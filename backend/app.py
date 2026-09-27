from __future__ import annotations
import copy, math, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from typing import Any, List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.config import ASSETS, TIMEFRAMES
from backend.data_adapter import fetch_market_dataframe_cached, fetch_market_klines, get_latest_quote, get_symbol_info
from backend.analyzer import analyze_market
from backend.history import HISTORY_LIMIT, clear_history, get_history, get_stats, record_signal, refresh_history
from backend.v7_engine import enrich, ENGINE_VERSION
from backend.news import fetch_news

APP_NAME='TraderAI V7.2.1 Speed Fix'; VERSION='7.2.1'; MODE='analysis_only'
app=FastAPI(title=APP_NAME,version=VERSION)
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_credentials=False,allow_methods=['*'],allow_headers=['*'])
class Candle(BaseModel):
    timestamp: Any; open: float; high: float; low: float; close: float; volume: float=0.0
class AnalyzeRequest(BaseModel): candles: List[Candle]=Field(min_length=220)
MARKET_CACHE_SECONDS=12.0
MTF_CACHE_SECONDS=45.0
SNAPSHOT_CACHE_SECONDS=12.0
SNAPSHOT_REQUEST_TIMEOUT=7.5

_market_cache={}
_mtf_cache={}
_snapshot_cache={}
_cache_lock=threading.RLock()

def json_safe(v):
    if v is None:return None
    if isinstance(v,dict):return {str(k):json_safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [json_safe(x) for x in v]
    if isinstance(v,np.ndarray):return [json_safe(x) for x in v.tolist()]
    if isinstance(v,np.generic):v=v.item()
    if isinstance(v,float):return v if math.isfinite(v) else None
    if isinstance(v,pd.Timestamp):return v.isoformat()
    return v

def validate_dataframe(candles):
    df=pd.DataFrame([c.model_dump() for c in candles])
    for c in ['open','high','low','close','volume']:df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=['open','high','low','close']).reset_index(drop=True)
    if len(df)<220:raise HTTPException(422,'Cần ít nhất 220 candle hợp lệ.')
    return df

def fetch_market_klines_cached(symbol, timeframe='M1', limit=300):
    key=(symbol.upper().strip(), timeframe.upper().strip(), int(limit))
    now=time.time()
    with _cache_lock:
        hit=_market_cache.get(key)
        if hit and now-hit[0] <= MARKET_CACHE_SECONDS:
            return copy.deepcopy(hit[1])
    data=fetch_market_klines(symbol, timeframe, limit)
    with _cache_lock:
        _market_cache[key]=(time.time(), copy.deepcopy(data))
    return data

def analyze_one(symbol,tf,limit,with_news=False):
    df=fetch_market_dataframe_cached(symbol,tf,limit); base=analyze_market(df)
    news=fetch_news(symbol,8) if with_news else {'items':[],'sentiment_score':0.0}
    a=enrich(df,base,news); a['timeframe']=tf
    return {'signal':a['signal'],'confidence':a['confidence'],'analysis':a,'source':get_symbol_info(symbol),'timeframe':tf}

def build_consensus(results):
    w={'M1':.05,'M5':.08,'M15':.12,'H1':.22,'H4':.18,'D1':.25,'W1':.10}; buy=sell=0.0
    for tf,r in results.items():
        x=w.get(tf,0)*float(r.get('confidence',50))/100
        if r.get('signal')=='BUY':buy+=x
        elif r.get('signal')=='SELL':sell+=x
    total=max(buy+sell,1e-9); edge=abs(buy-sell)/total; sig='BUY' if buy>sell else 'SELL' if sell>buy else 'NEUTRAL'
    directional=[r.get('signal') for r in results.values() if r.get('signal') in {'BUY','SELL'} and float(r.get('confidence',0))>=62]
    same=sum(x==sig for x in directional); h=[results.get(tf,{}).get('signal') for tf in ('H4','D1','W1')]; hd=[x for x in h if x in {'BUY','SELL'}]
    hs=sum(x==sig for x in hd)/len(hd) if hd else 0.0; high=sig!='NEUTRAL' and same>=2 and edge>=.20 and hs>=.50; conf=min(97,50+edge*45+same*2)
    if not high:sig='NEUTRAL'
    return {'consensus':sig,'signal':sig,'confidence':round(conf,2),'buy_score':round(buy,4),'sell_score':round(sell,4),'directional_edge':round(edge,4),'high_conviction':high,'directional_frames':directional,'same_direction_frames':same,'htf_support':round(hs,3),'engine_version':ENGINE_VERSION}

def history_payload(symbol,c,results):
    sig=c.get('signal','NEUTRAL')
    if sig not in {'BUY','SELL'}:return None
    for tf in ('H1','M15','H4','D1','W1','M5','M1'):
        a=results.get(tf,{}).get('analysis') or {}
        if a.get('signal')==sig:return {'symbol':symbol,'timeframe':'MTF','signal':sig,'entry':a.get('entry'),'sl':a.get('sl'),'tp1':a.get('tp1'),'confidence':c.get('confidence'),'signal_quality':a.get('signal_quality')}
    return None

def run_mtf(symbol,limit):
    results={}; errors={}
    with ThreadPoolExecutor(max_workers=7) as pool:
        jobs={pool.submit(analyze_one,symbol,tf,limit,tf in {'H1','H4','D1','W1'}):tf for tf in TIMEFRAMES}
        for job in as_completed(jobs):
            tf=jobs[job]
            try:results[tf]=job.result()
            except Exception as e:errors[tf]=str(e); results[tf]={'signal':'NEUTRAL','confidence':0,'analysis':None,'error':str(e),'timeframe':tf}
    c=build_consensus(results); rec=None
    try:
        p=history_payload(symbol,c,results)
        if p:rec=record_signal(p)
    except Exception as e:errors['history']=str(e)
    return json_safe({'status':'ok','symbol':symbol,'timeframes':results,'consensus':c,'errors':errors,'history_record':rec,'analysis_only':True,'engine_version':ENGINE_VERSION,'source':get_symbol_info(symbol),'speed_upgrade':'V7.2.1','cached_seconds':MTF_CACHE_SECONDS})

@app.get('/')
def root():return {'name':APP_NAME,'version':VERSION,'status':'running','mode':MODE,'analysis_only':True,'engine_version':ENGINE_VERSION}
@app.get('/api/health')
def health():return {'name':APP_NAME,'version':VERSION,'status':'running','mode':MODE,'analysis_only':True,'engine_version':ENGINE_VERSION,'timeframes':TIMEFRAMES,'speed_upgrade':'V7.2.1','cache':{'market_seconds':MARKET_CACHE_SECONDS,'mtf_seconds':MTF_CACHE_SECONDS,'snapshot_seconds':SNAPSHOT_CACHE_SECONDS,'snapshot_timeout_seconds':SNAPSHOT_REQUEST_TIMEOUT},'timeframes':TIMEFRAMES}
@app.get('/api/status')
def status():return {'name':APP_NAME,'version':VERSION,'status':'running','mode':MODE,'analysis_only':True,'execution':False,'order_placement':False,'engine_version':ENGINE_VERSION,'assets':ASSETS,'speed_upgrade':'V7.2.1','cache':{'market_seconds':MARKET_CACHE_SECONDS,'mtf_seconds':MTF_CACHE_SECONDS,'snapshot_seconds':SNAPSHOT_CACHE_SECONDS,'snapshot_timeout_seconds':SNAPSHOT_REQUEST_TIMEOUT}}
@app.get('/api/market-data')
def market_data(symbol:str,timeframe:str='M1',limit:int=Query(300,ge=1,le=1000)):
    try:return json_safe({'status':'ok','symbol':symbol.upper(),'timeframe':timeframe.upper(),'data':fetch_market_klines_cached(symbol,timeframe,limit),'source':get_symbol_info(symbol),'cached_seconds':MARKET_CACHE_SECONDS})
    except Exception as e:raise HTTPException(500,f'Market data thất bại: {e}')
@app.get('/api/market-snapshot')
def market_snapshot(symbols:str):
    requested=list(dict.fromkeys([x.strip().upper() for x in symbols.split(',') if x.strip()]))[:30]
    if not requested:raise HTTPException(422,'Cần ít nhất một symbol.')
    key=tuple(sorted(requested)); now=time.time()
    with _cache_lock:
        cached=_snapshot_cache.get(key)
        if cached and now-cached[0]<=SNAPSHOT_CACHE_SECONDS:
            out=copy.deepcopy(cached[1]); out['cached']=True; return json_safe(out)
    quotes={}; errors={}
    pool=ThreadPoolExecutor(max_workers=min(8,len(requested)))
    jobs={pool.submit(get_latest_quote,s):s for s in requested}
    done,pending=wait(list(jobs), timeout=SNAPSHOT_REQUEST_TIMEOUT)

    for job in done:
        s=jobs[job]
        try:
            quotes[s]=job.result()
        except Exception as e:
            errors[s]=str(e)

    for job in pending:
        s=jobs[job]
        errors[s]=f'Snapshot timeout after {SNAPSHOT_REQUEST_TIMEOUT:.1f}s'

    pool.shutdown(wait=False, cancel_futures=True)

    result={
        'status':'ok',
        'data':quotes,
        'items':quotes,
        'errors':errors,
        'cached':False,
        'partial':bool(pending) or bool(errors),
        'updated_at':time.time(),
        'analysis_only':True,
        'speed_upgrade':'V7.2.1',
        'request_timeout_seconds':SNAPSHOT_REQUEST_TIMEOUT,
    }
    # Only cache successful/partial data that actually arrived; never cache an empty timeout.
    if quotes:
        with _cache_lock:
            _snapshot_cache[key]=(time.time(),copy.deepcopy(result))
    return json_safe(result)
@app.post('/api/cache/clear')
def cache_clear():
    with _cache_lock:
        _market_cache.clear()
        _snapshot_cache.clear()
        _mtf_cache.clear()
    return json_safe({'status':'ok','message':'Đã xóa cache V7.2.1','analysis_only':True})

@app.get('/api/news')
def api_news(symbol:str,limit:int=Query(8,ge=1,le=20)):return json_safe({'status':'ok',**fetch_news(symbol,limit)})
@app.post('/api/analyze')
def analyze(request:AnalyzeRequest):
    try:
        df=validate_dataframe(request.candles); r=enrich(df,analyze_market(df),{'items':[],'sentiment_score':0}); return json_safe({'status':'ok','analysis':r})
    except HTTPException:raise
    except Exception as e:raise HTTPException(500,f'Phân tích thất bại: {e}')
@app.get('/api/analyze-live-mtf')
def analyze_live_mtf(symbol:str,limit:int=Query(300,ge=220,le=1000)):
    symbol=symbol.upper().strip(); key=(symbol,int(limit)); now=time.time()
    with _cache_lock:
        cached=_mtf_cache.get(key)
        if cached and now-cached[0]<=MTF_CACHE_SECONDS:
            out=copy.deepcopy(cached[1]); out['cached']=True; out['cache_age_seconds']=round(now-cached[0],2); return json_safe(out)
    result=run_mtf(symbol,int(limit))
    with _cache_lock:_mtf_cache[key]=(time.time(),copy.deepcopy(result))
    result['cached']=False; result['cache_age_seconds']=0; return json_safe(result)
@app.get('/api/history')
def api_history(limit:int=Query(HISTORY_LIMIT,ge=1,le=HISTORY_LIMIT)):
    h=get_history(limit); return json_safe({'status':'ok','history':h,'stats':get_stats(h),'limit':limit,'analysis_only':True})
@app.get('/api/history/refresh')
def api_history_refresh(limit:int=Query(HISTORY_LIMIT,ge=1,le=HISTORY_LIMIT)):return json_safe({'status':'ok',**refresh_history(limit),'analysis_only':True})
@app.post('/api/history/record')
def api_history_record(payload:dict):
    try:r=record_signal(payload); return json_safe({'status':'ok',**r,'stats':get_stats(),'analysis_only':True})
    except Exception as e:raise HTTPException(422,str(e))
@app.delete('/api/history')
def api_history_delete():return json_safe({'status':'ok',**clear_history(),'analysis_only':True})
