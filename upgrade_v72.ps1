$root='E:\TraderAI_V6'
$b=Join-Path $root 'backend\app.py'
$f=Join-Path $root 'frontend\index.html'
$s=Join-Path $PSScriptRoot 'app_v72_source.py'
if(!(Test-Path $b) -or !(Test-Path $f)){throw 'Khong tim thay E:\TraderAI_V6\backend\app.py hoac frontend\index.html'}
Copy-Item $b "$b.v71-backup" -Force
Copy-Item $f "$f.v71-backup" -Force
Copy-Item $s $b -Force
$h=Get-Content $f -Raw
$h=$h.Replace('TraderAI V7.1 High-Conviction','TraderAI V7.2 Speed Upgrade').Replace('V7.1 Settings','V7.2 Speed Settings').Replace('V7.1 Consensus','V7.2 Consensus').Replace('Phân tích V7.1','Phân tích V7.2')
$newPrices=@'
async function prices(){try{const d=await api("/api/market-snapshot?symbols="+encodeURIComponent(BN.join(",")),{},15000),items=d.items||{};Object.entries(items).forEach(([s,q])=>{const z=document.querySelector(`[data-p="${s}"]`),c=document.querySelector(`[data-c="${s}"]`),p=Number(q.price),ch=Number(q.change_percent);if(z)z.textContent=fmt(p,p>1000?2:5);if(c){c.textContent=Number.isFinite(ch)?(ch>=0?"+":"")+ch.toFixed(2)+"%":"--";c.className=ch>0?"buy":ch<0?"sell":"neutral"}})}catch(e){console.warn("Snapshot:",e.message)}}
'@
$h=[regex]::Replace($h,'async function prices\(\)\{.*?\n\}',$newPrices,[Text.RegularExpressions.RegexOptions]::Singleline,1000)
$newSelect=@'
function select(s){selected=ALL.find(x=>x.s===s)||selected;$("asset").textContent=selected.s;$("source").textContent=selected.src==="TV"?TV[selected.s]+" · TradingView":selected.s+" · Binance";renderWatch();chart();analyze(true);news()}
'@
$h=[regex]::Replace($h,'function select\(s\)\{.*?\n\}',$newSelect,[Text.RegularExpressions.RegexOptions]::Singleline,1000)
$newAnalyze=@'
let analysisInFlight=null,analysisCache={key:"",time:0,data:null};
async function analyze(force=false){const key=selected.s+"|"+selectedTF,now=Date.now();if(!force&&analysisCache.key===key&&analysisCache.data&&now-analysisCache.time<45000){render(analysisCache.data);return analysisCache.data}if(analysisInFlight)return analysisInFlight;analysisInFlight=(async()=>{try{const d=await api("/api/analyze-live-mtf?symbol="+encodeURIComponent(selected.s)+"&limit=300",{},90000);analysisCache={key:key,time:Date.now(),data:d};render(d);return d}catch(e){toast("Phan tich loi: "+e.message);throw e}finally{analysisInFlight=null}})();return analysisInFlight}
'@
$h=[regex]::Replace($h,'async function analyze\(\)\{.*?\n\}',$newAnalyze,[Text.RegularExpressions.RegexOptions]::Singleline,1000)
$h=$h.Replace('renderWatch();chart();history();news();analyze();setInterval(()=>{prices();analyze();history()},60000);setInterval(news,180000);','renderWatch();chart();history();news();prices();analyze();setInterval(prices,30000);setInterval(()=>analyze(false),60000);setInterval(history,120000);setInterval(news,180000);')
$h=$h.Replace('$("refresh").onclick=()=>{chart();renderWatch();analyze();news();history()};','$("refresh").onclick=()=>{chart();renderWatch();prices();analyze(true);news();history()};')
Set-Content $f $h -Encoding UTF8
Write-Host 'TraderAI V7.2 Speed Upgrade da duoc cai.'
Write-Host 'Backup: backend/app.py.v71-backup va frontend/index.html.v71-backup'
