"""Run an unmodified upstream HTTP check while preserving each exchange."""
import io,json,os,pathlib,runpy,sys,threading,time,urllib.request,urllib.error
script=pathlib.Path(sys.argv[1]).resolve(); receipt=pathlib.Path(sys.argv[2]).resolve()
args=sys.argv[3:]; old=urllib.request.urlopen; lock=threading.Lock()
# Optional stress suites are separate from the requested model/HTTP matrix.
if (receipt.parent/"skip-extended").exists() and script.name in ("serve_rewind.py","structured_served.py"):
    print("SKIP optional extended suite; no PASS claimed",flush=True)
    raise SystemExit(77)
def record(req,*a,**kw):
    url=req.full_url if isinstance(req,urllib.request.Request) else str(req)
    request=json.loads(req.data) if isinstance(req,urllib.request.Request) and req.data else None
    start=time.time()
    try:
        r=old(req,*a,**kw);raw=r.read();status=r.status;headers=r.headers;r.close()
    except urllib.error.HTTPError as e:
        raw=e.read();status=e.code;headers=e.headers
        with lock,receipt.open('a') as f:f.write(json.dumps({'url':url,'request':request,'status':status,'response':raw.decode(),'seconds':time.time()-start})+'\n')
        raise urllib.error.HTTPError(url,status,e.reason,headers,io.BytesIO(raw))
    with lock,receipt.open('a') as f:f.write(json.dumps({'url':url,'request':request,'status':status,'response':json.loads(raw),'seconds':time.time()-start})+'\n')
    reply=io.BytesIO(raw);reply.status=status;reply.headers=headers
    return reply
urllib.request.urlopen=record
sys.argv=[str(script),*args]
runpy.run_path(str(script),run_name='__main__')
