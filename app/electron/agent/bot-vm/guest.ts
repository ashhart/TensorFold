// This small original RPC program is passed to the guest through argv; request
// data travels over stdin, never through shell interpolation or host files.
export const BOT_VM_GUEST = String.raw`
import os, sys, json, stat, subprocess, signal, selectors, time, uuid, fnmatch, itertools
MAX=65536
def pathparts(value):
    if not isinstance(value,str) or len(value)>2048 or '\x00' in value: raise ValueError('Invalid workspace path.')
    if value=='/workspace': value=''
    elif value.startswith('/workspace/'): value=value[11:]
    elif value.startswith('/'): raise ValueError('Files must stay inside /workspace.')
    parts=value.split('/')
    if any(p=='..' for p in parts): raise ValueError('Parent paths are not allowed.')
    return [p for p in parts if p not in ('','.')]
def directory(parts,create=False):
    fd=os.open('/workspace',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for p in parts:
            if create:
                try: os.mkdir(p,0o700,dir_fd=fd)
                except FileExistsError: pass
            nextfd=os.open(p,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd); fd=nextfd
        return fd
    except: os.close(fd); raise
def readfile(parts,limit=1048576):
    if not parts: raise ValueError('Choose a file.')
    parent=directory(parts[:-1]); fd=None
    try:
        fd=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
        s=os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1: raise ValueError('Only regular files without links can be read.')
        if s.st_size>limit: raise ValueError('File exceeds the one MiB text limit.')
        with os.fdopen(fd,'rb') as f: fd=None; raw=f.read(limit+1)
        if len(raw)>limit: raise ValueError('File exceeds the text limit.')
        return raw.decode('utf-8')
    finally:
        if fd is not None: os.close(fd)
        os.close(parent)
def writefile(parts,content):
    if not parts or not isinstance(content,str) or len(content.encode('utf-8'))>1048576: raise ValueError('Write requires a file and at most one MiB of UTF-8 text.')
    parent=directory(parts[:-1],True); temp='.tensorfold-'+uuid.uuid4().hex
    try:
        try:
            s=os.stat(parts[-1],dir_fd=parent,follow_symlinks=False)
            if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1: raise ValueError('Refusing to replace a link or non-regular file.')
        except FileNotFoundError: pass
        fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
        with os.fdopen(fd,'w',encoding='utf-8') as f: f.write(content); f.flush(); os.fsync(f.fileno())
        os.replace(temp,parts[-1],src_dir_fd=parent,dst_dir_fd=parent)
    finally:
        try: os.unlink(temp,dir_fd=parent)
        except FileNotFoundError: pass
        os.close(parent)
def walk(parts):
    pending=[parts]; seen=0
    while pending and seen<2000:
        parentparts=pending.pop(); fd=directory(parentparts)
        try:
            with os.scandir(fd) as entries: names=sorted(entry.name for entry in itertools.islice(entries,1001))
            for name in names[:1000]:
                seen+=1
                if seen>2000: return
                info=os.stat(name,dir_fd=fd,follow_symlinks=False); rel=parentparts+[name]
                if stat.S_ISDIR(info.st_mode) and name not in ('node_modules','.git','__pycache__','.venv'): pending.append(rel)
                elif stat.S_ISREG(info.st_mode) and info.st_nlink==1: yield rel
        finally: os.close(fd)
def command(args):
    cmd=args.get('command'); timeout=args.get('timeoutMs',30000)
    if not isinstance(cmd,str) or not cmd.strip() or len(cmd)>16384 or '\x00' in cmd: raise ValueError('Command must contain at most 16,384 characters.')
    if not isinstance(timeout,(int,float)): raise ValueError('Invalid command timeout.')
    timeout=max(1000,min(120000,int(timeout)))/1000
    p=subprocess.Popen(['/bin/sh','-c',cmd],cwd='/workspace',stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/workspace','LANG':'C.UTF-8','TMPDIR':'/tmp'})
    sel=selectors.DefaultSelector(); sel.register(p.stdout,selectors.EVENT_READ); data=bytearray(); end=time.monotonic()+timeout; timedout=False
    try:
        while sel.get_map():
            remaining=end-time.monotonic()
            if remaining<=0: timedout=True; break
            for key,_ in sel.select(min(.1,remaining)):
                chunk=os.read(key.fileobj.fileno(),8192)
                if not chunk: sel.unregister(key.fileobj)
                elif len(data)<MAX: data.extend(chunk[:MAX-len(data)])
            if len(data)>=MAX: break
        bounded=len(data)>=MAX
        if timedout or bounded:
            try: os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError: pass
        try: code=p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL); code=p.wait(timeout=2)
        output=data.decode('utf-8',errors='replace')
        if timedout: output+='\n[Command stopped at its time limit.]'
        if bounded: output+='\n[Command stopped at its output limit.]'
        return {'ok':code==0 and not timedout and not bounded,'code':code,'output':output,'truncated':bounded,'interrupted':timedout or bounded}
    finally:
        sel.close(); p.stdout.close()
        try: os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        if p.poll() is None: p.wait()
def main(req):
    tool=req.get('tool'); args=req.get('args')
    if not isinstance(args,dict): raise ValueError('Invalid tool input.')
    if tool=='run_command': return command(args)
    parts=pathparts(args.get('path',''))
    if tool=='read_file':
        text=readfile(parts); lines=text.split('\n'); offset=max(1,min(1000000,int(args.get('offset',1)))); limit=max(1,min(2000,int(args.get('limit',2000))))
        out='\n'.join(str(i+offset).rjust(5)+'\t'+line for i,line in enumerate(lines[offset-1:offset-1+limit])); truncated=len(out)>MAX
        return {'ok':True,'output':out[:MAX]+ ('\n[Output limited; use a later offset.]' if truncated else ''),'truncated':truncated}
    if tool in ('write_file','edit_file'):
        if tool=='write_file': content=args.get('content')
        else:
            content=readfile(parts); old=args.get('old_string'); new=args.get('new_string')
            if not isinstance(old,str) or not old or not isinstance(new,str) or content.count(old)!=1: raise ValueError('The old string must occur exactly once.')
            content=content.replace(old,new,1)
        writefile(parts,content); return {'ok':True,'output':'Saved '+ '/'.join(parts),'changedPaths':['/'.join(parts)]}
    if tool=='list_dir':
        fd=directory(parts)
        try:
            with os.scandir(fd) as entries: names=sorted(entry.name for entry in itertools.islice(entries,501))
            out=[]
            for name in names[:500]:
                s=os.stat(name,dir_fd=fd,follow_symlinks=False)
                if not stat.S_ISLNK(s.st_mode): out.append(name+('/' if stat.S_ISDIR(s.st_mode) else ''))
            return {'ok':True,'output':'\n'.join(out)[:MAX] or '(empty)','truncated':len(names)>500}
        finally: os.close(fd)
    if tool in ('search','glob'):
        value=args.get('query' if tool=='search' else 'pattern')
        if not isinstance(value,str) or not value or len(value)>500 or '\x00' in value: raise ValueError('Provide a bounded search string or glob pattern.')
        if tool=='glob' and (value.startswith('/') or '..' in value.split('/')): raise ValueError('Glob must stay inside the workspace.')
        out=[]; count=0
        for rel in walk(parts):
            count+=1; name='/'.join(rel)
            if tool=='glob':
                relative='/'.join(rel[len(parts):])
                if fnmatch.fnmatchcase(relative,value) or (value.startswith('**/') and fnmatch.fnmatchcase(relative,value[3:])): out.append(name)
            else:
                try: text=readfile(rel,65536)
                except (ValueError,UnicodeError,OSError): continue
                for i,line in enumerate(text.split('\n')):
                    if value in line: out.append(name+':'+str(i+1)+':'+line[:1000])
                    if len(out)>=100: break
            if len(out)>=100 or sum(len(x)+1 for x in out)>MAX: break
        return {'ok':True,'output':'\n'.join(out)[:MAX] or '(no matches)','truncated':count>=2000 or len(out)>=100}
    raise ValueError('This tool is not available inside the bot VM.')
try:
    raw=sys.stdin.buffer.read(1100000)
    if len(raw)>1090000: raise ValueError('Request exceeds the VM limit.')
    result=main(json.loads(raw)); print(json.dumps(result,ensure_ascii=True))
except Exception as e:
    print(json.dumps({'ok':False,'output':'VM tool refused: '+str(e)[:500]}))
`
