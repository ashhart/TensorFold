import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { BOT_VM_GUEST } from './guest'

// Trace-only mirror of the guest program: same logic and limits as BOT_VM_GUEST, with P*/F*/X* markers on stderr localizing which syscall raises inside command(); assertions match guest.test.ts.
const DIAG = String.raw`import os, sys, json, stat, subprocess, signal, selectors, time, uuid, fnmatch, itertools
MAX=65536
def E(*a): print(*a, file=sys.stderr, flush=True)
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
    E('P1 pre-Popen: cmd=%r timeout=%r' % (cmd,timeout))
    p=subprocess.Popen(['/bin/sh','-c',cmd],cwd='/workspace',stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/workspace','LANG':'C.UTF-8','TMPDIR':'/tmp'})
    E('P2 Popen ok: pid=%r pgid=%r' % (p.pid, os.getpgid(p.pid)))
    sel=selectors.DefaultSelector(); E('P3 selector created: %s' % type(sel).__name__)
    sel.register(p.stdout,selectors.EVENT_READ); E('P4 registered fd=%r' % p.stdout.fileno())
    data=bytearray(); end=time.monotonic()+timeout; timedout=False
    try:
        while sel.get_map():
            remaining=end-time.monotonic()
            if remaining<=0: timedout=True; E('P5b time limit hit bytes=%d' % (len(data),)); break
            for key,_ in sel.select(min(.1,remaining)):
                chunk=os.read(key.fileobj.fileno(),8192)
                if not chunk: sel.unregister(key.fileobj); E('P5c EOF -> unregistered fd=%r' % key.fileobj.fileno())
                elif len(data)<MAX: data.extend(chunk[:MAX-len(data)])
            if len(data)>=MAX: E('P5a MAX reached bytes=%d -> break' % (len(data),)); break
        E('P6 loop done: bytes=%d timedout=%r bounded=%r' % (len(data),timedout,len(data)>=MAX))
        bounded=len(data)>=MAX
        if timedout or bounded:
            E('P7 primary killpg(%r,SIGKILL)' % p.pid)
            try: os.killpg(p.pid,signal.SIGKILL); E('P7a primary killpg returned')
            except (ProcessLookupError, PermissionError) as e: E('P7a primary killpg %s: %r' % (type(e).__name__, e))
        E('P8 primary p.wait(timeout=2)')
        try: code=p.wait(timeout=2); E('P8a reaped code=%r' % (code,))
        except subprocess.TimeoutExpired:
            E('P8b wait expired -> secondary killpg + wait')
            os.killpg(p.pid,signal.SIGKILL); code=p.wait(timeout=2); E('P8c reaped after secondary code=%r' % (code,))
        output=data.decode('utf-8',errors='replace')
        if timedout: output+='\n[Command stopped at its time limit.]'
        if bounded: output+='\n[Command stopped at its output limit.]'
        result={'ok':code==0 and not timedout and not bounded,'code':code,'output':output,'truncated':bounded,'interrupted':timedout or bounded}
        E('P9 returning normal/bounded result: ok=%r code=%r truncated=%r outlen=%d' % (result['ok'],result['code'],result['truncated'],len(output)))
        return result
    finally:
        E('F1 finally start (poll=%r)' % (p.poll(),))
        sel.close(); E('F2 sel.close() ok')
        p.stdout.close(); E('F3 p.stdout.close() ok')
        E('F4 FINAL killpg(%r,SIGKILL) -- after reap' % p.pid)
        try:
            os.killpg(p.pid,signal.SIGKILL); E('F4a final killpg returned (no exception)')
        except ProcessLookupError as e: E('F4a final killpg ProcessLookupError: %r' % (e,))
        except PermissionError as e: E('F4a final killpg PermissionError errno=%r' % (getattr(e,'errno',None),))
        E('F5 pre-poll: poll=%r' % (p.poll(),))
        if p.poll() is None: E('F5a still unreaped -> p.wait()'); p.wait(); E('F5b waited')
        E('F6 finally end')
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
    E('P0 stdin read')
    raw=sys.stdin.buffer.read(1100000)
    if len(raw)>1090000: raise ValueError('Request exceeds the VM limit.')
    result=main(json.loads(raw)); print(json.dumps(result,ensure_ascii=True))
except Exception as e:
    E('X1 outer except: %s: %r' % (type(e).__name__, str(e)))
    E('X2 __context__ chain: %r' % (repr(e.__context__),))
    print(json.dumps({'ok':False,'output':'VM tool refused: '+str(e)[:500]}))`

const root = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-vm-guest-test-'))
try {
  const workspace = path.join(root, 'workspace')
  mkdirSync(workspace)
  // Same two rewrites guest.test.ts applies to BOT_VM_GUEST, so both programs see one remapped root.
  const apply = (src: string) => src.replaceAll('/workspace', workspace).replace('value=value[11:]', `value=value[${workspace.length + 1}:]`)
  const realProgram = apply(BOT_VM_GUEST)
  const diagProgram = apply(DIAG)
  const probe = spawnSync('/usr/bin/python3', ['-I', '-c', "import sys,selectors,platform,json;print(json.dumps({'python':sys.version,'selector':type(selectors.DefaultSelector()).__name__,'platform':platform.platform()}))"], { encoding: 'utf8' })
  const envProbe = probe.status === 0 ? JSON.stringify(JSON.parse(probe.stdout)) : JSON.stringify({ probeStatus: probe.status, probeError: probe.error?.message, probeStderr: probe.stderr })
  const run = (program: string, tool: string, args: Record<string, unknown>) => spawnSync('/usr/bin/python3', ['-I', '-c', program], { input: JSON.stringify({ tool, args }), encoding: 'utf8', timeout: 10000 })
  type RunResult = ReturnType<typeof run>
  // Every message carries the case id, iteration, env probe and both stream tails so the CI log alone explains a failure.
  const detail = (caseId: string, iteration: number, result: RunResult) => `${caseId} iter=${iteration} env=${envProbe} signal=${result.signal ?? ''} error=${result.error?.message ?? ''} stdout_tail=${result.stdout.slice(-1500)} stderr_tail=${result.stderr.slice(-4000)}`
  const parsedOf = (caseId: string, iteration: number, result: RunResult) => {
    assert.equal(result.status, 0, `${caseId} iter=${iteration} guest exited abnormally: ${detail(caseId, iteration, result)}`)
    return JSON.parse(result.stdout) as { ok?: unknown; truncated?: unknown; interrupted?: unknown; code?: unknown }
  }
  for (let i = 1; i <= 8; i++) {
    const result = run(realProgram, 'run_command', { command: 'yes bounded-output' })
    const parsed = parsedOf('A-real-yes', i, result)
    assert.equal(parsed.ok, false, detail('A-real-yes', i, result))
    assert.equal(parsed.truncated, true, detail('A-real-yes', i, result))
    assert.equal(typeof parsed.code, 'number', detail('A-real-yes', i, result))
  }
  const hello = run(diagProgram, 'run_command', { command: 'printf hello' })
  const helloParsed = parsedOf('B-diag-printf', 1, hello)
  assert.equal(helloParsed.ok, true, detail('B-diag-printf', 1, hello))
  assert.equal(helloParsed.truncated, false, detail('B-diag-printf', 1, hello))
  const nap = run(diagProgram, 'run_command', { command: 'sleep 8', timeoutMs: 1000 })
  const napParsed = parsedOf('B-diag-timeout', 1, nap)
  assert.equal(napParsed.ok, false, detail('B-diag-timeout', 1, nap))
  assert.equal(napParsed.interrupted, true, detail('B-diag-timeout', 1, nap))
  let lastYesStderr = ''
  for (let i = 1; i <= 8; i++) {
    const result = run(diagProgram, 'run_command', { command: 'yes bounded-output' })
    const parsed = parsedOf('B-diag-yes', i, result)
    assert.equal(parsed.ok, false, detail('B-diag-yes', i, result))
    assert.equal(parsed.truncated, true, detail('B-diag-yes', i, result))
    lastYesStderr = result.stderr
  }
  console.log(`trace-diagnostic: 8 real runs, 10 instrumented runs, env ${envProbe}`)
  console.log(`trace-diagnostic last-instrumented-yes stderr:\n${lastYesStderr}`)
} finally { rmSync(root, { recursive: true, force: true }) }
