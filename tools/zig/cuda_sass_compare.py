"""Compare every emitted function of every reached CUDA image on the five served SMs; retain all evidence."""
import argparse
import concurrent.futures as futures
from datetime import datetime,timezone
import difflib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import cuda_sass_identity as sid
import cuda_sass_sources as sources


def select(trees,catalogs):
    manifests={tag:sources.manifest(tree) for tag,tree in trees.items()}
    changed={n for n in set(manifests['base'])|set(manifests['head'])
             if manifests['base'].get(n)!=manifests['head'].get(n)}
    left,right=catalogs['base'][1],catalogs['head'][1]
    if set(left)!=set(right):
        raise ValueError('image catalog additions/removals need their own gate: '+str(sorted(set(left)^set(right))))
    picked=[]
    for name in sorted(left):
        reach=set().union(*(sid.includes(trees[tag],catalogs[tag][1][name][0]) for tag in trees))
        if reach&changed or left[name]!=right[name] or catalogs["base"][0]!=catalogs["head"][0]:
            picked.append(name)
        elif name in sources.GENERATED and any(n.startswith('zig/src/families/') for n in changed):
            picked.append(name)
    return picked,sorted(changed)


def image_scope(reached,requested):
    if requested is None:
        return reached
    names=requested.split(',')
    if any(not name for name in names) or len(set(names))!=len(names):
        raise ValueError('image scope must contain distinct, nonempty image names')
    unknown=set(names)-set(reached)
    if unknown:
        raise ValueError('image scope is not reached by this change: '+', '.join(sorted(unknown)))
    return sorted(names)


def parsed(text,image):
    if text.startswith('BUILD FAIL'):
        raise RuntimeError(text)
    data=sid.functions(text)
    if not data:
        raise ValueError('no emitted SASS functions; no identity proof')
    expected={s for s in sid.SMS if s>=image[3]}
    actual=set()
    headers=0
    for line in text.splitlines():
        if re.match(r'\s*Function\s*:',line):headers+=1
    if headers!=len(data):
        raise ValueError('missing or duplicate architecture/function headers')
    for (arch,function),lines in data.items():
        match=re.fullmatch(r'sm_(\d+)a?',arch or '')
        if not match or not lines or not words(lines):
            raise ValueError(f'unparsed or empty SASS function: {arch} {function}')
        actual.add(int(match[1]))
    if actual!=expected:
        raise ValueError(f'architecture coverage differs: expected {sorted(expected)}, got {sorted(actual)}')
    return data


def words(lines):
    values=[]
    count=0
    for line in lines:
        start=re.match(r"/\*[0-9a-f]{4,}\*/",line,re.I)
        if start:
            if count not in (0,2):
                raise ValueError("incomplete instruction encoding")
            count=0
        elif count==0:
            raise ValueError("orphan instruction continuation")
        found=re.findall(r'/\*\s*(0x[0-9a-f]{16,32})\s*\*/',line,re.I)
        if not found:
            raise ValueError('instruction or continuation lacks its encoding: '+line)
        values.extend(v.lower() for v in found)
        count+=len(found)
        if count>2:
            raise ValueError("unexpected instruction encoding width")
    if count!=2:
        raise ValueError("incomplete final instruction encoding")
    return values


def compare_image(args,trees,catalogs,name,output,rows,passes):
    maps={}
    for run in range(passes):
        jobs=[(tag,output/'compiles'/tag/name/f'pass-{run}.fatbin') for tag in trees]
        with futures.ThreadPoolExecutor(args.jobs) as pool:
            pending={tag:pool.submit(sid.build,trees[tag],catalogs[tag][0],catalogs[tag][1][name],path)
                     for tag,path in jobs}
            for tag,job in pending.items():
                data=parsed(job.result(),catalogs[tag][1][name])
                maps[tag,run]=data
    base,head=maps['base',0],maps['head',0]
    changed,added,removed=sid.compare(base,head)
    unstable=[]
    if passes==2:
        _,unstable=sid.recheck((base,head),(maps['base',1],maps['head',1]))
    keys=sorted(set(base)|set(head))
    for key in keys:
        a,b=base.get(key),head.get(key)
        row={'image':name,'arch':key[0],'function':key[1],
             'base_sass_sha256':sid.digest(a) if a is not None else None,
             'head_sass_sha256':sid.digest(b) if b is not None else None,
             'base_encoding_sha256':sid.digest(words(a)) if a is not None else None,
             'head_encoding_sha256':sid.digest(words(b)) if b is not None else None,
             'status':'unstable' if key in unstable else 'added' if a is None else 'removed' if b is None
                      else 'changed' if key in changed else 'equal'}
        rows.append(row)
        if row['status']!='equal':
            target=output/'diffs'/name
            target.mkdir(parents=True,exist_ok=True)
            filename=sid.digest([key[0],key[1]])+'.diff'
            (target/filename).write_text('\n'.join(difflib.unified_diff(a or [],b or [],fromfile='base',tofile='head'))+'\n')
    return base,head,not(changed or added or removed or unstable)


def write_rows(output,rows):
    (output/'functions.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields=('image','arch','function','base_sass_sha256','head_sass_sha256','base_encoding_sha256','head_encoding_sha256','status')
    (output/'compared-functions.tsv').write_text('\t'.join(fields)+'\n'+''.join(
        '\t'.join(str(row.get(k) or '-') for k in fields)+'\n' for row in rows))


def controls(args,trees,catalogs,heads,output):
    record=[]
    for index,(label,relative,pattern,replacement) in enumerate(sid.CONTROLS):
        tree=sources.mutated(trees['head'],output/f'control-{index}',relative,pattern,replacement)
        path='zig/kernels/cuda/'+relative
        candidates=[name for name in sorted(catalogs['head'][1]) if path in sid.includes(tree,catalogs['head'][1][name][0])]
        images=[name for name in candidates if name in heads]+[name for name in candidates if name not in heads]
        if not images:
            raise ValueError('control reaches no compared image: '+label)
        changed_by_arch={}
        control_rows=[]
        used=[]
        for name in images:
            used.append(name)
            if name not in heads:
                first_base=parsed(sid.build(trees['head'],catalogs['head'][0],catalogs['head'][1][name],
                    output/'compiles'/f'control-base-{index}'/name/'pass-0.fatbin'),catalogs['head'][1][name])
                again_base=parsed(sid.build(trees['head'],catalogs['head'][0],catalogs['head'][1][name],
                    output/'compiles'/f'control-base-{index}'/name/'pass-1.fatbin'),catalogs['head'][1][name])
                if first_base!=again_base:
                    raise ValueError('unstable control baseline: '+name)
                heads[name]=first_base
            first=parsed(sid.build(tree,catalogs['head'][0],catalogs['head'][1][name],
                                  output/'compiles'/f'control-{index}'/name/'pass-0.fatbin'),catalogs['head'][1][name])
            again=parsed(sid.build(tree,catalogs['head'][0],catalogs['head'][1][name],
                                  output/'compiles'/f'control-{index}'/name/'pass-1.fatbin'),catalogs['head'][1][name])
            if first!=again:
                raise ValueError('unstable control build: '+label+' '+name)
            changed,added,removed=sid.compare(heads[name],first)
            for key in sorted(set(heads[name])|set(first)):
                a,b=heads[name].get(key),first.get(key)
                control_rows.append({'image':name,'arch':key[0],'function':key[1],
                    'base_sass_sha256':sid.digest(a) if a is not None else None,
                    'head_sass_sha256':sid.digest(b) if b is not None else None,
                    'base_encoding_sha256':sid.digest(words(a)) if a is not None else None,
                    'head_encoding_sha256':sid.digest(words(b)) if b is not None else None,
                    'status':'changed' if key in changed else 'added' if key in added else 'removed' if key in removed else 'equal'})
            write_rows(tree,control_rows)
            for arch,fn in changed+added+removed:
                changed_by_arch.setdefault(arch,[]).append({'image':name,'function':fn})
            if changed_by_arch:
                break
        entry={'control':label,'images':used,'changed_functions':changed_by_arch,
               'pass':bool(changed_by_arch)}
        record.append(entry)
        (output/'controls.json').write_text(json.dumps(record,indent=2)+'\n')
        if not entry['pass']:
            raise ValueError('control failed to detect any changed function: '+label)
    return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',help='checkout directory or local git revision')
    parser.add_argument('head',help='checkout directory or local git revision')
    parser.add_argument('--repo',default='.')
    parser.add_argument('--output','--proof-dir',type=Path)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--zig',default=os.environ.get('ZIG','zig'))
    parser.add_argument('--nvcc',default=sid.NVCC)
    parser.add_argument('--cuobjdump',default=sid.CUOBJDUMP)
    parser.add_argument('--cache-guard')
    parser.add_argument('--images',help='Comma-separated reached images; qualifies only this explicit scope')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--generate-only',action='store_true')
    parser.add_argument('--no-controls',action='store_true',help='diagnostic only, exit 2 after matching images')
    parser.add_argument('--no-recheck',action='store_true',help='diagnostic only, exit 2 after matching images')
    parser.add_argument('-j','--jobs',type=int,choices=(1,2),default=2)
    args=parser.parse_args(argv)
    output=(args.output or Path('sass-compare-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid()))).resolve()
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    report={'inputs':{},'sms':sid.SMS,'qualified':False,'output':str(output),'mode':'plan' if args.plan_only else 'generate' if args.generate_only else 'sass'}
    try:
        trees={tag:output/tag for tag in ('base','head')}
        for tag,value in zip(trees,(args.base,args.head)):
            report['inputs'][tag]=sources.snapshot(value,Path(args.repo).resolve(),trees[tag])
        if not(args.plan_only or args.generate_only):
            if not shutil.which(args.nvcc) or not shutil.which(args.cuobjdump):
                raise RuntimeError('nvcc and cuobjdump required; no SASS comparison was run')
            sid.NVCC,sid.CUOBJDUMP=args.nvcc,args.cuobjdump
            report['compiler']=subprocess.check_output([args.nvcc,'--version'],text=True)
            report['disassembler']=subprocess.check_output([args.cuobjdump,'--version'],text=True)
        if not args.plan_only:
            report['generated']={tag:sources.generate(tree,args,output,tag) for tag,tree in trees.items()}
        report['compiled_sources']={tag:sources.manifest(tree) for tag,tree in trees.items()}
        catalogs={tag:sid.catalog(tree) for tag,tree in trees.items()}
        report['torch_flags']={tag:catalogs[tag][0] for tag in trees}
        reached,changed=select(trees,catalogs)
        images=image_scope(reached,args.images)
        report.update(images=images,changed_files=changed,
                      all_reached_images=reached,omitted_images=sorted(set(reached)-set(images)),
                      comparison_scope='explicit images' if args.images is not None else 'all reached images',
                      catalogs={tag:{name:catalogs[tag][1][name] for name in images} for tag in trees},
                      function_scope='Every emitted cuobjdump Function on every supported SM, including outlined helpers and controls')
        if args.plan_only or args.generate_only:
            print(f'{report["mode"].upper()} ONLY: {len(images)} candidate images; no SASS comparison was run')
            return 0
        if not images:
            raise ValueError('no reached CUDA images; no identity qualification was performed')
        heads={}
        for name in images:
            _,heads[name],equal=compare_image(args,trees,catalogs,name,output,rows,1 if args.no_recheck else 2)
            write_rows(output,rows)
            print(f'{name}: {sum(row["image"]==name for row in rows)} functions; '+('equal' if equal else 'FAIL'),flush=True)
            if not equal:
                raise ValueError('first differing or unstable image: '+name)
        if not args.no_controls:
            report['controls']=controls(args,trees,catalogs,heads,output)
        report['qualified']=not(args.no_controls or args.no_recheck)
        print('SUMMARY '+('PASS' if report['qualified'] else 'UNQUALIFIED')+f' images={len(images)} functions={len(rows)}'
              +' scope='+('explicit' if args.images is not None else 'all-reached'))
        return 0 if report['qualified'] else 2
    except Exception as error:
        report['error']=str(error)
        print('FAIL: '+str(error),file=sys.stderr)
        return 1
    finally:
        write_rows(output,rows)
        (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        if args.receipt:
            args.receipt.write_bytes((output/'compared-functions.tsv').read_bytes())

if __name__=='__main__':
    raise SystemExit(main())
