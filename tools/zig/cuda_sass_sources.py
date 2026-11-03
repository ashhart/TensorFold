"""Snapshot either checkout paths or local revisions and generate the CUDA sources their builds actually use."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import cuda_sass_identity as sid

PATHS=("zig/kernels","zig/build","zig/src/families/flashnext","zig/src/families/qwen4")
GENERATED=("qwen4_gdn","qwen4_ple")
FILES=(".zig-version",)
GENERATOR='''const std = @import("std");
pub fn build(b: *std.Build) void {
    const files = b.addWriteFiles();
    const gdn = @import("zig/src/families/flashnext/hybrid_sources.zig").Gdn.cuda(b.allocator,
        @embedFile("zig/kernels/cuda/flashnext_gdn.cu")) catch @panic("GDN specialization");
    const gdn_control = @import("zig/src/families/qwen4/gdn_control_source.zig").cuda(b.allocator, gdn)
        catch @panic("GDN controls");
    const ple = @import("zig/src/families/flashnext/hybrid_ple_source.zig").cuda(b.allocator)
        catch @panic("PLE specialization");
    const ple_control = @import("zig/src/families/qwen4/ple_control_source.zig").cuda(b.allocator, ple)
        catch @panic("PLE controls");
    b.getInstallStep().dependOn(&b.addInstallFile(files.add("qwen4_gdn.cu", gdn_control), "qwen4_gdn.cu").step);
    b.getInstallStep().dependOn(&b.addInstallFile(files.add("qwen4_ple.cu", ple_control), "qwen4_ple.cu").step);
}
'''


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def manifest(tree):
    data={p.relative_to(tree).as_posix():sha(p) for prefix in PATHS for p in sorted((tree/prefix).rglob("*")) if p.is_file()}
    data.update({name:sha(tree/name) for name in FILES if (tree/name).is_file()})
    return data


def snapshot(value,repo,dest):
    source=Path(value)
    dest.mkdir(parents=True,exist_ok=False)
    if source.is_dir():
        source=source.resolve()
        for name in PATHS:
            if not (source/name).is_dir():
                raise ValueError(f"missing source directory: {source/name}")
            shutil.copytree(source/name,dest/name)
        for name in FILES:
            shutil.copyfile(source/name,dest/name)
        identity={"kind":"checkout","input":str(source)}
        head=subprocess.run(["git","-C",str(source),"rev-parse","HEAD"],capture_output=True,text=True)
        if head.returncode==0:
            identity["head"]=head.stdout.strip()
    else:
        revision=sid.git(str(repo),"rev-parse",value+"^{commit}").decode().strip()
        archive=sid.git(str(repo),"archive",revision,*PATHS,*FILES)
        subprocess.run(["tar","-x","-C",str(dest)],input=archive,check=True)
        identity={"kind":"revision","input":value,"head":revision}
    identity["files"]=manifest(dest)
    return identity


def guard(args,output):
    if not args.cache_guard:
        if sys.platform=="darwin":
            raise RuntimeError("--cache-guard is required before compiler builds on a Mac")
        return
    run=subprocess.run(["bash",args.cache_guard],capture_output=True,text=True)
    with (output/"disk.log").open("a") as stream:
        stream.write(run.stdout+run.stderr)
    if run.returncode and "a build is running, cache kept" not in run.stdout:
        raise RuntimeError("cache guard refused this build; see disk.log")
    if sys.platform=="darwin":
        state=subprocess.check_output(["ps","-axo","comm"],text=True)
        if sum(line.endswith("/maker") for line in state.splitlines())>=2:
            raise RuntimeError("two build roots are already active")


def generate(tree,args,output,tag):
    catalog=sid.catalog(tree)[1]
    if not set(GENERATED)&set(catalog):
        return {}
    if not set(GENERATED)<=set(catalog):
        raise ValueError("incomplete generated-image catalog")
    version=subprocess.check_output([args.zig,"version"],text=True).strip()
    if version!=(tree/".zig-version").read_text().strip():
        raise ValueError("use the source tree's pinned Zig version for production generation")
    build_file=tree/"build.sass-sources.zig"
    build_file.write_text(GENERATOR)
    guard(args,output)
    command=["nice","-n","19",args.zig,"build","--build-file",build_file.name,
             "--prefix",str(tree/"generated"),"-j2","--summary","all"]
    run=subprocess.run(command,cwd=tree,capture_output=True,text=True)
    (output/(tag+"-generation.log")).write_text(run.stdout+run.stderr)
    if run.returncode:
        raise RuntimeError(f"{tag}: production CUDA source generation failed")
    record={"command":command,"zig_version":version,"sources":{}}
    for name in GENERATED:
        source=tree/"generated"/(name+".cu")
        (tree/"zig/kernels/cuda"/(name+".cu")).write_bytes(source.read_bytes())
        record["sources"][name]=sha(source)
    return record


def mutated(head,dest,relative,pattern,replacement):
    import re
    dest.mkdir(parents=True,exist_ok=False)
    for prefix in ("zig/kernels","zig/build"):
        shutil.copytree(head/prefix,dest/prefix)
    source=dest/"zig/kernels/cuda"/relative
    changed,n=re.subn(pattern,replacement,source.read_text(),count=1)
    if n!=1:
        raise ValueError(f"control needle missing: {relative}")
    source.write_text(changed)
    return dest
