"""Emit Metal wrappers around the owned CUDA learner algebra; no model or compiler is invoked."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALUES = dict(DK=128, DV=128, TAPS=4, CK=4, SPLIT=4, ROWS=32, ATTN_KEYS=2048, WARPS=8)


def close_brace(text, start):
    depth = 1
    at = start + 1
    while depth:
        if text[at] == '{':
            depth += 1
        elif text[at] == '}':
            depth -= 1
        at += 1
    return at


def convert(text):
    text = text.replace('__device__ __forceinline__', 'inline').replace('__restrict__', '')
    text = text.replace('__bfloat162float', 'float').replace('__float2bfloat16_rn', 'bfloat').replace('__float2bfloat16', 'bfloat')
    text = text.replace('__syncthreads()', 'threadgroup_barrier(mem_flags::mem_threadgroup)')
    for before, after in [('expf', 'precise::exp'), ('sqrtf', 'precise::sqrt'), ('log1pf', 'tf_log1p'),
                          ('cosf', 'precise::cos'), ('sinf', 'precise::sin'), ('fmaxf', 'max'), ('make_float4', 'float4')]:
        text = re.sub(r'\b'+before+r'\b', after, text)
    text = re.sub(r'\bconst\s+(float4|float|bf16|unsigned|int)\s*\*', r'const device \1*', text)
    text = re.sub(r'(?<!device )\b(float4|float|bf16|unsigned|int)\s*\*', r'device \1*', text)
    text = re.sub(r'\b(const )?float\s*\(&', lambda m: 'thread '+(m[1] or '')+'float (&', text)
    text = re.sub(r'(?<!thread )\bfloat\s*&', 'thread float&', text)
    text = text.replace('const GdnShape&', 'thread const GdnShape&')
    text = text.replace('const device float* vrow', 'const threadgroup float* vrow')
    text = text.replace('threadgroup device ', 'threadgroup ').replace('threadgroup thread ', 'threadgroup ')
    return text.replace('long long', 'size_t')


REDUCE = '''  inline float warp_sum(float v) {
    for (ushort mask = 16; mask > 0; mask >>= 1) v += simd_shuffle_xor(v, mask);
    return v;
  }
  inline float warp_max(float v) {
    for (ushort mask = 16; mask > 0; mask >>= 1) v = max(v, simd_shuffle_xor(v, mask));
    return v;
  }
  inline float block_sum_tree(float v, threadgroup float* scratch, int warps) {
    const uint lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = warp == 0 && lane < uint(warps) ? scratch[lane] : 0.0f;
    if (warp == 0) total = warp_sum(total);
    if (threadIdx.x == 0) scratch[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
  }
  inline float block_max_tree(float v, threadgroup float* scratch, int warps) {
    const uint lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) scratch[warp] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = warp == 0 && lane < uint(warps) ? scratch[lane] : -INFINITY;
    if (warp == 0) total = warp_max(total);
    if (threadIdx.x == 0) scratch[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
  }
'''


def scratch(body):
    count = 0
    def allocation(match):
        nonlocal count
        declarations = []
        for term in match[1].split(','):
            term = term.strip()
            name = re.match(r'\w+', term)[0]
            dims = re.findall(r'\[([^]]+)\]', term)
            size = 1
            for dim in dims:
                size *= int(dim) if dim.isdecimal() else VALUES[dim]
            offset = count
            count += size
            if not dims:
                declarations.append(f'threadgroup float& {name} = shared[{offset}];')
            elif len(dims) == 1:
                declarations.append(f'threadgroup float* {name} = shared + {offset};')
            else:
                tail = ''.join('['+d+']' for d in dims[1:])
                declarations.append(f'threadgroup float (*{name}){tail} = reinterpret_cast<threadgroup float (*){tail}>(shared + {offset});')
        return '\n    '.join(declarations)
    return re.sub(r'__shared__\s+float\s+([^;]+);', allocation, body), max(count, 1)


def native_policy(text, stem):
    """STE through BF16 nodes; native deltanet.metal is the forward, never this recomputation."""
    if stem == 'attn_train':
        before = 'dpq[at] = g * f32(gated[(size_t)r * qw + c]) * (1.0f - sg);'
        assert before in text
        # The rounded gated row cannot recover its pre-gate context. A separate native-probability product fills this.
        return text.replace(before, 'dpq[at] = 0.0f;')
    if stem != 'gdn_train':
        return text
    changes = [
        ('const float s = sigmoidf_(v);\n    return s * (1.0f + v * (1.0f - s));',
         'v = bf(v);\n    const float s = sigmoidf_(v);\n    return bf(s) + v * s * (1.0f - s);'),
        ('const float aq = conv_acc(P, hist, cw, r, cq, s), ak = conv_acc(P, hist, cw, r, ck, s);',
         'const float aq = bf(conv_acc(P, hist, cw, r, cq, s)), ak = bf(conv_acc(P, hist, cw, r, ck, s));'),
        ('bf(aq / (1.0f + expf(-aq)))', 'bf(aq * bf(sigmoidf_(aq)))'),
        ('bf(ak / (1.0f + expf(-ak)))', 'bf(ak * bf(sigmoidf_(ak)))'),
        ('float inv = 1.0f / sqrtf(ss + 1e-6f);\n            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));',
         'const float inv = 1.0f / sqrtf(ss / float(DK) + 1e-6f);\n            const float scale = warp == 0 ? bf(1.0f / float(DK)) : bf(1.0f / sqrtf(float(DK)));'),
        ('out[i] = v4[i] * inv;', 'out[i] = bf(scale * bf(v4[i] * inv));'),
        ('const float av = conv_acc(P, hist, cw, r, c, s);', 'const float av = bf(conv_acc(P, hist, cw, r, c, s));'),
        ('b16(av / (1.0f + expf(-av)))', 'b16(av * bf(sigmoidf_(av)))'),
        ('softplusf_(a + dt_bias[hv])', 'bf(max(bf(a + dt_bias[hv]), 0.0f) + log1pf(expf(-abs(bf(a + dt_bias[hv])))))'),
        ('const float yn = bf(bf(yv * ri) * w);', 'const float yn = bf(yv * ri * w);'),
        ('const float sg = sigmoidf_(f32(P[zc])), d = dout[row + t];\n    dproj[zc] = d * yn * sg * (1.0f - sg);\n    const float dn = d * sg * w;',
         'const float z = f32(P[zc]), sg = sigmoidf_(z), d = dout[row + t];\n    dproj[zc] = d * yn * sg * (1.0f + z * (1.0f - sg));\n    const float dn = d * z * sg * w;'),
        ('const float* saved = ckpt + (size_t)h * nch * size + rb * DK;', 'const float* saved = ckpt + (size_t)h * size + rb * DK;'),
        ('load_state(st, saved + c * size, warp, lane);', 'load_state(st, saved + (size_t)c * s.nv * size, warp, lane);'),
        ('const float inv = 1.0f / sqrtf(ss + 1e-6f), sc = which == 0 ? 1.0f / sqrtf((float)DK) : 1.0f;',
         'const float inv = 1.0f / sqrtf(ss / float(DK) + 1e-6f), sc = which == 0 ? bf(1.0f / float(DK)) : bf(1.0f / sqrtf(float(DK)));'),
        ('sc * inv * (d - inv * inv * x * dot);', 'sc * inv * (d - inv * inv * x * dot / float(DK));'),
        ('x = f32(P[at + s.nv + h]) + dt_bias[h];', 'x = bf(f32(P[at + s.nv + h]) + dt_bias[h]);'),
        ('sx = x > 20.0f ? 1.0f : sigmoidf_(x);', 'sx = sigmoidf_(x);'),
    ]
    for before, after in changes:
        assert before in text, (stem, before)
        text = text.replace(before, after)
    return text


def owned_source(stem):
    folder = ROOT/'zig/kernels/cuda'
    path = folder/f'{stem}.cu'
    original = path.read_text()
    text = original
    dependencies = {}
    if stem == 'attn_train':
        header = folder/'attn_back.cuh'
        shared = header.read_text()
        dependencies[str(header.relative_to(ROOT))] = hashlib.sha256(shared.encode()).hexdigest()
        text = text.replace('#include "attn_back.cuh"', shared)
        text = re.sub(r'^#define TF_ATTN_BACK.*$', '', text, flags=re.M)
        text = re.sub(r'TF_ATTN_BACK\((\w+)\)', r'tf_attn_\1', text)
    header = folder/'common/act.cuh'
    helpers = header.read_text()
    dependencies[str(header.relative_to(ROOT))] = hashlib.sha256(helpers.encode()).hexdigest()
    for name in ('sigmoidf_', 'softplusf_', 'silu_grad'):
        token = f'using tf_common::{name};'
        if token in text:
            found = re.search(r'__device__ __forceinline__ float '+name+r'\([^)]*\)\s*\{', helpers)
            assert found, name
            definition = helpers[found.start():close_brace(helpers, found.end()-1)]
            text = text.replace(token, definition)
    return original, text, dependencies


def emit(stem):
    path = ROOT/'zig/kernels/cuda'/f'{stem}.cu'
    original, owned, dependencies = owned_source(stem)
    native = re.sub(r'\bhalf\b', 'rope_half', native_policy(owned, stem))
    bounds = {name: int(bound) for bound, name in re.findall(r'__launch_bounds__\(\s*(\d+)[^)]*\)\s*(\w+)', native)}
    text = re.sub(r'//[^\n]*', '', native.replace('namespace {', '').replace('}  // namespace', ''))
    text = re.sub(r'^#include.*$', '', text, flags=re.M)
    text = re.sub(r'^#pragma once.*$', '', text, flags=re.M)
    text = re.sub(r'typedef __nv_bfloat16 bf16;', '', text)
    text = re.sub(r'using tf_common::\w+;', '', text)
    text = re.sub(r'__launch_bounds__\([^)]*\)', '', text)
    shapes = []
    for match in list(re.finditer(r'struct (?:GdnShape|Heads)\s*\{[^}]*};', text)):
        shapes.append(match[0])
    for shape in shapes:
        text = text.replace(shape, '')
    constants = re.findall(r'constexpr int [^;]+;', text)
    for constant in constants:
        text = text.replace(constant, '')
    defines = re.findall(r'^#define .+$', text, flags=re.M)
    for define in defines:
        text = text.replace(define, '')
    pattern = r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\((.*?)\)\s*\{'
    matches = list(re.finditer(pattern, text, re.S))
    assert matches
    helpers = text[:matches[0].start()].strip()
    methods, wrappers, metadata = [], [], []
    for match in matches:
        name, signature = match[1], match[2]
        used = {'gdn_train': {'tf_gdn_norm_back', 'tf_gdn_scan_back', 'tf_gdn_qk_back', 'tf_gdn_gates_back', 'tf_gdn_conv_back'},
                'attn_train': {'tf_attn_q', 'tf_attn_gate_back', 'tf_attn_prep_back'}}
        if name not in used[stem]:
            continue
        body = text[match.end():close_brace(text, match.end()-1)-1]
        body, floats = scratch(body)
        params = []
        pointer_args, scalar_args, calls = [], [], []
        for item in signature.split(','):
            item = item.replace('__restrict__', '').strip()
            arg = re.search(r'(\w+)$', item)[1]
            ty = item[:item.rfind(arg)].strip()
            params.append((ty,arg))
            if '*' in ty:
                slot = len(pointer_args)
                pointer_args.append(f'{convert(ty)} {arg} [[buffer({slot})]]')
                calls.append(arg)
            else:
                scalar_args.append((ty,arg))
                calls.append('a.'+arg)
        argument_struct = f'{name}_args'
        fields = '\n'.join(f'  {convert(ty)} {arg};' for ty,arg in scalar_args)
        wrappers.append(f'struct {argument_struct} {{\n{fields}\n}};')
        attributes = pointer_args+[f'constant {argument_struct}& a [[buffer({len(pointer_args)})]]',
            'uint3 threadIdx [[thread_position_in_threadgroup]]',
            'uint3 blockIdx [[threadgroup_position_in_grid]]', 'uint3 blockDim [[threads_per_threadgroup]]']
        wrappers.append(f'[[max_total_threads_per_threadgroup({bounds.get(name,256)})]]\nkernel void {name}(\n  '+',\n  '.join(attributes)+f') {{\n  threadgroup float memory[{floats}];\n  Context c = {{threadIdx, blockIdx, blockDim, memory}};\n  c.{name}('+', '.join(calls)+');\n}')
        methods.append('  void '+name+'('+convert(signature)+') {\n'+convert(body)+'\n  }')
        metadata.append(dict(name=name,pointers=[dict(type=t,name=n) for t,n in params if '*' in t],
            scalars=[dict(type=t,name=n) for t,n in scalar_args],threadgroup_bytes=floats*4,max_threads=bounds.get(name,256)))
    out = '// Owned learner algebra with generated Metal bindings; regenerate with tools/zig/metal_learning_sources.py.\n'
    out += '#include <metal_stdlib>\nusing namespace metal;\ntypedef bfloat bf16;\n'
    out += '\n'.join(shapes)+'\n'+'\n'.join('constant '+v for v in constants)+'\n'+'\n'.join(defines)+'\n'
    out += 'namespace tf_metal_learn {\nstruct Context {\n  uint3 threadIdx, blockIdx, blockDim;\n  threadgroup float* shared;\n'
    out += '  inline float tf_log1p(float x) { return metal::log(1.0f + x); }\n'+REDUCE+convert(helpers)+'\n'+'\n'.join(methods)+'\n};\n}\n'
    tail = 'using tf_metal_learn::Context;\n'+'\n'.join(wrappers)+'\n'
    out = re.sub(r'\n\s*\n\s*\n', '\n\n', out)
    out = '\n'.join(line.rstrip() for line in out.splitlines()) + '\n'
    return out, tail, dict(source=str(path.relative_to(ROOT)),sha256=hashlib.sha256(original.encode()).hexdigest(),dependencies=dependencies,kernels=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Refuse stale generated files without changing them')
    args = parser.parse_args()
    manifests = {}
    files = {}
    for source in ['gdn_train','attn_train']:
        text, tail, meta = emit(source)
        destination = ROOT/'zig/kernels/metal/qwen27'/f'{source}.metal'
        files[destination] = f'#include "{source}_impl.h"\n#include "{source}_entries.h"\n'
        files[destination.with_name(source+'_entries.h')] = tail
        files[destination.with_name(source+'_impl.h')] = text
        manifests[source] = meta
    files[ROOT/'zig/kernels/metal/qwen27/learning-bindings.json'] = json.dumps(manifests,indent=2)+'\n'
    for path, content in files.items():
        if args.check:
            if path.read_text() != content:
                parser.exit(1, 'stale generated source: '+str(path.relative_to(ROOT))+'\n')
        else:
            path.write_text(content)
        print(path.relative_to(ROOT), 'checked' if args.check else 'generated')


if __name__ == '__main__':
    main()
