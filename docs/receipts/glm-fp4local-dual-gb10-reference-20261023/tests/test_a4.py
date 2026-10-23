"""Independent dense FP64 reference for compact-pack GLM routed A4 prompt kernels; no model loaded."""
import argparse
import ctypes
import json
import statistics
import numpy as np
import torch

ptr = ctypes.c_void_p
I = ctypes.c_int
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('artifact')
parser.add_argument('--baseline-artifact', help='Optional frozen original fatbin for the N64 comparison')
parser.add_argument('--bench', action='store_true', help='Alternating CUDA-event microbench; weights fit in cache')
parser.add_argument('--bench-reps', type=int, default=30)
args = parser.parse_args()
if args.artifact.endswith('.fatbin'):
    # Test the actual native build artifact, not just a separately compiled CUDA bridge.
    torch.cuda.init()
    context_anchor = torch.empty(1, device='cuda')
    driver = ctypes.CDLL('libcuda.so.1')
    module = ptr()
    driver.cuModuleLoad.argtypes = [ctypes.POINTER(ptr),ctypes.c_char_p]
    loaded = driver.cuModuleLoad(ctypes.byref(module),args.artifact.encode())
    assert loaded == 0, ('cuModuleLoad', loaded)
    baseline_module = module
    if args.baseline_artifact:
        baseline_module = ptr()
        loaded = driver.cuModuleLoad(ctypes.byref(baseline_module),args.baseline_artifact.encode())
        assert loaded == 0, ('baseline cuModuleLoad', loaded)
    address, size = ctypes.c_uint64(), ctypes.c_size_t()
    driver.cuModuleGetGlobal_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_size_t),ptr,ctypes.c_char_p]
    assert driver.cuModuleGetGlobal_v2(ctypes.byref(address),ctypes.byref(size),module,b'tf_glm_a4_native_mma') == 0, 'Missing SM121a A4 marker'
    kernels = {}
    driver.cuModuleGetFunction.argtypes = [ctypes.POINTER(ptr),ptr,ctypes.c_char_p]
    driver.cuFuncGetAttribute.argtypes = [ctypes.POINTER(I),I,ptr]
    for parts,name in [(2,'up'),(1,'down')]:
        for tile in (64,128):
            f = ptr()
            symbol = 'tf_glm_a4_prompt_' + name + ('_n128' if tile == 128 else '')
            source = baseline_module if tile == 64 else module
            assert driver.cuModuleGetFunction(ctypes.byref(f),source,symbol.encode()) == 0
            kernels[parts,tile] = f
            resources = {}
            for label,attribute in [('bound',0),('static',1),('local',3),('regs',4)]:
                value = I()
                assert driver.cuFuncGetAttribute(ctypes.byref(value),attribute,f) == 0
                resources[label] = value.value
            print(json.dumps(dict(symbol=symbol,resources=resources,
                                  frozen_baseline=bool(tile == 64 and args.baseline_artifact))),flush=True)
    driver.cuLaunchKernel.argtypes = [ptr,ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,ptr,ctypes.POINTER(ptr),ptr]
    def fn(tile,parts,x,ldx,slots,w,g,inp,k,n,items,counts,members,out,limit,skip,max_items,stream):
        values = [ptr(x),I(ldx),I(slots),ptr(w),ptr(g),ptr(inp),I(k),I(n),ptr(items),ptr(counts),ptr(members),ptr(out),ctypes.c_float(limit),I(skip)]
        args = (ptr*len(values))(*(ctypes.cast(ctypes.byref(v),ptr) for v in values))
        return driver.cuLaunchKernel(kernels[parts,tile],max_items*((n+tile-1)//tile),1,1,tile*2,1,1,0,ptr(stream),args,None)
else:
    lib = ctypes.CDLL(args.artifact)
    fn = lib.run_a4_tile
    fn.argtypes = [I, I, ptr, I, I, ptr, ptr, ptr, I, I, ptr, ptr, ptr, ptr, ctypes.c_float, I, I, ptr]
    fn.restype = I
torch.set_num_threads(4)
torch.manual_seed(123)
lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])

def pack(c, s):
    # Input codes [expert, projection, N, K], output A16 pack; reference never reads this pack.
    E, P, N, K = c.shape
    data = np.zeros((E, N // 32, K // 32, P, 144), dtype=np.uint32)
    cc, ss = c.numpy().astype(np.uint32), s.view(torch.uint8).numpy()
    for cb in range(N // 32):
        for kg in range(K // 32):
            block = data[:, cb, kg]
            for g in range(8):
                for t in range(4):
                    for j in range(4):
                        for h in range(2):
                            for q in range(4):
                                n, k = cb * 32 + j * 8 + g, kg * 32 + h * 16 + t * 4 + q
                                block[:, :, (g * 4 + t) * 4 + j] |= cc[:, :, n, k] << (4 * (2*h + q//2 + 4*(q%2)))
            sb = block[:, :, 128:].copy().view(np.uint8).reshape(E, P, 64)
            for t in range(4):
                for h in range(2):
                    for j in range(4):
                        for v in range(2):
                            sb[:, :, ((t*2+h)*4+j)*2+v] = ss[:, :, cb*32+j*8+t*2+v, kg*2+h]
            block[:, :, 128:] = sb.reshape(E, P, 64).view(np.uint32)
    return torch.from_numpy(data.copy()).to('cuda')

def quant(x, scale):
    x = x.float().reshape(x.shape[0], -1, 16)
    g = 1.0 / scale
    sf = (g * (x.abs().amax(-1) * (1.0 / 6.0))).clamp(max=448).to(torch.float8_e4m3fn).float()
    mul = torch.where(sf != 0, g / sf, 0)
    v = x * mul[..., None]
    a = v.abs()
    c = torch.zeros_like(a, dtype=torch.long)
    for cond, code in [(a > .25,1),(a >= .75,2),(a > 1.25,3),(a >= 1.75,4),
                       (a > 2.5,5),(a >= 3.5,6),(a > 5,7)]:
        c[cond] = code
    c += ((v < 0) & (c != 0)).long() * 8
    return (lut[c] * sf[..., None]).reshape(x.shape[0], -1)

def case(P, N, K, rows, slots, special=False):
    E = 3
    picks = torch.randint(E, (rows * slots,), dtype=torch.int32)
    picks[-1] = E  # shared expert must be skipped, sentinel output preserved
    items, members = [], []
    for e in range(E + 1):
        ids = torch.where(picks == e)[0].tolist()
        for start in range(0, len(ids), 64):
            part = ids[start:start+64]
            items.append([e, len(members), len(part)])
            members += part
    c = torch.randint(16, (E,P,N,K), dtype=torch.uint8)
    s = (2.0 ** torch.randint(-4, 2, (E,P,N,K//16))).to(torch.float8_e4m3fn)
    glob = torch.rand(E,P) * .025 + .005
    ins = torch.tensor([[.25,.5,.375],[.75,.125,.625],[.375,.875,.25]])
    x = (torch.randn(rows, K+16) * (30 if special else 1)).to(torch.bfloat16)
    if special:
        x[0].zero_()
        x[1, :16] = torch.tensor([0,.25,.75,1.25,1.75,2.5,3.5,5,-.25,-.75,-1.25,-1.75,-2.5,-3.5,-5,6])
    xx = x.cuda()
    packed = pack(c, s)
    gg, ii = glob.cuda(), ins.cuda()
    it = torch.tensor(items,dtype=torch.int32,device='cuda')
    ct = torch.tensor([len(items)],dtype=torch.int32,device='cuda')
    mem = torch.tensor(members,dtype=torch.int32,device='cuda')
    guarded = torch.full((rows*slots*N+64,),-123,dtype=torch.bfloat16,device='cuda')
    out = guarded[32:-32].view(rows*slots,N)
    inputs = [xx,packed,gg,ii,it,ct,mem]
    snapshots = [t.clone() for t in inputs]
    def launch(tile):
        err = fn(tile,P,xx.data_ptr(),K+16,slots if P==2 else 0,packed.data_ptr(),gg.data_ptr(),ii.data_ptr(),K,N,
                 it.data_ptr(),ct.data_ptr(),mem.data_ptr(),out.data_ptr(),10.0,E,len(items)+3,
                 torch.cuda.current_stream().cuda_stream)
        assert err == 0, err
    first = None
    for tile in (64,128):
        for repetition in range(5):
            out.fill_(-123)
            launch(tile)
            torch.cuda.synchronize()
            assert (guarded[:32] == -123).all() and (guarded[-32:] == -123).all(), 'Output guard modified'
            for before,after in zip(snapshots,inputs):
                assert torch.equal(before,after), 'Input modified'
            if first is None:
                first = out.clone()
            else:
                assert torch.equal(first.view(torch.uint8),out.view(torch.uint8)), 'Baseline/candidate bytes differ'
    ref = torch.full((rows*slots,N),-123,dtype=torch.bfloat16)
    for e in range(E):
        dest = torch.where(picks == e)[0]
        if dest.numel() == 0:
            continue
        src = dest // slots if P == 2 else dest
        products = []
        for p in range(P):
            qx = quant(x[src,:K], ins[e,p if P==2 else 2])
            qw = lut[c[e,p].long()] * s[e,p].float().repeat_interleave(16,-1)
            alpha = glob[e,p] * ins[e,p if P==2 else 2]
            products.append((qx.double() @ qw.double().T).float() * alpha)
        z = products[0]
        if P == 2:
            gate = z.bfloat16().float().clamp(max=10)
            up = products[1].bfloat16().float().clamp(-10,10)
            z = (gate / (1 + (-gate).exp())).bfloat16().float() * up
        ref[dest] = z.bfloat16()
    got = out.cpu()
    delta = (got.float()-ref.float()).abs()
    # BF16 rounding permits one adjacent value when the FP64 result straddles a rounding midpoint.
    tol = ref.float().abs() * .008 + .002
    assert (delta <= tol).all(), (P,N,K,rows,float(delta.max()), torch.where(delta > tol))
    assert torch.isfinite(got).all()
    assert (got[picks==E] == -123).all()
    print(json.dumps(dict(parts=P,N=N,K=K,rows=rows,slots=slots,special=special,
                          exact_fraction=float((got==ref).float().mean()),max_abs=float(delta.max()),repeats=5,
                          inputs_unchanged=True,guards_intact=True,baseline_byte_equal=True,status='PASS')),flush=True)
    if args.bench and K >= 768:
        for _ in range(5):
            launch(64)
            launch(128)
        samples = {64: [],128: []}
        # Alternating order reduces temperature/order bias. This is NOT a served throughput result.
        for repeat in range(6):
            for tile in ((64,128) if repeat % 2 == 0 else (128,64)):
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.bench_reps):
                    launch(tile)
                end.record()
                end.synchronize()
                samples[tile].append(start.elapsed_time(end)/args.bench_reps)
        print(json.dumps(dict(bench='cache_resident_microbenchmark',parts=P,N=N,K=K,rows=rows,slots=slots,
                              samples_ms=samples,median_ms={t:statistics.median(v) for t,v in samples.items()},
                              candidate_speedup=statistics.median(samples[64])/statistics.median(samples[128]))),flush=True)

for shape in [(2,32,64,1,1,False),(2,32,64,1,2,False),(1,32,64,3,1,False),(2,64,64,2,3,True),(2,96,128,17,4,False),(2,128,256,80,8,True),
              (1,96,256,19,1,True),(2,768,4096,65,2,False),(1,4096,768,67,1,False),
              (2,1024,4096,65,2,False),(1,4096,1024,67,1,False)]:
    case(*shape)
