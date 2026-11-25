# fmax() of 16-lane vector arguments against fmax() of the same values as uniform scalars, bit for bit. Uses ctypes against libOpenCL (no headers).
import ctypes as C, numpy as np, sys, time
cl = C.CDLL("libOpenCL.so.1")
SRC = b"""
#pragma OPENCL FP_CONTRACT OFF
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void probe(global const float *X, global float *V, global float *S, global const float *Y) {
  const int lane = get_sub_group_local_id(), g = get_group_id(0);
  const float x = X[g * 16 + lane], y = Y[g * 16 + lane];
  V[g * 16 + lane] = fmax(x, y);
  float s[16];
  for (int k = 0; k < 16; k++) s[k] = fmax(sub_group_broadcast(x, k), sub_group_broadcast(y, k));
  S[g * 16 + lane] = s[lane];
}
"""
u32, u64, i32 = C.c_uint32, C.c_uint64, C.c_int32
cl.clGetPlatformIDs.argtypes = [u32, C.POINTER(C.c_void_p), C.POINTER(u32)]
n = u32(0); cl.clGetPlatformIDs(0, None, C.byref(n))
plats = (C.c_void_p * n.value)(); cl.clGetPlatformIDs(n.value, plats, None)
dev = C.c_void_p()
cl.clGetDeviceIDs.argtypes = [C.c_void_p, u64, u32, C.POINTER(C.c_void_p), C.POINTER(u32)]
cl.clGetDeviceIDs(plats[0], 1 << 2, 1, C.byref(dev), None)  # CL_DEVICE_TYPE_GPU, first device (after ZE_AFFINITY_MASK)
err = i32(0)
cl.clCreateContext.restype = C.c_void_p
ctx = cl.clCreateContext(None, 1, C.byref(dev), None, None, C.byref(err)); assert err.value == 0
cl.clCreateCommandQueueWithProperties.restype = C.c_void_p
q = cl.clCreateCommandQueueWithProperties(C.c_void_p(ctx), dev, None, C.byref(err)); assert err.value == 0
cl.clCreateProgramWithSource.restype = C.c_void_p
ln = C.c_size_t(len(SRC)); sp = C.c_char_p(SRC)
prog = cl.clCreateProgramWithSource(C.c_void_p(ctx), 1, C.byref(sp), C.byref(ln), C.byref(err)); assert err.value == 0
r = cl.clBuildProgram(C.c_void_p(prog), 1, C.byref(dev), b"-cl-std=CL3.0 -cl-intel-256-GRF-per-thread", None, None)
if r != 0: sys.exit("build failed %d" % r)
cl.clCreateKernel.restype = C.c_void_p
k = cl.clCreateKernel(C.c_void_p(prog), b"probe", C.byref(err)); assert err.value == 0
if len(sys.argv) > 1 and sys.argv[1] == "bin":  # dump the device binary for disassembly
    sz = C.c_size_t(0); cl.clGetProgramInfo(C.c_void_p(prog), 0x1165, C.sizeof(C.c_size_t), C.byref(sz), None)
    buf = C.create_string_buffer(sz.value); ptr = C.c_char_p(C.addressof(buf))
    cl.clGetProgramInfo(C.c_void_p(prog), 0x1166, C.sizeof(C.c_void_p), C.byref(ptr), None)
    open("expprobe.bin", "wb").write(buf.raw); sys.exit(0)
cl.clCreateBuffer.restype = C.c_void_p
CHUNK = 1 << 26
def buf(sz): 
    b = cl.clCreateBuffer(C.c_void_p(ctx), 1, C.c_size_t(sz), None, C.byref(err)); assert err.value == 0; return b
bx, bv, bs, by = buf(CHUNK * 4), buf(CHUNK * 4), buf(CHUNK * 4), buf(CHUNK * 4)
for i, b in enumerate((bx, bv, bs, by)):
    cl.clSetKernelArg(C.c_void_p(k), i, C.c_size_t(8), C.byref(C.c_void_p(b)))
spec = np.array([0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x3f800000,
                 0xbf800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7f800001, 0xff800001,
                 0x7fc00001, 0x7fffffff, 0xffffffff, 0x40490fdb, 0xc0490fdb, 0x3f000000, 0xbf000000], np.uint32)
px, py = np.meshgrid(spec, spec)
rng = np.random.default_rng(7)
rx = rng.integers(0, 1 << 32, CHUNK - px.size, dtype=np.uint64).astype(np.uint32)
ry = rng.integers(0, 1 << 32, CHUNK - px.size, dtype=np.uint64).astype(np.uint32)
# also random pairs of nearby values (max of close numbers)
ry[: len(ry) // 2] = rx[: len(rx) // 2] ^ rng.integers(0, 4, len(rx) // 2, dtype=np.uint64).astype(np.uint32)
X = np.concatenate([px.ravel(), rx]).astype(np.uint32); Y = np.concatenate([py.ravel(), ry]).astype(np.uint32)
bad = 0; total = 0
for rep in range(8):
    if rep:
        X = rng.integers(0, 1 << 32, CHUNK, dtype=np.uint64).astype(np.uint32)
        Y = np.where(rng.random(CHUNK) < 0.5, X ^ rng.integers(0, 4, CHUNK, dtype=np.uint64).astype(np.uint32), rng.integers(0, 1 << 32, CHUNK, dtype=np.uint64).astype(np.uint32)).astype(np.uint32)
    cnt = CHUNK
    cl.clEnqueueWriteBuffer(C.c_void_p(q), C.c_void_p(bx), 1, 0, C.c_size_t(cnt * 4), X.ctypes.data_as(C.c_void_p), 0, None, None)
    cl.clEnqueueWriteBuffer(C.c_void_p(q), C.c_void_p(by), 1, 0, C.c_size_t(cnt * 4), Y.ctypes.data_as(C.c_void_p), 0, None, None)
    gs = C.c_size_t(cnt); ls = C.c_size_t(16)
    cl.clEnqueueNDRangeKernel(C.c_void_p(q), C.c_void_p(k), 1, None, C.byref(gs), C.byref(ls), 0, None, None)
    v = np.empty(cnt, np.uint32); s = np.empty(cnt, np.uint32)
    cl.clEnqueueReadBuffer(C.c_void_p(q), C.c_void_p(bv), 1, 0, C.c_size_t(cnt * 4), v.ctypes.data_as(C.c_void_p), 0, None, None)
    cl.clEnqueueReadBuffer(C.c_void_p(q), C.c_void_p(bs), 1, 0, C.c_size_t(cnt * 4), s.ctypes.data_as(C.c_void_p), 0, None, None)
    d = np.nonzero(v != s)[0]; bad += len(d); total += cnt
    if len(d): print("DIFF", [(hex(int(X[i])), hex(int(Y[i])), hex(int(v[i])), hex(int(s[i]))) for i in d[:5]])
print("fmax: checked", total, "pairs (all special-value pairs, nearby and random patterns), mismatches", bad)
