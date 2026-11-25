# Exhaustive check that exp() of a 16-lane vector argument and exp() of the same value as a uniform scalar agree bit
# for bit, over all 2^32 float bit patterns (NaNs and infinities included). Uses ctypes against libOpenCL (no headers).
import ctypes as C, numpy as np, sys, time
cl = C.CDLL("libOpenCL.so.1")
SRC = b"""
#pragma OPENCL FP_CONTRACT OFF
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void probe(global const float *X, global float *V, global float *S) {
  const int lane = get_sub_group_local_id(), g = get_group_id(0);
  const float x = X[g * 16 + lane];
  V[g * 16 + lane] = exp(x);
  float s[16];
  for (int k = 0; k < 16; k++) s[k] = exp(sub_group_broadcast(x, k));
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
bx, bv, bs = buf(CHUNK * 4), buf(CHUNK * 4), buf(CHUNK * 4)
for i, b in enumerate((bx, bv, bs)):
    cl.clSetKernelArg(C.c_void_p(k), i, C.c_size_t(8), C.byref(C.c_void_p(b)))
bad = 0; total = 0
# every float bit pattern, NaNs and infinities included
starts = list(range(0, 1 << 32, CHUNK))
t0 = time.time()
for st in starts:
    cnt = CHUNK
    x = (np.arange(cnt, dtype=np.uint64) + st).astype(np.uint32)
    cnt = (len(x) + 15) // 16 * 16
    xp = np.zeros(cnt, np.uint32); xp[:len(x)] = x
    cl.clEnqueueWriteBuffer(C.c_void_p(q), C.c_void_p(bx), 1, 0, C.c_size_t(cnt * 4), xp.ctypes.data_as(C.c_void_p), 0, None, None)
    gs = C.c_size_t(cnt); ls = C.c_size_t(16)
    cl.clEnqueueNDRangeKernel(C.c_void_p(q), C.c_void_p(k), 1, None, C.byref(gs), C.byref(ls), 0, None, None)
    v = np.empty(cnt, np.uint32); s = np.empty(cnt, np.uint32)
    cl.clEnqueueReadBuffer(C.c_void_p(q), C.c_void_p(bv), 1, 0, C.c_size_t(cnt * 4), v.ctypes.data_as(C.c_void_p), 0, None, None)
    cl.clEnqueueReadBuffer(C.c_void_p(q), C.c_void_p(bs), 1, 0, C.c_size_t(cnt * 4), s.ctypes.data_as(C.c_void_p), 0, None, None)
    n_ = len(x); d = np.nonzero(v[:n_] != s[:n_])[0]
    bad += len(d); total += n_
    if len(d): print("DIFF at", [(hex(int(x[i])), hex(int(v[i])), hex(int(s[i]))) for i in d[:5]])
    if st == 0: print("exp(+0) vector %08x scalar %08x (1.0 = 3f800000)" % (v[0], s[0]))
    if st == 0x80000000: print("exp(-0) vector %08x scalar %08x" % (v[0], s[0]))
print("checked", total, "values, mismatches", bad, "in %.1f s" % (time.time() - t0))
