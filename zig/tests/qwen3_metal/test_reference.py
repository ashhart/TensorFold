"""Small hand-computable checks of the independent oracle's C ABI; no checkpoint or GPU."""
import ctypes as C, math, struct, sys
from reference import F,U,P,W,array
def main():
    lib=C.CDLL(sys.argv[1]);lib.q3_linear.argtypes=[P,P,W,C.c_int,C.c_int,C.c_int];lib.q3_linear.restype=C.c_int
    lib.q3_norm.argtypes=[P,P,W,C.c_int,C.c_int,F];lib.q3_rope.argtypes=[P,C.c_int,C.c_int,C.c_double]
    lib.q3_attention.argtypes=[P,P,P,P,C.c_int,C.c_int,C.c_int];lib.q3_attention.restype=C.c_int
    def bf(x):return struct.unpack('<I',struct.pack('<f',x))[0]>>16
    x=(F*3)(1,2,3);w=(U*6)(*[bf(v) for v in [1,0,2,-1,2,0]]);y=array(2)
    assert lib.q3_linear(y,x,w,1,2,3)==0 and list(y)==[7,3]
    x=(F*256)(*([3]*256));w=(U*128)(*([bf(2)]*128));y=array(256)
    lib.q3_norm(y,x,w,2,128,0);assert max(abs(v-2) for v in y)<1e-6
    q=array(2*2*128);k=array(2*128);v=(F*256)(*([2]*128+[6]*128));y=array(2*2*128)
    assert lib.q3_attention(y,q,k,v,2,2,1)==0
    assert list(y[:256])==[2]*256 and list(y[256:])==[4]*256 # causal and grouped heads
    x=array(256);x[0]=x[128]=1
    lib.q3_rope(x,2,1,1e6);assert x[0]==1 and abs(x[128]-math.cos(1))<1e-6 and abs(x[192]-math.sin(1))<1e-6
    print('PASS independent reference: projection, RMS norm, causal GQA, split-half rotary')

if __name__=="__main__":main()
