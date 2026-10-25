// CPU collective and BF16 instruments execute generated HC bodies, without a Metal API or GPU.
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
namespace std {
using ::size_t; using ::memcpy; using ::memset; using ::malloc; using ::free; using ::abort;
template<class R,class... Args> struct coroutine_traits { using promise_type=typename R::promise_type; };
template<class P=void> struct coroutine_handle {
    void* frame;
    static coroutine_handle from_address(void* p) { return {p}; }
    static coroutine_handle from_promise(P& p) { return {__builtin_coro_promise(&p,alignof(P),true)}; }
    void resume() const { __builtin_coro_resume(frame); }
    void destroy() const { __builtin_coro_destroy(frame); }
    bool done() const { return __builtin_coro_done(frame); }
};
template<> struct coroutine_handle<void> {
    void* frame;
    static coroutine_handle from_address(void* p) { return {p}; }
};
struct suspend_always {
    bool await_ready() const noexcept { return false; }
    template<class P> void await_suspend(coroutine_handle<P>) const noexcept {}
    void await_resume() const noexcept {}
};
}
namespace {
using uint = unsigned;
inline bool isnan(float x) { return __builtin_isnan(x); }
inline bool isfinite(float x) { return __builtin_isfinite(x); }
struct uint2 { uint x, y; };
struct uint3 { uint x, y, z; };
using ulong = unsigned long;
using half = _Float16;
using atomic_uint = unsigned;
constexpr int memory_order_relaxed=0;
inline int min(int x,int y){return x<y?x:y;}
inline int clamp(int x,int lo,int hi){return x<lo?lo:(x>hi?hi:x);}
inline void atomic_store_explicit(uint* p,uint v,int){*p=v;}
inline uint atomic_load_explicit(const uint* p,int){return *p;}
inline uint atomic_fetch_add_explicit(uint* p,uint v,int){uint old=*p;*p+=v;return old;}
template<class T,class X> T as_type(X x){static_assert(sizeof(T)==sizeof(X));T out;std::memcpy(&out,&x,sizeof(T));return out;}
struct alignas(16) float4 {
    float x,y,z,w;
    float4(float a,float b,float c,float d):x(a),y(b),z(c),w(d){}
};
struct bfloat {
    uint16_t bits;
    bfloat() = default;
    bfloat(float value) {
        uint32_t u; std::memcpy(&u, &value, 4);
        bits = isnan(value) ? (u >> 16) | 0x40 : (u + 0x7fff + ((u >> 16) & 1)) >> 16;
    }
    operator float() const { uint32_t u = uint32_t(bits) << 16; float v; std::memcpy(&v, &u, 4); return v; }
};
using __nv_bfloat16 = bfloat;
inline float __bfloat162float(bfloat x) { return float(x); }
inline bfloat __float2bfloat16_rn(float x) { return bfloat(x); }
inline float __half2float(half x) { return float(x); }
inline half __float2half_rn(float x) { return half(x); }
using bfloat16_t = bfloat;
namespace metal {
inline float abs(float x) { return fabsf(x); }
inline float exp(float x) { return expf(x); }
inline int min(int x,int y) { return x<y?x:y; }
inline float max(float x,float y) { return fmaxf(x,y); }
inline float sign(float x) { return isnan(x) ? x : (x>0 ? 1.0f : (x<0 ? -1.0f : 0.0f)); }
namespace precise {
inline float exp2(float x) { return exp2f(x); }
inline float cos(float x) { return cosf(x); }
inline float sin(float x) { return sinf(x); }
inline float sqrt(float x) { return sqrtf(x); }
inline float exp(float x) { return expf(x); }
}
inline float rsqrt(float x) { return 1.0f / sqrtf(x); }
inline float log(float x) { return logf(x); }
}
inline float fma(float x,float y,float z) { return fmaf(x,y,z); }
inline float __fadd_rn(float x,float y) { return x+y; }
inline float __fsub_rn(float x,float y) { return x-y; }
inline float __fmul_rn(float x,float y) { return x*y; }
inline float __fdiv_rn(float x,float y) { return x/y; }
inline float __fmaf_rn(float x,float y,float z) { return fmaf(x,y,z); }
inline float __fsqrt_rn(float x) { return sqrtf(x); }
inline float __int_as_float(int x) { float out;std::memcpy(&out,&x,4);return out; }
struct Task {
    struct promise_type {
        static void* operator new(std::size_t n) { void* p = std::malloc(n); if (!p) std::abort(); return p; }
        static void operator delete(void* p) { std::free(p); }
        Task get_return_object() { return {std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() { std::abort(); }
    };
    std::coroutine_handle<promise_type> handle;
};
static int cpu_lane;
static int cpu_phase[1024];
static float cpu_values[1024], cpu_result[1024];
struct Await {
    float value;
    int lane, kind;
    bool await_ready() const { return false; }
    template<class P> void await_suspend(std::coroutine_handle<P>) const {
        cpu_values[lane] = value; cpu_phase[lane] = kind;
    }
    float await_resume() const { return cpu_result[lane]; }
};
static Await simd_sum(float v) { return {v, cpu_lane, 1}; }
static Await cpu_block_sum(float v) { return {v, cpu_lane, 4}; }
static Await prefix_sum(uint v) { return {float(v), cpu_lane, 5}; }
static Await group_barrier() { return {0, cpu_lane, 2}; }
static Await warp_barrier() { return {0, cpu_lane, 3}; }
static void run_tasks(Task* tasks,int threads=256) {
    if (threads<32 || threads>1024 || threads%32) std::abort();
    std::memset(cpu_phase,0,sizeof(cpu_phase));
    for (int i=0;i<threads;++i) {cpu_lane=i;tasks[i].handle.resume();}
    for (;;) {
        bool live=false,progress=false;
        for (int i=0;i<threads;++i) live=live||!tasks[i].handle.done();
        if (!live) break;
        for (int base=0;base<threads;base+=32) {
            const int kind=cpu_phase[base];
            bool ready=kind==1||kind==3||kind==5;
            for (int i=0;i<32;++i) ready=ready&&cpu_phase[base+i]==kind;
            if (!ready) continue;
            float v[32],next[32];
            for(int i=0;i<32;++i)v[i]=cpu_values[base+i];
            if(kind==5){float total=0;for(int i=0;i<32;++i){const float x=v[i];v[i]=total;total+=x;}}
            for(int off=kind==1?16:0;off;off>>=1) {
                for(int i=0;i<32;++i)next[i]=v[i]+v[i^off];
                std::memcpy(v,next,sizeof(v));
            }
            for(int i=0;i<32;++i){cpu_result[base+i]=v[i];cpu_phase[base+i]=0;}
            for(int i=0;i<32;++i){cpu_lane=base+i;tasks[base+i].handle.resume();}
            progress=true;
        }
        bool block=true;
        for(int i=0;i<threads;++i)block=block&&cpu_phase[i]==4;
        if(block) {
            float total=0;
            for(int base=0;base<threads;base+=32){
                float v[32],next[32];for(int i=0;i<32;++i)v[i]=cpu_values[base+i];
                for(int off=16;off;off>>=1){for(int i=0;i<32;++i)next[i]=v[i]+v[i^off];std::memcpy(v,next,sizeof(v));}
                total=total+v[0];
            }
            for(int i=0;i<threads;++i){cpu_result[i]=total;cpu_phase[i]=0;}
            for(int i=0;i<threads;++i){cpu_lane=i;tasks[i].handle.resume();}
            progress=true;
        }
        bool barrier=true;
        for(int i=0;i<threads;++i)barrier=barrier&&cpu_phase[i]==2;
        if(barrier) {
            std::memset(cpu_phase,0,sizeof(cpu_phase));
            for(int i=0;i<threads;++i){cpu_lane=i;tasks[i].handle.resume();}
            progress=true;
        }
        if(!progress)std::abort();
    }
    for (int i=0;i<threads;++i)tasks[i].handle.destroy();
}
}
