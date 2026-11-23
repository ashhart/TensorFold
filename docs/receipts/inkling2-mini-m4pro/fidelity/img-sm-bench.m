// bench <metal file> <n> <lc> <reps>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <objc/message.h>
#include <string.h>
#include <math.h>
static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 11); }
static float gauss(void) { float s = 0; for (int i = 0; i < 4; i++) s += (rnd() & 0xFFFFF) / 1048576.0f - 0.5f; return s * 1.7f; }
static id<MTLDevice> dev; static id<MTLCommandQueue> q; static id<MTLLibrary> lib;
static id<MTLComputePipelineState> P(NSString *n) {
  NSError *e = nil; id<MTLFunction> f = [lib newFunctionWithName:n];
  id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&e]; if (!p) { NSLog(@"%@ %@", n, e); exit(1); } return p; }
static uint16_t bf(float x) { uint32_t u; memcpy(&u, &x, 4); u += 0x7fff + ((u >> 16) & 1); return u >> 16; }
static float unbf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
typedef void (^Enc)(id<MTLComputeCommandEncoder>);
static double run(Enc enc) {
  id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  enc(e); [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  if (cb.error) { NSLog(@"%@", cb.error); exit(1); }
  return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
}
static double med(int reps, Enc enc) {
  double t[64]; run(enc); for (int i = 0; i < reps; i++) t[i] = run(enc);
  for (int i = 0; i < reps; i++) for (int j = i + 1; j < reps; j++) if (t[j] < t[i]) { double x = t[i]; t[i] = t[j]; t[j] = x; }
  return t[reps / 2];
}
static uint64_t fnv(const void *p, size_t n) { const uint8_t *b = p; uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; } return h; }
int main(int argc, char **argv) {
  @autoreleasepool {
    NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:nil];
    uint32_t n = atoi(argv[2]), lc = atoi(argv[3]); int reps = atoi(argv[4]); const uint32_t H = 16, D = 64;
    dev = MTLCreateSystemDefaultDevice(); q = [dev newCommandQueue];
    MTLCompileOptions *o = [MTLCompileOptions new]; o.mathMode = MTLMathModeSafe; o.languageVersion = MTLLanguageVersion4_0;
    ((void(*)(id,SEL,long))objc_msgSend)(o, sel_registerName("setMathFloatingPointFunctions:"), 0);
    NSError *err = nil; lib = [dev newLibraryWithSource:src options:o error:&err];
    if (!lib) { NSLog(@"%@", err); return 1; }
    size_t qn = (size_t)H * n * D, sn = (size_t)H * lc * n, on = (size_t)H * lc * D;
    id<MTLBuffer> Q = [dev newBufferWithLength:qn * 4 options:MTLResourceStorageModeShared], K = [dev newBufferWithLength:qn * 4 options:MTLResourceStorageModeShared], V = [dev newBufferWithLength:qn * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> S32 = [dev newBufferWithLength:sn * 4 options:MTLResourceStorageModeShared], S16 = [dev newBufferWithLength:sn * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> ST = [dev newBufferWithLength:(size_t)H * lc * 8 options:MTLResourceStorageModeShared]; id<MTLBuffer> O3 = [dev newBufferWithLength:(size_t)H * lc * D * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> O0 = [dev newBufferWithLength:on * 4 options:MTLResourceStorageModeShared], O1 = [dev newBufferWithLength:on * 4 options:MTLResourceStorageModeShared], O2 = [dev newBufferWithLength:on * 4 options:MTLResourceStorageModeShared];
    float *qf = Q.contents, *kf = K.contents, *vf = V.contents;
    for (size_t i = 0; i < qn; i++) { qf[i] = unbf(bf(gauss())); kf[i] = unbf(bf(gauss())); vf[i] = unbf(bf(gauss())); }
    float scale = 1.0f / sqrtf(64.0f); int32_t sb; memcpy(&sb, &scale, 4);
    static int32_t pqk[16]; static int32_t ppv[16]; int32_t tq[16] = { (int)lc, (int)n, 64, 64, 64, (int)n, 0, 0, 0, 0, (int)(n * 64), (int)(n * 64), (int)(lc * n), sb, 0, 0 };
    int32_t tp[16] = { (int)lc, 64, (int)n, (int)n, 64, 64, 0, 0, 0, 0, (int)(lc * n), (int)(n * 64), (int)(lc * 64), 0, 0, 0 };
    id<MTLComputePipelineState> stb = P(@"st_bf"), pve = P(@"pv_ex"); id<MTLComputePipelineState> qk32 = P(@"ik2_gemm_f32_nt_00"), pv32 = P(@"ik2_gemm_f32_nn_00"), sm32 = P(@"ik2v_softmax"), qkb = P(@"qk_bf"), pvb = P(@"pv_bf"), smb = P(@"sm_bf"), sms = P(@"sm_sg");
    memcpy(pqk, tq, 64); memcpy(ppv, tp, 64);
    MTLSize gq = MTLSizeMake((n + 63) / 64, (lc + 63) / 64, H), gp = MTLSizeMake(1, (lc + 63) / 64, H), t256 = MTLSizeMake(256, 1, 1);
    uint32_t rows = H * lc; float sc = scale;
    Enc eQk32 = ^(id<MTLComputeCommandEncoder> e) { [e setComputePipelineState:qk32]; [e setBuffer:Q offset:0 atIndex:0]; [e setBuffer:K offset:0 atIndex:1]; [e setBytes:pqk length:64 atIndex:2]; [e setBuffer:S32 offset:0 atIndex:3]; int32_t pp[16]; memcpy(pp, pqk, 64); [e dispatchThreadgroups:gq threadsPerThreadgroup:t256]; };
    Enc eSm32 = ^(id<MTLComputeCommandEncoder> e) { [e setComputePipelineState:sm32]; [e setBuffer:S32 offset:0 atIndex:0]; [e setBytes:&n length:4 atIndex:1]; [e setBytes:&sc length:4 atIndex:2]; [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:t256]; };
    Enc ePv32 = ^(id<MTLComputeCommandEncoder> e) { [e setComputePipelineState:pv32]; [e setBuffer:S32 offset:0 atIndex:0]; [e setBuffer:V offset:0 atIndex:1]; [e setBytes:ppv length:64 atIndex:2]; [e setBuffer:O0 offset:0 atIndex:3]; [e dispatchThreadgroups:gp threadsPerThreadgroup:t256]; };
    Enc eQkb = ^(id<MTLComputeCommandEncoder> e) { [e setComputePipelineState:qkb]; [e setBuffer:Q offset:0 atIndex:0]; [e setBuffer:K offset:0 atIndex:1]; [e setBytes:pqk length:64 atIndex:2]; [e setBuffer:S16 offset:0 atIndex:3]; [e setBuffer:ST offset:0 atIndex:4]; [e dispatchThreadgroups:gq threadsPerThreadgroup:t256]; };
    Enc eSmb = ^(id<MTLComputeCommandEncoder> e) { [e setComputePipelineState:smb]; [e setBuffer:S16 offset:0 atIndex:0]; [e setBytes:&n length:4 atIndex:1]; [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:t256]; };
    Enc eSms = ^(id<MTLComputeCommandEncoder> e) { [e setComputePipelineState:sms]; [e setBuffer:S16 offset:0 atIndex:0]; [e setBytes:&n length:4 atIndex:1]; [e setBytes:&rows length:4 atIndex:2]; [e dispatchThreadgroups:MTLSizeMake((rows + 7) / 8, 1, 1) threadsPerThreadgroup:t256]; };
    Enc ePvb = ^(id<MTLComputeCommandEncoder> e) { [e setComputePipelineState:pvb]; [e setBuffer:S16 offset:0 atIndex:0]; [e setBuffer:V offset:0 atIndex:1]; [e setBytes:ppv length:64 atIndex:2]; [e setBuffer:O1 offset:0 atIndex:3]; [e setBuffer:ST offset:0 atIndex:4]; [e dispatchThreadgroups:gp threadsPerThreadgroup:t256]; };
    double gb = (double)sn * 4 / 1e6;
    { run(eQk32); run(eQkb); uint16_t *a16 = S16.contents; float *a32 = S32.contents; size_t bad = 0;
      for (size_t i = 0; i < sn; i++) { float v = unbf(bf(unbf(bf(a32[i])) * scale)); uint16_t w = bf(v); if (a16[i] != w) bad++; }
      printf("scores: %zu of %zu differ from bf(bf(fp32 dot) * scale)\n", bad, sn); }
    printf("n=%u lc=%u heads=16: fp32 score chunk %.1f MB\n", n, lc, gb);
    printf("base  qk32 %.3f ms\n", med(reps, eQk32));
    run(eQk32); printf("base  sm32 %.3f ms (floor r+w at 250GB/s %.3f)\n", med(reps, ^(id<MTLComputeCommandEncoder> e) { eSm32(e); }), 2 * gb / 250.0);
    // fresh scores again for correctness, then softmax once
    run(eQk32); run(eSm32); printf("base  pv32 %.3f ms\n", med(reps, ePv32)); run(ePv32);
    uint64_t h32 = fnv(S32.contents, sn * 4), hO0 = fnv(O0.contents, on * 4);
    // new
    printf("new   qk_bf %.3f ms\n", med(reps, eQkb));
    // restore fresh scores per timed softmax: timings of idempotent re-softmax differ numerically but not in cost
    run(eQkb); printf("new   sm_bf %.3f ms\n", med(reps, ^(id<MTLComputeCommandEncoder> e) { eSmb(e); }));
    run(eQkb); printf("new   sm_sg %.3f ms\n", med(reps, ^(id<MTLComputeCommandEncoder> e) { eSms(e); }));
    run(eQkb); run(eSmb); printf("new   pv_bf %.3f ms\n", med(reps, ePvb)); run(ePvb);
    uint16_t *s16 = S16.contents; float *s32 = S32.contents; size_t bad = 0;
    for (size_t i = 0; i < sn; i++) if (memcmp(&s32[i], &(uint32_t[]){ (uint32_t)s16[i] << 16 }, 4)) bad++;
    printf("sm_bf probs vs fp32 probs: %zu of %zu differ; O hash %016llx vs %016llx\n", bad, sn, (unsigned long long)fnv(O1.contents, on * 4), (unsigned long long)hO0);
    run(eQkb); run(eSms); run(^(id<MTLComputeCommandEncoder> e) { [e setComputePipelineState:pvb]; [e setBuffer:S16 offset:0 atIndex:0]; [e setBuffer:V offset:0 atIndex:1]; [e setBytes:ppv length:64 atIndex:2]; [e setBuffer:O2 offset:0 atIndex:3]; [e dispatchThreadgroups:gp threadsPerThreadgroup:t256]; });
    bad = 0; for (size_t i = 0; i < sn; i++) if (memcmp(&s32[i], &(uint32_t[]){ (uint32_t)s16[i] << 16 }, 4)) bad++;
    Enc eSt = ^(id<MTLComputeCommandEncoder> e) { [e setComputePipelineState:stb]; [e setBuffer:S16 offset:0 atIndex:0]; [e setBytes:&n length:4 atIndex:1]; [e setBuffer:ST offset:0 atIndex:2]; [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:t256]; };
    Enc ePve = ^(id<MTLComputeCommandEncoder> e) { [e setComputePipelineState:pve]; [e setBuffer:S16 offset:0 atIndex:0]; [e setBuffer:V offset:0 atIndex:1]; [e setBytes:ppv length:64 atIndex:2]; [e setBuffer:O3 offset:0 atIndex:3]; [e setBuffer:ST offset:0 atIndex:4]; [e dispatchThreadgroups:gp threadsPerThreadgroup:t256]; };
    run(eQkb); printf("fused st_bf %.3f ms\n", med(reps, eSt)); run(eSt); printf("fused pv_ex %.3f ms\n", med(reps, ePve)); run(ePve);
    printf("fused O hash %016llx\n", (unsigned long long)fnv(O3.contents, on * 4));
    printf("sm_sg probs vs fp32 probs: %zu differ; O hash %016llx\n", bad, (unsigned long long)fnv(O2.contents, on * 4)); (void)h32;
  }
}
