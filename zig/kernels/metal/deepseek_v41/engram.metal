// Math/layout attribution: DeepSeek AI DeepSeek-V4.1-Flash inference/{engram,model}.py (MIT).
#include <metal_stdlib>
using namespace metal;
inline float rd(const device uchar* p, ulong i, uint dtype) {
  if (dtype == 1u) return ((const device float*)p)[i];
  if (dtype == 2u) return float(((const device half*)p)[i]);
  return as_type<float>(uint(((const device ushort*)p)[i]) << 16);
}

struct EngramLayer {
  uint layer_id, rows;
  ulong multipliers[4];
  uint primes[24], offsets[24];
};
struct EngramHistory { int tail[3]; uint available; };
struct EngramHashArgs { uint tokens, pad, masked; };

inline int engram_token(const device uint* ids, const device uint* map,
                         const device uchar* mask, uint pos, uint masked) {
  return masked && mask[pos] == 0 ? -1 : int(map[ids[pos]]);
}

// One lane per token; history is immutable until all layers have hashed this chunk.
kernel void tf_v41_engram_hash(const device uint* ids [[buffer(0)]],
                               const device uint* map [[buffer(1)]],
                               const device uchar* mask [[buffer(2)]],
                               const device EngramHistory& history [[buffer(3)]],
                               device uint* out [[buffer(4)]],
                               constant EngramLayer& layer [[buffer(5)]],
                               constant EngramHashArgs& args [[buffer(6)]],
                               uint pos [[thread_position_in_grid]]) {
  if (pos >= args.tokens) return;
  bool blocked = false;
  ulong rolling = 0;
  for (uint shift = 0; shift < 4; shift++) {
    int token;
    if (shift <= pos) token = engram_token(ids, map, mask, pos - shift, args.masked);
    else {
      const uint back = shift - pos - 1;
      token = back < history.available ? history.tail[back] : -1;
    }
    blocked = blocked || token == -1;
    const ulong compressed = blocked ? args.pad : uint(token);
    rolling ^= compressed * layer.multipliers[shift];
    if (shift != 0) for (uint head = 0; head < 8; head++) {
      const uint col = (shift - 1) * 8 + head;
      out[size_t(pos) * 24 + col] = uint(rolling % layer.primes[col]) + layer.offsets[col];
    }
  }
}

// Called exactly once after all Engram layers hash a chunk, never once per layer.
kernel void tf_v41_engram_commit(const device uint* ids [[buffer(0)]],
                                 const device uint* map [[buffer(1)]],
                                 const device uchar* mask [[buffer(2)]],
                                 device EngramHistory& history [[buffer(3)]],
                                 constant EngramHashArgs& args [[buffer(4)]]) {
  int tail[3];
  for (uint i = 0; i < 3; i++) {
    tail[i] = i < args.tokens ? engram_token(ids, map, mask, args.tokens - i - 1, args.masked)
      : i - args.tokens < history.available ? history.tail[i - args.tokens] : -1;
  }
  for (uint i = 0; i < 3; i++) history.tail[i] = tail[i];
  history.available = min(3u, history.available + args.tokens);
}

kernel void tf_v41_engram_reset(device EngramHistory& history [[buffer(0)]]) {
  for (uint i = 0; i < 3; i++) history.tail[i] = -1;
  history.available = 0;
}

inline float engram_e4m3(uchar bits) {
  const uint magnitude = bits & 127, exponent = magnitude >> 3, mantissa = magnitude & 7;
  float value = magnitude == 127 ? NAN : exponent == 0 ? float(mantissa) * 0x1p-9f
    : ldexp(1.0f + float(mantissa) * 0.125f, int(exponent) - 7);
  return bits & 128 ? -value : value;
}
inline float engram_ue8m0(uchar bits) {
  return bits == 255 ? NAN : as_type<float>(bits == 0 ? 0x00400000u : uint(bits) << 23);
}

struct EngramGatherArgs {
  ulong weight_stride, scale_stride, output_stride;
  uint tokens, columns, head_dim, scale_block;
};

// Read only the selected E4M3FN rows; one UE8M0 scale per 32 values of each row.
kernel void tf_v41_engram_gather(const device uint* ids [[buffer(0)]],
                                 const device uchar* table [[buffer(1)]],
                                 const device uchar* scales [[buffer(2)]],
                                 device bfloat* out [[buffer(3)]],
                                 constant EngramGatherArgs& args [[buffer(4)]],
                                 uint2 pos [[thread_position_in_grid]]) {
  if (pos.y >= args.tokens || pos.x >= args.columns * args.head_dim) return;
  const uint col = pos.x / args.head_dim, d = pos.x % args.head_dim;
  const size_t row = ids[size_t(pos.y) * args.columns + col];
  const float value = engram_e4m3(table[row * args.weight_stride + d]);
  const float scale = engram_ue8m0(scales[row * args.scale_stride + d / args.scale_block]);
  out[size_t(pos.y) * args.output_stride + pos.x] = bfloat(value * scale);
}

struct EngramGateArgs {
  ulong hidden_stride, kv_stride, output_stride;
  uint dim, copies, masked, weight_dtype;
  float eps;
};

// Each copy gets its own RMS dot gate; all copies share the value following the flattened keys.
kernel void tf_v41_engram_gate(const device bfloat* hidden [[buffer(0)]],
                               const device bfloat* kv [[buffer(1)]],
                               const device uchar* q_weight [[buffer(2)]],
                               const device uchar* k_weight [[buffer(3)]],
                               const device uchar* mask [[buffer(4)]],
                               device bfloat* out [[buffer(5)]],
                               constant EngramGateArgs& args [[buffer(6)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint t [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float3 sums[8];
  threadgroup float gate;
  const size_t hbase = size_t(group.x) * args.hidden_stride + size_t(group.y) * args.dim;
  const size_t kbase = size_t(group.x) * args.kv_stride + size_t(group.y) * args.dim;
  const size_t wbase = size_t(group.y) * args.dim;
  float3 acc = 0;
  for (uint d = t; d < args.dim; d += 256) {
    const float h = float(hidden[hbase + d]), k = float(kv[kbase + d]);
    const float weight = rd(q_weight, wbase + d, args.weight_dtype) * rd(k_weight, wbase + d, args.weight_dtype);
    acc += float3(h * h, k * k, (h * weight) * k);
  }
  acc = float3(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z));
  if (lane == 0) sums[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    acc = lane < 8 ? sums[lane] : float3(0);
    acc = float3(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z));
    if (lane == 0) {
      float rstd;
      {
#pragma clang fp contract(off)
        const float factor = 1.0f / float(args.dim);
        rstd = metal::precise::rsqrt(acc.x * factor + args.eps)
             * metal::precise::rsqrt(acc.y * factor + args.eps);
      }
      const float dot = (acc.z * rstd) * metal::precise::rsqrt(float(args.dim));
      const float signed_root = copysign(metal::precise::sqrt(max(abs(dot), 1e-6f)), dot);
      gate = args.masked && mask[group.x] == 0 ? 0.0f : 1.0f / (1.0f + metal::precise::exp(-signed_root));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const size_t vbase = size_t(group.x) * args.kv_stride + size_t(args.copies) * args.dim;
  const size_t obase = size_t(group.x) * args.output_stride + size_t(group.y) * args.dim;
  for (uint d = t; d < args.dim; d += 256)
    out[obase + d] = bfloat(float(hidden[hbase + d]) + gate * float(kv[vbase + d]));
}
