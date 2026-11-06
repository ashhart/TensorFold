// Slot steps preserve parent-state arithmetic while visiting the largest child last and recycling dead ancestors.
struct TfDeltaSlotStep { uint row; int read_slot; int write_slot; };
// SLOTS register slots: a chain needs one, a tree its peak of live branch states; fewer slots, fewer selects a node.
template <uint SLOTS>
kernel void tf_delta_gdn_tree_slots(device const bfloat* query [[buffer(0)]],
                         device const bfloat* key [[buffer(1)]],
                         device const bfloat* value [[buffer(2)]],
                         device const float* decay [[buffer(3)]],
                         device const bfloat* mixing [[buffer(4)]],
                         device const float* committed [[buffer(5)]],
                         device const int* parents [[buffer(6)]],
                         device const TfDeltaSegment* segments [[buffer(7)]],
                         constant TfDeltaParams& p [[buffer(8)]],
                         device bfloat* output [[buffer(9)]],
                         device float* snapshots [[buffer(10)]],
                         device const TfDeltaSlotStep* steps [[buffer(11)]],
                         device float* advanced [[buffer(12)]],
                         device float* before [[buffer(13)]],
                         uint3 position [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint dimension = position.y, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + lane * 4;
  float4 root;
  for (uint part = 0; part < 4; ++part) root[part] = committed[span.state_slot * state_stride + column + part];
  float4 kept[SLOTS];
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < SLOTS; ++j) kept[j] = 0.0f;
  float4 last = root;
  for (uint at = 0; at < span.rows; ++at) {
    const TfDeltaSlotStep step = steps[span.first + at];
    float4 s = root;
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < SLOTS; ++j) if (int(j) == step.read_slot) s = kept[j];
    s = tf_delta_tree_node(s, query, key, value, decay, mixing, p, output, step.row, head, key_head, dimension, lane);
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < SLOTS; ++j) if (int(j) == step.write_slot) kept[j] = s;
    last = s;
  }
  // `advanced` aliases `committed`: each thread rewrites only the words it read
  if (p.advance != 0 && span.rows == 1)
    for (uint part = 0; part < 4; ++part) {
      const uint at = span.state_slot * state_stride + column + part;
      before[at] = root[part];
      advanced[at] = last[part];
    }
}
#define TF_TREE_SLOTS(N) template [[host_name("tf_delta_gdn_tree_slots_" #N)]] kernel void tf_delta_gdn_tree_slots<N>(device const bfloat*, device const bfloat*, device const bfloat*, device const float*, device const bfloat*, device const float*, device const int*, device const TfDeltaSegment*, constant TfDeltaParams&, device bfloat*, device float*, device const TfDeltaSlotStep*, device float*, device float*, uint3, uint);
TF_TREE_SLOTS(1)
TF_TREE_SLOTS(2)
TF_TREE_SLOTS(4)
TF_TREE_SLOTS(8)
TF_TREE_SLOTS(16)

// The slot scan for 128-wide heads, 4 value columns a simdgroup; a flow word holds a row and slots to restore and save.
template <uint SLOTS>
kernel void tf_delta_gdn_columns(device const vec<bfloat, 4>* query [[buffer(0)]],
                                 device const vec<bfloat, 4>* key [[buffer(1)]],
                                 device const bfloat* value [[buffer(2)]],
                                 device const float* decay [[buffer(3)]],
                                 device const bfloat* mixing [[buffer(4)]],
                                 device const float4* committed [[buffer(5)]],
                                 device const TfDeltaSegment* segments [[buffer(6)]],
                                 constant TfDeltaParams& p [[buffer(7)]],
                                 device bfloat* output [[buffer(8)]],
                                 device const uint* flows [[buffer(9)]],
                                 uint3 group [[threadgroup_position_in_grid]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 kept[SLOTS][4][32];
  const uint head = group.z % p.nv, first = group.y * 4, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[group.z / p.nv];
  float4 s[4];
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < 4; k++) s[k] = committed[(span.state_slot * p.nv * p.dv + head * p.dv + first + k) * 32 + lane];
  for (uint at = 0; at < span.rows; at++) {
    const uint flow = flows[span.first + at], row = flow & 0xffffu;
    if (SLOTS > 1) {
      const uint restore = (flow >> 16) & 0xffu;
      if (restore != 0) {
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < 4; k++) s[k] = kept[restore - 1][k][lane];
      }
    }
    const uint cell = row * p.nv + head, key_at = (row * p.nk + key_head) * 32 + lane;
    const float forget = decay[cell], mix = float(mixing[cell]);
    const float4 ki = float4(key[key_at]), qi = float4(query[key_at]);
    const float4 incoming = float4(*(device const vec<bfloat, 4>*)(value + cell * p.dv + first));
    float4 results;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 4; k++) {
      float4 t = s[k];
      float remembered = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint part = 0; part < 4; part++) {
        t[part] = t[part] * forget;
        remembered += t[part] * ki[part];
      }
      remembered = simd_sum(remembered);
      const float correction = (incoming[k] - remembered) * mix;
      float result = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint part = 0; part < 4; part++) {
        t[part] = t[part] + ki[part] * correction;
        result += t[part] * qi[part];
      }
      s[k] = t;
      results[k] = simd_sum(result);
    }
    if (lane == 0) *(device vec<bfloat, 4>*)(output + cell * p.dv + first) = vec<bfloat, 4>(results);
    if (SLOTS > 1) {
      const uint save = flow >> 24;
      if (save != 0) {
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < 4; k++) kept[save - 1][k][lane] = s[k];
      }
    }
  }
}
#define TF_COLUMNS(N) template [[host_name("tf_delta_gdn_columns_" #N)]] kernel void tf_delta_gdn_columns<N>(device const vec<bfloat, 4>*, device const vec<bfloat, 4>*, device const bfloat*, device const float*, device const bfloat*, device const float4*, device const TfDeltaSegment*, constant TfDeltaParams&, device bfloat*, device const uint*, uint3, ushort);
TF_COLUMNS(1)
TF_COLUMNS(2)
TF_COLUMNS(4)

// tf_delta_gdn_pre for 128-wide heads, four taps and bf16 weights: a lane's channels load together, and no tails.
kernel void tf_delta_gdn_pre_vectors(device const vec<bfloat, 4>* projection [[buffer(0)]],
                                     device const bfloat* a [[buffer(1)]],
                                     device const bfloat* b [[buffer(2)]],
                                     device const vec<bfloat, 4>* history [[buffer(3)]],
                                     device const packed_ushort4* weights [[buffer(4)]],
                                     device const bfloat* decay_weight [[buffer(5)]],
                                     device const bfloat* time_bias [[buffer(6)]],
                                     device const packed_uint4* windows [[buffer(7)]],
                                     device const uint* row_slot [[buffer(8)]],
                                     constant TfDeltaParams& p [[buffer(9)]],
                                     device vec<bfloat, 4>* query [[buffer(10)]],
                                     device vec<bfloat, 4>* key [[buffer(11)]],
                                     device vec<bfloat, 4>* value [[buffer(12)]],
                                     device float* decay [[buffer(13)]],
                                     device bfloat* mixing [[buffer(14)]],
                                     uint3 group [[threadgroup_position_in_grid]],
                                     ushort lane [[thread_index_in_simdgroup]]) {
  const uint row = group.z, head = group.y, keep = p.taps - 1;
  const uint channels = (2 * p.nk * p.dk + p.nv * p.dv) / 4, at = head * 32 + lane;
  const uint4 sources = uint4(windows[row]);
  float4 taps[4];
  _Pragma("clang loop unroll(full)")
  for (uint tap = 0; tap < 4; tap++) {
    const uint source = sources[tap];
    device const vec<bfloat, 4>* from = source < keep ? history + (row_slot[row] * keep + source) * channels : projection + (source - keep) * channels;
    taps[tap] = float4(from[at]);
  }
  float activated[4];
  _Pragma("clang loop unroll(full)")
  for (uint part = 0; part < 4; part++) {
    const float4 w = as_type<float4>(uint4(ushort4(weights[head * p.dk + lane * 4 + part])) << 16);
    float result = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint tap = 0; tap < 4; tap++) result += w[tap] * taps[tap][part];
    const float rounded = float(bfloat(result));
    const float sigmoid = float(bfloat(1.0f / (1.0f + metal::exp(-rounded))));
    activated[part] = float(bfloat(rounded * sigmoid));
  }
  if (head < 2 * p.nk) {
    float square = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint part = 0; part < 4; part++) square += activated[part] * activated[part];
    square = simd_sum(square);
    const float inverse = metal::rsqrt(square / float(p.dk) + 1e-6f);
    const float scale = head < p.nk ? float(bfloat(1.0f / float(p.dk))) : float(bfloat(metal::rsqrt(float(p.dk))));
    float4 out;
    _Pragma("clang loop unroll(full)")
    for (uint part = 0; part < 4; part++) out[part] = scale * float(bfloat(activated[part] * inverse));
    const uint position = (row * p.nk + head % p.nk) * 32 + lane;
    if (head < p.nk) query[position] = vec<bfloat, 4>(out);
    else key[position] = vec<bfloat, 4>(out);
    return;
  }
  const uint output_head = head - 2 * p.nk;
  value[(row * p.nv + output_head) * 32 + lane] = vec<bfloat, 4>(float4(activated[0], activated[1], activated[2], activated[3]));
  if (lane == 0) {
    const uint cell = row * p.nv + output_head, gate_at = p.zba_stride ? row * p.zba_stride + output_head : cell;
    const float shifted = float(bfloat(float(a[gate_at]) + float(time_bias[output_head])));
    const float softplus = float(bfloat(metal::max(shifted, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(shifted)))));
    decay[cell] = metal::exp(-metal::exp(float(decay_weight[output_head])) * softplus);
    mixing[cell] = bfloat(1.0f / (1.0f + metal::exp(-float(b[gate_at]))));
  }
}
