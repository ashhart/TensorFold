[[kernel]] void tf_markov_pick_two(const device float2* parts [[buffer(0)]],
                                  const device float2* runners [[buffer(1)]],
                                  device uint* ids [[buffer(2)]],
                                  device MarkovRank* ranks [[buffer(3)]],
                                  device RankState* state [[buffer(4)]],
                                  constant RankArgs& a [[buffer(5)]]) {
  if (a.guard && !state->score_step) return;
  MarkovRank rank = {0xffffffffu, 0xffffffffu, -INFINITY, -INFINITY};
  for (uint j = 0; j < a.groups; ++j) {
    const float2 p = parts[j];
    const uint id = as_type<uint>(p.y);
    if (tf_markov_better(p.x, id, rank.first_score, rank.first)) {
      rank.first_score = p.x;
      rank.first = id;
    }
  }
  for (uint j = 0; j < a.groups; ++j) {
    const float2 first = parts[j];
    const float2 p = as_type<uint>(first.y) == rank.first ? runners[j] : first;
    const uint id = as_type<uint>(p.y);
    if (tf_markov_better(p.x, id, rank.second_score, rank.second)) {
      rank.second_score = p.x;
      rank.second = id;
    }
  }
  ranks[a.row] = rank;
  ids[a.row + 1] = rank.first;
  if (a.margin) *state = estimate_rank_acceptance(*state, a, rank);
}

// The reference recalculates all scores with the winning ID excluded.
[[kernel]] void tf_markov_audit_two(const device uint2* reference [[buffer(0)]],
                                   const device uint2* actual [[buffer(1)]],
                                   const device MarkovRank* ranks [[buffer(2)]],
                                   device uint* mismatch [[buffer(3)]],
                                   constant uint4& a [[buffer(4)]]) {
  if (a.z == 0) {
    for (uint j = 0; j < a.x; ++j)
      if (any(reference[j] != actual[j])) mismatch[0] = 1;
    return;
  }
  float best = -INFINITY;
  uint id = 0xffffffffu;
  for (uint j = 0; j < a.x; ++j) {
    const uint2 p = reference[j];
    const float score = as_type<float>(p.x);
    if (tf_markov_better(score, p.y, best, id)) {
      best = score;
      id = p.y;
    }
  }
  const MarkovRank rank = ranks[a.y];
  if (id != rank.second || as_type<uint>(best) != as_type<uint>(rank.second_score)) mismatch[0] = 1;
}

[[kernel]] void ds_markov_top_two(const device bfloat* LOGITS [[buffer(0)]], const device bfloat* EMB [[buffer(1)]],
                          const device bfloat* HEAD [[buffer(2)]], const device uint* IDS [[buffer(3)]],
                          device float2* PART [[buffer(4)]], constant MarkovArgs& a [[buffer(5)]], device float2* RUNNER [[buffer(6)]],
                          uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float e[256];
  threadgroup float bv[8];
  threadgroup int bi[8];
  threadgroup float rv[8];
  threadgroup uint ri[8];
  const uint tok = IDS[a.step];
  for (int j = int(t); j < a.rank; j += 256) e[j] = float(EMB[size_t(tok) * a.rank + j]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int v = int(tg) * 256 + int(t);
  float val = -INFINITY;
  int id = 0x7fffffff;
  if (v < a.vocab && uint(v) != a.exclude) {
    const device bfloat* h = HEAD + size_t(v) * a.rank;
    float bias = 0.0f;
    for (int j = 0; j < a.rank; j++) bias += e[j] * float(h[j]);
    val = float(LOGITS[size_t(a.step) * a.vocab + v]) + bias;
    id = v;
  }
  const float own_score = val;
  const int own_id = id;
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_down(val, o);
    const int oi = simd_shuffle_down(id, o);
    if (ov > val || (ov == val && oi < id)) {
      val = ov;
      id = oi;
    }
  }
  if (lane == 0) {
    bv[sg] = val;
    bi[sg] = id;
  }
  const int best_id = simd_broadcast(id, 0);
  float next = own_id == best_id ? -INFINITY : own_score;
  uint next_id = own_id == best_id ? 0x7fffffffu : uint(own_id);
  for (ushort o = 16; o > 0; o >>= 1) {
    const float score = simd_shuffle_down(next, o);
    const uint candidate = simd_shuffle_down(next_id, o);
    if (tf_markov_better(score, candidate, next, next_id)) {
      next = score;
      next_id = candidate;
    }
  }
  if (lane == 0) {
    rv[sg] = next;
    ri[sg] = next_id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float m = bv[0];
    int mi = bi[0];
    for (int k = 1; k < 8; k++)
      if (bv[k] > m || (bv[k] == m && bi[k] < mi)) {
        m = bv[k];
        mi = bi[k];
      }
    PART[tg] = float2(m, as_type<float>(mi));
    float second = -INFINITY;
    uint second_id = 0x7fffffffu;
    for (int k = 0; k < 8; ++k) {
      const bool exclude_best = bi[k] == mi;
      const float score = exclude_best ? rv[k] : bv[k];
      const uint candidate = exclude_best ? ri[k] : uint(bi[k]);
      if (tf_markov_better(score, candidate, second, second_id)) {
        second = score;
        second_id = candidate;
      }
    }
    RUNNER[tg] = float2(second, as_type<float>(second_id));
  }
}

[[kernel]] void tf_markov_cache_score(const device bfloat* LOGITS [[buffer(0)]], const device bfloat* EMB [[buffer(1)]],
                          const device bfloat* HEAD [[buffer(2)]], const device uint* IDS [[buffer(3)]],
                          device float2* PART [[buffer(4)]], constant MarkovArgs& a [[buffer(5)]], device float2* RUNNER [[buffer(6)]],
                          device uint* tags [[buffer(7)]], device float* cached_bias [[buffer(8)]],
                          device atomic_uint* audit_counts [[buffer(9)]], constant uint2& cache [[buffer(10)]],
                          uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float e[256];
  threadgroup float bv[8];
  threadgroup int bi[8];
  threadgroup float rv[8];
  threadgroup uint ri[8];
  const uint tok = IDS[a.step];
  const uint slot = (tok * 2654435761u) >> (32u - cache.x);
  const uint tag = tg * (1u << cache.x) + slot;
  const bool hit = tags[tag] == tok;
  const bool audit = cache.y != 0;
  if (!hit || audit) {
    for (int j = int(t); j < a.rank; j += 256) e[j] = float(EMB[size_t(tok) * a.rank + j]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int v = int(tg) * 256 + int(t);
  float val = -INFINITY;
  int id = 0x7fffffff;
  if (v < a.vocab && uint(v) != a.exclude) {
    const size_t address = size_t(slot) * a.vocab + v;
    float bias = hit ? cached_bias[address] : 0.0f;
    if (!hit || audit) {
      const device bfloat* h = HEAD + size_t(v) * a.rank;
      float exact = 0.0f;
      for (int j = 0; j < a.rank; j++) exact += e[j] * float(h[j]);
      if (!hit) {
        cached_bias[address] = exact;
        bias = exact;
      } else if (as_type<uint>(bias) != as_type<uint>(exact)) {
        atomic_store_explicit(audit_counts + 2, 1u, memory_order_relaxed);
      }
    }
    val = float(LOGITS[size_t(a.step) * a.vocab + v]) + bias;
    id = v;
  }
  const float own_score = val;
  const int own_id = id;
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_down(val, o);
    const int oi = simd_shuffle_down(id, o);
    if (ov > val || (ov == val && oi < id)) {
      val = ov;
      id = oi;
    }
  }
  if (lane == 0) {
    bv[sg] = val;
    bi[sg] = id;
  }
  const int best_id = simd_broadcast(id, 0);
  float next = own_id == best_id ? -INFINITY : own_score;
  uint next_id = own_id == best_id ? 0x7fffffffu : uint(own_id);
  for (ushort o = 16; o > 0; o >>= 1) {
    const float score = simd_shuffle_down(next, o);
    const uint candidate = simd_shuffle_down(next_id, o);
    if (tf_markov_better(score, candidate, next, next_id)) {
      next = score;
      next_id = candidate;
    }
  }
  if (lane == 0) {
    rv[sg] = next;
    ri[sg] = next_id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float m = bv[0];
    int mi = bi[0];
    for (int k = 1; k < 8; k++)
      if (bv[k] > m || (bv[k] == m && bi[k] < mi)) {
        m = bv[k];
        mi = bi[k];
      }
    PART[tg] = float2(m, as_type<float>(mi));
    float second = -INFINITY;
    uint second_id = 0x7fffffffu;
    for (int k = 0; k < 8; ++k) {
      const bool exclude_best = bi[k] == mi;
      const float score = exclude_best ? rv[k] : bv[k];
      const uint candidate = exclude_best ? ri[k] : uint(bi[k]);
      if (tf_markov_better(score, candidate, second, second_id)) {
        second = score;
        second_id = candidate;
      }
    }
    RUNNER[tg] = float2(second, as_type<float>(second_id));
    if (!hit) tags[tag] = tok; // Each group owns its tag and a disjoint range of bias values.
    if (audit) atomic_fetch_add_explicit(audit_counts + uint(!hit), 1u, memory_order_relaxed);
  }
}

[[kernel]] void ds_markov_without_winner(const device bfloat* LOGITS [[buffer(0)]], const device bfloat* EMB [[buffer(1)]],
                          const device bfloat* HEAD [[buffer(2)]], const device uint* IDS [[buffer(3)]],
                          device float2* PART [[buffer(4)]], constant MarkovArgs& a [[buffer(5)]],
                          uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float e[256];
  threadgroup float bv[8];
  threadgroup int bi[8];
  const uint tok = IDS[a.step];
  for (int j = int(t); j < a.rank; j += 256) e[j] = float(EMB[size_t(tok) * a.rank + j]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int v = int(tg) * 256 + int(t);
  float val = -INFINITY;
  int id = 0x7fffffff;
  if (v < a.vocab && uint(v) != IDS[a.step + 1]) {
    const device bfloat* h = HEAD + size_t(v) * a.rank;
    float bias = 0.0f;
    for (int j = 0; j < a.rank; j++) bias += e[j] * float(h[j]);
    val = float(LOGITS[size_t(a.step) * a.vocab + v]) + bias;
    id = v;
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_down(val, o);
    const int oi = simd_shuffle_down(id, o);
    if (ov > val || (ov == val && oi < id)) {
      val = ov;
      id = oi;
    }
  }
  if (lane == 0) {
    bv[sg] = val;
    bi[sg] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float m = bv[0];
    int mi = bi[0];
    for (int k = 1; k < 8; k++)
      if (bv[k] > m || (bv[k] == m && bi[k] < mi)) {
        m = bv[k];
        mi = bi[k];
      }
    PART[tg] = float2(m, as_type<float>(mi));
  }
}


// Compare exact FP32 scores and keep each winning ID attached to its score bits.
static uint2 tf_rank_pair_max(uint2 a, uint2 b) {
  return tf_markov_better(as_type<float>(b.x), b.y, as_type<float>(a.x), a.y) ? b : a;
}

static uint2 tf_rank_block_max(uint2 best, threadgroup uint2* scratch,
                               uint t, uint lane, uint sg) {
  for (ushort offset = 16; offset > 0; offset >>= 1) {
    const uint2 other = uint2(simd_shuffle_down(best.x, offset), simd_shuffle_down(best.y, offset));
    if (lane + offset < 32) best = tf_rank_pair_max(best, other);
  }
  if (lane == 0) scratch[sg] = best;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    for (uint j = 1; j < 8; ++j) best = tf_rank_pair_max(best, scratch[j]);
    scratch[8] = best;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint2 winner = scratch[8];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return winner;
}

[[kernel]] void tf_markov_pick_parallel(const device uint2* leaders [[buffer(0)]],
                                       const device uint2* runners [[buffer(1)]],
                                       device uint* ids [[buffer(2)]],
                                       device MarkovRank* ranks [[buffer(3)]],
                                       device RankState* state [[buffer(4)]],
                                       constant RankArgs& a [[buffer(5)]],
                                       uint t [[thread_position_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]]) {
  if (a.guard && !state->score_step) return;
  const uint2 empty = uint2(as_type<uint>(-INFINITY), 0xffffffffu);
  threadgroup uint2 winners[9];
  uint2 best = empty;
  for (uint j = t; j < a.groups; j += 256) best = tf_rank_pair_max(best, leaders[j]);
  const uint2 first = tf_rank_block_max(best, winners, t, lane, sg);
  best = empty;
  for (uint j = t; j < a.groups; j += 256) {
    const uint2 leader = leaders[j];
    if (leader.y == first.y)
      best = tf_rank_pair_max(best, runners[j]);
    else
      best = tf_rank_pair_max(best, leader);
  }
  const uint2 second = tf_rank_block_max(best, winners, t, lane, sg);
  if (t == 0) {
    const MarkovRank rank = {first.y, second.y, as_type<float>(first.x), as_type<float>(second.x)};
    ranks[a.row] = rank;
    ids[a.row + 1] = rank.first;
    if (a.margin) *state = estimate_rank_acceptance(*state, a, rank);
  }
}

[[kernel]] void tf_markov_pick_leader_policy(const device uint2* leaders [[buffer(0)]],
                                       const device uint2* runners [[buffer(1)]],
                                       device uint* ids [[buffer(2)]],
                                       device MarkovRank* ranks [[buffer(3)]],
                                       device RankState* state [[buffer(4)]],
                                       constant RankArgs& a [[buffer(5)]],
                                       uint t [[thread_position_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]]) {
  if (a.guard && !state->score_step) return;
  const uint2 empty = uint2(as_type<uint>(-INFINITY), 0xffffffffu);
  threadgroup uint2 winners[9];
  uint2 best = empty;
  for (uint j = t; j < a.groups; j += 256) best = tf_rank_pair_max(best, leaders[j]);
  const uint2 first = tf_rank_block_max(best, winners, t, lane, sg);
  threadgroup float masses[8];
  const float peak = as_type<float>(first.x);
  float total = 0.0f;
  best = empty;
  for (uint j = t; j < a.groups; j += 256) {
    const uint2 leader = leaders[j];
    const float score = as_type<float>(leader.x);
    if (isfinite(score) && isfinite(peak)) total += precise::exp(score - peak);
    if (leader.y == first.y)
      best = tf_rank_pair_max(best, runners[j]);
    else
      best = tf_rank_pair_max(best, leader);
  }
  const float wave_mass = simd_sum(total);
  if (lane == 0) masses[sg] = wave_mass;
  const uint2 second = tf_rank_block_max(best, winners, t, lane, sg);
  if (t == 0) {
    const MarkovRank rank = {first.y, second.y, as_type<float>(first.x), as_type<float>(second.x)};
    ranks[a.row] = rank;
    ids[a.row + 1] = rank.first;
    if (a.margin) {
      float mass = masses[0];
      for (uint k = 1; k < 8; ++k) mass += masses[k];
      *state = estimate_rank_acceptance(*state, a, rank, mass > 0.0f ? 1.0f / mass : -1.0f);
    }
  }
}
