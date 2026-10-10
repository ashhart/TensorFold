// One-shot two-rank exchange through pinned host memory both GPUs map (the TP mailbox; mailbox.zig is its host side).

#include <cuda_bf16.h>
#include <stdint.h>

// Protocol adapted from b12x's RoCEnante (Apache-2.0): a device epoch, slots by seq parity, a seq flag after the data.
namespace tftp {

constexpr int kMaxBlocks = 16;
constexpr int kFlagStride = 32;  // words: one 128-byte line per flag
// ctrl words: [0] abort (any rank, any host thread); per rank r at 16 + 8 r: failed, seq, seen, block, why, waited us
constexpr int kCtrlAbort = 0;
constexpr int kCtrlRank = 16;

enum Mode : uint32_t { kGather = 0, kSumF32 = 1, kSumBf16 = 2, kExchange = 3 };

struct MailArgs {
    const uint8_t *in;          // this rank's shard (device)
    uint8_t *out;               // gather [2 n], exchange / sums [n]
    uint8_t *peer_slots;        // the peer's inbox data, as this GPU maps it
    uint32_t *peer_flags;       // the peer's inbox flags
    const uint8_t *my_slots;    // this rank's inbox data
    const uint32_t *my_flags;   // this rank's inbox flags
    uint32_t *state;            // device: [0] epoch, [1] tail counter, [2] poison
    uint32_t *ctrl;             // shared control words
    uint64_t nbytes;            // bytes a rank
    uint64_t slot_bytes;
    uint64_t timeout_ns;
    uint32_t rank;
    uint32_t mode;
};

__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint4 ld_relaxed_sys_v4(const void *p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t ld_relaxed_sys_u8(const uint8_t *p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u8 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_relaxed_sys(uint32_t *p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_release_sys(uint32_t *p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_release_gpu(uint32_t *p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t atom_add_acq_rel_gpu(uint32_t *p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ uint32_t atom_add_acq_rel_sys(uint32_t *p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.acq_rel.sys.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ uint32_t atom_cas_gpu(uint32_t *p, uint32_t cmp, uint32_t v) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.cas.b32 %0, [%1], %2, %3;" : "=r"(old) : "l"(p), "r"(cmp), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// dst[i] <- src[i] for i in [0, n), grid-stride, 16-byte vectors when aligned; SysSrc: system-scope loads.
template <bool SysSrc>
__device__ __forceinline__ void copy_bytes(uint8_t *dst, const uint8_t *src, uint64_t n, uint64_t index,
                                           uint64_t stride) {
    const uint64_t a = reinterpret_cast<uint64_t>(dst) | reinterpret_cast<uint64_t>(src) | n;
    if ((a & 15u) == 0u) {
        const uint64_t packs = n >> 4;
        for (uint64_t i = index; i < packs; i += stride) {
            const uint4 v = SysSrc ? ld_relaxed_sys_v4(src + (i << 4)) : reinterpret_cast<const uint4 *>(src)[i];
            reinterpret_cast<uint4 *>(dst)[i] = v;
        }
    } else {
        for (uint64_t i = index; i < n; i += stride) {
            dst[i] = static_cast<uint8_t>(SysSrc ? ld_relaxed_sys_u8(src + i) : src[i]);
        }
    }
}

// TF_TP_ROCE_FAST: copy_bytes with two grid strides' loads issued before their stores (same bytes, one round trip).
template <bool SysSrc>
__device__ __forceinline__ void copy_bytes2(uint8_t *dst, const uint8_t *src, uint64_t n, uint64_t index,
                                            uint64_t stride) {
    const uint64_t a = reinterpret_cast<uint64_t>(dst) | reinterpret_cast<uint64_t>(src) | n;
    if ((a & 15u) != 0u) {
        copy_bytes<SysSrc>(dst, src, n, index, stride);
        return;
    }
    const uint64_t packs = n >> 4;
    for (uint64_t i = index; i < packs; i += 2u * stride) {
        const uint64_t j = i + stride;
        uint4 v0, v1 = make_uint4(0u, 0u, 0u, 0u);
        if (SysSrc) {
            v0 = ld_relaxed_sys_v4(src + (i << 4));
            if (j < packs) v1 = ld_relaxed_sys_v4(src + (j << 4));
        } else {
            v0 = reinterpret_cast<const uint4 *>(src)[i];
            if (j < packs) v1 = reinterpret_cast<const uint4 *>(src)[j];
        }
        reinterpret_cast<uint4 *>(dst)[i] = v0;
        if (j < packs) reinterpret_cast<uint4 *>(dst)[j] = v1;
    }
}

template <bool SysSrc>
__device__ __forceinline__ void copy_any(bool fast, uint8_t *dst, const uint8_t *src, uint64_t n, uint64_t index,
                                         uint64_t stride) {
    if (fast) {
        copy_bytes2<SysSrc>(dst, src, n, index, stride);
    } else {
        copy_bytes<SysSrc>(dst, src, n, index, stride);
    }
}

__device__ __forceinline__ uint32_t add_bf16x2(uint32_t x, uint32_t y) {
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&y));
    const __nv_bfloat162 r = __floats2bfloat162_rn(a.x + b.x, a.y + b.y);
    return *reinterpret_cast<const uint32_t *>(&r);
}

__device__ __forceinline__ uint32_t add_f32(uint32_t x, uint32_t y) {
    return __float_as_uint(__uint_as_float(x) + __uint_as_float(y));
}

__device__ __forceinline__ uint32_t ld_relaxed_sys_u16(const uint8_t *p) {
    uint16_t v;
    asm volatile("ld.relaxed.sys.global.u16 %0, [%1];" : "=h"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint16_t add_bf16(uint16_t x, uint16_t y) {
    const __nv_bfloat16 r = __float2bfloat16_rn(__bfloat162float(__ushort_as_bfloat16(x)) +
                                                __bfloat162float(__ushort_as_bfloat16(y)));
    return __bfloat16_as_ushort(r);
}

// out <- mine + peer's, rank 0's first; IEEE addition commutes, so both ranks get the same bits.
template <bool Bf16>
__device__ __forceinline__ void sum_bytes(uint8_t *out, const uint8_t *mine, const uint8_t *peer, uint64_t n,
                                          uint32_t rank, uint64_t index, uint64_t stride) {
    const uint64_t al = reinterpret_cast<uint64_t>(out) | reinterpret_cast<uint64_t>(mine) |
                        reinterpret_cast<uint64_t>(peer) | n;
    if ((al & 15u) == 0u) {
        const uint64_t packs = n >> 4;
        for (uint64_t i = index; i < packs; i += stride) {
            const uint4 m = reinterpret_cast<const uint4 *>(mine)[i];
            const uint4 p = ld_relaxed_sys_v4(peer + (i << 4));
            const uint4 a = rank == 0 ? m : p;
            const uint4 b = rank == 0 ? p : m;
            uint4 r;
            r.x = Bf16 ? add_bf16x2(a.x, b.x) : add_f32(a.x, b.x);
            r.y = Bf16 ? add_bf16x2(a.y, b.y) : add_f32(a.y, b.y);
            r.z = Bf16 ? add_bf16x2(a.z, b.z) : add_f32(a.z, b.z);
            r.w = Bf16 ? add_bf16x2(a.w, b.w) : add_f32(a.w, b.w);
            reinterpret_cast<uint4 *>(out)[i] = r;
        }
    } else if (Bf16) {
        for (uint64_t i = index; i < (n >> 1); i += stride) {
            const uint16_t m = reinterpret_cast<const uint16_t *>(mine)[i];
            const uint16_t p = static_cast<uint16_t>(ld_relaxed_sys_u16(peer + (i << 1)));
            reinterpret_cast<uint16_t *>(out)[i] = rank == 0 ? add_bf16(m, p) : add_bf16(p, m);
        }
    } else {
        for (uint64_t i = index; i < (n >> 2); i += stride) {
            const uint32_t m = reinterpret_cast<const uint32_t *>(mine)[i];
            const uint32_t p = ld_relaxed_sys(reinterpret_cast<const uint32_t *>(peer) + i);
            reinterpret_cast<uint32_t *>(out)[i] = rank == 0 ? add_f32(m, p) : add_f32(p, m);
        }
    }
}

// Thread 0 of a block: spin until the peer block's flag holds seq; false on abort (why 1) or timeout (why 2).
__device__ __noinline__ bool wait_flag(const uint32_t *flag, uint32_t seq, const uint32_t *ctrl, uint64_t timeout_ns,
                                       uint32_t *seen, uint32_t *why, uint32_t *waited_us) {
    const uint64_t t0 = globaltimer();
    uint32_t polls = 0;
    while (true) {
        const uint32_t v = ld_acquire_sys(flag);
        if (v == seq) {
            *seen = v;
            return true;
        }
        if ((++polls & 63u) == 0u) {
            const uint64_t now = globaltimer();
            const bool aborted = ld_relaxed_sys(ctrl + kCtrlAbort) != 0u;
            const bool late = now > t0 && now - t0 > timeout_ns;
            if (aborted || late) {
                *seen = ld_acquire_sys(flag);
                *why = aborted ? 1u : 2u;
                *waited_us = now > t0 ? static_cast<uint32_t>((now - t0) / 1000u) : 0u;
                return *seen == seq;
            }
        }
    }
}

// TF_TP_ROCE_FAST: wait_flag for one of several pollers of a flag; `done` (shared) ends the others once one saw it.
__device__ __noinline__ bool wait_flag_shared(const uint32_t *flag, uint32_t seq, const uint32_t *ctrl,
                                              uint64_t timeout_ns, uint32_t *seen, uint32_t *why, uint32_t *waited_us,
                                              volatile uint32_t *done) {
    const uint64_t t0 = globaltimer();
    uint32_t polls = 0;
    while (true) {
        const uint32_t v = ld_acquire_sys(flag);
        if (v == seq) {
            *done = 1u;
            *seen = v;
            return true;
        }
        if (*done != 0u) {
            *seen = seq;
            return true;
        }
        if ((++polls & 63u) == 0u) {
            const uint64_t now = globaltimer();
            const bool aborted = ld_relaxed_sys(ctrl + kCtrlAbort) != 0u;
            const bool late = now > t0 && now - t0 > timeout_ns;
            if (aborted || late) {
                *seen = ld_acquire_sys(flag);
                *why = aborted ? 1u : 2u;
                *waited_us = now > t0 ? static_cast<uint32_t>((now - t0) / 1000u) : 0u;
                if (*seen == seq) {
                    *done = 1u;
                    return true;
                }
                return *done != 0u;
            }
        }
    }
}

}  // namespace tftp

using namespace tftp;

extern "C" __global__ void __launch_bounds__(512) tp_mailbox(MailArgs a) {
    __shared__ uint32_t s_head[2];  // poison, seq
    __shared__ uint32_t s_bad;
    const uint32_t tid = threadIdx.x;
    const uint32_t gdim = gridDim.x;
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + tid;
    const uint64_t stride = static_cast<uint64_t>(gdim) * blockDim.x;

    // a poisoned transport does nothing (the host raises on its next check); poison and epoch in one round trip
    if (tid == 0) {
        s_head[0] = ld_relaxed_gpu(a.state + 2);
        s_head[1] = ld_relaxed_gpu(a.state) + 1u;
        s_bad = 0u;
    }
    __syncthreads();
    if (s_head[0] != 0u) {
        return;
    }
    const uint32_t seq = s_head[1];
    const uint64_t slot = seq & 1u;
    const uint64_t flag_at = (slot * kMaxBlocks + blockIdx.x) * kFlagStride;

    // 1. this block's share of the shard into the peer's inbox, then its flag after it (system-scope release)
    copy_bytes<false>(a.peer_slots + slot * a.slot_bytes, a.in, a.nbytes, index, stride);
    __syncthreads();
    if (tid == 0) {
        fence_sc_sys();
        st_release_sys(a.peer_flags + flag_at, seq);
    }
    // the local shard while the peer's writes travel
    if (a.mode == kGather) {
        copy_bytes<false>(a.out + static_cast<uint64_t>(a.rank) * a.nbytes, a.in, a.nbytes, index, stride);
    }

    // 2. the peer's block of the same index (it wrote exactly the bytes this block reads)
    if (tid == 0) {
        uint32_t seen = 0u, why = 0u, waited = 0u;
        if (!wait_flag(a.my_flags + flag_at, seq, a.ctrl, a.timeout_ns, &seen, &why, &waited)) {
            s_bad = 1u;
            if (atom_cas_gpu(a.state + 2, 0u, 1u) == 0u) {  // one failure record a rank
                uint32_t *rec = a.ctrl + kCtrlRank + 8u * a.rank;
                st_relaxed_sys(rec + 1, seq);
                st_relaxed_sys(rec + 2, seen);
                st_relaxed_sys(rec + 3, blockIdx.x);
                st_relaxed_sys(rec + 4, why);
                st_relaxed_sys(rec + 5, waited);
                fence_sc_sys();
                st_relaxed_sys(rec, 1u);
                fence_sc_sys();
            }
        }
    }
    __syncthreads();

    // 3. copy out, gated by this block's own wait
    if (s_bad == 0u) {
        const uint8_t *inbox = a.my_slots + slot * a.slot_bytes;
        switch (a.mode) {
            case kGather:
                copy_bytes<true>(a.out + static_cast<uint64_t>(1u - a.rank) * a.nbytes, inbox, a.nbytes, index,
                                 stride);
                break;
            case kExchange:
                copy_bytes<true>(a.out, inbox, a.nbytes, index, stride);
                break;
            case kSumF32:
                sum_bytes<false>(a.out, a.in, inbox, a.nbytes, a.rank, index, stride);
                break;
            default:
                sum_bytes<true>(a.out, a.in, inbox, a.nbytes, a.rank, index, stride);
                break;
        }
    }

    // 4. the last block publishes the epoch, unless a wait of this launch failed (later launches stay no-ops)
    __syncthreads();
    if (tid == 0) {
        // the counter is back at 0 for the next launch (stream order), whatever its grid
        const bool last = gdim == 1u || atom_add_acq_rel_gpu(a.state + 1, 1u) == gdim - 1u;
        const bool ok = gdim == 1u ? s_bad == 0u : ld_relaxed_gpu(a.state + 2) == 0u;
        if (last && gdim > 1u) {
            st_release_gpu(a.state + 1, 0u);
        }
        if (last && ok) {
            st_release_gpu(a.state, seq);
        }
    }
}

// The RoCE variant (roce.zig): blocks stage into the send slot, the last rings the doorbell, a host proxy RDMA-writes.
namespace tftp {

constexpr int kRoceHcas = 2;
// ctrl words shared with roce.zig's proxy: [0] doorbell seq, [4 + slot] padded bytes, [8] completed, [16..21] failure
constexpr int kRoceSeq = 0;
constexpr int kRoceSlotBytes = 4;
constexpr int kRoceCompleted = 8;
constexpr int kRoceFailed = 16;
constexpr int kRoceAbort = 12;  // wait_flag reads ctrl[kCtrlAbort]: the RoCE abort word is passed as that ctrl

struct RoceArgs {
    const uint8_t *in;
    uint8_t *out;
    uint8_t *send_slots;        // this rank's send slots (pinned host, read by the NIC)
    const uint8_t *recv_slots;  // this rank's receive slots (written by the peer's NIC)
    const uint32_t *flags;      // this rank's flags [2][kRoceHcas], written by the peer's NIC
    uint32_t *ctrl;
    uint32_t *state;            // device: [0] epoch, [1] tail counter, [2] poison, [3] stage counter
    uint64_t nbytes;
    uint64_t slot_bytes;
    uint64_t timeout_ns;
    uint32_t rank;
    uint32_t mode;
    uint32_t n_hca;
    uint32_t opts;  // TF_TP_ROCE_FAST: bit 0 fast, bits 8-15 pollers a flag, bits 16-31 start stagger ns; 0: classic
    // gather only (nullptr: every shard is nbytes): lens[r] = rank r's bytes, padded to 16 on the wire
    const int32_t *lens;
};

// Rank r's bytes this op: ``nbytes``, or lens[r] capped at nbytes.
__device__ __forceinline__ uint64_t roce_shard(const RoceArgs &a, uint32_t r) {
    if (a.lens == nullptr) {
        return a.nbytes;
    }
    const int32_t v = __ldg(a.lens + r);
    const uint64_t n = v > 0 ? static_cast<uint64_t>(v) : 0u;
    return n < a.nbytes ? n : a.nbytes;
}

// The bytes a copy of an n-byte shard moves: n rounded up to 16 when that stays inside a 16-aligned stride, else n.
__device__ __forceinline__ uint64_t roce_span(const RoceArgs &a, uint64_t n) {
    if (a.lens == nullptr || (a.nbytes & 15u) != 0u) {
        return n;
    }
    return (n + 15u) & ~uint64_t(15u);
}

}  // namespace tftp

extern "C" __global__ void __launch_bounds__(512) tp_roce(RoceArgs a) {
    __shared__ uint32_t s_head[2];  // poison, seq
    __shared__ uint32_t s_bad;
    __shared__ uint32_t s_done[kRoceHcas];  // fast: some poller of HCA h saw its flag
    const bool fast = (a.opts & 1u) != 0u;
    const uint32_t tid = threadIdx.x;
    const uint32_t gdim = gridDim.x;
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + tid;
    const uint64_t stride = static_cast<uint64_t>(gdim) * blockDim.x;

    if (tid == 0) {
        s_head[0] = ld_relaxed_gpu(a.state + 2);
        s_head[1] = ld_relaxed_gpu(a.state) + 1u;
        s_bad = 0u;
    }
    if (tid < static_cast<uint32_t>(kRoceHcas)) {
        s_done[tid] = 0u;
    }
    __syncthreads();
    if (s_head[0] != 0u) {
        return;
    }
    const uint32_t seq = s_head[1];
    const uint64_t slot = seq & 1u;
    const bool var = a.mode == kGather && a.lens != nullptr;
    const uint64_t mine = var ? roce_shard(a, a.rank) : a.nbytes;
    const uint64_t theirs = var ? roce_shard(a, 1u - a.rank) : a.nbytes;
    const uint64_t padded = var ? (mine < 16u ? 16u : (mine + 15u) & ~uint64_t(15u)) : (a.nbytes + 15u) & ~uint64_t(15u);

    // 1. stage this block's share into the send slot; the last block to finish rings the doorbell
    copy_any<false>(fast, a.send_slots + slot * a.slot_bytes, a.in, var ? roce_span(a, mine) : a.nbytes, index, stride);
    __syncthreads();
    if (tid == 0 && fast) {
        // lean doorbell: release RMW per block arrival, the last acquires, then a release store (no SC system fences)
        const bool last = gdim == 1u || atom_add_acq_rel_sys(a.state + 3, 1u) == gdim - 1u;
        if (last) {
            if (gdim > 1u) {
                st_release_gpu(a.state + 3, 0u);
            }
            st_relaxed_sys(a.ctrl + kRoceSlotBytes + slot, static_cast<uint32_t>(padded));
            st_release_sys(a.ctrl + kRoceSeq, seq);
        }
    } else if (tid == 0) {
        fence_sc_sys();
        const bool last = gdim == 1u || atom_add_acq_rel_gpu(a.state + 3, 1u) == gdim - 1u;
        if (last) {
            if (gdim > 1u) {
                st_release_gpu(a.state + 3, 0u);
                fence_sc_sys();  // every block's staging (seen through the counter) before the doorbell
            }
            st_relaxed_sys(a.ctrl + kRoceSlotBytes + slot, static_cast<uint32_t>(padded));
            st_release_sys(a.ctrl + kRoceSeq, seq);
        }
    }
    if (a.mode == kGather) {
        copy_any<false>(fast, a.out + static_cast<uint64_t>(a.rank) * a.nbytes, a.in,
                        var ? roce_span(a, mine) : a.nbytes, index, stride);
    }

    // 2. every HCA's flag for this slot; fast: poller k of HCA h is thread 32 k + h, started k x stagger ns late
    const uint32_t pollers = (a.opts >> 8) & 0xffu;
    const uint32_t stagger_ns = a.opts >> 16;
    if (fast) {
        const uint32_t lane = tid & 31u;
        const uint32_t k = tid >> 5;
        if (lane < a.n_hca && k < (pollers == 0u ? 1u : pollers)) {
            if (k > 0u && stagger_ns != 0u) {
                __nanosleep(k * stagger_ns);
            }
            uint32_t seen = 0u, why = 0u, waited = 0u;
            if (!wait_flag_shared(a.flags + (slot * kRoceHcas + lane) * kFlagStride, seq, a.ctrl + kRoceAbort,
                                  a.timeout_ns, &seen, &why, &waited, &s_done[lane])) {
                s_bad = 1u;
                if (atom_cas_gpu(a.state + 2, 0u, 1u) == 0u) {
                    st_relaxed_sys(a.ctrl + kRoceFailed + 1, seq);
                    st_relaxed_sys(a.ctrl + kRoceFailed + 2, seen);
                    st_relaxed_sys(a.ctrl + kRoceFailed + 3, lane);
                    st_relaxed_sys(a.ctrl + kRoceFailed + 4, why);
                    st_relaxed_sys(a.ctrl + kRoceFailed + 5, waited);
                    fence_sc_sys();
                    st_relaxed_sys(a.ctrl + kRoceFailed, 1u);
                    fence_sc_sys();
                }
            }
        }
    } else if (tid == 0) {
        for (uint32_t h = 0; h < a.n_hca; h++) {
            uint32_t seen = 0u, why = 0u, waited = 0u;
            if (!wait_flag(a.flags + (slot * kRoceHcas + h) * kFlagStride, seq, a.ctrl + kRoceAbort, a.timeout_ns, &seen,
                           &why, &waited)) {
                s_bad = 1u;
                if (atom_cas_gpu(a.state + 2, 0u, 1u) == 0u) {
                    st_relaxed_sys(a.ctrl + kRoceFailed + 1, seq);
                    st_relaxed_sys(a.ctrl + kRoceFailed + 2, seen);
                    st_relaxed_sys(a.ctrl + kRoceFailed + 3, h);
                    st_relaxed_sys(a.ctrl + kRoceFailed + 4, why);
                    st_relaxed_sys(a.ctrl + kRoceFailed + 5, waited);
                    fence_sc_sys();
                    st_relaxed_sys(a.ctrl + kRoceFailed, 1u);
                    fence_sc_sys();
                }
                break;
            }
        }
    }
    __syncthreads();

    // 3. copy out of the receive slot the peer's NIC wrote
    if (s_bad == 0u) {
        const uint8_t *inbox = a.recv_slots + slot * a.slot_bytes;
        switch (a.mode) {
            case kGather:
                copy_any<true>(fast, a.out + static_cast<uint64_t>(1u - a.rank) * a.nbytes, inbox,
                               var ? roce_span(a, theirs) : a.nbytes, index, stride);
                break;
            case kExchange:
                copy_any<true>(fast, a.out, inbox, a.nbytes, index, stride);
                break;
            case kSumF32:
                sum_bytes<false>(a.out, a.in, inbox, a.nbytes, a.rank, index, stride);
                break;
            default:
                sum_bytes<true>(a.out, a.in, inbox, a.nbytes, a.rank, index, stride);
                break;
        }
    }

    // 4. the epoch, as tp_mailbox; the host also sees the last completed seq
    __syncthreads();
    if (tid == 0) {
        const bool last = gdim == 1u || atom_add_acq_rel_gpu(a.state + 1, 1u) == gdim - 1u;
        const bool ok = gdim == 1u ? s_bad == 0u : ld_relaxed_gpu(a.state + 2) == 0u;
        if (last && gdim > 1u) {
            st_release_gpu(a.state + 1, 0u);
        }
        if (last && ok) {
            st_release_gpu(a.state, seq);
            st_relaxed_sys(a.ctrl + kRoceCompleted, seq);
        }
    }
}
