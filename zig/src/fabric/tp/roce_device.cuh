// The RoCE kernel's device side (mailbox.cu includes it): region [ctrl, flags, send, recv] per roce_guard.zig's Layout.
namespace tftp_multi {
using namespace tftp;

constexpr int kRoceHcas = 2;
constexpr int kRoceMaxPeers = 8;  // roce_verbs.zig max_peers: (world - 1) x kRoceHcas pollers fit one warp
constexpr int kRoceSlots = 2;
// ctrl words shared with roce.zig's proxy: [0] doorbell, [4 + slot] bytes, [8] completed, [12] abort, [16..22] failure
constexpr int kRoceSeq = 0;
constexpr int kRoceSlotBytes = 4;
constexpr int kRoceCompleted = 8;
constexpr int kRoceFailed = 16;
constexpr int kRoceAbort = 12;  // wait_flag reads ctrl[kCtrlAbort]: the RoCE abort word is passed as that ctrl

struct RoceArgs {
    const uint8_t *in;
    uint8_t *out;
    uint8_t *send_slots;        // this rank's send slots (pinned host, read by the NIC)
    const uint8_t *recv_slots;  // this rank's receive slots [src][2] (written by the peers' NICs)
    const uint32_t *flags;      // this rank's flags [src][2][kRoceHcas], written by the peers' NICs
    uint32_t *ctrl;
    uint32_t *state;            // device: [0] epoch, [1] tail counter, [2] poison, [3] stage counter
    uint64_t nbytes;
    uint64_t slot_bytes;
    uint64_t timeout_ns;
    uint32_t rank;
    uint32_t mode;
    uint32_t n_hca;
    // TF_TP_ROCE_FAST (local): bit 0 fast, bits 8-15 pollers a flag, bits 16-31 their start stagger in ns; 0: classic
    uint32_t opts;
    // Variable lengths (gather only; nullptr: every shard is nbytes): lens[r] = rank r's bytes, the same on every rank
    const int32_t *lens;
    uint64_t chunk;  // chunked gathers: this op's offset into each shard (at most a slot of each from there; else 0)
    uint32_t world;  // 2..kRoceMaxPeers
    uint32_t pad_;
};

// Rank `src`'s flag for `slot` on HCA `h` here, and its receive slot (roce_verbs.zig Layout.flagAt / recvAt).
__device__ __forceinline__ const uint32_t *roce_flag(const RoceArgs &a, uint32_t src, uint64_t slot, uint32_t h) {
    return a.flags + ((static_cast<uint64_t>(src) * kRoceSlots + slot) * kRoceHcas + h) * kFlagStride;
}

__device__ __forceinline__ const uint8_t *roce_inbox(const RoceArgs &a, uint32_t src, uint64_t slot) {
    return a.recv_slots + (static_cast<uint64_t>(src) * kRoceSlots + slot) * a.slot_bytes;
}

// The k-th peer to wait for (k < world - 1), the previous rank first: every proxy posts to the next rank first.
__device__ __forceinline__ uint32_t roce_src(const RoceArgs &a, uint32_t k) {
    return (a.rank + a.world - 1u - k) % a.world;
}

// A failed wait's record (one a rank: the first failure poisons the transport; later launches are no-ops).
__device__ __noinline__ void roce_fail(const RoceArgs &a, uint32_t seq, uint32_t seen, uint32_t src, uint32_t h,
                                       uint32_t why, uint32_t waited) {
    if (atom_cas_gpu(a.state + 2, 0u, 1u) == 0u) {
        st_relaxed_sys(a.ctrl + kRoceFailed + 1, seq);
        st_relaxed_sys(a.ctrl + kRoceFailed + 2, seen);
        st_relaxed_sys(a.ctrl + kRoceFailed + 3, h);
        st_relaxed_sys(a.ctrl + kRoceFailed + 4, why);
        st_relaxed_sys(a.ctrl + kRoceFailed + 5, waited);
        st_relaxed_sys(a.ctrl + kRoceFailed + 6, src);
        fence_sc_sys();
        st_relaxed_sys(a.ctrl + kRoceFailed, 1u);
        fence_sc_sys();
    }
}

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

// A shard's bytes past `off`, at most a slot (chunked gathers; off 0 and a shard within the slot: the shard).
__device__ __forceinline__ uint64_t roce_part(const RoceArgs &a, uint64_t shard, uint64_t off) {
    const uint64_t rest = shard > off ? shard - off : 0u;
    return rest < a.slot_bytes ? rest : a.slot_bytes;
}

extern "C" __global__ void tp_roce_multi_abi_v1() {}

// Present in images whose RoceArgs carry ``chunk`` (roce.zig refuses TF_TP_ROCE_CHUNKED_KB without it); never launched.
extern "C" __global__ void tp_roce_multi_chunk_abi() {}

extern "C" __global__ void __launch_bounds__(512) tp_roce_multi(RoceArgs a) {
    if (a.world < 3u || a.world > 8u || a.rank >= a.world || a.mode != kGather ||
        a.n_hca == 0u || a.n_hca > 2u || a.slot_bytes == 0u || a.slot_bytes > (1u << 20)) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            fail_record(a.state + 2, a.ctrl + kRoceFailed, 0u, 0u, 0u, 3u, 0u);
        }
        return;
    }
    __shared__ uint32_t s_head[2];  // poison, seq
    __shared__ uint32_t s_bad;
    __shared__ uint32_t s_done[kRoceMaxPeers * kRoceHcas];  // fast: some poller of (peer q, HCA h) saw its flag
    const bool fast = (a.opts & 1u) != 0u;
    const uint32_t tid = threadIdx.x;
    const uint32_t gdim = gridDim.x;
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + tid;
    const uint64_t stride = static_cast<uint64_t>(gdim) * blockDim.x;
    const uint32_t peers = a.world - 1u;

    if (tid == 0) {
        s_head[0] = ld_relaxed_gpu(a.state + 2);
        s_head[1] = ld_relaxed_gpu(a.state) + 1u;
        s_bad = 0u;
    }
    if (tid < static_cast<uint32_t>(kRoceMaxPeers * kRoceHcas)) {
        s_done[tid] = 0u;
    }
    __syncthreads();
    if (s_head[0] != 0u) {
        return;
    }
    const uint32_t seq = s_head[1];
    const uint64_t slot = seq & 1u;
    const bool var = a.mode == kGather && a.lens != nullptr;
    const uint64_t off = var ? a.chunk : 0u;
    const uint64_t mine = var ? roce_part(a, roce_shard(a, a.rank), off) : a.nbytes;
    const uint64_t padded = var ? (mine < 16u ? 16u : (mine + 15u) & ~uint64_t(15u)) : (a.nbytes + 15u) & ~uint64_t(15u);
    if (off != 0u && mine == 0u) {
        bool rest = false;
        for (uint32_t q = 0; q < peers; q++) {
            rest = rest || roce_part(a, roce_shard(a, roce_src(a, q)), off) != 0u;
        }
        if (!rest) {
            return;  // nothing past this offset on any rank: no op, the epoch stays (every rank skips alike)
        }
    }
    if (padded > a.slot_bytes) {  // never staged past a slot: a refused wait (why 3), the transport poisoned
        if (tid == 0 && blockIdx.x == 0) {
            fail_record(a.state + 2, a.ctrl + kRoceFailed, seq, 0u, 0u, 3u, 0u);
        }
        return;
    }

    // 1. stage this block's share into the send slot (once for every peer); the last block to finish rings the doorbell
    copy_any<false>(fast, a.send_slots + slot * a.slot_bytes, a.in + off, var ? roce_span(a, mine) : a.nbytes, index,
                    stride);
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
        copy_any<false>(fast, a.out + static_cast<uint64_t>(a.rank) * a.nbytes + off, a.in + off,
                        var ? roce_span(a, mine) : a.nbytes, index, stride);
    }

    // 2. every peer's flag on every HCA for this slot; fast: one poller a (peer, HCA), thread 32 k + q x n_hca + h.
    const uint32_t pollers = (a.opts >> 8) & 0xffu;
    const uint32_t stagger_ns = a.opts >> 16;
    if (fast) {
        const uint32_t lane = tid & 31u;
        const uint32_t k = tid >> 5;
        if (lane < peers * a.n_hca && k < (pollers == 0u ? 1u : pollers)) {
            if (k > 0u && stagger_ns != 0u) {
                __nanosleep(k * stagger_ns);
            }
            const uint32_t src = roce_src(a, lane / a.n_hca);
            const uint32_t h = lane % a.n_hca;
            uint32_t seen = 0u, why = 0u, waited = 0u;
            if (!wait_flag_shared(roce_flag(a, src, slot, h), seq, a.ctrl + kRoceAbort, a.timeout_ns, &seen, &why,
                                  &waited, &s_done[lane])) {
                s_bad = 1u;
                roce_fail(a, seq, seen, src, h, why, waited);
            }
        }
    } else if (tid == 0) {
        bool failed = false;
        for (uint32_t q = 0; q < peers && !failed; q++) {
            const uint32_t src = roce_src(a, q);
            for (uint32_t h = 0; h < a.n_hca; h++) {
                uint32_t seen = 0u, why = 0u, waited = 0u;
                if (!wait_flag(roce_flag(a, src, slot, h), seq, a.ctrl + kRoceAbort, a.timeout_ns, &seen, &why,
                               &waited)) {
                    s_bad = 1u;
                    roce_fail(a, seq, seen, src, h, why, waited);
                    failed = true;
                    break;
                }
            }
        }
    }
    __syncthreads();

    // 3. copy out of the receive slots the peers' NICs wrote, each sender's shard at its rank
    if (s_bad == 0u) {
        switch (a.mode) {
            case kGather:
                for (uint32_t q = 0; q < peers; q++) {
                    const uint32_t src = roce_src(a, q);
                    const uint64_t theirs = var ? roce_part(a, roce_shard(a, src), off) : a.nbytes;
                    copy_any<true>(fast, a.out + static_cast<uint64_t>(src) * a.nbytes + off,
                                   roce_inbox(a, src, slot), var ? roce_span(a, theirs) : a.nbytes, index, stride);
                }
                break;
            default:
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

}  // namespace tftp_multi
