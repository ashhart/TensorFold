// The RoCE kernel's device side (mailbox.cu includes it): region [ctrl, flags, send, recv] per roce_guard.zig's Layout.
namespace tftp {

constexpr int kRoceHcas = 2;
constexpr int kRoceMaxPeers = 16;  // roce_verbs.zig max_peers: (world - 1) x kRoceHcas pollers fit one warp
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

// Rank r's operand of a sum: this rank's input, else rank r's receive slot (system-scope loads: the NIC wrote it).
__device__ __forceinline__ uint4 roce_v4(const RoceArgs &a, uint32_t r, uint64_t slot, uint64_t i) {
    return r == a.rank ? reinterpret_cast<const uint4 *>(a.in)[i] : ld_relaxed_sys_v4(roce_inbox(a, r, slot) + (i << 4));
}

__device__ __forceinline__ uint16_t roce_u16(const RoceArgs &a, uint32_t r, uint64_t slot, uint64_t i) {
    return r == a.rank ? reinterpret_cast<const uint16_t *>(a.in)[i]
                       : static_cast<uint16_t>(ld_relaxed_sys_u16(roce_inbox(a, r, slot) + (i << 1)));
}

__device__ __forceinline__ uint32_t roce_u32(const RoceArgs &a, uint32_t r, uint64_t slot, uint64_t i) {
    return r == a.rank ? reinterpret_cast<const uint32_t *>(a.in)[i]
                       : ld_relaxed_sys(reinterpret_cast<const uint32_t *>(roce_inbox(a, r, slot)) + i);
}

// out <- every rank's operand added in rank order (bf16 through fp32), the same bits everywhere.
template <bool Bf16>
__device__ __forceinline__ void roce_sum(const RoceArgs &a, uint64_t slot, uint64_t index, uint64_t stride) {
    const uint64_t n = a.nbytes;
    const uint64_t al = reinterpret_cast<uint64_t>(a.out) | reinterpret_cast<uint64_t>(a.in) |
                        reinterpret_cast<uint64_t>(roce_inbox(a, 0u, slot)) | a.slot_bytes | n;
    if ((al & 15u) == 0u) {
        const uint64_t packs = n >> 4;
        for (uint64_t i = index; i < packs; i += stride) {
            uint4 acc = roce_v4(a, 0u, slot, i);
            for (uint32_t r = 1u; r < a.world; r++) {
                const uint4 b = roce_v4(a, r, slot, i);
                acc.x = Bf16 ? add_bf16x2(acc.x, b.x) : add_f32(acc.x, b.x);
                acc.y = Bf16 ? add_bf16x2(acc.y, b.y) : add_f32(acc.y, b.y);
                acc.z = Bf16 ? add_bf16x2(acc.z, b.z) : add_f32(acc.z, b.z);
                acc.w = Bf16 ? add_bf16x2(acc.w, b.w) : add_f32(acc.w, b.w);
            }
            reinterpret_cast<uint4 *>(a.out)[i] = acc;
        }
    } else if (Bf16) {
        for (uint64_t i = index; i < (n >> 1); i += stride) {
            uint16_t acc = roce_u16(a, 0u, slot, i);
            for (uint32_t r = 1u; r < a.world; r++) {
                acc = add_bf16(acc, roce_u16(a, r, slot, i));
            }
            reinterpret_cast<uint16_t *>(a.out)[i] = acc;
        }
    } else {
        for (uint64_t i = index; i < (n >> 2); i += stride) {
            uint32_t acc = roce_u32(a, 0u, slot, i);
            for (uint32_t r = 1u; r < a.world; r++) {
                acc = add_f32(acc, roce_u32(a, r, slot, i));
            }
            reinterpret_cast<uint32_t *>(a.out)[i] = acc;
        }
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

}  // namespace tftp
