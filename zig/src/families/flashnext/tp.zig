//! Speed-up mode for Flash Next: two Macs with the whole model each, half the work each, over MCDMA. Decode: each rank computes every routed and the shared expert for its half of the window's rows (`ownRows`) and those rows' MoE branch with one Mac's sums; after each target layer's experts the GPU posts a sequence; the host sends the branch rows (bf16) to the peer, and the next layer's combine waits for the flag the peer's message sets, then updates every row's streams (one Mac's bits). DeltaNet layers split their heads: each rank's out-projection partial (fp32), exchanged and added in rank order. Prefill: a chunk's rows split across the two Macs (replay.zig `chunkPair`); the host sends each layer's handoff into the peer's slot for that layer and signals the layer's flag, which the peer's GPU waits for.
const std = @import("std");
const mtl = @import("metal");
const fabric = @import("fabric");

const D = 2560;
const WIDE = 4 * D;
const MAXR = 16;
const LAYERS = 48;
const PMAX = 8192;
pub const CS_ROW = 3 * WIDE * 2;
pub const SO_ROW = 48 * 128 * 128 * 4;
const PAGE = 16384;

fn pages(n: usize) usize {
    return (n + PAGE - 1) / PAGE * PAGE;
}

/// The window, the same on both ranks: decode slots (alternating by sequence parity), then a slot a layer for prefill handoffs, the n-gram tail, the last row's streams and first token, the MTP head's keys; flags; the sync words.
const SLOT = MAXR * 11 * D * 2; // one layer's expert outputs at the widest window (bf16): 55 pages
pub const DN_SLOT = pages(CS_ROW + SO_ROW); // a DeltaNet layer: conv state row, then recurrent state row
pub const ATT_SLOT = 4 * PMAX * 512 + PMAX * 256; // an attention layer: keys of both heads, values of both, raw keys
pub const KV_HEAD = PMAX * 512; // one head's keys (or values) at the most rows, inside ATT_SLOT
const DECODE = 0;
const PART = MAXR * D * 4; // one rank's fp32 partial of a split projection, at the widest window
const REDUCE = 2 * SLOT; // the peer's partials, alternating by parity
const PREFILL = REDUCE + 2 * PART;
pub const TAIL = PREFILL + 36 * DN_SLOT + 12 * ATT_SLOT; // the n-gram tail: PLE_TAIL rows of WIDE
pub const LAST = TAIL + pages(9 * WIDE * 2); // the last prompt row's streams, then its first token
pub const MTP = LAST + pages(WIDE * 2 + 16); // the MTP head's keys, an attention slot
const FLAGS = MTP + ATT_SLOT;
const SYNC = FLAGS + PAGE;
const SENDX = SYNC + PAGE; // this rank's packed expert slots, sent from here: two by parity, a page of room before each
const SENDR = SENDX + 2 * (PAGE + SLOT); // this rank's fp32 partials, sent from here, the same way
const SENDA = SENDR + 2 * (PAGE + PART); // this rank's head argmax a row (value, index), sent from here the same way
const RECVA = SENDA + 4 * PAGE; // the peer's, alternating by parity
const REQ_TOKENS = 262144; // a served request's prompt tokens at most (the engine's context)
const REQ = RECVA + 2 * PAGE; // served: rank 0's request for rank 1 (a 64-byte head, then the prompt)
pub const MARK = REQ + pages(64 + 4 * REQ_TOKENS); // a mark a pair chunk passed: its DeltaNet states (a DN_SLOT a layer), then its n-gram tail
const WINDOW = MARK + 36 * DN_SLOT + pages(9 * WIDE * 2);

fn sendX(x: u32) usize {
    return SENDX + (x % 2) * (PAGE + SLOT) + PAGE;
}

fn sendR(x: u32) usize {
    return SENDR + (x % 2) * (PAGE + PART) + PAGE;
}

fn sendA(x: u32) usize {
    return SENDA + (x % 2) * 2 * PAGE + PAGE;
}
const FLAG = FLAGS; // decode: the last sequence whose outputs have landed
const HELLO = FLAGS + 8;
pub const LAYER_FLAG = FLAGS + 64; // prefill: a word a layer, then the n-gram tail, back, and MTP words
pub const TAIL_FLAG = LAYER_FLAG + LAYERS * 8;
pub const BACK_FLAG = TAIL_FLAG + 8;
pub const MTP_FLAG = BACK_FLAG + 8;
const REQ_FLAG = MTP_FLAG + 8; // served: the last request rank 0 has written
const CTRL_FLAG = REQ_FLAG + 8; // rank 0's decision at each step both ranks take: (step << 6) | (next depth << 1) | quit
const ACK_FLAG = CTRL_FLAG + 8; // served: rank 1's answer to a request's resume, (request << 1) | restored
pub const MARK_FLAG = FLAGS + 512; // a passed mark's states have landed in MARK (ACK_FLAG + 8 stays free for the link probe)
pub const MARK_READY = MARK_FLAG + 8; // the receiver's MARK is free for that sequence
const GPU = 1024;
const GAVE_UP = 2048;

/// The GPU posts (sequence << 5) | rows in one 32-bit word, so posted sequences wrap at 2^27.
const SEQ_BITS = 27;

/// Whether post word `w` has reached sequence `seq`, modulo 2^27 (the GPU is never 2^26 sequences ahead).
fn postReached(w: u32, seq: u32) bool {
    return ((w >> 5) -% seq) & ((1 << SEQ_BITS) - 1) < 1 << (SEQ_BITS - 1);
}

/// Whether wrapping counter `a` has reached `b` (host sequences and flag values; never 2^31 apart).
fn reached(a: u32, b: u32) bool {
    return @as(i32, @bitCast(a -% b)) >= 0;
}

/// The receiving window's slot for layer i's prefill handoff (attention layers are i % 4 == 3).
pub fn layerSlot(i: usize) usize {
    const att = i / 4; // attention layers before i (3, 7, ... below it)
    return PREFILL + (i - att) * DN_SLOT + att * ATT_SLOT;
}

test "control words are distinct and inside the flag page" {
    var words: [LAYERS + 12]usize = undefined;
    var n: usize = 0;
    for ([_]usize{ FLAG, HELLO, TAIL_FLAG, BACK_FLAG, MTP_FLAG, REQ_FLAG, CTRL_FLAG, ACK_FLAG, ACK_FLAG + 8, MARK_FLAG, MARK_READY }) |w| {
        words[n] = w;
        n += 1;
    }
    for (0..LAYERS) |i| {
        words[n] = LAYER_FLAG + 8 * i;
        n += 1;
    }
    for (words[0..n], 0..) |w, i| {
        try std.testing.expect(w >= FLAGS and w + 8 <= FLAGS + PAGE and w % 8 == 0);
        for (words[0..i]) |o| try std.testing.expect(o != w);
    }
}

test "prefill slots tile the region without overlap" {
    var end: usize = PREFILL;
    for (0..LAYERS) |i| {
        try std.testing.expectEqual(end, layerSlot(i));
        end += if (i % 4 == 3) ATT_SLOT else DN_SLOT;
    }
    try std.testing.expectEqual(TAIL, end);
}

pub const Settings = settings.Settings;
const settings = @import("tp_settings.zig");

const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// this sequence and the window's rows (seq << 5 | rows), after every earlier dispatch of the serial encoder
    \\kernel void tp_post(device atomic_uint* sync [[buffer(0)]], constant uint& seq [[buffer(1)]],
    \\                    const constant int* rows [[buffer(2)]]) {
    \\  atomic_store_explicit(&sync[1024], (seq << 5) | uint(rows[0]), memory_order_relaxed);
    \\}
    \\// post sx.x (every threadgroup stores the same word: the host sends this rank's part), then a threadgroup's first
    \\// thread polls until the peer's message for exchange sx.y has landed (its flag, stored after its bytes)
    \\inline void tp_post_poll(device atomic_uint* sync, device atomic_uint* flag, uint2 sx, int rows, bool poll, uint t) {
    \\  if (t != 0) return;
    \\  atomic_store_explicit(&sync[1024], (sx.x << 5) | uint(rows), memory_order_relaxed);
    \\  if (!poll) return;
    \\  uint polls = 0;
    \\  while (int(atomic_load_explicit(flag, memory_order_relaxed) - sx.y) < 0) {
    \\    if (++polls > 400000000u) { atomic_fetch_add_explicit(&sync[2048], 1u, memory_order_relaxed); return; }
    \\  }
    \\}
    \\// a row's argmax over this rank's vocab columns [lo, lo + n) of the full-width logits: (value, index), the
    \\// larger value and on a tie the lower index, as fz_argmax
    \\kernel void tp_argmax_part(device const bfloat* logits [[buffer(0)]], device uint* out [[buffer(1)]],
    \\    constant uint4& dims [[buffer(2)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    \\    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  device const bfloat* x = logits + row * dims.x;
    \\  float best = -INFINITY; uint at = 0xffffffffu;
    \\  for (uint i = dims.y + t; i < dims.y + dims.z; i += 1024) { const float v = float(x[i]); if (v > best) { best = v; at = i; } }
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  threadgroup float vb[32]; threadgroup uint va[32];
    \\  if (lane == 0) { vb[sg] = best; va[sg] = at; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (sg != 0) return;
    \\  best = vb[lane]; at = va[lane];
    \\  for (ushort o = 16; o > 0; o >>= 1) {
    \\    const float ob = simd_shuffle_xor(best, o); const uint oa = simd_shuffle_xor(at, o);
    \\    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
    \\  }
    \\  if (lane == 0) { out[2 * row] = as_type<uint>(best); out[2 * row + 1] = at; }
    \\}
    \\// post, poll until the peer's half lands, then each row's pick from both halves: the larger value, the lower index on a tie
    \\kernel void tp_pick(device atomic_uint* sync [[buffer(0)]], constant uint2& sx [[buffer(1)]],
    \\    const constant int* rows [[buffer(2)]], const device uint* mine [[buffer(3)]],
    \\    device atomic_uint* peer [[buffer(4)]], device uint* picks [[buffer(5)]], device atomic_uint* flag [[buffer(6)]]) {
    \\  tp_post_poll(sync, flag, sx, rows[0], true, 0);
    \\  for (int r = 0; r < rows[0]; r++) {
    \\    const float vm = as_type<float>(mine[2 * r]);
    \\    const uint am = mine[2 * r + 1];
    \\    const float vp = as_type<float>(atomic_load_explicit(&peer[2 * r], memory_order_relaxed));
    \\    const uint ap = atomic_load_explicit(&peer[2 * r + 1], memory_order_relaxed);
    \\    picks[r] = (vp > vm || (vp == vm && ap < am)) ? ap : am;
    \\  }
    \\}
    \\// the MTP head's merged draft pick (an index into its draft list) as a token id
    \\kernel void tp_ids(const device uint* pick [[buffer(0)]], const device uint* ids [[buffer(1)]],
    \\                   device uint* out [[buffer(2)]]) {
    \\  out[0] = ids[pick[0]];
    \\}
    \\// one thread polls until the 32-bit word reaches `value`; a give-up is counted, never silent
    \\kernel void tp_wait(device atomic_uint* word [[buffer(0)]], constant uint& value [[buffer(1)]],
    \\                    device atomic_uint* sync [[buffer(2)]]) {
    \\  uint polls = 0;
    \\  while (int(atomic_load_explicit(word, memory_order_relaxed) - value) < 0) {
    \\    if (++polls > 400000000u) { atomic_fetch_add_explicit(&sync[2048], 1u, memory_order_relaxed); return; }
    \\  }
    \\}
    \\// a stream update of row r at dims j * 256 + t from its branch, and each stream's partial sum of squares over those
    \\// dims: q4_hc_norm_plain's and q4_hc_norm_grouped's arithmetic
    \\inline void tp_update(const device bfloat* H, const device bfloat* INJ, float branch, device bfloat* HN, device float* SSP,
    \\    int j, int r, uint t, uint g, uint lane, threadgroup float (*part)[4]) {
    \\  constexpr int S = 4, D = 2560, W = S * D, NT = D / 256;
    \\  const int d = j * 256 + int(t);
    \\  float ss[S];
    \\  for (int s = 0; s < S; s++) {
    \\    const int e = s * D + d;
    \\    float hv = float(H[r * W + e]);
    \\    hv = float(bfloat(hv + float(bfloat(branch * float(INJ[r * S + s])))));
    \\    HN[r * W + e] = bfloat(hv);
    \\    ss[s] = simd_sum(hv * hv);
    \\  }
    \\  if (lane == 0) for (int s = 0; s < S; s++) part[g][s] = ss[s];
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (t < S) {
    \\    float total = 0.0f;
    \\    for (int k = 0; k < 8; k++) total += part[k][t];
    \\    SSP[(r * NT + j) * S + t] = total;
    \\  }
    \\}
    \\// post, wait for the peer's partial, then q4_hc_norm_plain's update with the branch from the two ranks' fp32
    \\// partials added in rank order and rounded once; the peer's words read as atomics (written by the host mid-buffer)
    \\kernel void tp_plain(device atomic_uint* sync [[buffer(0)]], constant uint2& sx [[buffer(1)]],
    \\    const constant int* rows [[buffer(2)]], const device float* MINE [[buffer(3)]], device atomic_uint* PEER [[buffer(4)]],
    \\    constant uint& rank [[buffer(5)]], const device bfloat* H [[buffer(6)]], const device bfloat* INJ [[buffer(7)]],
    \\    device bfloat* HN [[buffer(8)]], device float* SSP [[buffer(9)]], device atomic_uint* flag [[buffer(10)]],
    \\    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tpos [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
    \\  const int j = int(tg.x), r = int(tg.y), i = r * 2560 + j * 256 + int(tpos.x);
    \\  tp_post_poll(sync, flag, sx, rows[0], true, tpos.x);
    \\  threadgroup_barrier(mem_flags::mem_device);
    \\  const float mine = MINE[i], peer = as_type<float>(atomic_load_explicit(&PEER[i], memory_order_relaxed));
    \\  threadgroup float part[8][4];
    \\  tp_update(H, INJ, float(bfloat(rank == 0u ? mine + peer : peer + mine)), HN, SSP, j, r, tpos.x, g, lane, part);
    \\}
    \\// this rank's rows [own.x, own.y) of the MoE branch, as q4_hc_norm_grouped makes it on one Mac (the routed slots
    \\// summed in slot order, fp32; the gated shared expert; one bf16 rounding each): what the peer's streams take
    \\inline float tp_bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
    \\kernel void tp_branch(const device bfloat* Y [[buffer(0)]], const device float* WTS [[buffer(1)]],
    \\    const device float* LG [[buffer(2)]], device bfloat* OUT [[buffer(3)]], constant uint2& own [[buffer(4)]],
    \\    uint2 pos [[thread_position_in_grid]]) {
    \\  constexpr int D = 2560, TOPK = 10, NL = 513;
    \\  const int d = int(pos.x), r = int(own.x) + int(pos.y);
    \\  if (r >= int(own.y)) return;
    \\  float routed = 0.0f;
    \\  for (int k = 0; k < TOPK; k++) routed = fma(float(Y[(r * (TOPK + 1) + k) * D + d]), WTS[r * TOPK + k], routed);
    \\  const float shared = float(bfloat(float(Y[(r * (TOPK + 1) + TOPK) * D + d]) * tp_bsig(float(bfloat(LG[r * NL + NL - 1])))));
    \\  OUT[(r - int(own.x)) * D + d] = bfloat(float(bfloat(routed)) + shared);
    \\}
    \\// post, then q4_hc_norm_grouped's stream update with each row's branch from the rank that made it (rows
    \\// [split.x, split.y) this rank's, in MINE, at once; the rest the peer's, from split.z, in THEIRS, once landed)
    \\kernel void tp_merge(device atomic_uint* sync [[buffer(0)]], constant uint2& sx [[buffer(1)]],
    \\    const constant int* rows [[buffer(2)]], const device bfloat* H [[buffer(3)]], const device bfloat* INJ [[buffer(4)]],
    \\    const device bfloat* MINE [[buffer(5)]], device atomic_uint* THEIRS [[buffer(6)]], constant uint4& split [[buffer(7)]],
    \\    device bfloat* HN [[buffer(8)]], device float* SSP [[buffer(9)]], device atomic_uint* flag [[buffer(10)]],
    \\    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tpos [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
    \\  const int j = int(tg.x), r = int(tg.y), d = j * 256 + int(tpos.x);
    \\  const bool mine = uint(r) >= split.x && uint(r) < split.y;
    \\  tp_post_poll(sync, flag, sx, rows[0], !mine, tpos.x);
    \\  threadgroup_barrier(mem_flags::mem_device);
    \\  float branch;
    \\  if (mine) branch = float(MINE[(r - int(split.x)) * 2560 + d]);
    \\  else {
    \\    const uint at = uint(r - int(split.z)) * 2560u + uint(d);
    \\    branch = float(as_type<bfloat2>(atomic_load_explicit(&THEIRS[at >> 1], memory_order_relaxed))[at & 1]);
    \\  }
    \\  threadgroup float part[8][4];
    \\  tp_update(H, INJ, branch, HN, SSP, j, r, tpos.x, g, lane, part);
    \\}
;

/// Bytes the host copies into the peer's window once the GPU has posted the job's sequence.
pub const Write = struct { src: [*]const u8, len: usize, dst: usize };

/// A posted sequence's work: a decode exchange, or prefill writes and their flag.
const Job = struct {
    kind: enum { exchange, reduce, pick, send } = .send,
    writes: [8]Write = undefined,
    n: usize = 0,
    flag: usize = 0,
    value: u64 = 0,
};
const JOBS = 1024;

pub const Tp2 = struct {
    rank: u32,
    peer: u32,
    ep: *fabric.mcdma.Endpoint,
    rd: fabric.rdma.Rdma,
    win: []u8,
    wbuf: mtl.Buffer,
    post: mtl.Pipeline,
    wait: mtl.Pipeline,
    branch_pipe: mtl.Pipeline,
    argmax_pipe: mtl.Pipeline,
    pick_pipe: mtl.Pipeline,
    merge_pipe: mtl.Pipeline,
    plain_pipe: mtl.Pipeline,
    last_x: u32 = 0, // the last expert exchange: where the next combine finds both partials
    last_seq: u32 = 0, // and the sequence it posts
    req: u64 = 0, // served requests so far, the same on both ranks
    ctrl: u64 = 0, // stop decisions so far, the same on both ranks
    one: mtl.Buffer, // a rows word holding 1, for posts that carry no rows
    pick_tmp: mtl.Buffer, // the MTP head's merged draft index
    ids_pipe: mtl.Pipeline,
    seq: u32 = 0, // the last sequence this Mac's GPU posts (its own count: prefill sends are rank 0's alone)
    xseq: u32 = 0, // decode exchanges so far, the same count on both Macs: their slots' parity and flag values
    call: u32 = 0, // prefill chunks split so far (both ranks count the same): their flags' values
    mark_seq: u64 = 0, // marks exchanged after pair chunks (both ranks count the same)
    jobs: []Job,
    queued: std.atomic.Value(u32) = .init(0), // jobs written: the service reads a job only below this
    thread: ?std.Thread = null,
    stop: std.atomic.Value(bool) = .init(false),
    quitting: std.atomic.Value(bool) = .init(false), // served rank 1: end the wait for the next request
    failed: std.atomic.Value(bool) = .init(false),
    trace: bool = false, // TF_TP_TRACE: log every job the host serves
    local: bool = false, // TF_TP_LOCAL: serve every job at once, nothing sent (the GPU side's cost alone; wrong replies)
    stats: bool = false, // TF_TP_STATS: the host's time from a job's post to its serve, every 4096 jobs
    held_ns: u64 = 0,
    held_max: u64 = 0,
    held_kind: [4][2]u64 = @splat(.{ 0, 0 }), // by job kind: total ticks, jobs

    /// Connect to the peer named in the settings file and start the host's service thread.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, settings_path: []const u8) !*Tp2 {
        const f = try mtl.MappedFile.open(try gpa.dupeSentinel(u8, settings_path, 0));
        const s = try settings.read(gpa, f.bytes[0..f.size]);
        if (s.rank > 1 or s.links.len != 1) return error.TpTwoRanksOneLink;
        const lib = try gpa.dupeSentinel(u8, s.library, 0);
        const ep = try fabric.mcdma.Endpoint.create(gpa, lib, .{ .rank = s.rank, .ranks = 2, .window_bytes = WINDOW, .staging_bytes = 32 << 20, .links = s.links, .timeout_ns = 60 * std.time.ns_per_s, .connect_timeout_ns = 300 * std.time.ns_per_s });
        const rd = ep.rdma();
        const win = rd.window(); // zeroed by the endpoint before it connected: a fast peer's first words may be here already
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const lib_m = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        const t = try gpa.create(Tp2);
        t.* = .{
            .rank = s.rank,
            .peer = 1 - s.rank,
            .ep = ep,
            .rd = rd,
            .win = win,
            .wbuf = try device.bufferNoCopy(win.ptr, win.len, opts),
            .post = try mtl.Pipeline.init(device, lib_m, "tp_post", false),
            .wait = try mtl.Pipeline.init(device, lib_m, "tp_wait", false),
            .branch_pipe = try mtl.Pipeline.init(device, lib_m, "tp_branch", false),
            .argmax_pipe = try mtl.Pipeline.init(device, lib_m, "tp_argmax_part", false),
            .pick_pipe = try mtl.Pipeline.init(device, lib_m, "tp_pick", false),
            .merge_pipe = try mtl.Pipeline.init(device, lib_m, "tp_merge", false),
            .plain_pipe = try mtl.Pipeline.init(device, lib_m, "tp_plain", false),
            .one = try device.buffer(16, opts),
            .pick_tmp = try device.buffer(16, opts),
            .ids_pipe = try mtl.Pipeline.init(device, lib_m, "tp_ids", false),
            .jobs = try gpa.alloc(Job, JOBS),
        };
        t.one.slice(i32, 1)[0] = 1;
        t.trace = std.c.getenv("TF_TP_TRACE") != null;
        t.local = std.c.getenv("TF_TP_LOCAL") != null;
        t.stats = std.c.getenv("TF_TP_STATS") != null;
        // both ranks up before the first round: a word each way
        try rd.signal(t.peer, HELLO, 1);
        while (@atomicLoad(u64, t.word64(HELLO), .acquire) < 1) std.atomic.spinLoopHint();
        t.thread = try std.Thread.spawn(.{}, service, .{ t, t.seq +% 1 });
        std.log.info("TP=2 rank {d} connected", .{t.rank});
        return t;
    }

    /// Stop the host's service thread and close the link; every GPU use of the window has ended.
    pub fn deinit(t: *Tp2) void {
        t.rd.flush() catch {}; // what this rank sent (rank 0's last request) is out before the link closes
        t.stop.store(true, .release);
        if (t.thread) |th| th.join();
        t.ep.deinit();
    }

    /// The window rows whose experts (every routed slot and the shared one) and MoE branch this rank computes, [lo, hi):
    /// rank 0 the first half, rounded up, rank 1 the rest.
    pub fn ownRows(t: *const Tp2, rows: usize) [2]usize {
        const half = (rows + 1) / 2;
        return if (t.rank == 0) .{ 0, half } else .{ half, rows };
    }

    /// GPU waits that gave up (a peer that never answered); nonzero means the run is invalid.
    pub fn gaveUp(t: *const Tp2) u32 {
        return @atomicLoad(u32, t.word32(SYNC + GAVE_UP * 4), .acquire);
    }

    /// A new sequence with its job queued for the service thread.
    fn queue(t: *Tp2, job: Job) u32 {
        t.seq +%= 1;
        t.jobs[t.seq % JOBS] = job;
        t.queued.store(t.seq, .release);
        return t.seq;
    }

    fn encodePost(t: *Tp2, enc: mtl.ComputeEncoder, seq: u32, rows_buf: mtl.Buffer, rows_off: usize) void {
        enc.setPipeline(t.post);
        enc.setBuffer(t.wbuf, SYNC, 0);
        enc.setBytes(std.mem.asBytes(&seq), 1);
        enc.setBuffer(rows_buf, rows_off, 2);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }

    /// On the serial encoder: the GPU waits until the window's 32-bit word at `off` reaches `value`.
    pub fn waitWord(t: *Tp2, enc: mtl.ComputeEncoder, off: usize, value: u32) void {
        enc.setPipeline(t.wait);
        enc.setBuffer(t.wbuf, off, 0);
        enc.setBytes(std.mem.asBytes(&value), 1);
        enc.setBuffer(t.wbuf, SYNC, 2);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }

    /// After a target layer's experts for this rank's rows (`ownRows`), on the serial encoder: their MoE branch (bf16)
    /// into the send buffer; `combine` posts it and updates every row.
    pub fn exchange(t: *Tp2, enc: mtl.ComputeEncoder, y: anytype, wts: anytype, lg: anytype, rows: usize) void {
        t.xseq +%= 1;
        const x = t.xseq;
        t.last_x = x;
        t.last_seq = t.queue(.{ .kind = .exchange, .value = x });
        const own = t.ownRows(rows);
        if (own[1] > own[0]) {
            const own2 = [2]u32{ @intCast(own[0]), @intCast(own[1]) };
            enc.setPipeline(t.branch_pipe);
            for ([_]@TypeOf(y){ y, wts, lg }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
            enc.setBuffer(t.wbuf, sendX(x), 3);
            enc.setBytes(std.mem.asBytes(&own2), 4);
            enc.dispatchThreads(mtl.Size.of(D, own[1] - own[0], 1), mtl.Size.of(256, 1, 1));
        }
    }

    /// The last exchange posted, and its MoE branches into the streams (q4_hc_norm_grouped's job): this rank's rows
    /// from its send buffer at once, the peer's once they land.
    pub fn combine(t: *Tp2, enc: mtl.ComputeEncoder, h: anytype, inj: anytype, out: anytype, ssp: anytype, rows_buf: anytype, rows: usize) void {
        const own = t.ownRows(rows);
        const peer_lo: usize = if (t.rank == 0) own[1] else 0;
        const split = [4]u32{ @intCast(own[0]), @intCast(own[1]), @intCast(peer_lo), 0 };
        enc.setPipeline(t.merge_pipe);
        enc.setBuffer(t.wbuf, SYNC, 0);
        enc.setBytes(std.mem.asBytes(&[2]u32{ t.last_seq, t.last_x }), 1);
        enc.setBuffer(rows_buf.b, rows_buf.off, 2);
        enc.setBuffer(h.b, h.off, 3);
        enc.setBuffer(inj.b, inj.off, 4);
        enc.setBuffer(t.wbuf, sendX(t.last_x), 5);
        enc.setBuffer(t.wbuf, DECODE + (t.last_x % 2) * SLOT, 6);
        enc.setBytes(std.mem.asBytes(&split), 7);
        enc.setBuffer(out.b, out.off, 8);
        enc.setBuffer(ssp.b, ssp.off, 9);
        enc.setBuffer(t.wbuf, FLAG, 10);
        enc.dispatchThreads(mtl.Size.of(D, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// The head's picks from this rank's vocab columns [lo, lo + n) of `logits` (rows x vocab, bf16) and the peer's: each half's argmax, swapped, merged (the larger value, the lower index on a tie: one Mac's argmax exactly).
    pub fn argmax(t: *Tp2, enc: mtl.ComputeEncoder, logits: anytype, vocab: usize, lo: usize, n: usize, picks: anytype, rows_buf: anytype, rows: usize) void {
        t.xseq +%= 1;
        const x = t.xseq;
        const seq = t.queue(.{ .kind = .pick, .value = x });
        const dims = [4]u32{ @intCast(vocab), @intCast(lo), @intCast(n), 0 };
        enc.setPipeline(t.argmax_pipe);
        enc.setBuffer(logits.b, logits.off, 0);
        enc.setBuffer(t.wbuf, sendA(x), 1);
        enc.setBytes(std.mem.asBytes(&dims), 2);
        enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
        enc.setPipeline(t.pick_pipe);
        enc.setBuffer(t.wbuf, SYNC, 0);
        enc.setBytes(std.mem.asBytes(&[2]u32{ seq, x }), 1);
        enc.setBuffer(rows_buf.b, rows_buf.off, 2);
        enc.setBuffer(t.wbuf, sendA(x), 3);
        enc.setBuffer(t.wbuf, RECVA + (x % 2) * PAGE, 4);
        enc.setBuffer(picks.b, picks.off, 5);
        enc.setBuffer(t.wbuf, FLAG, 6);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }

    /// After `argmax` over the MTP head's draft logits into `pick_tmp`: the merged pick's token id into `out`.
    pub fn mapIds(t: *Tp2, enc: mtl.ComputeEncoder, ids: anytype, out: anytype) void {
        enc.setPipeline(t.ids_pipe);
        enc.setBuffer(t.pick_tmp, 0, 0);
        enc.setBuffer(ids.b, ids.off, 1);
        enc.setBuffer(out.b, out.off, 2);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }

    /// Where the next reduce's partial goes (fp32, rows x 2560): the host sends it from there.
    pub fn partNext(t: *const Tp2) struct { b: mtl.Buffer, off: usize } {
        return .{ .b = t.wbuf, .off = sendR(t.xseq +% 1) };
    }

    /// After a split projection wrote this rank's partial into `part`: post, wait for the host to swap partials with
    /// the peer, then q4_hc_norm_plain's stream update of `h` into `out` and `ssp` with the branch rank 0's partial +
    /// rank 1's, rounded once (bf16, rows x 2560).
    pub fn plain(t: *Tp2, enc: mtl.ComputeEncoder, h: anytype, inj: anytype, out: anytype, ssp: anytype, rows_buf: anytype, rows: usize) void {
        t.xseq +%= 1;
        const x = t.xseq;
        const seq = t.queue(.{ .kind = .reduce, .value = x });
        enc.setPipeline(t.plain_pipe);
        enc.setBuffer(t.wbuf, SYNC, 0);
        enc.setBytes(std.mem.asBytes(&[2]u32{ seq, x }), 1);
        enc.setBuffer(rows_buf.b, rows_buf.off, 2);
        enc.setBuffer(t.wbuf, sendR(x), 3);
        enc.setBuffer(t.wbuf, REDUCE + (x % 2) * PART, 4);
        enc.setBytes(std.mem.asBytes(&t.rank), 5);
        enc.setBuffer(h.b, h.off, 6);
        enc.setBuffer(inj.b, inj.off, 7);
        enc.setBuffer(out.b, out.off, 8);
        enc.setBuffer(ssp.b, ssp.off, 9);
        enc.setBuffer(t.wbuf, FLAG, 10);
        enc.dispatchThreads(mtl.Size.of(D, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// On the serial encoder, after the work that wrote the sources: once the GPU gets here the host copies `writes` into the peer's window and stores `value` at its `flag`; the GPU does not wait.
    pub fn send(t: *Tp2, enc: mtl.ComputeEncoder, writes: []const Write, flag: usize, value: u64) void {
        var job: Job = .{ .kind = .send, .n = writes.len, .flag = flag, .value = value };
        @memcpy(job.writes[0..writes.len], writes);
        const seq = t.queue(job);
        t.encodePost(enc, seq, t.one, 0);
    }

    /// The host side alone: copy `writes` and signal now (after a command buffer has completed).
    pub fn sendNow(t: *Tp2, writes: []const Write, flag: usize, value: u64) !void {
        for (writes, 0..) |w, j| {
            if (j + 1 == writes.len) try t.rd.write2Signal(t.peer, w.dst, w.src[0..w.len], &.{}, flag, value) else try t.rd.write(t.peer, w.dst, w.src[0..w.len]);
        }
        if (writes.len == 0) try t.rd.signal(t.peer, flag, value);
    }

    /// Served, rank 0: hand rank 1 the next request (`head` then the prompt's tokens) in one message.
    pub fn sendRequest(t: *Tp2, head: []const u8, prompt: []const u32) !void {
        if (head.len > 64 or prompt.len > REQ_TOKENS) return error.TpRequestTooLarge;
        t.req += 1;
        var h64: [64]u8 = @splat(0);
        @memcpy(h64[0..head.len], head);
        try t.rd.write2Signal(t.peer, REQ, &h64, std.mem.sliceAsBytes(prompt), REQ_FLAG, t.req);
    }

    /// After a pair chunk: this Mac's MARK is free for the peer's states of mark exchange `seq`.
    pub fn markReady(t: *Tp2, seq: u64) !void {
        try t.rd.signal(t.peer, MARK_READY, seq);
    }

    /// Served, rank 1: tell rank 0 whether this request's resume state is in place (rank 0 waits before its prompt pass).
    pub fn ackRequest(t: *Tp2, ok: bool) !void {
        try t.rd.signal(t.peer, ACK_FLAG, (t.req << 1) | @intFromBool(ok));
    }

    /// Served, rank 0: rank 1's answer for the request just sent: true when it restored the same resume state.
    pub fn waitAck(t: *Tp2) !bool {
        const t0 = std.c.mach_absolute_time();
        while (true) {
            const v = @atomicLoad(u64, t.word64(ACK_FLAG), .acquire);
            if (v >> 1 >= t.req) return v >> 1 == t.req and v & 1 != 0;
            if (std.c.mach_absolute_time() - t0 > 7_200_000_000) return error.TpPeerSilent; // 300 s: rank 1 may still be loading
            std.atomic.spinLoopHint();
        }
    }

    /// Served, rank 1, between requests: whether rank 0 has written one this Mac has not taken (or the wait is ending).
    pub fn requestWaiting(t: *const Tp2) bool {
        return t.quitting.load(.acquire) or @atomicLoad(u64, t.word64(REQ_FLAG), .acquire) > t.req;
    }

    /// Served, rank 1: wait for rank 0's next request; its 64-byte head and the window's prompt tokens after it; null once `quitting` ends the wait.
    pub fn waitRequest(t: *Tp2) ?struct { head: *const [64]u8, tokens: [*]const u32 } {
        t.req += 1;
        var spins: usize = 0;
        while (@atomicLoad(u64, t.word64(REQ_FLAG), .acquire) < t.req) {
            if (t.quitting.load(.acquire)) return null;
            spins += 1;
            if (spins > 100_000) { // idle: back off
                const ts: std.c.timespec = .{ .sec = 0, .nsec = 50_000 };
                _ = std.c.nanosleep(&ts, null);
            } else std.atomic.spinLoopHint();
        }
        return .{ .head = @ptrCast(t.win.ptr + REQ), .tokens = @ptrCast(@alignCast(t.win.ptr + REQ + 64)) };
    }

    /// A step both ranks take in the same order (a prompt chunk, the first token, a round): rank 0's decision to stop there (a stop string, a cancel) reaches rank 1, so both end on the same step. Rank 1 waits for it.
    pub fn agree(t: *Tp2, quit: bool, depth: u32) !struct { quit: bool, depth: u32 } {
        t.ctrl += 1;
        if (t.rank == 0) {
            try t.rd.signal(t.peer, CTRL_FLAG, (t.ctrl << 6) | (@as(u64, depth & 31) << 1) | @intFromBool(quit));
            return .{ .quit = quit, .depth = depth };
        }
        const t0 = std.c.mach_absolute_time();
        while (true) {
            const v = @atomicLoad(u64, t.word64(CTRL_FLAG), .acquire);
            if (v >> 6 >= t.ctrl) return .{ .quit = quit or (v >> 6 == t.ctrl and v & 1 != 0), .depth = @intCast((v >> 1) & 31) };
            if (std.c.mach_absolute_time() - t0 > 240_000_000) return error.TpPeerSilent; // 10 s
            std.atomic.spinLoopHint();
        }
    }

    /// The host waits until the window's word at `off` reaches `value`.
    pub fn hostWait(t: *Tp2, off: usize, value: u64) void {
        if (t.trace) std.debug.print("TP rank{d} host waits for word {d} >= {d} (now {d})\n", .{ t.rank, off, value, @atomicLoad(u64, t.word64(off), .acquire) });
        while (!reached(@truncate(@atomicLoad(u64, t.word64(off), .acquire)), @truncate(value))) std.atomic.spinLoopHint(); // call counts wrap
        if (t.trace) std.debug.print("TP rank{d} word {d} reached {d}\n", .{ t.rank, off, value });
    }

    pub fn window(t: *const Tp2) mtl.Buffer {
        return t.wbuf;
    }

    pub fn bytes(t: *const Tp2, off: usize) [*]u8 {
        return t.win.ptr + off;
    }

    fn word32(t: *const Tp2, off: usize) *u32 {
        return @ptrCast(@alignCast(t.win.ptr + off));
    }

    fn word64(t: *const Tp2, off: usize) *u64 {
        return @ptrCast(@alignCast(t.win.ptr + off));
    }

    /// Each sequence in order, once the GPU posts it: a decode exchange (send this rank's part; the GPU itself waits
    /// for the peer's, by the flag its message sets) or a prefill send (its writes, then its flag). While idle it
    /// watches the last exchange: 10 s without the peer's answer fails the link.
    fn service(t: *Tp2, first: u32) void {
        const posted = t.word32(SYNC + GPU * 4);
        const flag = t.word64(FLAG);
        var seq = first;
        var want: u64 = 0; // the last exchange sent, and when (0: none yet)
        var want_at: u64 = 0;
        while (true) {
            var w = @atomicLoad(u32, posted, .acquire);
            while (!postReached(w, seq) or !reached(t.queued.load(.acquire), seq)) {
                if (t.stop.load(.acquire)) return;
                if (want_at > 0 and !reached(@truncate(@atomicLoad(u64, flag, .acquire)), @truncate(want)) and !t.failed.load(.acquire) and std.c.mach_absolute_time() - want_at > 240_000_000) { // 10 s of 24 MHz ticks
                    t.fail(seq, error.PeerSilent);
                    t.land(want);
                }
                std.atomic.spinLoopHint();
                w = @atomicLoad(u32, posted, .acquire);
            }
            const job = t.jobs[seq % JOBS];
            const seen = if (t.stats) std.c.mach_absolute_time() else 0;
            if ((t.local and job.kind != .send) or t.failed.load(.acquire)) { // a failed link drains: the GPU never hangs
                if (job.kind != .send) t.land(job.value);
                seq +%= 1;
                continue;
            }
            if (t.trace) std.debug.print("TP rank{d} seq {d} {s} writes {d} flag {d} value {d} gave_up {d}\n", .{ t.rank, seq, @tagName(job.kind), job.n, job.flag, job.value, t.gaveUp() });
            const x = job.value;
            switch (job.kind) {
                .exchange => {
                    const own = t.ownRows(w & 31);
                    t.ep.writeSignalFrom(t.peer, sendX(@intCast(x)), DECODE + (x % 2) * SLOT, (own[1] - own[0]) * D * 2, FLAG, x) catch |err| t.fail(seq, err);
                },
                .reduce => t.ep.writeSignalFrom(t.peer, sendR(@intCast(x)), REDUCE + (x % 2) * PART, @as(usize, w & 31) * D * 4, FLAG, x) catch |err| t.fail(seq, err),
                .pick => t.ep.writeSignalFrom(t.peer, sendA(@intCast(x)), RECVA + (x % 2) * PAGE, @as(usize, w & 31) * 8, FLAG, x) catch |err| t.fail(seq, err),
                .send => t.sendNow(job.writes[0..job.n], job.flag, job.value) catch |err| t.fail(seq, err),
            }
            if (job.kind != .send) {
                want = x;
                want_at = std.c.mach_absolute_time();
                if (t.failed.load(.acquire)) t.land(x);
            }
            if (t.stats) {
                const held = std.c.mach_absolute_time() - seen; // ticks: 41.67 ns each on Apple silicon
                t.held_ns += held;
                t.held_max = @max(t.held_max, held);
                t.held_kind[@backingInt(job.kind)][0] += held;
                t.held_kind[@backingInt(job.kind)][1] += 1;
                if (seq % 4096 == 0) {
                    std.debug.print("TP rank{d} jobs to {d}: host held {d:.1} us a job, max {d:.1} us\n", .{ t.rank, seq, @as(f64, @floatFromInt(t.held_ns)) / 4096.0 / 24.0, @as(f64, @floatFromInt(t.held_max)) / 24.0 });
                    for (t.held_kind, 0..) |hk, kk| if (hk[1] > 0) std.debug.print("TP rank{d}   {s}: {d:.1} us a job ({d} jobs)\n", .{ t.rank, @tagName(@as(@TypeOf(job.kind), @fromBackingInt(@intCast(kk)))), @as(f64, @floatFromInt(hk[0])) / @as(f64, @floatFromInt(hk[1])) / 24.0, hk[1] });
                    t.held_ns = 0;
                    t.held_max = 0;
                    t.held_kind = @splat(.{ 0, 0 });
                }
            }
            seq +%= 1;
        }
    }

    /// A drained exchange: the flag the GPU waits on set here, as if the peer's message had landed (its bytes stale).
    fn land(t: *Tp2, x: u64) void {
        const flag = t.word64(FLAG);
        if (!reached(@truncate(@atomicLoad(u64, flag, .acquire)), @truncate(x))) @atomicStore(u64, flag, x, .release);
    }

    fn fail(t: *Tp2, seq: u32, err: anyerror) void {
        std.log.err("TP=2 rank {d}: sequence {d} failed: {s}", .{ t.rank, seq, @errorName(err) });
        t.failed.store(true, .release);
    }
};

test "post words and counters compare across their wraps" {
    const top: u32 = 1 << SEQ_BITS;
    for ([_]u32{ 1, top - 2, top - 1, top, top + 1, 0xFFFF_FFFE, 0xFFFF_FFFF, 0 }) |seq| {
        const w = (seq << 5) | 7; // the GPU's word: the sequence's low 27 bits
        try std.testing.expect(postReached(w, seq) and postReached(w, seq -% 5) and !postReached(w, seq +% 1));
        try std.testing.expect(reached(seq, seq) and reached(seq +% 3, seq) and !reached(seq, seq +% 1));
    }
    try std.testing.expect(!postReached(0, 1)); // a fresh window: nothing posted
}

test "the host service follows the GPU's posts across the 27-bit post wrap and the 32-bit counter wrap" {
    const win = try std.heap.page_allocator.alloc(u8, WINDOW);
    defer std.heap.page_allocator.free(win);
    const jobs = try std.testing.allocator.alloc(Job, JOBS);
    defer std.testing.allocator.free(jobs);
    for ([_][2]u32{ .{ (1 << SEQ_BITS) - 4, 7 }, .{ 0xFFFF_FFFC, 0xFFFF_FFFD } }) |start| { // last sequence, last flag
        var t: Tp2 = .{ .rank = 0, .peer = 1, .ep = undefined, .rd = undefined, .win = win, .wbuf = undefined, .post = undefined, .wait = undefined, .branch_pipe = undefined, .argmax_pipe = undefined, .pick_pipe = undefined, .merge_pipe = undefined, .plain_pipe = undefined, .one = undefined, .pick_tmp = undefined, .ids_pipe = undefined, .jobs = jobs, .local = true };
        t.seq = start[0];
        t.queued.store(start[0], .release);
        @atomicStore(u32, t.word32(SYNC + GPU * 4), (start[0] << 5) | 4, .release);
        @atomicStore(u64, t.word64(FLAG), start[1], .release);
        const th = try std.Thread.spawn(.{}, Tp2.service, .{ &t, start[0] +% 1 });
        defer {
            t.stop.store(true, .release);
            th.join();
        }
        var x = start[1];
        for (0..8) |_| { // local serving lands each exchange's flag once its post is seen
            x +%= 1;
            const seq = t.queue(.{ .kind = .exchange, .value = x });
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 2_000_000 };
            _ = std.c.nanosleep(&ts, null);
            try std.testing.expect(@as(u32, @truncate(@atomicLoad(u64, t.word64(FLAG), .acquire))) != x); // not before its post
            @atomicStore(u32, t.word32(SYNC + GPU * 4), (seq << 5) | 4, .release);
            const t0 = std.c.mach_absolute_time();
            while (@as(u32, @truncate(@atomicLoad(u64, t.word64(FLAG), .acquire))) != x) {
                if (std.c.mach_absolute_time() - t0 > 24_000_000) return error.ServiceStuck; // 1 s
                std.atomic.spinLoopHint();
            }
        }
    }
}

test "partNext and hostWait across the 32-bit wrap: the next exchange's slot, and a wait for a call past the wrap" {
    const win = try std.heap.page_allocator.alloc(u8, WINDOW);
    defer std.heap.page_allocator.free(win);
    var t: Tp2 = .{ .rank = 0, .peer = 1, .ep = undefined, .rd = undefined, .win = win, .wbuf = undefined, .post = undefined, .wait = undefined, .branch_pipe = undefined, .argmax_pipe = undefined, .pick_pipe = undefined, .merge_pipe = undefined, .plain_pipe = undefined, .one = undefined, .pick_tmp = undefined, .ids_pipe = undefined, .jobs = &.{} };
    for ([_]u32{ 0xFFFF_FFFE, 0xFFFF_FFFF, 0 }) |x| { // partNext names the slot `plain` sends the next partial from
        t.xseq = x;
        const at = t.partNext().off;
        t.xseq +%= 1;
        try std.testing.expectEqual(sendR(t.xseq), at);
    }
    const Peer = struct { // the peer's message landing a call's flag, a little later
        fn land(w: *u64, v: u64) void {
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000_000 };
            _ = std.c.nanosleep(&ts, null);
            @atomicStore(u64, w, v, .release);
        }
    };
    @atomicStore(u64, t.word64(BACK_FLAG), 0xFFFF_FFFE, .release);
    for ([_]u64{ 0xFFFF_FFFF, 0, 1 }) |call| { // tp.call's flags just before and after its wrap
        const th = try std.Thread.spawn(.{}, Peer.land, .{ t.word64(BACK_FLAG), call });
        t.hostWait(BACK_FLAG, call);
        try std.testing.expectEqual(call, @atomicLoad(u64, t.word64(BACK_FLAG), .acquire)); // it waited for the landing
        th.join();
    }
}
