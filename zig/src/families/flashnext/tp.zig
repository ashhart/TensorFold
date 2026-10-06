//! Speed-up mode for Flash Next: two Macs with the whole model each, half the work each, over MCDMA. Decode: each rank computes the routed experts it owns (rank 0 the first 256, rank 1 the rest) and both the shared one; after each target layer's experts the GPU sums its routed slots (fp32, slot order) and posts a sequence; the host sends the sum to the peer and serves the sequence once the peer's has landed; the next layer's combine adds the two in rank order (an fp32 reorder of one Mac's slot sum, the same on both ranks). DeltaNet layers split their heads the same way: each rank's out-projection partial, exchanged and added in rank order. Prefill: a chunk's rows split across the two Macs (replay.zig `chunkPair`); the host sends each layer's handoff into the peer's slot for that layer and signals the layer's flag, which the peer's GPU waits for.
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
const WINDOW = REQ + pages(64 + 4 * REQ_TOKENS);

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
const HOST = 0;
const GPU = 1024;
const GAVE_UP = 2048;

/// The receiving window's slot for layer i's prefill handoff (attention layers are i % 4 == 3).
pub fn layerSlot(i: usize) usize {
    const att = i / 4; // attention layers before i (3, 7, ... below it)
    return PREFILL + (i - att) * DN_SLOT + att * ATT_SLOT;
}

test "prefill slots tile the region without overlap" {
    var end: usize = PREFILL;
    for (0..LAYERS) |i| {
        try std.testing.expectEqual(end, layerSlot(i));
        end += if (i % 4 == 3) ATT_SLOT else DN_SLOT;
    }
    try std.testing.expectEqual(TAIL, end);
}

pub const Settings = struct { rank: u32, library: []const u8, links: []const fabric.mcdma.Link };

const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// this sequence and the window's rows (seq << 5 | rows), after every earlier dispatch of the serial encoder
    \\kernel void tp_post(device atomic_uint* sync [[buffer(0)]], constant uint& seq [[buffer(1)]],
    \\                    const constant int* rows [[buffer(2)]]) {
    \\  atomic_store_explicit(&sync[1024], (seq << 5) | uint(rows[0]), memory_order_relaxed);
    \\}
    \\// post then poll in one launch: the host sees the post while this thread polls for its serve
    \\kernel void tp_post_wait(device atomic_uint* sync [[buffer(0)]], constant uint& seq [[buffer(1)]],
    \\                         const constant int* rows [[buffer(2)]]) {
    \\  atomic_store_explicit(&sync[1024], (seq << 5) | uint(rows[0]), memory_order_relaxed);
    \\  uint polls = 0;
    \\  while (int(atomic_load_explicit(&sync[0], memory_order_relaxed) - seq) < 0) {
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
    \\// post, poll for the serve, then each row's pick from both halves: the larger value, the lower index on a tie
    \\kernel void tp_pick(device atomic_uint* sync [[buffer(0)]], constant uint& seq [[buffer(1)]],
    \\    const constant int* rows [[buffer(2)]], const device uint* mine [[buffer(3)]],
    \\    device atomic_uint* peer [[buffer(4)]], device uint* picks [[buffer(5)]]) {
    \\  atomic_store_explicit(&sync[1024], (seq << 5) | uint(rows[0]), memory_order_relaxed);
    \\  uint polls = 0;
    \\  while (int(atomic_load_explicit(&sync[0], memory_order_relaxed) - seq) < 0) {
    \\    if (++polls > 400000000u) { atomic_fetch_add_explicit(&sync[2048], 1u, memory_order_relaxed); return; }
    \\  }
    \\  for (int r = 0; r < rows[0]; r++) {
    \\    const float vm = as_type<float>(mine[2 * r]);
    \\    const uint am = mine[2 * r + 1];
    \\    const float vp = as_type<float>(atomic_load_explicit(&peer[2 * r], memory_order_relaxed));
    \\    const uint ap = atomic_load_explicit(&peer[2 * r + 1], memory_order_relaxed);
    \\    picks[r] = (vp > vm || (vp == vm && ap < am)) ? ap : am;
    \\  }
    \\}
    \\// one thread polls until the 32-bit word reaches `value`; a give-up is counted, never silent
    \\kernel void tp_wait(device atomic_uint* word [[buffer(0)]], constant uint& value [[buffer(1)]],
    \\                    device atomic_uint* sync [[buffer(2)]]) {
    \\  uint polls = 0;
    \\  while (int(atomic_load_explicit(word, memory_order_relaxed) - value) < 0) {
    \\    if (++polls > 400000000u) { atomic_fetch_add_explicit(&sync[2048], 1u, memory_order_relaxed); return; }
    \\  }
    \\}
    \\// the two ranks' fp32 partials added in rank order and rounded once
    \\kernel void tp_sum(const device float* mine [[buffer(0)]], const device float* peer [[buffer(1)]],
    \\                   device bfloat* out [[buffer(2)]], const constant int* rows [[buffer(3)]],
    \\                   constant uint& rank [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    \\  if (i >= uint(rows[0]) * 2560u) return;
    \\  out[i] = bfloat(rank == 0u ? mine[i] + peer[i] : peer[i] + mine[i]);
    \\}
    \\// this rank's routed slots summed in slot order, fp32 (the peer's slots are zero here): what the peer adds
    \\kernel void tp_moe_part(const device bfloat* y [[buffer(0)]], const device float* wts [[buffer(1)]],
    \\                        device float* part [[buffer(2)]], const constant int* rows [[buffer(3)]],
    \\                        uint i [[thread_position_in_grid]]) {
    \\  const uint r = i / 2560u, d = i % 2560u;
    \\  if (r >= uint(rows[0])) return;
    \\  float acc = 0.0f;
    \\  for (uint k = 0; k < 10u; k++) acc = fma(float(y[(r * 11u + k) * 2560u + d]), wts[r * 10u + k], acc);
    \\  part[i] = acc;
    \\}
    \\inline float tp_bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
    \\// q4_hc_norm_grouped with the routed sum from the two ranks' partials (rank 0's + rank 1's): the MoE branch into
    \\// the 4 streams, and each stream's partial sum of squares over this threadgroup's 256 dims
    \\kernel void tp_combine(const device bfloat* H [[buffer(0)]], const device bfloat* INJ [[buffer(1)]],
    \\    const device bfloat* Y [[buffer(2)]], const device float* LG [[buffer(3)]], const device float* P0 [[buffer(4)]],
    \\    const device float* P1 [[buffer(5)]], device bfloat* HN [[buffer(6)]], device float* SSP [[buffer(7)]],
    \\    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint3 tpos [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
    \\  constexpr int S = 4, D = 2560, TOPK = 10, NL = 513, W = S * D, NT = D / 256;
    \\  const uint t = tpos.x;
    \\  const int j = int(tg.x), r = int(tg.y), d = j * 256 + int(t);
    \\  threadgroup float part[8][S];
    \\  float ss[S];
    \\  const float routed = P0[r * D + d] + P1[r * D + d];
    \\  const float shared = float(bfloat(float(Y[(r * (TOPK + 1) + TOPK) * D + d]) * tp_bsig(float(bfloat(LG[r * NL + NL - 1])))));
    \\  const float branch = float(bfloat(float(bfloat(routed)) + shared));
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
    moe_part_pipe: mtl.Pipeline,
    argmax_pipe: mtl.Pipeline,
    pick_pipe: mtl.Pipeline,
    combine_pipe: mtl.Pipeline,
    last_x: u32 = 0, // the last expert exchange: where the next combine finds both partials
    req: u64 = 0, // served requests so far, the same on both ranks
    ctrl: u64 = 0, // stop decisions so far, the same on both ranks
    post_wait: mtl.Pipeline,
    fused: bool = true, // post and wait in one launch (TF_TP_FUSED=0: two)
    sum_pipe: mtl.Pipeline,
    one: mtl.Buffer, // a rows word holding 1, for posts that carry no rows
    seq: u32 = 0, // the last sequence this Mac's GPU posts (its own count: prefill sends are rank 0's alone)
    xseq: u32 = 0, // decode exchanges so far, the same count on both Macs: their slots' parity and flag values
    call: u32 = 0, // prefill chunks split so far (both ranks count the same): their flags' values
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
        const s = try std.json.parseFromSliceLeaky(Settings, gpa, f.bytes[0..f.size], .{ .allocate = .alloc_always });
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
            .moe_part_pipe = try mtl.Pipeline.init(device, lib_m, "tp_moe_part", false),
            .argmax_pipe = try mtl.Pipeline.init(device, lib_m, "tp_argmax_part", false),
            .pick_pipe = try mtl.Pipeline.init(device, lib_m, "tp_pick", false),
            .combine_pipe = try mtl.Pipeline.init(device, lib_m, "tp_combine", false),
            .post_wait = try mtl.Pipeline.init(device, lib_m, "tp_post_wait", false),
            .sum_pipe = try mtl.Pipeline.init(device, lib_m, "tp_sum", false),
            .one = try device.buffer(16, opts),
            .jobs = try gpa.alloc(Job, JOBS),
        };
        t.one.slice(i32, 1)[0] = 1;
        t.trace = std.c.getenv("TF_TP_TRACE") != null;
        t.local = std.c.getenv("TF_TP_LOCAL") != null;
        t.stats = std.c.getenv("TF_TP_STATS") != null;
        t.fused = if (std.c.getenv("TF_TP_FUSED")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        // both ranks up before the first round: a word each way
        try rd.signal(t.peer, HELLO, 1);
        while (@atomicLoad(u64, t.word64(HELLO), .acquire) < 1) std.atomic.spinLoopHint();
        t.thread = try std.Thread.spawn(.{}, service, .{t});
        std.log.info("TP=2 rank {d} connected; experts {any}", .{ t.rank, t.own() });
        return t;
    }

    /// Stop the host's service thread and close the link; every GPU use of the window has ended.
    pub fn deinit(t: *Tp2) void {
        t.rd.flush() catch {}; // what this rank sent (rank 0's last request) is out before the link closes
        t.stop.store(true, .release);
        if (t.thread) |th| th.join();
        t.ep.deinit();
    }

    /// The experts this rank computes: [lo, hi) and whether the shared expert is its (both compute it).
    pub fn own(t: *const Tp2) [4]u32 {
        return if (t.rank == 0) .{ 0, 256, 1, 0 } else .{ 256, 512, 1, 0 };
    }

    /// GPU waits that gave up (a peer that never answered); nonzero means the run is invalid.
    pub fn gaveUp(t: *const Tp2) u32 {
        return @atomicLoad(u32, t.word32(SYNC + GAVE_UP * 4), .acquire);
    }

    /// A new sequence with its job queued for the service thread.
    fn queue(t: *Tp2, job: Job) u32 {
        t.seq += 1;
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

    /// After a target layer's experts (`y` its slots, the peer's zero), on the serial encoder: sum this rank's routed slots into the send buffer, post, wait for the host to swap sums with the peer; `combine` then adds them.
    pub fn exchange(t: *Tp2, enc: mtl.ComputeEncoder, y: anytype, wts: anytype, rows_buf: anytype, rows: usize) void {
        t.xseq += 1;
        const x = t.xseq;
        t.last_x = x;
        const seq = t.queue(.{ .kind = .exchange, .value = x });
        enc.setPipeline(t.moe_part_pipe);
        enc.setBuffer(y.b, y.off, 0);
        enc.setBuffer(wts.b, wts.off, 1);
        enc.setBuffer(t.wbuf, sendX(x), 2);
        enc.setBuffer(rows_buf.b, rows_buf.off, 3);
        enc.dispatchThreads(mtl.Size.of(rows * D, 1, 1), mtl.Size.of(256, 1, 1));
        t.postWait(enc, seq, rows_buf.b, rows_buf.off);
    }

    /// The last exchange's MoE branch into the streams (q4_hc_norm_grouped's job): routed = rank 0's sum + rank 1's.
    pub fn combine(t: *Tp2, enc: mtl.ComputeEncoder, h: anytype, inj: anytype, y: anytype, lg: anytype, out: anytype, ssp: anytype, rows: usize) void {
        const mine = sendX(t.last_x);
        const theirs = DECODE + (t.last_x % 2) * SLOT;
        enc.setPipeline(t.combine_pipe);
        for ([_]@TypeOf(h){ h, inj, y, lg }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
        enc.setBuffer(t.wbuf, if (t.rank == 0) mine else theirs, 4);
        enc.setBuffer(t.wbuf, if (t.rank == 0) theirs else mine, 5);
        enc.setBuffer(out.b, out.off, 6);
        enc.setBuffer(ssp.b, ssp.off, 7);
        enc.dispatchThreads(mtl.Size.of(D, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// The head's picks from this rank's vocab columns [lo, lo + n) of `logits` (rows x vocab, bf16) and the peer's: each half's argmax, swapped, merged (the larger value, the lower index on a tie: one Mac's argmax exactly).
    pub fn argmax(t: *Tp2, enc: mtl.ComputeEncoder, logits: anytype, vocab: usize, lo: usize, n: usize, picks: anytype, rows_buf: anytype, rows: usize) void {
        t.xseq += 1;
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
        enc.setBytes(std.mem.asBytes(&seq), 1);
        enc.setBuffer(rows_buf.b, rows_buf.off, 2);
        enc.setBuffer(t.wbuf, sendA(x), 3);
        enc.setBuffer(t.wbuf, RECVA + (x % 2) * PAGE, 4);
        enc.setBuffer(picks.b, picks.off, 5);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }

    /// Post `seq` and wait for its serve: one launch, or two with TF_TP_FUSED=0.
    fn postWait(t: *Tp2, enc: mtl.ComputeEncoder, seq: u32, rows_buf: mtl.Buffer, rows_off: usize) void {
        if (!t.fused) {
            t.encodePost(enc, seq, rows_buf, rows_off);
            t.waitWord(enc, SYNC + HOST * 4, seq);
            return;
        }
        enc.setPipeline(t.post_wait);
        enc.setBuffer(t.wbuf, SYNC, 0);
        enc.setBytes(std.mem.asBytes(&seq), 1);
        enc.setBuffer(rows_buf, rows_off, 2);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }

    /// Where the next reduce's partial goes (fp32, rows x 2560): the host sends it from there.
    pub fn partNext(t: *const Tp2) struct { b: mtl.Buffer, off: usize } {
        return .{ .b = t.wbuf, .off = sendR(t.xseq + 1) };
    }

    /// After a split projection wrote this rank's partial into `part`: post, wait for the host to swap partials with the peer, then `out` (bf16, rows x 2560) = rank 0's partial + rank 1's, rounded once.
    pub fn reduce(t: *Tp2, enc: mtl.ComputeEncoder, out: anytype, rows_buf: anytype, rows: usize) void {
        t.xseq += 1;
        const x = t.xseq;
        const seq = t.queue(.{ .kind = .reduce, .value = x });
        t.postWait(enc, seq, rows_buf.b, rows_buf.off);
        enc.setPipeline(t.sum_pipe);
        enc.setBuffer(t.wbuf, sendR(x), 0);
        enc.setBuffer(t.wbuf, REDUCE + (x % 2) * PART, 1);
        enc.setBuffer(out.b, out.off, 2);
        enc.setBuffer(rows_buf.b, rows_buf.off, 3);
        enc.setBytes(std.mem.asBytes(&t.rank), 4);
        enc.dispatchThreads(mtl.Size.of(rows * D, 1, 1), mtl.Size.of(256, 1, 1));
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
        while (@atomicLoad(u64, t.word64(off), .acquire) < value) std.atomic.spinLoopHint();
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

    /// Each sequence in order, once the GPU posts it: a decode exchange (send this rank's packed slots, wait for the peer's, serve the GPU) or a prefill send (its writes, then its flag).
    fn service(t: *Tp2) void {
        const posted = t.word32(SYNC + GPU * 4);
        const served = t.word32(SYNC + HOST * 4);
        const flag = t.word64(FLAG);
        var seq: u32 = 1;
        while (true) {
            var w = @atomicLoad(u32, posted, .acquire);
            while (w >> 5 < seq or t.queued.load(.acquire) < seq) {
                if (t.stop.load(.acquire)) return;
                std.atomic.spinLoopHint();
                w = @atomicLoad(u32, posted, .acquire);
            }
            const job = t.jobs[seq % JOBS];
            const seen = if (t.stats) std.c.mach_absolute_time() else 0;
            if ((t.local and job.kind != .send) or t.failed.load(.acquire)) { // a failed link drains: the GPU never hangs
                @atomicStore(u32, served, seq, .release);
                seq += 1;
                continue;
            }
            if (t.trace) std.debug.print("TP rank{d} seq {d} {s} writes {d} flag {d} value {d} gave_up {d}\n", .{ t.rank, seq, @tagName(job.kind), job.n, job.flag, job.value, t.gaveUp() });
            switch (job.kind) {
                .exchange => {
                    const x = job.value;
                    t.ep.writeSignalFrom(t.peer, sendX(@intCast(x)), DECODE + (x % 2) * SLOT, @as(usize, w & 31) * D * 4, FLAG, x) catch |err| t.fail(seq, err);
                    if (!t.peerAt(flag, x)) return;
                    @atomicStore(u32, served, seq, .release);
                },
                .reduce => {
                    const x = job.value;
                    t.ep.writeSignalFrom(t.peer, sendR(@intCast(x)), REDUCE + (x % 2) * PART, @as(usize, w & 31) * D * 4, FLAG, x) catch |err| t.fail(seq, err);
                    if (!t.peerAt(flag, x)) return;
                    @atomicStore(u32, served, seq, .release);
                },
                .pick => {
                    const x = job.value;
                    t.ep.writeSignalFrom(t.peer, sendA(@intCast(x)), RECVA + (x % 2) * PAGE, @as(usize, w & 31) * 8, FLAG, x) catch |err| t.fail(seq, err);
                    if (!t.peerAt(flag, x)) return;
                    @atomicStore(u32, served, seq, .release);
                },
                .send => t.sendNow(job.writes[0..job.n], job.flag, job.value) catch |err| t.fail(seq, err),
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
            seq += 1;
        }
    }

    /// Wait for the peer's flag to reach `x`: false when stopping; a failed link or 10 s of silence fails and drains.
    fn peerAt(t: *Tp2, flag: *const u64, x: u64) bool {
        const t0 = std.c.mach_absolute_time();
        while (@atomicLoad(u64, flag, .acquire) < x) {
            if (t.stop.load(.acquire)) return false;
            if (t.failed.load(.acquire)) return true;
            if (std.c.mach_absolute_time() - t0 > 240_000_000) { // 10 s of 24 MHz ticks
                t.fail(@intCast(x), error.PeerSilent);
                return true;
            }
            std.atomic.spinLoopHint();
        }
        return true;
    }

    fn fail(t: *Tp2, seq: u32, err: anyerror) void {
        std.log.err("TP=2 rank {d}: sequence {d} failed: {s}", .{ t.rank, seq, @errorName(err) });
        t.failed.store(true, .release);
    }
};
