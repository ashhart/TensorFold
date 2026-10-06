//! Expert parallel over two Macs (MCDMA): half the routed experts each, a MoE call's (row, slot) outputs swapped, all combined on both.
const std = @import("std");
const mtl = @import("metal");
const fabric = @import("fabric");
const settings = @import("../flashnext/tp_settings.zig");
const control = @import("ep_control.zig");
const Ref = @import("weights.zig").Ref;

pub const Settings = settings.Settings;
pub const readSettings = settings.read;
pub const Identity = control.Identity;
pub const Request = control.Request;

const D = 4096;
const TOPK = 8;
const MAXR = 16;
const MAXP = MAXR * TOPK; // a window's picks (row * TOPK + slot) at most
const ENTRY = D * 2; // one pick's routed output, bf16
const PAGE = 16384;
/// A prompt chunk's rows at most: by rows, each row's fp32 routed sum (16 KB) goes in one exchange (64 MB at 4,096).
pub const PROMPT_ROWS = 4096;
const SLOT = @max(MAXP * ENTRY, PROMPT_ROWS * D * 4); // one exchange's entries at most

// The window (the same on both Macs): the peer's entries and ours by parity (a page before each send), flags, GPU words.
const RECV = 0;
const SEND = RECV + 2 * SLOT;
const FLAGS = SEND + 2 * (PAGE + SLOT);
const FLAG = FLAGS; // u64: the last exchange whose peer entries have landed
const SYNC = FLAGS + PAGE;
const POSTED = SYNC; // u32: the last exchange the GPU has packed (its low 32 bits)
const COUNT = SYNC + 64; // u32 by parity: each exchange's packed entries
const GAVE_UP = SYNC + 128; // u32: GPU waits that gave up (nonzero: the run is invalid)
const CONTROL = SYNC + PAGE; // the control protocol's region (ep_control.zig)
pub const WINDOW = CONTROL + control.BYTES;

fn recvAt(x: u64) usize {
    return RECV + @as(usize, @intCast(x % 2)) * SLOT;
}

fn sendAt(x: u64) usize {
    return SEND + @as(usize, @intCast(x % 2)) * (PAGE + SLOT) + PAGE;
}

/// Whether the GPU's posted word (an exchange's low 32 bits) has reached exchange x, across the 32-bit wrap.
fn reached(posted: u32, x: u64) bool {
    return @as(i32, @bitCast(posted -% @as(u32, @truncate(x)))) >= 0;
}

// The lists one exchange keeps (i32): this Mac's unique experts with local ids and their members, the picks each Mac computes.
const L_IDS = 0;
const L_MEM = L_IDS + MAXP * 4;
const L_COUNT = L_MEM + MAXP * MAXR * 4;
const MINE = L_COUNT + 256;
const THEIRS = MINE + MAXP * 4;
const COUNTS = THEIRS + MAXP * 4; // [mine, theirs]
const LISTS = COUNTS + 256;

const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\constant constexpr int TOPK = 8, MAXR = 16, WORDS = 4096 / 2;
    \\// one threadgroup of 128: the route's unique experts held here ([lo, hi), ascending, local ids, their members),
    \\// and the window's picks this Mac and the peer compute (ascending), with this exchange's packed count
    \\kernel void ep_localize(const device int* PICK [[buffer(0)]], const device int* UIDS [[buffer(1)]],
    \\    const device int* UMEM [[buffer(2)]], const device int* UCOUNT [[buffer(3)]], constant int4& arg [[buffer(4)]],
    \\    device int* LIDS [[buffer(5)]], device int* LMEM [[buffer(6)]], device int* LCOUNT [[buffer(7)]],
    \\    device int* MINE [[buffer(8)]], device int* THEIRS [[buffer(9)]], device int* COUNTS [[buffer(10)]],
    \\    device atomic_uint* CNT [[buffer(11)]], uint t [[thread_position_in_threadgroup]],
    \\    uint lane [[thread_index_in_simdgroup]], uint g [[simdgroup_index_in_threadgroup]]) {
    \\  const int rows = arg.x, lo = arg.y, hi = arg.z, i = int(t);
    \\  const int e = i < UCOUNT[0] ? UIDS[i] : -1;
    \\  const int held = e >= lo && e < hi ? 1 : 0;
    \\  const int pe = i < rows * TOPK ? PICK[i] : -1; // pick i's expert
    \\  const int mine = pe >= lo && pe < hi ? 1 : 0;
    \\  const int theirs = pe >= 0 && mine == 0 ? 1 : 0;
    \\  const int b0 = simd_prefix_exclusive_sum(held), b1 = simd_prefix_exclusive_sum(mine), b2 = simd_prefix_exclusive_sum(theirs);
    \\  threadgroup int part[3][4];
    \\  if (lane == 31) { part[0][g] = b0 + held; part[1][g] = b1 + mine; part[2][g] = b2 + theirs; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  int o0 = 0, o1 = 0, o2 = 0, n0 = 0, n1 = 0, n2 = 0;
    \\  for (uint q = 0; q < 4; q++) {
    \\    if (q < g) { o0 += part[0][q]; o1 += part[1][q]; o2 += part[2][q]; }
    \\    n0 += part[0][q]; n1 += part[1][q]; n2 += part[2][q];
    \\  }
    \\  if (held) { LIDS[o0 + b0] = e - lo; for (int j = 0; j < MAXR; j++) LMEM[(o0 + b0) * MAXR + j] = UMEM[i * MAXR + j]; }
    \\  if (mine) MINE[o1 + b1] = i;
    \\  if (theirs) THEIRS[o2 + b2] = i;
    \\  if (t == 0) { LCOUNT[0] = n0; COUNTS[0] = n1; COUNTS[1] = n2; atomic_store_explicit(CNT, uint(n1), memory_order_relaxed); }
    \\}
    \\// A send slot is reused every second exchange: before exchange x writes it, the peer's exchange x - 1 must have
    \\// landed here. The peer posts x - 1 only after our x - 2 (the slot's last use) reached it, so the link is done reading
    \\// the slot. Thread 0 of each threadgroup waits; a give-up is counted, never silent, and the slot is left alone
    \\// (false: the threadgroup stores nothing, and the post that follows sends nothing).
    \\inline bool ep_slot_free(device atomic_uint* flag, uint x, device atomic_uint* gave_up, uint t, threadgroup uint* ok) {
    \\  if (t == 0) {
    \\    uint polls = 0;
    \\    *ok = 1u;
    \\    while (int(atomic_load_explicit(flag, memory_order_relaxed) - (x - 1u)) < 0) {
    \\      if (++polls > 400000000u) { atomic_fetch_add_explicit(gave_up, 1u, memory_order_relaxed); *ok = 0u; break; }
    \\    }
    \\    if (atomic_load_explicit(gave_up, memory_order_relaxed) != 0u) *ok = 0u;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    \\  return *ok != 0u;
    \\}
    \\// this Mac's picks' outputs (Y [picks, 4096] bf16) packed in MINE order for the peer; a threadgroup a pick
    \\kernel void ep_pack(const device uint* Y [[buffer(0)]], const device int* MINE [[buffer(1)]],
    \\    const device int* COUNTS [[buffer(2)]], device uint* OUT [[buffer(3)]], device atomic_uint* flag [[buffer(4)]],
    \\    constant uint& x [[buffer(5)]], device atomic_uint* gave_up [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    \\    uint3 tpos [[thread_position_in_threadgroup]]) {
    \\  const int i = int(tg.y);
    \\  const uint t = tpos.x;
    \\  threadgroup uint ok;
    \\  if (!ep_slot_free(flag, x, gave_up, t, &ok)) return;
    \\  if (i >= COUNTS[0]) return;
    \\  const device uint* src = Y + size_t(MINE[i]) * WORDS;
    \\  device uint* dst = OUT + size_t(i) * WORDS;
    \\  for (uint j = t; j < uint(WORDS); j += 256) dst[j] = src[j];
    \\}
    \\// by rows: this Mac's routed sum a row (each slot's fp32 partial weighted in slot order, the combine's arithmetic)
    \\// into the send slot, and the entries the host sends (two a row) in the count word
    \\#pragma clang fp contract(off)
    \\inline float ep_mul_add(float acc, float a, float b) { return a * b + acc; }
    \\#pragma clang fp contract(on)
    \\kernel void ep_rcombine(const device float* YP [[buffer(0)]], const device float* WTS [[buffer(1)]],
    \\    constant int& rows [[buffer(2)]], device float* OUT [[buffer(3)]], device atomic_uint* CNT [[buffer(4)]],
    \\    device atomic_uint* flag [[buffer(5)]], constant uint& x [[buffer(6)]], device atomic_uint* gave_up [[buffer(7)]],
    \\    uint gid [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    \\  threadgroup uint ok;
    \\  if (!ep_slot_free(flag, x, gave_up, t, &ok)) return;
    \\  const int r = int(gid) / 4096, d = int(gid) % 4096;
    \\  if (gid == 0) atomic_store_explicit(CNT, uint(rows * 2), memory_order_relaxed);
    \\  if (r >= rows) return;
    \\  const device float* y = YP + size_t(r) * TOPK * 4096 + d;
    \\  float acc = WTS[r * TOPK] * y[0];
    \\  for (int k = 1; k < TOPK; k++) acc = ep_mul_add(acc, WTS[r * TOPK + k], y[size_t(k) * 4096]);
    \\  OUT[size_t(r) * 4096 + d] = acc;
    \\}
    \\// by rows: wait for the peer's sums of exchange x; each row's branch is the two Macs' sums added in rank order,
    \\// rounded, plus the shared expert (the combine's last step)
    \\kernel void ep_rfinal(device atomic_uint* flag [[buffer(0)]], constant uint& x [[buffer(1)]],
    \\    device atomic_uint* gave_up [[buffer(2)]], const device float* MINE [[buffer(3)]], device atomic_uint* PEER [[buffer(4)]],
    \\    constant uint& rank [[buffer(5)]], const device bfloat* YS [[buffer(6)]], device bfloat* OUT [[buffer(7)]],
    \\    constant int& rows [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]], uint3 tpos [[thread_position_in_threadgroup]]) {
    \\  const uint t = tpos.x;
    \\  if (t == 0) {
    \\    uint polls = 0;
    \\    while (int(atomic_load_explicit(flag, memory_order_relaxed) - x) < 0) {
    \\      if (++polls > 400000000u) { atomic_fetch_add_explicit(gave_up, 1u, memory_order_relaxed); break; }
    \\    }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_device);
    \\  const int i = int(tg.x) * 256 + int(t);
    \\  if (i >= rows * 4096) return;
    \\  const float mine = MINE[i], peer = as_type<float>(atomic_load_explicit(&PEER[i], memory_order_relaxed));
    \\  const float total = rank == 0u ? mine + peer : peer + mine;
    \\  OUT[i] = bfloat(total) + YS[i];
    \\}
    \\// exchange x packed: the host sends it once the GPU gets here
    \\kernel void ep_post(device atomic_uint* posted [[buffer(0)]], constant uint& x [[buffer(1)]],
    \\    device atomic_uint* gave_up [[buffer(2)]]) {
    \\  if (atomic_load_explicit(gave_up, memory_order_relaxed) == 0u) atomic_store_explicit(posted, x, memory_order_relaxed);
    \\}
    \\// every threadgroup's first thread waits for the peer's exchange x (its flag, stored after its bytes); then the
    \\// peer's entries (read as atomics: written mid-buffer) into their picks' rows of Y; a give-up is counted, never silent
    \\kernel void ep_unpack(device atomic_uint* flag [[buffer(0)]], constant uint& x [[buffer(1)]],
    \\    device atomic_uint* gave_up [[buffer(2)]], device atomic_uint* IN [[buffer(3)]], const device int* THEIRS [[buffer(4)]],
    \\    const device int* COUNTS [[buffer(5)]], device uint* Y [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    \\    uint3 tpos [[thread_position_in_threadgroup]]) {
    \\  const uint t = tpos.x;
    \\  if (t == 0) {
    \\    uint polls = 0;
    \\    while (int(atomic_load_explicit(flag, memory_order_relaxed) - x) < 0) {
    \\      if (++polls > 400000000u) { atomic_fetch_add_explicit(gave_up, 1u, memory_order_relaxed); break; }
    \\    }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_device);
    \\  const int i = int(tg.y);
    \\  if (i >= COUNTS[1]) return;
    \\  device atomic_uint* src = IN + size_t(i) * WORDS;
    \\  device uint* dst = Y + size_t(THEIRS[i]) * WORDS;
    \\  for (uint j = t; j < uint(WORDS); j += 256) dst[j] = atomic_load_explicit(&src[j], memory_order_relaxed);
    \\}
;

pub const Ep = struct {
    rank: u32,
    peer: u32,
    own: [2]u32,
    link: *fabric.mcdma.Endpoint,
    rd: fabric.rdma.Rdma,
    win: []u8,
    wbuf: mtl.Buffer,
    lists: mtl.Buffer,
    localize_pipe: mtl.Pipeline,
    pack_pipe: mtl.Pipeline,
    post_pipe: mtl.Pipeline,
    unpack_pipe: mtl.Pipeline,
    rcombine_pipe: mtl.Pipeline,
    rfinal_pipe: mtl.Pipeline,
    x: u64 = 0, // exchanges encoded so far, the same count on both Macs (the GPU sees the low 32 bits)
    ctl: control.Control, // identities, requests and stop decisions
    thread: ?std.Thread = null,
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    trace: bool = false, // GLM_EP_TRACE: log every exchange the host sends
    delay_ns: u64 = 0, // GLM_EP_DELAY_US: hold every prompt-chunk send (over 1 MiB) this long: a slow link, for stress runs
    sent: u64 = 0, // exchanges the host has sent
    held_ticks: u64 = 0, // their time from the GPU's post to the send returning
    big_sent: u64 = 0, // prompt-chunk exchanges (over 1 MiB) sent, and their time likewise
    big_held_ticks: u64 = 0,
    st: Stats = .{}, // when the peer's entries land against our post, and how the picks split

    /// Connect to the peer in `s`, refuse it unless it runs the same model with the other half of the experts (`me`).
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, s: Settings, me: Identity) !*Ep {
        if (s.rank > 1 or s.links.len != 1) return error.EpTwoRanksOneLink;
        const link = blk: {
            const lib = try gpa.dupeSentinel(u8, s.library, 0);
            defer gpa.free(lib);
            break :blk try fabric.mcdma.Endpoint.create(gpa, lib, .{ .rank = s.rank, .ranks = 2, .window_bytes = WINDOW, .staging_bytes = 4 << 20, .links = s.links, .timeout_ns = 60 * std.time.ns_per_s, .connect_timeout_ns = 300 * std.time.ns_per_s });
        };
        errdefer link.deinit();
        const rd = link.rdma();
        const win = rd.window(); // zeroed before the link connected: a fast peer's first words may be here already
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const t = try gpa.create(Ep);
        errdefer gpa.destroy(t);
        const wbuf = try device.bufferNoCopy(win.ptr, win.len, opts);
        errdefer wbuf.deinit();
        const lists = try device.buffer(LISTS, opts);
        errdefer lists.deinit();
        const lib_m = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib_m.deinit();
        var pipes: [6]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pp| pp.deinit();
        for ([_][]const u8{ "ep_localize", "ep_pack", "ep_post", "ep_unpack", "ep_rcombine", "ep_rfinal" }) |name| {
            pipes[made] = try mtl.Pipeline.init(device, lib_m, name, false);
            made += 1;
        }
        t.* = .{ .rank = s.rank, .peer = 1 - s.rank, .own = .{ me.own_lo, me.own_hi }, .link = link, .rd = rd, .win = win, .wbuf = wbuf, .lists = lists, .localize_pipe = pipes[0], .pack_pipe = pipes[1], .post_pipe = pipes[2], .unpack_pipe = pipes[3], .rcombine_pipe = pipes[4], .rfinal_pipe = pipes[5], .ctl = undefined };
        t.ctl = try control.Control.init(gpa, rd, CONTROL, &t.failed);
        errdefer t.ctl.deinit(gpa);
        _ = try t.ctl.hello(me); // both Macs up, running the same thing, before the first exchange
        t.trace = std.c.getenv("GLM_EP_TRACE") != null;
        if (std.c.getenv("GLM_EP_DELAY_US")) |v| t.delay_ns = 1000 * (std.fmt.parseInt(u64, std.mem.span(v), 10) catch 0);
        t.thread = try std.Thread.spawn(.{}, service, .{t});
        std.log.info("expert parallel: rank {d} of 2 holds experts {d}-{d}, connected", .{ t.rank, me.own_lo, me.own_hi - 1 });
        return t;
    }

    /// Say goodbye (rank 0), stop the sending thread and close the link (no GPU work may still use the window).
    pub fn deinit(t: *Ep, gpa: std.mem.Allocator) void {
        t.ctl.bye();
        t.rd.flush() catch {};
        t.stop.store(true, .release);
        if (t.thread) |th| th.join();
        if (t.sent > 0) {
            const n: f64 = @floatFromInt(t.sent);
            const late: f64 = @floatFromInt(@max(t.st.late, 1));
            if (t.st.big_n > 0) std.log.info("expert parallel rank {d}: {d} prompt-chunk exchanges, {d:.0} us a send, the peer's sums landed {d:.0} us after our post on average", .{ t.rank, t.st.big_n, @as(f64, @floatFromInt(t.big_held_ticks)) / @as(f64, @floatFromInt(@max(t.big_sent, 1))) / 24.0, @as(f64, @floatFromInt(t.st.big_ticks)) / @as(f64, @floatFromInt(t.st.big_n)) / 24.0 });
            std.log.info("expert parallel rank {d}: {d} exchanges, {d:.1} us a send; the peer's entries landed {d:.1} us after our post on average ({d} times, {d} before it); entries a exchange: ours {d:.2}, theirs {d:.2}, |difference| {d:.2}; {d} GPU waits gave up", .{ t.rank, t.sent, @as(f64, @floatFromInt(t.held_ticks)) / n / 24.0, @as(f64, @floatFromInt(t.st.late_ticks)) / late / 24.0, t.st.late, t.st.early, @as(f64, @floatFromInt(t.st.mine)) / n, @as(f64, @floatFromInt(t.st.theirs)) / n, @as(f64, @floatFromInt(t.st.imbalance)) / n, t.gaveUp() });
        }
        t.ctl.deinit(gpa);
        for ([_]mtl.Pipeline{ t.localize_pipe, t.pack_pipe, t.post_pipe, t.unpack_pipe, t.rcombine_pipe, t.rfinal_pipe }) |pp| pp.deinit();
        t.lists.deinit();
        t.wbuf.deinit();
        t.link.deinit();
        gpa.destroy(t);
    }

    /// GPU waits that gave up (a peer that never answered); nonzero means the replies since are invalid.
    pub fn gaveUp(t: *const Ep) u32 {
        return @atomicLoad(u32, t.word32(GAVE_UP), .acquire);
    }

    fn list(t: *const Ep, off: usize) Ref {
        return .{ .buf = t.lists, .off = off };
    }

    /// The gate/up and down kernels' unique-expert group for this Mac's experts (local ids).
    pub fn group(t: *const Ep) [3]Ref {
        return .{ t.list(L_IDS), t.list(L_MEM), t.list(L_COUNT) };
    }

    /// The next exchange (both Macs count every MoE call the same).
    pub fn begin(t: *Ep) void {
        t.x += 1;
    }

    /// The Python family's route's lists localized (the core route's selection makes them itself).
    pub fn localize(t: *Ep, enc: mtl.ComputeEncoder, pick: Ref, uids: Ref, umem: Ref, ucount: Ref, rows: u32) void {
        enc.setPipeline(t.localize_pipe);
        for ([_]Ref{ pick, uids, umem, ucount }, 0..) |r, i| enc.setBuffer(r.buf, r.off, i);
        enc.setValue([4]i32{ @intCast(rows), @intCast(t.own[0]), @intCast(t.own[1]), 0 }, 4);
        for ([_]usize{ L_IDS, L_MEM, L_COUNT, MINE, THEIRS, COUNTS }, 5..) |off, i| enc.setBuffer(t.lists, off, i);
        enc.setBuffer(t.wbuf, COUNT + 4 * @as(usize, @intCast(t.x % 2)), 11);
        enc.dispatchThreads(mtl.Size.of(MAXP, 1, 1), mtl.Size.of(MAXP, 1, 1));
    }

    /// Where the route's selection puts this exchange's lists and this Mac's count (core/moe_route.zig).
    pub fn outputs(t: *const Ep, pick: Ref, wts: Ref) struct { pick: Ref, wts: Ref, ids: Ref, members: Ref, count: Ref, mine: Ref, theirs: Ref, counts: Ref, word: Ref } {
        return .{ .pick = pick, .wts = wts, .ids = t.list(L_IDS), .members = t.list(L_MEM), .count = t.list(L_COUNT), .mine = t.list(MINE), .theirs = t.list(THEIRS), .counts = t.list(COUNTS), .word = .{ .buf = t.wbuf, .off = COUNT + 4 * @as(usize, @intCast(t.x % 2)) } };
    }

    /// After this Mac's routed outputs are in `ye` [rows * TOPK, D]: pack them and post; the host sends them.
    pub fn send(t: *Ep, enc: mtl.ComputeEncoder, ye: Ref, rows: u32, pack: bool, post: bool) void {
        if (pack) {
            enc.setPipeline(t.pack_pipe);
            enc.setBuffer(ye.buf, ye.off, 0);
            enc.setBuffer(t.lists, MINE, 1);
            enc.setBuffer(t.lists, COUNTS, 2);
            enc.setBuffer(t.wbuf, sendAt(t.x), 3);
            t.slotGuard(enc, 4);
            enc.dispatchGroups(mtl.Size.of(1, rows * TOPK, 1), mtl.Size.of(256, 1, 1));
        }
        if (!post) return;
        enc.setPipeline(t.post_pipe);
        enc.setBuffer(t.wbuf, POSTED, 0);
        enc.setValue(@as(u32, @truncate(t.x)), 1);
        enc.setBuffer(t.wbuf, GAVE_UP, 2); // after a give-up nothing more is sent: the run is already invalid
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }

    /// By rows: this Mac's routed sums (`yp` partials weighted by `wts`) into the send slot, then the post (a window or a prompt chunk).
    pub fn sendRows(t: *Ep, enc: mtl.ComputeEncoder, yp: Ref, wts: Ref, rows: u32) void {
        std.debug.assert(rows <= PROMPT_ROWS);
        enc.setPipeline(t.rcombine_pipe);
        enc.setBuffer(yp.buf, yp.off, 0);
        enc.setBuffer(wts.buf, wts.off, 1);
        enc.setValue(@as(i32, @intCast(rows)), 2);
        enc.setBuffer(t.wbuf, sendAt(t.x), 3);
        enc.setBuffer(t.wbuf, COUNT + 4 * @as(usize, @intCast(t.x % 2)), 4);
        t.slotGuard(enc, 5);
        enc.dispatchThreads(mtl.Size.of(rows * 4096, 1, 1), mtl.Size.of(256, 1, 1));
        t.send(enc, yp, rows, false, true);
    }

    /// The send slot guard's operands (ep_slot_free) from buffer index `first`: the flag, this exchange, the give-up count.
    fn slotGuard(t: *const Ep, enc: mtl.ComputeEncoder, first: usize) void {
        enc.setBuffer(t.wbuf, FLAG, first);
        enc.setValue(@as(u32, @truncate(t.x)), first + 1);
        enc.setBuffer(t.wbuf, GAVE_UP, first + 2);
    }

    /// By rows: once the peer's sums land, each row's branch: both sums in rank order, rounded, plus the shared `ys`.
    pub fn receiveRows(t: *Ep, enc: mtl.ComputeEncoder, ys: Ref, out: Ref, rows: u32) void {
        enc.setPipeline(t.rfinal_pipe);
        enc.setBuffer(t.wbuf, FLAG, 0);
        enc.setValue(@as(u32, @truncate(t.x)), 1);
        enc.setBuffer(t.wbuf, GAVE_UP, 2);
        enc.setBuffer(t.wbuf, sendAt(t.x), 3);
        enc.setBuffer(t.wbuf, recvAt(t.x), 4);
        enc.setValue(t.rank, 5);
        enc.setBuffer(ys.buf, ys.off, 6);
        enc.setBuffer(out.buf, out.off, 7);
        enc.setValue(@as(i32, @intCast(rows)), 8);
        enc.dispatchGroups(mtl.Size.of(rows * 16, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// Wait for the peer's outputs of this exchange and put them in their picks' rows of `ye`.
    pub fn receive(t: *Ep, enc: mtl.ComputeEncoder, ye: Ref, rows: u32) void {
        enc.setPipeline(t.unpack_pipe);
        enc.setBuffer(t.wbuf, FLAG, 0);
        enc.setValue(@as(u32, @truncate(t.x)), 1);
        enc.setBuffer(t.wbuf, GAVE_UP, 2);
        enc.setBuffer(t.wbuf, recvAt(t.x), 3);
        enc.setBuffer(t.lists, THEIRS, 4);
        enc.setBuffer(t.lists, COUNTS, 5);
        enc.setBuffer(ye.buf, ye.off, 6);
        enc.dispatchGroups(mtl.Size.of(1, rows * TOPK, 1), mtl.Size.of(256, 1, 1));
    }

    fn word32(t: *const Ep, off: usize) *u32 {
        return @ptrCast(@alignCast(t.win.ptr + off));
    }

    fn word64(t: *const Ep, off: usize) *u64 {
        return @ptrCast(@alignCast(t.win.ptr + off));
    }

    /// Each posted exchange's entries to the peer, then its flag; 10 s without an answer fails the link and lands every flag.
    fn service(t: *Ep) void {
        const posted = t.word32(POSTED);
        const flag = t.word64(FLAG);
        var x: u64 = 1;
        var waiting = false; // an exchange sent whose peer entries have not landed yet
        var want: u64 = 0;
        var want_at: u64 = 0;
        while (true) {
            var spins: usize = 0;
            while (!reached(@atomicLoad(u32, posted, .acquire), x)) {
                t.st.watch(@atomicLoad(u64, flag, .acquire), x - 1);
                if (t.stop.load(.acquire)) return;
                if (waiting and @atomicLoad(u64, flag, .acquire) >= want) waiting = false;
                if (waiting and !t.failed.load(.acquire) and std.c.mach_absolute_time() - want_at > 240_000_000) {
                    t.fail(want, error.PeerSilent);
                    t.land(want);
                    waiting = false;
                }
                spins += 1;
                if (spins > 1_000_000) { // idle: back off
                    const ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000 };
                    _ = std.c.nanosleep(&ts, null);
                } else std.atomic.spinLoopHint();
            }
            const seen = std.c.mach_absolute_time();
            const n: usize = @atomicLoad(u32, t.word32(COUNT + 4 * @as(usize, @intCast(x % 2))), .acquire);
            const theirs: u32 = @bitCast(@as([*]const i32, @ptrCast(@alignCast(t.lists.contents() + COUNTS)))[1]);
            t.st.posted(x, seen, @intCast(n), theirs);
            t.st.watch(@atomicLoad(u64, flag, .acquire), x);
            if (t.trace) std.debug.print("EP rank{d} exchange {d}: {d} entries\n", .{ t.rank, x, n });
            if (t.delay_ns > 0 and n * ENTRY > 1 << 20) {
                const ts: std.c.timespec = .{ .sec = @intCast(t.delay_ns / std.time.ns_per_s), .nsec = @intCast(t.delay_ns % std.time.ns_per_s) };
                _ = std.c.nanosleep(&ts, null);
            }
            if (t.failed.load(.acquire) or n * ENTRY > SLOT) {
                if (n * ENTRY > SLOT) t.fail(x, error.BadCount);
                t.land(x);
            } else t.link.writeSignalFrom(t.peer, sendAt(x), recvAt(x), n * ENTRY, FLAG, x) catch |err| {
                t.fail(x, err);
                t.land(x);
            };
            want = x;
            want_at = std.c.mach_absolute_time();
            waiting = true;
            t.sent += 1;
            t.held_ticks += want_at - seen;
            if (n * ENTRY > 1 << 20) {
                t.big_sent += 1;
                t.big_held_ticks += want_at - seen;
            }
            x += 1;
        }
    }

    /// The flag the GPU waits on, set here as if the peer's message had landed (its bytes stale).
    fn land(t: *Ep, x: u64) void {
        const flag = t.word64(FLAG);
        if (@atomicLoad(u64, flag, .acquire) < x) @atomicStore(u64, flag, x, .release);
    }

    /// The host's view of each exchange: when the GPU posted it and when the peer's entries landed (64 in flight).
    const Stats = struct {
        post_t: [64]u64 = @splat(0),
        flag_t: [64]u64 = @splat(0),
        flag_seen: u64 = 0,
        done: u64 = 0,
        late: u64 = 0,
        late_ticks: u64 = 0,
        early: u64 = 0,
        mine: u64 = 0,
        theirs: u64 = 0,
        imbalance: u64 = 0,
        big: [64]bool = @splat(false), // a prompt chunk's exchange (over 1 MiB)
        big_n: u64 = 0,
        big_ticks: u64 = 0, // their landing after our post (0 when the peer's came first)

        fn posted(s: *Stats, x: u64, at: u64, mine: u32, theirs: u32) void {
            s.post_t[x % 64] = at;
            s.big[x % 64] = @as(usize, mine) * ENTRY > 1 << 20;
            s.mine += mine;
            s.theirs += theirs;
            s.imbalance += if (mine > theirs) mine - theirs else theirs - mine;
        }

        /// The flag's new value seen now; every exchange up to `upto` whose post and landing are both known is counted.
        fn watch(s: *Stats, flag: u64, upto: u64) void {
            if (flag > s.flag_seen) {
                const now = std.c.mach_absolute_time();
                var v = s.flag_seen + 1;
                while (v <= flag) : (v += 1) s.flag_t[v % 64] = now;
                s.flag_seen = flag;
            }
            while (s.done < @min(upto, s.flag_seen)) {
                s.done += 1;
                const p = s.post_t[s.done % 64];
                const f = s.flag_t[s.done % 64];
                if (f >= p) {
                    s.late += 1;
                    s.late_ticks += f - p;
                } else s.early += 1;
                if (s.big[s.done % 64]) {
                    s.big_n += 1;
                    s.big_ticks += if (f >= p) f - p else 0;
                }
            }
        }
    };

    fn fail(t: *Ep, x: u64, err: anyerror) void {
        std.log.err("expert parallel rank {d}: exchange {d} failed: {s}", .{ t.rank, x, @errorName(err) });
        t.failed.store(true, .release);
    }
};

test "the window's regions tile without overlap, slots by parity, a page of room before each send" {
    try std.testing.expectEqual(@as(usize, 0), WINDOW % fabric.mcdma.alignment);
    try std.testing.expect(recvAt(1) + SLOT <= SEND and recvAt(2) == RECV);
    try std.testing.expect(sendAt(1) - PAGE >= sendAt(2) + SLOT and sendAt(2) == SEND + PAGE);
    try std.testing.expect(sendAt(1) + SLOT <= FLAGS and FLAG + 8 <= SYNC);
    try std.testing.expect(GAVE_UP + 4 <= CONTROL and COUNT + 8 <= GAVE_UP and CONTROL % PAGE == 0);
    try std.testing.expectEqual(@as(usize, 4096), D);
    try std.testing.expect(COUNTS + 8 <= LISTS);
}

test "the posted word reaches an exchange across the 32-bit wrap" {
    try std.testing.expect(reached(5, 5) and reached(6, 5) and !reached(4, 5));
    const top: u64 = 0xffff_fffe;
    try std.testing.expect(reached(0xffff_fffe, top) and !reached(0xffff_fffd, top));
    try std.testing.expect(reached(0, top + 2) and reached(1, top + 2) and !reached(0xffff_ffff, top + 2)); // 2^32 posts as 0
    try std.testing.expect(reached(3, (1 << 33) + 3) and !reached(2, (1 << 33) + 3));
}
