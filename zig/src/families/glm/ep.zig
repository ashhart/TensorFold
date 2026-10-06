//! Expert parallel over two Macs (MCDMA): each holds half of every MoE layer's routed experts. A MoE call's own (row, slot)
//! outputs are packed and sent while the GPU runs the shared expert; the peer's land in their rows; both combine all.
//! Served, rank 0 hands rank 1 each request and its decision at every step both take (a stop, a cancel).
const std = @import("std");
const mtl = @import("metal");
const fabric = @import("fabric");
const settings = @import("../flashnext/tp_settings.zig");
const Ref = @import("weights.zig").Ref;

pub const Settings = settings.Settings;
pub const readSettings = settings.read;

const D = 4096;
const TOPK = 8;
const MAXR = 16;
const MAXP = MAXR * TOPK; // a window's picks (row * TOPK + slot) at most
const ENTRY = D * 2; // one pick's routed output, bf16
const PAGE = 16384;
const SLOT = MAXP * ENTRY; // one exchange's entries at most

// The window, the same on both Macs: the peer's entries (two slots by parity), this Mac's (a page of room before each
// for the library's head), the flags the peer sets, the words the GPU posts.
const RECV = 0;
const SEND = RECV + 2 * SLOT;
const FLAGS = SEND + 2 * (PAGE + SLOT);
const FLAG = FLAGS; // u64: the last exchange whose peer entries have landed
const HELLO = FLAGS + 8;
const CTRL = FLAGS + 16; // u64 by parity: rank 0's decision at each step both Macs take, (step << 1) | quit
const REQ_FLAG = FLAGS + 32; // u64: the last request rank 0 has written
const SYNC = FLAGS + PAGE;
const POSTED = SYNC; // u32: the last exchange the GPU has packed
const COUNT = SYNC + 64; // u32 by parity: each exchange's packed entries
const GAVE_UP = SYNC + 128; // u32: GPU waits that gave up (nonzero: the run is invalid)
const REQ = SYNC + PAGE; // served: rank 0's request for rank 1, a 64-byte head and then the prompt's tokens
pub const REQ_TOKENS = 262144;
const MAX_EOS = 11;
pub const WINDOW = REQ + std.mem.alignForward(usize, 64 + 4 * REQ_TOKENS, PAGE);

fn recvAt(x: u32) usize {
    return RECV + (x % 2) * SLOT;
}

fn sendAt(x: u32) usize {
    return SEND + (x % 2) * (PAGE + SLOT) + PAGE;
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
    \\// this Mac's picks' outputs (Y [picks, 4096] bf16) packed in MINE order for the peer; a threadgroup a pick
    \\kernel void ep_pack(const device uint* Y [[buffer(0)]], const device int* MINE [[buffer(1)]],
    \\    const device int* COUNTS [[buffer(2)]], device uint* OUT [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
    \\    uint3 tpos [[thread_position_in_threadgroup]]) {
    \\  const int i = int(tg.y);
    \\  const uint t = tpos.x;
    \\  if (i >= COUNTS[0]) return;
    \\  const device uint* src = Y + size_t(MINE[i]) * WORDS;
    \\  device uint* dst = OUT + size_t(i) * WORDS;
    \\  for (uint j = t; j < uint(WORDS); j += 256) dst[j] = src[j];
    \\}
    \\// exchange x packed: the host sends it once the GPU gets here
    \\kernel void ep_post(device atomic_uint* posted [[buffer(0)]], constant uint& x [[buffer(1)]]) {
    \\  atomic_store_explicit(posted, x, memory_order_relaxed);
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
    x: u32 = 0, // exchanges encoded so far, the same count on both Macs
    ctrl: u64 = 0, // steps agreed so far, the same count on both Macs
    req: u64 = 0, // requests handed over so far
    eos: [MAX_EOS]u32 = undefined, // rank 1: the request's end tokens, copied out of the window
    thread: ?std.Thread = null,
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    quitting: std.atomic.Value(bool) = .init(false), // served rank 1: end the wait for rank 0's next request

    /// Connect to the peer in `s` (this Mac holds routed experts `own`), then start the host's sending thread.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, s: Settings, own: [2]u32) !*Ep {
        if (s.rank > 1 or s.links.len != 1) return error.EpTwoRanksOneLink;
        const lib = try gpa.dupeSentinel(u8, s.library, 0);
        const link = try fabric.mcdma.Endpoint.create(gpa, lib, .{ .rank = s.rank, .ranks = 2, .window_bytes = WINDOW, .staging_bytes = 4 << 20, .links = s.links, .timeout_ns = 60 * std.time.ns_per_s, .connect_timeout_ns = 300 * std.time.ns_per_s });
        errdefer link.deinit();
        const rd = link.rdma();
        const win = rd.window(); // zeroed before the link connected: a fast peer's first words may be here already
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const lib_m = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        const t = try gpa.create(Ep);
        errdefer gpa.destroy(t);
        t.* = .{
            .rank = s.rank,
            .peer = 1 - s.rank,
            .own = own,
            .link = link,
            .rd = rd,
            .win = win,
            .wbuf = try device.bufferNoCopy(win.ptr, win.len, opts),
            .lists = try device.buffer(LISTS, opts),
            .localize_pipe = try mtl.Pipeline.init(device, lib_m, "ep_localize", false),
            .pack_pipe = try mtl.Pipeline.init(device, lib_m, "ep_pack", false),
            .post_pipe = try mtl.Pipeline.init(device, lib_m, "ep_post", false),
            .unpack_pipe = try mtl.Pipeline.init(device, lib_m, "ep_unpack", false),
        };
        // both Macs up before the first exchange: a word each way
        try rd.signal(t.peer, HELLO, 1);
        const t0 = std.c.mach_absolute_time();
        while (@atomicLoad(u64, t.word64(HELLO), .acquire) < 1) {
            if (std.c.mach_absolute_time() - t0 > 7_200_000_000) return error.EpPeerSilent; // 300 s
            std.atomic.spinLoopHint();
        }
        t.thread = try std.Thread.spawn(.{}, service, .{t});
        std.log.info("expert parallel: rank {d} of 2 holds experts {d}-{d}, connected", .{ t.rank, own[0], own[1] - 1 });
        return t;
    }

    /// Stop the sending thread and close the link (no GPU work may still use the window).
    pub fn deinit(t: *Ep, gpa: std.mem.Allocator) void {
        t.rd.flush() catch {};
        t.stop.store(true, .release);
        if (t.thread) |th| th.join();
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

    /// The next exchange: from the route's picks and unique experts, this Mac's experts with local ids and the picks each Mac computes.
    pub fn localize(t: *Ep, enc: mtl.ComputeEncoder, pick: Ref, uids: Ref, umem: Ref, ucount: Ref, rows: u32) void {
        t.x += 1;
        enc.setPipeline(t.localize_pipe);
        for ([_]Ref{ pick, uids, umem, ucount }, 0..) |r, i| enc.setBuffer(r.buf, r.off, i);
        enc.setValue([4]i32{ @intCast(rows), @intCast(t.own[0]), @intCast(t.own[1]), 0 }, 4);
        for ([_]usize{ L_IDS, L_MEM, L_COUNT, MINE, THEIRS, COUNTS }, 5..) |off, i| enc.setBuffer(t.lists, off, i);
        enc.setBuffer(t.wbuf, COUNT + 4 * (t.x % 2), 11);
        enc.dispatchThreads(mtl.Size.of(MAXP, 1, 1), mtl.Size.of(MAXP, 1, 1));
    }

    /// After this Mac's routed outputs are in `ye` [rows * TOPK, D]: pack them and post; the host sends them.
    pub fn send(t: *Ep, enc: mtl.ComputeEncoder, ye: Ref, rows: u32) void {
        enc.setPipeline(t.pack_pipe);
        enc.setBuffer(ye.buf, ye.off, 0);
        enc.setBuffer(t.lists, MINE, 1);
        enc.setBuffer(t.lists, COUNTS, 2);
        enc.setBuffer(t.wbuf, sendAt(t.x), 3);
        enc.dispatchGroups(mtl.Size.of(1, rows * TOPK, 1), mtl.Size.of(256, 1, 1));
        enc.setPipeline(t.post_pipe);
        enc.setBuffer(t.wbuf, POSTED, 0);
        enc.setValue(t.x, 1);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }

    /// Wait for the peer's outputs of this exchange and put them in their picks' rows of `ye`.
    pub fn receive(t: *Ep, enc: mtl.ComputeEncoder, ye: Ref, rows: u32) void {
        enc.setPipeline(t.unpack_pipe);
        enc.setBuffer(t.wbuf, FLAG, 0);
        enc.setValue(t.x, 1);
        enc.setBuffer(t.wbuf, GAVE_UP, 2);
        enc.setBuffer(t.wbuf, recvAt(t.x), 3);
        enc.setBuffer(t.lists, THEIRS, 4);
        enc.setBuffer(t.lists, COUNTS, 5);
        enc.setBuffer(ye.buf, ye.off, 6);
        enc.dispatchGroups(mtl.Size.of(1, rows * TOPK, 1), mtl.Size.of(256, 1, 1));
    }

    /// A step both Macs take in the same order (a prompt window, the first token, a round): rank 0's decision to stop
    /// there reaches rank 1, so both end on the same step. Rank 1 waits for it. Rank 0 runs at most two steps ahead
    /// (a step's exchanges need rank 1; only a new request's first step follows a step without any), so two words.
    pub fn agree(t: *Ep, quit: bool) !bool {
        t.ctrl += 1;
        const word = CTRL + 8 * (t.ctrl % 2);
        if (t.rank == 0) {
            try t.rd.signal(t.peer, word, (t.ctrl << 1) | @intFromBool(quit));
            return quit;
        }
        const t0 = std.c.mach_absolute_time();
        while (true) {
            const v = @atomicLoad(u64, t.word64(word), .acquire);
            if (v >> 1 == t.ctrl) return v & 1 != 0;
            if (v >> 1 > t.ctrl) return error.EpOutOfStep;
            if (std.c.mach_absolute_time() - t0 > 240_000_000) return error.EpPeerSilent; // 10 s
            std.atomic.spinLoopHint();
        }
    }

    pub const Request = struct { max_tokens: usize, depth: usize, eos: []const u32, prompt: []const u32 };

    /// Served, rank 0: hand rank 1 the next request in one message.
    pub fn sendRequest(t: *Ep, r: Request) !void {
        if (r.prompt.len > REQ_TOKENS or r.eos.len > MAX_EOS) return error.EpRequestTooLarge;
        var head: [16]u32 = @splat(0);
        head[0] = @intCast(@min(r.max_tokens, std.math.maxInt(u32)));
        head[1] = @intCast(r.depth);
        head[2] = @intCast(r.prompt.len);
        head[3] = @intCast(r.eos.len);
        @memcpy(head[4..][0..r.eos.len], r.eos);
        t.req += 1;
        try t.rd.write2Signal(t.peer, REQ, std.mem.sliceAsBytes(&head), std.mem.sliceAsBytes(r.prompt), REQ_FLAG, t.req);
    }

    /// Served, rank 1: rank 0's next request (its tokens in the window until the next one), or null once `quitting`.
    pub fn waitRequest(t: *Ep) ?Request {
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
        const head: [*]const u32 = @ptrCast(@alignCast(t.win.ptr + REQ));
        const tokens: [*]const u32 = @ptrCast(@alignCast(t.win.ptr + REQ + 64));
        const n_eos = @min(head[3], MAX_EOS);
        @memcpy(t.eos[0..n_eos], head[4..][0..n_eos]); // the next request may land before this one's last step
        return .{ .max_tokens = head[0], .depth = head[1], .eos = t.eos[0..n_eos], .prompt = tokens[0..@min(head[2], REQ_TOKENS)] };
    }

    fn word32(t: *const Ep, off: usize) *u32 {
        return @ptrCast(@alignCast(t.win.ptr + off));
    }

    fn word64(t: *const Ep, off: usize) *u64 {
        return @ptrCast(@alignCast(t.win.ptr + off));
    }

    /// Each exchange in order, once the GPU posts it: its packed entries to the peer's slot, then the peer's flag. Ten
    /// seconds without the peer's answer to the last one fails the link; a failed link lands every flag (the GPU never hangs).
    fn service(t: *Ep) void {
        const posted = t.word32(POSTED);
        const flag = t.word64(FLAG);
        var x: u32 = 1;
        var want: u64 = 0;
        var want_at: u64 = 0;
        while (true) {
            var spins: usize = 0;
            while (@as(i32, @bitCast(@atomicLoad(u32, posted, .acquire) -% x)) < 0) {
                if (t.stop.load(.acquire)) return;
                if (want > 0 and @atomicLoad(u64, flag, .acquire) < want and !t.failed.load(.acquire) and std.c.mach_absolute_time() - want_at > 240_000_000) {
                    t.fail(x, error.PeerSilent);
                    t.land(want);
                }
                spins += 1;
                if (spins > 1_000_000) { // idle: back off
                    const ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000 };
                    _ = std.c.nanosleep(&ts, null);
                } else std.atomic.spinLoopHint();
            }
            const n: usize = @atomicLoad(u32, t.word32(COUNT + 4 * (x % 2)), .acquire);
            if (t.failed.load(.acquire) or n > MAXP) {
                if (n > MAXP) t.fail(x, error.BadCount);
                t.land(x);
            } else t.link.writeSignalFrom(t.peer, sendAt(x), recvAt(x), n * ENTRY, FLAG, x) catch |err| {
                t.fail(x, err);
                t.land(x);
            };
            want = x;
            want_at = std.c.mach_absolute_time();
            x +%= 1;
        }
    }

    /// The flag the GPU waits on, set here as if the peer's message had landed (its bytes stale).
    fn land(t: *Ep, x: u64) void {
        const flag = t.word64(FLAG);
        if (@atomicLoad(u64, flag, .acquire) < x) @atomicStore(u64, flag, x, .release);
    }

    fn fail(t: *Ep, x: u32, err: anyerror) void {
        std.log.err("expert parallel rank {d}: exchange {d} failed: {s}", .{ t.rank, x, @errorName(err) });
        t.failed.store(true, .release);
    }
};

test "the window's regions tile without overlap, slots by parity, a page of room before each send" {
    try std.testing.expectEqual(@as(usize, 0), WINDOW % fabric.mcdma.alignment);
    try std.testing.expect(recvAt(1) + SLOT <= SEND and recvAt(2) == RECV);
    try std.testing.expect(sendAt(1) - PAGE >= sendAt(2) + SLOT and sendAt(2) == SEND + PAGE);
    try std.testing.expect(sendAt(1) + SLOT <= FLAGS and HELLO + 8 <= SYNC);
    try std.testing.expect(GAVE_UP + 4 <= REQ and COUNT + 8 <= GAVE_UP and REQ_FLAG + 8 <= SYNC and CTRL + 16 <= REQ_FLAG);
    try std.testing.expect(REQ + 64 + 4 * REQ_TOKENS <= WINDOW and 4 + MAX_EOS <= 16);
    try std.testing.expectEqual(@as(usize, 4096), D);
    try std.testing.expect(COUNTS + 8 <= LISTS);
}
