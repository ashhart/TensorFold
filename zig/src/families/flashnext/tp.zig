//! Speed-up mode for Flash Next: two Macs with the whole model each, half the work each, over MCDMA.
//! Decode: each rank computes the routed experts it owns (rank 0 the first 256 and the shared expert, rank 1 the rest);
//! after each target layer's experts the GPU packs its own slots in row and slot order and posts a sequence; the host
//! sends them to the peer and serves the sequence once the peer's have landed; the GPU unpacks them. Both ranks route
//! the same, so each knows the other's slots, and every slot holds one rank's value: the bits equal one Mac's.
//! Prefill: a chunk's rows split across the two Macs (replay.zig `chunkPair`); the host sends each layer's handoff into
//! the peer's slot for that layer and signals the layer's flag, which the peer's GPU waits for.
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

/// The window, the same on both ranks: decode slots (alternating by sequence parity), then a slot a layer for prefill
/// handoffs, the n-gram tail, the last row's streams and first token, the MTP head's keys; flags; the sync words.
const SLOT = MAXR * 11 * D * 2; // one layer's expert outputs at the widest window (bf16): 55 pages
pub const DN_SLOT = pages(CS_ROW + SO_ROW); // a DeltaNet layer: conv state row, then recurrent state row
pub const ATT_SLOT = 4 * PMAX * 512 + PMAX * 256; // an attention layer: keys of both heads, values of both, raw keys
pub const KV_HEAD = PMAX * 512; // one head's keys (or values) at the most rows, inside ATT_SLOT
const DECODE = 0;
const PREFILL = 2 * SLOT;
pub const TAIL = PREFILL + 36 * DN_SLOT + 12 * ATT_SLOT; // the n-gram tail: PLE_TAIL rows of WIDE
pub const LAST = TAIL + pages(9 * WIDE * 2); // the last prompt row's streams, then its first token
pub const MTP = LAST + pages(WIDE * 2 + 16); // the MTP head's keys, an attention slot
const FLAGS = MTP + ATT_SLOT;
const SYNC = FLAGS + PAGE;
const WINDOW = SYNC + PAGE;
const FLAG = FLAGS; // decode: the last sequence whose outputs have landed
const HELLO = FLAGS + 8;
pub const LAYER_FLAG = FLAGS + 64; // prefill: a word a layer, then the n-gram tail, back, and MTP words
pub const TAIL_FLAG = LAYER_FLAG + LAYERS * 8;
pub const BACK_FLAG = TAIL_FLAG + 8;
pub const MTP_FLAG = BACK_FLAG + 8;
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

/// Which rank computes slot k of a row whose routed experts are `pick` (k == 10: the shared expert, rank 0).
fn owner(pick: []const u32, k: usize) u32 {
    return if (k == 10) 0 else @intFromBool(pick[k] >= 256);
}

const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// this sequence and the window's rows (seq << 5 | rows), after every earlier dispatch of the serial encoder
    \\kernel void tp_post(device atomic_uint* sync [[buffer(0)]], constant uint& seq [[buffer(1)]],
    \\                    const constant int* rows [[buffer(2)]]) {
    \\  atomic_store_explicit(&sync[1024], (seq << 5) | uint(rows[0]), memory_order_relaxed);
    \\}
    \\// one thread polls until the 32-bit word reaches `value`; a give-up is counted, never silent
    \\kernel void tp_wait(device atomic_uint* word [[buffer(0)]], constant uint& value [[buffer(1)]],
    \\                    device atomic_uint* sync [[buffer(2)]]) {
    \\  uint polls = 0;
    \\  while (int(atomic_load_explicit(word, memory_order_relaxed) - value) < 0) {
    \\    if (++polls > 400000000u) { atomic_fetch_add_explicit(&sync[2048], 1u, memory_order_relaxed); return; }
    \\  }
    \\}
    \\// the slot's owner: the shared expert (slot 10) is rank 0's, routed experts below 256 rank 0's, the rest rank 1's
    \\inline uint tp_owner(const device uint* pick, uint p) {
    \\  const uint r = p / 11u, k = p % 11u;
    \\  return k == 10u ? 0u : (pick[r * 10u + k] >= 256u ? 1u : 0u);
    \\}
    \\// slot p's place among `rank`'s slots, in row and slot order
    \\inline uint tp_index(const device uint* pick, uint p, uint rank) {
    \\  uint n = 0;
    \\  for (uint q = 0; q < p; q++) n += tp_owner(pick, q) == rank ? 1u : 0u;
    \\  return n;
    \\}
    \\// threadgroup per slot: this rank's slots, packed in order
    \\kernel void tp_pack(const device bfloat* y [[buffer(0)]], const device uint* pick [[buffer(1)]],
    \\                    device bfloat* packed [[buffer(2)]], const constant int* rows [[buffer(3)]],
    \\                    constant uint& rank [[buffer(4)]], uint t [[thread_position_in_threadgroup]],
    \\                    uint p [[threadgroup_position_in_grid]]) {
    \\  if (p >= uint(rows[0]) * 11u || tp_owner(pick, p) != rank) return;
    \\  const uint at = tp_index(pick, p, rank);
    \\  for (uint d = t; d < 2560u; d += 256u) packed[at * 2560u + d] = y[p * 2560u + d];
    \\}
    \\// threadgroup per slot: the peer's packed slots into theirs
    \\kernel void tp_unpack(device bfloat* y [[buffer(0)]], const device uint* pick [[buffer(1)]],
    \\                      const device bfloat* packed [[buffer(2)]], const constant int* rows [[buffer(3)]],
    \\                      constant uint& peer [[buffer(4)]], uint t [[thread_position_in_threadgroup]],
    \\                      uint p [[threadgroup_position_in_grid]]) {
    \\  if (p >= uint(rows[0]) * 11u || tp_owner(pick, p) != peer) return;
    \\  const uint at = tp_index(pick, p, peer);
    \\  for (uint d = t; d < 2560u; d += 256u) y[p * 2560u + d] = packed[at * 2560u + d];
    \\}
;

/// Bytes the host copies into the peer's window once the GPU has posted the job's sequence.
pub const Write = struct { src: [*]const u8, len: usize, dst: usize };

/// A posted sequence's work: a decode exchange, or prefill writes and their flag.
const Job = struct {
    kind: enum { exchange, send } = .send,
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
    pack_pipe: mtl.Pipeline,
    unpack_pipe: mtl.Pipeline,
    pack_buf: mtl.Buffer, // this rank's slots, packed: what the host sends
    pick: [*]const u32, // the routed experts of each row (the model's pick), so the host knows the packed size
    one: mtl.Buffer, // a rows word holding 1, for posts that carry no rows
    seq: u32 = 0, // the last sequence this Mac's GPU posts (its own count: prefill sends are rank 0's alone)
    xseq: u32 = 0, // decode exchanges so far, the same count on both Macs: their slots' parity and flag values
    call: u32 = 0, // prefill chunks split so far (both ranks count the same): their flags' values
    jobs: []Job,
    queued: std.atomic.Value(u32) = .init(0), // jobs written: the service reads a job only below this
    thread: ?std.Thread = null,
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    trace: bool = false, // TF_TP_TRACE: log every job the host serves

    /// Connect to the peer named in the settings file and start the host's service thread; `pick` holds each
    /// window row's routed experts.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, settings_path: []const u8, pick: anytype) !*Tp2 {
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
            .pack_pipe = try mtl.Pipeline.init(device, lib_m, "tp_pack", false),
            .unpack_pipe = try mtl.Pipeline.init(device, lib_m, "tp_unpack", false),
            .pack_buf = try device.buffer(SLOT, opts),
            .pick = @ptrCast(@alignCast(pick.b.contents() + pick.off)),
            .one = try device.buffer(16, opts),
            .jobs = try gpa.alloc(Job, JOBS),
        };
        t.one.slice(i32, 1)[0] = 1;
        t.trace = std.c.getenv("TF_TP_TRACE") != null;
        // both ranks up before the first round: a word each way
        try rd.signal(t.peer, HELLO, 1);
        while (@atomicLoad(u64, t.word64(HELLO), .acquire) < 1) std.atomic.spinLoopHint();
        t.thread = try std.Thread.spawn(.{}, service, .{t});
        std.log.info("TP=2 rank {d} connected; experts {any}", .{ t.rank, t.own() });
        return t;
    }

    pub fn deinit(t: *Tp2) void {
        t.stop.store(true, .release);
        if (t.thread) |th| th.join();
        t.ep.deinit();
    }

    /// The experts this rank computes: [lo, hi) and whether the shared expert is its.
    pub fn own(t: *const Tp2) [4]u32 {
        return if (t.rank == 0) .{ 0, 256, 1, 0 } else .{ 256, 512, 0, 0 };
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

    /// After a target layer's experts, on the serial encoder: pack this rank's slots, post the layer's sequence and
    /// rows, wait for the host to serve it, unpack the peer's slots into theirs.
    pub fn exchange(t: *Tp2, enc: mtl.ComputeEncoder, y: anytype, pick: anytype, rows_buf: anytype) void {
        t.xseq += 1;
        const x = t.xseq;
        const seq = t.queue(.{ .kind = .exchange, .value = x });
        enc.setPipeline(t.pack_pipe);
        enc.setBuffer(y.b, y.off, 0);
        enc.setBuffer(pick.b, pick.off, 1);
        enc.setBuffer(t.pack_buf, 0, 2);
        enc.setBuffer(rows_buf.b, rows_buf.off, 3);
        enc.setBytes(std.mem.asBytes(&t.rank), 4);
        enc.dispatchGroups(mtl.Size.of(MAXR * 11, 1, 1), mtl.Size.of(256, 1, 1));
        t.encodePost(enc, seq, rows_buf.b, rows_buf.off);
        t.waitWord(enc, SYNC + HOST * 4, seq);
        enc.setPipeline(t.unpack_pipe);
        enc.setBuffer(y.b, y.off, 0);
        enc.setBuffer(pick.b, pick.off, 1);
        enc.setBuffer(t.wbuf, DECODE + (x % 2) * SLOT, 2);
        enc.setBuffer(rows_buf.b, rows_buf.off, 3);
        enc.setBytes(std.mem.asBytes(&t.peer), 4);
        enc.dispatchGroups(mtl.Size.of(MAXR * 11, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// On the serial encoder, after the work that wrote the sources: once the GPU gets here the host copies `writes`
    /// into the peer's window and stores `value` at its `flag`; the GPU does not wait.
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

    /// Each sequence in order, once the GPU posts it: a decode exchange (send this rank's packed slots, wait for the
    /// peer's, serve the GPU) or a prefill send (its writes, then its flag).
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
            if (t.trace) std.debug.print("TP rank{d} seq {d} {s} writes {d} flag {d} value {d} gave_up {d}\n", .{ t.rank, seq, @tagName(job.kind), job.n, job.flag, job.value, t.gaveUp() });
            switch (job.kind) {
                .exchange => {
                    var mine: usize = 0; // this rank's slots in the window's rows: the packed size
                    for (0..@as(usize, w & 31)) |r| {
                        for (0..11) |k| mine += @intFromBool(owner(t.pick[r * 10 ..][0..10], k) == t.rank);
                    }
                    const src: [*]const u8 = @ptrCast(t.pack_buf.contents());
                    const x = job.value;
                    t.rd.write2Signal(t.peer, DECODE + (x % 2) * SLOT, src[0 .. mine * D * 2], &.{}, FLAG, x) catch |err| return t.fail(seq, err);
                    while (@atomicLoad(u64, flag, .acquire) < x) {
                        if (t.stop.load(.acquire)) return;
                        std.atomic.spinLoopHint();
                    }
                    @atomicStore(u32, served, seq, .release);
                },
                .send => t.sendNow(job.writes[0..job.n], job.flag, job.value) catch |err| return t.fail(seq, err),
            }
            seq += 1;
        }
    }

    fn fail(t: *Tp2, seq: u32, err: anyerror) void {
        std.log.err("TP=2 rank {d}: sequence {d} failed: {s}", .{ t.rank, seq, @errorName(err) });
        t.failed.store(true, .release);
    }
};
