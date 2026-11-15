//! Flash Next paired serving: split work, exchange exact partials, and own the endpoint until GPU aliases are gone.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const fabric = @import("fabric");
const Handoff = @import("tp_handoff.zig").Handoff;
const relay = @import("tp_service.zig");
const host_wait = @import("tp_host_wait.zig");
const tp_seq = @import("tp_seq.zig");
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
/// Both ranks share the layout: decode, prefill, tail, final streams, draft keys, flags and send slots.
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
pub const MARK = REQ + pages(64 + 4 * REQ_TOKENS); // a pair chunk's DeltaNet states, then its n-gram tail
const WINDOW = MARK + 36 * DN_SLOT + pages(9 * WIDE * 2);
fn sendX(slot: u32) usize {
    return SENDX + (slot % 2) * (PAGE + SLOT) + PAGE;
}
fn sendR(slot: u32) usize {
    return SENDR + (slot % 2) * (PAGE + PART) + PAGE;
}
fn sendA(slot: u32) usize {
    return SENDA + (slot % 2) * 2 * PAGE + PAGE;
}
const FLAG = FLAGS; // decode: the last sequence whose outputs have landed
const HELLO = FLAGS + 8;
pub const LAYER_FLAG = FLAGS + 64; // prefill: a word a layer, then the n-gram tail, back, and MTP words
pub const TAIL_FLAG = LAYER_FLAG + LAYERS * 8;
pub const BACK_FLAG = TAIL_FLAG + 8;
pub const MTP_FLAG = BACK_FLAG + 8;
const REQ_FLAG = MTP_FLAG + 8; // served: the last request rank 0 has written
const CTRL_FLAG = REQ_FLAG + 8; // rank 0's step decision: (step << 6) | (next depth << 1) | quit
const ACK_FLAG = CTRL_FLAG + 8; // served: rank 1's answer to a request's resume, (request << 1) | restored
const SUBMIT_FLAG = ACK_FLAG + 16; // both hosts finished encoding through this job before either submits GPU waits
pub const MARK_FLAG = FLAGS + 512; // states landed in MARK; ACK_FLAG + 8 remains the link probe
pub const MARK_READY = MARK_FLAG + 8; // the receiver's MARK is free for that sequence
/// The receiving window's slot for layer i's prefill handoff (attention layers are i % 4 == 3).
pub fn layerSlot(i: usize) usize {
    const att = i / 4; // attention layers before i (3, 7, ... below it)
    return PREFILL + (i - att) * DN_SLOT + att * ATT_SLOT;
}
test "control words are distinct and inside the flag page" {
    var words: [LAYERS + 12]usize = undefined;
    var n: usize = 0;
    for ([_]usize{ FLAG, HELLO, TAIL_FLAG, BACK_FLAG, MTP_FLAG, REQ_FLAG, CTRL_FLAG, ACK_FLAG, ACK_FLAG + 8, SUBMIT_FLAG, MARK_FLAG, MARK_READY }) |w| {
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
const source = @import("tp_source.zig").source;
pub const Write = @import("tp_job.zig").Write;
const Job = @import("tp_job.zig").Job;
const JOBS = 1024;
pub const Tp2 = struct {
    pub const handoff_module = @import("tp_handoff.zig");
    pub const relay_module = relay;
    pub const shader = source;
    pub const QueueJob = Job;
    owned: @import("owned.zig").Owner(mtl) = .{},
    allocator: ?std.mem.Allocator = null,
    benchmark_group: ?[]const u8 = null,
    rank: u32,
    peer: u32,
    ep: *fabric.mcdma.Endpoint,
    rd: fabric.rdma.Rdma,
    win: []u8,
    wbuf: mtl.Buffer,
    handoff: Handoff,
    copy_pipe: mtl.Pipeline,
    partial: mtl.Buffer,
    branch_pipe: mtl.Pipeline,
    argmax_pipe: mtl.Pipeline,
    pick_pipe: mtl.Pipeline,
    merge_pipe: mtl.Pipeline,
    plain_pipe: mtl.Pipeline,
    last_x: tp_seq.Msg = .{ .flag = 0, .slot = 0 }, // the last expert exchange's partials for combine
    last_seq: u64 = 0, // and the sequence it posts
    req: u64 = 0, // served requests so far, the same on both ranks
    ctrl: u64 = 0, // stop decisions so far, the same on both ranks
    one: mtl.Buffer, // a rows word holding 1, for posts that carry no rows
    pick_tmp: mtl.Buffer, // the MTP head's merged draft index
    ids_pipe: mtl.Pipeline,
    seq: u64 = 0, // the last job: each prefill send on rank 0 matches a wait on rank 1
    seqs: tp_seq.Seqs = .{}, // decode messages so far, numbered the same on both Macs: flag values and slots
    call: u32 = 0, // prefill chunks split so far (both ranks count the same): their flags' values
    mark_seq: u64 = 0, // marks exchanged after pair chunks (both ranks count the same)
    jobs: []Job,
    queued: std.atomic.Value(u64) = .init(0), // jobs written: the service reads a job only below this
    submitted: std.atomic.Value(u64) = .init(0), // both hosts are ready to submit GPU work through this job
    thread: ?std.Thread = null,
    monitor: ?std.Thread = null,
    stop: std.atomic.Value(bool) = .init(false),
    quitting: std.atomic.Value(bool) = .init(false), // served rank 1: end the wait for the next request
    failed: std.atomic.Value(bool) = .init(false),
    trace: bool = false, // TF_TP_TRACE: log every job the host serves
    local: bool = false, // TF_TP_LOCAL: serve jobs locally to time the GPU alone; replies are wrong
    stats: bool = false, // TF_TP_STATS: the host's time from a job's post to its serve, every 4096 jobs
    held_ns: u64 = 0,
    held_max: u64 = 0,
    held_kind: [5][2]u64 = @splat(.{ 0, 0 }), // by job kind: total ticks, jobs
    /// Metal aliases the endpoint window; count it once beside staging, tiny buffers and the host queue.
    pub fn startupBytes() usize {
        return WINDOW + (32 << 20) + PART + 48 + @sizeOf(Tp2) + JOBS * @sizeOf(Job);
    }
    /// Connect to the peer named in the settings file and start the host's service thread.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, settings_path: []const u8) !*Tp2 {
        const f = try mtl.MappedFile.open(try gpa.dupeSentinel(u8, settings_path, 0));
        const s = try settings.read(gpa, f.bytes[0..f.size]);
        if (s.rank > 1 or s.links.len != 1) return error.TpTwoRanksOneLink;
        const lib = try gpa.dupeSentinel(u8, s.library, 0);
        const ep = try fabric.mcdma.Endpoint.create(gpa, lib, .{ .rank = s.rank, .ranks = 2, .window_bytes = WINDOW, .staging_bytes = 32 << 20, .links = s.links, .timeout_ns = 60 * std.time.ns_per_s, .connect_timeout_ns = 300 * std.time.ns_per_s });
        errdefer ep.deinit();
        var owned: @import("owned.zig").Owner(mtl) = .{};
        errdefer owned.deinit();
        const rd = ep.rdma();
        const win = rd.window(); // zeroed before connection; a fast peer's first words may already be here
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const lib_m = try owned.keep(gpa, try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx()));
        const t = try gpa.create(Tp2);
        errdefer gpa.destroy(t);
        t.* = .{
            .allocator = gpa,
            .rank = s.rank,
            .peer = 1 - s.rank,
            .ep = ep,
            .rd = rd,
            .win = win,
            .wbuf = try owned.keep(gpa, try device.bufferNoCopy(win.ptr, win.len, opts)),
            .handoff = undefined,
            .copy_pipe = try owned.keep(gpa, try mtl.Pipeline.init(device, lib_m, "tp_copy", false)),
            .partial = try owned.keep(gpa, try device.buffer(PART, opts)),
            .branch_pipe = try owned.keep(gpa, try mtl.Pipeline.init(device, lib_m, "tp_branch", false)),
            .argmax_pipe = try owned.keep(gpa, try mtl.Pipeline.init(device, lib_m, "tp_argmax_part", false)),
            .pick_pipe = try owned.keep(gpa, try mtl.Pipeline.init(device, lib_m, "tp_pick", false)),
            .merge_pipe = try owned.keep(gpa, try mtl.Pipeline.init(device, lib_m, "tp_merge", false)),
            .plain_pipe = try owned.keep(gpa, try mtl.Pipeline.init(device, lib_m, "tp_plain", false)),
            .one = try owned.keep(gpa, try device.buffer(16, opts)),
            .pick_tmp = try owned.keep(gpa, try device.buffer(16, opts)),
            .ids_pipe = try owned.keep(gpa, try mtl.Pipeline.init(device, lib_m, "tp_ids", false)),
            .jobs = try gpa.alloc(Job, JOBS),
        };
        errdefer gpa.free(t.jobs);
        t.handoff = try Handoff.init(device, &t.failed, &t.stop, t.rank);
        errdefer t.handoff.deinit();
        t.owned = owned;
        t.one.slice(i32, 1)[0] = 1;
        t.trace = platform.c.getenv("TF_TP_TRACE") != null;
        t.local = platform.c.getenv("TF_TP_LOCAL") != null;
        t.stats = platform.c.getenv("TF_TP_STATS") != null;
        // both ranks up before the first round: a word each way
        t.benchmark_group = try @import("core").benchmark_group.metal(gpa, io, device, rd, FLAGS + 4096, HELLO, 2);
        errdefer if (t.benchmark_group) |b| gpa.free(b);
        t.monitor = try std.Thread.spawn(.{}, Handoff.watch, .{&t.handoff});
        errdefer {
            t.stop.store(true, .release);
            t.monitor.?.join();
        }
        t.thread = try std.Thread.spawn(.{}, relay.service, .{t});
        std.log.info("TP=2 rank {d} connected", .{t.rank});
        return t;
    }
    /// Stop the host's service thread and close the link; every GPU use of the window has ended.
    pub fn deinit(t: *Tp2) void {
        t.stopService();
        t.handoff.deinit();
        t.owned.deinit();
        t.ep.deinit();
        if (t.allocator) |a| {
            if (t.benchmark_group) |b| a.free(b);
            a.free(t.jobs);
            a.destroy(t);
        }
    }
    pub fn stopService(t: *Tp2) void {
        const healthy = !t.failed.load(.acquire);
        t.stop.store(true, .release);
        t.handoff.cancel();
        if (t.thread) |th| th.join();
        t.thread = null;
        if (t.monitor) |th| th.join();
        t.monitor = null;
        if (healthy) {
            const now = fabric.words.checkedNowNs() catch return;
            t.ep.flushPeerUntil(t.peer, now +| std.time.ns_per_s) catch {};
        }
    }
    /// This rank's expert and MoE rows [lo, hi): rank 0 takes the first half rounded up, rank 1 the rest.
    pub fn ownRows(t: *const Tp2, rows: usize) [2]usize {
        return tp_seq.ownRows(t.rank, rows);
    }
    /// A failed handoff makes every associated GPU result invalid.
    pub fn gaveUp(t: *const Tp2) u32 {
        return @intFromBool(t.failed.load(.acquire));
    }
    /// Cold pipeline compilation stays on the host, ahead of the bounded GPU event waits.
    pub fn prepareSubmit(t: *Tp2) !void {
        @import("tp_submit.zig").prepare(t, SUBMIT_FLAG) catch |err| {
            t.handoff.fail(t.seq, err);
            return err;
        };
    }
    /// A new sequence with its job queued for the service thread.
    fn queue(t: *Tp2, job: Job) u64 {
        t.seq += 1;
        t.jobs[t.seq % JOBS] = job;
        t.queued.store(t.seq, .release);
        return t.seq;
    }
    /// The host establishes this receive before the event releases any GPU payload reads.
    pub fn waitWord(t: *Tp2, enc: *mtl.ComputeEncoder, off: usize, value: u32) void {
        const seq = t.queue(.{ .kind = .wait, .flag = off, .value = value });
        t.handoff.seam(enc, seq);
    }

    pub fn copyPeer(t: *Tp2, enc: mtl.ComputeEncoder, src: anytype, dst: anytype, words: usize) void {
        t.copy(enc, src, dst, words, t.seq);
    }

    fn copy(t: *Tp2, enc: mtl.ComputeEncoder, src: anytype, dst: anytype, words: usize, expected: u64) void {
        enc.setPipeline(t.copy_pipe);
        t.handoff.bind(enc, expected, 0);
        enc.setBuffer(src.b, src.off, 2);
        enc.setBuffer(dst.b, dst.off, 3);
        enc.setValue(@as(u32, @intCast(words)), 4);
        enc.dispatchThreads(mtl.Size.of(words, 1, 1), mtl.Size.of(256, 1, 1));
    }
    /// Encode owned MoE branches into the send buffer; combine posts them and updates every row.
    pub fn exchange(t: *Tp2, enc: mtl.ComputeEncoder, y: anytype, wts: anytype, lg: anytype, rows: usize) void {
        const m = t.seqs.exchange();
        t.last_x = m;
        t.last_seq = t.queue(.{ .kind = .exchange, .value = m.flag, .flag = FLAG, .local = sendX(m.slot), .remote = DECODE + m.slot * SLOT, .bytes = tp_seq.bytes(.exchange, t.rank, rows) });
        const own = t.ownRows(rows);
        if (own[1] > own[0]) {
            const own2 = [2]u32{ @intCast(own[0]), @intCast(own[1]) };
            enc.setPipeline(t.branch_pipe);
            for ([_]@TypeOf(y){ y, wts, lg }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
            enc.setBuffer(t.wbuf, sendX(m.slot), 3);
            enc.setBytes(std.mem.asBytes(&own2), 4);
            t.handoff.bind(enc, t.last_seq - 1, 5);
            enc.dispatchThreads(mtl.Size.of(D, own[1] - own[0], 1), mtl.Size.of(256, 1, 1));
        }
    }
    /// Publish the last exchange, then update streams from each row's owning rank's MoE branch.
    pub fn combine(t: *Tp2, enc: *mtl.ComputeEncoder, h: anytype, inj: anytype, out: anytype, ssp: anytype, rows_buf: anytype, rows: usize) void {
        const own = t.ownRows(rows);
        const peer_lo: usize = if (t.rank == 0) own[1] else 0;
        const split = [4]u32{ @intCast(own[0]), @intCast(own[1]), @intCast(peer_lo), 0 };
        t.handoff.seam(enc, t.last_seq);
        enc.setPipeline(t.merge_pipe);
        t.handoff.bind(enc.*, t.last_seq, 0);
        enc.setBuffer(rows_buf.b, rows_buf.off, 2);
        enc.setBuffer(h.b, h.off, 3);
        enc.setBuffer(inj.b, inj.off, 4);
        enc.setBuffer(t.wbuf, sendX(t.last_x.slot), 5);
        enc.setBuffer(t.wbuf, DECODE + t.last_x.slot * SLOT, 6);
        enc.setBytes(std.mem.asBytes(&split), 7);
        enc.setBuffer(out.b, out.off, 8);
        enc.setBuffer(ssp.b, ssp.off, 9);
        enc.dispatchThreads(mtl.Size.of(D, rows, 1), mtl.Size.of(256, 1, 1));
    }
    /// Exchange half-vocabulary argmax picks; choose the larger value, then the lower index on ties.
    pub fn argmax(t: *Tp2, enc: *mtl.ComputeEncoder, logits: anytype, vocab: usize, lo: usize, n: usize, picks: anytype, rows_buf: anytype, rows: usize) void {
        const m = t.seqs.pick();
        const x = m.flag;
        const seq = t.queue(.{ .kind = .pick, .value = x, .flag = FLAG, .local = sendA(m.slot), .remote = RECVA + m.slot * PAGE, .bytes = tp_seq.bytes(.pick, t.rank, rows) });
        const dims = [4]u32{ @intCast(vocab), @intCast(lo), @intCast(n), 0 };
        enc.setPipeline(t.argmax_pipe);
        enc.setBuffer(logits.b, logits.off, 0);
        enc.setBuffer(t.wbuf, sendA(m.slot), 1);
        enc.setBytes(std.mem.asBytes(&dims), 2);
        t.handoff.bind(enc.*, seq - 1, 3);
        enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
        t.handoff.seam(enc, seq);
        enc.setPipeline(t.pick_pipe);
        t.handoff.bind(enc.*, seq, 0);
        enc.setBuffer(rows_buf.b, rows_buf.off, 2);
        enc.setBuffer(t.wbuf, sendA(m.slot), 3);
        enc.setBuffer(t.wbuf, RECVA + m.slot * PAGE, 4);
        enc.setBuffer(picks.b, picks.off, 5);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }

    /// After `argmax` over the MTP head's draft logits into `pick_tmp`: the merged pick's token id into `out`.
    pub fn mapIds(t: *Tp2, enc: mtl.ComputeEncoder, ids: anytype, out: anytype) void {
        enc.setPipeline(t.ids_pipe);
        enc.setBuffer(t.pick_tmp, 0, 0);
        enc.setBuffer(ids.b, ids.off, 1);
        enc.setBuffer(out.b, out.off, 2);
        t.handoff.bind(enc, t.seq, 3);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }

    /// Scratch for the next projection partial (fp32, rows x 2560), copied to its outgoing slot before posting.
    pub fn partNext(t: *const Tp2) struct { b: mtl.Buffer, off: usize } {
        return .{ .b = t.partial, .off = 0 };
    }

    /// Exchange projection partials, add rank 0 then rank 1, and round the branch once to bf16 for the stream update.
    pub fn plain(t: *Tp2, enc: *mtl.ComputeEncoder, h: anytype, inj: anytype, out: anytype, ssp: anytype, rows_buf: anytype, rows: usize) void {
        const m = t.seqs.reduce();
        const x = m.flag;
        const seq = t.queue(.{ .kind = .reduce, .value = x, .flag = FLAG, .local = sendR(m.slot), .remote = REDUCE + m.slot * PART, .bytes = tp_seq.bytes(.reduce, t.rank, rows) });
        t.copy(enc.*, .{ .b = t.partial, .off = @as(usize, 0) }, .{ .b = t.wbuf, .off = sendR(m.slot) }, rows * D, seq - 1);
        t.handoff.seam(enc, seq);
        enc.setPipeline(t.plain_pipe);
        t.handoff.bind(enc.*, seq, 0);
        enc.setBuffer(rows_buf.b, rows_buf.off, 2);
        enc.setBuffer(t.wbuf, sendR(m.slot), 3);
        enc.setBuffer(t.wbuf, REDUCE + m.slot * PART, 4);
        enc.setBytes(std.mem.asBytes(&t.rank), 5);
        enc.setBuffer(h.b, h.off, 6);
        enc.setBuffer(inj.b, inj.off, 7);
        enc.setBuffer(out.b, out.off, 8);
        enc.setBuffer(ssp.b, ssp.off, 9);
        enc.dispatchThreads(mtl.Size.of(D, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// Hand off the GPU's sources to the host, then resume after the send and flush finish.
    pub fn send(t: *Tp2, enc: *mtl.ComputeEncoder, writes: []const Write, flag: usize, value: u64) void {
        var job: Job = .{ .kind = .send, .n = writes.len, .flag = flag, .value = value };
        @memcpy(job.writes[0..writes.len], writes);
        const seq = t.queue(job);
        t.handoff.seam(enc, seq);
    }

    /// The host side alone: copy `writes` and signal now (after a command buffer has completed).
    pub fn sendNow(t: *Tp2, writes: []const Write, flag: usize, value: u64) !void {
        if (t.failed.load(.acquire)) return error.TpLinkFailed;
        for (writes, 0..) |w, j| {
            if (t.failed.load(.acquire)) return error.TpLinkFailed;
            if (j + 1 == writes.len) try t.rd.write2Signal(t.peer, w.dst, w.src[0..w.len], &.{}, flag, value) else try t.rd.write(t.peer, w.dst, w.src[0..w.len]);
        }
        if (writes.len == 0) try t.rd.signal(t.peer, flag, value);
    }

    /// Served, rank 0: hand rank 1 the next request (`head` then the prompt's tokens) in one message.
    pub fn sendRequest(t: *Tp2, head: []const u8, prompt: []const u32) !void {
        if (t.failed.load(.acquire)) return error.TpLinkFailed;
        if (head.len > 64 or prompt.len > REQ_TOKENS) return error.TpRequestTooLarge;
        t.req += 1;
        var h64: [64]u8 = @splat(0);
        @memcpy(h64[0..head.len], head);
        try t.rd.write2Signal(t.peer, REQ, &h64, std.mem.sliceAsBytes(prompt), REQ_FLAG, t.req);
    }

    /// After a pair chunk: this Mac's MARK is free for the peer's states of mark exchange `seq`.
    pub fn markReady(t: *Tp2, seq: u64) !void {
        if (t.failed.load(.acquire)) return error.TpLinkFailed;
        try t.rd.signal(t.peer, MARK_READY, seq);
    }

    /// Rank 1 tells rank 0 whether this request's resume state is ready before its prompt pass.
    pub fn ackRequest(t: *Tp2, ok: bool) !void {
        if (t.failed.load(.acquire)) return error.TpLinkFailed;
        try t.rd.signal(t.peer, ACK_FLAG, (t.req << 1) | @intFromBool(ok));
    }

    /// Served, rank 0: rank 1's answer for the request just sent: true when it restored the same resume state.
    pub fn waitAck(t: *Tp2) !bool {
        const t0 = platform.c.mach_absolute_time();
        while (true) {
            const v = @atomicLoad(u64, t.word64(ACK_FLAG), .acquire);
            if (v >> 1 >= t.req) return v >> 1 == t.req and v & 1 != 0;
            if (platform.c.mach_absolute_time() - t0 > 7_200_000_000) return error.TpPeerSilent; // 300 s for loading
            std.atomic.spinLoopHint();
        }
    }

    /// Served, rank 1, between requests: whether rank 0 has written one this Mac has not taken (or the wait is ending).
    pub fn requestWaiting(t: *const Tp2) bool {
        return t.quitting.load(.acquire) or t.stop.load(.acquire) or t.failed.load(.acquire) or
            @atomicLoad(u64, t.word64(REQ_FLAG), .acquire) > t.req;
    }

    /// Rank 1 waits for the next request, probing transport while idle; null once shutdown ends the wait.
    pub fn waitRequest(t: *Tp2) ?struct { head: *const [64]u8, tokens: [*]const u32 } {
        t.req += 1;
        var spins: usize = 0;
        while (@atomicLoad(u64, t.word64(REQ_FLAG), .acquire) < t.req) {
            if (t.quitting.load(.acquire)) return null;
            spins += 1;
            if (spins > 100_000) { // idle: back off
                const ts: platform.c.timespec = .{ .sec = 0, .nsec = 50_000 };
                _ = platform.c.nanosleep(&ts, null);
            } else std.atomic.spinLoopHint();
        }
        return .{ .head = @ptrCast(t.win.ptr + REQ), .tokens = @ptrCast(@alignCast(t.win.ptr + REQ + 64)) };
    }

    /// Rank 0 sends its stop decision and draft depth at each shared step; rank 1 waits for that decision.
    pub fn agree(t: *Tp2, quit: bool, depth: u32) !struct { quit: bool, depth: u32 } {
        if (t.failed.load(.acquire)) return error.TpLinkFailed;
        t.ctrl += 1;
        if (t.rank == 0) {
            try t.rd.signal(t.peer, CTRL_FLAG, (t.ctrl << 6) | (@as(u64, depth & 31) << 1) | @intFromBool(quit));
            return .{ .quit = quit, .depth = depth };
        }
        const t0 = platform.c.mach_absolute_time();
        while (true) {
            const v = @atomicLoad(u64, t.word64(CTRL_FLAG), .acquire);
            if (v >> 6 >= t.ctrl) return .{ .quit = quit or (v >> 6 == t.ctrl and v & 1 != 0), .depth = @intCast((v >> 1) & 31) };
            if (platform.c.mach_absolute_time() - t0 > 240_000_000) return error.TpPeerSilent; // 10 s
            std.atomic.spinLoopHint();
        }
    }

    /// The host waits until the window's word at `off` reaches `value`.
    pub fn hostWait(t: *Tp2, off: usize, value: u64) !void {
        if (t.trace) std.debug.print("TP rank{d} host waits for word {d} >= {d} (now {d})\n", .{ t.rank, off, value, @atomicLoad(u64, t.word64(off), .acquire) });
        host_wait.wait(t.word64(off), value, .{ .failed = &t.failed, .quitting = &t.quitting, .stop = &t.stop }, 10 * std.time.ns_per_s) catch |err| {
            if (err != error.TpStopping) t.handoff.fail(t.seq, err);
            return err;
        };
        if (t.trace) std.debug.print("TP rank{d} word {d} reached {d}\n", .{ t.rank, off, value });
    }

    pub fn window(t: *const Tp2) mtl.Buffer {
        return t.wbuf;
    }

    pub fn bytes(t: *const Tp2, off: usize) [*]u8 {
        return t.win.ptr + off;
    }

    pub fn word64(t: *const Tp2, off: usize) *u64 {
        return @ptrCast(@alignCast(t.win.ptr + off));
    }

    pub fn recordStats(t: *Tp2, seq: u64, job: Job, seen: u64) void {
        if (!t.stats) return;
        const held = platform.c.mach_absolute_time() - seen;
        t.held_ns += held;
        t.held_max = @max(t.held_max, held);
        t.held_kind[@backingInt(job.kind)][0] += held;
        t.held_kind[@backingInt(job.kind)][1] += 1;
        if (seq % 4096 != 0) return;
        std.debug.print("TP rank{d} jobs to {d}: host held {d:.1} us a job, max {d:.1} us\n", .{ t.rank, seq, @as(f64, @floatFromInt(t.held_ns)) / 4096.0 / 24.0, @as(f64, @floatFromInt(t.held_max)) / 24.0 });
        for (t.held_kind, 0..) |hk, kk| if (hk[1] > 0) std.debug.print("TP rank{d}   {s}: {d:.1} us a job ({d} jobs)\n", .{ t.rank, @tagName(@as(@TypeOf(job.kind), @fromBackingInt(@intCast(kk)))), @as(f64, @floatFromInt(hk[0])) / @as(f64, @floatFromInt(hk[1])) / 24.0, hk[1] });
        t.held_ns = 0;
        t.held_max = 0;
        t.held_kind = @splat(.{ 0, 0 });
    }
};

test {
    _ = @import("tp_deadline.zig");
    _ = tp_seq; // the decode messages' numbers and slots
}

test "host waits retain the wrapping prefill call counter" {
    const win = try std.heap.page_allocator.alloc(u8, WINDOW);
    defer std.heap.page_allocator.free(win);
    const Peer = struct {
        fn land(w: *u64, v: u64) void {
            @import("tp_handoff.zig").sleep(20_000_000);
            @atomicStore(u64, w, v, .release);
        }
    };
    var failed: std.atomic.Value(bool) = .init(false);
    var quitting: std.atomic.Value(bool) = .init(false);
    var stop: std.atomic.Value(bool) = .init(false);
    const word: *u64 = @ptrCast(@alignCast(win.ptr + BACK_FLAG));
    word.* = 0xFFFF_FFFE;
    for ([_]u64{ 0xFFFF_FFFF, 0, 1 }) |call| {
        const th = try std.Thread.spawn(.{}, Peer.land, .{ word, call });
        defer th.join();
        try host_wait.wait(word, call, .{ .failed = &failed, .quitting = &quitting, .stop = &stop }, std.time.ns_per_s);
        try std.testing.expectEqual(call, @atomicLoad(u64, word, .acquire));
    }
}
