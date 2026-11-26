//! N-rank (1..8 Macs) tensor- and pipeline-parallel exchanges in a Metal command buffer over MCDMA RDMA.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const fabric = @import("fabric");
const settings = @import("tp_settings.zig");

pub const Settings = settings.Settings;
pub const Ref = @import("exlr8_projection.zig").Ref;
pub const Mesh = Pair;
pub const max_ranks = 8;

const layout_mod = @import("pair_layout.zig");
pub const Layout = layout_mod.Layout;
pub const command_words = layout_mod.command_words;
const PAGE = layout_mod.PAGE;
const CMD_BYTES = layout_mod.CMD_BYTES;
const pages = layout_mod.pages;
const tpFlag = layout_mod.tpFlag;
const cmdAck = layout_mod.cmdAck;
const readyFlag = layout_mod.readyFlag;
const PIPE_FLAG = layout_mod.PIPE_FLAG;
const PICK_FLAG = layout_mod.PICK_FLAG;
const CMD_FLAG = layout_mod.CMD_FLAG;
const reached = layout_mod.reached;
const JOBS = 4096;
const GPU_PACKED_BIT: u32 = 0x8000_0000;

threadlocal var raised = false;

/// Raise this thread to user-interactive QoS once so busy CPUs never delay RDMA sends or commands.
pub fn raise() void {
    if (raised) return;
    _ = platform.c.pthread_set_qos_class_self_np(.USER_INTERACTIVE, 0);
    raised = true;
}

fn spinOrSleep(spins: *usize, limit: usize) void {
    spins.* += 1;
    if (spins.* > limit) {
        const ts: platform.c.timespec = .{ .sec = 0, .nsec = 20_000 };
        _ = platform.c.nanosleep(&ts, null);
    } else std.atomic.spinLoopHint();
}

const PeerTx = struct {
    p: *Pair = undefined,
    dst_tp: u32 = 0,
    thread: ?std.Thread = null,
    gen: std.atomic.Value(u32) = .init(0),
    done: std.atomic.Value(u32) = .init(0),
};

pub const Pair = struct {
    benchmark_group: ?[]const u8 = null,
    benchmark_allocator: ?std.mem.Allocator = null,
    rank: u32,
    ranks: u32,
    peer: u32, // legacy alias for 2-rank callers
    tp: u32,
    pp: u32,
    tp_rank: u32,
    pp_rank: u32,
    tp_peers: [max_ranks]u32 = @splat(0),
    prev_stage_peer: ?u32 = null,
    next_stage_peer: ?u32 = null,
    last_stage_rank0: u32 = 0,
    cfg: Settings,
    l: Layout,
    ep: ?*fabric.mcdma.Endpoint = null,
    rd: ?fabric.rdma.Rdma = null,
    owned_win: ?[]align(PAGE) u8 = null,
    win: []u8,
    wbuf: mtl.Buffer,
    pipes: layout_mod.Pipes,
    /// TP exchanges encoded so far (same count on every rank in a TP stage).
    x: u64 = 0,
    bytes: []u32,
    queued: std.atomic.Value(u32) = .init(0),
    /// PP activation sends/recvs encoded so far.
    pipe_tx: u64 = 0,
    pipe_rx: u64 = 0,
    pipe_bytes: []u32,
    pipe_queued: std.atomic.Value(u32) = .init(0),
    /// PP final-stage token picks returns encoded so far.
    pick_tx: u64 = 0,
    pick_rx: u64 = 0,
    pick_bytes: []u32,
    pick_queued: std.atomic.Value(u32) = .init(0),
    thread: ?std.Thread = null,
    peer_tx: [max_ranks]PeerTx = @splat(.{}),
    tx_x: u32 = 0,
    tx_len: u32 = 0,
    tx_packed: bool = false,
    tx_gen: u32 = 0,
    first_tp_peer: u32 = 0,
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    /// Commands sent (rank 0) or taken (followers) so far.
    cmd: u64 = 0,
    /// Optional Q8 wire row threshold (TF_PAIR_Q8_MIN_ROWS, default 32).
    q8_min_rows: u32 = 32,
    /// Fused coherent(device) wait+reduce BF16 decode kernel (TF_PAIR_FUSED_WAIT, default true) & inline TX limit.
    fused_wait: bool = true,
    inline_tx_max: u32 = 65536,
    /// Live RDMA telemetry counters.
    exchanges_total: std.atomic.Value(u64) = .init(0),
    bytes_sent_total: std.atomic.Value(u64) = .init(0),
    rdma_send_ns_total: std.atomic.Value(u64) = .init(0),

    /// Initialize a standalone 1-rank Mesh (no MCDMA library needed).
    pub fn initSingle(gpa: std.mem.Allocator, device: mtl.Device, bytes: usize) !*Pair {
        const s: Settings = .{ .rank = 0, .ranks = 1, .tp = 1, .pp = 1, .library = "", .links = &.{} };
        return initFromSettings(gpa, null, device, s, bytes);
    }

    /// Connect to all peers named in `settings_path` (1..8 ranks, TP/PP), and start the RDMA relay thread.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, settings_path: []const u8, bytes: usize) !*Pair {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, settings_path, gpa, .limited(1 << 20));
        const s = try settings.read(gpa, text);
        return initFromSettings(gpa, io, device, s, bytes);
    }

    pub fn initFromSettings(gpa: std.mem.Allocator, io_opt: ?std.Io, device: mtl.Device, s: Settings, bytes: usize) !*Pair {
        const ranks = s.worldSize();
        const tp = s.tpSize();
        const pp = s.ppSize();
        const tp_rank = s.tpRank();
        const pp_rank = s.ppRank();
        if (ranks == 0 or ranks > max_ranks or s.rank >= ranks) return error.BadTpSettings;
        const l: Layout = .{ .slot = pages(bytes), .tp = tp, .pp = pp, .ranks = ranks };
        var ep: ?*fabric.mcdma.Endpoint = null;
        var rd: ?fabric.rdma.Rdma = null;
        var owned_win: ?[]align(PAGE) u8 = null;
        var win: []u8 = undefined;
        if (ranks > 1) {
            const lib = try gpa.dupeSentinel(u8, s.library, 0);
            const created = try fabric.mcdma.Endpoint.create(gpa, lib, .{
                .rank = s.rank,
                .ranks = ranks,
                .window_bytes = l.window(),
                .staging_bytes = 4 << 20,
                .links = s.links,
                .timeout_ns = 60 * std.time.ns_per_s,
                .connect_timeout_ns = 300 * std.time.ns_per_s,
            });
            ep = created;
            rd = created.rdma();
            win = rd.?.window();
        } else {
            owned_win = try gpa.alignedAlloc(u8, .fromByteUnits(PAGE), l.window());
            @memset(owned_win.?, 0);
            win = owned_win.?;
        }
        errdefer {
            if (ep) |e| e.deinit();
            if (owned_win) |w| gpa.free(w);
        }
        const pipes = try layout_mod.Pipes.load(device);
        const p = try gpa.create(Pair);
        const q8_min: u32 = platform.environment.read(u32, "TF_PAIR_Q8_MIN_ROWS", 32) catch 32;
        const fused_wait = platform.environment.read(bool, "TF_PAIR_FUSED_WAIT", true) catch true;
        const inline_tx: u32 = platform.environment.read(u32, "TF_PAIR_INLINE_TX_MAX", 65536) catch 65536;
        p.* = .{
            .benchmark_allocator = gpa,
            .rank = s.rank,
            .ranks = ranks,
            .peer = if (ranks == 2) 1 - s.rank else (s.rank + 1) % ranks,
            .tp = tp,
            .pp = pp,
            .tp_rank = tp_rank,
            .pp_rank = pp_rank,
            .last_stage_rank0 = (pp - 1) * tp,
            .cfg = s,
            .l = l,
            .ep = ep,
            .rd = rd,
            .owned_win = owned_win,
            .win = win,
            .wbuf = try device.bufferNoCopy(win.ptr, win.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked),
            .pipes = pipes,
            .bytes = try gpa.alloc(u32, JOBS),
            .pipe_bytes = try gpa.alloc(u32, JOBS),
            .pick_bytes = try gpa.alloc(u32, JOBS),
            .q8_min_rows = q8_min,
            .fused_wait = fused_wait,
            .inline_tx_max = inline_tx,
        };
        for (0..tp) |i| p.tp_peers[i] = pp_rank * tp + @as(u32, @intCast(i));
        if (pp > 1) {
            if (pp_rank > 0) p.prev_stage_peer = (pp_rank - 1) * tp + tp_rank;
            if (pp_rank + 1 < pp) p.next_stage_peer = (pp_rank + 1) * tp + tp_rank;
        }
        if (ranks > 1) {
            if (io_opt) |io| {
                var peer_ids: [max_ranks]u32 = undefined;
                for (s.links, 0..) |lnk, i| peer_ids[i] = lnk.peer;
                p.benchmark_group = try @import("benchmark_group.zig").metalMesh(
                    gpa,
                    io,
                    device,
                    rd.?,
                    l.flags() + layout_mod.WIRE_BASE,
                    l.flags() + layout_mod.HELLO_BASE,
                    peer_ids[0..s.links.len],
                    2,
                );
            }
            var assigned_first = false;
            for (0..tp) |dst_i| {
                const dst_tp: u32 = @intCast(dst_i);
                if (dst_tp == tp_rank) continue;
                if (!assigned_first) {
                    p.first_tp_peer = dst_tp;
                    assigned_first = true;
                } else {
                    p.peer_tx[dst_tp].p = p;
                    p.peer_tx[dst_tp].dst_tp = dst_tp;
                    p.peer_tx[dst_tp].thread = try std.Thread.spawn(.{}, peerWorker, .{&p.peer_tx[dst_tp]});
                }
            }
            p.thread = try std.Thread.spawn(.{}, service, .{p});
        }
        return p;
    }

    /// Rank 0 notifies all followers it is done; then the relay threads and MCDMA links close cleanly.
    pub fn deinit(p: *Pair) void {
        if (p.ranks > 1 and p.rank == 0 and !p.failed.load(.acquire)) p.send(0, &.{}) catch {};
        if (p.rd) |r| r.flush() catch {};
        p.stop.store(true, .release);
        if (p.thread) |t| t.join();
        for (&p.peer_tx) |*h| if (h.thread) |t| t.join();
        p.pipes.deinit();
        if (p.ep) |e| e.deinit();
        if (p.benchmark_allocator) |a| {
            if (p.owned_win) |w| a.free(w);
            if (p.benchmark_group) |b| a.free(b);
            a.free(p.bytes);
            a.free(p.pipe_bytes);
            a.free(p.pick_bytes);
            a.destroy(p);
        }
    }

    fn word64(p: *const Pair, off: usize) *u64 {
        return @ptrCast(@alignCast(p.win.ptr + off));
    }
    fn word32(p: *const Pair, off: usize) *u32 {
        return @ptrCast(@alignCast(p.win.ptr + off));
    }

    /// GPU waits that gave up (a peer that never answered): nonzero means the run is invalid.
    pub fn gaveUp(p: *const Pair) u32 {
        return @atomicLoad(u32, p.word32(p.l.sync() + layout_mod.SYNC_GAVE_UP), .acquire);
    }

    /// Where the next TP exchange's local partial goes (rows x width fp32): producer writes here.
    pub fn part(p: *const Pair) Ref {
        return .{ .buf = p.wbuf, .off = p.l.sendPeer(p.x + 1, p.tp_rank) };
    }

    /// Direct-pack target for producer kernels on decode (r=1): base of peer send slots + uint2 stride.
    pub fn packTarget(p: *const Pair) struct { base: Ref, info: [4]u32 } {
        return .{ .base = .{ .buf = p.wbuf, .off = p.l.sendPeer(p.x + 1, 0) }, .info = .{ p.tp_rank, p.tp, @intCast((PAGE + p.l.slot) / 8), 0 } };
    }

    fn queue(p: *Pair, bytes: usize, gpu_packed: bool) u32 {
        p.x += 1;
        const x: u32 = @truncate(p.x);
        p.bytes[x % JOBS] = @as(u32, @intCast(bytes)) | (if (gpu_packed) GPU_PACKED_BIT else 0);
        p.queued.store(x, .release);
        return x;
    }

    pub fn addRms(p: *Pair, e: mtl.ComputeEncoder, h: Ref, w: Ref, y: Ref, rows: u32, width: u32, eps: f32) void {
        _ = p.addRmsOpt(e, h, w, y, rows, width, eps, false, 0, 0, null);
    }
    pub fn addRmsXh(p: *Pair, e: mtl.ComputeEncoder, h: Ref, w: Ref, y: Ref, rows: u32, width: u32, eps: f32, xh_off4: u32) bool {
        return p.addRmsOpt(e, h, w, y, rows, width, eps, false, 0, xh_off4, null);
    }
    pub fn addRmsPrepacked(p: *Pair, e: mtl.ComputeEncoder, h: Ref, w: Ref, y: Ref, rows: u32, width: u32, eps: f32) void {
        _ = p.addRmsOpt(e, h, w, y, rows, width, eps, p.tp > 1 and rows <= 16 and (width % 4 == 0), 0, 0, null);
    }
    pub fn addRmsPrepackedGated(p: *Pair, e: mtl.ComputeEncoder, h: Ref, w: Ref, y: Ref, rows: u32, width: u32, eps: f32, gate_meta: ?mtl.Buffer) void {
        _ = p.addRmsOpt(e, h, w, y, rows, width, eps, p.tp > 1 and rows <= 16 and (width % 4 == 0), 0, 0, gate_meta);
    }
    pub fn addRmsPrepackedXhGated(p: *Pair, e: mtl.ComputeEncoder, h: Ref, w: Ref, y: Ref, rows: u32, width: u32, eps: f32, xh_off4: u32, gate_meta: ?mtl.Buffer) bool {
        return p.addRmsOpt(e, h, w, y, rows, width, eps, p.tp > 1 and rows <= 16 and (width % 4 == 0), 0, xh_off4, gate_meta);
    }
    pub fn addRmsReplace(p: *Pair, e: mtl.ComputeEncoder, h: Ref, w: Ref, y: Ref, rows: u32, width: u32, eps: f32, prepacked: bool) void {
        _ = p.addRmsOpt(e, h, w, y, rows, width, eps, prepacked and p.tp > 1 and rows <= 16 and (width % 4 == 0), 1, 0, null);
    }
    pub fn addRmsReplaceGated(p: *Pair, e: mtl.ComputeEncoder, h: Ref, w: Ref, y: Ref, rows: u32, width: u32, eps: f32, prepacked: bool, gate_meta: ?mtl.Buffer) void {
        _ = p.addRmsOpt(e, h, w, y, rows, width, eps, prepacked and p.tp > 1 and rows <= 16 and (width % 4 == 0), 1, 0, gate_meta);
    }
    pub fn addRmsReplaceXhGated(p: *Pair, e: mtl.ComputeEncoder, h: Ref, w: Ref, y: Ref, rows: u32, width: u32, eps: f32, prepacked: bool, xh_off4: u32, gate_meta: ?mtl.Buffer) bool {
        return p.addRmsOpt(e, h, w, y, rows, width, eps, prepacked and p.tp > 1 and rows <= 16 and (width % 4 == 0), 1, xh_off4, gate_meta);
    }
    pub fn sumOnly(p: *Pair, e: mtl.ComputeEncoder, dst: Ref, rows: u32, width: u32, prepacked: bool) void {
        _ = p.addRmsOpt(e, dst, dst, dst, rows, width, 1e-5, prepacked and p.tp > 1 and rows <= 16 and (width % 4 == 0), 2, 0, null);
    }
    fn addRmsOpt(p: *Pair, e: mtl.ComputeEncoder, h: Ref, w: Ref, y: Ref, rows: u32, width: u32, eps: f32, prepacked: bool, replace_mode: u32, xh_off4: u32, gate_meta: ?mtl.Buffer) bool {
        if (replace_mode == 0 and gate_meta == null and p.tp >= 3 and rows >= 64 and (width % 64 == 0)) {
            const nbytes_slice = ((rows + p.tp - 1) / p.tp) * (width / 64) * 68;
            p.pipes.encodeAddRmsRsAg(e, p.wbuf, p.l, p.queue(nbytes_slice, true), p.queue(nbytes_slice, true), p.tp_rank, p.tp, h, w, y, rows, width, eps);
            return false;
        }
        const elems = @as(usize, rows) * width;
        const use_q8 = p.tp > 1 and rows >= p.q8_min_rows and (width % 64 == 0);
        const use_bf16 = p.tp > 1 and !use_q8 and (width % 4 == 0);
        const nbytes = if (use_q8) (elems / 64) * 68 else if (use_bf16) elems * 2 else elems * 4;
        p.pipes.encodeAddRms(e, p.wbuf, p.l, p.queue(nbytes, use_q8 or use_bf16), p.tp_rank, p.tp, h, w, y, rows, width, eps, use_q8, use_bf16, prepacked and use_bf16, p.fused_wait, replace_mode, xh_off4, gate_meta);
        return use_bf16 and xh_off4 != 0;
    }
    pub fn reserveRaw(p: *Pair, nbytes: usize) struct { x: u32, send: Ref, recv: Ref, slot_u32: u32 } {
        const x = p.queue(nbytes, false);
        return .{ .x = x, .send = .{ .buf = p.wbuf, .off = p.l.sendPeer(x, p.tp_rank) }, .recv = .{ .buf = p.wbuf, .off = p.l.recvBase(x) }, .slot_u32 = @intCast(p.l.slot / 4) };
    }
    pub fn waitRaw(p: *Pair, e: mtl.ComputeEncoder, x: u32) void {
        p.pipes.encodeWaitRaw(e, p.wbuf, p.l, x, p.tp_rank, p.tp);
    }

    /// Reserve a TP argmax exchange (`with_prob`: 36B top-4 + sum_exp else 8B [val, id] per row).
    pub fn reservePick(p: *Pair, rows: u32, with_prob: bool) struct { x: u32, send: Ref } {
        const x = p.queue(@as(usize, rows) * (if (with_prob) @as(usize, 36) else 8), false);
        return .{ .x = x, .send = .{ .buf = p.wbuf, .off = p.l.sendPeer(x, p.tp_rank) } };
    }

    /// Exchange per-row argmax (+ optional softmax sum_exp) across TP peers into `picks`.
    pub fn finishPick(p: *Pair, e: mtl.ComputeEncoder, x: u32, rows: u32, picks: Ref, with_prob: bool) void {
        p.pipes.encodeFinishPick(e, p.wbuf, p.l, x, p.tp_rank, p.tp, rows, picks, with_prob, null);
    }
    pub fn finishPickGated(p: *Pair, e: mtl.ComputeEncoder, x: u32, rows: u32, picks: Ref, with_prob: bool, gate_meta: ?mtl.Buffer) void {
        p.pipes.encodeFinishPick(e, p.wbuf, p.l, x, p.tp_rank, p.tp, rows, picks, with_prob, gate_meta);
    }

    /// Each row's argmax over all TP peers' vocab slices (this rank holds `n` cols from `lo`) into `picks`.
    pub fn argmax(p: *Pair, e: mtl.ComputeEncoder, logits: Ref, stride: u32, lo: u32, n: u32, rows: u32, picks: Ref) void {
        const pk = p.reservePick(rows, false);
        e.setPipeline(p.pipes.argmax);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setValue([4]u32{ stride, lo, n, 0 }, 1);
        e.setBuffer(pk.send.buf, pk.send.off, 2);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
        e.barrier();
        p.finishPick(e, pk.x, rows, picks, false);
    }

    /// PP: send `rows * width` f32 hidden state to `next_stage_peer` (BF16 for decode, Q8_F32 for prefill).
    pub fn pipeSend(p: *Pair, e: mtl.ComputeEncoder, src: Ref, rows: u32, width: u32) void {
        if (p.next_stage_peer == null) return;
        const elems: usize = @as(usize, rows) * width;
        const use_q8 = rows >= 32 and (width % 64 == 0);
        const use_bf16 = !use_q8 and (width % 4 == 0);
        p.pipe_tx += 1;
        const seq: u32 = @truncate(p.pipe_tx);
        p.pipe_bytes[seq % JOBS] = @intCast(if (use_q8) (elems / 64) * 68 else if (use_bf16) elems * 2 else elems * 4);
        p.pipe_queued.store(seq, .release);
        p.pipes.encodePipeSend(e, p.wbuf, p.l, seq, src, elems, use_q8, use_bf16);
    }

    /// PP: receive `rows * width` f32 hidden state from `prev_stage_peer` into `dst` (BF16/Q8_F32 compressed).
    pub fn pipeRecv(p: *Pair, e: mtl.ComputeEncoder, dst: Ref, rows: u32, width: u32) void {
        if (p.prev_stage_peer == null) return;
        const elems: usize = @as(usize, rows) * width;
        const use_q8 = rows >= 32 and (width % 64 == 0);
        const use_bf16 = !use_q8 and (width % 4 == 0);
        p.pipe_rx += 1;
        const seq: u32 = @truncate(p.pipe_rx);
        p.pipes.encodePipeRecv(e, p.wbuf, p.l, seq, dst, elems, use_q8, use_bf16);
    }

    /// PP: final stage's tp_rank 0 relays `rows` picked token IDs back to rank 0.
    pub fn returnPicks(p: *Pair, e: mtl.ComputeEncoder, picks: Ref, rows: u32) void {
        if (p.pp <= 1 or p.rank != p.last_stage_rank0) return;
        p.pick_tx += 1;
        const seq: u32 = @truncate(p.pick_tx);
        p.pick_bytes[seq % JOBS] = rows * 4;
        p.pick_queued.store(seq, .release);
        e.setPipeline(p.pipes.pipe_send);
        e.setBuffer(p.wbuf, p.l.sync(), 0);
        e.setBuffer(picks.buf, picks.off, 1);
        e.setBuffer(p.wbuf, p.l.pickSend(seq), 2);
        e.setValue(layout_mod.PipeArgs{ .count_u32 = rows, .seq = seq, .sync_idx = 48 }, 3);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 256 });
    }

    /// PP: rank 0 receives `rows` picked token IDs from final stage's tp_rank 0 into `picks`.
    pub fn waitPicks(p: *Pair, e: mtl.ComputeEncoder, picks: Ref, rows: u32) void {
        if (p.pp <= 1 or p.rank != 0) return;
        p.pick_rx += 1;
        const seq: u32 = @truncate(p.pick_rx);
        e.setPipeline(p.pipes.pipe_recv);
        e.setBuffer(p.wbuf, p.l.sync(), 0);
        e.setBuffer(p.wbuf, p.l.flags() + PICK_FLAG, 1);
        e.setBuffer(p.wbuf, p.l.pickRecv(seq), 2);
        e.setBuffer(picks.buf, picks.off, 3);
        e.setValue(layout_mod.PipeArgs{ .count_u32 = rows, .seq = seq, .sync_idx = 0 }, 4);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 256 });
    }

    /// Rank 0: broadcast command `kind` (0 stops followers) to all followers `1..ranks-1`.
    pub fn send(p: *Pair, kind: u32, words: []const u32) !void {
        if (p.ranks <= 1) return;
        if (p.rank != 0) return error.FollowsPeer;
        if (words.len > command_words) return error.CommandTooLong;
        raise();
        for (1..p.ranks) |peer_usize| {
            try p.reach(p.l.flags() + cmdAck(@intCast(peer_usize)), p.cmd, true);
        }
        p.cmd += 1;
        const head: [2]u32 = .{ kind, @intCast(words.len) };
        const payload_len = 8 + 4 * words.len;
        const ep = p.ep orelse return error.PairLinkFailed;
        for (1..p.ranks) |peer_usize| {
            const peer_id: u32 = @intCast(peer_usize);
            const out = p.l.cmdOutPeer(peer_id);
            @memcpy(p.win[out..][0..8], std.mem.asBytes(&head));
            if (words.len > 0) @memcpy(p.win[out + 8 ..][0 .. 4 * words.len], std.mem.sliceAsBytes(words));
            try ep.writeSignalFrom(peer_id, out, p.l.cmdIn(), payload_len, p.l.flags() + CMD_FLAG, p.cmd);
        }
    }

    /// Followers (rank >= 1): receive rank 0's next command; null once rank 0 closes or this rank stops.
    pub fn next(p: *Pair, gpa: std.mem.Allocator, out: *std.ArrayList(u32)) !?u32 {
        if (p.ranks <= 1) return null;
        raise();
        p.reach(p.l.flags() + CMD_FLAG, p.cmd + 1, false) catch |err| return if (err == error.PairStopped) null else err;
        p.cmd += 1;
        const in = p.l.cmdIn();
        const head: *const [2]u32 = @ptrCast(@alignCast(p.win.ptr + in));
        if (head[1] > command_words) return error.CommandOutOfStep;
        out.clearRetainingCapacity();
        try out.appendSlice(gpa, @as([*]const u32, @ptrCast(@alignCast(p.win.ptr + in + 8)))[0..head[1]]);
        const kind = head[0];
        if (p.rd) |r| try r.signal(0, p.l.flags() + cmdAck(p.rank), p.cmd);
        return if (kind == 0) null else kind;
    }

    /// All ranks loaded before the first command: signal every linked peer and wait for every linked peer.
    pub fn ready(p: *Pair) !void {
        if (p.ranks <= 1) return;
        const r = p.rd orelse return;
        for (p.cfg.links) |lnk| {
            try r.signal(lnk.peer, p.l.flags() + readyFlag(p.rank), 1);
        }
        try r.flush();
        for (p.cfg.links) |lnk| {
            try p.reach(p.l.flags() + readyFlag(lnk.peer), 1, false);
        }
    }

    /// This rank's command wait ends (followers at close).
    pub fn stopFollowing(p: *Pair) void {
        p.stop.store(true, .release);
    }

    fn reach(p: *Pair, off: usize, value: u64, bounded: bool) !void {
        const t0 = platform.c.mach_absolute_time();
        var spins: usize = 0;
        while (@atomicLoad(u64, p.word64(off), .acquire) < value) {
            if (p.stop.load(.acquire)) return error.PairStopped;
            if (p.failed.load(.acquire)) return error.PairLinkFailed;
            if (bounded and platform.c.mach_absolute_time() - t0 > 60 * 24_000_000) return error.PairPeerSilent;
            spinOrSleep(&spins, 200_000);
        }
    }

    fn markAllArrived(p: *Pair, x: u32, pipe_s: u32, pick_s: u32) void {
        for (0..p.tp) |i| @atomicStore(u64, p.word64(p.l.flags() + tpFlag(@intCast(i))), x, .release);
        @atomicStore(u64, p.word64(p.l.flags() + PIPE_FLAG), pipe_s, .release);
        @atomicStore(u64, p.word64(p.l.flags() + PICK_FLAG), pick_s, .release);
    }

    fn sendOnePeer(p: *Pair, dst_tp: u32, next_x: u32, len: u32, gpu_packed: bool) void {
        const ep = p.ep orelse return;
        const src_off = p.l.sendPeer(next_x, p.tp_rank);
        const peer_send = if (gpu_packed or p.tp > 2) p.l.sendPeer(next_x, dst_tp) else src_off;
        if (!gpu_packed and p.tp > 2 and len > 0) @memcpy(p.win[peer_send..][0..len], p.win[src_off..][0..len]);
        ep.writeSignalFrom(p.tp_peers[dst_tp], peer_send, p.l.recv(next_x, p.tp_rank), len, p.l.flags() + tpFlag(p.tp_rank), next_x) catch |err| {
            std.log.err("mesh rank {d}: TP exchange {d} to peer {d} failed: {s}", .{ p.rank, next_x, p.tp_peers[dst_tp], @errorName(err) });
            p.failed.store(true, .release);
        };
    }

    fn peerWorker(h: *PeerTx) void {
        raise();
        const p = h.p;
        var last_gen: u32 = 0;
        var idle: usize = 0;
        while (!p.stop.load(.acquire)) {
            const g = h.gen.load(.acquire);
            if (g != last_gen) {
                last_gen = g;
                idle = 0;
                if (!p.failed.load(.acquire)) p.sendOnePeer(h.dst_tp, p.tx_x, p.tx_len, p.tx_packed);
                h.done.store(g, .release);
            } else spinOrSleep(&idle, 1_000_000);
        }
    }

    /// Relay thread: watches GPU-posted TP/PP sends and fires parallel RDMA write-signals across all TB5 links.
    fn service(p: *Pair) void {
        raise();
        const ep = p.ep orelse return;
        var next_x: u32 = 1;
        var next_pipe: u32 = 1;
        var next_pick: u32 = 1;
        var idle: usize = 0;
        while (!p.stop.load(.acquire)) {
            var did_work = false;
            // 1. TP all-reduce exchange across all tp-1 peers in parallel (one dedicated sender thread per TB5 link).
            if (p.tp > 1) {
                const posted = @atomicLoad(u32, p.word32(p.l.sync() + layout_mod.SYNC_POSTED), .acquire);
                if (reached(posted, next_x) and reached(p.queued.load(.acquire), next_x)) {
                    did_work = true;
                    const skip_off = p.l.sync() + layout_mod.SYNC_SKIP_BASE + (@as(usize, next_x & 2047) * 4);
                    if (@atomicLoad(u32, p.word32(skip_off), .acquire) == next_x) {
                        next_x +%= 1;
                        continue;
                    }
                    const raw_len = p.bytes[next_x % JOBS];
                    const gpu_packed = (raw_len & GPU_PACKED_BIT) != 0;
                    const len = raw_len & ~GPU_PACKED_BIT;
                    const t0 = platform.c.mach_absolute_time();
                    if (!p.failed.load(.acquire)) {
                        if (p.tp == 2) {
                            p.sendOnePeer(p.first_tp_peer, next_x, len, gpu_packed);
                        } else if (len <= p.inline_tx_max) {
                            for (1..p.tp) |step| {
                                p.sendOnePeer((p.tp_rank + @as(u32, @intCast(step))) % p.tp, next_x, len, gpu_packed);
                            }
                        } else {
                            p.tx_x = next_x;
                            p.tx_len = len;
                            p.tx_packed = gpu_packed;
                            p.tx_gen +%= 1;
                            const g = p.tx_gen;
                            for (&p.peer_tx) |*h| if (h.thread != null) h.gen.store(g, .release);
                            p.sendOnePeer(p.first_tp_peer, next_x, len, gpu_packed);
                            for (&p.peer_tx) |*h| {
                                if (h.thread == null) continue;
                                while (h.done.load(.acquire) != g and !p.stop.load(.acquire)) std.atomic.spinLoopHint();
                            }
                        }
                    }
                    const dt = platform.c.mach_absolute_time() - t0;
                    _ = p.exchanges_total.fetchAdd(1, .monotonic);
                    _ = p.bytes_sent_total.fetchAdd(@as(u64, len) * (p.tp - 1), .monotonic);
                    _ = p.rdma_send_ns_total.fetchAdd(dt * 1000 / 24, .monotonic);
                    if (p.failed.load(.acquire)) p.markAllArrived(next_x, next_pipe, next_pick);
                    next_x +%= 1;
                }
            }
            // 2. PP hidden-state send to next pipeline stage.
            if (p.next_stage_peer) |peer_id| {
                const posted = @atomicLoad(u32, p.word32(p.l.sync() + layout_mod.SYNC_PIPE_POSTED), .acquire);
                if (reached(posted, next_pipe) and reached(p.pipe_queued.load(.acquire), next_pipe)) {
                    did_work = true;
                    const len = p.pipe_bytes[next_pipe % JOBS];
                    if (!p.failed.load(.acquire)) {
                        ep.writeSignalFrom(peer_id, p.l.pipeSend(next_pipe), p.l.pipeRecv(next_pipe), len, p.l.flags() + PIPE_FLAG, next_pipe) catch |err| {
                            std.log.err("mesh rank {d}: PP send {d} to peer {d} failed: {s}", .{ p.rank, next_pipe, peer_id, @errorName(err) });
                            p.failed.store(true, .release);
                        };
                    }
                    _ = p.bytes_sent_total.fetchAdd(len, .monotonic);
                    if (p.failed.load(.acquire)) p.markAllArrived(next_x, next_pipe, next_pick);
                    next_pipe +%= 1;
                }
            }
            // 3. PP final-stage token picks return to rank 0.
            if (p.pp > 1 and p.rank == p.last_stage_rank0) {
                const posted = @atomicLoad(u32, p.word32(p.l.sync() + layout_mod.SYNC_PICK_POSTED), .acquire);
                if (reached(posted, next_pick) and reached(p.pick_queued.load(.acquire), next_pick)) {
                    did_work = true;
                    const len = p.pick_bytes[next_pick % JOBS];
                    if (!p.failed.load(.acquire)) {
                        ep.writeSignalFrom(0, p.l.pickSend(next_pick), p.l.pickRecv(next_pick), len, p.l.flags() + PICK_FLAG, next_pick) catch |err| {
                            std.log.err("mesh rank {d}: PP picks {d} to rank 0 failed: {s}", .{ p.rank, next_pick, @errorName(err) });
                            p.failed.store(true, .release);
                        };
                    }
                    if (p.failed.load(.acquire)) p.markAllArrived(next_x, next_pipe, next_pick);
                    next_pick +%= 1;
                }
            }
            if (did_work) {
                idle = 0;
            } else spinOrSleep(&idle, 1_000_000);
        }
    }
};

test {
    _ = layout_mod;
}
