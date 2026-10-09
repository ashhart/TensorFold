//! N Macs' partial-sum exchange inside the command buffer: the GPU writes its partial into its own slot of the
//! registered window and posts a sequence word; a host thread sends that slot to every peer's same slot with one
//! write-and-signal each; the GPU spins until every peer's flag reaches the sequence, then sums slots 0..N-1 in order.
const std = @import("std");
const mtl = @import("metal");
const fabric = @import("fabric");
const m = @import("model.zig");
const Ref = m.Ref;

const room = 256; // ahead of each slot's data: the library's head room for a zero-copy write-and-signal
const arg_bytes = 1024;
/// Slot sets (by seq % ring): a peer can run ahead of this rank by every exchange it can post before needing one of
/// ours, which is 1 for one block at a time.
const ring_sets: u32 = 2;
fn envOn(name: [*:0]const u8) bool {
    return if (std.c.getenv(name)) |v| v[0] == '1' else false;
}
/// The start flag's word: every setting that changes the exchange count, the window layout or the draft counts, which
/// must be the same on every rank (a mismatch would hang or misread slots): the ring, max_rows, G53_OWN_FRONT, G53_KSPLIT
/// (+_MIN), G53_LSPLIT (+_FROM, _OWNER), G53_COPY and G53_COPY_ROW.
fn cfgWord(max_rows: usize) u64 {
    var h = std.hash.Wyhash.init(0x6735_3363_6667);
    h.update(std.mem.asBytes(&ring_sets));
    h.update(std.mem.asBytes(&max_rows)); // sets every slot offset a peer writes to
    h.update(&.{ @intFromBool(envOn("G53_OWN_FRONT")), @intFromBool(envOn("G53_KSPLIT")) });
    if (envOn("G53_LSPLIT")) { // the layer-split cache adds exchanges past G53_LSPLIT_FROM
        h.update("lsplit");
        h.update(if (std.c.getenv("G53_LSPLIT_FROM")) |z| std.mem.span(z) else "0");
        h.update(if (std.c.getenv("G53_LSPLIT_OWNER")) |z| std.mem.span(z) else "rr");
    }
    for ([_][*:0]const u8{ "G53_KSPLIT_MIN", "G53_COPY", "G53_COPY_ROW" }) |n| {
        const v: []const u8 = if (std.c.getenv(n)) |z| std.mem.span(z) else "";
        h.update(v);
        h.update(&.{0}); // a separator: values cannot run into each other
    }
    return h.final() | 1;
}

fn slotBytes(max_rows: usize) usize {
    return std.mem.alignForward(usize, room + max_rows * m.D * 4, 16384);
}

pub const Kind = enum(u8) { partial, arg, peers }; // peers: a different payload a peer, from its send region

pub const Exchange = struct {
    ep: *fabric.mcdma.Endpoint,
    rd: fabric.Rdma,
    wbuf: mtl.Buffer, // the window as a Metal buffer, no copy
    win: []u8,
    rank: u32,
    ranks: u32,
    slots_at: usize,
    args_at: usize,
    flags_at: usize,
    sync_at: usize,
    slot_bytes: usize,
    sends_at: usize = 0, // big partials: a copy per peer (index j), so each zero-copy send has its own head room
    kinds: [1024]Kind = @splat(.partial),
    lens: [1024]usize = @splat(0),
    sent: u64 = 0,
    ring: u32 = ring_sets,
    conc: bool = false, // the engine's encoder is concurrent: barriers around the post and the wait
    stall_s: f64 = 5.0, // G53_STALL_S: print a stalled wait after this long
    xchunk: usize = 0, // G53_XCHUNK=<bytes> (0 = off): see budget()
    xlimit: usize = 4 << 20, // G53_XLIMIT: most unfenced bytes on one link (half of TB_RING's 8 MiB)
    unfenced: [64]usize = @splat(0),
    fences: u64 = 0,

    pub fn windowBytes(ranks: u32, max_rows: usize, ring: u32) usize {
        const n = ring * ranks * slotBytes(max_rows) + ring * ranks * arg_bytes + 16384 + 16384 + ring * (ranks - 1) * slotBytes(max_rows);
        return std.mem.alignForward(usize, n, fabric.mcdma.alignment);
    }

    pub fn create(gpa: std.mem.Allocator, device: mtl.Device, library: [:0]const u8, rank: u32, ranks: u32, links: []const fabric.mcdma.Link, max_rows: usize) !*Exchange {
        const ring = ring_sets;
        const wb = windowBytes(ranks, max_rows, ring);
        const cfg = cfgWord(max_rows);
        const slot_bytes = slotBytes(max_rows);
        const ep = try fabric.mcdma.Endpoint.create(gpa, library, .{ .rank = rank, .ranks = ranks, .window_bytes = wb, .staging_bytes = std.mem.alignForward(usize, slot_bytes + (1 << 20), 16384), .links = links, .timeout_ns = 60 * std.time.ns_per_s, .connect_timeout_ns = 300 * std.time.ns_per_s });
        errdefer ep.deinit();
        const rd = ep.rdma();
        const win = rd.window(); // zeroed by Endpoint.create; never clear it here: peers' signals may already have landed
        const x = try gpa.create(Exchange);
        x.* = .{ .ep = ep, .rd = rd, .win = win, .wbuf = try device.bufferNoCopy(win.ptr, win.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked), .rank = rank, .ranks = ranks, .slot_bytes = slot_bytes, .ring = ring, .slots_at = 0, .args_at = ring * ranks * slot_bytes, .flags_at = ring * ranks * slot_bytes + ring * ranks * arg_bytes, .sync_at = 0 };
        x.sync_at = x.flags_at + 16384;
        x.sends_at = x.sync_at + 16384;
        if (std.c.getenv("G53_XCHUNK")) |v| x.xchunk = std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0;
        if (std.c.getenv("G53_XLIMIT")) |v| x.xlimit = std.fmt.parseInt(usize, std.mem.span(v), 10) catch (4 << 20);
        if (x.xchunk > 0) x.xchunk = @max(4096, x.xchunk / 4096 * 4096);
        if (std.c.getenv("G53_STALL_S")) |v| x.stall_s = std.fmt.parseFloat(f64, std.mem.span(v)) catch 5.0;
        // every rank up before any step: a start flag each way, re-sent each second until every peer's has landed
        var tries: u32 = 0;
        while (true) : (tries += 1) {
            for (0..ranks) |p| if (p != rank) try rd.signal(@intCast(p), x.flags_at + 8 * 64 + 8 * rank, cfg);
            rd.flush() catch |err| std.debug.print("glm53: start flush: {t}\n", .{err});
            const t0 = mtl.clock.seconds();
            var all = false;
            while (!all and mtl.clock.seconds() - t0 < 1.0) {
                all = true;
                for (0..ranks) |p| if (p != rank) {
                    const w: *const u64 = @ptrCast(@alignCast(win.ptr + x.flags_at + 8 * 64 + 8 * p));
                    const got = @atomicLoad(u64, w, .acquire);
                    if (got != 0 and got != 1 and got != cfg) {
                        std.debug.print("glm53: rank {d}: rank {d} started with different exchange settings (G53_OWN_FRONT/KSPLIT/LSPLIT/COPY, max_rows or binary) - refusing\n", .{ rank, p });
                        return error.ConfigMismatch;
                    }
                    if (got != cfg) all = false;
                };
            }
            if (all) break;
            var missing: [64]u8 = undefined;
            var nm: usize = 0;
            for (0..ranks) |p| if (p != rank) {
                const w: *const u64 = @ptrCast(@alignCast(win.ptr + x.flags_at + 8 * 64 + 8 * p));
                if (@atomicLoad(u64, w, .acquire) != cfg and nm < 60) {
                    missing[nm] = '0' + @as(u8, @intCast(p));
                    nm += 1;
                }
            };
            if (tries % 5 == 0) std.debug.print("glm53: rank {d} waiting for start flags from ranks {s}\n", .{ rank, missing[0..nm] });
            if (tries > 120) return error.StartTimeout;
        }
        return x;
    }

    pub fn deinit(x: *Exchange, gpa: std.mem.Allocator) void {
        x.rd.flush() catch {};
        x.wbuf.deinit();
        x.ep.deinit();
        gpa.destroy(x);
    }

    fn ref(x: *Exchange, off: usize) Ref {
        return .{ .buf = x.wbuf, .off = off };
    }

    /// Slot 0's data for exchange `seq` (slots `slotFloats` apart).
    pub fn slot(x: *Exchange, seq: u32) Ref {
        return x.ref(x.slots_at + (seq % x.ring) * x.ranks * x.slot_bytes + room);
    }
    pub fn slotFloats(x: *Exchange) usize {
        return x.slot_bytes / 4;
    }
    pub fn argBase(x: *Exchange, seq: u32) Ref {
        return x.ref(x.args_at + (seq % x.ring) * x.ranks * arg_bytes + room);
    }
    pub fn argSlot(x: *Exchange, seq: u32) Ref {
        return x.argBase(seq).at(x.rank * arg_bytes);
    }
    pub fn argFloats(x: *Exchange) usize {
        _ = x;
        return arg_bytes / 4;
    }

    /// Peer p's send region for exchange `seq` (the GPU writes p's own payload there; encodePeers sends it).
    pub fn sendRef(x: *Exchange, seq: u32, p: usize) Ref {
        const j: usize = if (p < x.rank) p else p - 1;
        return x.ref(x.sendRegion(seq, j));
    }
    /// Exchange `seq` with a payload a peer (`len` bytes each, written to sendRef(seq, p) before this);
    /// each peer receives its own payload in this rank's slot.
    pub fn encodePeers(x: *Exchange, enc: mtl.ComputeEncoder, post: mtl.Pipeline, wait: mtl.Pipeline, seq: u32, len: usize) void {
        x.encodeKind(enc, post, wait, seq, .peers, len);
    }

    /// Partials at least this big go zero-copy from per-peer copies; smaller ones are staged.
    pub const zero_copy_min: usize = 256 << 10;

    fn sendRegion(x: *Exchange, seq: u32, j: usize) usize {
        return x.sends_at + ((seq % x.ring) * (x.ranks - 1) + j) * x.slot_bytes + room;
    }

    /// Copy this rank's partial into every per-peer send region (big partials only), before the post.
    pub fn encodeCopies(x: *Exchange, enc: mtl.ComputeEncoder, copy: mtl.Pipeline, seq: u32, len: usize) void {
        if (len < zero_copy_min) return;
        if (x.conc) enc.barrier(); // the partial is written before it is copied
        const mine = x.slots_at + ((seq % x.ring) * x.ranks + x.rank) * x.slot_bytes + room;
        const n: i32 = @intCast(len / 4);
        for (0..x.ranks - 1) |j| {
            enc.setPipeline(copy);
            enc.setBuffer(x.wbuf, mine, 0);
            enc.setBuffer(x.wbuf, x.sendRegion(seq, j), 1);
            enc.setBytes(std.mem.asBytes(&n), 2);
            enc.dispatchThreads(mtl.Size.of(@intCast(n), 1, 1), mtl.Size.of(256, 1, 1));
        }
    }

    fn encodeKind(x: *Exchange, enc: mtl.ComputeEncoder, post: mtl.Pipeline, wait: mtl.Pipeline, seq: u32, kind: Kind, len: usize) void {
        x.kinds[seq % x.kinds.len] = kind;
        x.lens[seq % x.lens.len] = len;
        const word: u32 = seq + 1;
        if (x.conc) enc.barrier(); // the partial (or arg pair) is written before the post
        enc.setPipeline(post);
        enc.setBuffer(x.wbuf, x.sync_at, 0);
        enc.setBytes(std.mem.asBytes(&word), 1);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
        if (x.conc) enc.barrier();
        enc.setPipeline(wait);
        enc.setBuffer(x.wbuf, x.flags_at, 0);
        enc.setBuffer(x.wbuf, x.sync_at + 1024, 1);
        const a = [3]u32{ word, x.ranks, x.rank };
        enc.setBytes(std.mem.asBytes(&a), 2);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
        if (x.conc) enc.barrier(); // every peer's slot has landed before the sum reads it
    }

    /// The post alone (G53_FWAIT: the wait rides in the consuming launch, see waitBind).
    pub fn encodePost(x: *Exchange, enc: mtl.ComputeEncoder, post: mtl.Pipeline, seq: u32, len: usize) void {
        x.kinds[seq % x.kinds.len] = .partial;
        x.lens[seq % x.lens.len] = len;
        const word: u32 = seq + 1;
        if (x.conc) enc.barrier(); // the partial is written before the post
        enc.setPipeline(post);
        enc.setBuffer(x.wbuf, x.sync_at, 0);
        enc.setBytes(std.mem.asBytes(&word), 1);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
        if (x.conc) enc.barrier();
    }

    /// The wait's buffers for a consuming launch: flags at `first`, give-ups at `first + 1`, {seq, ranks, me} at `first + 2`.
    pub fn waitBind(x: *Exchange, enc: mtl.ComputeEncoder, seq: u32, first: usize) void {
        enc.setBuffer(x.wbuf, x.flags_at, first);
        enc.setBuffer(x.wbuf, x.sync_at + 1024, first + 1);
        const a = [3]u32{ seq + 1, x.ranks, x.rank };
        enc.setBytes(std.mem.asBytes(&a), first + 2);
    }

    pub fn encode(x: *Exchange, enc: mtl.ComputeEncoder, post: mtl.Pipeline, wait: mtl.Pipeline, seq: u32, len: usize) void {
        x.encodeKind(enc, post, wait, seq, .partial, len);
    }
    pub fn encodeArgs(x: *Exchange, enc: mtl.ComputeEncoder, post: mtl.Pipeline, wait: mtl.Pipeline, seq: u32) void {
        x.encodeKind(enc, post, wait, seq, .arg, 8);
    }
    /// `len` bytes of (value, index) pairs (a pair a row; at most arg_bytes - room).
    pub fn encodeArgsN(x: *Exchange, enc: mtl.ComputeEncoder, post: mtl.Pipeline, wait: mtl.Pipeline, seq: u32, len: usize) void {
        std.debug.assert(len <= arg_bytes - room);
        x.encodeKind(enc, post, wait, seq, .arg, len);
    }

    /// Send exchange `s` (this rank's slot, or arg pair) to peer `p`: data then the flag word s+1, in one write-and-signal.
    /// G53_XCHUNK: before a send that would put more than xlimit bytes in flight on peer p's link, fence that link (the
    /// peer has applied everything before it); zero-copy partials go in pieces of xchunk bytes (the pieces signal a
    /// scratch word, the last one the real flag; one queue pair keeps them in order). Without it, a 6.3 MB prompt partial
    /// arriving while the 8 MiB receive ring refilled was dropped without an error, and every GPU spun in g53_wait.
    fn budget(x: *Exchange, p: u32, n: usize) !void {
        if (x.xchunk == 0) return;
        if (x.unfenced[p] + n > x.xlimit) {
            try x.ep.flushPeer(p);
            x.unfenced[p] = 0;
            x.fences += 1;
        }
        x.unfenced[p] += n;
    }

    fn fenceNow(x: *Exchange, p: u32) !void {
        try x.ep.flushPeer(p);
        x.unfenced[p] = 0;
        x.fences += 1;
    }

    fn sendOne(x: *Exchange, p: u32, s: u32) !void {
        const word = s + 1;
        const kind = x.kinds[s % x.kinds.len];
        const off = switch (kind) {
            .partial, .peers => x.slots_at + ((s % x.ring) * x.ranks + x.rank) * x.slot_bytes + room,
            .arg => x.args_at + ((s % x.ring) * x.ranks + x.rank) * arg_bytes + room,
        };
        const len = x.lens[s % x.lens.len];
        if (kind == .peers and len < zero_copy_min) { // a small per-peer payload: staged from its send region
            const j: usize = if (p < x.rank) p else p - 1;
            const src = x.sendRegion(s, j);
            try x.budget(p, len + 64);
            return x.rd.write2Signal(p, off, &.{}, x.win[src..][0..len], x.flags_at + 8 * x.rank, word);
        }
        if ((kind == .partial or kind == .peers) and len >= zero_copy_min) {
            const j: usize = if (p < x.rank) p else p - 1;
            const src = x.sendRegion(s, j);
            var done: usize = 0;
            // a piece's head goes into the 256 bytes before it = the previous piece's tail: fence before every later piece
            if (x.xchunk > 0) while (len - done > x.xchunk) : (done += x.xchunk) {
                if (done > 0) try x.fenceNow(p);
                try x.budget(p, x.xchunk);
                try x.ep.writeSignalFrom(p, src + done, off + done, x.xchunk, x.flags_at + 8 * (256 + x.rank), word);
            };
            if (done > 0) try x.fenceNow(p);
            try x.budget(p, len - done);
            return x.ep.writeSignalFrom(p, src + done, off + done, len - done, x.flags_at + 8 * x.rank, word);
        }
        try x.budget(p, len + 64);
        return x.rd.write2Signal(p, off, &.{}, x.win[off..][0..len], x.flags_at + 8 * x.rank, word);
    }

    fn flagOf(x: *Exchange, p: usize) u64 {
        const fw: *const u64 = @ptrCast(@alignCast(x.win.ptr + x.flags_at + 8 * p));
        return @atomicLoad(u64, fw, .acquire);
    }

    /// A wait that lasts too long: after `stall_s` print what this rank sees, once a wait, so a stuck exchange names its
    /// rank and peer instead of hanging silently.
    fn stallTick(x: *Exchange, p: u32, what: []const u8, s: u32, t0: f64, printed: *bool) void {
        const now = mtl.clock.seconds();
        if (!printed.* and now - t0 > x.stall_s) {
            printed.* = true;
            const sync: *const u32 = @ptrCast(@alignCast(x.win.ptr + x.sync_at));
            var buf: [256]u8 = undefined;
            var n: usize = 0;
            for (0..x.ranks) |q| if (q != x.rank) {
                const w = std.fmt.bufPrint(buf[n..], " flag[{d}]={d}", .{ q, x.flagOf(q) }) catch break;
                n += w.len;
            };
            std.debug.print("glm53 STALL rank {d} peer-thread {d}: waiting {s} for exchange {d} (word {d}) {d:.1} s; gpu post word {d};{s}; gaveup {d}\n", .{ x.rank, p, what, s, s + 1, now - t0, @atomicLoad(u32, sync, .acquire), buf[0..n], x.gaveUp() });
        }
    }

    /// Serve exchanges [first, last) to one peer: wait for the GPU's post, send this rank's slot (staged per peer).
    fn servePeer(x: *Exchange, p: u32, first: u32, last: u32, failed: *std.atomic.Value(bool)) void {
        const sync: *const u32 = @ptrCast(@alignCast(x.win.ptr + x.sync_at));
        var s = first;
        while (s < last) : (s += 1) {
            const word = s + 1;
            if (@as(i32, @bitCast(@atomicLoad(u32, sync, .acquire) -% word)) < 0) {
                const t0 = mtl.clock.seconds();
                var printed = false;
                var spins: u32 = 0;
                while (@as(i32, @bitCast(@atomicLoad(u32, sync, .acquire) -% word)) < 0) {
                    std.atomic.spinLoopHint();
                    spins +%= 1;
                    // our GPU has not posted s: it may be waiting for a peer's s-1, or for OUR s-1 to land somewhere
                    if (spins & 0xffff == 0) x.stallTick(p, "for our GPU's post", s, t0, &printed);
                }
            }
            x.sendOne(p, s) catch {
                failed.store(true, .release);
                return;
            };
        }
    }

    /// Serve exchanges [first, last): a thread a peer (staging is per peer), so sends to different peers overlap.
    pub fn serve(x: *Exchange, first: u32, last: u32) !void {
        var failed = std.atomic.Value(bool).init(false);
        var threads: [64]?std.Thread = @splat(null);
        var self_p: ?u32 = null;
        for (0..x.ranks) |pi| {
            const p: u32 = @intCast(pi);
            if (p == x.rank) continue;
            if (self_p == null) { // the calling thread serves the first peer itself
                self_p = p;
                continue;
            }
            threads[pi] = std.Thread.spawn(.{}, servePeer, .{ x, p, first, last, &failed }) catch null;
            if (threads[pi] == null) servePeer(x, p, first, last, &failed);
        }
        if (self_p) |p| servePeer(x, p, first, last, &failed);
        for (threads) |t| if (t) |th| th.join();
        if (failed.load(.acquire)) return error.PeerDown;
    }

    pub fn gaveUp(x: *Exchange) u32 {
        return @as(*const u32, @ptrCast(@alignCast(x.win.ptr + x.sync_at + 1024))).*;
    }
};
