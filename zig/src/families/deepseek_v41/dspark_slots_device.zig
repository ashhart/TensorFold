//! SlotPass on the device: rounds started from the row window's pick, graphed rounds, the passes' glue steps.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");
const emit = @import("dspark_emit.zig");
const rows = @import("dspark_rows.zig");
const gpu = @import("dspark_gpu.zig");
const iface = @import("draft/iface.zig");
const dspark = @import("draft/dspark.zig");
const dsd = @import("dspark_dev.zig");
const ph = @import("phases.zig");
const core = @import("dspark_slots.zig");
const SlotPass = core.SlotPass;
const self = core.SlotPass.self;
const propose = core.SlotPass.propose;
const bySlot = core.SlotPass.bySlot;
const Begun = core.SlotPass.Begun;
const scr_ids = core.SlotPass.scr_ids;
const scr_picks = core.SlotPass.scr_picks;
const scr_segs = core.SlotPass.scr_segs;
const scr_acc = core.SlotPass.scr_acc;
const scr_mem = core.SlotPass.scr_mem;
const scr_len = core.SlotPass.scr_len;

/// The device path is on and past its checks.
pub fn devReady(x: *const SlotPass) bool {
    const d = x.dev orelse return false;
    return d.on() and d.stats.passes >= dsd.check_passes;
}

/// Pass.arm (rank 0): the slot's coming row window; followers get op 43 [op, slot, start, rows, sampled] first.
pub fn arm(p: *anyopaque, slot: u32, start: u64, n_rows: u32, sampled: bool) void {
    const x = self(p);
    if (x.dl != null) x.dropDevice();
    if (x.bat == null or slot >= x.nslots or !x.devReady()) return;
    if (sampled and x.f.after_sample == null) return;
    x.sendOp(&.{ gpu.GpuPass.op_arm, slot, @intCast(start), n_rows, @intFromBool(sampled) }) catch |e| {
        std.log.err("slot drafts: the arm did not reach the followers ({t})", .{e});
        return;
    };
    x.armed[slot] = .{ .start = start, .n = n_rows, .sampled = sampled };
}

pub fn dropDevice(x: *SlotPass) void {
    if (x.dl != null) x.dstats.device_dropped += 1;
    x.dl = null;
    x.armed = @splat(null);
}

/// After a row window's GPU pick (every rank): picks into the device choices; the round starts if all slots chose.
pub fn afterPick(ctx: *anyopaque, pick: fwd.Forward.PickDevice) anyerror!void {
    const x: *SlotPass = @ptrCast(@alignCast(ctx));
    const b = x.bat orelse return;
    if (pick.chain or pick.n != b.n or pick.n > 64) return;
    const r = x.f.runner;
    try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.dscr.?.ptr + 8 * scr_picks, pick.out, 8 * @as(usize, pick.n), r.stream.handle), "cuMemcpyDtoDAsync");
    for (&x.armed) |*am| if (am.*) |*a| if (!a.sampled) {
        a.chosen = true;
    };
    return x.tryLaunch();
}

/// After a sampled call's device choice (every rank): a sampled armed slot gets its rows; the round may start.
pub fn afterSample(ctx: *anyopaque, row0: u32, n: u32, chosen: u64) anyerror!void {
    const x: *SlotPass = @ptrCast(@alignCast(ctx));
    const b = x.bat orelse return;
    for (&x.armed, 0..) |*am, sl| if (am.*) |*a| if (a.sampled) {
        const pd = b.pend[sl] orelse continue;
        if (pd.stashed or pd.a != row0 or pd.n != n or pd.a + pd.n > 64) continue;
        const r = x.f.runner;
        try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.dscr.?.ptr + 8 * (scr_picks + pd.a), chosen, 8 * @as(usize, n), r.stream.handle), "cuMemcpyDtoDAsync");
        a.chosen = true;
    };
    return x.tryLaunch();
}

/// The round from the window's device choices once every armed slot has its choice: accept, ingest, stage, pass.
pub fn tryLaunch(x: *SlotPass) !void {
    const b = x.bat orelse return;
    if (x.begun != null or !x.devReady()) return;
    var slots: [16]u32 = undefined;
    var k: usize = 0;
    for (x.armed[0..x.nslots], 0..) |am, sl| if (am) |a| {
        const pd = b.pend[sl] orelse continue;
        if (pd.stashed or pd.start != a.start or pd.n != a.n or pd.a + pd.n > b.n or a.n > x.one.ingest_rows) continue;
        if (!a.chosen) return; // a slot of this forward still waits for its choice
        slots[k] = @intCast(sl);
        k += 1;
    };
    if (k == 0) return;
    const d = x.dev.?;
    const f = x.f;
    const r = f.runner;
    const n: u32 = f.cfg.dspark_block;
    if (x.capture_first and d.active == .graphs) {
        // TF_DSV41_DRAFT_CAPTURE: a group whose pass no regular round has issued yet: the regular path this round
        var gs0: [16][2]usize = undefined;
        for (gs0[0..rows.groups(k, x.group, &gs0)]) |g| if (!x.wasSeen(passKey(g[1] - g[0], g[0] * n))) {
            x.armed = @splat(null);
            x.declined += 1;
            return;
        };
    }
    x.armed = @splat(null);
    // the accept's inputs: the armed slots' window ids at their rows, their (first row, rows, start)
    const h: []i64 = @alignCast(std.mem.bytesAsSlice(i64, x.hscr.?.bytes[0 .. 8 * scr_len]));
    for (slots[0..k], 0..) |sl, j| {
        const pd = b.pend[sl].?;
        for (pd.ids[0..pd.n], 0..) |t, i| h[scr_ids + pd.a + i] = t;
        h[scr_segs + 3 * j ..][0..3].* = .{ pd.a, pd.n, @intCast(pd.start) };
    }
    const dev = x.dscr.?.ptr;
    // ids, then segs (the picks between them are the device's, written above)
    try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = dev, .len = 8 * scr_picks }, 0, std.mem.sliceAsBytes(h[0..scr_picks]), r.stream.handle);
    try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = dev + 8 * scr_segs, .len = 8 * (scr_acc - scr_segs) }, 0, std.mem.sliceAsBytes(h[scr_segs..scr_acc]), r.stream.handle);
    const ops = r.kernels.others(r.stream);
    try ops.dsAcceptRows(dev + 8 * scr_picks, dev + 8 * scr_ids, dev + 8 * scr_segs, k, dev + 8 * scr_acc);
    try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(x.hscr.?.bytes.ptr + 8 * scr_acc, dev + 8 * scr_acc, 8 * 3 * k, r.stream.handle), "cuMemcpyDtoHAsync");
    try x.staged.?.record(r.stream);
    // each slot's whole window into its rings: overlapped from the taps mark when recorded, else here in the stream
    if (x.side != null and b.marked) {
        try x.ingestRound(slots[0..k], true);
    } else if (x.bproj != .off) {
        try x.ingestRound(slots[0..k], false);
        if (x.side) |*sd| sd.serial += 1;
    } else {
        for (slots[0..k]) |sl| {
            const pd = b.pend[sl].?;
            try x.ingestAsync(sl, pd.start, .{ .src = 0, .skip = pd.a }, pd.n);
        }
        if (x.side) |*sd| sd.serial += 1;
    }
    // the groups as devLaunchRound forms them (slots sorted, rows.groups), each staged on the device and launched
    var gs: [16][2]usize = undefined;
    const ng = rows.groups(k, x.group, &gs);
    const W: u32 = f.cfg.window;
    const ring: u32 = @intCast(emit.ringRows(f.cfg));
    const s64 = r.addressOf("w.ds.i64") orelse return error.Unbound;
    const s32 = r.addressOf("w.ds.i32") orelse return error.Unbound;
    for (gs[0..ng]) |g| {
        var ms: [16]rows.Member = undefined;
        const gk = g[1] - g[0];
        const mem = h[scr_mem + 48 * (g[0] % 16) ..][0 .. 3 * gk];
        for (slots[g[0]..g[1]], ms[0..gk], 0..) |sl, *m, i| {
            m.* = .{ .slot = sl, .anchor = 0, .start = b.pend[sl].?.start, .valid = x.valid[sl] };
            mem[3 * i ..][0..3].* = .{ @intCast(g[0] + i), sl, @intCast(x.valid[sl]) };
        }
        const lay = emit.Layout.of(@intCast(gk), n, W);
        const at = g[0] * n;
        const sv = try d.statics(at / n, @intCast(lay.len64), @intCast(lay.len32));
        rows.stage(lay, n, W, ring, x.one.shape.noise, ms[0..gk], sv[0], sv[1]);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = s64, .len = 8 * @as(usize, @intCast(lay.len64)) }, 0, std.mem.sliceAsBytes(sv[0]), r.stream.handle);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = s32, .len = 4 * @as(usize, @intCast(lay.end32)) }, 0, std.mem.sliceAsBytes(sv[1][0..@intCast(lay.end32)]), r.stream.handle);
        const moff = 8 * (scr_mem + 48 * (g[0] % 16));
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = dev + moff, .len = 8 * 3 * gk }, 0, std.mem.sliceAsBytes(mem), r.stream.handle);
        try ops.dsStageRows(dev + 8 * scr_acc, dev + moff, gk, s64, s32, n, W, ring, x.one.shape.noise, .{ lay.ids, lay.positions, lay.tokens, lay.counts, lay.lo, lay.hi });
        x.stats.passes += 1;
        x.stats.slots += gk;
        x.cur_at = at;
        x.cur_k = gk;
        r.glue = .{ .ctx = x, .run = glueFn };
        const extra = [_]u64{ if (x.send) |sb| sb.ptr else 0, if (x.send) |sb| sb.len else 0 };
        try d.issueReplay(passKey(gk, at), dsd.fingerprint(r, d, &extra), .{ .ctx = x, .run = groupBody });
        try d.settle(); // counted here on every rank (rank 0's collect does not count it again)
    }
    try d.mark();
    if (x.f.comm.rank() == 0 or x.f.link == null) {
        x.dl = .{ .k = k, .slots = undefined };
        @memcpy(x.dl.?.slots[0..k], slots[0..k]);
    }
}

pub fn devOn(x: *const SlotPass) bool {
    const d = x.dev orelse return false;
    return d.on();
}

/// The speculative pass over several slots: every group launched (device path), collected by `collect`.
pub fn begin(p: *anyopaque, asks: []const iface.Ask) anyerror!void {
    const x = self(p);
    if (asks.len == 0 or asks.len > x.nslots) return error.Slots;
    if (x.begun != null) return error.PassBegun;
    @memcpy(x.begun_asks[0..asks.len], asks);
    if (x.dl != null and try x.takeDevice(x.begun_asks[0..asks.len])) return;
    if (!x.devOn()) {
        // the checks turned the device path off: the pass runs at `collect` (exact; only the overlap is lost)
        x.begun = .{ .n = asks.len, .groups = undefined, .ng = 0 };
        x.begun_host = true;
        return;
    }
    try x.devLaunchRound(x.begun_asks[0..asks.len]);
}

/// `begin` over the device-launched round when its slots and (bonus, next position) match the asks; else false.
pub fn takeDevice(x: *SlotPass, asks: []const iface.Ask) !bool {
    const dl = x.dl.?;
    x.dl = null;
    for (x.order[0..asks.len], 0..) |*o, i| o.* = i;
    std.sort.insertion(usize, x.order[0..asks.len], asks, bySlot);
    var ok = asks.len == dl.k;
    if (ok) {
        try x.staged.?.synchronize(); // the accept's copy back (not the passes)
        const h: []const i64 = @alignCast(std.mem.bytesAsSlice(i64, x.hscr.?.bytes[0 .. 8 * scr_len]));
        for (x.order[0..asks.len], 0..) |i, j| {
            const a = asks[i];
            if (a.slot != dl.slots[j] or h[scr_acc + 3 * j + 1] != a.anchor or h[scr_acc + 3 * j + 2] != @as(i64, @intCast(a.start))) ok = false;
        }
    }
    if (!ok) {
        x.dstats.device_dropped += 1;
        return false;
    }
    var bg: Begun = .{ .n = asks.len, .groups = undefined, .ng = 0 };
    bg.ng = rows.groups(asks.len, x.group, &bg.groups);
    x.begun = bg;
    x.begun_dev = true;
    x.dstats.device_starts += 1;
    // the chain after the round's passes (enqueued at the pick), the round's event re-recorded after it
    if (try x.launchChain(asks)) try x.dev.?.mark();
    return true;
}

pub fn collect(p: *anyopaque, out: []iface.Proposal) anyerror!void {
    const x = self(p);
    const b = x.begun orelse return error.NoPassBegun;
    if (out.len != b.n) return error.Slots;
    if (x.begun_host) {
        x.begun = null;
        x.begun_host = false;
        errdefer {
            x.begun = b;
            x.begun_host = true;
        }
        try propose(p, x.begun_asks[0..b.n], out);
    } else try x.devFinishRound(x.begun_asks[0..b.n], out);
    x.begun_dev = false;
}

/// Every group of the round on the device in slot order; during the check passes each is waited for and checked.
pub fn devLaunchRound(x: *SlotPass, asks: []const iface.Ask) !void {
    const d = x.dev.?;
    for (x.order[0..asks.len], 0..) |*o, i| o.* = i;
    std.sort.insertion(usize, x.order[0..asks.len], asks, bySlot);
    var b: Begun = .{ .n = asks.len, .groups = undefined, .ng = 0 };
    b.ng = rows.groups(asks.len, x.group, &b.groups);
    const n: usize = x.f.cfg.dspark_block;
    for (b.groups[0..b.ng], 0..) |g, gi| {
        var ms: [16]rows.Member = undefined;
        const idx = x.order[g[0]..g[1]];
        for (idx, ms[0..idx.len]) |i, *m| {
            const a = asks[i];
            if (a.slot >= x.nslots) return error.Slots;
            m.* = .{ .slot = a.slot, .anchor = a.anchor, .start = a.start, .valid = x.valid[a.slot] };
        }
        // the device path's op carries the group's first row of the round (its region and graph key), last
        var buf: [3 + 3 * 16]i64 = undefined;
        const msg = try rows.proposeMsg(ms[0..idx.len], buf[0 .. 2 + 3 * 16]);
        buf[msg.len] = @intCast(g[0] * n);
        try x.sendOp(buf[0 .. msg.len + 1]);
        _ = gi;
        try x.devGroup(ms[0..idx.len], g[0] * n);
        if (d.stats.passes < dsd.check_passes) {
            try d.mark();
            try d.wait();
            try d.check(x.last_logits, x.last_v, idx.len * n, g[0] * n);
        }
        try d.settle();
    }
    _ = try x.launchChain(asks);
    try d.mark();
    x.begun = b;
}

/// TF_DSV41_DRAFT_CHAIN: the round's chain on the device after its passes, slot order; false: host chain later.
pub fn launchChain(x: *SlotPass, asks: []const iface.Ask) !bool {
    const ch = (x.dev orelse return false).chain orelse return false;
    var an: [16]u32 = undefined;
    var ps: [16]dspark.Params = undefined;
    for (x.order[0..asks.len], 0..) |i, j| {
        an[j] = asks[i].anchor;
        ps[j] = asks[i].params;
    }
    return ch.launch(x.f.runner, x.f.cfg, x.f.widths, x.f.opts, an[0..asks.len], ps[0..asks.len]);
}

/// The round's results: the event waited for, each group's candidates unpacked, each slot's chain.
pub fn devFinishRound(x: *SlotPass, asks: []const iface.Ask, out: []iface.Proposal) !void {
    const d = x.dev.?;
    const b = x.begun.?;
    {
        const tw = ph.start(x.f.phases, .@"draft.wait");
        defer tw.stop();
        try d.wait();
    }
    const n: usize = x.f.cfg.dspark_block;
    const c: usize = x.f.comm.world() * x.one.k;
    const D: usize = x.f.cfg.hidden;
    if (d.chain) |ch| if (ch.launched == b.n) {
        try x.chainResults(ch, asks, out, b);
        x.begun = null;
        return;
    };
    for (b.groups[0..b.ng]) |g| {
        const at = g[0] * n;
        const R = (g[1] - g[0]) * n;
        try d.unpack(R, at, x.f.cfg.vocab, x.cand[at * c ..], x.cval[at * c ..], if (x.one.conf != null) x.hid[at * D ..] else null);
    }
    var scratch = std.heap.ArenaAllocator.init(x.gpa);
    defer scratch.deinit();
    const tc = ph.start(x.f.phases, .@"draft.chain");
    defer tc.stop();
    // rows of the round are in sorted-slot order: the j-th sorted ask's rows at j x n
    for (x.order[0..b.n], 0..) |i, j| {
        const a = asks[i];
        const o = out[i];
        const cd = x.cand[j * n * c ..][0 .. n * c];
        const cv = x.cval[j * n * c ..][0 .. n * c];
        try dspark.chain(scratch.allocator(), x.one.markov, cd, cv, c, a.anchor, a.params, if (x.one.conf != null) x.hid[j * n * D ..][0 .. n * D] else null, o.drafts, o.conf);
        if (o.cand) |t| {
            if (t.len != cd.len) return error.CandidateWidth;
            @memcpy(t, cd);
        }
        if (o.base) |t| {
            if (t.len != cv.len) return error.CandidateWidth;
            @memcpy(t, cv);
        }
    }
    x.begun = null;
}

/// The device chain's drafts and confidences into each ask's proposal; a tree's candidates and base logits too.
pub fn chainResults(x: *SlotPass, ch: *dsd.Chain, asks: []const iface.Ask, out: []iface.Proposal, b: Begun) !void {
    const tc = ph.start(x.f.phases, .@"draft.chain");
    defer tc.stop();
    const n: usize = x.f.cfg.dspark_block;
    const c: usize = x.f.comm.world() * x.one.k;
    const res = ch.results();
    var trees = false;
    for (out) |o| trees = trees or o.cand != null or o.base != null;
    // the first rounds: the host's chain over the same candidates too (a near tie may differ by fp32 order)
    const checking = ch.stats.rounds <= dsd.Chain.check_rounds;
    const D: usize = x.f.cfg.hidden;
    if (trees or checking) for (b.groups[0..b.ng]) |g| {
        const at = g[0] * n;
        try x.dev.?.unpack((g[1] - g[0]) * n, at, x.f.cfg.vocab, x.cand[at * c ..], x.cval[at * c ..], if (checking and x.one.conf != null) x.hid[at * D ..] else null);
    };
    if (checking) {
        var scratch = std.heap.ArenaAllocator.init(x.gpa);
        defer scratch.deinit();
        var drafts: [64]u32 = undefined;
        var conf: [64]f32 = undefined;
        for (x.order[0..b.n], 0..) |i, j| {
            const a = asks[i];
            try dspark.chain(scratch.allocator(), x.one.markov, x.cand[j * n * c ..][0 .. n * c], x.cval[j * n * c ..][0 .. n * c], c, a.anchor, a.params, if (x.one.conf != null) x.hid[j * n * D ..][0 .. n * D] else null, drafts[0..n], conf[0..n]);
            for (drafts[0..n], res.draft[j * n ..][0..n]) |h, d| {
                ch.checked += 1;
                if (h != @as(u32, @intCast(d))) ch.differ += 1;
            }
        }
        if (ch.stats.rounds == dsd.Chain.check_rounds) ch.verdict();
    }
    for (x.order[0..b.n], 0..) |i, j| {
        const o = out[i];
        if (o.drafts.len != n or o.conf.len != n) return error.CandidateWidth;
        for (o.drafts, res.draft[j * n ..][0..n]) |*t, v| t.* = @intCast(v);
        if (ch.conf) @memcpy(o.conf, res.conf[j * n ..][0..n]) else @memset(o.conf, 0.5);
        if (o.cand) |t| {
            if (t.len != n * c) return error.CandidateWidth;
            @memcpy(t, x.cand[j * n * c ..][0 .. n * c]);
        }
        if (o.base) |t| {
            if (t.len != n * c) return error.CandidateWidth;
            @memcpy(t, x.cval[j * n * c ..][0 .. n * c]);
        }
    }
    ch.launched = 0;
}

/// One group on the device: statics from its pinned region (no sync), the pass; candidates land at round row `at`.
pub fn devGroup(x: *SlotPass, ms: []const rows.Member, at: usize) !void {
    const d = x.dev.?;
    const f = x.f;
    const n: u32 = f.cfg.dspark_block;
    const lay = emit.Layout.of(@intCast(ms.len), n, f.cfg.window);
    const sv = try d.statics(at / @as(usize, n), @intCast(lay.len64), @intCast(lay.len32));
    rows.stage(lay, n, f.cfg.window, @intCast(emit.ringRows(f.cfg)), x.one.shape.noise, ms, sv[0], sv[1]);
    const r = f.runner;
    try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i64") orelse return error.Unbound, .len = 8 * @as(usize, @intCast(lay.len64)) }, 0, std.mem.sliceAsBytes(sv[0]), r.stream.handle);
    try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i32") orelse return error.Unbound, .len = 4 * @as(usize, @intCast(lay.end32)) }, 0, std.mem.sliceAsBytes(sv[1][0..@intCast(lay.end32)]), r.stream.handle);
    x.stats.passes += 1;
    x.stats.slots += ms.len;
    x.cur_at = at;
    r.glue = .{ .ctx = x, .run = glueFn };
    const extra = [_]u64{ if (x.send) |b| b.ptr else 0, if (x.send) |b| b.len else 0 };
    x.cur_k = ms.len;
    const key = passKey(ms.len, at);
    if (x.capture_first and d.active == .graphs) x.markSeen(key);
    try d.issue(key, dsd.fingerprint(r, d, &extra), .{ .ctx = x, .run = groupBody });
}

/// A pass's graph key: its slot count and its first row in the round.
pub fn passKey(k: usize, at: usize) u32 {
    return @intCast(k << 16 | at);
}

pub fn wasSeen(x: *const SlotPass, key: u32) bool {
    return std.mem.indexOfScalar(u32, x.seen[0..x.nseen], key) != null;
}

pub fn markSeen(x: *SlotPass, key: u32) void {
    if (x.wasSeen(key) or x.nseen == x.seen.len) return;
    x.seen[x.nseen] = key;
    x.nseen += 1;
}

pub fn groupBody(ctx: *anyopaque, stream: cuda.Stream) anyerror!void {
    const x: *SlotPass = @ptrCast(@alignCast(ctx));
    _ = stream;
    try x.f.runner.window(x.programs[x.cur_k - 1]);
}

// glue ---------------------------------------------------------------------------------------------------------

pub fn device(x: *SlotPass, slot: *?cuda.DeviceBuffer, bytes: usize) !u64 {
    return @import("grow_device.zig").grow(slot, x.f.runner.d, bytes);
}

pub fn numel(t: calls.Tensor) usize {
    var n: usize = 1;
    for (t.shape) |d| n *= @intCast(d);
    return n;
}

/// dspark_gpu.zig's glue steps at the pass's rows (`R`: every slot's).
pub fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
    const x: *SlotPass = @ptrCast(@alignCast(ctx));
    const step = c.name["glue.".len..];
    const s = r.stream.handle;
    const k = r.kernels.others(r.stream);
    const comm = x.f.comm;
    const D: usize = x.f.cfg.hidden;
    const Wd: usize = comm.world();
    if (std.mem.eql(u8, step, "ds_embed")) {
        const n: usize = numel(c.args[2].arg.t);
        const V: usize = x.f.cfg.vocab / Wd;
        const embed = r.weights.get("embed") orelse return error.MissingWeight;
        const rws = try x.device(&x.send, 2 * n * D);
        try k.embedSend(try r.tensorAddr(c.args[2].arg.t), n, embed.ptr, @intCast(comm.rank() * V), V, D, rws);
        const recv = try r.tensorAddr(c.args[1].arg.t);
        try comm.allGather(rws, recv, n * D, .bf16, s);
        return k.embedSum(recv, Wd, n, D, try r.tensorAddr(c.args[0].arg.t));
    }
    if (std.mem.eql(u8, step, "exchange_f32")) {
        const part = c.args[0].arg.t;
        const n = numel(part);
        const half = try x.device(&x.send, 2 * n);
        try k.castBf16(try r.tensorAddr(part), half, n);
        return comm.allGather(half, try r.tensorAddr(c.args[1].arg.t), n, .bf16, s);
    }
    if (std.mem.eql(u8, step, "widen"))
        return k.widenF32(try r.tensorAddr(c.args[0].arg.t), try r.tensorAddr(c.args[1].arg.t), numel(c.args[0].arg.t));
    if (std.mem.eql(u8, step, "ds_out")) {
        if (x.dev) |dv| if (dv.on()) {
            x.last_logits = try r.tensorAddr(c.args[0].arg.t);
            x.last_v = @intCast(c.args[0].arg.t.shape[1]);
            return dv.launch(r, c.args[0].arg.t, c.args[1].arg.t, @intCast(c.args[0].arg.t.shape[0]), x.cur_at);
        };
        return x.hostStep(r, c.args[0].arg.t, c.args[1].arg.t);
    }
    return error.GlueNotBuilt;
}

/// The host step over the pass's rows: each rank's best k a row, lists gathered, head hidden for confidences.
pub fn hostStep(x: *SlotPass, r: *run.Runner, logits_t: calls.Tensor, hid_t: calls.Tensor) !void {
    const th = ph.start(x.f.phases, .@"draft.host");
    defer th.stop();
    const comm = x.f.comm;
    const R: usize = @intCast(logits_t.shape[0]);
    const V: usize = @intCast(logits_t.shape[1]);
    const D: usize = x.f.cfg.hidden;
    const k: usize = x.one.k;
    const Wd: usize = comm.world();
    const logits = x.logits[0 .. R * V];
    const hid16 = x.hid16[0 .. R * D];
    try r.stream.synchronize();
    try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(logits_t), .len = 4 * logits.len }, 0, std.mem.sliceAsBytes(logits));
    try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(hid_t), .len = 2 * hid16.len }, 0, std.mem.sliceAsBytes(hid16));
    for (hid16, x.hid[0 .. R * D]) |b, *h| h.* = dspark.bf16(b);
    const packed_ = x.packed_[0 .. R * 2 * k];
    const id0: u32 = @intCast(comm.rank() * V);
    for (0..R) |i| {
        const row = packed_[i * 2 * k ..][0 .. 2 * k];
        try dspark.candidates(x.gpa, logits[i * V ..][0..V], id0, x.ids, row[0..k]);
        for (x.ids, row[k..]) |id, *o| o.* = @bitCast(id);
    }
    const bytes = 4 * packed_.len;
    const src = try x.device(&x.send, bytes);
    const dst = try x.device(&x.gathered, bytes * Wd);
    try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = src, .len = bytes }, 0, std.mem.sliceAsBytes(packed_), r.stream.handle);
    try comm.allGather(src, dst, packed_.len, .f32, r.stream.handle);
    try r.stream.synchronize();
    const all = x.all[0 .. packed_.len * Wd];
    try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = dst, .len = 4 * all.len }, 0, std.mem.sliceAsBytes(all));
    rows.ungather(all, Wd, R, k, x.cand, x.cval);
    try gpu.checkCandidates(x.cand[0 .. R * Wd * k], x.f.cfg.vocab); // the pass's rows (x.cand holds the most)
}
