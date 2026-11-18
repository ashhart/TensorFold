//! Drafts behind the verify: a window's picks (greedy or keyed), acceptance, ingest and next pass queued on the GPU.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const dspark = @import("dspark.zig");
const plan_ = @import("dspark_plan.zig");
const pipelines = @import("pipelines.zig");
const picks_ = @import("early_picks.zig");

const iface = host.iface;
const lanes = host.lanes;
const R = pipelines.max_rows;
const Plan = plan_.Plan;
const threads = 1024;
pub const raced = picks_.raced;

pub const Stats = struct {
    /// chains queued behind a window (`keyed`: of sampled windows); their passes the lanes took, and those dropped
    queued: u64 = 0,
    keyed: u64 = 0,
    used: u64 = 0,
    dropped: u64 = 0,
    /// chains whose accepted count a test swapped (`skew_every`)
    skewed: u64 = 0,
};

/// A window about to be verified whose slot may draft behind it (Pass.arm).
const Arm = struct { slot: u32, start: u64, rows: u32 };

/// A chain queued behind a window: its slot's window, its command buffer, and what the lanes asked of it since.
const Chain = struct {
    slot: u32,
    start: u64,
    rows: u32,
    /// its pass's chain runs on the GPU (dspark.Pass.drawsOnGpu)
    gpu: bool,
    /// the drafts its pass makes
    block: u32,
    taps: mtl.Buffer,
    cb: mtl.CommandBuffer,
    /// the window's first rows the lanes asked ingested: the chain's own ingest when its picks accepted as many
    deferred: ?u32 = null,
    /// the pass the lanes began over it
    ask: ?dspark.Ask = null,
};

pub const Early = struct {
    a: std.mem.Allocator,
    x: *dspark.Pass,
    picks: picks_.Picks,
    /// tokens [R] u32, picks [R] u32, the accepted count and the bonus, then each sampled row's key base [R] u64
    state: mtl.Buffer,
    /// the plan of each accepted count: [R][Plan.ints] i32
    plans: mtl.Buffer,
    armed: ?Arm = null,
    /// more than one greedy slot armed for the coming window (no chain)
    crowded: bool = false,
    chain: ?Chain = null,
    stats: Stats = .{},
    /// tests: every this many chains (0: none) the GPU swaps an accepted count of 0 and 1, so the chain is dropped
    skew_every: u32 = 0,

    const tokens_at = 0;
    const picks_at = 4 * R;
    const result_at = 8 * R;
    const bases_at = 8 * R + 16;

    pub fn init(a: std.mem.Allocator, x: *dspark.Pass) !*Early {
        var picks = try picks_.Picks.init(x.device);
        errdefer picks.deinit();
        const state = try x.device.buffer(bases_at + 8 * R, mtl.ResourceOptions.shared);
        errdefer state.deinit();
        const plans = try x.device.buffer(@as(usize, R) * Plan.ints * 4, mtl.ResourceOptions.shared);
        errdefer plans.deinit();
        @memset(plans.contents()[0 .. @as(usize, R) * Plan.ints * 4], 0);
        if (x.resident) |set| {
            set.add(state);
            set.add(plans);
            set.commit();
        }
        const e = try a.create(Early);
        e.* = .{ .a = a, .x = x, .picks = picks, .state = state, .plans = plans };
        return e;
    }

    pub fn deinit(e: *Early) void {
        e.drain() catch {};
        e.picks.deinit();
        e.state.deinit();
        e.plans.deinit();
        e.a.destroy(e);
    }

    // the window ---------------------------------------------------------------------------------------------------

    /// Before a window: `slot`'s rows from `start` may draft behind it (a keyed window's sampling is read at `queue`).
    pub fn arm(e: *Early, slot: u32, start: u64, rows: u32, sampled: bool) void {
        _ = sampled;
        if (e.armed != null) e.crowded = true else e.armed = .{ .slot = slot, .start = start, .rows = rows };
    }

    /// Right after a window's command buffers: its slot's chain (picks, accepted drafts, ingest and pass), when armed.
    pub fn queue(e: *Early, group: []const iface.Segment, logits: mtl.Buffer, logits_off: usize, taps: mtl.Buffer) void {
        const armed = e.armed orelse return;
        defer {
            e.armed = null;
            e.crowded = false;
        }
        if (e.crowded or group.len != 1 or e.chain != null) return;
        const g = group[0];
        const x = e.x;
        const c = x.c;
        const B = x.blockOf(g.slot);
        const n: u32 = @intCast(g.tokens.len);
        if (g.slot != armed.slot or g.start != armed.start or n != armed.rows or g.slot >= x.runs.len) return;
        const keyed = if (g.sampling) |s| s.temperature > 0 else false;
        if (keyed and !raced(g.sampling.?, c.vocab)) return;
        if (g.draws.len != n) return;
        // rows past the accepted ones land in the block's own rows, which every pass writes before it reads them
        if (n == 0 or n > c.dspark_block + 1 or n > R or g.start + n + B > x.tables.positions) return;
        // the slot's rows from the window on as the drafter last had them (a cut window's put back), then kept again
        x.undo.settle(g.slot, g.start);
        e.stage(g, n, B) catch return;
        x.undo.save(g.slot, g.start, n);
        const skew = e.skew_every > 0 and n > 1 and (e.stats.queued + 1) % e.skew_every == 0;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = x.queue.commandBuffer();
        // a buffer never committed never runs: the rings stay as they are
        e.encode(cb, logits, logits_off, taps, g.start, n, B, skew, if (keyed) g.sampling.? else null) catch return;
        cb.commit();
        e.stats.skewed += @intFromBool(skew);
        e.stats.keyed += @intFromBool(keyed);
        e.chain = .{ .slot = g.slot, .start = g.start, .rows = n, .block = B, .gpu = x.mk != null, .taps = taps, .cb = cb.retain() };
        e.stats.queued += 1;
    }

    /// The host's part of a chain: the window's ids (and keyed rows' key bases), each accepted count's plan, the block.
    fn stage(e: *Early, g: iface.Segment, n: u32, B: u32) !void {
        const x = e.x;
        const c = x.c;
        const ids = e.state.slice(u32, R);
        for (ids[0..n], g.tokens) |*o, t| o.* = t;
        if (g.sampling) |s| picks_.bases(s, g.draws, @alignCast(std.mem.bytesAsSlice(u64, e.state.contents()[bases_at..][0 .. 8 * n])));
        // the GPU chain's draws for the next ask: its first draft's position is the accept's (pos_base + accepted)
        if (x.mk) |*m| m.o.asks()[0] = dspark.askOf(host.dspark.Params.of(g.sampling, 0));
        const all = e.plans.slice(i32, @as(usize, R) * Plan.ints);
        for (0..n) |acc| {
            const m = all[acc * Plan.ints ..][0..Plan.ints];
            for (0..n) |r| {
                m[Plan.ipos + r] = @intCast(g.start + r);
                const dest = if (r <= acc) plan_.ringRow(c, g.slot, g.start + r) else plan_.blockRow(c, g.slot, @intCast(r - acc - 1));
                m[Plan.idest + r] = @intCast(dest);
            }
            var run = x.runs[g.slot];
            run.land(g.start, acc + 1);
            try Plan.writeBlockTo(m, c, 0, g.slot, g.start + acc + 1, run, B);
        }
        const block: []u32 = @alignCast(std.mem.bytesAsSlice(u32, x.b.bytes(x.b.ids, 4 * B)));
        host.dspark.blockIds(.{ .block = B, .noise = c.dspark_noise_token }, 0, block);
    }

    /// The chain's encoders: picks (keyed by `smp`) and acceptance, every window row's ingest (serial), the pass.
    fn encode(e: *Early, cb: mtl.CommandBuffer, logits: mtl.Buffer, logits_off: usize, taps: mtl.Buffer, start: u64, n: u32, B: u32, skew: bool, smp: ?lanes.Sampling) !void {
        const x = e.x;
        const se = cb.compute(.serial);
        e.picks.encode(se, .{ .buf = logits, .off = logits_off }, .{ .buf = e.state, .off = picks_at }, .{ .buf = e.state, .off = bases_at }, n, x.c.vocab, smp);
        se.setPipeline(e.picks.accept);
        const pos_base = start + 2;
        se.setValue([5]u32{ n, Plan.ints, @intFromBool(skew), @truncate(pos_base), @truncate(pos_base >> 32) }, 0);
        se.setBuffer(e.state, tokens_at, 1);
        se.setBuffer(e.state, picks_at, 2);
        se.setBuffer(e.plans, 0, 3);
        se.setBuffer(x.plan.buf, 0, 4);
        se.setBuffer(x.b.ids.buf, x.b.ids.off, 5);
        se.setBuffer(e.state, result_at, 6);
        if (x.mk) |*m| se.setBuffer(m.o.buf, @import("markov.zig").Out.asks_at + 8, 7) else se.setBuffer(e.state, result_at + 8, 7);
        se.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(threads, 1, 1));
        const ok = x.encodeIngest(se, .{ .buf = taps, .rows = n }, n);
        se.end();
        try ok;
        const ce = cb.compute(.concurrent);
        const l: dspark.Launch = .{ .rows = B, .n = B, .asks = &.{}, .gpu = x.mk != null };
        const done = x.encodeAll(ce, &l);
        ce.end();
        try done;
    }

    // the lanes' asks ----------------------------------------------------------------------------------------------

    /// An ingest of the queued chain's window rows 0 .. m - 1 from its taps: left to the chain (true), or not ours.
    pub fn covers(e: *Early, slot: u32, start: u64, taps: mtl.Buffer, off: usize, rows: []const u32) bool {
        const ch = &(e.chain orelse return false);
        if (ch.slot != slot or ch.start != start or ch.deferred != null or taps.id != ch.taps.id or off != 0) return false;
        if (rows.len == 0 or rows.len > ch.rows) return false;
        for (rows, 0..) |r, i| if (r != i) return false;
        ch.deferred = @intCast(rows.len);
        return true;
    }

    /// The lanes begin the pass after the deferred rows: the chain's own (true), or not ours.
    pub fn begin(e: *Early, asks: []const iface.Ask) bool {
        const ch = &(e.chain orelse return false);
        const m = ch.deferred orelse return false;
        if (asks.len != 1 or ch.ask != null) return false;
        const k = asks[0];
        if (k.slot != ch.slot or k.start != ch.start + m) return false;
        ch.ask = .{ .slot = k.slot, .anchor = k.anchor, .start = k.start, .params = k.params };
        return true;
    }

    /// The begun pass's results: the chain's when its picks accepted the rows the host kept, else the host's own pass.
    pub fn collect(e: *Early, out: []dspark.Proposal) !void {
        const ch = e.chain orelse return error.NothingBegun;
        const ask = ch.ask orelse return error.NothingBegun;
        e.chain = null;
        const got = try e.wait(ch);
        const m = ch.deferred.?;
        if (got.acc + 1 == m and got.bonus == ask.anchor) {
            e.keep(ch, m);
            const l: dspark.Launch = .{ .rows = ch.block, .n = ch.block, .asks = (&ask)[0..1], .gpu = ch.gpu };
            try e.x.finish(&l, out);
            e.stats.used += 1;
            return;
        }
        e.stats.dropped += 1;
        try e.replace(ch, m);
        try e.x.propose((&ask)[0..1], out);
    }

    /// The queued chain finished and settled: a deferred ingest kept or redone, every other ring row put back.
    pub fn drain(e: *Early) !void {
        const ch = e.chain orelse return;
        e.chain = null;
        const got = try e.wait(ch);
        if (ch.ask != null) e.stats.dropped += 1;
        const m = ch.deferred orelse return e.x.undo.settle(ch.slot, ch.start);
        if (got.acc + 1 == m) return e.keep(ch, m);
        try e.replace(ch, m);
    }

    const Got = struct { acc: u32, bonus: u32 };

    fn wait(e: *Early, ch: Chain) !Got {
        defer ch.cb.release();
        ch.cb.wait();
        if (ch.cb.failure()) |msg| {
            std.debug.print("dsv41 early drafts: command buffer failed: {s}\n", .{msg});
            return error.CommandBufferFailed;
        }
        const r = e.state.slice(u32, result_at / 4 + 2)[result_at / 4 ..][0..2];
        return .{ .acc = @min(r[0], ch.rows - 1), .bonus = r[1] };
    }

    /// The chain's ingest is the host's: its ring rows landed, kept in case a later commit is shorter.
    fn keep(e: *Early, ch: Chain, m: u32) void {
        e.x.runs[ch.slot].land(ch.start, m);
        e.x.undo.shrink(ch.slot, m);
    }

    /// The chain's picks accepted other rows than the host keeps (`m`): its rows put back, the host's ingested.
    fn replace(e: *Early, ch: Chain, m: u32) !void {
        e.x.undo.settle(ch.slot, ch.start);
        var rows: [R]u32 = undefined;
        for (rows[0..m], 0..) |*r, i| r.* = @intCast(i);
        try e.x.ingest(ch.slot, ch.start, .{ .buf = ch.taps, .rows = ch.rows }, rows[0..m]);
    }

    pub fn describe(e: *const Early, buf: []u8) []const u8 {
        const s = e.stats;
        return std.fmt.bufPrint(buf, "drafts behind the verify: {d} queued ({d} keyed), {d} used, {d} dropped; {d} drafter ring rows put back", .{ s.queued, s.keyed, s.used, s.dropped, e.x.undo.restored }) catch buf[0..0];
    }
};
