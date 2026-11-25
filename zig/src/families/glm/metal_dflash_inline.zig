//! A round's DFlash2 pass encoded in the verify window's command buffer on one Mac, one wait a round.
const std = @import("std");
const mtl = @import("metal");
const op = @import("../qwen27/dflash/operators.zig");
const rounds_mod = @import("metal_dflash_rounds.zig");
const slots_mod = @import("slots.zig");
const ahead = @import("metal_dflash_ahead.zig");
const same = @import("metal_dflash_same.zig");
const follow = @import("metal_dflash_follow.zig");
const accept = @import("metal_dflash_accept.zig");
const max_held = rounds_mod.max_held;

/// A stream's window the pass follows: its rounds, slot, the window's first position and its rows.
const Entry = struct { sr: *rounds_mod.StreamRounds, slot: u32 = 0, first: u64, rows: u32 };

/// The pass in a window's buffer, its acceptance and lattices once the window completes, until the round takes them.
pub const Inline = struct {
    gpa: std.mem.Allocator,
    r: *rounds_mod.Rounds,
    kernel: follow.Follow,
    acc: accept.Accept,
    table: mtl.Buffer, // u32 offsets[8] then rows[8]: each stream's rows in the window's ids and picks
    words: mtl.Buffer, // the accept kernel's two words a stream
    entries: [8]Entry = undefined,
    n: u32 = 0,
    pass: same.Pass = undefined,
    armed: bool = false, // encoded in the open window, not yet landed
    ready: bool = false, // landed: `kept` and `lattices` hold the pass's
    kept: [16]u32 = undefined,
    lattices: [8]op.Lattice = undefined,
    window: u64 = 0, // the window the pass followed (Slots.windows after it)
    served: u64 = 0,
    missed: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, r: *rounds_mod.Rounds, device: mtl.Device) !*Inline {
        const x = try gpa.create(Inline);
        errdefer gpa.destroy(x);
        const kernel = try follow.Follow.init(gpa, device, null);
        errdefer kernel.deinit();
        const acc = try accept.Accept.init(gpa, device, null);
        errdefer acc.deinit();
        const table = try device.buffer(64, mtl.ResourceOptions.shared);
        errdefer table.deinit();
        x.* = .{ .gpa = gpa, .r = r, .kernel = kernel, .acc = acc, .table = table, .words = try device.buffer(64, mtl.ResourceOptions.shared) };
        return x;
    }

    pub fn deinit(x: *Inline) void {
        x.drop();
        x.kernel.deinit();
        x.acc.deinit();
        x.table.deinit();
        x.words.deinit();
        x.gpa.destroy(x);
    }

    /// The slots' hook: the pass encoded behind each window's forward, told when the window completed.
    pub fn hook(x: *Inline) slots_mod.Follow {
        return .{ .ctx = x, .encode = encode, .land = land };
    }

    fn of(ctx: *anyopaque) *Inline {
        return @ptrCast(@alignCast(ctx));
    }

    fn encode(ctx: *anyopaque, sl: *slots_mod.Slots, cb: mtl.CommandBuffer, e: mtl.ComputeEncoder, wins: []const slots_mod.Win) void {
        const x = of(ctx);
        x.drop();
        x.encodeWins(sl, cb, e, wins) catch |err| {
            std.log.warn("glm drafter: the pass in the window's buffer ({s}); the round drafts as before", .{@errorName(err)});
        };
    }

    /// Every stream of the window expecting DFlash2 and caught up to it: their block pass behind its forward.
    fn encodeWins(x: *Inline, sl: *slots_mod.Slots, cb: mtl.CommandBuffer, e: mtl.ComputeEncoder, wins: []const slots_mod.Win) !void {
        const r = x.r;
        const c = r.model.graph.config;
        const n: u32 = @intCast(wins.len);
        const block = r.model.blockFor(max_held);
        if (wins.len == 0 or wins.len > x.entries.len or n * block > 128) return;
        if (!r.expectInlineDflash(wins)) return;
        var aheads: [8]ahead.Ahead = undefined;
        const table = x.table.slice(u32, 16);
        for (wins, 0..) |w, k| {
            const slot = &sl.slots[w.slot];
            const sr = r.get(w.slot) orelse return;
            if (!sr.dflash or !ahead.ready(c, sr.ctx, slot.s.pos, slot.rows)) return;
            x.entries[k] = .{ .sr = sr, .slot = w.slot, .first = slot.s.pos, .rows = slot.rows };
            aheads[k] = .{ .context = sr.ctx, .first = slot.s.pos, .rows = slot.rows };
            table[k] = slot.row0;
            table[8 + k] = slot.rows;
        }
        const m = r.model;
        const epoch = try same.beginOn(m, cb, e);
        errdefer same.release(m);
        const ids = sl.e.sc.ids;
        const picks = sl.e.sc.picks;
        try x.acc.encode(e, .{ .buf = ids.buf, .off = ids.off }, .{ .buf = picks.buf, .off = picks.off }, .{ .buf = x.table }, .{ .buf = x.table, .off = 32 }, .{ .buf = x.words }, n);
        e.barrier();
        x.pass = try same.encode(m, &x.kernel, epoch, aheads[0..n], block, .{ .gpu = .{ .buf = x.words } });
        x.n = n;
        x.armed = true;
    }

    fn land(ctx: *anyopaque, sl: *slots_mod.Slots, ok: bool) void {
        const x = of(ctx);
        if (!x.armed) return;
        x.armed = false;
        const m = x.r.model;
        defer same.release(m);
        if (!ok) return;
        same.landed(m) catch return;
        same.lattices(m, &x.pass, x.lattices[0..x.n]) catch |err| {
            std.log.warn("glm drafter: the pass in the window's buffer read back ({s}); the round drafts as before", .{@errorName(err)});
            return;
        };
        @memcpy(x.kept[0 .. 2 * x.n], x.words.slice(u32, 2 * x.n));
        x.ready = true;
        x.window = sl.windows + 1; // the window counts itself once it has landed
    }

    /// The landed pass's lattices let go.
    pub fn drop(x: *Inline) void {
        if (x.ready) for (x.lattices[0..x.n]) |*l| l.deinit();
        x.ready = false;
    }

    /// The round's `jobs` take their chains from the landed pass when every job matches; false: fallback.
    pub fn serve(x: *Inline, sl: *const slots_mod.Slots, jobs: []const rounds_mod.Job, out: []rounds_mod.Round) !bool {
        defer x.drop();
        if (!x.ready or x.window != sl.windows or jobs.len == 0) return false;
        var at: [8]usize = undefined;
        const fits = match(x.entries[0..x.n], x.kept[0 .. 2 * x.n], x.pass.block, x.r.model, jobs, at[0..jobs.len]);
        if (sl.log) std.log.info("glm inline: streams {d} served {}", .{ jobs.len, fits });
        if (!fits) {
            x.missed += 1;
            return false;
        }
        const c = x.r.model.graph.config;
        for (x.entries[0..x.n], 0..) |en, k| {
            if (x.r.get(en.slot)) |live| if (live == en.sr) ahead.settle(c, live.ctx, en.first, x.kept[2 * k]);
        }
        for (jobs, at[0..jobs.len], out[0..jobs.len]) |j, s, *o| {
            const kept = x.kept[2 * s];
            j.sr.held_n = 0;
            var t = try same.tree(x.r.model, &x.pass, s, x.lattices[s], kept, x.kept[2 * s + 1], .{ .nodes = @min(j.depth, max_held), .children = 1 });
            defer t.deinit();
            x.r.holdChain(j.sr, &t, j.depth, o);
        }
        x.served += 1;
        return true;
    }
};

/// Whether every job in `jobs` matches a stream entry's GPU acceptance and anchor within the fixed block `block`.
pub fn match(entries: []const Entry, kept: []const u32, block: u32, model: anytype, jobs: []const rounds_mod.Job, at: []usize) bool {
    _ = model;
    if (jobs.len == 0 or jobs.len > entries.len) return false;
    for (jobs, at) |j, *a| {
        const s = for (entries, 0..) |en, k| {
            if (en.sr == j.sr) break k;
        } else return false;
        // the core kept what the GPU kept, its next anchor is the pick after them, and the fixed block covers j.depth
        if (j.anchor != kept[2 * s + 1] or j.position != entries[s].first + kept[2 * s] + 2) return false;
        if (j.depth == 0 or @min(j.depth, max_held) + 1 > block) return false;
        a.* = s;
    }
    return true;
}

test "a landed pass serves every matching job up to the fixed block, and misses on count or anchor mismatch" {
    const Blocks = struct {
        fn blockFor(_: @This(), nodes: u32) u32 {
            return @min(16, nodes + 1);
        }
    };
    var srs: [2]rounds_mod.StreamRounds = undefined;
    const entries = [2]Entry{ .{ .sr = &srs[0], .first = 100, .rows = 8 }, .{ .sr = &srs[1], .first = 40, .rows = 5 } };
    const kept = [4]u32{ 3, 77, 0, 9 };
    var at: [2]usize = undefined;
    // the core's requests in another order: stream 1's anchor at 40 + 0 + 1, stream 0's at 100 + 3 + 1
    var jobs = [2]rounds_mod.Job{ .{ .sr = &srs[1], .position = 42, .anchor = 9, .depth = 7 }, .{ .sr = &srs[0], .position = 105, .anchor = 77, .depth = 7 } };
    try std.testing.expect(match(&entries, &kept, 8, Blocks{}, &jobs, &at));
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, &at);
    jobs[1].position = 104; // the core kept fewer (a stop or a cut)
    try std.testing.expect(!match(&entries, &kept, 8, Blocks{}, &jobs, &at));
    jobs[1].position = 105;
    jobs[0].anchor = 8; // another anchor
    try std.testing.expect(!match(&entries, &kept, 8, Blocks{}, &jobs, &at));
    jobs[0].anchor = 9;
    jobs[0].depth = 3; // fixed-block pass covers any positive depth up to 7
    try std.testing.expect(match(&entries, &kept, 8, Blocks{}, &jobs, &at));
    try std.testing.expect(match(&entries, &kept, 8, Blocks{}, jobs[0..1], at[0..1])); // subset after sibling ends
}
