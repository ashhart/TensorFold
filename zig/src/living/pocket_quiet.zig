//! A pocket lesson's earlier neurons: other answers its subject neurons fire on, and refits of those on its rows.
const std = @import("std");
const pk = @import("pocket.zig");
const pocket_fit = @import("pocket_fit.zig");
const decouple = @import("pocket_decouple.zig");
const subspace = @import("subspace.zig");
const clock = @import("pocket_clock.zig");
const pocket_dump = @import("pocket_dump.zig");
const joint = @import("pocket_joint.zig");
const Kind = pocket_fit.Kind;
const Fit = pocket_fit.Fit;
const fit = pocket_fit.fit;
const fitLayer = pocket_fit.fitLayer;
const dotBf16 = pocket_fit.dotBf16;
const fit_steps = pocket_fit.fit_steps;

/// Other answers where the fact's subject neurons fire, each at its strongest row: these stay as they were.
pub fn findKeeps(p: anytype, d: usize) !void {
    if (p.split[1] == 0 or !p.keeping) return;
    const n = p.slots();
    var i: usize = 0;
    while (i < p.labels.items.len) {
        const which = p.labels.items[i].ex;
        var most: f32 = 0;
        var at: ?usize = null;
        while (i < p.labels.items.len and p.labels.items[i].ex == which) : (i += 1) {
            const l = p.labels.items[i];
            // the fact's own mentions are free but already its answers, not ones to keep
            if (l.kind != .none or l.ex < p.answered.len + p.prompts.len + p.held.len) continue;
            var sum: f32 = 0;
            for (p.taken.items) |x| if (x.key == .subject) {
                sum += @max(p.act(x.slot, x.neuron, p.rows.items[(i * n + x.slot) * d ..][0..d]), 0);
            };
            if (sum > most) {
                most = sum;
                at = l.row;
            }
        }
        const row = at orelse continue;
        const ex = p.pickRead(which).?.ex;
        if (ex.ids.len > ex.start + 1) try p.keeps.append(p.gpa, .{ .ex = ex, .row = row });
    }
    std.log.info("slide pocket: {d} other answers name the fact's subject where its @TypeOf(p.*).neurons fire; they are to stay as they were", .{p.keeps.items.len});
}

/// Earlier neurons firing on this fact's rows refitted silent there, undo copies kept; same-subject ones stay.
pub fn quiet(p: anytype, t: anytype, host: anytype) !void {
    const d = p.hidden;
    const Job = struct {
        e: pk.Learned,
        most: f32,
        u: []f32, // its input weights, refitted in this copy
        xs: std.ArrayList([]const u16) = .empty,
        kinds: std.ArrayList(Kind) = .empty,
        fixed: usize = 0,
        f: Fit = .{},
        failed: bool = false,
    };
    var jobs: std.ArrayList(Job) = .empty;
    defer {
        for (jobs.items) |*j| {
            p.gpa.free(j.u);
            j.xs.deinit(p.gpa);
            j.kinds.deinit(p.gpa);
        }
        jobs.deinit(p.gpa);
    }
    for (p.learned.items[0..p.learned_before]) |e| {
        if (e.key == .subject and std.mem.indexOfScalar(u32, p.subjects.items, e.subject) != null) continue;
        const u = p.neuron(p.up.items, e.slot, e.neuron, d);
        var most: f32 = -std.math.inf(f32);
        const n = p.slots();
        for (p.labels.items, 0..) |l, i| if (l.kind == .fact or l.kind == .held) {
            most = @max(most, dotBf16(u, p.rows.items[(i * n + e.slot) * d ..][0..d]));
        };
        if (most <= pk.refit_above) continue; // weak there: left exactly as it is, this fact's outputs fitted with it
        try jobs.append(p.gpa, .{ .e = e, .most = most, .u = try p.gpa.dupe(f32, u) });
        try refitRows(p, &jobs.items[jobs.items.len - 1]);
    }
    const scales: []const f32 = if (p.decoupled and jobs.items.len > 0) try p.scales.get(p, t) else &.{};
    const Run = struct {
        fn one(pp: anytype, sc: []const f32, j: *Job, dd: usize) void {
            const gpa = pp.gpa;
            j.f = fit(gpa, j.xs.items, j.kinds.items, dd, j.u, true, fit_steps / 2, j.fixed, @TypeOf(p.*).fam.neuron.margin) catch {
                j.failed = true;
                return;
            };
            if (sc.len == 0) return;
            // refitted, still kept off what the neurons below it write, and judged as it is then
            var basis: std.ArrayList(f32) = .empty;
            defer basis.deinit(gpa);
            const dirs = decouple.below(pp, gpa, sc, j.e.slot, &basis) catch {
                j.failed = true;
                return;
            };
            if (dirs == 0) return;
            decouple.project(j.u, basis.items, dirs, dd);
            j.f = decouple.evaluate(j.xs.items, j.kinds.items, j.u, gpa) catch {
                j.failed = true;
                return;
            };
        }
        fn share(pp: anytype, sc: []const f32, js: []Job, cursor: *std.atomic.Value(usize), dd: usize) void {
            while (true) {
                const i = cursor.fetchAdd(1, .monotonic);
                if (i >= js.len) return;
                one(pp, sc, &js[i], dd);
            }
        }
    };
    var cursor = std.atomic.Value(usize).init(0);
    const workers = @min(jobs.items.len, @min(std.Thread.getCpuCount() catch 1, 32));
    var spawned: [32]?std.Thread = @splat(null);
    for (spawned[0..workers]) |*th| th.* = std.Thread.spawn(.{}, Run.share, .{ p, scales, jobs.items, &cursor, d }) catch null;
    Run.share(p, scales, jobs.items, &cursor, d);
    for (spawned[0..workers]) |th| if (th) |x| x.join();
    for (jobs.items) |*j| {
        if (j.failed) return error.OutOfMemory;
        const e = j.e;
        const u = p.neuron(p.up.items, e.slot, e.neuron, d);
        const f = j.f;
        if (f.fact_least < pk.kept_least or @max(f.steady_most, f.other_most) > 0) {
            // a refit that forgets its fact or spreads is undone; this fact's outputs are fitted with it firing
            std.log.info("slide pocket: layer {d}, neuron {d}, an earlier fact's, fires on this fact's questions (up to {d:.2}); fitted again its own would read at least {d:.2} and other rows at most {d:.2}, so it stays as it was", .{ p.moe.items[e.slot], e.neuron, j.most, f.fact_least, @max(f.steady_most, f.other_most) });
            continue;
        }
        try p.refits.append(p.gpa, .{ .slot = e.slot, .neuron = e.neuron });
        try p.refit_up.appendSlice(p.gpa, u);
        if (@TypeOf(p.*).swiglu) try p.refit_gate.appendSlice(p.gpa, p.neuron(p.gate.items, e.slot, e.neuron, d));
        @memcpy(u, j.u);
        p.gateFrom(e.slot, e.neuron);
        try p.upload(t, host, e.slot);
        std.log.info("slide pocket: layer {d}, neuron {d}, an earlier fact's, fired on this fact's questions (up to {d:.2}); fitted again: there and elsewhere at most {d:.2}, its own questions' least {d:.2} ({d} rows held as they read, {d} of them read now)", .{ p.moe.items[e.slot], e.neuron, j.most, @max(f.steady_most, f.other_most), f.fact_least, j.fixed, j.fixed - e.count });
    }
}

/// A refit's rows: its fact's stored rows, then the rows it fires on now held as they read, then the rest.
pub fn refitRows(p: anytype, j: anytype) !void {
    const d = p.hidden;
    const n = p.slots();
    const e = j.e;
    const u = j.u;
    const own = p.answered.len + p.prompts.len + p.held.len; // read examples below this are this fact's
    for (0..e.count) |r| {
        try j.xs.append(p.gpa, p.pos.items[(e.start + r) * d ..][0..d]);
        try j.kinds.append(p.gpa, .fact);
    }
    // the rows of other prompts it fires on now are its own fact's as the model reads them today.
    const live = try p.gpa.alloc(bool, p.labels.items.len);
    defer p.gpa.free(live);
    var fixed = e.count;
    for (p.labels.items, live, 0..) |l, *on, i| {
        const x = p.rows.items[(i * n + e.slot) * d ..][0..d];
        on.* = (l.ex >= own or l.row < p.pickRead(l.ex).?.ex.context) and dotBf16(u, x) > 0;
        if (!on.*) continue;
        try j.xs.append(p.gpa, x);
        try j.kinds.append(p.gpa, .none); // fixed, so read as it was; a weak one is no stored row to judge by
        fixed += 1;
    }
    for (p.labels.items, live, 0..) |l, on, i| {
        if (on) continue;
        try j.xs.append(p.gpa, p.rows.items[(i * n + e.slot) * d ..][0..d]);
        try j.kinds.append(p.gpa, switch (l.kind) {
            .fact, .held => .steady, // this fact's questions are another fact's to it
            .none => .other,
            else => l.kind,
        });
    }
    j.fixed = fixed;
}
