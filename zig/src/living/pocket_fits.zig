//! A pocket lesson's input fits: every candidate layer fitted (host or device), kept off lower outputs, the best taken.
const std = @import("std");
const pk = @import("pocket.zig");
const pocket_fit = @import("pocket_fit.zig");
const decouple = @import("pocket_decouple.zig");
const subspace = @import("subspace.zig");
const clock = @import("pocket_clock.zig");
const pocket_dump = @import("pocket_dump.zig");
const joint = @import("pocket_joint.zig");
const quieting = @import("pocket_quiet.zig");
const Kind = pocket_fit.Kind;
const Fit = pocket_fit.Fit;
const fit = pocket_fit.fit;
const fitLayer = pocket_fit.fitLayer;
const dotBf16 = pocket_fit.dotBf16;
const fit_steps = pocket_fit.fit_steps;

/// Every key's candidate layers fitted in threads, the best taken; then keeps found, earlier neurons refitted.
pub fn fitAll(p: anytype, t: anytype, host: anytype) !bool {
    const d = p.hidden;
    const n = p.slots();
    const m = p.labels.items.len;
    if (std.c.getenv("TF_SLIDE_POCKET_DUMP")) |dir| pocket_dump.write(p, std.mem.span(dir), p.learned_before, d) catch |e| std.log.warn("slide pocket: the read rows were not written: {t}", .{e});
    const kinds = try p.gpa.alloc(Kind, 3 * m);
    defer p.gpa.free(kinds);
    for (p.labels.items, 0..) |l, i| for (0..3) |k| {
        kinds[k * m + i] = pk.kindFor(l, @fromBackingInt(@intCast(k)));
    };
    const us = try p.gpa.alloc(f32, 3 * n * d);
    defer p.gpa.free(us);
    const fits = try p.gpa.alloc(Fit, 3 * n);
    defer p.gpa.free(fits);
    @memset(fits, .{ .fires = std.math.maxInt(usize) });
    const t0 = clock.now(host.io);
    const below = @TypeOf(p.*).fam.neuron.margin;
    var fitted: usize = 0;
    for (0..3) |k| for (p.moe.items) |l| if (p.split[k] > 0 and pk.candidate(l, p.layers, @fromBackingInt(@intCast(k)))) {
        fitted += 1;
    };
    const device = comptime @hasDecl(@TypeOf(t.*), "fitDevice");
    if (device and !pk.set("TF_SLIDE_FIT", "host")) {
        try fitOnDevice(p, t, kinds, us, fits, m, below);
    } else try fitOnHost(p, kinds, us, fits, m, below);
    if (p.decoupled) try inputsOff(p, t, kinds, us, fits, m);
    const t1 = clock.now(host.io);
    clock.add(.fit_layers, t1 - t0);
    for (0..3) |k| if (p.split[k] > 0) {
        const key: pk.Key = @fromBackingInt(@intCast(k));
        var order: std.ArrayList(usize) = .empty;
        defer order.deinit(p.gpa);
        // a layer whose pocket is full is passed over
        for (p.moe.items, 0..) |l, j| if (pk.candidate(l, p.layers, key) and p.used.items[j] < @TypeOf(p.*).neurons) try order.append(p.gpa, j);
        if (order.items.len == 0) return error.PocketFull;
        std.mem.sort(usize, order.items, @as([]const Fit, fits[k * n ..][0..n]), pk.better);
        for (order.items[0..@min(p.split[k], order.items.len)]) |j| try take(p, j, key, us[(k * n + j) * d ..][0..d], fits[k * n + j], kinds[k * m ..][0..m], d);
    };
    logSources(p, d, m, kinds);
    try quieting.findKeeps(p, d);
    const tq = clock.now(host.io);
    try p.quiet(t, host);
    clock.add(.fit_quiet, clock.now(host.io) - tq);
    std.log.info("slide pocket: {d} candidate @TypeOf(p.*).neurons fitted on {d} read rows in {d:.1} s; earlier @TypeOf(p.*).neurons checked in {d:.1} s", .{ fitted, m, t1 - t0, clock.now(host.io) - t1 });
    try joint.find(p, d);
    try p.v.begin(p.gpa, p.taken.items.len + p.joint.items.len, d, p.answered.len + p.joint_answers.items.len + p.keeps.items.len);
    joint.seed(p, d);
    p.next(.values);
    return false;
}

/// Every key's candidate layers fitted in threads on the host (pocket_fit.fitLayer), into us and fits.
pub fn fitOnHost(p: anytype, kinds: []const Kind, us: []f32, fits: []Fit, m: usize, below: f32) !void {
    const d = p.hidden;
    const n = p.slots();
    const threads = try p.gpa.alloc(?std.Thread, 3 * n);
    defer p.gpa.free(threads);
    @memset(threads, null);
    for (0..3) |k| for (p.moe.items, 0..) |l, j| if (p.split[k] > 0 and pk.candidate(l, p.layers, @fromBackingInt(@intCast(k)))) {
        threads[k * n + j] = std.Thread.spawn(.{}, fitLayer, .{ p.gpa, p.rows.items, kinds[k * m ..][0..m], n, j, d, us[(k * n + j) * d ..][0..d], &fits[k * n + j], below }) catch null;
    };
    for (0..3) |k| for (p.moe.items, 0..) |l, j| if (p.split[k] > 0 and pk.candidate(l, p.layers, @fromBackingInt(@intCast(k)))) {
        if (threads[k * n + j]) |th| th.join() else fitLayer(p.gpa, p.rows.items, kinds[k * m ..][0..m], n, j, d, us[(k * n + j) * d ..][0..d], &fits[k * n + j], below);
    };
}

/// The same fits with their Adam steps on the trainer's device (T.fitDevice).
pub fn fitOnDevice(p: anytype, t: anytype, kinds: []const Kind, us: []f32, fits: []Fit, m: usize, below: f32) !void {
    const d = p.hidden;
    const n = p.slots();
    const Job = struct { k: usize, j: usize, xs: [][]const u16 = &.{}, scale: f32 = 0, inv: f32 = 0, failed: bool = false };
    var jobs: std.ArrayList(Job) = .empty;
    defer {
        for (jobs.items) |jb| p.gpa.free(jb.xs);
        jobs.deinit(p.gpa);
    }
    for (0..3) |k| for (p.moe.items, 0..) |l, j| if (p.split[k] > 0 and pk.candidate(l, p.layers, @fromBackingInt(@intCast(k)))) {
        const xs = try p.gpa.alloc([]const u16, m);
        for (xs, 0..) |*x, i| x.* = p.rows.items[(i * n + j) * d ..][0..d];
        try jobs.append(p.gpa, .{ .k = k, .j = j, .xs = xs });
    };
    const Ends = struct {
        fn start(gpa: std.mem.Allocator, jb: *Job, ks: []const Kind, u: []f32, dd: usize) void {
            const got = pocket_fit.prepare(gpa, jb.xs, ks, dd, u) catch {
                jb.failed = true;
                return;
            };
            jb.scale = got.scale;
            jb.inv = got.inv;
        }
        fn end(jb: *Job, ks: []const Kind, u: []f32, f: *Fit) void {
            f.* = pocket_fit.finish(jb.xs, ks, u, jb.inv, jb.scale);
        }
    };
    var threads: [192]?std.Thread = @splat(null);
    if (jobs.items.len > threads.len) return fitOnHost(p, kinds, us, fits, m, below);
    for (jobs.items, threads[0..jobs.items.len]) |*jb, *th| th.* = std.Thread.spawn(.{}, Ends.start, .{ p.gpa, jb, kinds[jb.k * m ..][0..m], us[(jb.k * n + jb.j) * d ..][0..d], d }) catch null;
    for (jobs.items, threads[0..jobs.items.len]) |*jb, th| if (th) |x| x.join() else Ends.start(p.gpa, jb, kinds[jb.k * m ..][0..m], us[(jb.k * n + jb.j) * d ..][0..d], d);
    for (jobs.items) |jb| if (jb.failed) return error.OutOfMemory;
    const dev = try p.gpa.alloc(pocket_fit.DeviceFit, jobs.items.len);
    defer p.gpa.free(dev);
    for (jobs.items, dev) |jb, *x| x.* = .{ .rows = p.rows.items, .slots = n, .slot = jb.j, .d = d, .kinds = kinds[jb.k * m ..][0..m], .u = us[(jb.k * n + jb.j) * d ..][0..d], .inv = jb.inv, .below = below, .steps = fit_steps };
    try t.fitDevice(dev);
    @memset(threads[0..jobs.items.len], null);
    for (jobs.items, threads[0..jobs.items.len]) |*jb, *th| th.* = std.Thread.spawn(.{}, Ends.end, .{ jb, kinds[jb.k * m ..][0..m], us[(jb.k * n + jb.j) * d ..][0..d], &fits[jb.k * n + jb.j] }) catch null;
    for (jobs.items, threads[0..jobs.items.len]) |*jb, th| if (th) |x| x.join() else Ends.end(jb, kinds[jb.k * m ..][0..m], us[(jb.k * n + jb.j) * d ..][0..d], &fits[jb.k * n + jb.j]);
    if (std.c.getenv("TF_SLIDE_FIT_CHECK") != null) {
        const us2 = try p.gpa.dupe(f32, us);
        defer p.gpa.free(us2);
        const fits2 = try p.gpa.dupe(Fit, fits);
        defer p.gpa.free(fits2);
        try fitOnHost(p, kinds, us2, fits2, m, below);
        var same: usize = 0;
        for (jobs.items) |jb| {
            const at = jb.k * n + jb.j;
            const eq = std.mem.eql(u8, std.mem.sliceAsBytes(us[at * d ..][0..d]), std.mem.sliceAsBytes(us2[at * d ..][0..d])) and std.meta.eql(fits[at], fits2[at]);
            same += @intFromBool(eq);
        }
        std.log.info("slide pocket: fit check: {d} of {d} @TypeOf(p.*).neurons fitted on the device have the host's bits", .{ same, jobs.items.len });
    }
}

/// Every candidate neuron's input row kept off what the live neurons below its layer write there.
pub fn inputsOff(p: anytype, t: anytype, kinds: []const Kind, us: []f32, fits: []Fit, m: usize) !void {
    const n = p.slots();
    const scales = try p.scales.get(p, t);
    const Job = struct { j: usize, dirs: usize = 0, failed: bool = false, lost: f32 = 0, least: [3]f32 = @splat(0) };
    const jobs = try p.gpa.alloc(Job, n);
    defer p.gpa.free(jobs);
    for (jobs, 0..) |*jb, j| jb.* = .{ .j = j };
    const Run = struct {
        fn one(pp: anytype, jb: *Job, sc: []const f32, ks: []const Kind, uu: []f32, ff: []Fit, mm: usize) void {
            const dd = pp.hidden;
            const nn = pp.slots();
            var any = false;
            for (0..3) |k| any = any or (pp.split[k] > 0 and pk.candidate(pp.moe.items[jb.j], pp.layers, @fromBackingInt(@intCast(k))) and ff[k * nn + jb.j].fires != std.math.maxInt(usize));
            if (!any) return;
            var basis: std.ArrayList(f32) = .empty;
            defer basis.deinit(pp.gpa);
            jb.dirs = decouple.below(pp, pp.gpa, sc, jb.j, &basis) catch {
                jb.failed = true;
                return;
            };
            if (jb.dirs == 0) return;
            const xs = pp.gpa.alloc([]const u16, mm) catch {
                jb.failed = true;
                return;
            };
            defer pp.gpa.free(xs);
            for (xs, 0..) |*x, i| x.* = pp.rows.items[(i * nn + jb.j) * dd ..][0..dd];
            for (0..3) |k| {
                const at = k * nn + jb.j;
                if (pp.split[k] == 0 or !pk.candidate(pp.moe.items[jb.j], pp.layers, @fromBackingInt(@intCast(k))) or ff[at].fires == std.math.maxInt(usize)) continue;
                const u = uu[at * dd ..][0..dd];
                const was = subspace.dot(u, u);
                decouple.project(u, basis.items, jb.dirs, dd);
                jb.lost = @max(jb.lost, 1 - subspace.dot(u, u) / @max(was, 1e-30));
                jb.least[k] = ff[at].fact_least;
                ff[at] = decouple.evaluate(xs, ks[k * mm ..][0..mm], u, pp.gpa) catch {
                    jb.failed = true;
                    return;
                };
            }
        }
    };
    var threads: [192]?std.Thread = @splat(null);
    const k_n = @min(n, threads.len);
    for (jobs[0..k_n], threads[0..k_n]) |*jb, *th| th.* = std.Thread.spawn(.{}, Run.one, .{ p, jb, scales, kinds, us, fits, m }) catch null;
    for (jobs[0..k_n], threads[0..k_n]) |*jb, th| if (th) |x| x.join() else Run.one(p, jb, scales, kinds, us, fits, m);
    for (jobs[k_n..]) |*jb| Run.one(p, jb, scales, kinds, us, fits, m);
    var most: usize = 0;
    var lost: f32 = 0;
    var drop: f32 = 0;
    for (jobs) |jb| {
        if (jb.failed) return error.OutOfMemory;
        most = @max(most, jb.dirs);
        lost = @max(lost, jb.lost);
        for (0..3) |k| if (jb.dirs > 0 and p.split[k] > 0 and fits[k * n + jb.j].fires != std.math.maxInt(usize) and jb.least[k] != 0) {
            drop = @max(drop, jb.least[k] - fits[k * n + jb.j].fact_least);
        };
    }
    std.log.info("slide pocket: candidate inputs kept off what the @TypeOf(p.*).neurons below them write (up to {d} directions a layer): at most {d:.1}% of an input's size removed, a fact's least product down by at most {d:.3}", .{ most, 100 * lost, drop });
}

/// Logs each key's first neuron's firing share by source: answers, short and long wordings, held-out answers.
pub fn logSources(p: anytype, d: usize, m: usize, kinds: []const Kind) void {
    const n = p.slots();
    for (p.taken.items) |x| {
        if (x.neuron + 1 != p.used.items[x.slot]) continue;
        const k = @backingInt(x.key);
        var on: [5]usize = @splat(0);
        var all: [5]usize = @splat(0);
        var quiet_most: f32 = -std.math.inf(f32); // the highest it reads on another prompt in a conversation
        const u = p.neuron(@constCast(p.up.items), x.slot, x.neuron, d);
        for (p.labels.items, 0..) |l, i| {
            const kind = kinds[k * m + i];
            const ex = p.pickRead(l.ex).?.ex;
            if (kind == .steady and ex.context > 0) quiet_most = @max(quiet_most, dotBf16(u, p.rows.items[(i * n + x.slot) * d ..][0..d]));
            if (kind != .fact and kind != .held) continue;
            const src: usize = if (ex.context > 0) 4 else if (l.ex < p.answered.len) 0 else if (l.ex < p.answered.len + p.prompts.len) (if (ex.ids.len > 60) 2 else 1) else 3;
            all[src] += 1;
            on[src] += @intFromBool(dotBf16(u, p.rows.items[(i * n + x.slot) * d ..][0..d]) >= 0.5);
        }
        std.log.info("slide pocket: layer {d} at the {s} fires on answers {d}/{d}, short wordings {d}/{d}, long ones {d}/{d}, held-out {d}/{d}, in a conversation {d}/{d} (other prompts there at most {d:.2})", .{ p.moe.items[x.slot], @tagName(x.key), on[0], all[0], on[1], all[1], on[2], all[2], on[3], all[3], on[4], all[4], quiet_most });
    }
}

/// Slot j's next neuron taken for the open fact on `key`, its input weights `u` as fitted (`f`, on `kinds`).
pub fn take(p: anytype, j: usize, key: pk.Key, u: []const f32, f: Fit, kinds: []const Kind, d: usize) !void {
    const n = p.slots();
    const k = p.used.items[j];
    p.used.items[j] += 1;
    try p.taken.append(p.gpa, .{ .slot = j, .neuron = k, .key = key });
    @memcpy(p.neuron(p.up.items, j, k, d), u);
    p.gateFrom(j, k);
    @memset(p.neuron(p.down.items, j, k, d), 0);
    std.log.info("slide pocket: layer {d}, neuron {d}, at the {s}: on for {d:.0}% of the fact's questions (least {d:.2}), held-out {d:.0}% (mean {d:.2}); other prompts at most {d:.2}, other rows {d:.2}, {d} above zero", .{ p.moe.items[j], k, @tagName(key), 100 * f.fact_on, f.fact_least, 100 * f.held_on, f.held_mean, f.steady_most, f.other_most, f.fires });
    const start = p.pos.items.len / d;
    for (kinds, 0..) |kind, i| if (kind == .fact) try p.pos.appendSlice(p.gpa, p.rows.items[(i * n + j) * d ..][0..d]);
    const subject: u32 = if (key == .subject and p.subjects.items.len > 0) p.subjects.items[0] else 0;
    try p.learned.append(p.gpa, .{ .slot = j, .neuron = k, .key = key, .start = start, .count = p.pos.items.len / d - start, .subject = subject, .lesson = p.lesson });
}
