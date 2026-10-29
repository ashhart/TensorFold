//! hip-affine-tune: every MLX affine prompt tile the GPU has, timed on the served cells, and proposed table rows.

const std = @import("std");
const base = @import("../rig.zig");
const affine = @import("../../launches/affine.zig");
const registry = @import("core").registry;
const tuning = @import("../../tuning/tuning.zig");
const search = @import("../../tuning/search.zig");
const mlx_entries = @import("../../launches/mlx_entries.zig");
const Event = @import("../../stream.zig").Event;
const prod = @import("product.zig");
const cells = @import("cells.zig");
const Product = prod.Product;
const Rig = prod.Rig;
const Entry = affine.Entry;
const Buffer = base.Buffer;
const gpa = base.gpa;

/// Bytes a rotation over the table copies must read, so launches miss the 64 MiB Infinity Cache and the L2.
const read_target = 512 << 20;
/// Device memory a class's table copies may take; a cell that cannot reach `read_target` within it is unqualified.
const memory_cap = 2 << 30;
const batches = 7;
/// The least GPU time of one batch, so clock ramps and launch gaps stay small against it.
const batch_us = 5000;

const Tuner = struct {
    rig: Rig,
    env: registry.Env,
    classes: []const cells.Class,
    s: search.Search,

    fn deinit(t: *Tuner) void {
        t.s.deinit();
    }

    /// A class's parent product at its most rows or tokens, the same in every pass, with its table copies.
    fn parent(t: *Tuner, class: usize, most: usize) !Product {
        const c = t.classes[class];
        t.rig.rng.state = 0x7E57_0000_0000_0001 +% class *% 0x9E37_79B9_7F4A_7C15;
        var p = try Product.init(&t.rig, c.case(most));
        errdefer p.deinit();
        const table_bytes = p.copy_words * 4 + p.copy_tab * 4;
        const room = @max(1, memory_cap / table_bytes);
        const want = if (c.routed) room else (read_target + table_bytes - 1) / table_bytes;
        try p.grow(@min(room, want));
        return p;
    }

    /// Times every prompt entry the GPU has on `count` rows or tokens of the parent's class.
    fn timeCell(t: *Tuner, class: usize, par: *const Product, count: usize) !void {
        if (t.s.cellOf(class, count) != null) return;
        var view: ?Product = if (count == par.c.rows) null else try Product.prefix(par, count);
        defer if (view) |*v| v.deinit();
        const p: *const Product = if (view) |*v| v else par;
        const op: registry.Op = if (p.c.routed()) .routed else .project;
        const arg = p.arg(0, 0);
        const reads = @as(f64, @floatFromInt(p.copies)) * p.bytes();
        var shape = affine.Kernels.shapeOf(arg, @intCast(p.items));
        shape.plan_pairs = @intCast(p.pairCount());
        const c = t.classes[class];
        var two: [2]usize = undefined;
        const bound = cells.sides(c, if (c.routed) shape.items else @intCast(count), &two);
        const ci = try t.s.addCell(.{ .class = class, .count = count, .op = op, .shape = shape, .qualified = reads >= read_target, .bound = bound });
        var out = try Buffer.alloc(t.rig.base, p.outBytes(false));
        defer out.free();
        for (t.rig.kernels.reg.entries, 0..) |*e, i| {
            if (e.format != .mlx or e.op != op or e.path != .prefill) continue;
            if (!e.caps(t.env) or !e.policy(t.env) or !e.fits(prod.shapeFor(arg, p.items, p.pairCount(), e))) continue;
            // a noisy timing is taken again, up to twice, and the steadiest kept; one still noisy cannot move a bound
            var r = try t.timeEntry(p, e, out);
            for (0..2) |_| {
                if (r.noise <= search.noise_limit) break;
                const again = try t.timeEntry(p, e, out);
                if (again.noise < r.noise) r = again;
            }
            t.s.setTime(ci, i, .{ .us = r.us, .noise = r.noise });
            std.debug.print("TUNE {s} n{d} k{d} b{d} g{d} {s}={d} items={d} pairs={d} {s} {d:.2} us noise {d:.1}%{s}\n", .{
                c.name, c.n,  c.k,           c.bits,                                                    c.group, if (c.routed) "tokens" else "rows", count, p.items, p.pairCount(),
                e.id,   r.us, r.noise * 100, if (t.s.cells.items[ci].qualified) "" else " UNQUALIFIED",
            });
        }
    }

    /// The median launch time over `batches` batches of whole rotations, and the spread of the batch means less one.
    fn timeEntry(t: *Tuner, p: *const Product, e: *const Entry, out: Buffer) !struct { us: f64, noise: f64 } {
        var start = try Event.init(&t.rig.base.r, true);
        defer start.deinit();
        var end = try Event.init(&t.rig.base.r, true);
        defer end.deinit();
        // a warm-up rotation, then batches of whole rotations grown until one measures at least `batch_us`
        const rotation = p.copies * ((8 + p.copies - 1) / p.copies);
        _ = try timeBatch(p, e, out, rotation, start, end);
        var per = rotation;
        while (true) {
            const took = try timeBatch(p, e, out, per, start, end);
            if (took >= batch_us) break;
            const reps = per / rotation;
            const need: usize = @intFromFloat(@ceil(batch_us / @max(took, 1.0) * @as(f64, @floatFromInt(reps))));
            per = rotation * @max(reps + 1, need);
        }
        var us: [batches]f64 = undefined;
        for (&us) |*u| u.* = try timeBatch(p, e, out, per, start, end) / @as(f64, @floatFromInt(per));
        std.mem.sort(f64, &us, {}, std.sort.asc(f64));
        return .{ .us = us[batches / 2], .noise = us[batches - 1] / us[0] - 1 };
    }

    /// Bisects between timed counts of a class wherever a choice changes, until adjacent or within the noise.
    fn refine(t: *Tuner, class: usize, par: *const Product, a: usize, b: usize) !void {
        if (b <= a + 1 or !t.s.changes(class, a, b)) return;
        if (t.s.close(class, a) and t.s.close(class, b)) {
            std.debug.print("INTERVAL {s} n{d} k{d}: {d} to {d} within the noise\n", .{ t.classes[class].name, t.classes[class].n, t.classes[class].k, a, b });
            return;
        }
        const m = a + (b - a) / 2;
        try t.timeCell(class, par, m);
        try t.refine(class, par, a, m);
        try t.refine(class, par, m, b);
    }

    fn countsOf(t: *const Tuner, class: usize, list: *std.ArrayList(usize)) !void {
        for (t.s.cells.items) |c| if (c.class == class) try list.append(gpa, c.count);
        std.mem.sort(usize, list.items, {}, std.sort.asc(usize));
    }
};

fn batch(p: *const Product, e: *const Entry, out: Buffer, n: usize) !void {
    for (0..n) |i| try prod.launchEntry(p, e, out, false, 0, i % p.copies);
}

/// The GPU time of `n` launches between two events, in microseconds.
fn timeBatch(p: *const Product, e: *const Entry, out: Buffer, n: usize, start: Event, end: Event) !f64 {
    try start.record(p.rig.base.stream);
    try batch(p, e, out, n);
    try end.record(p.rig.base.stream);
    try end.synchronize();
    return @as(f64, try Event.elapsedMs(start, end)) * 1000.0;
}

/// One `.zon` row as the tables write it.
fn printRow(r: tuning.Row) void {
    std.debug.print("ZON .{{ .id = \"{s}\", .cost = {d}", .{ r.id, r.cost });
    if (r.rows != 0) std.debug.print(", .rows = {d}", .{r.rows});
    if (r.min_rows != 0) std.debug.print(", .min_rows = {d}", .{r.min_rows});
    if (r.items != 0) std.debug.print(", .items = {d}", .{r.items});
    if (r.pairs_per_item != 0) std.debug.print(", .pairs_per_item = {d}", .{r.pairs_per_item});
    if (!r.plain) std.debug.print(", .plain = false", .{});
    if (!r.routed) std.debug.print(", .routed = false", .{});
    std.debug.print(" }},\n", .{});
}

test "time the prompt tiles" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var t: Tuner = .{ .rig = Rig.open(b), .env = undefined, .classes = undefined, .s = undefined };
    t.env = t.rig.kernels.env();
    t.s = search.Search.init(gpa, t.env, &tuning.gfx1201);
    defer t.deinit();
    var class_buf: [cells.max_classes]cells.Class = undefined;
    t.classes = cells.classes(&class_buf);

    // the coarse sweep, a class at a time, each from one parent product with its table copies
    var most: [cells.max_classes]usize = undefined;
    for (t.classes, 0..) |c, ci| {
        var list: std.ArrayList(usize) = .empty;
        defer list.deinit(gpa);
        try cells.coarse(c, &list, gpa);
        most[ci] = std.mem.max(usize, list.items);
        var par = try t.parent(ci, most[ci]);
        defer par.deinit();
        for (list.items) |v| try t.timeCell(ci, &par, v);
    }
    // propose, then bisect where either table's choice changes; bisection only adds counts, so the passes end
    while (true) {
        t.s.reset();
        try t.s.propose(false);
        const before = t.s.cells.items.len;
        for (t.classes, 0..) |_, ci| {
            var list: std.ArrayList(usize) = .empty;
            defer list.deinit(gpa);
            try t.countsOf(ci, &list);
            var to_refine = false;
            for (list.items[0 .. list.items.len - 1], list.items[1..]) |x, y| {
                if (y <= cells.sampledMax(t.classes[ci]) and y > x + 1 and t.s.changes(ci, x, y)) to_refine = true;
            }
            // both sides of every place the proposal's routed choice changes, so no accepted bound rests on a guess
            var edge_list: std.ArrayList(usize) = .empty;
            defer edge_list.deinit(gpa);
            if (t.classes[ci].routed) {
                const table = t.s.proposal();
                const reg = affine.Registry.init(&table, &mlx_entries.all);
                const template = t.s.cells.items[t.s.cellOf(ci, list.items[0]).?].shape;
                try cells.edges(t.classes[ci], template, &reg, t.env, &edge_list, gpa);
            }
            var new_edges = false;
            for (edge_list.items) |v| if (t.s.cellOf(ci, v) == null) {
                new_edges = true;
            };
            if (!to_refine and !new_edges) continue;
            var par = try t.parent(ci, most[ci]);
            defer par.deinit();
            for (edge_list.items) |v| try t.timeCell(ci, &par, v);
            for (list.items[0 .. list.items.len - 1], list.items[1..]) |x, y| {
                if (y <= cells.sampledMax(t.classes[ci])) try t.refine(ci, &par, x, y);
            }
        }
        if (t.s.cells.items.len == before) break;
    }
    t.s.reset();
    try t.s.propose(true);
    for (t.s.rows[0..t.s.n_rows]) |r| if (cells.isPrompt(r.id, false) or cells.isPrompt(r.id, true)) printRow(r);

    // the compiled gfx1201 table against gfx1100's, cell by cell: the receipt once the measured rows are written
    const now = affine.Registry.init(&tuning.gfx1201, &mlx_entries.all);
    for (t.s.cells.items, 0..) |c, ci| {
        const x = t.s.old.select(t.env, .mlx, c.op, .prefill, c.shape) orelse continue;
        const y = now.select(t.env, .mlx, c.op, .prefill, c.shape) orelse continue;
        const tx = t.s.timeOf(ci, search.index(x)) orelse continue;
        const ty = t.s.timeOf(ci, search.index(y)) orelse continue;
        const cl = t.classes[c.class];
        std.debug.print("TABLES {s} n{d} k{d} b{d} g{d} {s}={d}: gfx1100 {s} {d:.2} us, gfx1201 {s} {d:.2} us, ratio {d:.3}\n", .{
            cl.name, cl.n, cl.k, cl.bits, cl.group, if (cl.routed) "tokens" else "rows", c.count, x.id, tx.us, y.id, ty.us, ty.us / tx.us,
        });
    }
}
