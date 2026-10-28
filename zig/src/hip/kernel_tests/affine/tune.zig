//! hip-affine-tune: every MLX affine prompt tile the GPU has, timed on the served cells, and proposed table rows.

const std = @import("std");
const base = @import("../rig.zig");
const affine = @import("../../launches/affine.zig");
const mlx_entries = @import("../../launches/mlx_entries.zig");
const registry = @import("core").registry;
const tuning = @import("../../tuning/tuning.zig");
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
/// The noise above which a point cannot move a bound.
const noise_limit = 0.05;

const Cell = struct { class: usize, count: usize, op: registry.Op, shape: registry.Shape, qualified: bool };
const Time = struct { cell: usize, entry: usize, us: f64, noise: f64 };
const Score = struct { loss: f64, saving: f64, changed: usize };

/// What a bound moves: a plain tile's rows, those rows with 0 for off (`plain = false`), or a routed tile's items.
const Kind = enum { rows, gate_rows, items };
const Bound = struct { id: []const u8, kind: Kind };
/// The plain K-parallel tile of plans first, so a cheaper tile's rows cannot hide it; then by cost.
const bounds = [_]Bound{
    .{ .id = "mlx.project.prefill.kp3", .kind = .gate_rows },
    .{ .id = "mlx.project.prefill.kp0", .kind = .gate_rows },
    .{ .id = "mlx.project.prefill.kp1", .kind = .rows },
    .{ .id = "mlx.project.prefill.kp2", .kind = .rows },
    .{ .id = "mlx.routed.prefill.kp3", .kind = .items },
};

const Tuner = struct {
    rig: Rig,
    env: registry.Env,
    classes: []const cells.Class,
    cells: std.ArrayList(Cell) = .empty,
    times: std.ArrayList(Time) = .empty,
    old: affine.Registry,
    rows: [registry.max_entries]tuning.Row = undefined,
    n_rows: usize = 0,

    fn deinit(t: *Tuner) void {
        t.cells.deinit(gpa);
        t.times.deinit(gpa);
    }

    fn proposal(t: *const Tuner) tuning.Table {
        return .{ .gfx = "proposal", .rows = t.rows[0..t.n_rows] };
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

    fn timed(t: *const Tuner, class: usize, count: usize) bool {
        for (t.cells.items) |c| if (c.class == class and c.count == count) return true;
        return false;
    }

    /// Times every prompt entry the GPU has on `count` rows or tokens of the parent's class.
    fn timeCell(t: *Tuner, class: usize, par: *const Product, count: usize) !void {
        if (t.timed(class, count)) return;
        var view: ?Product = if (count == par.c.rows) null else try Product.prefix(par, count);
        defer if (view) |*v| v.deinit();
        const p: *const Product = if (view) |*v| v else par;
        const op: registry.Op = if (p.c.routed()) .routed else .project;
        const arg = p.arg(0, 0);
        const reads = @as(f64, @floatFromInt(p.copies)) * p.bytes();
        var shape = affine.Kernels.shapeOf(arg, @intCast(p.items));
        shape.plan_pairs = @intCast(p.pairCount());
        try t.cells.append(gpa, .{ .class = class, .count = count, .op = op, .shape = shape, .qualified = reads >= read_target });
        const ci = t.cells.items.len - 1;
        var out = try Buffer.alloc(t.rig.base, p.outBytes(false));
        defer out.free();
        for (t.rig.kernels.reg.entries, 0..) |*e, i| {
            if (e.format != .mlx or e.op != op or e.path != .prefill) continue;
            if (!e.caps(t.env) or !e.policy(t.env) or !e.fits(prod.shapeFor(arg, p.items, p.pairCount(), e))) continue;
            const r = try t.timeEntry(p, e, out);
            try t.times.append(gpa, .{ .cell = ci, .entry = i, .us = r.us, .noise = r.noise });
            const c = t.classes[class];
            std.debug.print("TUNE {s} n{d} k{d} b{d} g{d} {s}={d} items={d} {s} {d:.2} us noise {d:.1}%{s}\n", .{
                c.name, c.n,  c.k,           c.bits,                                                  c.group, if (c.routed) "tokens" else "rows", count, p.items,
                e.id,   r.us, r.noise * 100, if (t.cells.items[ci].qualified) "" else " UNQUALIFIED",
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

    fn timeOf(t: *const Tuner, cell: usize, entry: usize) ?Time {
        for (t.times.items) |x| if (x.cell == cell and x.entry == entry) return x;
        return null;
    }

    /// The candidate table's score against gfx1100's choices, or null when the per-cell veto refuses it.
    fn score(t: *const Tuner, table: *const tuning.Table) ?Score {
        const cand = affine.Registry.init(table, &mlx_entries.all);
        var s: Score = .{ .loss = 0, .saving = 0, .changed = 0 };
        for (t.cells.items, 0..) |c, ci| {
            const a = t.old.select(t.env, .mlx, c.op, .prefill, c.shape) orelse continue;
            const b = cand.select(t.env, .mlx, c.op, .prefill, c.shape) orelse return null;
            if (a == b) continue;
            if (!c.qualified) return null;
            const ta = t.timeOf(ci, index(a)) orelse return null;
            const tb = t.timeOf(ci, index(b)) orelse return null;
            if (ta.noise > noise_limit or tb.noise > noise_limit) return null;
            const ratio = tb.us / ta.us;
            if (ratio > 1 + @max(ta.noise, tb.noise)) return null;
            s.loss = @max(s.loss, ratio - 1);
            s.saving += ta.us - tb.us;
            s.changed += 1;
        }
        return s;
    }

    /// Moves each bound in turn to its best accepted value: least loss, then most saving, then nearest gfx1100's.
    fn propose(t: *Tuner, print: bool) void {
        for (bounds) |bd| {
            const at = rowIndex(t.rows[0..t.n_rows], bd.id) orelse continue;
            const was = field(t.rows[at], bd.kind);
            // each candidate starts from the row as it was, so an off value keeps its other fields
            const orig = t.rows[at];
            const first = tuning.find(&tuning.gfx1100, bd.id).?;
            const old_v = field(first, bd.kind);
            var best_v = was;
            var best: Score = t.score(&t.proposal()) orelse .{ .loss = 0, .saving = 0, .changed = 0 };
            var values: [1024]u32 = undefined;
            for (t.candidates(bd.kind, &values)) |v| {
                t.rows[at] = orig;
                setField(&t.rows[at], bd.kind, v);
                const s = t.score(&t.proposal()) orelse continue;
                const better = s.loss < best.loss or (s.loss == best.loss and (s.saving > best.saving or
                    (s.saving == best.saving and dist(v, old_v) < dist(best_v, old_v))));
                if (better) {
                    best = s;
                    best_v = v;
                }
            }
            t.rows[at] = orig;
            setField(&t.rows[at], bd.kind, best_v);
            if (print) std.debug.print("BOUND {s} {t} {d} -> {d}: {d} cells changed, loss {d:.3}, saving {d:.2} us\n", .{
                bd.id, bd.kind, old_v, best_v, best.changed, best.loss, best.saving,
            });
        }
    }

    /// Bound values to try: off for a gate, and the rows or items of timed cells whose next count refinement reaches.
    fn candidates(t: *const Tuner, kind: Kind, buf: []u32) []u32 {
        const routed = kind == .items;
        var len: usize = 0;
        if (kind == .gate_rows) {
            buf[0] = 0;
            len = 1;
        }
        for (t.cells.items) |c| {
            if ((c.op == .routed) != routed or len == buf.len) continue;
            const v: u32 = if (routed) c.shape.items else @intCast(c.count);
            var two: [2]usize = undefined;
            if (!cells.sides(t.classes[c.class], v, &two)) continue;
            if (std.mem.indexOfScalar(u32, buf[0..len], v) == null) {
                buf[len] = v;
                len += 1;
            }
        }
        return buf[0..len];
    }

    /// Whether the choice between two timed counts of a class changes, under gfx1100's table or the proposal.
    fn changes(t: *const Tuner, class: usize, a: usize, b: usize) bool {
        const table = t.proposal();
        const cand = affine.Registry.init(&table, &mlx_entries.all);
        const ca = t.cellOf(class, a) orelse return false;
        const cb = t.cellOf(class, b) orelse return false;
        const c = t.cells.items;
        const oa = t.old.select(t.env, .mlx, c[ca].op, .prefill, c[ca].shape);
        const ob = t.old.select(t.env, .mlx, c[cb].op, .prefill, c[cb].shape);
        const pa = cand.select(t.env, .mlx, c[ca].op, .prefill, c[ca].shape);
        const pb = cand.select(t.env, .mlx, c[cb].op, .prefill, c[cb].shape);
        if (oa != ob or pa != pb) return true;
        // the two tables' choices trade places in speed between the counts
        return t.lead(ca, oa, pa) != t.lead(cb, ob, pb);
    }

    /// Whether gfx1100's choice and the proposal's at a count differ by less than both their noises.
    fn close(t: *const Tuner, class: usize, count: usize) bool {
        const table = t.proposal();
        const cand = affine.Registry.init(&table, &mlx_entries.all);
        const ci = t.cellOf(class, count) orelse return false;
        const c = t.cells.items[ci];
        const ta = t.timeOf(ci, index(t.old.select(t.env, .mlx, c.op, .prefill, c.shape) orelse return false)) orelse return false;
        const tb = t.timeOf(ci, index(cand.select(t.env, .mlx, c.op, .prefill, c.shape) orelse return false)) orelse return false;
        return @abs(ta.us - tb.us) < @min(ta.noise, tb.noise) * @min(ta.us, tb.us);
    }

    fn lead(t: *const Tuner, cell: usize, a: ?*const Entry, b: ?*const Entry) i2 {
        const ta = t.timeOf(cell, index(a orelse return 0)) orelse return 0;
        const tb = t.timeOf(cell, index(b orelse return 0)) orelse return 0;
        return if (ta.us < tb.us) -1 else if (ta.us > tb.us) 1 else 0;
    }

    fn cellOf(t: *const Tuner, class: usize, count: usize) ?usize {
        for (t.cells.items, 0..) |c, i| if (c.class == class and c.count == count) return i;
        return null;
    }

    /// Bisects between timed counts of a class wherever a choice changes, until adjacent or within the noise.
    fn refine(t: *Tuner, class: usize, par: *const Product, a: usize, b: usize) !void {
        if (b <= a + 1 or !t.changes(class, a, b)) return;
        if (t.close(class, a) and t.close(class, b)) {
            std.debug.print("INTERVAL {s} n{d} k{d}: {d} to {d} within the noise\n", .{ t.classes[class].name, t.classes[class].n, t.classes[class].k, a, b });
            return;
        }
        const m = a + (b - a) / 2;
        try t.timeCell(class, par, m);
        try t.refine(class, par, a, m);
        try t.refine(class, par, m, b);
    }

    fn countsOf(t: *const Tuner, class: usize, list: *std.ArrayList(usize)) !void {
        for (t.cells.items) |c| if (c.class == class) try list.append(gpa, c.count);
        std.mem.sort(usize, list.items, {}, std.sort.asc(usize));
    }
};

fn index(e: *const Entry) usize {
    return (@intFromPtr(e) - @intFromPtr(&mlx_entries.all[0])) / @sizeOf(Entry);
}

fn rowIndex(rows: []const tuning.Row, id: []const u8) ?usize {
    for (rows, 0..) |r, i| if (std.mem.eql(u8, r.id, id)) return i;
    return null;
}

fn field(r: tuning.Row, kind: Kind) u32 {
    return switch (kind) {
        .rows => r.rows,
        .gate_rows => if (r.plain) r.rows else 0,
        .items => r.items,
    };
}

fn setField(r: *tuning.Row, kind: Kind, v: u32) void {
    switch (kind) {
        .rows => r.rows = v,
        .gate_rows => {
            r.plain = v != 0;
            if (v != 0) r.rows = v;
        },
        .items => r.items = v,
    }
}

fn dist(a: u32, b: u32) u32 {
    return if (a > b) a - b else b - a;
}

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
    if (!r.plain) std.debug.print(", .plain = false", .{});
    if (!r.routed) std.debug.print(", .routed = false", .{});
    std.debug.print(" }},\n", .{});
}

test "time the prompt tiles" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var t: Tuner = .{ .rig = Rig.open(b), .env = undefined, .classes = undefined, .old = affine.Registry.init(&tuning.gfx1100, &mlx_entries.all) };
    defer t.deinit();
    t.env = t.rig.kernels.env();
    var class_buf: [cells.max_classes]cells.Class = undefined;
    t.classes = cells.classes(&class_buf);
    for (tuning.gfx1201.rows, 0..) |r, i| t.rows[i] = r;
    t.n_rows = tuning.gfx1201.rows.len;

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
        for (t.rows[0..t.n_rows]) |*r| r.* = tuning.find(&tuning.gfx1201, r.id).?;
        t.propose(false);
        const before = t.cells.items.len;
        for (t.classes, 0..) |_, ci| {
            var list: std.ArrayList(usize) = .empty;
            defer list.deinit(gpa);
            try t.countsOf(ci, &list);
            var to_refine = false;
            for (list.items[0 .. list.items.len - 1], list.items[1..]) |x, y| {
                if (y <= cells.sampledMax(t.classes[ci]) and y > x + 1 and t.changes(ci, x, y)) to_refine = true;
            }
            if (!to_refine) continue;
            var par = try t.parent(ci, most[ci]);
            defer par.deinit();
            for (list.items[0 .. list.items.len - 1], list.items[1..]) |x, y| {
                if (y <= cells.sampledMax(t.classes[ci])) try t.refine(ci, &par, x, y);
            }
        }
        if (t.cells.items.len == before) break;
    }
    for (t.rows[0..t.n_rows]) |*r| r.* = tuning.find(&tuning.gfx1201, r.id).?;
    t.propose(true);
    for (t.rows[0..t.n_rows]) |r| if (cells.isPrompt(r.id, false) or cells.isPrompt(r.id, true)) printRow(r);

    // the compiled gfx1201 table against gfx1100's, cell by cell: the receipt once the measured rows are written
    const now = affine.Registry.init(&tuning.gfx1201, &mlx_entries.all);
    for (t.cells.items, 0..) |c, ci| {
        const x = t.old.select(t.env, .mlx, c.op, .prefill, c.shape) orelse continue;
        const y = now.select(t.env, .mlx, c.op, .prefill, c.shape) orelse continue;
        const tx = t.timeOf(ci, index(x)) orelse continue;
        const ty = t.timeOf(ci, index(y)) orelse continue;
        const cl = t.classes[c.class];
        std.debug.print("TABLES {s} n{d} k{d} b{d} g{d} {s}={d}: gfx1100 {s} {d:.2} us, gfx1201 {s} {d:.2} us, ratio {d:.3}\n", .{
            cl.name, cl.n, cl.k, cl.bits, cl.group, if (cl.routed) "tokens" else "rows", c.count, x.id, tx.us, y.id, ty.us, ty.us / tx.us,
        });
    }
}
