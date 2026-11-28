//! Production-image GLPv3 additive device proof: FP64/FP32 refs, strides, controls, alias refusals and graph lifetime.

// Format and paired site: Matt Suiche/weightless (spec b181a07d3980fc48abe39d95ddb044459dd0b868).
const std = @import("std");
// Additive convention: llama.cpp; Arditi et al. (NeurIPS 2024) is background only. Run `tf-dsv41-test glp <dir>`.
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");
const ref = @import("glp_ref.zig");
const rig = @import("glp_rig.zig");
const fold = @import("glp_fold.zig");
const final = @import("glp_final.zig");
const controls = @import("glp_controls.zig");
const graph = @import("glp_graph.zig");
const program = @import("glp_program.zig");

const glp = rig.glp;

var counter: u64 = 0x1000;

fn nextSeed() u64 {
    counter += 1;
    return counter;
}

fn foldSuite(c: rig.Ctx) !void {
    var name: [64]u8 = undefined;
    for ([_]usize{ 5120, 128, 1032 }) |w| for ([_]usize{ 1, 3, 16, 17, 64, 255 }) |rows| {
        for ([_]bool{ false, true }) |strided| for ([_]bool{ false, true }) |tap| {
            if (w == 5120 and rows == 255 and !(strided and tap)) continue; // one layout at the widest case
            const label = try std.fmt.bufPrint(&name, "w{d} R{d} {s}{s}", .{ w, rows, if (strided) "strided" else "dense", if (tap) " tap" else "" });
            try fold.foldCase(c, .{
                .name = label,
                .width = w,
                .rows = rows,
                .xs = if (strided) 4 * w + 64 else 0,
                .tap = tap,
                .ts = if (strided and tap) w + 64 else 0,
                .alpha = 1.5,
                .seed = nextSeed(),
            });
        };
    };
}

/// Exercise ties, BF16 maxima, tail columns, nonfinite copies and nonfinite directions.
fn fillSuite(c: rig.Ctx) !void {
    for ([_]usize{ 5120, 1032 }) |w| {
        const seed = nextSeed();
        const o = try fold.foldRun(c, c.ops, .{ .name = "ties", .width = w, .rows = 17, .alpha = 1, .fill = .ties, .seed = seed });
        if (o.t.flips != 0) return rig.fail("ties w{d}: {d} bf16 ties rounded differently from FP64", .{ w, o.t.flips });
        try fold.report("fold", "add ties (every element on a bf16 tie)", o, seed, false, false);
    }
    const a = try fold.foldRun(c, c.ops, .{ .name = "extreme", .width = 128, .rows = 4, .alpha = 0.5, .fill = .extreme, .tap = true, .seed = nextSeed() });
    if (a.nonfinite != 0 or !a.ok()) return rig.fail("extreme copies (0x7f7f / 0xff7f): not finite and equal to FP64", .{});
    check.pass("glp fold extreme: four copies of the largest finite bf16 in both signs stay finite, {d} elements equal the FP64 math (no fp32 overflow of the mean)", .{a.t.n});
    for ([_]usize{ 1032, 5120 }) |w| try fold.foldCase(c, .{ .name = "last column edited", .width = w, .rows = 3, .tap = true, .witness = true, .seed = nextSeed() });
    for ([_]usize{ 5120, 1032 }) |w| try fold.foldCase(c, .{ .name = "nonfinite copies", .width = w, .rows = 6, .fill = .nonfinite, .tap = true, .seed = nextSeed() });
    try fold.foldCase(c, .{ .name = "NaN in the direction", .width = 1032, .rows = 4, .fault = .nan, .seed = nextSeed() });
    try fold.foldCase(c, .{ .name = "infinity in the direction", .width = 1032, .rows = 4, .fault = .inf, .seed = nextSeed() });
}

fn writerSuite(c: rig.Ctx) !void {
    var name: [64]u8 = undefined;
    for ([_]usize{ 1, 2, 3, 8 }) |world| for ([_]usize{ 5120, 1032 }) |w| for ([_]usize{ 1, 5, 17 }) |rows| for ([_]bool{ false, true }) |strided| {
        const label = try std.fmt.bufPrint(&name, "w{d} R{d} {s}", .{ w, rows, if (strided) "strided" else "dense" });
        try fold.writerCase(c, .{
            .name = label,
            .width = w,
            .rows = rows,
            .total = if (strided) rows + 3 else 0,
            .world = world,
            .gs = if (strided) w + 64 else 0,
            .gr_extra = if (strided) 64 else 0,
            .alpha = 1.5,
            .seed = nextSeed(),
        });
    };
    for ([_]usize{ 1, 2, 3, 8 }) |world| try fold.writerCase(c, .{ .name = "nonfinite partials", .width = 1032, .rows = 6, .world = world, .fill = .nonfinite, .seed = nextSeed() });
    // the cancellation fill makes the fixed rank order show in the bits; one rank has no order to show
    for ([_]usize{ 2, 3, 8 }) |world| try fold.writerCase(c, .{ .name = "rank cancellation", .width = 5120, .rows = 8, .world = world, .alpha = 1, .fill = .cancel, .seed = nextSeed() });
    for ([_]usize{ 1, 3 }) |world| try fold.writerCase(c, .{ .name = "extreme partials", .width = 128, .rows = 4, .world = world, .alpha = 0.5, .fill = .extreme, .seed = nextSeed() });
    for ([_]usize{ 1032, 5120 }) |w| try fold.writerCase(c, .{ .name = "last column edited", .width = w, .rows = 3, .world = 3, .witness = true, .seed = nextSeed() });
}

fn finalSuite(c: rig.Ctx, k: *const dsv41.Kernels) !void {
    var name: [64]u8 = undefined;
    for ([_]bool{ false, true }) |nw16| for ([_]usize{ 1, 3, 17 }) |rows| for ([_]usize{ 5120, 128 }) |w| for ([_]bool{ false, true }) |strided| {
        const label = try std.fmt.bufPrint(&name, "w{d} R{d} {s}", .{ w, rows, if (strided) "strided" else "dense" });
        try final.finalCase(c, .{ .name = label, .width = w, .rows = rows, .xs = if (strided) 4 * w + 64 else 0, .nw16 = nw16, .seed = nextSeed() });
    };
    for ([_]bool{ false, true }) |nw16| for ([_]usize{ 1, 3, 16 }) |rows| try final.mhcEqual(c, k, rows, nw16, nextSeed());
}

fn expectError(n: *usize, what: []const u8, want: anyerror, result: anytype) !void {
    if (result) |_| {
        return rig.fail("{s}: expected {t}, the call was accepted", .{ what, want });
    } else |e| {
        if (e != want) return rig.fail("{s}: expected {t}, got {t}", .{ what, want, e });
    }
    n.* += 1;
}

/// Reject invalid widths, alignment, strides and aliasing without changing any byte of real device buffers.
fn refusals(c: rig.Ctx) !void {
    const a = c.gpu.gpa;
    const W = 5120;
    const zeros = try a.alloc(u8, 2 * 4 * W * 2 + 256);
    defer a.free(zeros);
    @memset(zeros, 0x3c);
    var xb = try rig.Buf.init(c.gpu, zeros);
    defer xb.free();
    var db = try rig.Buf.init(c.gpu, zeros[0 .. W * 4]);
    defer db.free();
    const ops = c.ops;
    const x = xb.ptr();
    const d = db.ptr();
    var refused: usize = 0;
    const n = &refused;
    for ([_]usize{ 0, 4, 12, 1028, 5121, 5124, 5128, 6144, 10240, 10248 }) |w| { // 5120 is the widest row
        try expectError(n, "fold width", error.BadWidth, rig.fold(ops, x, 4 * W, 0, 0, d, 1, 2, w));
        try expectError(n, "writer width", error.BadWidth, rig.writer(ops, x, W, W, d, 1, 2, w, 2));
    }
    for (1..8) |extra| { // a partial last group of any size
        try expectError(n, "fold partial group", error.BadWidth, rig.fold(ops, x, 4 * W, 0, 0, d, 1, 2, 1024 + extra));
        try expectError(n, "writer partial group", error.BadWidth, rig.writer(ops, x, W, W, d, 1, 2, 1024 + extra, 2));
    }
    for ([_]usize{ 0, 5121, 5124, 5128, 130, 64, 5248, 10240, 10368 }) |w| {
        try expectError(n, "final width", error.BadWidth, c.ops.final(.{ .x = x, .xs = 4 * W, .pre = d, .nw = d, .out = x, .eps = 1e-6, .nwbf16 = 0, .R = 2, .width = @intCast(w) }));
    }
    try expectError(n, "fold stride below 4 width", error.BadStride, rig.fold(ops, x, 4 * W - 8, 0, 0, d, 1, 2, W));
    try expectError(n, "fold stride not a multiple of 8", error.BadStride, rig.fold(ops, x, 4 * W + 4, 0, 0, d, 1, 2, W));
    try expectError(n, "tap stride not a multiple of 8", error.Misaligned, rig.fold(ops, x, 4 * W, x, W + 4, d, 1, 2, W));
    try expectError(n, "fold x misaligned", error.Misaligned, rig.fold(ops, x + 8, 4 * W, 0, 0, d, 1, 2, W));
    try expectError(n, "fold dir misaligned", error.Misaligned, rig.fold(ops, x, 4 * W, 0, 0, d + 8, 1, 2, W));
    try expectError(n, "writer g misaligned", error.Misaligned, rig.writer(ops, x + 8, W, W, d, 1, 2, W, 2));
    try expectError(n, "writer row stride", error.BadStride, rig.writer(ops, x, W, W + 4, d, 1, 2, W, 2));
    try expectError(n, "writer rank stride", error.BadStride, rig.writer(ops, x, W + 4, W, d, 1, 2, W, 2));
    try expectError(n, "world 0", error.BadRanks, rig.writer(ops, x, W, W, d, 1, 2, W, 0));
    try expectError(n, "world 9", error.BadRanks, rig.writer(ops, x, W, W, d, 1, 2, W, 9));
    try expectError(n, "final pre misaligned", error.Misaligned, c.ops.final(.{ .x = x, .xs = 4 * W, .pre = d + 4, .nw = d, .out = x, .eps = 1e-6, .nwbf16 = 0, .R = 2, .width = W }));
    // the copies' bytes and the tap's never overlap; negative strides are never valid
    try expectError(n, "tap inside the copies", error.Overlap, rig.fold(ops, x, 4 * W, x + 32, W, d, 1, 2, W));
    try expectError(n, "tap reaching into the copies", error.Overlap, rig.fold(ops, x, 4 * W, x - 2 * W, W, d, 1, 2, W));
    try expectError(n, "negative fold stride", error.BadStride, ops.stream(.{ .x = x, .xs = -8 * W, .dir = d, .alpha = 1, .R = 2, .width = W }));
    try expectError(n, "negative tap stride", error.BadStride, ops.stream(.{ .x = x, .xs = 4 * W, .tap = x + 8 * W, .ts = -8, .dir = d, .alpha = 1, .R = 1, .width = W }));
    try expectError(n, "negative writer row stride", error.BadStride, ops.writer(.{ .g = x, .gr = 2 * W, .gs = -8, .dir = d, .alpha = 1, .R = 2, .width = W, .world = 2 }));
    try expectError(n, "rank stride 0", error.BadStride, rig.writer(ops, x, 0, W, d, 1, 2, W, 2));
    try expectError(n, "rank slots overlapping the rows", error.BadStride, rig.writer(ops, x, W, W, d, 1, 2, W, 2));
    try expectError(n, "negative rank stride", error.BadStride, ops.writer(.{ .g = x, .gr = -2 * W, .gs = W, .dir = d, .alpha = 1, .R = 2, .width = W, .world = 2 }));
    // the boundary cases the checks accept (strength 0: the kernels return before touching a byte)
    try rig.fold(ops, x, 4 * W, x + 8 * W, W, d, 0, 1, W); // the tap right after the copies
    try rig.writer(ops, x, 2 * W, W, d, 0, 2, W, 2); // the rank stride just clearing the rows
    try rig.writer(ops, x, 0, W, d, 0, 2, W, 1); // one rank: the rank stride is not read
    // a negative or non-finite strength never reaches the kernels (zero, either sign, is the kernel's own bypass)
    for ([_]f32{ -0.5, -std.math.inf(f32), std.math.inf(f32), std.math.nan(f32), -1e-30 }) |bad| {
        try expectError(n, "fold strength", error.BadStrength, rig.fold(ops, x, 4 * W, 0, 0, d, bad, 2, W));
        try expectError(n, "writer strength", error.BadStrength, rig.writer(ops, x, W, W, d, bad, 2, W, 2));
    }
    try c.sync();
    const after = try xb.read(a);
    defer a.free(after);
    const dafter = try db.read(a);
    defer a.free(dafter);
    if (!std.mem.eql(u8, after, zeros) or !std.mem.eql(u8, dafter, zeros[0 .. W * 4])) return rig.fail("a refused call changed a buffer", .{});
    if (!try xb.guards(a) or !try db.guards(a)) return rig.fail("a refused call wrote a canary band", .{});
    check.pass("glp refusals: {d} launches refused (partial last groups of 1-7 columns, widths above 5120, strides, tap overlap, rank slots, alignments, ranks, strengths), no buffer byte changed", .{refused});
}

/// Alpha zero must preserve every payload byte, including nonfinite values, taps and direction buffers.
fn alphaZero(c: rig.Ctx) !void {
    const a = c.gpu.gpa;
    const w = 1032;
    const xs = 4 * w + 16;
    const ts = w + 8;
    var rng = ref.Rng.init(0x7101);
    const x = try a.alloc(u16, 6 * xs);
    defer a.free(x);
    const tap = try a.alloc(u16, 6 * ts);
    defer a.free(tap);
    const g = try a.alloc(u16, 3 * 5 * (w + 8));
    defer a.free(g);
    const d = try a.alloc(f32, w);
    defer a.free(d);
    for ([_][]u16{ x, tap, g }) |s| for (s) |*v| {
        v.* = @truncate(rng.next());
    };
    for (0..6) |r| {
        x[r * xs + r] = ref.nan16 | @as(u16, @intCast(r)); // NaN payloads, infinities of both signs
        x[r * xs + 2 * w + r] = if (r % 2 == 0) ref.inf16 else ref.ninf16;
        tap[r * ts + r] = 0xffc1;
    }
    for (d, 0..) |*v, i| v.* = if (i % 97 == 0) std.math.nan(f32) else ref.dirEntry(&rng);
    var xb = try rig.Buf.init(c.gpu, rig.bytes(x));
    defer xb.free();
    var tb = try rig.Buf.init(c.gpu, rig.bytes(tap));
    defer tb.free();
    var gb = try rig.Buf.init(c.gpu, rig.bytes(g));
    defer gb.free();
    var db = try rig.Buf.init(c.gpu, rig.bytes(d));
    defer db.free();
    const neg_zero: f32 = -0.0;
    var launches: usize = 0;
    for ([_]f32{ 0.0, neg_zero }) |alpha| {
        try rig.fold(c.ops, xb.ptr(), xs, tb.ptr(), ts, db.ptr(), alpha, 6, w);
        try rig.writer(c.ops, gb.ptr(), 5 * (w + 8), w + 8, db.ptr(), alpha, 5, w, 3);
        try c.sync();
        launches += 2;
        for ([_]struct { b: rig.Buf, s: []const u8 }{ .{ .b = xb, .s = rig.bytes(x) }, .{ .b = tb, .s = rig.bytes(tap) }, .{ .b = gb, .s = rig.bytes(g) }, .{ .b = db, .s = rig.bytes(d) } }) |p| {
            const got = try p.b.read(a);
            defer a.free(got);
            if (!std.mem.eql(u8, got, p.s)) return rig.fail("alpha {e}: the bypass changed a byte (first at {d})", .{ alpha, std.mem.indexOfDiff(u8, got, p.s).? });
            if (!try p.b.guards(a)) return rig.fail("alpha {e}: the bypass wrote a canary band", .{alpha});
        }
    }
    check.pass("glp alpha 0: {d} launches (+0 and -0, fold with tap and writer) over NaN / Inf / random payload rows left every byte of x, g, tap and the direction unchanged", .{launches});
}

/// Rows of other slots in one buffer stay untouched, and two launches on adjacent row ranges equal one launch on both.
fn separation(c: rig.Ctx) !void {
    const a = c.gpu.gpa;
    const w = 1032;
    const xs = 4 * w + 64;
    const total = 14;
    const first = 4; // the steered slot's rows: [4, 10)
    const n = 6;
    var rng = ref.Rng.init(0x7201);
    const d = try a.alloc(f32, w);
    defer a.free(d);
    ref.fillDir(.random, &rng, d);
    const x = try a.alloc(u16, total * xs);
    defer a.free(x);
    for (x, 0..) |*v, i| v.* = if (i / xs < first) 0x4a00 | @as(u16, @truncate(i)) & 0xff else ref.padWord(i) ^ 0x2000; // slots with their own canaries
    for (first..first + n) |r| ref.fillRow(.random, &rng, r, x[r * xs ..][0 .. 4 * w], d);
    var db = try rig.Buf.init(c.gpu, rig.bytes(d));
    defer db.free();
    var xb = try rig.Buf.init(c.gpu, rig.bytes(x));
    defer xb.free();
    const row_bytes = xs * 2;
    try xb.write(rig.bytes(x));
    try rig.fold(c.ops, xb.ptr() + first * row_bytes, xs, 0, 0, db.ptr(), 0.75, n, w);
    try c.sync();
    const single = try xb.read(a);
    defer a.free(single);
    try xb.write(rig.bytes(x));
    try rig.fold(c.ops, xb.ptr() + first * row_bytes, xs, 0, 0, db.ptr(), 0.75, 2, w);
    try rig.fold(c.ops, xb.ptr() + (first + 2) * row_bytes, xs, 0, 0, db.ptr(), 0.75, n - 2, w);
    try c.sync();
    const split = try xb.read(a);
    defer a.free(split);
    if (!std.mem.eql(u8, single, split)) return rig.fail("separation: two launches on adjacent rows differ from one launch", .{});
    const ym = try a.alloc(u16, 4 * w);
    defer a.free(ym);
    var edited: usize = 0;
    for (0..total) |r| {
        const got = rig.words(single)[r * xs ..][0..xs];
        if (r < first or r >= first + n) {
            if (!std.mem.eql(u16, got, x[r * xs ..][0..xs])) return rig.fail("separation: row {d} of another slot changed", .{r});
            continue;
        }
        ref.foldRow(x[r * xs ..][0 .. 4 * w], d, 0.75, .{}, ym, null);
        for (ym, got[0 .. 4 * w]) |m, v| {
            if (!ref.same(v, m)) return rig.fail("separation: steered row {d} is off the model", .{r});
            edited += 1;
        }
        if (!std.mem.eql(u16, got[4 * w ..], x[r * xs + 4 * w ..][0 .. xs - 4 * w])) return rig.fail("separation: pad columns of row {d} changed", .{r});
    }
    // the writer: the same over the rows of a gathered buffer [world, total, gs] with a steered row range
    const world = 2;
    const gs = w + 8;
    const g = try a.alloc(u16, world * total * gs);
    defer a.free(g);
    for (g, 0..) |*v, i| v.* = if ((i / gs) % total < first) 0x4b00 | @as(u16, @truncate(i)) & 0xff else ref.spread(&rng);
    var gb = try rig.Buf.init(c.gpu, rig.bytes(g));
    defer gb.free();
    const gbytes = gs * 2;
    try gb.write(rig.bytes(g));
    try rig.writer(c.ops, gb.ptr() + first * gbytes, total * gs, gs, db.ptr(), 0.75, n, w, world);
    try c.sync();
    const wsingle = try gb.read(a);
    defer a.free(wsingle);
    try gb.write(rig.bytes(g));
    try rig.writer(c.ops, gb.ptr() + first * gbytes, total * gs, gs, db.ptr(), 0.75, 3, w, world);
    try rig.writer(c.ops, gb.ptr() + (first + 3) * gbytes, total * gs, gs, db.ptr(), 0.75, n - 3, w, world);
    try c.sync();
    const wsplit = try gb.read(a);
    defer a.free(wsplit);
    if (!std.mem.eql(u8, wsingle, wsplit)) return rig.fail("separation: two writer launches on adjacent rows differ from one launch", .{});
    var changed_other: usize = 0;
    for (0..world) |rk| for (0..total) |r| {
        if (r >= first and r < first + n) continue;
        const at = rk * total * gs + r * gs;
        if (!std.mem.eql(u16, rig.words(wsingle)[at..][0..gs], g[at..][0..gs])) changed_other += 1;
    };
    if (changed_other != 0) return rig.fail("separation: {d} writer rows of other slots changed", .{changed_other});
    check.pass("glp separation: a steered row range inside buffers holding other slots' rows (canaries of their own) leaves every other row and pad untouched;" ++
        " two launches on adjacent ranges equal one ({d} fold elements on the model; writer too)", .{edited});
}

pub fn run(gpu: check.Gpu, k: *dsv41.Kernels, dir: []const u8) !void {
    try k.loadGlp(gpu.d);
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads on the default stream
    defer stream.deinit();
    const ops = k.glpOps(stream) orelse return error.GlpKernelNotBuilt;
    const c: rig.Ctx = .{ .gpu = gpu, .stream = stream, .ops = ops };
    var ctl = try controls.Controls.load(c, dir, k.glp_fns.?); // required: a missing controls image fails the command
    defer ctl.unload();
    try foldSuite(c);
    try fillSuite(c);
    try writerSuite(c);
    try finalSuite(c, k);
    try refusals(c);
    try alphaZero(c);
    try separation(c);
    try controls.run(c, &ctl);
    try program.run(c, k);
    try graph.run(c);
    check.pass("glp: additive fold, writer, final (and mHC's final), repeats, nonfinite, tails, rank cancellation, separation, alpha 0, refusals, 6 planted controls, program controls on the real mHC post and the graph lifetime all green", .{});
}
