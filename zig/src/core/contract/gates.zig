//! Gates 1-4 over a registered kernel's hooks (gate 5, the limits, is limits.zig): each prints one line a kernel.
const std = @import("std");
const c = @import("contract.zig");
const rows = @import("rows.zig");

pub const Fill = rows.Fill;
pub const Order = rows.Order;

pub const Gate = c.Gate;

/// Host tests of the gates themselves set this, so a planted fault's FAIL line never reads as a real one.
pub var quiet = false;

pub const Verdict = enum { pass, fail, na };

/// A registered kernel: its spec, its test's hooks over the buffers it owns, and (a planted control) the gate it fails.
pub const Kernel = struct {
    spec: c.Spec,
    ctx: *anyopaque,
    vt: *const VTable,
    control: ?Gate = null,
};

/// The hooks a kernel's test gives the harness; the harness owns no device memory.
pub const VTable = struct {
    /// Inputs for a shape filled as asked (seeded), outputs poisoned and guarded; false when the shape is refused.
    prepare: *const fn (*anyopaque, c.Shape, Fill, u64) anyerror!bool,
    /// One launch on the prepared buffers, waited for.
    run: *const fn (*anyopaque) anyerror!void,
    /// The bytes the last run wrote, on the host.
    output: *const fn (*anyopaque) anyerror![]const u8,
    /// Every input's bytes now, on the host: a run must leave them as prepared.
    inputs: *const fn (*anyopaque) anyerror![]const u8,
    /// The guard bytes around every output still hold their pattern.
    intact: *const fn (*anyopaque) anyerror!bool,
    /// Some output range overlaps some input range in the call as bound.
    overlap: *const fn (*anyopaque) bool,
    /// Output elements of the prepared call.
    count: *const fn (*anyopaque) usize,
    /// Float64 outputs of the prepared inputs, and each one's sums' term magnitude (the bound's mag).
    reference: *const fn (*anyopaque, []f64, []f64) anyerror!void,
    /// A derived absolute error allowance for every output; null retains the registered generic formula.
    bounds: ?*const fn (*anyopaque, []const f64, []const f64, []f64) anyerror!void = null,
    /// The outputs if every sum ran in an accumulator, in an order (gate 4); null: no sums to simulate.
    simulate: ?*const fn (*anyopaque, c.Accum, Order, []f64) anyerror!void = null,
    /// A raw output or a stated NaN rule checks itself at a gate: null when it holds, else what failed.
    check: ?*const fn (*anyopaque, Gate) anyerror!?[]const u8 = null,
    /// A replacement's baseline: the replaced kernel's bytes on the prepared inputs.
    replaced: ?*const fn (*anyopaque) anyerror![]const u8 = null,
};

/// The VTable of a test type T: its prepare, run, count and reference, its `rig`'s bytes, and the optional hooks.
pub fn hooks(comptime T: type) VTable {
    const H = struct {
        fn of(ctx: *anyopaque) *T {
            return @ptrCast(@alignCast(ctx));
        }
        fn prepare(ctx: *anyopaque, shape: c.Shape, fill: Fill, seed: u64) anyerror!bool {
            return of(ctx).prepare(shape, fill, seed);
        }
        fn run(ctx: *anyopaque) anyerror!void {
            return of(ctx).run();
        }
        fn output(ctx: *anyopaque) anyerror![]const u8 {
            return of(ctx).rig.outputBytes();
        }
        fn inputs(ctx: *anyopaque) anyerror![]const u8 {
            return of(ctx).rig.inputBytes();
        }
        fn intact(ctx: *anyopaque) anyerror!bool {
            return of(ctx).rig.intact();
        }
        fn overlap(ctx: *anyopaque) bool {
            return of(ctx).rig.overlap();
        }
        fn count(ctx: *anyopaque) usize {
            return of(ctx).count();
        }
        fn reference(ctx: *anyopaque, want: []f64, mag: []f64) anyerror!void {
            return of(ctx).reference(want, mag);
        }
        fn bounds(ctx: *anyopaque, want: []const f64, mag: []const f64, allow: []f64) anyerror!void {
            return of(ctx).bounds(want, mag, allow);
        }
        fn simulate(ctx: *anyopaque, acc: c.Accum, order: Order, out: []f64) anyerror!void {
            return of(ctx).simulate(acc, order, out);
        }
        fn check(ctx: *anyopaque, at: Gate) anyerror!?[]const u8 {
            return of(ctx).check(at);
        }
        fn replaced(ctx: *anyopaque) anyerror![]const u8 {
            return of(ctx).replaced();
        }
    };
    return .{
        .prepare = H.prepare,
        .run = H.run,
        .output = H.output,
        .inputs = H.inputs,
        .intact = H.intact,
        .overlap = H.overlap,
        .count = H.count,
        .reference = H.reference,
        .bounds = if (@hasDecl(T, "bounds")) H.bounds else null,
        .simulate = if (@hasDecl(T, "simulate")) H.simulate else null,
        .check = if (@hasDecl(T, "check")) H.check else null,
        .replaced = if (@hasDecl(T, "replaced")) H.replaced else null,
    };
}

fn say(v: Verdict, at: Gate, name: []const u8, comptime fmt: []const u8, args: anytype) Verdict {
    const word = switch (v) {
        .pass => "PASS",
        .fail => "FAIL",
        .na => "N/A ",
    };
    if (!quiet) std.debug.print("{s} {t} {s}: " ++ fmt ++ "\n", .{ word, at, name } ++ args);
    return v;
}

/// A shape for the single-shape gates: the first served one with `lo` to `hi` rows.
fn small(s: c.Spec, lo: usize, hi: usize) c.Shape {
    var x = s.served[0];
    x.rows = std.math.clamp(x.rows, lo, hi);
    return x;
}

/// Gate 1: NaN and both infinities at known inputs; every output's class equals the float64 reference's.
pub fn nonfinite(a: std.mem.Allocator, k: Kernel, seed: u64) !Verdict {
    const name = k.spec.name;
    if (!try k.vt.prepare(k.ctx, small(k.spec, 5, 8), .nonfinite, seed)) return say(.fail, .nonfinite, name, "refused a served shape", .{});
    try k.vt.run(k.ctx);
    if (k.vt.check) |own| {
        if (try own(k.ctx, .nonfinite)) |why| return say(.fail, .nonfinite, name, "{s}", .{why});
        const rule = switch (k.spec.nan) {
            .stated => |text| text,
            .propagate => "outputs equal the float64 reference's bytes",
        };
        return say(.pass, .nonfinite, name, "the stated rule holds: {s}", .{rule});
    }
    const n = k.vt.count(k.ctx);
    const want = try a.alloc(f64, n);
    defer a.free(want);
    const mag = try a.alloc(f64, n);
    defer a.free(mag);
    try k.vt.reference(k.ctx, want, mag);
    const out = try k.vt.output(k.ctx);
    var planted: usize = 0;
    var differ: usize = 0;
    var first: usize = n;
    for (want, 0..) |w, i| {
        if (c.classOf(w) != .finite) planted += 1;
        if (c.classOf(c.read(k.spec.out, out, i)) != c.classOf(w)) {
            differ += 1;
            if (first == n) first = i;
        }
    }
    if (planted == 0 or planted == n) return say(.fail, .nonfinite, name, "the planted values make {d} of {d} reference outputs non-finite: no proof", .{ planted, n });
    if (differ > 0) return say(.fail, .nonfinite, name, "{d} of {d} outputs differ in class; first at {d}: {t} where float64 gives {t}", .{ differ, n, first, c.classOf(c.read(k.spec.out, out, first)), c.classOf(want[first]) });
    return say(.pass, .nonfinite, name, "{d} non-finite and {d} finite outputs match the float64 classes", .{ planted, n - planted });
}

/// Gate 2: five runs give the same bytes, leave the inputs and guards as they were, and bind no output over an input.
pub fn repeat(a: std.mem.Allocator, k: Kernel, seed: u64) !Verdict {
    const name = k.spec.name;
    if (!try k.vt.prepare(k.ctx, small(k.spec, 1, 64), .random, seed)) return say(.fail, .repeat, name, "refused a served shape", .{});
    const before = try a.dupe(u8, try k.vt.inputs(k.ctx));
    defer a.free(before);
    try k.vt.run(k.ctx);
    const first = try a.dupe(u8, try k.vt.output(k.ctx));
    defer a.free(first);
    var differ: usize = 0;
    var at: usize = 0;
    for (1..5) |_| {
        try k.vt.run(k.ctx);
        const got = try k.vt.output(k.ctx);
        if (!std.mem.eql(u8, first, got)) {
            if (differ == 0) at = std.mem.indexOfDiff(u8, first, got).?;
            differ += 1;
        }
    }
    if (differ > 0) return say(.fail, .repeat, name, "{d} of 4 repeats differ from the first run's bytes, first at byte {d}", .{ differ, at });
    if (!std.mem.eql(u8, before, try k.vt.inputs(k.ctx))) return say(.fail, .repeat, name, "a run changed its inputs", .{});
    if (!try k.vt.intact(k.ctx)) return say(.fail, .repeat, name, "a run wrote outside its outputs", .{});
    if (k.vt.overlap(k.ctx) and !k.spec.in_place) return say(.fail, .repeat, name, "an output overlaps an input and the spec does not declare in place", .{});
    return say(.pass, .repeat, name, "five runs byte-identical ({d} bytes), inputs and guards untouched, no overlap", .{first.len});
}

/// Gate 3: the served shapes and their edges against float64 within the declared bound, or a replaced kernel's bits.
pub fn shapes(a: std.mem.Allocator, k: Kernel, seed: u64) !Verdict {
    const name = k.spec.name;
    const list = try c.shapes(a, k.spec);
    defer a.free(list);
    var worst: f64 = 0;
    var ran: usize = 0;
    var refused: usize = 0;
    for (list, 0..) |shape, si| {
        if (!try k.vt.prepare(k.ctx, shape, .random, seed + si)) {
            if (si < k.spec.served.len) return say(.fail, .shapes, name, "refused served shape {d}x{d}x{d}", .{ shape.rows, shape.n, shape.k });
            refused += 1;
            continue;
        }
        try k.vt.run(k.ctx);
        ran += 1;
        const out = try k.vt.output(k.ctx);
        if (k.spec.replaces) |what| {
            const base = k.vt.replaced orelse return say(.fail, .shapes, name, "replaces {s} but its test gives no baseline", .{what});
            const want = try base(k.ctx);
            if (!std.mem.eql(u8, out, want)) return say(.fail, .shapes, name, "at {d}x{d}x{d} its bytes differ from {s}'s from byte {d}", .{ shape.rows, shape.n, shape.k, what, std.mem.indexOfDiff(u8, out, want).? });
            continue;
        }
        if (k.vt.check) |own| {
            if (try own(k.ctx, .shapes)) |why| return say(.fail, .shapes, name, "at {d}x{d}x{d}: {s}", .{ shape.rows, shape.n, shape.k, why });
            continue;
        }
        const n = k.vt.count(k.ctx);
        const want = try a.alloc(f64, n);
        defer a.free(want);
        const mag = try a.alloc(f64, n);
        defer a.free(mag);
        try k.vt.reference(k.ctx, want, mag);
        const allow = try a.alloc(f64, n);
        defer a.free(allow);
        try allowances(k, shape, want, mag, allow, false);
        for (want, allow, 0..) |w, limit, i| {
            const got = c.read(k.spec.out, out, i);
            const q = if (got == w) 0 else @abs(got - w) / limit;
            if (!(q <= 1)) return say(.fail, .shapes, name, "at {d}x{d}x{d} output {d} is {e:.6} where float64 gives {e:.6}: {d:.2} of its bound", .{ shape.rows, shape.n, shape.k, i, got, w, q });
            worst = @max(worst, q);
        }
    }
    const how = if (k.spec.replaces != null) "bit-equal to the kernel it replaces" else if (k.vt.check != null) "exact by its own check" else "within the float64 bound";
    return say(.pass, .shapes, name, "{d} shapes {s} (worst {d:.3} of the bound), {d} edges refused by the kernel", .{ ran, how, worst, refused });
}

fn allowances(k: Kernel, shape: c.Shape, want: []const f64, mag: []const f64, allow: []f64, exact: bool) !void {
    for (allow, want, mag) |*l, w, m| l.* = if (exact) c.boundExact(k.spec, w) else c.bound(k.spec, shape, w, m);
    if (k.vt.bounds) |bound| {
        try bound(k.ctx, want, mag, allow);
        for (allow) |l| if (!std.math.isFinite(l) or l < 0) return error.InvalidErrorBound;
    }
}

/// The worst |got - want| / allow over a row of outputs; a non-finite where float64 is finite counts as infinite.
fn misses(got: []const f64, want: []const f64, allow: []const f64) f64 {
    var most: f64 = 0;
    for (got, want, allow) |g, w, l| most = @max(most, if (g == w) 0 else if (std.math.isFinite(g) or !std.math.isFinite(w)) @abs(g - w) / l else std.math.inf(f64));
    return most;
}

/// Gate 4: sums built so fp32 is exact in any order; fp16 and bf16 sums proven to miss first, then the kernel run.
pub fn sums(a: std.mem.Allocator, k: Kernel, seed: u64) !Verdict {
    const s = k.spec;
    if (s.accum != .f32) return say(.na, .sums, s.name, "declares {t} sums", .{s.accum});
    const sim = k.vt.simulate orelse return say(.fail, .sums, s.name, "declares fp32 sums but its test simulates none", .{});
    var shape = s.served[0];
    for (s.served) |x| if (x.k > shape.k) {
        shape = x;
    };
    shape.rows = @min(shape.rows, 4);
    if (!try k.vt.prepare(k.ctx, shape, .stress, seed)) return say(.fail, .sums, s.name, "refused a served shape", .{});
    const n = k.vt.count(k.ctx);
    const buf = try a.alloc(f64, 4 * n);
    defer a.free(buf);
    const want = buf[0..n];
    const allow = buf[n .. 2 * n];
    const got = buf[2 * n .. 3 * n];
    const mag = buf[3 * n ..];
    try k.vt.reference(k.ctx, want, mag);
    try allowances(k, shape, want, mag, allow, s.sums.exact);
    var margin = [2]f64{ std.math.inf(f64), std.math.inf(f64) };
    for ([_]Order{ .sequential, .pairwise, .strided }) |o| {
        try sim(k.ctx, .f32, o, got);
        if (!(misses(got, want, allow) <= 1)) return say(.fail, .sums, s.name, "fp32 sums of its stress inputs leave the bound ({t} order)", .{o});
        for ([_]c.Accum{ .f16, .bf16 }, &margin) |acc, *m| {
            try sim(k.ctx, acc, o, got);
            m.* = @min(m.*, misses(got, want, allow));
        }
    }
    const need = [2]f64{ s.sums.f16, s.sums.bf16 };
    for (margin, need, [_][]const u8{ "fp16", "bf16" }) |m, r, what| if (m < r)
        return say(.fail, .sums, s.name, "its stress inputs make {s} sums miss by only {d:.1}x (needs {d:.0}x)", .{ what, m, r });
    try k.vt.run(k.ctx);
    const out = try k.vt.output(k.ctx);
    for (got, 0..) |*g, i| g.* = c.read(s.out, out, i);
    const q = misses(got, want, allow);
    if (!(q <= 1)) return say(.fail, .sums, s.name, "{d:.1}x over its exact-sum bound at k {d}: not fp32 sums (fp16 would miss {d:.0}x, bf16 {d:.0}x)", .{ q, shape.k, margin[0], margin[1] });
    if (need[1] == 0 or need[0] == 0) return say(.pass, .sums, s.name, "within its bound at k {d}; fp16 sums would miss {d:.0}x, bf16 {d:.1}x ({s})", .{ shape.k, margin[0], margin[1], s.sums.why });
    return say(.pass, .sums, s.name, "exact sums at k {d}, where fp16 sums would miss by {d:.0}x and bf16 by {d:.0}x", .{ shape.k, margin[0], margin[1] });
}

/// A kernel's or control's gates 1-4; a control must fail its own gate and passes when it does.
pub const Tally = struct {
    kernels: usize = 0,
    passed: usize = 0,
    failed: usize = 0,
    controls: usize = 0,
    caught: usize = 0,
    missed: usize = 0,
    known: usize = 0,

    pub fn problems(t: Tally) usize {
        return t.failed + t.missed;
    }
};

pub fn one(a: std.mem.Allocator, which: Gate, k: Kernel, seed: u64) !Verdict {
    return switch (which) {
        .nonfinite => nonfinite(a, k, seed),
        .repeat => repeat(a, k, seed),
        .shapes => shapes(a, k, seed),
        .sums => sums(a, k, seed),
        .limits => .na,
    };
}

/// Gates 1-4 on each kernel, and each control at its own gate (which must fail).
pub fn runAll(a: std.mem.Allocator, kernels: []const Kernel, seed: u64, t: *Tally) !void {
    for (kernels, 0..) |k, i| {
        const s = seed +% 0x9e37 *% i;
        if (k.control) |g| {
            t.controls += 1;
            const v = one(a, g, k, s) catch |err| blk: {
                if (!quiet) std.debug.print("CONTROL ERROR {s}: {t} at {t}, so it proves nothing\n", .{ k.spec.name, err, g });
                break :blk .na;
            };
            if (v == .fail) {
                t.caught += 1;
                if (!quiet) std.debug.print("CONTROL {s}: red at {t}, as planted\n", .{ k.spec.name, g });
            } else {
                t.missed += 1;
                if (!quiet) std.debug.print("CONTROL MISSED {s}: {t} did not go red\n", .{ k.spec.name, g });
            }
            continue;
        }
        t.kernels += 1;
        var bad = false;
        for ([_]Gate{ .nonfinite, .repeat, .shapes, .sums }) |g| {
            const v = one(a, g, k, s) catch |err| blk: {
                if (!quiet) std.debug.print("FAIL {t} {s}: {t}\n", .{ g, k.spec.name, err });
                break :blk .fail;
            };
            const known = if (k.spec.known) |kn| kn.gate == g else false;
            if (known and v == .fail) {
                t.known += 1;
                if (!quiet) std.debug.print("KNOWN {t} {s}: {s}\n", .{ g, k.spec.name, k.spec.known.?.finding });
            } else if (known) {
                bad = true;
                if (!quiet) std.debug.print("FAIL {t} {s}: passes now; drop its known finding\n", .{ g, k.spec.name });
            } else if (v == .fail) bad = true;
        }
        if (bad) t.failed += 1 else t.passed += 1;
    }
}
