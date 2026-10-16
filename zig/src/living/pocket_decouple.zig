//! Pocket neurons kept from driving one another along the residual stream.
const std = @import("std");
const subspace = @import("subspace.zig");
const pocket_fit = @import("pocket_fit.zig");

const Kind = pocket_fit.Kind;
const Fit = pocket_fit.Fit;

/// Whether lessons keep pocket neurons decoupled (the default).
pub fn on() bool {
    const v = std.c.getenv("TF_SLIDE_POCKET_COUPLE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "on");
}

/// Each slot's read scale.
pub const Scales = struct {
    rows: std.ArrayList(f32) = .empty,

    pub fn deinit(s: *Scales, gpa: std.mem.Allocator) void {
        s.rows.deinit(gpa);
    }

    pub fn get(s: *Scales, p: anytype, t: anytype) ![]const f32 {
        const d = p.hidden;
        const n = p.moe.items.len;
        if (s.rows.items.len == n * d) return s.rows.items;
        s.rows.clearRetainingCapacity();
        try s.rows.appendNTimes(p.gpa, 1, n * d);
        const T = @TypeOf(t.*);
        if (comptime @hasDecl(T, "readScale")) {
            for (p.moe.items, 0..) |l, j| try t.readScale(l, s.rows.items[j * d ..][0..d]);
        }
        return s.rows.items;
    }
};

/// v less its part along the first `count` orthonormal rows of `basis` ([*, d]), twice for accuracy.
pub fn project(v: []f32, basis: []const f32, count: usize, d: usize) void {
    for (0..2) |_| for (0..count) |r| {
        const b = basis[r * d ..][0..d];
        subspace.axpy(v, -subspace.dot(v, b), b);
    };
}

/// `v` added to the orthonormal rows `basis` if it is not (nearly) in their span: Gram-Schmidt, run twice.
fn add(gpa: std.mem.Allocator, basis: *std.ArrayList(f32), v: []f32, d: usize) !bool {
    const before = @sqrt(subspace.dot(v, v));
    if (before == 0) return false;
    project(v, basis.items, basis.items.len / d, d);
    const len = @sqrt(subspace.dot(v, v));
    if (len <= 1e-4 * before) return false;
    for (v) |*x| x.* /= len;
    try basis.appendSlice(gpa, v);
    return true;
}

/// What the neurons above each slot read: orthonormal rows built from the top slot down.
pub const Above = struct {
    rows: std.ArrayList(f32) = .empty,
    above: std.ArrayList(usize) = .empty,

    pub fn deinit(a: *Above, gpa: std.mem.Allocator) void {
        a.rows.deinit(gpa);
        a.above.deinit(gpa);
    }

    pub fn build(a: *Above, p: anytype, scales: []const f32) !void {
        const d = p.hidden;
        const n = p.moe.items.len;
        a.rows.clearRetainingCapacity();
        a.above.clearRetainingCapacity();
        try a.above.appendNTimes(p.gpa, 0, n);
        const v = try p.gpa.alloc(f32, d);
        defer p.gpa.free(v);
        var j = n;
        while (j > 0) {
            j -= 1;
            a.above.items[j] = a.rows.items.len / d;
            for (0..p.used.items[j]) |k| {
                const u = p.neuron(p.up.items, j, k, d);
                for (v, u, scales[j * d ..][0..d]) |*o, x, s| o.* = x * s;
                _ = try add(p.gpa, &a.rows, v, d);
            }
        }
    }

    /// Slot j's output row `v` off what every neuron above it reads.
    pub fn keepOff(a: *const Above, v: []f32, j: usize, d: usize) void {
        project(v, a.rows.items, a.above.items[j], d);
    }
};

/// What the neurons below slot j write where it reads: orthonormal rows of s_j ⊙ v_i for every live output v_i at a.
pub fn below(p: anytype, gpa: std.mem.Allocator, scales: []const f32, j: usize, out: *std.ArrayList(f32)) !usize {
    const d = p.hidden;
    out.clearRetainingCapacity();
    const v = try gpa.alloc(f32, d);
    defer gpa.free(v);
    for (0..j) |i| for (0..p.used.items[i]) |k| {
        const w = p.neuron(p.down.items, i, k, d);
        for (v, w, scales[j * d ..][0..d]) |*o, x, s| o.* = x * s;
        _ = try add(gpa, out, v, d);
    };
    return out.items.len / d;
}

/// How input row `u` (unscaled, as a neuron multiplies x) sits on each kind of read row `xs`, as pocket_fit reports it.
pub fn evaluate(xs: []const []const u16, kinds: []const Kind, u: []const f32, gpa: std.mem.Allocator) !Fit {
    const copy = try gpa.dupe(f32, u);
    defer gpa.free(copy);
    return pocket_fit.finish(xs, kinds, copy, 1, 1);
}

test "an output kept off what the neurons above read moves none of their products; a slot's own and lower ones are free" {
    const gpa = std.testing.allocator;
    const d = 32;
    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    const P = struct {
        gpa: std.mem.Allocator,
        hidden: usize,
        moe: std.ArrayList(usize) = .empty,
        up: std.ArrayList(f32) = .empty,
        down: std.ArrayList(f32) = .empty,
        used: std.ArrayList(usize) = .empty,
        pub fn neuron(_: *const @This(), which: []f32, slot: usize, n: usize, dd: usize) []f32 {
            return which[(slot * 4 + n) * dd ..][0..dd];
        }
    };
    var p: P = .{ .gpa = gpa, .hidden = d };
    defer {
        p.moe.deinit(gpa);
        p.up.deinit(gpa);
        p.down.deinit(gpa);
        p.used.deinit(gpa);
    }
    try p.moe.appendSlice(gpa, &.{ 3, 5, 9 });
    try p.used.appendSlice(gpa, &.{ 2, 3, 1 });
    try p.up.appendNTimes(gpa, 0, 3 * 4 * d);
    try p.down.appendNTimes(gpa, 0, 3 * 4 * d);
    for (p.up.items, p.down.items) |*u, *v| {
        u.* = r.floatNorm(f32);
        v.* = r.floatNorm(f32);
    }
    const scales = try gpa.alloc(f32, 3 * d);
    defer gpa.free(scales);
    for (scales) |*s| s.* = 0.5 + r.float(f32);
    var a: Above = .{};
    defer a.deinit(gpa);
    try a.build(&p, scales);
    try std.testing.expectEqualSlices(usize, &.{ 4, 1, 0 }, a.above.items);
    // slot 0's first output, kept off: every neuron at slots 1 and 2 reads (s ⊙ u) . v = 0; slot 0's own do not
    const v = p.neuron(p.down.items, 0, 0, d);
    a.keepOff(v, 0, d);
    for (1..3) |j| for (0..p.used.items[j]) |k| {
        var dot: f32 = 0;
        for (p.neuron(p.up.items, j, k, d), scales[j * d ..][0..d], v) |u, s, x| dot += u * s * x;
        try std.testing.expectApproxEqAbs(@as(f32, 0), dot, 1e-4);
    };
    // an input row at slot 2 kept off what slots 0 and 1 write there reads none of their outputs
    var basis: std.ArrayList(f32) = .empty;
    defer basis.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 5), try below(&p, gpa, scales, 2, &basis));
    const u = p.neuron(p.up.items, 2, 0, d);
    project(u, basis.items, 5, d);
    for (0..2) |i| for (0..p.used.items[i]) |k| {
        var dot: f32 = 0;
        for (u, scales[2 * d ..][0..d], p.neuron(p.down.items, i, k, d)) |x, s, w| dot += x * s * w;
        try std.testing.expectApproxEqAbs(@as(f32, 0), dot, 1e-4);
    };
}
