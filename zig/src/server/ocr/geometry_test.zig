//! Quad validation and the Mistral integer form.
const std = @import("std");
const api = @import("engine_api");
const g = @import("geometry.zig");
const t = std.testing;
const Quad = api.ocr.Quad;

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) Quad {
    return .{ .{ x0, y0 }, .{ x1, y0 }, .{ x1, y1 }, .{ x0, y1 } };
}

fn expectQuad(want: Quad, got: Quad) !void {
    for (want, got) |w, q| {
        try t.expectEqual(w[0], q[0]);
        try t.expectEqual(w[1], q[1]);
    }
}

test "ocr geometry: a non-finite coordinate fails the page" {
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    try t.expectError(error.Unreadable, g.validate(.{ .{ nan, 0 }, .{ 10, 0 }, .{ 10, 10 }, .{ 0, 10 } }, 100, 100));
    try t.expectError(error.Unreadable, g.validate(.{ .{ 0, 0 }, .{ 10, 0 }, .{ 10, inf }, .{ 0, 10 } }, 100, 100));
    try t.expectError(error.Unreadable, g.validate(.{ .{ 0, 0 }, .{ 10, 0 }, .{ 10, 10 }, .{ -inf, 10 } }, 100, 100));
}

test "ocr geometry: points clamp into the frame and a valid quad keeps its corner order" {
    try expectQuad(rect(0, 5, 100, 50), try g.validate(rect(-20, 5, 140, 50), 100, 80));
    try expectQuad(rect(10, 70, 60, 80), try g.validate(rect(10, 70, 60, 95), 100, 80));
    const rotated: Quad = .{ .{ 30, 0 }, .{ 30, 60 }, .{ 0, 60 }, .{ 0, 0 } }; // 90° text: TL top right
    try expectQuad(rotated, try g.validate(rotated, 100, 100));
    const upside: Quad = .{ .{ 90, 40 }, .{ 10, 40 }, .{ 10, 20 }, .{ 90, 20 } }; // 180° text
    try expectQuad(upside, try g.validate(upside, 100, 100));
    const slanted: Quad = .{ .{ 5, 10 }, .{ 50, 2 }, .{ 52, 14 }, .{ 7, 22 } };
    try expectQuad(slanted, try g.validate(slanted, 100, 100));
}

test "ocr geometry: a degenerate quad becomes its box, a zero side widened to 1 px inside the frame" {
    // collinear: zero height
    try expectQuad(rect(5, 5, 20, 6), try g.validate(.{ .{ 5, 5 }, .{ 20, 5 }, .{ 20, 5 }, .{ 5, 5 } }, 100, 100));
    // self-intersecting bow tie
    try expectQuad(rect(0, 0, 10, 10), try g.validate(.{ .{ 0, 0 }, .{ 10, 10 }, .{ 10, 0 }, .{ 0, 10 } }, 100, 100));
    // all on the bottom edge after clamping: widened upward, inside the frame
    try expectQuad(rect(10, 79, 30, 80), try g.validate(rect(10, 85, 30, 90), 100, 80));
    // a point on the right edge: widened leftward
    try expectQuad(rect(99, 40, 100, 41), try g.validate(rect(120, 40, 130, 40), 100, 100));
    // under 1 px² of area: replaced, and its sub-pixel height grown to 1 px
    try expectQuad(rect(0, 3, 1.5, 4), try g.validate(rect(0, 3, 1.5, 3.5), 100, 100));
}

test "ocr geometry: validated quads are simple and convex, and validation is idempotent" {
    var prng = std.Random.DefaultPrng.init(0x0c7);
    const r = prng.random();
    for (0..2000) |_| {
        var q: Quad = undefined;
        for (&q) |*p| p.* = .{ r.float(f32) * 140 - 20, r.float(f32) * 90 - 5 };
        if (r.boolean()) q[2] = q[1]; // repeated corners are common in model output
        const v = try g.validate(q, 100, 64);
        try t.expect(g.isSimpleConvex(v));
        for (v) |p| try t.expect(p[0] >= 0 and p[0] <= 100 and p[1] >= 0 and p[1] <= 64);
        try expectQuad(v, try g.validate(v, 100, 64));
    }
}

test "ocr geometry: simple convex needs one turn sign and 1 px² of area" {
    try t.expect(g.isSimpleConvex(rect(0, 0, 1, 1)));
    try t.expect(g.isSimpleConvex(.{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 1 }, .{ 1, 0 } })); // the other winding
    try t.expect(!g.isSimpleConvex(rect(0, 0, 0.9, 1)));
    try t.expect(!g.isSimpleConvex(.{ .{ 0, 0 }, .{ 10, 10 }, .{ 10, 0 }, .{ 0, 10 } }));
    try t.expect(!g.isSimpleConvex(.{ .{ 0, 0 }, .{ 5, 0 }, .{ 10, 0 }, .{ 0, 10 } })); // collinear corner
    try t.expect(!g.isSimpleConvex(.{ .{ 0, 0 }, .{ 10, 0 }, .{ 2, 2 }, .{ 0, 10 } })); // concave
    try t.expectEqual([4]f32{ 0, 1, 10.4, 11 }, g.minMax(.{ .{ 0, 1 }, .{ 10, 11 }, .{ 10.4, 10.6 }, .{ 0.4, 1 } }));
}

test "ocr geometry: Mistral floors the minimum and ceils the maximum" {
    try t.expectEqual([4]i32{ 1, 2, 11, 21 }, g.mistralBox(.{ .{ 1.2, 2.7 }, .{ 10.1, 2.7 }, .{ 10.1, 20.0001 }, .{ 1.2, 20.0001 } }));
    try t.expectEqual([4]i32{ 3, 4, 7, 8 }, g.mistralBox(rect(3, 4, 7, 8)));
    try t.expectEqual([4]i32{ 2, 1, 9, 9 }, g.mistralBox(.{ .{ 2.5, 1.5 }, .{ 8.5, 3 }, .{ 8, 8.5 }, .{ 3, 7 } }));
}
