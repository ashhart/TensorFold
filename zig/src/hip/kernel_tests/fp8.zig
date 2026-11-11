//! ModelOpt FP8's product against the format's float64 reference, at every row count the engine serves.

const std = @import("std");
const rig = @import("rig.zig");
const launch = @import("../launches/fp8.zig");
const quant = @import("core").quant;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

/// A projection as ModelOpt stores an FP8 one: E4M3 codes one a byte, one f32 scale a projection.
const Proj = struct {
    codes: []u8,
    scale: f32,
    n: usize,
    k: usize,

    fn init(a: std.mem.Allocator, rng: *rig.Rng, n: usize, k: usize, scale: f32) !Proj {
        const codes = try a.alloc(u8, n * k);
        for (codes) |*c| {
            var b: u8 = @intCast(rng.next() & 0xff);
            if (b & 0x7f == 0x7f) b &= 0xfe; // E4M3's NaN codes are refused at read
            c.* = b;
        }
        return .{ .codes = codes, .scale = scale, .n = n, .k = k };
    }

    fn deinit(p: Proj) void {
        gpa.free(p.codes);
    }
};

/// `hx`'s rows times `p`, f32 out, against the format's f64 values and the bound the kernel's own chain allows.
fn product(t: *Rig, p: Proj, hx: []const u16) ![]f32 {
    const rows = hx.len / p.k;
    const want = try gpa.alloc(f64, rows * p.n);
    defer gpa.free(want);
    const mag = try gpa.alloc(f64, rows * p.n);
    defer gpa.free(mag);
    for (0..rows) |r| {
        for (0..p.n) |c| {
            var acc: f64 = 0;
            var abs: f64 = 0;
            for (0..p.k) |j| {
                const term = t.value(hx[r * p.k + j]) * quant.fp8.e4m3(p.codes[c * p.k + j]);
                acc += term;
                abs += @abs(term);
            }
            want[r * p.n + c] = acc * @as(f64, p.scale);
            mag[r * p.n + c] = abs * @as(f64, p.scale);
        }
    }

    var x = try t.upload(hx);
    defer x.free();
    var codes = try t.upload(p.codes);
    defer codes.free();
    const one = [_]f32{p.scale};
    var scale = try t.upload(std.mem.sliceAsBytes(&one));
    defer scale.free();
    var out = try t.alloc(rows * p.n * 4);
    defer out.free();

    const w: launch.Weight = .{
        .x = at(x),
        .codes = at(codes),
        .scale = at(scale),
        .out = at(out),
        .m = @intCast(rows),
        .n = @intCast(p.n),
        .k = @intCast(p.k),
    };
    try launch.Kernels.gemmGo(&t.on.fp8, &t.r, w, t.stream.handle);
    try t.stream.synchronize();

    const bits = try rig.download(u32, out);
    errdefer gpa.free(bits);
    const got: []f32 = @ptrCast(bits);
    // the kernel's chain: k fmaf into the sum, then one multiply by the scale
    const u = std.math.pow(f64, 2, -24);
    const per_element = @as(f64, @floatFromInt(p.k + 1)) * u;
    var worst: f64 = 0;
    for (want, 0..) |wv, i| {
        const tol = per_element * mag[i];
        if (mag[i] > 0) worst = @max(worst, @abs(@as(f64, got[i]) - wv) / tol);
        if (@abs(@as(f64, got[i]) - wv) > tol) {
            std.debug.print("fp8 {d}x{d} rows {d}: [{d}, {d}] is {d} against {d}, bound {d}\n", .{ p.n, p.k, rows, i / p.n, i % p.n, got[i], wv, tol });
            return error.TestUnexpectedResult;
        }
    }
    std.debug.print("fp8 {d}x{d} rows {d}: worst |got - want| is {d:.3} of the derived bound\n", .{ p.n, p.k, rows, worst });
    return got;
}

test "FP8: the product holds the format reference at one row, at three and at five" {
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x6A31C7 };
    for ([_][3]usize{ .{ 64, 64, 0 }, .{ 160, 128, 1 }, .{ 48, 96, 2 } }) |shape| {
        // one global of each kind: a power of two, an awkward f32, and one below it
        const scale: f32 = switch (shape[2]) {
            0 => 0.25,
            1 => 0.3,
            else => 0.01171875,
        };
        const p = try Proj.init(gpa, &rng, shape[0], shape[1], scale);
        defer p.deinit();
        for ([_]usize{ 1, 3, 5 }) |rows| {
            const hx = try gpa.alloc(u16, rows * p.k);
            defer gpa.free(hx);
            for (hx) |*v| v.* = t.bits(rng.unit());
            const got = try product(t, p, hx);
            gpa.free(got);
        }
    }
}

test "FP8: a row's bits are the same alone and in a round of four" {
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x1D4B93 };
    const p = try Proj.init(gpa, &rng, 96, 128, 0.375);
    defer p.deinit();
    const hx = try gpa.alloc(u16, 4 * p.k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    const alone = try product(t, p, hx[0..p.k]);
    defer gpa.free(alone);
    const round = try product(t, p, hx);
    defer gpa.free(round);
    try std.testing.expectEqualSlices(f32, alone, round[0..p.n]);
}
