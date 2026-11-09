//! ModelOpt NVFP4 products: the prompt GEMM and the decode stream tile against a float64 oracle.

const std = @import("std");
const rig = @import("rig.zig");
const launch = @import("../launches/nvfp4.zig");
const quant = @import("core").quant;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

/// A projection as the checkpoint stores it: e2m1 nibbles, an E4M3 scale a 16 inputs, one global scale.
const Proj = struct {
    codes: []u8,
    scales: []u8,
    global: f32,
    n: usize,
    k: usize,

    fn init(a: std.mem.Allocator, rng: *rig.Rng, n: usize, k: usize, global: f32) !Proj {
        const codes = try a.alloc(u8, n * k / 2);
        for (codes) |*c| c.* = @intCast(rng.next() & 0xff);
        const scales = try a.alloc(u8, n * (k / quant.nvfp4.group));
        // E4M3 in [0.5, 1.875]: every byte holds a finite positive value
        for (scales) |*s| s.* = @intCast(0x30 + (rng.next() & 0x0f));
        return .{ .codes = codes, .scales = scales, .global = global, .n = n, .k = k };
    }

    fn deinit(p: Proj) void {
        gpa.free(p.codes);
        gpa.free(p.scales);
    }
};

/// The ModelOpt tensors behind one key, as the loader reads them.
const Fake = struct {
    w: quant.Tensor,
    s: quant.Tensor,
    g: quant.Tensor,

    pub fn get(f: *const Fake, key: []const u8) !quant.Tensor {
        if (std.mem.endsWith(u8, key, ".weight")) return f.w;
        if (std.mem.endsWith(u8, key, ".weight_scale_2")) return f.g;
        if (std.mem.endsWith(u8, key, ".weight_scale")) return f.s;
        return error.MissingTensor;
    }

    pub fn has(f: *const Fake, key: []const u8) bool {
        _ = f;
        return std.mem.endsWith(u8, key, ".weight_scale_2");
    }
};

/// `hx`'s rows (each `k` wide, the activation's bits) times `p`, f32 out: the one product kernel.
fn product(t: *Rig, p: Proj, hx: []const u16) ![]f32 {
    const rows = hx.len / p.k;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // the projection through the format's own reader, so the oracle reads the table the engine uploads
    const fake: Fake = .{
        .w = .{ .dtype = .u8, .rank = 2, .shape = .{ p.n, p.k / 2, 1, 1, 1 }, .bytes = p.codes },
        .s = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ p.n, p.k / quant.nvfp4.group, 1, 1, 1 }, .bytes = p.scales },
        .g = .{ .dtype = .f32, .rank = 1, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes((&[_]f32{p.global})[0..]) },
    };
    const h = try quant.nvfp4.read(a, &fake, "p");
    const wq = try a.alloc(f64, p.n * p.k);
    try quant.nvfp4.reference(h, wq);
    // the product is the test's own f64 sum: the format reads back exactly, and the tiles round once to f32
    const want = try a.alloc(f64, rows * p.n);
    for (0..rows) |r| for (0..p.n) |c| {
        var acc: f64 = 0;
        for (0..p.k) |j| acc += t.value(hx[r * p.k + j]) * wq[c * p.k + j];
        want[r * p.n + c] = acc;
    };

    var x = try t.upload(hx);
    defer x.free();
    var codes = try t.upload(p.codes);
    defer codes.free();
    var scale = try t.upload(std.mem.bytesAsSlice(u16, h.scale.bytes));
    defer scale.free();
    var out = try t.alloc(rows * p.n * 4);
    defer out.free();

    const w: launch.Weight = .{
        .x = at(x),
        .words = at(codes),
        .scale = .{ .p = at(scale), .kind = @backingInt(quant.Tables.bf16) },
        .out = at(out),
        .m = @intCast(rows),
        .n = @intCast(p.n),
        .k = @intCast(p.k),
        .fp16 = @intFromBool(!t.bf16),
        .global = p.global,
    };
    try launch.Kernels.gemmGo(&t.on.nvfp4, &t.r, w, t.stream.handle);
    try t.stream.synchronize();

    const bits = try rig.download(u32, out);
    errdefer gpa.free(bits);
    const got: []f32 = @ptrCast(bits);
    for (want, 0..) |wv, i| {
        const tol = 2e-4 * @max(1.0, @abs(wv));
        if (@abs(@as(f64, got[i]) - wv) > tol) {
            std.debug.print("nvfp4 {d}x{d} rows {d} x[0]={d} wq[0]={d} wq[1]={d} want[{d}]={d} got={d}\n", .{ p.n, p.k, rows, t.value(hx[0]), wq[0], wq[1], i, wv, got[i] });
            return error.TestUnexpectedResult;
        }
    }
    return got;
}

fn draw(a: std.mem.Allocator, t: *const Rig, rng: *rig.Rng, rows: usize, k: usize) ![]u16 {
    const hx = try a.alloc(u16, rows * k);
    for (hx) |*v| v.* = t.bits(rng.unit());
    return hx;
}

test "NVFP4: the product holds the format reference at one row, at three and at five" {
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x4F9D21 };
    for ([_][2]usize{ .{ 64, 64 }, .{ 160, 128 } }) |shape| {
        const p = try Proj.init(gpa, &rng, shape[0], shape[1], 0.25);
        defer p.deinit();
        for ([_]usize{ 1, 3, 5 }) |rows| {
            const hx = try draw(gpa, t, &rng, rows, p.k);
            defer gpa.free(hx);
            const got = try product(t, p, hx);
            gpa.free(got);
        }
    }
}

test "NVFP4: a row's bits are the same alone and in a round of four" {
    const t = try Rig.open(1 << 16);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x51A6E7 };
    const p = try Proj.init(gpa, &rng, 96, 128, 0.5);
    defer p.deinit();
    const hx = try draw(gpa, t, &rng, 4, p.k);
    defer gpa.free(hx);
    const alone = try product(t, p, hx[0..p.k]);
    defer gpa.free(alone);
    const round = try product(t, p, hx);
    defer gpa.free(round);
    try std.testing.expectEqualSlices(f32, alone, round[0..p.n]);
}
