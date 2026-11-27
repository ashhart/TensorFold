//! Whole HC pre outputs, including every partial, equal the original launch on identical BF16 inputs.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const contract = @import("core").contract;
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("contract_rig.zig");

const wide = 16384;
const bytes = [_]usize{ 4096 * 2, 64 * 4, 4 * 4, 16 * 4, 16 * 32 * 4 };
pub const served = blk: {
    var list: [32]c.Shape = undefined;
    for (&list, 1..) |*s, r| s.* = .{ .rows = r, .n = 25, .k = wide };
    break :blk list;
};
fn noSums(_: c.Shape) usize {
    return 0;
}

pub const Fixture = struct {
    rig: rig_.Rig,
    glue: glm.glue.Kernels,
    fault: bool = false,
    rows: usize = 0,
    x: u64 = 0,
    w: [4]u64 = undefined,
    out: [5]u64 = undefined,
    baseline: [5]u64 = undefined,
    saved: std.ArrayList(u8) = .empty,

    pub fn deinit(f: *Fixture) void {
        f.rig.deinit();
        f.saved.deinit(f.rig.gpu.gpa);
    }

    pub fn kernel(f: *Fixture) g.Kernel {
        return .{ .spec = .{
            .name = if (f.fault) "control: glm HC shared weights read the next input row" else "cuda glm HC shared-row partial8 and finish",
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .replaces = "tf_glm_hc_partial8 plus unchanged hc_finish",
            .nan = .{ .stated = "Every output byte, including nonfinite payloads, equals the original per-row launch" },
            .served = &served,
        }, .ctx = f, .vt = &g.hooks(Fixture), .control = if (f.fault) .shapes else null };
    }

    pub fn prepare(f: *Fixture, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != 25 or shape.k != wide or shape.rows == 0 or shape.rows > 32) return false;
        f.rig.reset();
        f.rows = shape.rows;
        const a = f.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const x = try a.alloc(u16, (f.rows + 1) * wide);
        defer a.free(x);
        const fnw = try a.alloc(u16, 24 * wide);
        defer a.free(fnw);
        for (x, 0..) |*v, i| v.* = c.toBf16(if (fill == .stress) (if (i % 2 == 0) @as(f32, 1) else -0.5) else rng.float(f32) * 2 - 1);
        for (fnw) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.02);
        if (fill == .nonfinite) {
            const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32) };
            for (values, 0..) |v, r| if (r < f.rows) {
                x[r * wide + 129] = c.toBf16(v);
            };
            if (f.rows > 3) x[3 * wide + 143] = c.toBf16(-std.math.inf(f32));
            fnw[wide + 129] = c.toBf16(std.math.nan(f32));
        }
        const norm = try a.alloc(u16, 4096);
        defer a.free(norm);
        @memset(norm, c.toBf16(1));
        const base: [24]f32 = @splat(0.1);
        const scale = [3]f32{ 0.7, 0.9, 1.1 };
        f.x = try f.rig.input(std.mem.sliceAsBytes(x));
        f.w = .{ try f.rig.input(std.mem.sliceAsBytes(fnw)), try f.rig.input(std.mem.asBytes(&base)), try f.rig.input(std.mem.asBytes(&scale)), try f.rig.input(std.mem.sliceAsBytes(norm)) };
        for (&f.out, &f.baseline, bytes) |*out, *old, per| {
            out.* = try f.rig.output(f.rows * per);
            old.* = try f.rig.scratch(f.rows * per);
            const b: cuda.DeviceBuffer = .{ .d = f.rig.gpu.d, .ptr = old.*, .len = f.rows * per };
            try b.fill8(0xff, f.rig.s.handle);
        }
        return true;
    }

    fn launch(f: *Fixture, shared: bool, out: [5]u64) !void {
        try glm.glue.hcPre(.{ .glue = &f.glue, .hc_rows = shared }, f.rig.s, f.x, f.w[0], f.w[1], f.w[2], .{ .bf16 = f.w[3] }, out[0], out[1], out[2], out[3], out[4], 1e-6, 1e-6, 20, f.rows, 4096, 4);
        try f.rig.s.synchronize();
    }

    pub fn run(f: *Fixture) !void {
        try f.launch(true, f.out);
    }

    pub fn replaced(f: *Fixture) ![]const u8 {
        try f.launch(false, f.baseline);
        f.saved.clearRetainingCapacity();
        for (f.baseline, bytes) |ptr, per| try f.saved.appendSlice(f.rig.gpu.gpa, try f.rig.bytes(ptr, f.rows * per));
        return f.saved.items;
    }

    pub fn check(f: *Fixture, _: g.Gate) !?[]const u8 {
        const out = try f.rig.outputBytes();
        return if (std.mem.eql(u8, out, try f.replaced())) null else "HC shared-row outputs differ from the original per-row launch";
    }

    pub fn count(f: *Fixture) usize {
        var total: usize = 0;
        for (bytes) |n| total += n;
        return f.rows * total;
    }

    pub fn reference(f: *Fixture, want: []f64, mag: []f64) !void {
        for (try f.replaced(), want, mag) |v, *w, *m| {
            w.* = @floatFromInt(v);
            m.* = 0;
        }
    }
};
