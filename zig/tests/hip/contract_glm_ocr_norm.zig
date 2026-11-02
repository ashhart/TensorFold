//! GLM-OCR's BF16-node RMS and affine LayerNorm under the common gates, including source/output alias equality.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const Ops = @import("glm_ocr_ops").Ops;
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;

pub const Which = enum { head, vision, text, prompt, decode, layer };
fn width(which: Which) usize {
    return switch (which) {
        .head => 64,
        .vision => 1024,
        .text, .prompt, .decode, .layer => 1536,
    };
}
fn depth(s: c.Shape) usize {
    return s.k / 4 + 32;
}
fn served(comptime which: Which) []const c.Shape {
    const k = width(which);
    return &.{ .{ .rows = 1, .n = k, .k = k }, .{ .rows = 4, .n = k, .k = k }, .{ .rows = 17, .n = k, .k = k } };
}
fn spec(comptime which: Which) c.Spec {
    return .{
        .name = "hip GLM-OCR norm " ++ @tagName(which),
        .accum = .f32,
        .depth = depth,
        .ops = 12,
        .mids = if (which == .layer) 0 else 1,
        .out = .bf16,
        .served = served(which),
        .tiles = .{ .{ .size = 16 }, .{}, .{} },
        .sums = .{ .bf16 = 0, .why = "BF16 output and the intermediate BF16 RMS node cap the visible error of BF16 sums" },
    };
}

pub const Norm = struct {
    const vt = g.hooks(Norm);
    rig: Rig,
    ops: *const Ops,
    which: Which,
    rows: ?rows.Rows = null,
    shape: c.Shape = undefined,
    x: u64 = 0,
    weight: u64 = 0,
    bias: u64 = 0,
    out: u64 = 0,
    baseline: u64 = 0,

    pub fn init(a: std.mem.Allocator, r: *const hip.Runtime, stream: hip.Stream, ops: *const Ops, which: Which) Norm {
        return .{ .rig = .{ .a = a, .r = r, .s = stream }, .ops = ops, .which = which };
    }
    pub fn deinit(n: *Norm) void {
        if (n.rows) |r| r.deinit();
        n.rig.deinit();
    }
    pub fn kernel(n: *Norm) g.Kernel {
        const specification = switch (n.which) {
            inline else => |which| comptime spec(which),
        };
        return .{ .spec = specification, .ctx = n, .vt = &vt };
    }
    pub fn prepare(n: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 17 or shape.k != width(n.which) or shape.n != shape.k) return false;
        if (n.rows) |r| r.deinit();
        n.rows = null;
        n.rig.reset();
        var r = try rows.Rows.init(n.rig.a, .rms, shape);
        n.rows = r;
        r.fill(fill, seed);
        if (fill == .stress and n.which == .head)
            for (r.x) |*v| {
                v.* = c.toBf16(c.fromBf16(v.*) * 4);
            };
        r.eps = 1e-5;
        n.rows = r;
        n.shape = shape;
        n.x = try n.rig.input(std.mem.sliceAsBytes(r.x));
        n.weight = try n.rig.input(std.mem.sliceAsBytes(r.w));
        const zeros = try n.rig.a.alloc(u16, shape.k);
        defer n.rig.a.free(zeros);
        @memset(zeros, 0);
        n.bias = try n.rig.input(std.mem.sliceAsBytes(zeros));
        n.out = try n.rig.output(shape.rows * shape.k * 2);
        n.baseline = try n.rig.scratch(shape.rows * shape.k * 2);
        return true;
    }
    pub fn count(n: *Norm) usize {
        return n.shape.rows * n.shape.k;
    }
    pub fn run(n: *Norm) !void {
        const target: hip.DeviceBuffer = .{ .r = n.rig.r, .ptr = @ptrFromInt(n.out), .len = n.count() * 2 };
        try target.copyFromAsync(0, n.x, n.count() * 2, n.rig.s);
        try n.apply(n.out, n.out);
        try n.rig.s.synchronize();
        const actual = try n.rig.a.dupe(u8, try n.rig.bytes(n.out, n.count() * 2));
        defer n.rig.a.free(actual);
        const separate_bytes = try n.separate();
        if (!std.mem.eql(u8, actual, separate_bytes)) return error.NormAliasChangesOutput;
    }
    fn separate(n: *Norm) ![]const u8 {
        try n.apply(n.x, n.baseline);
        return n.rig.bytes(n.baseline, n.count() * 2);
    }
    fn apply(n: *Norm, input: u64, output: u64) !void {
        if (n.which == .prompt or n.which == .decode) {
            try n.ops.normText(n.rig.s, input, n.weight, output, n.shape.rows, if (n.which == .prompt) .prompt else .decode);
        } else try n.ops.norm(n.rig.s, input, n.weight, if (n.which == .layer) n.bias else null, output, n.shape.rows, n.shape.k);
    }
    pub fn reference(n: *Norm, want: []f64, mag: []f64) !void {
        if (n.which != .layer) return n.rows.?.reference(want, mag);
        try n.layerReference(null, .sequential, want, mag);
    }
    pub fn simulate(n: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (n.which != .layer) return n.rows.?.simulate(acc, order, out);
        const mag = try n.rig.a.alloc(f64, out.len);
        defer n.rig.a.free(mag);
        try n.layerReference(acc, order, out, mag);
    }
    fn layerReference(n: *Norm, acc: ?c.Accum, order: g.Order, out: []f64, mag: []f64) !void {
        const r = n.rows.?;
        const k = r.shape.k;
        const values = try n.rig.a.alloc(f64, k);
        defer n.rig.a.free(values);
        const squares = try n.rig.a.alloc(f64, k);
        defer n.rig.a.free(squares);
        const size: f64 = @floatFromInt(k);
        for (0..r.shape.rows) |row| {
            var sum: f64 = 0;
            var absolute: f64 = 0;
            for (values, 0..) |*v, i| {
                v.* = c.fromBf16(r.x[row * k + i]);
                sum += v.*;
                absolute += @abs(v.*);
            }
            const mean = (if (acc) |a| rows.sumIn(a, order, values) else sum) / size;
            sum = 0;
            for (squares, values) |*s, v| {
                s.* = (v - mean) * (v - mean);
                sum += s.*;
            }
            const variance = (if (acc) |a| rows.sumIn(a, order, squares) else sum) / size;
            const inverse = 1 / @sqrt(variance + r.eps);
            for (values, 0..) |v, i| {
                const weight: f64 = c.fromBf16(r.w[i]);
                out[row * k + i] = (v - mean) * inverse * weight;
                mag[row * k + i] = (@abs(v) + @abs(mean) + absolute / size) * inverse * @abs(weight);
            }
        }
    }
};
