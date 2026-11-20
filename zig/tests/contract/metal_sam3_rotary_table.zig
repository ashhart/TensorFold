//! sam3_rotary_table against float64 cos and sin of the spec's angles, within Metal's fast-function allowance.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;

pub const Rotary = extern struct { rows: u32, heads: u32, head_width: u32, rotated_rows: u32, positions: u32, side: u32, scale: f32 };

/// One SAM kernel file's text from the common sources (no tensor ops: builds on every Mac), optionally edited.
pub fn fileText(a: std.mem.Allocator, first: []const u8, end: ?[]const u8, edit: ?[2][]const u8) ![]u8 {
    const all = sources.sam3_baseline_source;
    const tail = sources.sam3_dense_small_baseline;
    if (!std.mem.endsWith(u8, all, tail)) return error.SourceLayoutChanged;
    const common = all[0 .. all.len - tail.len];
    const from = std.mem.indexOf(u8, common, first) orelse return error.SourceLayoutChanged;
    const to = if (end) |e| std.mem.indexOfPos(u8, common, from, e) orelse return error.SourceLayoutChanged else common.len;
    const body = try std.mem.concat(a, u8, &.{ "#include <metal_stdlib>\nusing namespace metal;\n", common[from..to] });
    const e = edit orelse return body;
    defer a.free(body);
    if (std.mem.count(u8, body, e[0]) != 1) return error.ControlTextNotFoundOnce;
    return std.mem.replaceOwned(u8, a, body, e[0], e[1]);
}

pub const table_edit = [2][]const u8{ "const float u = pair < quarter ? float(position % p.side) : float(position / p.side);", "const float u = float(position % p.side);" };

/// Angle: pow 16 ulps, division 9.2, two products, reduction 4, so 32 of 2^-23; then fast cos/sin's 2^-13.
fn allowance(theta: f64) f64 {
    return 0x1p-13 + 32 * 0x1p-23 * @abs(theta);
}

pub const Table = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    control: bool,
    shape: c.Shape = undefined,
    out: rig_.Ref = undefined,
    why: [192]u8 = undefined,

    pub const vt = g.hooks(Table);

    pub fn label(control: bool) []const u8 {
        return if (control) "control: sam3 rotary table uses the x angle for every pair" else "metal sam3 rotary table";
    }

    fn zero(_: c.Shape) usize {
        return 0;
    }

    /// rows = grid side S, n = head width D, k = 3 for the full-grid trunk's sigma f32(24/72), else 1.
    pub fn spec(p: *const Table) c.Spec {
        return .{
            .name = label(p.control),
            .accum = .none,
            .depth = zero,
            .out = .f32,
            .nan = .{ .stated = "no float inputs: every output finite and within fast cos/sin's allowance of float64" },
            .served = &.{
                .{ .rows = 24, .n = 64, .k = 1 },
                .{ .rows = 72, .n = 64, .k = 3 },
                .{ .rows = 72, .n = 32, .k = 1 },
                .{ .rows = 72, .n = 256, .k = 1 },
            },
        };
    }

    fn sigma(k: usize) f32 {
        return if (k == 3) @as(f32, 24) / @as(f32, 72) else 1;
    }

    pub fn prepare(p: *Table, shape: c.Shape, _: g.Fill, _: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 4 != 0 or (shape.k != 1 and shape.k != 3)) return false;
        p.rig.reset();
        p.shape = shape;
        p.out = try p.rig.output(p.count() * 4);
        return true;
    }

    pub fn run(p: *Table) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        pass.e.setBuffer(p.out.buf, p.out.off, 0);
        pass.e.setValue(Rotary{ .rows = 0, .heads = 0, .head_width = @intCast(s.n), .rotated_rows = 0, .positions = @intCast(s.rows * s.rows), .side = @intCast(s.rows), .scale = sigma(s.k) }, 1);
        pass.e.dispatchThreads(mtl.Size.of(p.count() / 2, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Table) usize {
        return p.shape.rows * p.shape.rows * p.shape.n;
    }

    fn theta(p: *const Table, pair_index: usize) f64 {
        const s = p.shape;
        const pairs = s.n / 2;
        const quarter = s.n / 4;
        const pair = pair_index % pairs;
        const position = pair_index / pairs;
        const u: f64 = @floatFromInt(if (pair < quarter) position % s.rows else position / s.rows);
        const e = -@as(f64, @floatFromInt(pair % quarter)) / @as(f64, @floatFromInt(quarter));
        return u * @as(f64, sigma(s.k)) * std.math.pow(f64, 10000, e);
    }

    pub fn reference(p: *Table, want: []f64, mag: []f64) !void {
        for (0..want.len / 2) |i| {
            const t = p.theta(i);
            want[2 * i] = @cos(t);
            want[2 * i + 1] = @sin(t);
        }
        @memset(mag, 0);
    }

    pub fn check(p: *Table, _: g.Gate) !?[]const u8 {
        const out = try p.rig.outputBytes();
        for (0..p.count() / 2) |i| {
            const t = p.theta(i);
            for ([_]f64{ @cos(t), @sin(t) }, 0..) |want, j| {
                const got = c.read(.f32, out, 2 * i + j);
                if (!(@abs(got - want) <= allowance(t))) return try std.fmt.bufPrint(&p.why, "S {d} D {d}: output {d} is {e:.6} where float64 gives {e:.6} (angle {d:.4}, allowance {e:.3})", .{ p.shape.rows, p.shape.n, 2 * i + j, got, want, t, allowance(t) });
            }
        }
        return null;
    }
};
