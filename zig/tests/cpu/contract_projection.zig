//! The CPU affine projection under the contract: Qwen3.5-2B's three chain classes, and four planted controls.
const std = @import("std");
const contract = @import("contract");
const k = @import("cpu_kernels");
const rig_ = @import("contract_rig.zig");

const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;

pub const Which = enum { chains16, chains32, chains8, finite, writes, tail, half };

/// Thread counts a registration's runs take in turn: run r on 1 + (r mod 8) threads, the caller one of them.
pub const thread_counts = 8;

fn chainsOf(w: Which) usize {
    return switch (w) {
        .chains32 => 32,
        .chains8 => 8,
        else => 16,
    };
}

/// A group's 64 products in one fma chain, two roundings a group along its chain, then the chains' pairwise tree.
fn depth(comptime chains: usize) fn (c.Shape) usize {
    return struct {
        fn of(s: c.Shape) usize {
            return 64 + 2 * ((s.k / k.group + chains - 1) / chains) + std.math.log2_int(usize, chains);
        }
    }.of;
}

/// Rows in blocks of eight, columns in tiles of eight and passes of four tiles, inputs in groups of 64.
const tiles: [3]c.Tile = .{ .{ .size = 8 }, .{ .size = 32, .step = 8, .min = 8 }, .{ .size = k.group, .step = k.group, .min = k.group } };

/// The 2B's 16-chain projections at a decode row, a 128-row prompt chunk and a 32-row verify window, each width.
const served16 = [_]c.Shape{
    .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 128, .n = 2048, .k = 2048 }, .{ .rows = 32, .n = 2048, .k = 2048 },
    .{ .rows = 1, .n = 6144, .k = 2048 }, .{ .rows = 128, .n = 6144, .k = 2048 }, .{ .rows = 32, .n = 6144, .k = 2048 },
    .{ .rows = 1, .n = 2048, .k = 6144 }, .{ .rows = 128, .n = 2048, .k = 6144 }, .{ .rows = 32, .n = 2048, .k = 6144 },
    .{ .rows = 1, .n = 512, .k = 2048 },  .{ .rows = 128, .n = 512, .k = 2048 },  .{ .rows = 32, .n = 512, .k = 2048 },
    .{ .rows = 1, .n = 4096, .k = 2048 }, .{ .rows = 128, .n = 4096, .k = 2048 }, .{ .rows = 32, .n = 4096, .k = 2048 },
};
/// The delta rule's a and b projections, 16 outputs each, at the same three row counts.
const served32 = [_]c.Shape{ .{ .rows = 1, .n = 16, .k = 2048 }, .{ .rows = 128, .n = 16, .k = 2048 }, .{ .rows = 32, .n = 16, .k = 2048 } };
/// The head: one row a decode step or a prompt's last row, and a verify window's rows.
const served8 = [_]c.Shape{ .{ .rows = 1, .n = 248320, .k = 2048 }, .{ .rows = 32, .n = 248320, .k = 2048 } };
const ctl = [_]c.Shape{ .{ .rows = 9, .n = 104, .k = 2048 }, .{ .rows = 2, .n = 40, .k = 6144 } };

pub fn spec(w: Which) c.Spec {
    return switch (w) {
        .chains16 => .{ .name = "cpu projection q4 g64, 16 chains", .accum = .f32, .depth = depth(16), .out = .bf16, .tiles = tiles, .served = &served16 },
        .chains32 => .{ .name = "cpu projection q4 g64, 32 chains", .accum = .f32, .depth = depth(32), .out = .bf16, .tiles = tiles, .served = &served32 },
        .chains8 => .{ .name = "cpu projection q4 g64, 8 chains", .accum = .f32, .depth = depth(8), .out = .bf16, .tiles = tiles, .served = &served8 },
        .finite => .{ .name = "control: projection zeroing a non-finite activation", .accum = .f32, .depth = depth(16), .out = .bf16, .tiles = tiles, .served = &ctl },
        .writes => .{ .name = "control: projection writing its activations", .accum = .f32, .depth = depth(16), .out = .bf16, .tiles = tiles, .served = &ctl },
        .tail => .{ .name = "control: projection skipping the rows past its last block of eight", .accum = .f32, .depth = depth(16), .out = .bf16, .tiles = tiles, .served = &ctl },
        .half => .{ .name = "control: projection summing each group in half", .accum = .f32, .depth = depth(16), .out = .bf16, .tiles = tiles, .served = &ctl },
    };
}

/// The gate each control must fail.
pub fn control(w: Which) ?g.Gate {
    return switch (w) {
        .chains16, .chains32, .chains8 => null,
        .finite => .nonfinite,
        .writes => .repeat,
        .tail => .shapes,
        .half => .sums,
    };
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    runs: usize = 0,
    f: ?Affine = null,
    l: k.Linear = undefined,
    x: []u16 = &.{},
    clean: []u16 = &.{},
    xf: []f32 = &.{},
    xs: []f32 = &.{},
    out: []u16 = &.{},

    pub fn init(a: std.mem.Allocator, which: Which) Proj {
        return .{ .a = a, .rig = .{ .a = a }, .which = which };
    }

    pub fn deinit(p: *Proj) void {
        p.drop();
        p.rig.deinit();
    }

    fn drop(p: *Proj) void {
        if (p.f) |f| f.deinit();
        p.f = null;
        p.rig.reset();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        return .{ .spec = spec(p.which), .ctx = p, .vt = &vt, .control = control(p.which) };
    }

    const vt = g.hooks(Proj);

    fn takes(shape: c.Shape) bool {
        return shape.rows > 0 and shape.rows <= k.max_rows and shape.n > 0 and shape.n % 8 == 0 and shape.k > 0 and shape.k % k.group == 0;
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (!takes(shape)) return false;
        var f = try Affine.init(p.a, shape, 4, k.group);
        errdefer f.deinit();
        try f.fill(fill, seed);
        const words = try p.a.alloc(u32, shape.n * shape.k / 8);
        defer p.a.free(words);
        f.words(words);
        const packed_ = try k.pack(p.a, words, f.scales, f.biases, shape.n, shape.k, chainsOf(p.which));
        defer k.free(p.a, packed_);
        p.l = .{ .tiles = try p.rig.input(u32, packed_.tiles), .scales = try p.rig.input(u16, packed_.scales), .biases = try p.rig.input(u16, packed_.biases), .inputs = shape.k, .outputs = shape.n, .chains = packed_.chains };
        p.x = try p.rig.input(u16, f.x);
        p.clean = try p.rig.scratch(u16, shape.k);
        p.xf = try p.rig.scratch(f32, shape.rows * shape.k);
        p.xs = try p.rig.scratch(f32, shape.rows * (shape.k / k.group));
        p.out = try p.rig.output(u16, f.outputs());
        p.f = f;
        return true;
    }

    pub fn run(p: *Proj) !void {
        const shape = p.f.?.shape;
        const groups = shape.k / k.group;
        const threads = 1 + p.runs % thread_counts;
        p.runs += 1;
        for (0..shape.rows) |r| {
            var row = p.x[r * shape.k ..][0..shape.k];
            if (p.which == .finite) {
                for (p.clean, row) |*v, h| v.* = if (std.math.isFinite(k.f32of(h))) h else 0;
                row = p.clean;
            }
            k.prepareRow(row, p.xf[r * shape.k ..][0..shape.k], p.xs[r * groups ..][0..groups]);
        }
        switch (p.which) {
            .half => p.halfRows(),
            .tail => try k.projectOn(p.l, p.xf, p.xs, shape.rows / 8 * 8, p.out, threads),
            else => try k.projectOn(p.l, p.xf, p.xs, shape.rows, p.out, threads),
        }
        if (p.which == .writes) p.x[p.x.len - 1] ^= 1;
    }

    /// The projection's own chains, each group's 64 products accumulated in half precision.
    fn halfRows(p: *Proj) void {
        const shape = p.f.?.shape;
        const groups = shape.k / k.group;
        for (0..shape.rows) |r| for (0..shape.n) |j| {
            var acc: [32]f32 = @splat(0);
            for (0..groups) |gi| {
                var sum: f16 = 0;
                for (0..k.group) |e| {
                    const code: f16 = @floatFromInt(p.l.code(j, gi * k.group + e));
                    sum = @floatCast(code * @as(f16, @floatCast(p.xf[r * shape.k + gi * k.group + e])) + sum);
                }
                const chain = gi % p.l.chains;
                acc[chain] = @mulAdd(f32, k.f32of(p.l.bias(j, gi)), p.xs[r * groups + gi], @mulAdd(f32, k.f32of(p.l.scale(j, gi)), @floatCast(sum), acc[chain]));
            }
            var width: usize = 1;
            while (width < p.l.chains) : (width *= 2) {
                var at: usize = 0;
                while (at + width < p.l.chains) : (at += 2 * width) acc[at] = acc[at + width] + acc[at];
            }
            p.out[r * shape.n + j] = k.bf16of(acc[0]);
        };
    }

    pub fn count(p: *Proj) usize {
        return p.f.?.outputs();
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        return p.f.?.reference(want, mag);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.f.?.simulate(acc, order, out);
    }
};

test "each control binds to its gate, and the chain classes serve the 2B's widths" {
    try std.testing.expectEqual(g.Gate.nonfinite, control(.finite).?);
    try std.testing.expectEqual(g.Gate.repeat, control(.writes).?);
    try std.testing.expectEqual(g.Gate.shapes, control(.tail).?);
    try std.testing.expectEqual(g.Gate.sums, control(.half).?);
    try std.testing.expectEqual(@as(usize, 64 + 4 + 4), spec(.chains16).depth(.{ .rows = 1, .n = 2048, .k = 2048 }));
    try std.testing.expectEqual(@as(usize, 64 + 12 + 4), spec(.chains16).depth(.{ .rows = 1, .n = 2048, .k = 6144 }));
    try std.testing.expectEqual(@as(usize, 64 + 2 + 5), spec(.chains32).depth(.{ .rows = 1, .n = 16, .k = 2048 }));
    try std.testing.expectEqual(@as(usize, 64 + 8 + 3), spec(.chains8).depth(.{ .rows = 1, .n = 248320, .k = 2048 }));
    for (served16 ++ served32 ++ served8) |s| try std.testing.expect(Proj.takes(s));
    try std.testing.expect(!Proj.takes(.{ .rows = 129, .n = 8, .k = 64 }));
    try std.testing.expect(!Proj.takes(.{ .rows = 1, .n = 12, .k = 64 }));
    try std.testing.expect(!Proj.takes(.{ .rows = 1, .n = 8, .k = 96 }));
}
