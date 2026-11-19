//! Routed BF16 rows over rank-local 4- or 8-bit MLX affine tensors in their checkpoint layout.
const std = @import("std");
const kernels = @import("kernels.zig");
const Driver = @import("driver.zig").Driver;
const launch = @import("launch.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;

pub const Shape = extern struct {
    width: u32,
    inner: u32,
    first: u32,
    experts: u32,
    pairs: u32,
    rows: u32,
    swiglu_limit: f32,

    pub fn check(self: Shape) !void {
        if (self.width == 0 or self.width % 64 != 0 or self.width > 65536 or
            self.inner == 0 or self.inner % 64 != 0 or self.inner > 65536 or
            self.experts == 0 or @as(u64, self.first) + self.experts > std.math.maxInt(i32) or
            self.pairs == 0 or self.pairs > std.math.maxInt(u16) or self.rows == 0 or
            !std.math.isFinite(self.swiglu_limit) or self.swiglu_limit < 0) return error.BadAffineExpertShape;
    }
};

pub const Weights = struct { words: u64, scales: u64, biases: u64 };

pub const Direction = enum { up, down };
pub const Precision = enum { q4, q8 };

pub const Arguments = struct {
    input: u64,
    ids: u64,
    src_rows: u64 = 0,
    weights: Weights,
    output: u64,
};

pub const Kernels = struct {
    image: Module,
    ops: [2][2]Function,

    pub fn load(driver: *const Driver) !Kernels {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var image = try Module.load(driver, kernels.affine_experts);
        errdefer image.unload();
        var ops: [2][2]Function = undefined;
        inline for ([_]u8{ 4, 8 }, 0..) |bits, i| {
            inline for ([_]Direction{ .up, .down }, 0..) |direction, j| {
                const name = std.fmt.comptimePrint("tf_affine_experts_{s}_{d}", .{ @tagName(direction), bits });
                ops[i][j] = try image.function(name);
            }
        }
        return .{ .image = image, .ops = ops };
    }

    pub fn deinit(self: *Kernels) void {
        self.image.unload();
        self.* = undefined;
    }

    /// Up returns pair rows in BF16. Down reads pair rows and returns FP32 expert contributions.
    /// IDs outside this rank's slice write zeros, preserving the global route ordering on each rank.
    pub fn run(self: Kernels, s: Stream, precision: Precision, direction: Direction, b: Arguments, a: Shape) !void {
        try a.check();
        if (direction == .up and b.src_rows == 0) return error.AffineExpertMissingRows;
        const size = if (direction == .up) a.inner else a.width;
        var args: launch.Args = .{};
        inline for (.{ b.input, b.ids, b.src_rows, b.weights.words, b.weights.scales, b.weights.biases, b.output, a }) |x| args.add(x);
        try launch.launch(self.ops[@backingInt(precision)][@backingInt(direction)], .{
            .grid = .{ .x = (size + 3) / 4, .y = a.pairs },
            .block = .{ .x = 128 },
        }, s, &args);
    }
};
