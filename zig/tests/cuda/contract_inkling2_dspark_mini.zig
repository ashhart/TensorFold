//! Mini's visit-tap and hidden projection bridges under the DSpark CUDA contract.
const std = @import("std");
const core = @import("core");
const dk = @import("inkling2").dspark_ops;
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;

/// The copy preserves every other channel, including non-finite BF16 words and tail columns.
pub const Tap = struct {
    kern: *const dk.Kernels,
    fault: bool = false,
    shape: c.Shape = undefined,
    src: []u16 = &.{},
    want: []u8 = &.{},
    input: u64 = 0,
    output: u64 = 0,

    pub fn spec(self: *const Tap) c.Spec {
        return .{
            .name = if (self.fault) "inkling2 dspark control: Mini tap writes the wrong channel" else "inkling2 dspark tf_ink2_mini_tap: interleaved visit rows preserve all four channels",
            .accum = .none,
            .depth = ref.noSums,
            .out = .raw,
            .nan = .{ .stated = "every BF16 bit copied exactly; all three other channels stay untouched" },
            .served = &.{ .{ .rows = 1, .n = 2304, .k = 0 }, .{ .rows = 4, .n = 2305, .k = 1 }, .{ .rows = 4, .n = 2304, .k = 3 } },
        };
    }

    pub fn control(self: *const Tap) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *Tap, a: std.mem.Allocator) void {
        a.free(self.src);
        a.free(self.want);
        self.src = &.{};
        self.want = &.{};
    }

    pub fn prepare(self: *Tap, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k > 3) return false;
        self.free(a);
        self.shape = shape;
        self.src = try a.alloc(u16, shape.rows * shape.n);
        for (self.src) |*word| word.* = rng.int(u16);
        if (fill == .nonfinite) {
            self.src[0] = c.toBf16(ref.nan);
            self.src[self.src.len - 1] = c.toBf16(ref.inf);
        }
        self.want = try a.alloc(u8, shape.rows * 4 * shape.n * 2);
        @memset(self.want, 0xff);
        for (0..shape.rows) |row| {
            const dst = (row * 4 + shape.k) * shape.n * 2;
            @memcpy(self.want[dst..][0 .. shape.n * 2], std.mem.sliceAsBytes(self.src[row * shape.n ..][0..shape.n]));
        }
        self.input = try ref.input(rig, self.src);
        self.output = try rig.output(self.want.len);
        return true;
    }

    pub fn run(self: *Tap, rig: *rig_.Rig) !void {
        try dk.miniTap(self.kern, rig.s, self.input, self.output, self.shape.rows, if (self.fault) (self.shape.k + 1) % 4 else self.shape.k, self.shape.n);
    }

    pub fn count(self: *const Tap) usize {
        return self.want.len;
    }

    pub fn check(self: *const Tap, _: std.mem.Allocator, got: []const u8, _: *[200]u8) ?[]const u8 {
        return if (std.mem.eql(u8, got, self.want)) null else "a tap row or an untouched channel changed";
    }
};

/// The fused Mini hidden_out adds target hidden to bias and the rank projection, then rounds BF16 once.
pub const Hidden = struct {
    kern: *const dk.Kernels,
    fault: bool = false,
    shape: c.Shape = undefined,
    stride: usize = 0,
    y: []u16 = &.{},
    latent: []u16 = &.{},
    w: []u16 = &.{},
    bias: []u16 = &.{},
    input: [4]u64 = @splat(0),
    output: u64 = 0,

    pub fn spec(self: *const Hidden) c.Spec {
        return .{
            .name = if (self.fault) "inkling2 dspark control: Mini hidden reads the next hidden column" else "inkling2 dspark tf_ink2_mini_hidden: BF16 hidden_out before target unembedding",
            .accum = .none,
            .depth = ref.noSums,
            .out = .bf16,
            .nan = .{ .stated = "NaN and infinities follow the BF16 projection and sum" },
            .served = &.{ .{ .rows = 1, .n = 64, .k = 256 }, .{ .rows = 4, .n = 257, .k = 256 } },
        };
    }

    pub fn control(self: *const Hidden) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *Hidden, a: std.mem.Allocator) void {
        inline for (.{ "y", "latent", "w", "bias" }) |name| {
            a.free(@field(self, name));
            @field(self, name) = &.{};
        }
    }

    pub fn prepare(self: *Hidden, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.k != 256 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        self.stride = shape.n + 3;
        self.y = try ref.bfs(a, rng, shape.rows * self.stride, -1, 1);
        self.latent = try ref.bfs(a, rng, shape.rows * shape.k, -0.5, 0.5);
        self.w = try ref.bfs(a, rng, shape.n * shape.k, -0.5, 0.5);
        self.bias = try ref.bfs(a, rng, shape.n, -1, 1);
        if (fill == .stress) {
            for (self.y) |*word| word.* = c.toBf16(ref.int(rng, -8, 8));
            for (self.w) |*word| word.* = c.toBf16(ref.int(rng, -8, 8));
        }
        if (fill == .nonfinite) {
            self.y[0] = c.toBf16(ref.nan);
            self.bias[shape.n - 1] = c.toBf16(ref.inf);
        }
        self.input = .{ try ref.input(rig, self.y), try ref.input(rig, self.latent), try ref.input(rig, self.w), try ref.input(rig, self.bias) };
        self.output = try rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(self: *Hidden, rig: *rig_.Rig) !void {
        try dk.miniHidden(self.kern, rig.s, self.input[0] + @as(u64, if (self.fault) 2 else 0), self.stride, self.input[1], self.input[2], self.input[3], self.output, self.shape.rows, self.shape.n, self.shape.k);
    }

    pub fn count(self: *const Hidden) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Hidden, _: std.mem.Allocator, _: ref.Sums, out: []ref.Iv) !void {
        const n = self.shape.n;
        const rank = self.shape.k;
        for (0..self.shape.rows) |row| for (0..n) |col| {
            var v = ref.Iv.of(ref.bf(self.y[row * self.stride + col])).add(ref.Iv.of(ref.bf(self.bias[col])));
            for (0..rank) |i| v = v.add(ref.Iv.of(ref.bf(self.w[col * rank + i])).mul(ref.Iv.of(ref.bf(self.latent[row * rank + i]))));
            out[row * n + col] = v.bf();
        };
    }
};
