//! Metal routed rows over the unmodified 4- or 8-bit affine checkpoint tensors.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

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

pub const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };
pub const Weights = struct { words: Ref, scales: Ref, biases: Ref };
pub const Direction = enum { up, down };
pub const Precision = enum { q4, q8 };

pub const Arguments = struct {
    input: Ref,
    ids: Ref,
    src_rows: ?Ref = null,
    weights: Weights,
    output: Ref,
};

pub const Kernels = struct {
    ops: [4]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Kernels {
        const lib = try mtl.Library.fromSource(device, sources.hy4_flash_affine_experts, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var self: Kernels = undefined;
        var loaded: usize = 0;
        errdefer for (self.ops[0..loaded]) |pipe| pipe.deinit();
        const names = [_][]const u8{
            "tf_affine_experts_up_4", "tf_affine_experts_down_4", "tf_affine_experts_up_8", "tf_affine_experts_down_8",
        };
        for (&self.ops, names) |*pipe, name| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            loaded += 1;
        }
        return self;
    }

    pub fn deinit(self: *Kernels) void {
        for (self.ops) |pipe| pipe.deinit();
        self.* = undefined;
    }

    /// Up returns BF16 activated pair rows. Down writes FP32 contributions for weighted combination.
    pub fn run(self: Kernels, e: mtl.ComputeEncoder, precision: Precision, direction: Direction, b: Arguments, a: Shape) !void {
        try a.check();
        if (direction == .up and b.src_rows == null) return error.AffineExpertMissingRows;
        const size = if (direction == .up) a.inner else a.width;
        e.setPipeline(self.ops[2 * @as(usize, @backingInt(precision)) + @backingInt(direction)]);
        const args = [_]Ref{ b.input, b.ids, b.src_rows orelse b.ids, b.weights.words, b.weights.scales, b.weights.biases, b.output };
        for (args, 0..) |ref, i| e.setBuffer(ref.buffer, ref.offset, i);
        e.setValue(a, args.len);
        e.dispatchGroups(mtl.Size.of((size + 3) / 4, a.pairs, 1), mtl.Size.of(128, 1, 1));
    }
};
