//! The prompt's skewed-row matrix-core tile against the padded 128-row tile it replaces, under the contract on HIP.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Kernels = @FieldType(hip.Launcher, "affine");
const Arg = @typeInfo(@TypeOf(Kernels.streamGo)).@"fn".param_types[2].?;
const Kind = @typeInfo(@TypeOf(Kernels.blockGo)).@"fn".param_types[5].?;
const Entry = @typeInfo(@FieldType(@FieldType(Kernels, "reg"), "entries")).pointer.child;
const Call = @typeInfo(@typeInfo(@FieldType(Entry, "launch")).pointer.child).@"fn".param_types[1].?;

const Case = struct { group: usize, name: []const u8 };

pub const cases = [_]Case{
    .{ .group = 32, .name = "hip affine_wmma_skew (4-bit, group 32, fp32 out)" },
    .{ .group = 64, .name = "hip affine_wmma_skew (4-bit, group 64, fp32 out)" },
};

/// The 128-row tile's chain: a group's products by k, its scale and bias, then the groups' sum.
fn depth(s: c.Shape) usize {
    return s.k / 2 + s.k / 32 + 8;
}

/// x [rows][k] bf16 times the MLX 4-bit matrix [n][k], fp32 [rows][n], 128 rows a block.
pub const Skew = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    arena: std.heap.ArenaAllocator,
    index: usize,
    f: ?contract.affine.Affine = null,
    arg: Arg = undefined,
    base: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, i: usize) Skew {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .arena = .init(a), .index = i };
    }

    pub fn deinit(self: *Skew) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Skew) g.Kernel {
        const cfg = cases[self.index];
        return .{
            .spec = .{
                .name = cfg.name,
                .accum = .f32,
                .depth = depth,
                .out = .f32,
                .replaces = "affine_wmma_gemm",
                .nan = .{ .stated = "the padded tile's bytes" },
                // Qwen3.8-27B's prompt widths and Qwen3.6-35B-A3B's, cut to a few blocks, with partial edge blocks
                .served = &.{ .{ .rows = 256, .n = 256, .k = 5120 }, .{ .rows = 130, .n = 1300, .k = 2048 }, .{ .rows = 300, .n = 136, .k = 6144 } },
                .tiles = .{ .{ .size = 128 }, .{ .size = 128 }, .{ .size = 256, .step = cfg.group, .min = cfg.group } },
            },
            .ctx = self,
            .vt = &vt,
        };
    }

    const vt = g.hooks(Skew);

    pub fn prepare(self: *Skew, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const cfg = cases[self.index];
        if (shape.rows == 0 or shape.n == 0 or shape.k % cfg.group != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const f = try contract.affine.Affine.init(a, shape, 4, cfg.group);
        try f.fill(fill, seed);
        self.f = f;
        const words = try a.alloc(u32, shape.n * shape.k / 8);
        f.words(words);
        self.base = try self.rig.scratch(f.outputs() * 4);
        self.arg = .{
            .x = try self.rig.input(std.mem.sliceAsBytes(f.x)),
            .words = try self.rig.input(std.mem.sliceAsBytes(words)),
            .scale = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.scales)), .kind = 1 },
            .bias = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.biases)), .kind = 1 },
            .out = try self.rig.output(f.outputs() * 4),
            .m = @intCast(shape.rows),
            .n = @intCast(shape.n),
            .k = @intCast(shape.k),
            .bits = 4,
            .group = @intCast(cfg.group),
            .fp16 = 0,
        };
        return true;
    }

    pub fn run(self: *Skew) !void {
        for (self.l.affine.reg.entries) |*e| if (std.mem.eql(u8, e.id, "mlx.project.prefill.skew")) {
            const call: Call = .{ .r = self.l.r, .s = self.rig.s.handle, .arg = self.arg, .items = 1 };
            try e.launch(&self.l.affine, call);
            try self.rig.s.synchronize();
            return;
        };
        return error.Invalid;
    }

    /// The padded matrix-core tile's fp32 bytes on the same inputs.
    pub fn replaced(self: *Skew) ![]const u8 {
        var plain = self.arg;
        plain.out = self.base;
        try self.l.affine.blockGo(self.l.r, plain, self.rig.s.handle, 1, @as(Kind, .matrix));
        return self.rig.bytes(self.base, self.f.?.outputs() * 4);
    }

    /// Every plant, finite or not, gives the padded tile's bytes.
    pub fn check(self: *Skew, at: c.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const out = try self.arena.allocator().dupe(u8, try self.rig.outputBytes());
        if (!std.mem.eql(u8, out, try self.replaced())) return "non-finite output differs from the padded tile's bytes";
        return null;
    }

    pub fn count(self: *Skew) usize {
        return self.f.?.outputs();
    }

    pub fn reference(self: *Skew, want: []f64, mag: []f64) !void {
        try self.f.?.reference(want, mag);
    }

    pub fn simulate(self: *Skew, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.f.?.simulate(acc, order, out);
    }
};
