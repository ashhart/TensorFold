//! The fused draw replaces the real float/topk/lookup/keyed pipeline on identical BF16 rows.
const std = @import("std");
const cuda = @import("cuda");
const nemo = @import("nemotron");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
fn noSums(_: c.Shape) usize {
    return 0;
}
pub const Greedy = struct {
    rig: rig_.Rig,
    fused: cuda.Function,
    keyed: cuda.Function,
    torch: *const nemo.torch_ops.Functions,
    mapped: bool = true,
    corrupt: bool = false,
    shape: c.Shape = undefined,
    input: u64 = 0,
    ids: u64 = 0,
    output: u64 = 0,
    prob: u64 = 0,
    oracle: std.ArrayList(u8) = .empty,
    pub fn deinit(self: *Greedy) void {
        self.oracle.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }
    pub fn kernel(self: *Greedy) g.Kernel {
        const name = if (self.corrupt) "control: Nemotron MTP fused draw output byte" else if (self.mapped) "cuda Nemotron MTP fused greedy subset" else "cuda Nemotron MTP fused greedy full";
        return .{ .spec = .{ .name = name, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "same top-28 radix candidates and top-20 keyed rule, including NaN/Inf probability classes" }, .replaces = "toF32/topk/lookup/tf_nemo_keyed_greedy", .tiles = .{ .{ .size = 1 }, .{}, .{ .size = 256 } }, .served = &.{ .{ .rows = 1, .n = 1, .k = 49152 }, .{ .rows = 5, .n = 1, .k = 131072 }, .{ .rows = 1, .n = 1, .k = 28 } } }, .ctx = self, .vt = &vt, .control = if (self.corrupt) .shapes else null };
    }
    const vt = g.hooks(Greedy);
    pub fn prepare(self: *Greedy, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows < 1 or shape.rows > 8 or shape.k < 1 or shape.k > 131072 or shape.n != 1) return false;
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const bits = try a.alloc(u16, shape.rows * shape.k);
        defer a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (bits) |*b| b.* = c.toBf16(@as(f32, @floatFromInt(rng.intRangeAtMost(i32, -16, 16))) / 8);
        if (fill == .nonfinite) for (0..shape.rows) |row| {
            bits[row * shape.k] = switch (row % 5) {
                0 => 0x7fc1,
                1 => 0x7f80,
                2 => 0xff80,
                3 => 0x7fc2,
                else => 0,
            };
            if (row % 5 == 3 and shape.k > 1) bits[row * shape.k + 1] = 0x7f80;
        };
        self.input = try self.rig.input(std.mem.sliceAsBytes(bits));
        const ids = try a.alloc(i64, shape.k);
        defer a.free(ids);
        for (ids, 0..) |*id, i| id.* = @intCast(shape.k - 1 - i);
        self.ids = if (self.mapped) try self.rig.input(std.mem.sliceAsBytes(ids)) else 0;
        self.output = try self.rig.output(shape.rows * 4);
        self.prob = try self.rig.output(shape.rows * 4);
        const n = shape.k;
        const candidates = @min(n, 28);
        const rows = shape.rows;
        const f32s = try self.rig.scratch(rows * n * 4);
        const values = try self.rig.scratch(rows * candidates * 4);
        const columns = try self.rig.scratch(rows * candidates * 8);
        const cand = try self.rig.scratch(rows * candidates * 8);
        const scratch = try self.rig.scratch(nemo.torch_ops.topkScratchBytes(rows, n));
        const invalid = try self.rig.scratch(4);
        const out = try self.rig.scratch(rows * 8);
        const torch: nemo.torch_ops.Torch = .{ .f = self.torch, .s = self.rig.s };
        try torch.toF32(self.input, f32s, rows * n);
        try torch.topk(f32s, n, rows, candidates, values, columns, scratch);
        if (self.mapped) try torch.lookup(self.ids, n, columns, cand, rows * candidates, invalid);
        var args: cuda.Args = .{};
        for ([_]u64{ values, if (self.mapped) cand else columns, out, out + rows * 4 }) |x| args.add(x);
        args.add(@as(c_int, @intCast(candidates)));
        args.add(@as(c_int, 20));
        try cuda.launch.launch(self.keyed, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, self.rig.s, &args);
        const want = try self.rig.bytes(out, rows * 8);
        self.oracle.clearRetainingCapacity();
        try self.oracle.appendSlice(a, want);
        return true;
    }
    pub fn run(self: *Greedy) !void {
        for (0..self.shape.rows) |row| {
            var args: cuda.Args = .{};
            for ([_]u64{ self.input + row * self.shape.k * 2, self.ids, self.output + row * 4, self.prob + row * 4 }) |x| args.add(x);
            args.add(@as(c_int, @intCast(self.shape.k)));
            try cuda.launch.launch(self.fused, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, self.rig.s, &args);
        }
        if (self.corrupt) {
            const bytes = try self.rig.bytes(self.output, 4);
            bytes[0] ^= 1;
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.output, .len = 4 };
            try b.upload(0, bytes);
        }
    }
    pub fn count(self: *Greedy) usize {
        return self.shape.rows * 8;
    }
    pub fn reference(_: *Greedy, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    pub fn replaced(self: *Greedy) ![]const u8 {
        return self.oracle.items;
    }
    pub fn check(self: *Greedy, _: g.Gate) !?[]const u8 {
        const got = try self.rig.outputBytes();
        const bytes = self.shape.rows * 4;
        if (!std.mem.eql(u8, got[0..bytes], self.oracle.items[0..bytes])) return "draft id differs from original candidates/tie rule";
        for (0..self.shape.rows) |i| {
            const at = bytes + i * 4;
            const x: f32 = @bitCast(std.mem.readInt(u32, got[at..][0..4], .little));
            const y: f32 = @bitCast(std.mem.readInt(u32, self.oracle.items[at..][0..4], .little));
            if (std.math.isNan(x) and std.math.isNan(y)) continue;
            if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) return "top-20 probability differs from original warp reduction";
        }
        return null;
    }
};
