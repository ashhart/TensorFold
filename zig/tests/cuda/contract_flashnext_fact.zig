//! Flash Next's prompt matmul on qmm.cu's factored arithmetic (flashnext_fact.cu) bit-equal to the 16x64 lane tile.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const parts = flash.batch_state;
const tk = flash.tile_kernels;

pub const Kind = enum { prompt, fold };

/// A group's 32 products from zero in two k16 steps, two fmas a group, then up to 8 slices added in order.
fn depth(s: c.Shape) usize {
    return 32 + 2 * (s.k / 32) + 8;
}

/// The engine's K slices: PLE values 4, out projections 8.
fn slicesOf(s: c.Shape) usize {
    if (s.n == 2560 and s.k == 2560) return 4;
    if (s.k == 6144) return 8;
    for ([_]usize{ 8, 4, 2 }) |sk| if ((s.k / 32) % (2 * sk) == 0) return sk;
    return 1;
}

const tiles = [3]c.Tile{ .{ .size = 64 }, .{ .size = 128, .step = 32, .min = 32 }, .{ .size = 256, .step = 64, .min = 64 } };

pub fn spec(kind: Kind) c.Spec {
    return switch (kind) {
        .prompt => .{ .name = "cuda flashnext factored prompt matmul 64x128", .accum = .f32, .depth = depth, .ops = 2, .out = .bf16, .replaces = "qmm.cu's 16x64 lane tile", .tiles = tiles, .served = &.{ .{ .rows = 64, .n = 13952, .k = 2560 }, .{ .rows = 100, .n = 2560, .k = 2560 } } },
        .fold => .{ .name = "cuda flashnext factored prompt matmul 64x128, K slices folded", .accum = .f32, .depth = depth, .ops = 2, .out = .f32, .replaces = "the lane tile's K slices then fn_sum_slices", .tiles = tiles, .served = &.{ .{ .rows = 64, .n = 2560, .k = 2560 }, .{ .rows = 100, .n = 2560, .k = 6144 } } },
    };
}

pub const Fact = struct {
    rig: rig_.Rig,
    kind: Kind,
    lane: cuda.Function,
    prompt: cuda.Function,
    fold: cuda.Function,
    sum: cuda.Function,
    f: ?Affine = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    sk: usize = 1,
    npad: usize = 0,
    x: u64 = 0,
    xs: u64 = 0,
    w: u64 = 0,
    s: u64 = 0,
    b: u64 = 0,
    part: u64 = 0,
    base: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Fact) void {
        if (self.f) |f| f.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Fact) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Fact);

    pub fn prepare(self: *Fact, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.f) |f| f.deinit();
        self.f = null;
        self.rig.reset();
        // two groups a stage: K in 64s (every served K is)
        if (shape.rows == 0 or shape.n == 0 or shape.n % 8 != 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        self.sk = if (self.kind == .prompt) 1 else slicesOf(shape);
        if (self.kind == .fold and self.sk == 1) return false;
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const k = shape.k;
        const f = try Affine.init(a, shape, 4, 32);
        self.f = f;
        try f.fill(fill, seed);
        // the producers' fp32 group sums, in order
        const xs = try a.alloc(f32, shape.rows * (k / 32));
        defer a.free(xs);
        for (xs, 0..) |*v, i| {
            var t: f32 = 0;
            for (f.x[i * 32 ..][0..32]) |h| t += c.fromBf16(h);
            v.* = t;
        }
        const words = try a.alloc(u32, shape.n * k / 8);
        defer a.free(words);
        f.words(words);
        var p = try flash.qmm.pack(a, std.mem.sliceAsBytes(words), std.mem.sliceAsBytes(f.scales), std.mem.sliceAsBytes(f.biases), shape.n, k / 8);
        defer p.deinit(a);
        self.npad = p.npad;
        self.x = try self.rig.input(std.mem.sliceAsBytes(f.x));
        self.xs = try self.rig.input(std.mem.sliceAsBytes(xs));
        self.w = try self.rig.input(p.weight);
        self.s = try self.rig.input(p.scales);
        self.b = try self.rig.input(p.biases);
        const esize = c.size(spec(self.kind).out);
        self.out = try self.rig.output(shape.rows * shape.n * esize);
        self.base = try self.rig.scratch(shape.rows * shape.n * esize);
        if (self.kind == .fold) self.part = try self.rig.scratch(self.sk * shape.rows * shape.n * 4);
        return true;
    }

    pub fn run(self: *Fact) !void {
        switch (self.kind) {
            .prompt => try self.launch(self.prompt, 64, 128, 256, tk.fact_smem, self.out, 0, 1, 1),
            .fold => try self.launch(self.fold, 64, 128, 256, tk.fact_smem, 0, self.out, self.sk, 1),
        }
        try self.rig.s.synchronize();
    }

    /// cuda_batch.zig's lane() launch: grid of row and column tiles, z the slices on the grid.
    fn launch(self: *Fact, f: cuda.Function, bm: usize, bn: usize, block: u32, shared: u32, out: u64, part: u64, sk: usize, z: usize) !void {
        const sh = self.shape;
        const rows_t = (sh.rows + bm - 1) / bm;
        const cols_t = (sh.n + bn - 1) / bn;
        const sweep = @max(@as(usize, 1), @min(rows_t, (12 << 20) / (bm * sh.k * 2)));
        var a: cuda.Args = .{};
        for ([_]u64{ self.x, self.xs, self.w, self.s, self.b, out, part }) |v| a.add(v);
        for ([_]usize{ sh.rows, sh.n, sh.k, sk, self.npad, sh.k, sweep }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows_t * cols_t), .z = @intCast(z) }, .block = .{ .x = block }, .shared = shared }, self.rig.s, &a);
    }

    /// The lane tile on the same inputs: bf16 out, or its K slices summed in order by fn_sum_slices (fp32).
    pub fn replaced(self: *Fact) ![]const u8 {
        const n = self.shape.rows * self.shape.n;
        switch (self.kind) {
            .prompt => {
                try self.launch(self.lane, 16, 64, 128, parts.qmm_smem, self.base, 0, 1, 1);
                try self.rig.s.synchronize();
                return self.rig.bytes(self.base, n * 2);
            },
            .fold => {
                try self.launch(self.lane, 16, 64, 128, parts.qmm_smem, 0, self.part, self.sk, self.sk);
                var a: cuda.Args = .{};
                a.add(self.part);
                a.add(self.base);
                a.add(@as(c_int, 0));
                a.add(@as(c_longlong, @intCast(n)));
                a.add(@as(c_int, @intCast(self.sk)));
                try cuda.launch.launch(self.sum, .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, self.rig.s, &a);
                try self.rig.s.synchronize();
                return self.rig.bytes(self.base, n * 4);
            },
        }
    }

    pub fn count(self: *Fact) usize {
        return self.f.?.outputs();
    }

    pub fn reference(self: *Fact, want: []f64, mag: []f64) !void {
        return self.f.?.reference(want, mag);
    }

    pub fn simulate(self: *Fact, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.f.?.simulate(acc, order, out);
    }
};
