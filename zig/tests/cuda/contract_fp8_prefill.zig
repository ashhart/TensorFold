//! Qwen3.8 prompt-only row-scaled E4M3 preparation and Q4/W8 projections.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;
const Rig = @import("contract_rig.zig").Rig;
const w4 = @import("w4a8_ref.zig");

fn noDepth(_: c.Shape) usize {
    return 0;
}

pub const Fns = struct {
    prepare: cuda.Function,
    q4: cuda.Function,
    w8: cuda.Function,
    expand: cuda.Function,
};

const prepare_shapes = [_]c.Shape{
    .{ .rows = 17, .n = 1, .k = 5120 },
    .{ .rows = 1, .n = 1, .k = 17408 },
    .{ .rows = 129, .n = 1, .k = 4096 },
};

pub const Prepare = struct {
    rig: Rig,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    fill: g.Fill = .random,
    x8: u64 = 0,
    sums: u64 = 0,
    scales: u64 = 0,
    args: cuda.launch.Args = .{},

    pub fn deinit(self: *Prepare) void {
        self.rig.deinit();
        self.arena.deinit();
    }
    pub fn kernel(self: *Prepare) g.Kernel {
        return .{ .spec = .{
            .name = "cuda qwen35_prefill8 tf_q35_prefill8_prepare quantize",
            .accum = .none,
            .depth = noDepth,
            .out = .raw,
            .nan = .{ .stated = "nonfinite BF16 input produces a deterministic nonfinite group sum" },
            .tiles = .{ .{ .size = 128 }, .{}, .{ .size = 256, .step = 64, .min = 64 } },
            .served = &prepare_shapes,
        }, .ctx = self, .vt = &vt };
    }
    const vt = g.hooks(Prepare);

    pub fn prepare(self: *Prepare, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        _ = seed;
        if (shape.rows == 0 or shape.rows > 4096 or shape.k == 0 or shape.k > 17408 or shape.k % 64 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.fill = fill;
        const x = try self.arena.allocator().alloc(u16, shape.rows * shape.k);
        @memset(x, c.toBf16(1));
        if (fill == .nonfinite) {
            x[0] = c.toBf16(std.math.nan(f32));
            if (shape.rows > 1) x[shape.k] = c.toBf16(std.math.inf(f32));
            if (shape.rows > 2) x[2 * shape.k] = c.toBf16(-std.math.inf(f32));
        }
        const dx = try self.rig.input(std.mem.sliceAsBytes(x));
        self.x8 = try self.rig.output(shape.rows * shape.k);
        self.sums = try self.rig.output(shape.rows * (shape.k / 64) * 2);
        self.scales = try self.rig.output(shape.rows * 4);
        var args: cuda.launch.Args = .{};
        inline for (.{ dx, @as(u64, 0), @as(u64, 0), self.x8, self.sums, self.scales }) |v| args.add(v);
        args.add(@as(c_int, @intCast(shape.k)));
        args.add(@as(c_int, 4));
        args.add(@as(f32, 0));
        self.args = args;
        return true;
    }
    pub fn run(self: *Prepare) !void {
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = 256 }, .shared = @intCast((self.shape.k + 256) * 4) }, self.rig.s, &self.args);
        try self.rig.s.synchronize();
    }
    pub fn count(self: *Prepare) usize {
        return self.shape.rows * self.shape.k + self.shape.rows * (self.shape.k / 64) * 2 + self.shape.rows * 4;
    }
    pub fn reference(_: *Prepare, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    pub fn check(self: *Prepare, at: g.Gate) !?[]const u8 {
        if (at == .nonfinite) {
            const raw = try self.rig.bytes(self.sums, self.shape.rows * (self.shape.k / 64) * 2);
            const first = c.fromBf16(std.mem.readInt(u16, raw[0..2], .little));
            return if (!std.math.isFinite(first)) null else "nonfinite input did not reach the group-sum output";
        }
        const q = try self.rig.bytes(self.x8, self.shape.rows * self.shape.k);
        for (q) |v| if (v != 0x7e) return "unit BF16 rows did not quantize to saturated positive E4M3";
        const sums = try self.rig.bytes(self.sums, self.shape.rows * (self.shape.k / 64) * 2);
        const want_sum = c.toBf16(64.0 * 448.0);
        var i: usize = 0;
        while (i < sums.len) : (i += 2) if (std.mem.readInt(u16, sums[i..][0..2], .little) != want_sum) return "unit BF16 rows produced the wrong unquantized group sum";
        const scales = try self.rig.bytes(self.scales, self.shape.rows * 4);
        const want_scale: u32 = @bitCast(@as(f32, 1.0 / 448.0));
        i = 0;
        while (i < scales.len) : (i += 4) if (std.mem.readInt(u32, scales[i..][0..4], .little) != want_scale) return "unit BF16 rows produced the wrong row scale";
        return null;
    }
};

const expand_shapes = [_]c.Shape{ .{ .rows = 1, .n = 1, .k = 64 }, .{ .rows = 1, .n = 128, .k = 5120 } };
pub const Expand = struct {
    rig: Rig,
    f: cuda.Function,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    out: u64 = 0,
    args: cuda.launch.Args = .{},
    pub fn deinit(self: *Expand) void {
        self.rig.deinit();
    }
    pub fn kernel(self: *Expand) g.Kernel {
        return .{ .spec = .{
            .name = "cuda qmm_prefill8 tf_qmm_prefill8_expand_w8",
            .accum = .none,
            .depth = noDepth,
            .out = .raw,
            .nan = .{ .stated = "integer Q4-to-E4M3 expansion has no floating-point nonfinite input" },
            .tiles = .{ .{}, .{ .size = 128 }, .{ .size = 64, .step = 64, .min = 64 } },
            .served = &expand_shapes,
        }, .ctx = self, .vt = &vt };
    }
    const vt = g.hooks(Expand);
    pub fn prepare(self: *Expand, shape: c.Shape, _: g.Fill, _: u64) !bool {
        if (shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        self.rig.reset();
        self.shape = shape;
        const words = shape.n * shape.k / 8;
        const src = try self.rig.gpu.gpa.alloc(u32, words);
        defer self.rig.gpu.gpa.free(src);
        for (src, 0..) |*v, i| v.* = @as(u32, @truncate(i)) *% 0x11111111;
        const input = try self.rig.input(std.mem.sliceAsBytes(src));
        self.out = try self.rig.output(words * 8);
        self.args = .{};
        self.args.add(input);
        self.args.add(self.out);
        self.args.add(@as(c_int, @intCast(words)));
        return true;
    }
    pub fn run(self: *Expand) !void {
        const words = self.shape.n * self.shape.k / 8;
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((words + 255) / 256) }, .block = .{ .x = 256 } }, self.rig.s, &self.args);
        try self.rig.s.synchronize();
    }
    pub fn count(self: *Expand) usize {
        return self.shape.n * self.shape.k;
    }
    pub fn reference(_: *Expand, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    pub fn check(self: *Expand, _: g.Gate) !?[]const u8 {
        const lut = [16]u8{ 0x00, 0x38, 0x40, 0x44, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57 };
        const raw = try self.rig.bytes(self.out, self.count());
        for (raw, 0..) |v, i| {
            const word: u32 = @as(u32, @truncate(i / 8)) *% 0x11111111;
            const byte = (i % 8) / 2 + (if (i % 2 == 0) @as(usize, 0) else 4);
            const code: u4 = @truncate(word >> @intCast(4 * byte));
            if (v != lut[code]) return "expanded E4M3 byte differs from the independent Q4 lookup";
        }
        return null;
    }
};

pub const MatmulKind = enum { q4, w8 };
const mm_shapes = [_]c.Shape{
    .{ .rows = 17, .n = 65, .k = 5120 },
    .{ .rows = 1, .n = 5120, .k = 6144 },
    .{ .rows = 129, .n = 129, .k = 256 },
};
pub const Matmul = struct {
    rig: Rig,
    kind: MatmulKind,
    f: cuda.Function,
    expand: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    out: u64 = 0,
    args: cuda.launch.Args = .{},
    expand_args: cuda.launch.Args = .{},
    words: usize = 0,
    nonfinite: bool = false,
    pub fn deinit(self: *Matmul) void {
        self.rig.deinit();
        self.arena.deinit();
    }
    pub fn kernel(self: *Matmul) g.Kernel {
        return .{ .spec = .{
            .name = if (self.kind == .q4) "cuda qmm_prefill8 tf_qmm_prefill8 Q4" else "cuda qmm_prefill8 tf_qmm_prefill8_w8",
            .accum = .none,
            .depth = noDepth,
            .out = .bf16,
            .nan = .{ .stated = "a nonfinite row scale propagates to every output in that row" },
            .tiles = .{ .{ .size = 128 }, .{ .size = 128 }, .{ .size = 64, .step = 64, .min = 64 } },
            .served = &mm_shapes,
        }, .ctx = self, .vt = &vt };
    }
    const vt = g.hooks(Matmul);
    pub fn prepare(self: *Matmul, shape: c.Shape, fill: g.Fill, _: u64) !bool {
        if (shape.rows == 0 or shape.rows > 4096 or shape.n == 0 or shape.k == 0 or shape.k > 17408 or shape.k % 64 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.nonfinite = fill == .nonfinite;
        const a = self.arena.allocator();
        const kg = shape.k / 64;
        const x = try a.alloc(u8, shape.rows * shape.k);
        @memset(x, 0x38);
        const xs = try a.alloc(u16, shape.rows * kg);
        @memset(xs, c.toBf16(0));
        const row_scales = try a.alloc(f32, shape.rows);
        @memset(row_scales, 1);
        if (fill == .nonfinite) {
            row_scales[0] = std.math.nan(f32);
            if (shape.rows > 1) row_scales[1] = std.math.inf(f32);
            if (shape.rows > 2) row_scales[2] = -std.math.inf(f32);
        }
        const codes = try a.alloc(u8, shape.n * shape.k);
        @memset(codes, 1);
        const layout = cuda.qlinear.Affine4.layout(shape.n, shape.k);
        const weight_words = try a.alloc(u32, layout.words / 4);
        w4.pack(weight_words, codes, shape.n, shape.k);
        const table_count = layout.scales / 2;
        const scales = try a.alloc(u16, table_count);
        @memset(scales, c.toBf16(1));
        const biases = try a.alloc(u16, table_count);
        @memset(biases, c.toBf16(0));
        const dx = try self.rig.input(x);
        const ds = try self.rig.input(std.mem.sliceAsBytes(xs));
        const da = try self.rig.input(std.mem.sliceAsBytes(row_scales));
        const dw = try self.rig.input(std.mem.sliceAsBytes(weight_words));
        const dscales = try self.rig.input(std.mem.sliceAsBytes(scales));
        const dbiases = try self.rig.input(std.mem.sliceAsBytes(biases));
        var actual_w = dw;
        self.words = layout.words / 4;
        if (self.kind == .w8) {
            actual_w = try self.rig.scratch(layout.words * 2);
            self.expand_args = .{};
            self.expand_args.add(dw);
            self.expand_args.add(actual_w);
            self.expand_args.add(@as(c_int, @intCast(self.words)));
        }
        self.out = try self.rig.output(shape.rows * shape.n * 2);
        self.args = .{};
        inline for (.{ dx, ds, da, actual_w, dscales, dbiases, self.out }) |v| self.args.add(v);
        const group = @max(1, @min((shape.rows + 127) / 128, (12 << 20) / (128 * shape.k)));
        inline for (.{ shape.rows, shape.n, shape.k, layout.npad, group }) |v| self.args.add(@as(c_int, @intCast(v)));
        return true;
    }
    pub fn run(self: *Matmul) !void {
        if (self.kind == .w8) try cuda.launch.launch(self.expand, .{ .grid = .{ .x = @intCast((self.words + 255) / 256) }, .block = .{ .x = 256 } }, self.rig.s, &self.expand_args);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((self.shape.rows + 127) / 128 * ((self.shape.n + 127) / 128)) }, .block = .{ .x = 128 }, .shared = if (self.kind == .w8) 49_920 else 37_632 }, self.rig.s, &self.args);
        try self.rig.s.synchronize();
    }
    pub fn count(self: *Matmul) usize {
        return self.shape.rows * self.shape.n;
    }
    pub fn reference(_: *Matmul, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    pub fn check(self: *Matmul, _: g.Gate) !?[]const u8 {
        const raw = try self.rig.bytes(self.out, self.count() * 2);
        const want = c.toBf16(@floatFromInt(self.shape.k));
        for (0..self.count()) |i| {
            const got = c.fromBf16(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little));
            const row = i / self.shape.n;
            if (self.nonfinite and row < 3) {
                const class = c.classOf(got);
                const expected: c.Class = if (row == 0) .nan else if (row == 1) .pos_inf else .neg_inf;
                if (class != expected) return "nonfinite row scale did not propagate to the output row";
            } else if (std.mem.readInt(u16, raw[i * 2 ..][0..2], .little) != want) return "unit E4M3 by unit Q4 projection differs from exact K";
        }
        return null;
    }
};
