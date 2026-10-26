//! cosmos3.cu under the contract: its rows (embeddings, widening, placement, norms, relu^2, M-RoPE) and attention.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_cosmos3_ref.zig");
const attn_k = @import("contract_cosmos3_attn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const ar = core.attention_range;
const D = ref.D;

pub const Which = enum { embed, widen8, place, norm, norm_add, relu2, qk };

/// A registration: a served kernel, or a control (contract_controls.cu's symbol) that drops one rounding point.
pub const Entry = struct { name: []const u8, which: Which, ctl: ?[:0]const u8 = null, gate: g.Gate = .shapes };

pub const entries = [_]Entry{
    .{ .name = "cuda cosmos3 tf_cos3_embed (MLX 8-bit rows, bf16 at MLX's points)", .which = .embed },
    .{ .name = "cuda cosmos3 tf_cos3_widen8 (the tower's 8-bit weights to bf16)", .which = .widen8 },
    .{ .name = "cuda cosmos3 tf_cos3_place (image rows over their placeholders)", .which = .place },
    .{ .name = "cuda cosmos3 tf_cos3_norm (RMS norm rows, MLX's bf16 points, 64-value sums)", .which = .norm },
    .{ .name = "cuda cosmos3 tf_cos3_norm with its residual add (bf16 h + r first)", .which = .norm_add },
    .{ .name = "cuda cosmos3 tf_cos3_relu2 (in place, 64-value sums of the bf16 squares)", .which = .relu2 },
    .{ .name = "cuda cosmos3 tf_cos3_qk (interleaved M-RoPE [24, 20, 20], keys and values into the cache)", .which = .qk },
    .{ .name = "control: cosmos3 embed rows without bf16(s q)", .which = .embed, .ctl = "tf_ctl_cos3_embed_drop_sq" },
    .{ .name = "control: cosmos3 widen8 without bf16(s q)", .which = .widen8, .ctl = "tf_ctl_cos3_widen8_drop_sq" },
    .{ .name = "control: cosmos3 place writing row 0 over every text row", .which = .place, .ctl = "tf_ctl_cos3_place_all" },
    .{ .name = "control: cosmos3 norm without bf16(h inv) before the weight", .which = .norm, .ctl = "tf_ctl_cos3_norm_drop_inner" },
    .{ .name = "control: cosmos3 norm reading h + r before its bf16 rounding", .which = .norm_add, .ctl = "tf_ctl_cos3_norm_drop_add" },
    .{ .name = "control: cosmos3 relu2 group sums of the fp32 squares", .which = .relu2, .ctl = "tf_ctl_cos3_relu2_fp32_sums" },
    .{ .name = "control: cosmos3 relu2 by fmaxf alone, a NaN input made 0", .which = .relu2, .ctl = "tf_ctl_cos3_relu2_drop_nan", .gate = .nonfinite },
    .{ .name = "control: cosmos3 M-RoPE with every pair on the t axis", .which = .qk, .ctl = "tf_ctl_cos3_qk_t_only" },
};

fn noDepth(_: c.Shape) usize {
    return 0;
}

/// 8 fmas a thread, a 5-level warp tree, then the block's 8 warps in order.
fn normDepth(_: c.Shape) usize {
    return 21;
}

const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub fn spec(e: Entry) c.Spec {
    const full = &[_]c.Shape{ .{ .rows = 1, .n = D, .k = D }, .{ .rows = 37, .n = D, .k = D }, .{ .rows = 300, .n = D, .k = D } };
    return switch (e.which) {
        .embed => .{ .name = e.name, .accum = .none, .depth = noDepth, .out = .bf16, .served = &.{ .{ .rows = 1, .n = D, .k = 1 }, .{ .rows = 37, .n = D, .k = 1 } } },
        .widen8 => .{ .name = e.name, .accum = .none, .depth = noDepth, .out = .bf16, .tiles = .{ .{}, .{ .size = 256, .step = 64, .min = 64 }, .{} }, .served = &.{ .{ .rows = 96, .n = 1152, .k = 1 }, .{ .rows = 24, .n = 4352, .k = 1 } } },
        .place => .{ .name = e.name, .accum = .none, .depth = noDepth, .out = .bf16, .served = &.{ .{ .rows = 9, .n = D, .k = 1 }, .{ .rows = 200, .n = D, .k = 1 } } },
        .norm, .norm_add => .{ .name = e.name, .accum = .f32, .depth = normDepth, .ops = 8, .mids = 1, .out = .bf16, .in_place = e.which == .norm_add, .served = full, .sums = .{ .bf16 = 0, .why = norm_why } },
        .relu2 => .{ .name = e.name, .accum = .none, .depth = noDepth, .out = .bf16, .in_place = true, .tiles = .{ .{}, .{ .size = 512, .step = 256, .min = 256 }, .{} }, .served = &.{ .{ .rows = 3, .n = 9216, .k = 1 }, .{ .rows = 17, .n = 9216, .k = 1 } } },
        .qk => .{ .name = e.name, .accum = .none, .depth = noDepth, .out = .bf16, .served = &.{ .{ .rows = 1, .n = 24, .k = 1 }, .{ .rows = 37, .n = 24, .k = 1 }, .{ .rows = 160, .n = 24, .k = 1 } } },
    };
}

const vocab = 97;
const qkv_w = (ref.QH + 2 * ref.KVH) * ref.HD;

pub const Glue = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    /// Host inputs, by kind: u16 bf16 rows, u8 codes, i32 ids, maps, positions and slots.
    h16: [4][]u16 = undefined,
    h8: []u8 = &.{},
    hi: [2][]i32 = undefined,
    dev: [8]u64 = @splat(0),
    cap: usize = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Glue) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Glue) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.ctl != null) self.entry.gate else null };
    }

    const vt = g.hooks(Glue);

    fn bf(rng: std.Random, lo: f32, hi: f32) u16 {
        return c.toBf16(lo + rng.float(f32) * (hi - lo));
    }

    fn put(h: []u16, at: usize, v: f32) void {
        if (at < h.len) h[at] = c.toBf16(v);
    }

    pub fn prepare(self: *Glue, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        self.fill = fill;
        const n = shape.rows;
        const nan = std.math.nan(f32);
        const inf = std.math.inf(f32);
        switch (self.entry.which) {
            .embed, .widen8 => {
                const width = if (self.entry.which == .embed) D else shape.n;
                if (width % 64 != 0 or n == 0) return false;
                const table_rows = if (self.entry.which == .embed) vocab else n;
                self.h8 = try a.alloc(u8, table_rows * width);
                rng.bytes(self.h8);
                for (0..2) |i| self.h16[i] = try a.alloc(u16, table_rows * width / 64);
                for (self.h16[0], self.h16[1]) |*s, *b| {
                    s.* = bf(rng, -0.03, 0.03);
                    b.* = bf(rng, -1.5, 1.5);
                }
                self.hi[0] = try a.alloc(i32, n);
                for (self.hi[0], 0..) |*id, i| id.* = @intCast(if (i == 0) vocab - 1 else rng.uintLessThan(usize, vocab));
                if (fill == .nonfinite) { // a row's group each: NaN scale, +Inf bias, -Inf scale
                    const row = [_]usize{ if (self.entry.which == .embed) @intCast(self.hi[0][0]) else 0, if (self.entry.which == .embed) @intCast(self.hi[0][1]) else 1, if (self.entry.which == .embed) @intCast(self.hi[0][2]) else 2 };
                    put(self.h16[0], row[0] * width / 64 + 1, nan);
                    put(self.h16[1], row[1] * width / 64 + 2, inf);
                    put(self.h16[0], row[2] * width / 64, -inf);
                }
                self.dev[0] = try self.rig.input(self.h8);
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.h16[0]));
                self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.h16[1]));
                self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(self.hi[0]));
                self.dev[4] = try self.rig.output(n * width * 2);
            },
            .place => {
                if (shape.n != D or n == 0) return false;
                const m = n / 2 + 1;
                self.h16[0] = try a.alloc(u16, n * D);
                self.h16[1] = try a.alloc(u16, m * D);
                for (self.h16[0]) |*x| x.* = bf(rng, -2, 2);
                for (self.h16[1]) |*x| x.* = bf(rng, -2, 2);
                self.hi[0] = try a.alloc(i32, n);
                for (self.hi[0], 0..) |*x, i| x.* = if (i % 3 == 0 or i >= 2 * m) -1 else @intCast(rng.uintLessThan(usize, m));
                if (fill == .nonfinite) {
                    self.hi[0][1] = 0;
                    put(self.h16[1], 5, nan);
                    put(self.h16[1], 6, inf);
                    put(self.h16[0], 7, -inf); // row 0 is text: kept
                }
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.h16[0]));
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.h16[1]));
                self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.hi[0]));
                self.dev[4] = try self.rig.output(n * D * 2);
            },
            .norm, .norm_add => {
                if (shape.k != D or n == 0) return false;
                for (0..3) |i| self.h16[i] = try a.alloc(u16, if (i == 2) D else n * D);
                for (self.h16[0]) |*x| x.* = bf(rng, -3, 3);
                for (self.h16[1]) |*x| x.* = if (fill == .stress) 0 else bf(rng, -1, 1);
                for (self.h16[2]) |*x| x.* = bf(rng, 0.5, 1.5);
                if (fill == .stress) for (0..n) |row| for (0..D) |i| { // rows.zig's rms stress: integers, big squares
                    self.h16[0][row * D + i] = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) - 31);
                };
                if (fill == .nonfinite) {
                    const into = if (self.entry.which == .norm_add) self.h16[1] else self.h16[0];
                    put(into, D / 3, nan);
                    put(into, D + D / 2, inf);
                    put(into, 2 * D + D - 1, -inf);
                }
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.h16[0]));
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.h16[1]));
                self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.h16[2]));
                self.dev[4] = try self.rig.output(n * D * 2); // x
                self.dev[5] = try self.rig.output(n * D / 64 * 4); // its group sums
                self.dev[6] = try self.rig.output(n * D * 2); // h, written in place (seeded from h0 each run)
            },
            .relu2 => {
                if (shape.n % 256 != 0 or n == 0) return false;
                self.h16[0] = try a.alloc(u16, n * shape.n);
                for (self.h16[0]) |*x| x.* = bf(rng, -4, 4);
                if (fill == .nonfinite) {
                    put(self.h16[0], 3, nan);
                    put(self.h16[0], shape.n + 70, inf);
                    put(self.h16[0], 2 * shape.n + 200, -inf);
                }
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.h16[0]));
                self.dev[4] = try self.rig.output(n * shape.n * 2);
                self.dev[5] = try self.rig.output(n * shape.n / 64 * 4);
            },
            .qk => {
                if (shape.n != ref.QH + ref.KVH or n == 0) return false;
                self.cap = 2 * n + 3;
                self.h16[0] = try a.alloc(u16, n * qkv_w);
                for (self.h16[0]) |*x| x.* = bf(rng, -2, 2);
                self.hi[0] = try a.alloc(i32, 3 * n);
                self.hi[1] = try a.alloc(i32, n);
                for (0..n) |r| {
                    const zero = fill == .nonfinite and r < 3; // planted rows at position 0: cos 1, sin 0 exactly
                    self.hi[0][3 * r] = if (zero) 0 else @intCast(rng.uintLessThan(u32, 20000));
                    self.hi[0][3 * r + 1] = if (zero) 0 else @intCast(rng.uintLessThan(u32, 64));
                    self.hi[0][3 * r + 2] = if (zero) 0 else @intCast(64 + rng.uintLessThan(u32, 64));
                }
                var perm = try a.alloc(i32, self.cap);
                for (perm, 0..) |*p, i| p.* = @intCast(i);
                rng.shuffle(i32, perm);
                @memcpy(self.hi[1], perm[0..n]);
                if (fill == .nonfinite) {
                    put(self.h16[0], 3, nan); // row 0 query head 0
                    put(self.h16[0], qkv_w + ref.QH * ref.HD + 5, inf); // row 1 key head 0
                    put(self.h16[0], 2 * qkv_w + (ref.QH + ref.KVH + 2) * ref.HD, -inf); // row 2 value head 2
                }
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.h16[0]));
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.hi[0]));
                self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.hi[1]));
                self.dev[4] = try self.rig.output(n * ref.QH * ref.HD * 2);
                self.dev[5] = try self.rig.output(ref.KVH * self.cap * ref.HD * 2);
                self.dev[6] = try self.rig.output(ref.KVH * self.cap * ref.HD * 2);
            },
        }
        return true;
    }

    fn reseed(self: *Glue, to: u64, from: u64, len: usize) !void {
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = to, .len = len };
        try b.copyFrom(0, from, len, self.rig.s.handle);
    }

    fn launch(self: *Glue, grid: [3]usize, block: usize, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |x| a.add(x);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block) } }, self.rig.s, &a);
    }

    pub fn run(self: *Glue) !void {
        const n = self.shape.rows;
        const v = self.dev;
        switch (self.entry.which) {
            .embed => try self.launch(.{ n, D / 256, 1 }, 256, .{ v[3], v[0], v[1], v[2], v[4] }),
            .widen8 => {
                const total: i64 = @intCast(n * self.shape.n);
                try self.launch(.{ (n * self.shape.n + 255) / 256, 1, 1 }, 256, .{ v[0], v[1], v[2], v[4], @as(c_int, @intCast(self.shape.n)), total });
            },
            .place => {
                try self.reseed(v[4], v[0], n * D * 2);
                try self.launch(.{ n, D / 256, 1 }, 256, .{ v[4], v[1], v[2] });
            },
            .norm, .norm_add => {
                try self.reseed(v[6], v[0], n * D * 2);
                try self.launch(.{ n, 1, 1 }, 256, .{ v[6], v[1], v[2], ref.eps, @as(c_int, @intFromBool(self.entry.which == .norm_add)), v[4], v[5] });
            },
            .relu2 => {
                try self.reseed(v[4], v[0], n * self.shape.n * 2);
                try self.launch(.{ n, self.shape.n / 256, 1 }, 256, .{ v[4], @as(c_int, @intCast(self.shape.n)), v[5] });
            },
            .qk => try self.launch(.{ ref.QH + ref.KVH, n, 1 }, 32, .{ v[0], v[1], v[2], @as(c_int, @intCast(self.cap)), std.math.log2(@as(f32, 1e8)), v[4], v[5], v[6] }),
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Glue) usize {
        const s = self.shape;
        return switch (self.entry.which) {
            .embed, .place, .norm, .norm_add => s.rows * D,
            .widen8, .relu2 => s.rows * s.n,
            .qk => s.rows * ref.QH * ref.HD,
        };
    }

    /// The norms' float64 outputs (no bf16 points) for gate 4; magnitude |y| / 2, as y ~ ss^-1/2.
    pub fn reference(self: *Glue, want: []f64, mag: []f64) !void {
        if (self.entry.which != .norm and self.entry.which != .norm_add) return error.NoSums;
        for (0..self.shape.rows) |r| {
            const row = try self.addedRow(r);
            var ss: f64 = 0;
            for (row) |x| ss += @as(f64, ref.val(x)) * ref.val(x);
            const inv = 1 / @sqrt(ss / D + ref.eps);
            for (0..D) |i| {
                want[r * D + i] = @as(f64, ref.val(row[i])) * inv * ref.val(self.h16[2][i]);
                mag[r * D + i] = @abs(want[r * D + i]) / 2;
            }
        }
    }

    pub fn simulate(self: *Glue, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.arena.allocator().alloc(f64, D);
        for (0..self.shape.rows) |r| {
            const row = try self.addedRow(r);
            for (t, row) |*x, h| x.* = @as(f64, ref.val(h)) * ref.val(h);
            const inv = 1 / @sqrt(rows.sumIn(acc, order, t) / D + ref.eps);
            for (0..D) |i| out[r * D + i] = @as(f64, ref.val(row[i])) * inv * ref.val(self.h16[2][i]);
        }
    }

    /// Row r of h as the norm reads it: h0, or bf16(h0 + r) with the residual added.
    fn addedRow(self: *Glue, r: usize) ![]const u16 {
        const h0 = self.h16[0][r * D ..][0..D];
        if (self.entry.which != .norm_add) return h0;
        const out = try self.arena.allocator().alloc(u16, D);
        for (out, h0, self.h16[1][r * D ..][0..D]) |*o, x, y| o.* = c.toBf16(ref.val(x) + ref.val(y));
        return out;
    }

    fn fail(self: *Glue, comptime fmt: []const u8, args: anytype) ?[]const u8 {
        return std.fmt.bufPrint(&self.msg, fmt, args) catch "check failed";
    }

    fn get(self: *Glue, comptime T: type, at: u64, len: usize) ![]T {
        const host = try self.arena.allocator().alloc(T, len);
        @memcpy(std.mem.sliceAsBytes(host), try self.rig.bytes(at, len * @sizeOf(T)));
        return host;
    }

    /// Each kernel's own check: bits at its rounding points, or within the fp32 spread where a step is inexact.
    pub fn check(self: *Glue, at: g.Gate) !?[]const u8 {
        const n = self.shape.rows;
        var nonfinite: usize = 0;
        switch (self.entry.which) {
            .embed, .widen8 => {
                const width = if (self.entry.which == .embed) D else self.shape.n;
                const out = try self.get(u16, self.dev[4], n * width);
                for (0..n) |r| for (0..width) |i| {
                    const t: usize = if (self.entry.which == .embed) @intCast(self.hi[0][r]) else r;
                    const gi = t * width / 64 + i / 64;
                    const want = ref.dequant(self.h16[0][gi], self.h8[t * width + i], self.h16[1][gi], true);
                    if (ref.isNan(want)) nonfinite += 1;
                    if (!ref.same(out[r * width + i], want)) return self.fail("row {d} col {d}: {x} where float64 at its points gives {x}", .{ r, i, out[r * width + i], want });
                };
            },
            .place => {
                const out = try self.get(u16, self.dev[4], n * D);
                for (0..n) |r| for (0..D) |i| {
                    const m = self.hi[0][r];
                    const want = if (m >= 0) self.h16[1][@as(usize, @intCast(m)) * D + i] else self.h16[0][r * D + i];
                    if (ref.isNan(want) or want & 0x7fff == 0x7f80) nonfinite += 1;
                    if (out[r * D + i] != want) return self.fail("row {d} col {d}: {x} where {x}", .{ r, i, out[r * D + i], want });
                };
            },
            .norm, .norm_add => {
                const x = try self.get(u16, self.dev[4], n * D);
                const xs = try self.get(f32, self.dev[5], n * D / 64);
                const h = try self.get(u16, self.dev[6], n * D);
                for (0..n) |r| {
                    const row = try self.addedRow(r);
                    for (h[r * D ..][0..D], row, 0..) |got, want, i| if (!ref.same(got, want)) return self.fail("row {d} col {d}: h is not bf16(h + r)", .{ r, i });
                    const nr = ref.NormRow.of(row);
                    for (0..D) |i| {
                        const got = x[r * D + i];
                        if (ref.isNan(got)) nonfinite += 1;
                        if (!nr.holds(row[i], self.h16[2][i], got)) return self.fail("row {d} col {d}: {x} outside w bf16(h inv) for inv in its fp32 spread", .{ r, i, got });
                    }
                    if (try sums(self, x[r * D ..][0..D], xs[r * D / 64 ..][0 .. D / 64])) |why| return why;
                }
            },
            .relu2 => {
                const w = self.shape.n;
                const x = try self.get(u16, self.dev[4], n * w);
                const xs = try self.get(f32, self.dev[5], n * w / 64);
                for (0..n * w) |i| {
                    const want = ref.relu2(self.h16[0][i]);
                    if (ref.isNan(want) or want == 0x7f80) nonfinite += 1;
                    if (!ref.same(x[i], want)) return self.fail("element {d}: {x} where bf16(relu(x)^2) is {x}", .{ i, x[i], want });
                }
                if (try sums(self, x, xs)) |why| return why;
            },
            .qk => if (try self.checkQk(&nonfinite)) |why| return why,
        }
        if (at == .nonfinite and nonfinite == 0) return self.fail("the planted values reach no output: no proof", .{});
        return null;
    }

    fn checkQk(self: *Glue, nonfinite: *usize) !?[]const u8 {
        const n = self.shape.rows;
        const hd = ref.HD;
        const q = try self.get(u16, self.dev[4], n * ref.QH * hd);
        const kc = try self.get(u16, self.dev[5], ref.KVH * self.cap * hd);
        const vc = try self.get(u16, self.dev[6], ref.KVH * self.cap * hd);
        var point: [ref.HD]f64 = undefined;
        var half: [ref.HD]f64 = undefined;
        var st: ar.Stats = .{};
        const log2_theta = std.math.log2(@as(f32, 1e8));
        var written = try self.arena.allocator().alloc(bool, self.cap);
        @memset(written, false);
        for (0..n) |r| {
            const pos = [3]i32{ self.hi[0][3 * r], self.hi[0][3 * r + 1], self.hi[0][3 * r + 2] };
            const slot: usize = @intCast(self.hi[1][r]);
            written[slot] = true;
            for (0..ref.QH + ref.KVH) |head| {
                const src = self.h16[0][r * qkv_w + head * hd ..][0..hd];
                ref.rotate(src, pos, log2_theta, &point, &half);
                const got = if (head < ref.QH) q[(r * ref.QH + head) * hd ..][0..hd] else kc[((head - ref.QH) * self.cap + slot) * hd ..][0..hd];
                for (got, point, half, 0..) |x, p, hw, i| {
                    if (!ref.within(p, hw, x, &st)) return self.fail("row {d} head {d} dim {d}: {x} outside {e:.6} +- {e:.3}", .{ r, head, i, x, p, hw });
                    if (ref.isNan(x)) nonfinite.* += 1;
                }
                if (head >= ref.QH) {
                    const kh = head - ref.QH;
                    const v = self.h16[0][r * qkv_w + (ref.QH + ref.KVH + kh) * hd ..][0..hd];
                    if (!std.mem.eql(u16, vc[(kh * self.cap + slot) * hd ..][0..hd], v)) return self.fail("row {d} value head {d}: not its copy", .{ r, kh });
                }
            }
        }
        for (written, 0..) |w, slot| if (!w) for (0..ref.KVH) |kh| {
            for (kc[(kh * self.cap + slot) * hd ..][0..hd], vc[(kh * self.cap + slot) * hd ..][0..hd]) |x, y|
                if (x != 0xffff or y != 0xffff) return self.fail("cache slot {d} written though no row has it", .{slot});
        };
        return null;
    }
};

/// Each 64 outputs' group sum, bit for bit: the fp32 chain in order (an fp16 or reordered sum leaves these bits).
fn sums(self: *Glue, x: []const u16, xs: []const f32) !?[]const u8 {
    var v: [64]f32 = undefined;
    for (xs, 0..) |got, j| {
        for (&v, x[64 * j ..][0..64]) |*o, b| o.* = ref.val(b);
        const want = ref.group64(&v);
        if (@as(u32, @bitCast(got)) != @as(u32, @bitCast(want)) and !(std.math.isNan(got) and std.math.isNan(want)))
            return self.fail("group sum {d}: {e} where the fp32 chain of the outputs gives {e}", .{ j, got, want });
    }
    return null;
}

/// Every cosmos3 registration, its controls from the controls image, and attention's from contract_cosmos3_attn.zig.
pub const Group = struct {
    mod: ?cuda.Module = null,
    glue: [entries.len]Glue = undefined,
    made: usize = 0,
    attn: attn_k.Group = .{},

    pub fn add(self: *Group, gpu: check.Gpu, s: cuda.Stream, ctl: cuda.Module, list: *std.ArrayList(g.Kernel)) !void {
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.cosmos3);
        for (entries) |e| {
            const f = if (e.ctl) |sym| try ctl.function(sym) else try self.mod.?.function(symbol(e.which));
            self.glue[self.made] = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = f, .arena = .init(gpu.gpa) };
            self.made += 1;
        }
        for (self.glue[0..self.made]) |*x| try list.append(gpu.gpa, x.kernel());
        try self.attn.add(gpu, s, self.mod.?, ctl, list);
    }

    pub fn deinit(self: *Group) void {
        self.attn.deinit();
        for (self.glue[0..self.made]) |*x| x.deinit();
        if (self.mod) |*m| m.unload();
    }
};

fn symbol(w: Which) [:0]const u8 {
    return switch (w) {
        .embed => "tf_cos3_embed",
        .widen8 => "tf_cos3_widen8",
        .place => "tf_cos3_place",
        .norm, .norm_add => "tf_cos3_norm",
        .relu2 => "tf_cos3_relu2",
        .qk => "tf_cos3_qk",
    };
}
