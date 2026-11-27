//! Contract cases for the remaining 13 `hy4_rows` kernels plus a wrong-page negative control for `kv_write_paged`.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hr = @import("hy4_route_rows_cuda");
const rig_ = @import("contract_rig.zig");
const exact = @import("contract_hy4_rows_exact.zig");

const c = core.contract.spec;
const g = core.contract.gates;

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

fn twoDepth(_: c.Shape) usize {
    return 2;
}

fn kvDepth(_: c.Shape) usize {
    return 512;
}

fn mropeAxis(pair: usize) usize {
    return if (pair < 30 and (pair % 3) != 0) (pair % 3) else 0;
}

fn randBf16(rng: std.Random) u16 {
    return c.toBf16(rng.float(f32) * 2.0 - 1.0);
}

fn hasMixedFinite(out_ty: c.Elem, bytes: []const u8, n: usize) bool {
    var fin: usize = 0;
    var nonfin: usize = 0;
    for (0..n) |i| {
        if (c.classOf(c.read(out_ty, bytes, i)) == .finite) fin += 1 else nonfin += 1;
    }
    return fin > 0 and nonfin > 0;
}

fn buildPageTable(a: std.mem.Allocator, streams: usize, pages_per_stream: usize, pt_stride: usize) !struct { pt: []u32, phys_pages: usize } {
    const mapped = streams * pages_per_stream;
    const phys_pages = mapped + 2;
    const perm = try a.alloc(u32, phys_pages);
    defer a.free(perm);
    for (0..phys_pages) |i| perm[i] = @intCast((i * 5 + 3) % phys_pages);
    if (phys_pages % 5 == 0) {
        for (0..phys_pages) |i| perm[i] = @intCast((phys_pages - 1 - i + 2) % phys_pages);
    }
    const pt = try a.alloc(u32, streams * pt_stride);
    @memset(pt, perm[mapped]);
    for (0..streams) |s| for (0..pages_per_stream) |p| {
        pt[s * pt_stride + p] = perm[s * pages_per_stream + p];
    };
    return .{ .pt = pt, .phys_pages = phys_pages };
}

// Float64-verified numeric helpers: comb_swiglu_bf16, vit_qkv, q_rope, q_rope_mrope, kv_write, kv_write_ex.
pub const NumKind = enum { comb_swiglu, vit_qkv, q_rope, q_rope_mrope, kv_write, kv_write_ex };

const comb_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 32, .k = 2 },
    .{ .rows = 4, .n = 32, .k = 2 },
    .{ .rows = 17, .n = 32, .k = 2 },
};
const vit_qkv_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 4, .k = 64 },
    .{ .rows = 5, .n = 4, .k = 64 },
    .{ .rows = 13, .n = 4, .k = 64 },
};
const rope_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 256, .k = 64 },
    .{ .rows = 4, .n = 256, .k = 64 },
    .{ .rows = 13, .n = 256, .k = 64 },
};
const kv_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 576, .k = 512 },
    .{ .rows = 4, .n = 576, .k = 512 },
    .{ .rows = 9, .n = 576, .k = 512 },
};

pub const NumCase = struct {
    rig: rig_.Rig,
    rows: *const hr.Rows,
    mla: *const exact.MlaRef,
    kind: NumKind,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    rank: u32 = 0,
    use_multi_cache: bool = false,
    ring: usize = 6,
    eps: f32 = 1e-6,
    qa_stride: usize = 640,
    q_lora: usize = 32,
    dev: [8]u64 = @splat(0),
    dev_out: u64 = 0,
    out_elems: usize = 0,
    // Host copies for reference().
    h_f32_a: std.ArrayList(f32) = .empty,
    h_f32_b: std.ArrayList(f32) = .empty,
    h_u16_a: std.ArrayList(u16) = .empty,
    h_u16_b: std.ArrayList(u16) = .empty,
    h_i32_a: std.ArrayList(i32) = .empty,
    h_i32_b: std.ArrayList(i32) = .empty,
    h_i32_c: std.ArrayList(i32) = .empty,

    pub fn deinit(self: *NumCase) void {
        const a = self.rig.gpu.gpa;
        self.h_f32_a.deinit(a);
        self.h_f32_b.deinit(a);
        self.h_u16_a.deinit(a);
        self.h_u16_b.deinit(a);
        self.h_i32_a.deinit(a);
        self.h_i32_b.deinit(a);
        self.h_i32_c.deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *NumCase) g.Kernel {
        const name: []const u8 = switch (self.kind) {
            .comb_swiglu => "cuda hy4_rows hy4_rows_comb_swiglu_bf16",
            .vit_qkv => "cuda hy4_rows hy4_rows_vit_qkv",
            .q_rope => "cuda hy4_rows hy4_rows_q_rope",
            .q_rope_mrope => "cuda hy4_rows hy4_rows_q_rope_mrope",
            .kv_write => "cuda hy4_rows hy4_rows_kv_write",
            .kv_write_ex => "cuda hy4_rows hy4_rows_kv_write_ex",
        };
        const served: []const c.Shape = switch (self.kind) {
            .comb_swiglu => &comb_shapes,
            .vit_qkv => &vit_qkv_shapes,
            .q_rope, .q_rope_mrope => &rope_shapes,
            .kv_write, .kv_write_ex => &kv_shapes,
        };
        return .{
            .spec = .{
                .name = name,
                .accum = .none,
                .depth = zeroDepth,
                .ops = 20,
                .out = .bf16,
                .served = served,
            },
            .ctx = self,
            .vt = &vt,
        };
    }

    pub fn bounds(self: *const NumCase, want: []const f64, mag: []const f64, allow: []f64) !void {
        const is_kv = self.kind == .kv_write or self.kind == .kv_write_ex;
        const is_bf_mid = is_kv or self.kind == .q_rope or self.kind == .q_rope_mrope;
        const spec: c.Spec = .{
            .name = "",
            .accum = if (is_kv or self.kind == .comb_swiglu) .f32 else .none,
            .depth = if (is_kv) kvDepth else if (self.kind == .comb_swiglu) twoDepth else zeroDepth,
            .ops = 20,
            .mids = if (is_bf_mid) 2 else 0,
            .out = .bf16,
            .served = &kv_shapes,
        };
        for (allow, want, mag) |*l, w, m| l.* = c.bound(spec, self.shape, w, m) + (if (is_bf_mid) 0x1p-7 * m else 0);
    }

    const vt = g.hooks(NumCase);

    pub fn prepare(self: *NumCase, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed ^ (@as(u64, @backingInt(self.kind)) * 0x9e37));
        const rng = prng.random();
        switch (self.kind) {
            .comb_swiglu => {
                self.rank = @intCast(shape.rows & 1);
                const half = 16;
                const sw = 16;
                const comb_stride = 40;
                const gu_stride = 48;
                try self.h_f32_a.resize(a, shape.rows * comb_stride);
                try self.h_f32_b.resize(a, shape.rows * half);
                try self.h_u16_a.resize(a, shape.rows * gu_stride);
                for (self.h_f32_a.items) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
                for (self.h_f32_b.items) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
                for (self.h_u16_a.items) |*v| v.* = randBf16(rng);
                if (fill == .nonfinite) {
                    self.h_f32_a.items[4] = std.math.nan(f32);
                    self.h_u16_a.items[8] = 0x7fc0;
                }
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.h_f32_a.items));
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.h_f32_b.items));
                self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16_a.items));
                self.out_elems = shape.rows * (half + sw);
            },
            .vit_qkv => {
                const heads = shape.n;
                const hd = shape.k;
                const d = heads * hd;
                try self.h_f32_a.resize(a, shape.rows * 3 * d);
                try self.h_u16_a.resize(a, 3 * d);
                for (self.h_f32_a.items) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
                for (self.h_u16_a.items) |*v| v.* = randBf16(rng);
                if (fill == .nonfinite) {
                    self.h_f32_a.items[0] = std.math.nan(f32);
                    self.h_f32_a.items[d] = std.math.inf(f32);
                }
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.h_f32_a.items));
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16_a.items));
                self.out_elems = 3 * heads * shape.rows * hd;
            },
            .q_rope, .q_rope_mrope => {
                const heads = 4;
                const xh = 72;
                const xs = heads * xh + 8;
                try self.h_u16_a.resize(a, shape.rows * xs);
                try self.h_f32_a.resize(a, 32);
                try self.h_i32_a.resize(a, shape.rows);
                try self.h_i32_b.resize(a, 3 * shape.rows);
                for (self.h_u16_a.items) |*v| v.* = randBf16(rng);
                for (0..32) |i| self.h_f32_a.items[i] = @floatCast(std.math.pow(f64, 10000.0, -2.0 * @as(f64, @floatFromInt(i)) / 64.0));
                for (0..shape.rows) |r| {
                    self.h_i32_a.items[r] = @intCast(r * 7 + 3);
                    self.h_i32_b.items[0 * shape.rows + r] = @intCast(r * 5 + 1);
                    self.h_i32_b.items[1 * shape.rows + r] = @intCast(r * 11 + 2);
                    self.h_i32_b.items[2 * shape.rows + r] = @intCast(r * 13 + 4);
                }
                if (fill == .nonfinite) self.h_u16_a.items[0] = 0x7fc0;
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16_a.items));
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.h_f32_a.items));
                self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.h_i32_a.items));
                self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(self.h_i32_b.items));
                self.out_elems = shape.rows * heads * 64;
            },
            .kv_write, .kv_write_ex => {
                self.use_multi_cache = shape.rows > 1;
                try self.h_u16_a.resize(a, shape.rows * self.qa_stride);
                try self.h_u16_b.resize(a, 512);
                try self.h_f32_a.resize(a, 32);
                try self.h_i32_a.resize(a, shape.rows);
                try self.h_i32_b.resize(a, shape.rows);
                try self.h_i32_c.resize(a, 3 * shape.rows);
                for (self.h_u16_a.items) |*v| v.* = randBf16(rng);
                for (self.h_u16_b.items) |*v| v.* = randBf16(rng);
                for (0..32) |i| self.h_f32_a.items[i] = @floatCast(std.math.pow(f64, 10000.0, -2.0 * @as(f64, @floatFromInt(i)) / 64.0));
                for (0..shape.rows) |r| {
                    // Ensure unique destination slot (r < ring when single_cache; per-row cache when multi_cache).
                    self.h_i32_a.items[r] = @intCast(if (self.use_multi_cache) (r + self.ring) else r);
                    self.h_i32_b.items[r] = @intCast(r * 9 + 5);
                    self.h_i32_c.items[0 * shape.rows + r] = @intCast(r * 3 + 2);
                    self.h_i32_c.items[1 * shape.rows + r] = @intCast(r * 7 + 4);
                    self.h_i32_c.items[2 * shape.rows + r] = @intCast(r * 11 + 6);
                }
                if (fill == .nonfinite) {
                    self.h_u16_a.items[self.q_lora + 5] = 0x7fc0;
                    self.h_u16_a.items[self.q_lora + 512] = 0x7f80;
                }
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16_a.items));
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16_b.items));
                self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.h_f32_a.items));
                self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(self.h_i32_a.items));
                self.dev[4] = try self.rig.input(std.mem.sliceAsBytes(self.h_i32_b.items));
                self.dev[5] = try self.rig.input(std.mem.sliceAsBytes(self.h_i32_c.items));
                // Each row writes into its own cache slice of `ring * 576` bf16s (or 1 slice when rows==1).
                const caches_n: usize = if (self.use_multi_cache) shape.rows else 1;
                self.out_elems = caches_n * self.ring * 576;
            },
        }
        self.dev_out = try self.rig.output(self.out_elems * 2);
        if (self.kind == .kv_write or self.kind == .kv_write_ex) {
            if (self.use_multi_cache) {
                const ptrs = try a.alloc(u64, shape.rows);
                defer a.free(ptrs);
                for (0..shape.rows) |r| ptrs[r] = self.dev_out + r * self.ring * 576 * 2;
                self.dev[6] = try self.rig.input(std.mem.sliceAsBytes(ptrs));
            } else {
                self.dev[6] = 0;
            }
        }
        return true;
    }

    pub fn run(self: *NumCase) !void {
        switch (self.kind) {
            .comb_swiglu => try self.rows.combSwigluBf16(self.rig.s, self.dev[0], 40, 4, self.dev[1], self.dev[2], 48, 8, 24, self.dev_out, 16, 16, self.shape.rows, self.rank),
            .vit_qkv => {
                const plane = self.shape.n * self.shape.rows * self.shape.k * 2;
                try self.rows.vitQkv(self.rig.s, self.dev[0], self.dev[1], self.dev_out, self.dev_out + plane, self.dev_out + 2 * plane, self.shape.rows, self.shape.n, self.shape.k);
            },
            .q_rope => try self.rows.qRope(self.rig.s, self.dev[0], 4 * 72 + 8, 72, self.dev_out, 4 * 64, 64, self.dev[2], self.dev[1], self.shape.rows, 4),
            .q_rope_mrope => try self.rows.qRopeMrope(self.rig.s, self.dev[0], 4 * 72 + 8, 72, self.dev_out, 4 * 64, 64, self.dev[2], self.dev[3], self.shape.rows, self.dev[1], self.shape.rows, 4),
            .kv_write => {
                const out_view: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.dev_out, .len = self.out_elems * 2 };
                try out_view.fill8(0, self.rig.s.handle);
                try self.rows.kvWrite(self.rig.s, self.dev[0], self.qa_stride, self.q_lora, self.dev[1], self.dev[2], self.dev[3], self.dev[6], self.dev_out, self.ring, self.shape.rows, self.eps);
            },
            .kv_write_ex => {
                const out_view: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.dev_out, .len = self.out_elems * 2 };
                try out_view.fill8(0, self.rig.s.handle);
                const pos3: u64 = if (self.shape.rows > 1) self.dev[5] else 0;
                try self.rows.kvWriteEx(self.rig.s, self.dev[0], self.qa_stride, self.q_lora, self.dev[1], self.dev[2], self.dev[3], self.dev[4], pos3, self.shape.rows, self.dev[6], self.dev_out, self.ring, self.shape.rows, self.eps);
            },
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *const NumCase) usize {
        return self.out_elems;
    }

    pub fn check(self: *NumCase, gate: g.Gate) !?[]const u8 {
        return exact.checkNum(self, gate);
    }

    pub fn reference(self: *NumCase, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
        switch (self.kind) {
            .comb_swiglu => {
                for (0..self.shape.rows) |r| {
                    for (0..16) |col| {
                        const mine: f64 = self.h_f32_a.items[r * 40 + 4 + col];
                        const peer: f64 = self.h_f32_b.items[r * 16 + col];
                        want[r * 32 + col] = mine + peer;
                        mag[r * 32 + col] = @abs(mine) + @abs(peer);
                    }
                    for (0..16) |col| {
                        const gv: f64 = c.fromBf16(self.h_u16_a.items[r * 48 + 8 + col]);
                        const uv: f64 = c.fromBf16(self.h_u16_a.items[r * 48 + 24 + col]);
                        const v = gv / (1.0 + @exp(-gv)) * uv;
                        want[r * 32 + 16 + col] = v;
                        mag[r * 32 + 16 + col] = @abs(v);
                    }
                }
            },
            .vit_qkv => {
                const heads = self.shape.n;
                const hd = self.shape.k;
                const d = heads * hd;
                const plane = heads * self.shape.rows * hd;
                for (0..self.shape.rows) |r| for (0..heads) |h| for (0..hd) |j| {
                    const c_idx = h * hd + j;
                    const o = (h * self.shape.rows + r) * hd + j;
                    inline for (0..3) |p| {
                        const av: f64 = self.h_f32_a.items[r * 3 * d + p * d + c_idx];
                        const bv: f64 = c.fromBf16(self.h_u16_a.items[p * d + c_idx]);
                        want[p * plane + o] = av + bv;
                        mag[p * plane + o] = @abs(av) + @abs(bv);
                    }
                };
            },
            .q_rope, .q_rope_mrope => {
                const heads = 4;
                const xh = 72;
                const xs = heads * xh + 8;
                for (0..self.shape.rows) |r| for (0..heads) |h| for (0..32) |i| {
                    const rp: i32 = if (self.kind == .q_rope_mrope)
                        self.h_i32_b.items[mropeAxis(i) * self.shape.rows + r]
                    else
                        self.h_i32_a.items[r];
                    const ang = @as(f64, self.h_f32_a.items[i]) * @as(f64, @floatFromInt(rp));
                    const sv = @sin(ang);
                    const cv = @cos(ang);
                    const base = r * xs + h * xh + 2 * i;
                    const x0: f64 = c.fromBf16(self.h_u16_a.items[base]);
                    const x1: f64 = c.fromBf16(self.h_u16_a.items[base + 1]);
                    const o = (r * heads + h) * 64 + 2 * i;
                    want[o] = x0 * cv - x1 * sv;
                    mag[o] = @abs(x0 * cv) + @abs(x1 * sv);
                    want[o + 1] = x1 * cv + x0 * sv;
                    mag[o + 1] = @abs(x1 * cv) + @abs(x0 * sv);
                };
            },
            .kv_write, .kv_write_ex => {
                for (0..self.shape.rows) |r| {
                    const src = self.h_u16_a.items[r * self.qa_stride + self.q_lora ..];
                    var ss: f64 = 0;
                    for (0..512) |i| {
                        const v: f64 = c.fromBf16(src[i]);
                        ss += v * v;
                    }
                    const inv = 1.0 / @sqrt(ss / 512.0 + @as(f64, self.eps));
                    const sp: usize = @intCast(self.h_i32_a.items[r]);
                    const slot = sp % self.ring;
                    const base = (if (self.use_multi_cache) r * self.ring else 0) * 576 + slot * 576;
                    for (0..512) |i| {
                        const v = @as(f64, c.fromBf16(src[i])) * inv * @as(f64, c.fromBf16(self.h_u16_b.items[i]));
                        want[base + i] = v;
                        mag[base + i] = @abs(v);
                    }
                    for (0..32) |i| {
                        const rp: i32 = if (self.kind == .kv_write_ex and self.shape.rows > 1)
                            self.h_i32_c.items[mropeAxis(i) * self.shape.rows + r]
                        else if (self.kind == .kv_write_ex)
                            self.h_i32_b.items[r]
                        else
                            self.h_i32_a.items[r];
                        const ang = @as(f64, self.h_f32_a.items[i]) * @as(f64, @floatFromInt(rp));
                        const sv = @sin(ang);
                        const cv = @cos(ang);
                        const x0: f64 = c.fromBf16(src[512 + 2 * i]);
                        const x1: f64 = c.fromBf16(src[512 + 2 * i + 1]);
                        want[base + 512 + 2 * i] = x0 * cv - x1 * sv;
                        mag[base + 512 + 2 * i] = @abs(x0 * cv) + @abs(x1 * sv);
                        want[base + 512 + 2 * i + 1] = x1 * cv + x0 * sv;
                        mag[base + 512 + 2 * i + 1] = @abs(x1 * cv) + @abs(x0 * sv);
                    }
                }
            },
        }
    }
};

// Paged BF16 KV write (`hy4_rows_kv_write_paged`) vs contiguous `kv_write_ex` plus wrong-page control.
const kv_paged_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 576, .k = 512 },
    .{ .rows = 5, .n = 576, .k = 512 },
    .{ .rows = 9, .n = 576, .k = 512 },
};

pub const KvWritePagedCase = struct {
    rig: rig_.Rig,
    rows: *const hr.Rows,
    wrong_page: bool = false,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    qa_stride: usize = 640,
    q_lora: usize = 32,
    pt_stride: usize = 4,
    phys_pages: usize = 0,
    eps: f32 = 1e-6,
    dev: [9]u64 = @splat(0),
    dev_out: u64 = 0,
    pt_true: []u32 = &.{},
    slot_pos: []i32 = &.{},
    row_stream: []i32 = &.{},
    want_bytes: std.ArrayList(u8) = .empty,

    fn freeHost(self: *KvWritePagedCase) void {
        const a = self.rig.gpu.gpa;
        if (self.pt_true.len > 0) a.free(self.pt_true);
        if (self.slot_pos.len > 0) a.free(self.slot_pos);
        if (self.row_stream.len > 0) a.free(self.row_stream);
        self.pt_true = &.{};
        self.slot_pos = &.{};
        self.row_stream = &.{};
    }

    pub fn deinit(self: *KvWritePagedCase) void {
        self.freeHost();
        self.want_bytes.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *KvWritePagedCase) g.Kernel {
        return .{
            .spec = .{
                .name = if (self.wrong_page) "control: hy4_rows hy4_rows_kv_write_paged wrong page table" else "cuda hy4_rows hy4_rows_kv_write_paged",
                .accum = .none,
                .depth = zeroDepth,
                .out = .raw,
                .nan = .{ .stated = "exact byte match to contiguous hy4_rows_kv_write_ex scattered into paged arena" },
                .served = &kv_paged_shapes,
            },
            .ctx = self,
            .vt = &vt,
            .control = if (self.wrong_page) .shapes else null,
        };
    }

    const vt = g.hooks(KvWritePagedCase);

    pub fn prepare(self: *KvWritePagedCase, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != 576 or shape.k != 512) return false;
        self.freeHost();
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x4b56);
        const rng = prng.random();
        const tbl = try buildPageTable(a, 2, 3, self.pt_stride);
        self.pt_true = tbl.pt;
        self.phys_pages = tbl.phys_pages;
        const pt_dev = try a.dupe(u32, self.pt_true);
        defer a.free(pt_dev);
        if (self.wrong_page) std.mem.swap(u32, &pt_dev[0], &pt_dev[self.pt_stride]);

        const boundary_pos = [_]i32{ 0, 255, 256, 257, 511, 512, 513, 17, 300 };
        self.slot_pos = try a.alloc(i32, shape.rows);
        self.row_stream = try a.alloc(i32, shape.rows);
        const qa = try a.alloc(u16, shape.rows * self.qa_stride);
        defer a.free(qa);
        const w = try a.alloc(u16, 512);
        defer a.free(w);
        const freqs = try a.alloc(f32, 32);
        defer a.free(freqs);
        const rot_pos = try a.alloc(i32, shape.rows);
        defer a.free(rot_pos);
        const pos3 = try a.alloc(i32, 3 * shape.rows);
        defer a.free(pos3);
        const lin_pos = try a.alloc(i32, shape.rows);
        defer a.free(lin_pos);
        for (qa) |*v| v.* = randBf16(rng);
        for (w) |*v| v.* = randBf16(rng);
        for (0..32) |i| freqs[i] = @floatCast(std.math.pow(f64, 10000.0, -2.0 * @as(f64, @floatFromInt(i)) / 64.0));
        for (0..shape.rows) |r| {
            self.slot_pos[r] = boundary_pos[r % boundary_pos.len];
            self.row_stream[r] = @intCast(r & 1);
            rot_pos[r] = @intCast(r * 7 + 1);
            pos3[0 * shape.rows + r] = @intCast(r * 5 + 3);
            pos3[1 * shape.rows + r] = @intCast(r * 9 + 1);
            pos3[2 * shape.rows + r] = @intCast(r * 13 + 2);
            lin_pos[r] = @intCast(r);
        }
        if (fill == .nonfinite) {
            qa[self.q_lora + 7] = 0x7fc0;
            qa[self.q_lora + 512] = 0x7f80;
        }
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(qa));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(w));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(freqs));
        self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(self.slot_pos));
        self.dev[4] = try self.rig.input(std.mem.sliceAsBytes(rot_pos));
        self.dev[5] = try self.rig.input(std.mem.sliceAsBytes(pos3));
        self.dev[6] = try self.rig.input(std.mem.sliceAsBytes(pt_dev));
        self.dev[7] = try self.rig.input(std.mem.sliceAsBytes(self.row_stream));
        self.dev[8] = try self.rig.input(std.mem.sliceAsBytes(lin_pos));
        self.dev_out = try self.rig.output(self.count());
        return true;
    }

    pub fn run(self: *KvWritePagedCase) !void {
        const out_view: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.dev_out, .len = self.count() };
        try out_view.fill8(0xff, self.rig.s.handle);
        try self.rows.kvWritePaged(self.rig.s, self.dev[0], self.qa_stride, self.q_lora, self.dev[1], self.dev[2], self.dev[3], self.dev[4], self.dev[5], self.shape.rows, self.dev[6], self.dev[7], self.pt_stride, self.dev_out, self.shape.rows, self.eps);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *const KvWritePagedCase) usize {
        return self.phys_pages * 256 * 576 * 2;
    }
    pub fn reference(_: *KvWritePagedCase, _: []f64, _: []f64) !void {}

    pub fn replaced(self: *KvWritePagedCase) ![]const u8 {
        const a = self.rig.gpu.gpa;
        const row_bytes = 576 * 2;
        var tmp = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, self.shape.rows * row_bytes);
        defer tmp.free();
        try self.rows.kvWriteEx(self.rig.s, self.dev[0], self.qa_stride, self.q_lora, self.dev[1], self.dev[2], self.dev[8], self.dev[4], self.dev[5], self.shape.rows, 0, tmp.ptr, 0, self.shape.rows, self.eps);
        try self.rig.s.synchronize();
        const lin = try a.alloc(u8, self.shape.rows * row_bytes);
        defer a.free(lin);
        try tmp.download(0, lin);
        try self.want_bytes.resize(a, self.count());
        @memset(self.want_bytes.items, 0xff);
        for (0..self.shape.rows) |r| {
            const sp: usize = @intCast(self.slot_pos[r]);
            const st: usize = @intCast(self.row_stream[r]);
            const pg = self.pt_true[st * self.pt_stride + (sp >> 8)];
            const dst_slot = pg * 256 + (sp & 255);
            @memcpy(self.want_bytes.items[dst_slot * row_bytes ..][0..row_bytes], lin[r * row_bytes ..][0..row_bytes]);
        }
        return self.want_bytes.items;
    }

    pub fn check(self: *KvWritePagedCase, gate: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (gate == .nonfinite and !hasMixedFinite(.bf16, want, want.len / 2)) return "kv_write_paged non-finite test lacked mixed finite/non-finite outputs";
        if (!std.mem.eql(u8, got, want)) return "hy4_rows_kv_write_paged output bytes differ from contiguous kv_write_ex";
        if (!try self.rig.intact()) return "hy4_rows_kv_write_paged overwrote guard region";
        if (!self.wrong_page and gate == .shapes) return exact.checkPagedSentinel(self);
        return null;
    }
};
