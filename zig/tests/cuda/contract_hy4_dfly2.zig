//! Contract cases for all 6 live `hy4_dfly2_glue` kernels, checking every written output buffer.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hr = @import("hy4_route_rows_cuda");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const dfly = hr.dfly2;

const H: usize = dfly.hidden; // 3072

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

fn normDepth(_: c.Shape) usize {
    return H;
}

fn randBf16(rng: std.Random) u16 {
    return c.toBf16(rng.float(f32) * 2.0 - 1.0);
}

fn rbf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

fn rmsNormRow(src: []const f64, w: []const u16, eps: f64, want: []f64, mag: []f64) void {
    var ss: f64 = 0;
    for (src) |v| ss += v * v;
    const ri = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(H)) + eps);
    for (0..H) |i| {
        const y = @as(f64, c.fromBf16(w[i])) * rbf(src[i] * ri);
        want[i] = y;
        mag[i] = @abs(y);
    }
}

// Float64-verified DFly2 row-norm kernels: dual_rmsnorm_cat_s1, fusion_gated_add_s1, sublayer_boundary_s1.
pub const NormKind = enum { dual, fusion, boundary };

const dual_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 2 * H, .k = H },
    .{ .rows = 4, .n = 2 * H, .k = H },
    .{ .rows = 7, .n = 2 * H, .k = H },
};
const fusion_shapes = [_]c.Shape{
    .{ .rows = 6, .n = 2 * H, .k = H },
    .{ .rows = 12, .n = 2 * H, .k = H },
};
const boundary_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 2 * H, .k = H },
    .{ .rows = 4, .n = 2 * H, .k = H },
    .{ .rows = 7, .n = 2 * H, .k = H },
};

pub const NormCase = struct {
    rig: rig_.Rig,
    k: *const dfly.Kernels,
    kind: NormKind,
    shape: c.Shape = undefined,
    eps: f32 = 1e-6,
    dev: [8]u64 = @splat(0),
    dev_out: u64 = 0,
    nrows_b: usize = 3,
    use_sw: bool = true,
    // Host copies for reference().
    h_u16: [5]std.ArrayList(u16) = @splat(.empty),
    h_f32: std.ArrayList(f32) = .empty,
    h_ids: std.ArrayList(i32) = .empty,

    pub fn deinit(self: *NormCase) void {
        const a = self.rig.gpu.gpa;
        for (&self.h_u16) |*l| l.deinit(a);
        self.h_f32.deinit(a);
        self.h_ids.deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *NormCase) g.Kernel {
        const name: []const u8 = switch (self.kind) {
            .dual => "cuda hy4_dfly2_glue hy4_dfly_dual_rmsnorm_cat_s1",
            .fusion => "cuda hy4_dfly2_glue hy4_dfly_fusion_gated_add_s1",
            .boundary => "cuda hy4_dfly2_glue hy4_dfly_sublayer_boundary_s1",
        };
        const served: []const c.Shape = switch (self.kind) {
            .dual => &dual_shapes,
            .fusion => &fusion_shapes,
            .boundary => &boundary_shapes,
        };
        return .{
            .spec = .{
                .name = name,
                .accum = .none,
                .depth = zeroDepth,
                .ops = 24,
                .out = .bf16,
                .served = served,
            },
            .ctx = self,
            .vt = &vt,
        };
    }

    pub fn bounds(self: *const NormCase, want: []const f64, mag: []const f64, allow: []f64) !void {
        const spec: c.Spec = .{ .name = "", .accum = .f32, .depth = normDepth, .ops = 24, .out = .bf16, .served = &dual_shapes };
        for (allow, want, mag) |*l, w, m| l.* = c.bound(spec, self.shape, w, m);
    }

    const vt = g.hooks(NormCase);

    pub fn prepare(self: *NormCase, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != 2 * H or shape.k != H) return false;
        if (self.kind == .fusion and (shape.rows % dfly.qw) != 0) return false;
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed ^ (@as(u64, @backingInt(self.kind)) * 0x51a7));
        const rng = prng.random();
        switch (self.kind) {
            .dual => {
                try self.h_u16[0].resize(a, shape.rows * H);
                try self.h_u16[1].resize(a, self.nrows_b * H);
                try self.h_u16[2].resize(a, H);
                try self.h_u16[3].resize(a, H);
                try self.h_ids.resize(a, shape.rows);
                for (self.h_u16[0].items) |*v| v.* = randBf16(rng);
                for (self.h_u16[1].items) |*v| v.* = randBf16(rng);
                for (self.h_u16[2].items) |*v| v.* = randBf16(rng);
                for (self.h_u16[3].items) |*v| v.* = randBf16(rng);
                for (0..shape.rows) |r| {
                    self.h_ids.items[r] = if (r == 1) -1 else if (r == 2) @intCast(self.nrows_b) else @intCast(r % self.nrows_b);
                }
                if (fill == .nonfinite) self.h_u16[0].items[3] = 0x7fc0;
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16[0].items));
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16[1].items));
                self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16[2].items));
                self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16[3].items));
                self.dev[4] = try self.rig.input(std.mem.sliceAsBytes(self.h_ids.items));
            },
            .fusion => {
                const n_anc = shape.rows / dfly.qw;
                try self.h_u16[0].resize(a, shape.rows * H);
                try self.h_f32.resize(a, 2 * n_anc * H);
                try self.h_u16[1].resize(a, H);
                try self.h_u16[2].resize(a, H);
                for (self.h_u16[0].items) |*v| v.* = randBf16(rng);
                for (self.h_f32.items) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
                for (self.h_u16[1].items) |*v| v.* = randBf16(rng);
                for (self.h_u16[2].items) |*v| v.* = randBf16(rng);
                if (fill == .nonfinite) self.h_f32.items[5] = std.math.nan(f32);
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16[0].items));
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.h_f32.items));
                self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16[1].items));
                self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16[2].items));
            },
            .boundary => {
                self.use_sw = shape.rows > 1;
                try self.h_f32.resize(a, 2 * shape.rows * H);
                try self.h_u16[0].resize(a, shape.rows * H);
                try self.h_u16[1].resize(a, H);
                try self.h_u16[2].resize(a, H);
                for (self.h_f32.items) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
                for (self.h_u16[0].items) |*v| v.* = randBf16(rng);
                for (self.h_u16[1].items) |*v| v.* = randBf16(rng);
                for (self.h_u16[2].items) |*v| v.* = randBf16(rng);
                if (fill == .nonfinite) self.h_f32.items[7] = std.math.nan(f32);
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.h_f32.items));
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16[0].items));
                self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16[1].items));
                self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(self.h_u16[2].items));
            },
        }
        self.dev_out = try self.rig.output(self.count() * 2);
        return true;
    }

    pub fn run(self: *NormCase) !void {
        const half_bytes = self.shape.rows * H * 2;
        switch (self.kind) {
            .dual => {
                const a_rows: dfly.DflyRows = .{ .base = self.dev[0], .ld = H };
                const b_rows: dfly.DflyRows = .{ .base = self.dev[1], .ids = self.dev[4], .ld = H, .ids_stride = 1, .nrows = @intCast(self.nrows_b) };
                try self.k.runDual(self.rig.s, a_rows, self.dev[2], b_rows, self.dev[3], self.eps, self.dev_out, 2 * H, self.shape.rows);
            },
            .fusion => {
                const n_anc = self.shape.rows / dfly.qw;
                const e_rows: dfly.DflyRows = .{ .base = self.dev[0], .ld = H };
                var args: cuda.Args = .{};
                args.add(e_rows);
                args.add(self.dev[1]);
                args.add(@as(i64, H));
                args.add(@as(i64, @intCast(n_anc * H)));
                args.add(@as(i32, 2));
                args.add(self.dev[2]);
                args.add(self.dev[3]);
                args.add(self.eps);
                args.add(@as(i32, dfly.qw));
                args.add(self.dev_out);
                args.add(@as(i64, H));
                args.add(self.dev_out + half_bytes);
                args.add(@as(i64, H));
                try cuda.launch.launch(self.k.fusion, .{ .grid = .{ .x = 1, .y = @intCast(self.shape.rows), .z = 1 }, .block = .{ .x = 384 } }, self.rig.s, &args);
            },
            .boundary => {
                const sw: u64 = if (self.use_sw) self.dev[2] else 0;
                try self.k.runBoundary(self.rig.s, self.dev[0], H, self.shape.rows * H, 2, sw, self.dev[1], self.dev[3], self.dev_out, self.dev_out + half_bytes, self.eps, self.shape.rows);
            },
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *const NormCase) usize {
        return self.shape.rows * 2 * H;
    }

    pub fn reference(self: *NormCase, want: []f64, mag: []f64) !void {
        var row_buf: [H]f64 = undefined;
        var tmp_w: [H]f64 = undefined;
        var tmp_m: [H]f64 = undefined;
        const half = self.shape.rows * H;
        switch (self.kind) {
            .dual => {
                for (0..self.shape.rows) |r| {
                    for (0..H) |i| row_buf[i] = c.fromBf16(self.h_u16[0].items[r * H + i]);
                    rmsNormRow(&row_buf, self.h_u16[2].items, self.eps, want[r * 2 * H ..][0..H], mag[r * 2 * H ..][0..H]);
                    const id = self.h_ids.items[r];
                    if (id < 0 or id >= @as(i32, @intCast(self.nrows_b))) {
                        @memset(&row_buf, 0);
                    } else {
                        const bi: usize = @intCast(id);
                        for (0..H) |i| row_buf[i] = c.fromBf16(self.h_u16[1].items[bi * H + i]);
                    }
                    rmsNormRow(&row_buf, self.h_u16[3].items, self.eps, want[r * 2 * H + H ..][0..H], mag[r * 2 * H + H ..][0..H]);
                }
            },
            .fusion => {
                const n_anc = self.shape.rows / dfly.qw;
                for (0..self.shape.rows) |r| {
                    for (0..H) |i| {
                        var h: f64 = c.fromBf16(self.h_u16[0].items[r * H + i]);
                        if (r % dfly.qw == 0) {
                            const anc = r / dfly.qw;
                            const s0: f64 = self.h_f32.items[anc * H + i];
                            const s1: f64 = self.h_f32.items[(n_anc + anc) * H + i];
                            const f_sum = rbf(s0 + s1);
                            const gv: f64 = c.fromBf16(self.h_u16[1].items[i]);
                            h = rbf(h + rbf(gv * f_sum));
                        }
                        row_buf[i] = h;
                        want[r * H + i] = h;
                        mag[r * H + i] = @abs(h);
                    }
                    rmsNormRow(&row_buf, self.h_u16[2].items, self.eps, want[half + r * H ..][0..H], mag[half + r * H ..][0..H]);
                }
            },
            .boundary => {
                for (0..self.shape.rows) |r| {
                    for (0..H) |i| {
                        const s0: f64 = self.h_f32.items[r * H + i];
                        const s1: f64 = self.h_f32.items[(self.shape.rows + r) * H + i];
                        row_buf[i] = rbf(s0 + s1);
                    }
                    if (self.use_sw) {
                        rmsNormRow(&row_buf, self.h_u16[1].items, self.eps, &tmp_w, &tmp_m);
                        for (0..H) |i| row_buf[i] = rbf(tmp_w[i]);
                    }
                    for (0..H) |i| {
                        const rv: f64 = c.fromBf16(self.h_u16[0].items[r * H + i]);
                        const h = rbf(rv + row_buf[i]);
                        row_buf[i] = h;
                        want[r * H + i] = h;
                        mag[r * H + i] = @abs(h);
                    }
                    rmsNormRow(&row_buf, self.h_u16[2].items, self.eps, want[half + r * H ..][0..H], mag[half + r * H ..][0..H]);
                }
            },
        }
    }
};

// Exact byte-verified DFly2 index/argmax/gather kernels: extend_topk_k, pick_argmax, gather_hidden.
pub const ExactKind = enum { extend_topk, pick_argmax, gather_hidden };

const topk_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 32, .k = 16 }, // mode=TOK + paged block_table
    .{ .rows = 2, .n = 32, .k = 16 }, // mode=LIST (dense_below + sparse) + paged block_table
    .{ .rows = 3, .n = 32, .k = 16 }, // mode=LIST + contiguous req_stream
};
const pick_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 4, .k = 5 },
    .{ .rows = 7, .n = 4, .k = 5 },
    .{ .rows = 65, .n = 4, .k = 5 },
};
const gather_shapes = [_]c.Shape{
    .{ .rows = 1, .n = H, .k = H },
    .{ .rows = 4, .n = H, .k = H },
    .{ .rows = 9, .n = H, .k = H },
};

pub const ExactCase = struct {
    rig: rig_.Rig,
    k: *const dfly.Kernels,
    kind: ExactKind,
    shape: c.Shape = undefined,
    ta: dfly.TopkArgs = .{},
    dev: [8]u64 = @splat(0),
    dev_out: u64 = 0,
    out_bytes: usize = 0,
    want: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *ExactCase) void {
        self.want.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *ExactCase) g.Kernel {
        const name: []const u8 = switch (self.kind) {
            .extend_topk => "cuda hy4_dfly2_glue hy4_dfly_extend_topk_k",
            .pick_argmax => "cuda hy4_dfly2_glue hy4_dfly_pick_argmax",
            .gather_hidden => "cuda hy4_dfly2_glue hy4_dfly_gather_hidden",
        };
        const served: []const c.Shape = switch (self.kind) {
            .extend_topk => &topk_shapes,
            .pick_argmax => &pick_shapes,
            .gather_hidden => &gather_shapes,
        };
        return .{
            .spec = .{
                .name = name,
                .accum = .none,
                .depth = zeroDepth,
                .out = .raw,
                .replaces = "host exact reference",
                .nan = .{ .stated = "exact byte match across all written outputs; non-finite argmax emits 0xffffffff" },
                .served = served,
            },
            .ctx = self,
            .vt = &vt,
        };
    }

    const vt = g.hooks(ExactCase);

    pub fn prepare(self: *ExactCase, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed ^ (@as(u64, @backingInt(self.kind)) * 0x7711));
        const rng = prng.random();
        switch (self.kind) {
            .extend_topk => {
                const b: usize = 3;
                const num_layers: usize = 2;
                const qw: usize = 6;
                const tt: usize = 16;
                const lw: usize = 8;
                const k_pad: usize = 32;
                const cap: usize = 4;
                const mode: i32 = if (shape.rows == 1) 0 else 1;
                const paged = shape.rows != 3;
                const dense_below: i32 = 30;
                const src_row_stride: usize = if (mode == 0) tt else 8;
                const src_slot_stride: usize = cap * src_row_stride;
                const src = try a.alloc(i32, num_layers * src_slot_stride);
                defer a.free(src);
                for (src, 0..) |*v, i| v.* = if (mode == 0) @intCast((i * 5 + 1) % 45) else @intCast((i * 3) % 12);
                const layer_slot = [_]i32{ 1, 0 };
                const active = [_]i32{ 1, 0, 1 };
                const source_rows = [_]i32{ 1, 0, 2 };
                const anchors = [_]i32{ 20, 40, 50 };
                const bt_stride: usize = 4;
                const block_size: i32 = 16;
                const blocks_per_req: i32 = 4;
                const max_model_len: i32 = 64;
                const block_table = [_]i32{ 3, 7, 2, 0, 1, 2, 3, 4, 5, 6, 8, 9 };
                const req_stream = [_]i32{ 2, 0, 1 };
                const ctx: i64 = 64;
                const pad_slot: i64 = -1;

                const out_layer_stride = b * qw * k_pad;
                const out_bytes = num_layers * out_layer_stride * 4;
                const slot_bytes = b * qw * 8;
                const pos_bytes = b * qw * 4;
                self.out_bytes = out_bytes + slot_bytes + pos_bytes;
                try self.want.resize(a, self.out_bytes);
                @memset(self.want.items, 0);
                const w_out = std.mem.bytesAsSlice(i32, self.want.items[0..out_bytes]);
                const w_slot = std.mem.bytesAsSlice(i64, self.want.items[out_bytes..][0..slot_bytes]);
                const w_pos = std.mem.bytesAsSlice(i32, self.want.items[out_bytes + slot_bytes ..][0..pos_bytes]);

                for (0..b) |bi| for (0..qw) |q| {
                    const row = bi * qw + q;
                    const act = active[bi] != 0;
                    const pos: i64 = @as(i64, anchors[bi]) + @as(i64, @intCast(q));
                    var slot: i64 = pad_slot;
                    if (act and pos >= 0 and pos < max_model_len) {
                        if (paged) {
                            const blk = @divTrunc(pos, block_size);
                            if (blk < blocks_per_req) {
                                const id = block_table[bi * bt_stride + @as(usize, @intCast(blk))];
                                if (id > 0) slot = @as(i64, id) * block_size + @rem(pos, block_size);
                            }
                        } else if (pos < ctx) {
                            const st: i64 = req_stream[bi];
                            if (st >= 0) slot = st * ctx + pos;
                        }
                    }
                    w_slot[row] = slot;
                    w_pos[row] = if (act) @intCast(pos) else @intCast(q);
                };
                for (0..num_layers) |l| for (0..b) |bi| {
                    const act = active[bi] != 0;
                    const anc = anchors[bi];
                    const ws = anc - @as(i32, lw);
                    for (0..k_pad) |col| {
                        var hv: i32 = -1;
                        var cj: i32 = -1;
                        if (act) {
                            if (col < tt) {
                                const sr = source_rows[bi];
                                if (sr >= 0 and sr < cap) {
                                    const base = @as(usize, @intCast(layer_slot[l])) * src_slot_stride + @as(usize, @intCast(sr)) * src_row_stride;
                                    const h: i32 = if (mode == 0)
                                        src[base + col]
                                    else if (anc - 1 < dense_below)
                                        @intCast(col)
                                    else blk: {
                                        const g_idx = src[base + (col >> 2)];
                                        break :blk if (g_idx >= 0) 4 * g_idx + @as(i32, @intCast(col & 3)) else -1;
                                    };
                                    hv = if (h >= 0 and h < ws) h else -1;
                                }
                            } else {
                                const cc = col - tt;
                                if (cc < lw) {
                                    hv = @max(ws + @as(i32, @intCast(cc)), -1);
                                } else if (cc - lw < qw) {
                                    cj = @intCast(cc - lw);
                                }
                            }
                        }
                        for (0..qw) |q| {
                            const qi: i32 = @intCast(q);
                            w_out[l * out_layer_stride + (bi * qw + q) * k_pad + col] = if (cj >= 0) (if (cj <= qi) anc + cj else -1) else hv;
                        }
                    }
                };
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(src));
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(&layer_slot));
                self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(&active));
                self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(&source_rows));
                self.dev[4] = try self.rig.input(std.mem.sliceAsBytes(&anchors));
                self.dev[5] = try self.rig.input(std.mem.sliceAsBytes(&block_table));
                self.dev[6] = try self.rig.input(std.mem.sliceAsBytes(&req_stream));
                self.dev_out = try self.rig.output(self.out_bytes);
                self.ta = .{
                    .mode = mode,
                    .src = self.dev[0],
                    .src_slot_stride = @intCast(src_slot_stride),
                    .src_row_stride = @intCast(src_row_stride),
                    .src_rows_cap = cap,
                    .dense_below = dense_below,
                    .layer_slot = self.dev[1],
                    .num_layers = num_layers,
                    .active = self.dev[2],
                    .source_rows = self.dev[3],
                    .anchors = self.dev[4],
                    .b = b,
                    .qw = qw,
                    .target_topk = tt,
                    .local_window = lw,
                    .k_pad = k_pad,
                    .out = self.dev_out,
                    .out_layer_stride = @intCast(out_layer_stride),
                    .slot_out = self.dev_out + out_bytes,
                    .row_pos_out = self.dev_out + out_bytes + slot_bytes,
                    .block_table = if (paged) self.dev[5] else 0,
                    .bt_stride = bt_stride,
                    .blocks_per_req = blocks_per_req,
                    .block_size = block_size,
                    .max_model_len = max_model_len,
                    .req_stream = if (paged) 0 else self.dev[6],
                    .ctx = ctx,
                    .pad_slot = pad_slot,
                };
            },
            .pick_argmax => {
                const world = shape.n;
                const b = shape.rows;
                const stride = shape.k;
                const step: usize = 2;
                const in_u32 = try a.alloc(u32, world * b * 2);
                defer a.free(in_u32);
                for (0..world) |rk| for (0..b) |bi| {
                    var v = rng.float(f32) * 10.0 - 5.0;
                    var id: u32 = @intCast(rk * 1000 + bi * 7 + 3);
                    if (rk == 2 and (bi & 1) == 0) {
                        // Exact tie with rank 0, lower id wins.
                        v = @bitCast(in_u32[(0 * b + bi) * 2]);
                        id = in_u32[(0 * b + bi) * 2 + 1] - 1;
                    }
                    in_u32[(rk * b + bi) * 2] = @bitCast(v);
                    in_u32[(rk * b + bi) * 2 + 1] = id;
                };
                if (fill == .nonfinite) in_u32[(1 * b + 0) * 2] = @bitCast(std.math.nan(f32));
                const prev_bytes = b * 4;
                const draft_bytes = b * stride * 4;
                self.out_bytes = prev_bytes + draft_bytes;
                try self.want.resize(a, self.out_bytes);
                @memset(self.want.items, 0xaa);
                const w_prev = std.mem.bytesAsSlice(u32, self.want.items[0..prev_bytes]);
                const w_draft = std.mem.bytesAsSlice(u32, self.want.items[prev_bytes..][0..draft_bytes]);
                for (0..b) |bi| {
                    var best_v: f32 = -std.math.inf(f32);
                    var best_id: u32 = 0xffffffff;
                    for (0..world) |rk| {
                        const v: f32 = @bitCast(in_u32[(rk * b + bi) * 2]);
                        const id = in_u32[(rk * b + bi) * 2 + 1];
                        if (!std.math.isFinite(v)) {
                            best_id = 0xffffffff;
                            break;
                        }
                        if (v > best_v or (v == best_v and id < best_id)) {
                            best_v = v;
                            best_id = id;
                        }
                    }
                    w_prev[bi] = best_id;
                    w_draft[bi * stride + step] = best_id;
                }
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(in_u32));
                self.dev_out = try self.rig.output(self.out_bytes);
            },
            .gather_hidden => {
                const pool_rows = shape.rows + 5;
                const pool = try a.alloc(u16, pool_rows * H);
                defer a.free(pool);
                for (pool) |*v| v.* = randBf16(rng);
                if (fill == .nonfinite) pool[H + 3] = 0x7fc0;
                self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(pool));
                const ptrs = try a.alloc(u64, shape.rows);
                defer a.free(ptrs);
                self.out_bytes = shape.rows * H * 2;
                try self.want.resize(a, self.out_bytes);
                const w16 = std.mem.bytesAsSlice(u16, self.want.items);
                for (0..shape.rows) |b| {
                    const src_row = (b * 3 + 1) % pool_rows;
                    ptrs[b] = self.dev[0] + src_row * H * 2;
                    @memcpy(w16[b * H ..][0..H], pool[src_row * H ..][0..H]);
                }
                self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(ptrs));
                self.dev_out = try self.rig.output(self.out_bytes);
            },
        }
        return true;
    }

    pub fn run(self: *ExactCase) !void {
        switch (self.kind) {
            .extend_topk => try self.k.runExtendTopk(self.rig.s, self.ta),
            .pick_argmax => {
                const out_view: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.dev_out, .len = self.out_bytes };
                try out_view.fill8(0xaa, self.rig.s.handle);
                try self.k.runPickArgmax(self.rig.s, self.dev[0], self.shape.n, self.shape.rows, 2, self.shape.k, self.dev_out, self.dev_out + self.shape.rows * 4);
            },
            .gather_hidden => try self.k.runGatherHidden(self.rig.s, self.dev[1], self.dev_out, self.shape.rows),
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *const ExactCase) usize {
        return self.out_bytes;
    }
    pub fn reference(_: *ExactCase, _: []f64, _: []f64) !void {}
    pub fn replaced(self: *ExactCase) ![]const u8 {
        return self.want.items;
    }
    pub fn check(self: *ExactCase, _: g.Gate) !?[]const u8 {
        const got = try self.rig.outputBytes();
        if (!std.mem.eql(u8, got, self.want.items)) return "hy4_dfly2_glue output bytes differ from exact reference";
        if (!try self.rig.intact()) return "hy4_dfly2_glue overwrote guard region";
        return null;
    }
};
