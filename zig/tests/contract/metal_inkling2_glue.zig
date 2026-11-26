//! Inkling 2 Mini glue under FP64 references, guarded buffers and real Metal dispatch.
const std = @import("std");
const mtl = @import("metal");
const controls = @import("metal_inkling2_glue_controls.zig");
const kv_conv = @import("metal_inkling2_glue_kv_conv.zig");
const split_controls = @import("metal_inkling2_split_controls.zig");
const fused = @import("metal_inkling2_glue_fused.zig");
const norm_fused = @import("metal_inkling2_glue_norm.zig");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.glue;
const p = host.plan;
const sources = @import("inkling2_metal");

const depth = fused.depth;
const value = fused.value;
const round = fused.round;
const at = fused.at;
fn fromBf(v: u16) f32 {
    return c.fromBf16(v);
}

const segments_label = "metal inkling2 glue conv_dynamic_segs tf_inkling2_conv_dynamic_segs";
const split_label = "metal inkling2 glue router split tf_inkling2_route_scores and tf_inkling2_route_select";

/// Each dot opens with +B x0 - B x0 = 0 (B = 2^18, exact): any other summation order loses the low terms' bits.
fn orderProbe(x: []f32, weight: []f32) void {
    for (0..x.len / p.hidden) |row| x[row * p.hidden + 1] = x[row * p.hidden];
    for (0..p.expert_count) |e| {
        weight[e * p.hidden] = 0x1p18;
        weight[e * p.hidden + 1] = -0x1p18;
    }
}

// The attention-residual probe runs a batch of rows, each with its own blocks, partial sums and two output queries.
const ar_batch = 2;
const ar_blocks = 4 * p.hidden;
fn arBlocks(f: []f32, row: usize) []f32 {
    return f[row * ar_blocks ..][0..ar_blocks];
}
fn arPartial(f: []f32, row: usize) []f32 {
    return f[row * p.hidden ..][0..p.hidden];
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: p.Kind,
    pipes: host.Pipelines,
    base_pipes: ?host.Pipelines = null,
    shape: c.Shape = undefined,
    served: []const c.Shape = &.{},
    f: [8][]f32 = @splat(&.{}),
    ids: []u32 = &.{},
    route_ids: rig_.Ref = undefined,
    refs: [8]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    out_v: rig_.Ref = undefined,
    base_out: [2]rig_.Ref = undefined,
    base_bytes: std.ArrayList(u8) = .empty,
    nonfinite: bool = false,
    /// The split router against main's tf_inkling2_router: 1, 33 and 64 rows, two select blocks at 64.
    split: bool = false,
    segmented: bool = false,
    scores: rig_.Ref = undefined,
    nan_keys_run: bool = false,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: p.Kind) !Probe {
        return initSource(a, device, queue, kind, sources.glue);
    }
    pub fn initSource(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: p.Kind, text: []const u8) !Probe {
        var pipes = try host.Pipelines.loadSource(device, text);
        errdefer pipes.deinit();
        var self: Probe = .{
            .a = a,
            .arena = .init(a),
            .rig = .{ .a = a, .device = device, .queue = queue },
            .kind = kind,
            .pipes = pipes,
        };
        self.served = switch (kind) {
            .embed_norm => &.{ .{ .rows = 1, .n = p.hidden, .k = p.hidden }, .{ .rows = 5, .n = p.hidden, .k = p.hidden }, .{ .rows = 7, .n = p.hidden, .k = p.hidden } },
            .ar => &.{ .{ .rows = 2, .n = p.hidden, .k = p.hidden }, .{ .rows = 4, .n = p.hidden, .k = p.hidden }, .{ .rows = 5, .n = p.hidden, .k = p.hidden } },
            .conv_static => &.{ .{ .rows = 1, .n = 128, .k = 4 }, .{ .rows = 5, .n = 1024, .k = 4 }, .{ .rows = 7, .n = p.hidden, .k = 4 } },
            .conv_dynamic => &.{ .{ .rows = 1, .n = 128, .k = 4 }, .{ .rows = 5, .n = 512, .k = 4 }, .{ .rows = 7, .n = 1024, .k = 4 } },
            .kv_conv_static => &kv_conv.static_served,
            .kv_conv_dynamic => &kv_conv.dynamic_served,
            .router => &.{ .{ .rows = 1, .n = 8, .k = p.hidden }, .{ .rows = 5, .n = 8, .k = p.hidden }, .{ .rows = 7, .n = 8, .k = p.hidden } },
            .swiglu => &.{ .{ .rows = 1, .n = p.intermediate, .k = 2 }, .{ .rows = 5, .n = p.intermediate, .k = 2 }, .{ .rows = 7, .n = p.intermediate, .k = 2 } },
            .logits => &.{ .{ .rows = 1, .n = 128, .k = p.hidden }, .{ .rows = 5, .n = 256, .k = p.hidden }, .{ .rows = 7, .n = 128, .k = p.hidden } },
            .expert_pick => &.{ .{ .rows = 1, .n = p.intermediate, .k = 8 }, .{ .rows = 5, .n = p.intermediate, .k = 8 }, .{ .rows = 7, .n = p.intermediate, .k = 8 } },
            .expert_combine => &.{ .{ .rows = 1, .n = p.hidden, .k = 8 }, .{ .rows = 5, .n = p.hidden, .k = 8 }, .{ .rows = 7, .n = p.hidden, .k = 8 } },
            .argmax => &.{ .{ .rows = 1, .n = 128, .k = 1 }, .{ .rows = 5, .n = 4096, .k = 1 }, .{ .rows = 7, .n = 200058, .k = 1 } },
            .cached_ar, .block_inv_rms, .attn_partial_ar, .mlp_partial_ar, .norm_bf16, .attn_partial_ar_norm, .mlp_partial_ar_norm => unreachable,
        };
        return self;
    }
    pub fn initSplit(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, text: []const u8) !Probe {
        var self = try initSource(a, device, queue, .router, text);
        self.split = true;
        self.served = &.{ .{ .rows = 1, .n = 8, .k = p.hidden }, .{ .rows = 33, .n = 8, .k = p.hidden }, .{ .rows = 64, .n = 8, .k = p.hidden } };
        return self;
    }
    pub fn deinit(self: *Probe) void {
        self.base_bytes.deinit(self.a);
        if (self.base_pipes) |*bp| bp.deinit();
        self.rig.deinit();
        self.arena.deinit();
        self.pipes.deinit();
    }
    const vt_checked = g.hooks(Probe);
    const vt_plain = blk: {
        var vt = g.hooks(Probe);
        vt.check = null;
        break :blk vt;
    };
    pub fn kernel(self: *Probe) g.Kernel {
        const out: c.Elem = switch (self.kind) {
            .ar, .router, .logits => .f32,
            .argmax => .raw,
            else => .bf16,
        };
        const replaces: ?[]const u8 = if (self.split) "tf_inkling2_router" else switch (self.kind) {
            .kv_conv_static => "two tf_inkling2_conv_static dispatches (K and V)",
            .kv_conv_dynamic => "two tf_inkling2_conv_dynamic dispatches (K and V)",
            else => null,
        };
        const tiles: [3]c.Tile = switch (self.kind) {
            .kv_conv_static => .{ .{}, .{ .size = 32, .step = 1, .min = 1 }, .{} },
            .kv_conv_dynamic => .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            else => .{ .{}, .{}, .{} },
        };
        return .{
            .ctx = self,
            .vt = if (self.kind == .router or self.kind == .argmax) &vt_checked else &vt_plain,
            .spec = .{
                .name = if (self.segmented) segments_label else if (self.split) split_label else self.kind.label(),
                .replaces = if (self.segmented) "tf_inkling2_conv_dynamic" else replaces,
                .accum = if (kv_conv.isKvConv(self.kind) or self.kind == .argmax) .none else .f32,
                .depth = depth,
                .ops = 96,
                .mids = if (self.kind == .logits) 1 else 0,
                .out = out,
                .tiles = tiles,
                .served = self.served,
                .nan = if (self.kind == .router)
                    .{ .stated = "a NaN selected score makes the shared normalization and every selected weight NaN" }
                else if (self.kind == .argmax)
                    .{ .stated = "index 0 wins when NaN; later NaNs are skipped; ties pick the lower index" }
                else
                    .propagate,
                // AR's integer stress terms are exact in FP32 and must discriminate FP16/BF16 by 100x.
                .sums = if (self.kind == .ar) .{} else if (kv_conv.isKvConv(self.kind)) .{ .f16 = 0, .bf16 = 0, .why = "each paired side preserves the single-dispatch kernel's exact sums" } else .{ .f16 = 0, .bf16 = 0, .why = "nonlinear norms, exponentials and BF16 endpoints have no uniform lower-precision separability factor" },
            },
        };
    }
    pub fn count(self: *Probe) usize {
        if (self.kind == .ar) return ar_batch * 2 * p.hidden;
        return p.count(self.kind, .{ .rows = self.shape.rows, .n = self.shape.n, .k = self.shape.k });
    }
    fn floats(self: *Probe, n: usize, rng: std.Random) ![]f32 {
        return kv_conv.fillFloats(self.arena.allocator(), n, rng);
    }
    fn input(self: *Probe, values: []const f32, bf: bool) !rig_.Ref {
        if (!bf) return self.rig.input(std.mem.sliceAsBytes(values));
        return kv_conv.uploadBf16(self.arena.allocator(), &self.rig, values);
    }
    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!p.valid(self.kind, .{ .rows = shape.rows, .n = shape.n, .k = shape.k })) return false;
        if (self.kind == .embed_norm and shape.n != p.hidden) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.nonfinite = fill == .nonfinite;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const r = shape.rows;
        const n = shape.n;
        const nf = fill == .nonfinite;
        self.f = @splat(&.{});
        switch (self.kind) {
            .embed_norm => {
                self.f[0] = try self.floats(64 * p.hidden, rng);
                self.f[1] = try self.floats(p.hidden, rng);
                self.ids = try self.arena.allocator().alloc(u32, r);
                for (self.ids, 0..) |*id, i| id.* = @intCast(i % 64);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.rig.input(std.mem.sliceAsBytes(self.ids));
                self.refs[1] = try self.input(self.f[0], true);
                self.refs[2] = try self.input(self.f[1], true);
            },
            .ar => {
                self.f[0] = try self.floats(ar_batch * ar_blocks, rng);
                self.f[1] = try self.floats(ar_batch * p.hidden, rng);
                self.f[2] = try self.floats(2 * p.hidden, rng);
                if (fill == .stress) {
                    // Each 2304-term square sum is exactly 9 * 2^24 in FP32 under every order.
                    for (0..ar_batch) |b| for (0..r) |s| {
                        const values = if (s + 1 == r) arPartial(self.f[1], b) else self.f[0][b * ar_blocks + s * p.hidden ..][0..p.hidden];
                        @memset(values, if (s % 2 == 0) 256 else -256);
                    };
                    // Normalized mirrored queries leave +/-8; partial sums stay exact integers in FP32.
                    fused.fillStressQueries(self.f[2], 2, rng);
                }
                if (nf) self.f[2][0] = std.math.inf(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], false);
                self.refs[2] = try self.input(self.f[2], false);
            },
            .conv_static => {
                self.f[0] = try self.floats((r + 3) * n, rng);
                self.f[1] = try self.floats(4 * n, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
            },
            .conv_dynamic => {
                self.f[0] = try self.floats((r + 3) * n, rng);
                self.f[1] = try self.floats(r * p.hidden, rng);
                self.f[2] = try self.floats(4 * n, rng);
                self.f[3] = try self.floats(32 * p.hidden, rng);
                if (nf) self.f[1][0] = std.math.nan(f32);
                for (0..4) |j| self.refs[j] = try self.input(self.f[j], true);
            },
            .kv_conv_static, .kv_conv_dynamic => try kv_conv.prepare(self.arena.allocator(), &self.rig, self.kind, r, n, nf, rng, &self.f, &self.refs),
            .router => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                self.f[1] = try self.floats(p.expert_count * p.hidden, rng);
                self.f[2] = try self.floats(p.expert_count, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                if (self.split and fill == .random) orderProbe(self.f[0], self.f[1]);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
                self.refs[2] = try self.input(self.f[2], false);
                self.route_ids = try self.rig.scratch(r * 8 * 4);
                if (self.split) self.scores = try self.rig.scratch(r * p.expert_count * 4);
            },
            .swiglu => {
                self.f[0] = try self.floats(r * n * 2, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
            },
            .logits => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                self.f[1] = try self.floats(n * p.hidden, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
            },
            .expert_pick => {
                self.f[0] = try self.floats(r * p.expert_count * n, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.ids = try self.arena.allocator().alloc(u32, r * 8);
                for (self.ids, 0..) |*id, j| id.* = @intCast(j % 8 * 13 + j / 8);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.rig.input(std.mem.sliceAsBytes(self.ids));
            },
            .expert_combine => {
                self.f[0] = try self.floats(r * 8 * n, rng);
                self.f[1] = try self.floats(r * 8, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], false);
            },
            .argmax => {
                self.f[0] = try self.floats(r * n, rng);
                if (n >= 4) {
                    self.f[0][1] = 10.0;
                    self.f[0][3] = 10.0;
                }
                if (nf) {
                    self.f[0][n - 1] = std.math.nan(f32);
                    if (r > 1) self.f[0][n] = std.math.nan(f32);
                }
                self.refs[0] = try self.input(self.f[0], false);
            },
            .cached_ar, .block_inv_rms, .attn_partial_ar, .mlp_partial_ar, .norm_bf16, .attn_partial_ar_norm, .mlp_partial_ar_norm => unreachable,
        }
        // Every BF16 input and the FP64 oracle see exactly the same stored values.
        for (self.f, 0..) |values, j| if (values.len > 0 and switch (self.kind) {
            .ar => j == 0,
            .router => j < 2,
            .expert_combine => j == 0,
            .argmax => false,
            else => true,
        }) round(values);
        const bytes = self.count() * @as(usize, if (self.kind == .ar or self.kind == .router or self.kind == .logits or self.kind == .argmax) 4 else 2);
        if (kv_conv.isKvConv(self.kind)) {
            const half_bytes = bytes / 2;
            self.out = try self.rig.output(half_bytes);
            self.out_v = try self.rig.output(half_bytes);
            self.base_out[0] = try self.rig.scratch(half_bytes);
            self.base_out[1] = try self.rig.scratch(half_bytes);
        } else {
            self.out = try self.rig.output(bytes);
        }
        return true;
    }
    /// Main's router on the same inputs; IDs that differ append main's, so no byte comparison can pass.
    pub fn run(self: *Probe) !void {
        const pass = self.rig.begin();
        const e = pass.e;
        const pl = &self.pipes;
        const R = self.refs;
        const y = at(self.out);
        const r: u32 = @intCast(self.shape.rows);
        const n: u32 = @intCast(self.shape.n);
        switch (self.kind) {
            .embed_norm => host.embedNorm(pl, e, at(R[0]), at(R[1]), at(R[2]), y, r),
            .ar => host.ar(pl, e, at(R[0]), at(R[1]), at(R[2]), y, r, 2, .{ .rows = ar_batch, .blocks_stride = ar_blocks, .partial_stride = p.hidden, .out_stride = 2 * p.hidden }),
            .conv_static => host.convStatic(pl, e, at(R[0]), at(R[1]), y, r, n),
            .conv_dynamic => if (self.segmented) {
                const split = @max(1, r / 2);
                const seg_count: usize = if (r == 1) 1 else 2;
                const table = try self.rig.input(std.mem.sliceAsBytes(&[_]host.InkConvDynSeg{
                    .{ .taps = R[0].buf.gpuAddress() + R[0].off, .input = R[1].buf.gpuAddress() + R[1].off, .out = y.buf.gpuAddress() + y.off, .row0 = 0, .rows = split },
                    .{ .taps = R[0].buf.gpuAddress() + R[0].off + split * n * 2, .input = R[1].buf.gpuAddress() + R[1].off + split * p.hidden * 2, .out = y.buf.gpuAddress() + y.off + split * n * 2, .row0 = split, .rows = r - split },
                }));
                e.useResources(&.{ R[0].buf.id, R[1].buf.id, y.buf.id }, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
                host.convDynamicSegs(pl, e, at(table), at(R[2]), at(R[3]), @intCast(seg_count), r, n);
            } else host.convDynamic(pl, e, at(R[0]), at(R[1]), at(R[2]), at(R[3]), y, r, n),
            .kv_conv_static => host.convStaticKv(pl, e, at(R[0]), at(R[1]), y, at(R[2]), at(R[3]), at(self.out_v), r, n),
            .kv_conv_dynamic => host.convDynamicKv(pl, e, at(R[0]), at(R[1]), at(R[2]), at(R[3]), y, at(R[4]), at(R[5]), at(R[6]), at(R[7]), at(self.out_v), r, n),
            .router => if (self.split)
                host.routerSplit(pl, e, at(R[0]), at(R[1]), at(R[2]), at(self.scores), at(self.route_ids), y, r)
            else
                host.router(pl, e, at(R[0]), at(R[1]), at(R[2]), at(self.route_ids), y, r),
            .swiglu => host.swiglu(pl, e, at(R[0]), y, r, n),
            .logits => host.logitsBaseline(pl, e, at(R[0]), at(R[1]), y, r, n),
            .expert_pick => host.expertPick(pl, e, at(R[0]), at(R[1]), y, r, n),
            .expert_combine => host.expertCombine(pl, e, at(R[0]), at(R[1]), y, r, n),
            .argmax => host.argmax(pl, e, at(R[0]), y, r, n),
            .cached_ar, .block_inv_rms, .attn_partial_ar, .mlp_partial_ar, .norm_bf16, .attn_partial_ar_norm, .mlp_partial_ar_norm => unreachable,
        }
        try self.rig.finish(pass);
    }
    /// Main's router or the two un-paired K+V conv dispatches on the exact same inputs.
    pub fn replaced(self: *Probe) ![]const u8 {
        const r: u32 = @intCast(self.shape.rows);
        const R = self.refs;
        if (self.segmented) {
            const out = try self.rig.scratch(self.count() * 2);
            const pass = self.rig.begin();
            host.convDynamic(&self.pipes, pass.e, at(R[0]), at(R[1]), at(R[2]), at(R[3]), at(out), r, @intCast(self.shape.n));
            try self.rig.finish(pass);
            return self.rig.bytes(out, self.count() * 2);
        }
        if (self.kind == .router) {
            const ids = try self.rig.scratch(r * 8 * 4);
            const weights = try self.rig.scratch(r * 8 * 4);
            const pass = self.rig.begin();
            host.router(&self.pipes, pass.e, at(R[0]), at(R[1]), at(R[2]), at(ids), at(weights), r);
            try self.rig.finish(pass);
            const want = self.rig.bytes(weights, r * 8 * 4);
            const main_ids = self.rig.bytes(ids, r * 8 * 4);
            if (std.mem.eql(u8, main_ids, self.rig.bytes(self.route_ids, r * 8 * 4))) return want;
            return std.mem.concat(self.arena.allocator(), u8, &.{ want, main_ids });
        }
        const pl = if (self.base_pipes) |*bp| bp else &self.pipes;
        return kv_conv.replaced(self.a, &self.rig, pl, self.kind, &R, &self.base_out, &self.base_bytes, self.shape);
    }
    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        const r = self.shape.rows;
        const n = self.shape.n;
        for (0..r) |row| switch (self.kind) {
            .embed_norm => {
                const x = self.f[0][self.ids[row] * p.hidden ..][0..p.hidden];
                for (0..p.hidden) |d| want[row * p.hidden + d] = p.embedValue(x, self.f[1], d);
            },
            .ar => {
                for (0..ar_batch) |b| for (0..2) |q| p.blockResidual(arBlocks(self.f[0], b), arPartial(self.f[1], b), self.f[2], r, q, want[(b * 2 + q) * p.hidden ..][0..p.hidden]);
                break;
            },
            .conv_static => {
                for (0..n) |d| want[row * n + d] = p.staticConv(self.f[0], self.f[1], n, row, d);
            },
            .conv_dynamic => {
                for (0..n) |d| want[row * n + d] = p.dynamicConv(self.f[0], self.f[1], self.f[2], self.f[3], n, row, d);
            },
            .kv_conv_static, .kv_conv_dynamic => kv_conv.referenceRow(self.kind, &self.f, r, n, row, want),
            .router => {
                const route = p.route(self.f[0][row * p.hidden ..][0..p.hidden], self.f[1], self.f[2]);
                @memcpy(want[row * 8 ..][0..8], &route.weights);
            },
            .swiglu => {
                for (0..n) |d| want[row * n + d] = p.swiglu(self.f[0][(row * n + d) * 2], self.f[0][(row * n + d) * 2 + 1]);
            },
            .logits => {
                for (0..n) |v| want[row * n + v] = p.logit(self.f[0][row * p.hidden ..][0..p.hidden], self.f[1], v);
            },
            .expert_pick => {
                for (0..8) |j| {
                    for (0..n) |d| want[(row * 8 + j) * n + d] = p.pick(self.f[0], self.ids, n, row, j, d);
                }
            },
            .expert_combine => {
                for (0..n) |d| want[row * n + d] = p.combine(self.f[0][row * 8 * n ..], self.f[1][row * 8 ..][0..8], n, d);
            },
            .argmax => want[row] = @floatFromInt(p.argmax(self.f[0][row * n ..][0..n])),
            .cached_ar, .block_inv_rms, .attn_partial_ar, .mlp_partial_ar, .norm_bf16, .attn_partial_ar_norm, .mlp_partial_ar_norm => unreachable,
        };
        for (mag, want) |*m, w| m.* = 10 * @abs(w) + 1e-5;
    }
    pub fn simulate(self: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.kind == .ar) {
            const sim = fused.arSim(acc, order);
            for (0..ar_batch) |b| for (0..2) |q| p.blockResidualIn(arBlocks(self.f[0], b), arPartial(self.f[1], b), self.f[2], self.shape.rows, q, sim, out[(b * 2 + q) * p.hidden ..][0..p.hidden]);
            return;
        }
        const mag = try self.a.alloc(f64, out.len);
        defer self.a.free(mag);
        return self.reference(out, mag);
    }
    pub fn check(self: *Probe, gate: g.Gate) !?[]const u8 {
        if (self.kind == .argmax) {
            const got = std.mem.bytesAsSlice(u32, try self.rig.outputBytes());
            const n = self.shape.n;
            for (0..self.shape.rows) |row| {
                if (got[row] != p.argmax(self.f[0][row * n ..][0..n])) return "argmax token differs";
            }
            return null;
        }
        if (self.kind != .router) return "numeric hooks must be used";
        const bytes = try self.rig.outputBytes();
        const want = try self.a.alloc(f64, self.count());
        defer self.a.free(want);
        const mag = try self.a.alloc(f64, self.count());
        defer self.a.free(mag);
        try self.reference(want, mag);
        if (gate == .nonfinite) {
            for (want, 0..) |w, i| if (std.math.isNan(c.read(.f32, bytes, i)) != std.math.isNan(w)) return "NaN normalization rule differs";
            return null;
        }
        const got_ids = std.mem.bytesAsSlice(u32, self.rig.bytes(self.route_ids, self.shape.rows * 8 * 4));
        for (0..self.shape.rows) |row| {
            const route = p.route(self.f[0][row * p.hidden ..][0..p.hidden], self.f[1], self.f[2]);
            for (0..8) |j| {
                if (got_ids[row * 8 + j] != route.ids[j]) return "selected IDs differ";
                const got = c.read(.f32, bytes, row * 8 + j);
                if (@abs(got - want[row * 8 + j]) > 1e-5 * @max(1, @abs(want[row * 8 + j]))) return "router weight differs";
            }
        }
        if (!self.split and !self.nan_keys_run) {
            self.nan_keys_run = true;
            return self.nanKeys();
        }
        return null;
    }
    /// Six rows with NaN keys (inf times a zero weight) at varied experts, 50 launches, each the serial scan's IDs.
    fn nanKeys(self: *Probe) !?[]const u8 {
        const sets = [_][]const u8{ &.{0}, &.{5}, &.{63}, &.{127}, &.{ 7, 90 }, &.{ 1, 2, 3, 126 } };
        const r = sets.len;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        var prng = std.Random.DefaultPrng.init(0x4919);
        const rng = prng.random();
        const x = try self.floats(r * p.hidden, rng);
        const w = try self.floats(p.expert_count * p.hidden, rng);
        const bias = try self.arena.allocator().alloc(f32, p.expert_count);
        for (bias, 0..) |*b, e| b.* = @as(f32, @floatFromInt(e * 37 % 128)) / 1024 - 0.0625;
        round(x);
        round(w);
        for (sets, 0..) |set, row| {
            x[row * p.hidden + row] = std.math.inf(f32);
            for (0..p.expert_count) |e| if (w[e * p.hidden + row] == 0) {
                w[e * p.hidden + row] = 0.125;
            };
            for (set) |e| w[@as(usize, e) * p.hidden + row] = 0;
        }
        const xs = try self.input(x, true);
        const ws = try self.input(w, true);
        const bs = try self.input(bias, false);
        const ids = try self.rig.scratch(r * 8 * 4);
        const weights = try self.rig.scratch(r * 8 * 4);
        var first: [r * 8 * 4]u8 = undefined;
        for (0..50) |launch| {
            const pass = self.rig.begin();
            host.router(&self.pipes, pass.e, at(xs), at(ws), at(bs), at(ids), at(weights), r);
            try self.rig.finish(pass);
            const got_ids = std.mem.bytesAsSlice(u32, self.rig.bytes(ids, r * 8 * 4));
            const got = self.rig.bytes(weights, r * 8 * 4);
            for (0..r) |row| {
                const route = p.route(x[row * p.hidden ..][0..p.hidden], w, bias);
                for (0..8) |j| {
                    if (got_ids[row * 8 + j] != route.ids[j]) return "NaN-key selected IDs differ from the serial scan";
                    const v = c.read(.f32, got, row * 8 + j);
                    const want = route.weights[j];
                    if (std.math.isNan(v) != std.math.isNan(want)) return "NaN-key weight class differs from the serial scan";
                    if (!std.math.isNan(want) and @abs(v - want) > 1e-5 * @max(1, @abs(want))) return "NaN-key router weight differs";
                }
            }
            if (launch == 0) {
                @memcpy(&first, got);
            } else if (!std.mem.eql(u8, &first, got)) return "a repeated NaN-key launch differs from the first";
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,
    fused_probes: std.ArrayList(*fused.Probe) = .empty,
    norm_probes: std.ArrayList(*norm_fused.Probe) = .empty,

    const InitCtx = struct {
        s: *Suite,
        kind: p.Kind,
        text: []const u8,
        split: bool = false,

        fn initProbe(ctx: InitCtx) !Probe {
            return if (ctx.split)
                Probe.initSplit(ctx.s.a, ctx.s.device, ctx.s.queue, ctx.text)
            else
                Probe.initSource(ctx.s.a, ctx.s.device, ctx.s.queue, ctx.kind, ctx.text);
        }

        fn initFused(ctx: InitCtx) !fused.Probe {
            return fused.Probe.initSource(ctx.s.a, ctx.s.device, ctx.s.queue, ctx.kind, ctx.text);
        }

        fn initNormFused(ctx: InitCtx) !norm_fused.Probe {
            return norm_fused.Probe.initSource(ctx.s.a, ctx.s.device, ctx.s.queue, ctx.kind, ctx.text);
        }

        fn loadBase(ctx: InitCtx, probe: *Probe) !void {
            probe.base_pipes = try host.Pipelines.load(ctx.s.device);
        }

        fn markSegmented(_: InitCtx, probe: *Probe) !void {
            probe.segmented = true;
        }
    };

    fn add(s: *Suite, kind: p.Kind, control: bool, only: ?[]const u8) !void {
        const name = if (control) controls.name(kind) else kind.label();
        if (only) |filter| if (std.mem.indexOf(u8, name, filter) == null) return;
        const owned = if (control) try controls.source(s.a, sources.glue, kind) else null;
        defer if (owned) |buf| s.a.free(buf);
        const ctx: InitCtx = .{ .s = s, .kind = kind, .text = owned orelse sources.glue };
        const ctrl = if (control) name else null;
        if (kind == .norm_bf16 or kind == .attn_partial_ar_norm or kind == .mlp_partial_ar_norm) {
            try controls.addAcquired(norm_fused.Probe, InitCtx, s.a, &s.norm_probes, s.kernels, ctx, InitCtx.initNormFused, null, ctrl);
            return;
        }
        if (kind == .cached_ar or kind == .block_inv_rms or kind == .attn_partial_ar or kind == .mlp_partial_ar) {
            try controls.addAcquired(fused.Probe, InitCtx, s.a, &s.fused_probes, s.kernels, ctx, InitCtx.initFused, null, ctrl);
            return;
        }
        const load_base: ?*const fn (InitCtx, *Probe) anyerror!void = if (control and kv_conv.isKvConv(kind)) InitCtx.loadBase else null;
        try controls.addAcquired(Probe, InitCtx, s.a, &s.probes, s.kernels, ctx, InitCtx.initProbe, load_base, ctrl);
    }

    fn addNanControl(s: *Suite, control: controls.NanControl, only: ?[]const u8) !void {
        const name = controls.nanName(control);
        if (only) |filter| if (std.mem.indexOf(u8, name, filter) == null) return;
        const text = try controls.nanSource(s.a, sources.glue, control);
        defer s.a.free(text);
        try controls.addAcquired(Probe, InitCtx, s.a, &s.probes, s.kernels, .{ .s = s, .kind = .router, .text = text }, InitCtx.initProbe, null, name);
    }

    fn addSplit(s: *Suite, control: ?split_controls.Control, only: ?[]const u8) !void {
        const name = if (control) |x| split_controls.name(x) else split_label;
        if (only) |filter| if (std.mem.indexOf(u8, name, filter) == null) return;
        const owned = if (control) |x| try split_controls.source(s.a, sources.glue, x) else null;
        defer if (owned) |buf| s.a.free(buf);
        try controls.addAcquired(Probe, InitCtx, s.a, &s.probes, s.kernels, .{ .s = s, .kind = .router, .text = owned orelse sources.glue, .split = true }, InitCtx.initProbe, null, if (control != null) name else null);
    }

    fn addSegments(s: *Suite, control: bool, only: ?[]const u8) !void {
        const name = if (control) "control: " ++ segments_label else segments_label;
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const from = "const uint r = global_r - seg.row0;";
        if (std.mem.count(u8, sources.glue, from) != 1) return error.ControlEditMissing;
        const text = if (control) try std.mem.replaceOwned(u8, s.a, sources.glue, from, "const uint r = 0;") else try s.a.dupe(u8, sources.glue);
        defer s.a.free(text);
        try controls.addAcquired(Probe, InitCtx, s.a, &s.probes, s.kernels, .{ .s = s, .kind = .conv_dynamic, .text = text }, InitCtx.initProbe, InitCtx.markSegmented, if (control) name else null);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(p.Kind)) |kind| {
            try s.add(kind, false, only);
            try s.add(kind, true, only);
        }
        try s.addSegments(false, only);
        try s.addSegments(true, only);
        try s.addSplit(null, only);
        for (std.enums.values(controls.NanControl)) |control| try s.addNanControl(control, only);
        try s.addSplit(.route_strided, only);
        try s.addSplit(.route_index, only);
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
        for (s.fused_probes.items) |fp| {
            fp.deinit();
            s.a.destroy(fp);
        }
        s.fused_probes.deinit(s.a);
        for (s.norm_probes.items) |np| {
            np.deinit();
            s.a.destroy(np);
        }
        s.norm_probes.deinit(s.a);
    }
};
