//! Inkling 2 Mini glue under FP64 references, guarded buffers and real Metal dispatch.
const std = @import("std");
const mtl = @import("metal");
const controls = @import("metal_inkling2_glue_controls.zig");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.glue;
const p = host.plan;
const sources = @import("inkling2_metal");

fn depth(s: c.Shape) usize {
    return s.k + 40;
}
fn value(rng: std.Random) f32 {
    return rng.float(f32) * 0.5 - 0.25;
}
fn fromBf(v: u16) f32 {
    return c.fromBf16(v);
}

fn round(values: []f32) void {
    for (values) |*v| v.* = c.fromBf16(c.toBf16(v.*));
}

fn dynSegSplit(total: usize, n_segs: usize, idx: usize) usize {
    if (total == 7 and n_segs == 3) return ([_]usize{ 1, 2, 4 })[idx];
    if (total == 5 and n_segs == 2) return ([_]usize{ 2, 3 })[idx];
    const q = total / n_segs;
    return if (idx + 1 == n_segs) total - q * idx else q;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: p.Kind,
    pipes: host.Pipelines,
    shape: c.Shape = undefined,
    served: [3]c.Shape = undefined,
    f: [4][]f32 = @splat(&.{}),
    ids: []u32 = &.{},
    route_ids: rig_.Ref = undefined,
    refs: [5]rig_.Ref = undefined,
    dyn_segs_ref: rig_.Ref = undefined,
    dyn_taps_refs: [4]rig_.Ref = undefined,
    dyn_out_refs: [4]rig_.Ref = undefined,
    dyn_seg_rows: [4]usize = @splat(0),
    dyn_n_segs: usize = 0,
    dyn_res_ids: [16]mtl.objc.Id = undefined,
    dyn_res_len: usize = 0,
    out: rig_.Ref = undefined,
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    nonfinite: bool = false,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: p.Kind) !Probe {
        return initSource(a, device, queue, kind, sources.glue);
    }
    pub fn initSource(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: p.Kind, text: []const u8) !Probe {
        var self: Probe = .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .kind = kind, .pipes = try host.Pipelines.loadSource(device, text) };
        self.served = switch (kind) {
            .embed_norm => .{ .{ .rows = 1, .n = p.hidden, .k = p.hidden }, .{ .rows = 5, .n = p.hidden, .k = p.hidden }, .{ .rows = 7, .n = p.hidden, .k = p.hidden } },
            .ar, .ar_bf16 => .{ .{ .rows = 2, .n = p.hidden, .k = p.hidden }, .{ .rows = 4, .n = p.hidden, .k = p.hidden }, .{ .rows = 5, .n = p.hidden, .k = p.hidden } },
            .conv_static => .{ .{ .rows = 1, .n = 128, .k = 4 }, .{ .rows = 5, .n = 1024, .k = 4 }, .{ .rows = 7, .n = p.hidden, .k = 4 } },
            .conv_dynamic => .{ .{ .rows = 1, .n = 128, .k = 4 }, .{ .rows = 5, .n = 512, .k = 4 }, .{ .rows = 7, .n = 1024, .k = 4 } },
            .conv_dynamic_segs => .{ .{ .rows = 2, .n = 128, .k = 4 }, .{ .rows = 5, .n = 512, .k = 4 }, .{ .rows = 7, .n = 1024, .k = 4 } },
            .router => .{ .{ .rows = 1, .n = 8, .k = p.hidden }, .{ .rows = 5, .n = 8, .k = p.hidden }, .{ .rows = 7, .n = 8, .k = p.hidden } },
            .swiglu => .{ .{ .rows = 1, .n = p.intermediate, .k = 2 }, .{ .rows = 5, .n = p.intermediate, .k = 2 }, .{ .rows = 7, .n = p.intermediate, .k = 2 } },
            .logits => .{ .{ .rows = 1, .n = 128, .k = p.hidden }, .{ .rows = 5, .n = 256, .k = p.hidden }, .{ .rows = 17, .n = 131, .k = p.hidden } },
            .argmax => .{ .{ .rows = 1, .n = 128, .k = p.hidden }, .{ .rows = 2, .n = 2057, .k = p.hidden }, .{ .rows = 17, .n = 131, .k = p.hidden } },
            .expert_pick => .{ .{ .rows = 1, .n = p.intermediate, .k = 8 }, .{ .rows = 5, .n = p.intermediate, .k = 8 }, .{ .rows = 7, .n = p.intermediate, .k = 8 } },
            .expert_combine => .{ .{ .rows = 1, .n = p.hidden, .k = 8 }, .{ .rows = 5, .n = p.hidden, .k = 8 }, .{ .rows = 7, .n = p.hidden, .k = 8 } },
        };
        return self;
    }
    pub fn deinit(self: *Probe) void {
        self.rig.deinit();
        self.arena.deinit();
        self.pipes.deinit();
    }
    const vt_checked = blk: {
        var vt = g.hooks(Probe);
        vt.bounds = null;
        vt.intervals = null;
        break :blk vt;
    };
    const vt_plain = blk: {
        var vt = g.hooks(Probe);
        vt.check = null;
        vt.bounds = null;
        vt.intervals = null;
        break :blk vt;
    };
    const vt_logits = blk: {
        var vt = g.hooks(Probe);
        vt.check = null;
        break :blk vt;
    };
    pub fn kernel(self: *Probe) g.Kernel {
        const out: c.Elem = switch (self.kind) {
            .ar, .ar_bf16, .router, .logits => .f32,
            .argmax => .raw,
            else => .bf16,
        };
        return .{
            .ctx = self,
            .vt = switch (self.kind) {
                .router, .argmax => &vt_checked,
                .logits => &vt_logits,
                else => &vt_plain,
            },
            .spec = .{
                .name = self.kind.label(),
                .accum = if (self.kind == .argmax) .none else .f32,
                .depth = depth,
                .ops = 96,
                .mids = if (self.kind == .logits) 1 else 0,
                .out = out,
                .tiles = if (self.kind == .logits or self.kind == .argmax) .{ .{ .size = 8 }, .{ .size = 8 }, .{} } else .{ .{}, .{}, .{} },
                .served = &self.served,
                .nan = if (self.kind == .router) .{ .stated = "a NaN selected score makes the shared normalization and every selected weight NaN" } else if (self.kind == .argmax) .{ .stated = "NaN at token 0 wins; later NaNs lose to finite tokens; finite ties pick the lowest id" } else .propagate,
                // AR's integer stress terms are exact in FP32 and must discriminate FP16/BF16 by 100x.
                .sums = if (self.kind == .ar or self.kind == .ar_bf16) .{} else .{ .f16 = 0, .bf16 = 0, .exact = self.kind != .logits, .why = "nonlinear norms, exponentials and BF16 endpoints have no uniform lower-precision separability factor" },
            },
        };
    }
    pub fn count(self: *Probe) usize {
        return if (self.kind == .ar or self.kind == .ar_bf16) 2 * p.hidden else p.count(self.kind, .{ .rows = self.shape.rows, .n = self.shape.n, .k = self.shape.k });
    }
    fn floats(self: *Probe, n: usize, rng: std.Random) ![]f32 {
        const out = try self.arena.allocator().alloc(f32, n);
        for (out) |*x| x.* = value(rng);
        return out;
    }
    fn input(self: *Probe, values: []const f32, bf: bool) !rig_.Ref {
        if (!bf) return self.rig.input(std.mem.sliceAsBytes(values));
        const bits = try self.arena.allocator().alloc(u16, values.len);
        for (bits, values) |*b, x| b.* = c.toBf16(x);
        return self.rig.input(std.mem.sliceAsBytes(bits));
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
        self.lo = &.{};
        self.hi = &.{};
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
            .ar, .ar_bf16 => {
                self.f[0] = try self.floats((r - 1) * p.hidden, rng);
                self.f[1] = try self.floats(p.hidden, rng);
                self.f[2] = try self.floats(2 * p.hidden, rng);
                if (fill == .stress) {
                    // Each 2304-term square sum is exactly 9 * 2^24 in FP32 under every order.
                    for (0..r) |s| {
                        const values = if (s + 1 == r) self.f[1] else self.f[0][s * p.hidden ..][0..p.hidden];
                        @memset(values, if (s % 2 == 0) 256 else -256);
                    }
                    // Normalized mirrored queries leave +/-8; partial sums stay exact integers in FP32.
                    for (0..2) |query_row| {
                        const row = self.f[2][query_row * p.hidden ..][0..p.hidden];
                        for (0..p.hidden / 2) |d| {
                            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                            const nudge: f32 = if (d < 8) 1 else 0;
                            row[d] = v;
                            row[p.hidden / 2 + d] = nudge - v;
                        }
                    }
                }
                if (nf) self.f[2][0] = std.math.inf(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], self.kind == .ar_bf16);
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
            .conv_dynamic_segs => {
                self.dyn_n_segs = if (r >= 7) 3 else 2;
                self.f[0] = try self.floats((r + 3 * self.dyn_n_segs) * n, rng);
                self.f[1] = try self.floats(r * p.hidden, rng);
                self.f[2] = try self.floats(4 * n, rng);
                self.f[3] = try self.floats(32 * p.hidden, rng);
                round(self.f[0]);
                round(self.f[1]);
                round(self.f[2]);
                round(self.f[3]);
                if (nf) self.f[1][0] = std.math.nan(f32);
                self.refs[1] = try self.input(self.f[1], true);
                self.refs[2] = try self.input(self.f[2], true);
                self.refs[3] = try self.input(self.f[3], true);
                var seg_descs: [4]host.InkConvDynSeg = undefined;
                self.dyn_res_len = 0;
                self.dyn_res_ids[self.dyn_res_len] = self.refs[1].buf.id;
                self.dyn_res_len += 1;
                var row0: usize = 0;
                var tap_off: usize = 0;
                for (0..self.dyn_n_segs) |si| {
                    const sr = dynSegSplit(r, self.dyn_n_segs, si);
                    self.dyn_seg_rows[si] = sr;
                    const t_len = (sr + 3) * n;
                    self.dyn_taps_refs[si] = try self.input(self.f[0][tap_off..][0..t_len], true);
                    self.dyn_out_refs[si] = try self.rig.output(sr * n * 2);
                    seg_descs[si] = .{
                        .taps = self.dyn_taps_refs[si].buf.gpuAddress() + self.dyn_taps_refs[si].off,
                        .input = self.refs[1].buf.gpuAddress() + self.refs[1].off + row0 * p.hidden * 2,
                        .out = self.dyn_out_refs[si].buf.gpuAddress() + self.dyn_out_refs[si].off,
                        .row0 = @intCast(row0),
                        .rows = @intCast(sr),
                    };
                    self.dyn_res_ids[self.dyn_res_len] = self.dyn_taps_refs[si].buf.id;
                    self.dyn_res_ids[self.dyn_res_len + 1] = self.dyn_out_refs[si].buf.id;
                    self.dyn_res_len += 2;
                    row0 += sr;
                    tap_off += t_len;
                }
                self.dyn_segs_ref = try self.rig.input(std.mem.sliceAsBytes(seg_descs[0..self.dyn_n_segs]));
            },
            .router => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                self.f[1] = try self.floats(p.expert_count * p.hidden, rng);
                self.f[2] = try self.floats(p.expert_count, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
                self.refs[2] = try self.input(self.f[2], false);
                self.route_ids = try self.rig.scratch(r * 8 * 4);
            },
            .swiglu => {
                self.f[0] = try self.floats(r * n * 2, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
            },
            .logits => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                self.f[1] = try self.floats(n * p.hidden, rng);
                self.lo = try self.arena.allocator().alloc(f64, self.count());
                self.hi = try self.arena.allocator().alloc(f64, self.count());
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
            },
            .argmax => {
                self.f[0] = try self.arena.allocator().alloc(f32, r * p.hidden);
                self.f[1] = try self.arena.allocator().alloc(f32, n * p.hidden);
                @memset(self.f[0], 0);
                @memset(self.f[1], 0);
                for (0..r) |row| {
                    for (0..8) |d| self.f[0][row * p.hidden + d] = if (d < 4) 1 else -0.5;
                    if (row % 3 == 2) {
                        for (8..16) |d| self.f[0][row * p.hidden + d] = 128;
                    }
                }
                for (0..n) |v| {
                    const pat = rng.uintLessThan(u32, 16);
                    for (0..4) |d| self.f[1][v * p.hidden + d] = @floatFromInt(@as(i32, @intCast((pat >> @intCast(d)) & 1)) * 2 - 1);
                    if (v == 1 or v == 2 or (n > 8 and v == n - 1)) {
                        for (0..4) |d| self.f[1][v * p.hidden + d] = 2;
                    }
                    if (v >= 16 and v < 20) self.f[1][v * p.hidden + 8] = 8;
                }
                if (nf) {
                    @memset(self.f[1][0..p.hidden], 0);
                    self.f[1][0] = std.math.inf(f32);
                    self.f[1][1] = std.math.inf(f32);
                    self.f[0][0] = 1;
                    self.f[0][1] = -1;
                    if (r > 1) {
                        @memset(self.f[0][p.hidden .. 2 * p.hidden], 0);
                        self.f[0][p.hidden + 0] = -1;
                        self.f[0][p.hidden + 1] = -1;
                        self.f[0][p.hidden + 2] = 1;
                        self.f[0][p.hidden + 3] = -1;
                        self.f[0][p.hidden + 4] = 20;
                    }
                    if (n > 5) {
                        @memset(self.f[1][2 * p.hidden .. 3 * p.hidden], 0);
                        self.f[1][2 * p.hidden + 4] = 1;
                        @memset(self.f[1][5 * p.hidden .. 6 * p.hidden], 0);
                        self.f[1][5 * p.hidden + 2] = std.math.inf(f32);
                        self.f[1][5 * p.hidden + 3] = std.math.inf(f32);
                    }
                }
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
                self.refs[2] = try self.rig.scratch(r * n * 4);
                const parts = (n + 7) / 8;
                self.refs[3] = try self.rig.scratch(r * parts * 4);
                self.refs[4] = try self.rig.scratch(r * parts * 4);
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
        }
        // Every BF16 input and the FP64 oracle see exactly the same stored values.
        for (self.f, 0..) |values, j| if (values.len > 0 and switch (self.kind) {
            .ar => j == 0,
            .ar_bf16, .router => j < 2,
            .expert_combine => j == 0,
            else => true,
        }) round(values);
        const bytes = self.count() * @as(usize, if (self.kind == .ar or self.kind == .ar_bf16 or self.kind == .router or self.kind == .logits or self.kind == .argmax) 4 else 2);
        if (self.kind != .conv_dynamic_segs) self.out = try self.rig.output(bytes);
        return true;
    }
    fn at(ref: rig_.Ref) host.Ref {
        return .{ .buf = ref.buf, .off = ref.off };
    }
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
            .ar => host.ar(pl, e, at(R[0]), at(R[1]), at(R[2]), y, r, 2),
            .ar_bf16 => host.arBf16(pl, e, at(R[0]), at(R[1]), at(R[2]), y, r, 2),
            .conv_static => host.convStatic(pl, e, at(R[0]), at(R[1]), y, r, n),
            .conv_dynamic => host.convDynamic(pl, e, at(R[0]), at(R[1]), at(R[2]), at(R[3]), y, r, n),
            .conv_dynamic_segs => {
                e.useResources(self.dyn_res_ids[0..self.dyn_res_len], mtl.ResourceUsage.read | mtl.ResourceUsage.write);
                host.convDynamicSegs(pl, e, at(self.dyn_segs_ref), at(R[2]), at(R[3]), @intCast(self.dyn_n_segs), r, n);
            },
            .router => host.router(pl, e, at(R[0]), at(R[1]), at(R[2]), at(self.route_ids), y, r),
            .swiglu => host.swiglu(pl, e, at(R[0]), y, r, n),
            .logits => host.logits(pl, e, at(R[0]), at(R[1]), y, r, n),
            .argmax => host.logitsArgmax(pl, e, at(R[0]), at(R[1]), at(R[2]), at(R[3]), at(R[4]), y, r, n, false),
            .expert_pick => host.expertPick(pl, e, at(R[0]), at(R[1]), y, r, n),
            .expert_combine => host.expertCombine(pl, e, at(R[0]), at(R[1]), y, r, n),
        }
        try self.rig.finish(pass);
    }
    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        const r = self.shape.rows;
        const n = self.shape.n;
        for (0..r) |row| switch (self.kind) {
            .embed_norm => {
                const x = self.f[0][self.ids[row] * p.hidden ..][0..p.hidden];
                for (0..p.hidden) |d| want[row * p.hidden + d] = p.embedValue(x, self.f[1], d);
            },
            .ar, .ar_bf16 => {
                var qout: [2 * p.hidden]f64 = undefined;
                for (0..2) |q| p.blockResidual(self.f[0], self.f[1], self.f[2], r, q, qout[q * p.hidden ..][0..p.hidden]);
                @memcpy(want, &qout);
                break;
            },
            .conv_static => {
                for (0..n) |d| want[row * n + d] = p.staticConv(self.f[0], self.f[1], n, row, d);
            },
            .conv_dynamic => {
                for (0..n) |d| want[row * n + d] = p.dynamicConv(self.f[0], self.f[1], self.f[2], self.f[3], n, row, d);
            },
            .conv_dynamic_segs => {
                var row0: usize = 0;
                var tap_off: usize = 0;
                for (0..self.dyn_n_segs) |si| {
                    const sr = self.dyn_seg_rows[si];
                    const t_slice = self.f[0][tap_off .. tap_off + (sr + 3) * n];
                    const in_slice = self.f[1][row0 * p.hidden ..];
                    for (0..sr) |lr| for (0..n) |d| {
                        want[(row0 + lr) * n + d] = p.dynamicConv(t_slice, in_slice, self.f[2], self.f[3], n, lr, d);
                    };
                    row0 += sr;
                    tap_off += (sr + 3) * n;
                }
                break;
            },
            .router => {
                const route = p.route(self.f[0][row * p.hidden ..][0..p.hidden], self.f[1], self.f[2]);
                @memcpy(want[row * 8 ..][0..8], &route.weights);
            },
            .swiglu => {
                for (0..n) |d| want[row * n + d] = p.swiglu(self.f[0][(row * n + d) * 2], self.f[0][(row * n + d) * 2 + 1]);
            },
            .logits => {
                const err_gamma = c.gamma(@floatFromInt(depth(self.shape)), c.unit(.f32));
                const h_row = self.f[0][row * p.hidden ..][0..p.hidden];
                for (0..n) |v| {
                    const u_row = self.f[1][v * p.hidden ..][0..p.hidden];
                    var dot_f64: f64 = 0;
                    var sum_mag: f64 = 0;
                    for (0..p.hidden) |d| {
                        const term = @as(f64, u_row[d]) * h_row[d];
                        dot_f64 += term;
                        sum_mag += @abs(term);
                    }
                    const idx = row * n + v;
                    want[idx] = p.cap(p.roundBf16(dot_f64), 30);
                    mag[idx] = sum_mag;
                    if (std.math.isFinite(dot_f64)) {
                        const err_dot = err_gamma * sum_mag;
                        const cap_lo = p.cap(p.roundBf16(dot_f64 - err_dot), 30);
                        const cap_hi = p.cap(p.roundBf16(dot_f64 + err_dot), 30);
                        const cap_err = 96.0 * 0x1p-24 * @max(@abs(cap_lo), @abs(cap_hi)) + 0x1p-149;
                        self.lo[idx] = cap_lo - cap_err;
                        self.hi[idx] = cap_hi + cap_err;
                    } else {
                        self.lo[idx] = want[idx];
                        self.hi[idx] = want[idx];
                    }
                }
            },
            .argmax => return error.CheckedByItsOwnRule,
            .expert_pick => {
                for (0..8) |j| {
                    for (0..n) |d| want[(row * 8 + j) * n + d] = p.pick(self.f[0], self.ids, n, row, j, d);
                }
            },
            .expert_combine => {
                for (0..n) |d| want[row * n + d] = p.combine(self.f[0][row * 8 * n ..], self.f[1][row * 8 ..][0..8], n, d);
            },
        };
        if (self.kind != .logits) {
            for (mag, want) |*m, w| m.* = 10 * @abs(w) + 1e-5;
        }
    }
    pub fn bounds(self: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, self.lo, self.hi) |*l, w, lo, hi| l.* = if (std.math.isFinite(w)) @max(@abs(w - lo), @abs(hi - w)) else 0;
    }
    pub fn intervals(self: *Probe, low: []f64, high: []f64) !void {
        @memcpy(low, self.lo);
        @memcpy(high, self.hi);
    }
    pub fn simulate(self: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.kind == .ar or self.kind == .ar_bf16) {
            const sim: p.ArSum = .{
                .accum = switch (acc) {
                    .none => .none,
                    .f32 => .f32,
                    .f16 => .f16,
                    .bf16 => .bf16,
                },
                .order = switch (order) {
                    .sequential => .sequential,
                    .pairwise => .pairwise,
                    .strided => .strided,
                },
            };
            for (0..2) |q| p.blockResidualIn(self.f[0], self.f[1], self.f[2], self.shape.rows, q, sim, out[q * p.hidden ..][0..p.hidden]);
            return;
        }
        const mag = try self.a.alloc(f64, out.len);
        defer self.a.free(mag);
        return self.reference(out, mag);
    }
    fn argmaxCheck(self: *Probe) !?[]const u8 {
        const r = self.shape.rows;
        const n = self.shape.n;
        for (self.rig.bytes(self.refs[2], r * n * 4)) |b| {
            if (b != 0xa5) return "write_logits=false touched out logits buffer";
        }
        const got_ids = std.mem.bytesAsSlice(u32, try self.rig.outputBytes());
        for (0..r) |row| {
            const h_row = self.f[0][row * p.hidden ..][0..p.hidden];
            var best_val: f32 = 0;
            var best_id: ?usize = null;
            for (0..n) |v| {
                const u_row = self.f[1][v * p.hidden ..][0..p.hidden];
                var dot_f64: f64 = 0;
                for (0..p.hidden) |d| dot_f64 += @as(f64, u_row[d]) * h_row[d];
                const y: f32 = @floatCast(p.cap(p.roundBf16(dot_f64), 30));
                if (best_id == null or y > best_val) {
                    best_val = y;
                    best_id = v;
                }
            }
            if (got_ids[row] != best_id.?) {
                return try std.fmt.bufPrint(&self.msg, "row {d} picked id {d} where exact argmax picks {d}", .{ row, got_ids[row], best_id.? });
            }
        }
        return null;
    }
    pub fn check(self: *Probe, gate: g.Gate) !?[]const u8 {
        if (self.kind == .argmax) return self.argmaxCheck();
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
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    fn add(s: *Suite, kind: p.Kind, control: bool, only: ?[]const u8) !void {
        const name = if (control) controls.name(kind) else kind.label();
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        if (control) {
            const text = try controls.source(s.a, sources.glue, kind);
            defer s.a.free(text);
            probe.* = try Probe.initSource(s.a, s.device, s.queue, kind, text);
        } else {
            probe.* = try Probe.init(s.a, s.device, s.queue, kind);
        }
        errdefer probe.deinit();
        try s.probes.append(s.a, probe);
        var k = probe.kernel();
        if (control) {
            k.spec.name = name;
            k.control = .shapes;
        }
        try s.kernels.append(s.a, k);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(p.Kind)) |kind| {
            try s.add(kind, false, only);
            try s.add(kind, true, only);
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
    }
};
