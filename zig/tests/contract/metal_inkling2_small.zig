//! Inkling 2 Small (H=4608, E=384, 8 blocks/taps) Metal ops under FP64 references and guarded buffers.
const std = @import("std");
const mtl = @import("metal");
const controls = @import("metal_inkling2_small_controls.zig");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.small_ops;
const p = host.plan;
const sources = @import("inkling2_metal");

fn depth(s: c.Shape) usize {
    return s.k + 40;
}
fn value(rng: std.Random) f32 {
    return rng.float(f32) * 0.5 - 0.25;
}
fn round(values: []f32) void {
    for (values) |*v| v.* = c.fromBf16(c.toBf16(v.*));
}

const ar_batch = 2;
const ar_blocks = p.block_slots * p.hidden;
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
    shape: c.Shape = undefined,
    served: [3]c.Shape = undefined,
    f: [4][]f32 = @splat(&.{}),
    ids: []u32 = &.{},
    route_ids: rig_.Ref = undefined,
    refs: [4]rig_.Ref = undefined,
    blocks: rig_.Ref = undefined,
    blocks_before: []u16 = &.{},
    out: rig_.Ref = undefined,
    nan_keys_run: bool = false,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: p.Kind) !Probe {
        return initSource(a, device, queue, kind, sources.small_ops);
    }
    pub fn initSource(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: p.Kind, text: []const u8) !Probe {
        var self: Probe = .{
            .a = a,
            .arena = .init(a),
            .rig = .{ .a = a, .device = device, .queue = queue },
            .kind = kind,
            .pipes = try host.Pipelines.loadSource(device, text),
        };
        self.served = switch (kind) {
            .embed_norm => .{ .{ .rows = 1, .n = p.hidden, .k = p.hidden }, .{ .rows = 5, .n = p.hidden, .k = p.hidden }, .{ .rows = 7, .n = p.hidden, .k = p.hidden } },
            .ar => .{ .{ .rows = 2, .n = p.hidden, .k = p.hidden }, .{ .rows = 5, .n = p.hidden, .k = p.hidden }, .{ .rows = 9, .n = p.hidden, .k = p.hidden } },
            .conv_dynamic => .{ .{ .rows = 1, .n = 128, .k = 4 }, .{ .rows = 5, .n = 512, .k = 4 }, .{ .rows = 7, .n = 1024, .k = 4 } },
            .router => .{ .{ .rows = 1, .n = p.topk, .k = p.hidden }, .{ .rows = 5, .n = p.topk, .k = p.hidden }, .{ .rows = 7, .n = p.topk, .k = p.hidden } },
            .logits => .{ .{ .rows = 1, .n = 128, .k = p.hidden }, .{ .rows = 5, .n = 256, .k = p.hidden }, .{ .rows = 7, .n = 128, .k = p.hidden } },
            .attention_partial => .{ .{ .rows = 1, .n = p.hidden, .k = 0 }, .{ .rows = 3, .n = p.hidden, .k = 4 }, .{ .rows = 5, .n = p.hidden, .k = 8 } },
            .capture_tap => .{ .{ .rows = 1, .n = p.hidden, .k = 0 }, .{ .rows = 3, .n = p.hidden, .k = 3 }, .{ .rows = 5, .n = p.hidden, .k = 7 } },
        };
        return self;
    }
    pub fn deinit(self: *Probe) void {
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
            .ar, .router, .logits, .attention_partial => .f32,
            .embed_norm, .conv_dynamic, .capture_tap => .bf16,
        };
        const checked = self.kind == .router or self.kind == .attention_partial;
        return .{
            .ctx = self,
            .vt = if (checked) &vt_checked else &vt_plain,
            .spec = .{
                .name = self.kind.label(),
                .accum = if (self.kind == .attention_partial or self.kind == .capture_tap) .none else .f32,
                .depth = depth,
                .ops = if (self.kind == .attention_partial) 1 else if (self.kind == .capture_tap) 0 else 96,
                .mids = if (self.kind == .logits or self.kind == .conv_dynamic) 1 else 0,
                .out = out,
                .served = &self.served,
                .nan = if (self.kind == .router) .{ .stated = "a NaN selected score makes the shared normalization and every selected weight NaN" } else .propagate,
                .sums = if (self.kind == .ar) .{} else .{ .f16 = 0, .bf16 = 0, .why = "nonlinear norms, exponentials, BF16 endpoints or pointwise ops have no uniform lower-precision separability factor" },
            },
        };
    }

    pub fn count(self: *Probe) usize {
        return if (self.kind == .ar)
            ar_batch * 2 * p.hidden
        else
            p.outputCount(self.kind, .{ .rows = self.shape.rows, .n = self.shape.n, .k = self.shape.k });
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
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.f = @splat(&.{});
        self.blocks_before = &.{};
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const r = shape.rows;
        const n = shape.n;
        const nf = fill == .nonfinite;
        switch (self.kind) {
            .embed_norm => {
                self.f[0] = try self.floats(32 * p.hidden, rng);
                self.f[1] = try self.floats(p.hidden, rng);
                self.ids = try self.arena.allocator().alloc(u32, r);
                for (self.ids, 0..) |*id, i| id.* = @intCast(i % 32);
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
                    // 4608 * 256^2 = 9 * 2^25 is exact in FP32, overflows FP16, and stalls BF16.
                    for (0..ar_batch) |b| for (0..r) |s| {
                        const vals = if (s + 1 == r) arPartial(self.f[1], b) else arBlocks(self.f[0], b)[s * p.hidden ..][0..p.hidden];
                        @memset(vals, if (s % 2 == 0) 256 else -256);
                    };
                    for (0..2) |q| {
                        const row = self.f[2][q * p.hidden ..][0..p.hidden];
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
                self.refs[1] = try self.input(self.f[1], false);
                self.refs[2] = try self.input(self.f[2], false);
            },
            .conv_dynamic => {
                self.f[0] = try self.floats((r + 3) * n, rng);
                self.f[1] = try self.floats(r * p.hidden, rng);
                self.f[2] = try self.floats(4 * n, rng);
                self.f[3] = try self.floats(32 * p.hidden, rng);
                if (nf) self.f[1][0] = std.math.nan(f32);
                for (0..4) |j| self.refs[j] = try self.input(self.f[j], true);
            },
            .router => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                self.f[1] = try self.floats(p.expert_count * p.hidden, rng);
                self.f[2] = try self.floats(p.expert_count, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
                self.refs[2] = try self.input(self.f[2], false);
                self.route_ids = try self.rig.scratch(r * p.topk * 4);
            },
            .logits => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                self.f[1] = try self.floats(n * p.hidden, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
            },
            .attention_partial => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                self.f[1] = try self.floats(r * p.hidden, rng);
                round(self.f[0]);
                round(self.f[1]);
                if (nf) self.f[0][0] = std.math.nan(f32);
                const blks = try self.arena.allocator().alloc(u16, r * p.block_slots * p.hidden);
                for (blks) |*b| b.* = c.toBf16(value(rng));
                self.blocks_before = blks;
                self.blocks = try self.rig.scratch(std.mem.sliceAsBytes(blks).len);
                @memcpy(self.rig.bytes(self.blocks, std.mem.sliceAsBytes(blks).len), std.mem.sliceAsBytes(blks));
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
            },
            .capture_tap => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                round(self.f[0]);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
            },
        }
        for (self.f, 0..) |vals, j| if (vals.len > 0 and switch (self.kind) {
            .ar => j == 0,
            .router => j < 2,
            .attention_partial, .capture_tap => false,
            else => true,
        }) round(vals);
        const elem: c.Elem = switch (self.kind) {
            .ar, .router, .logits, .attention_partial => .f32,
            .embed_norm, .conv_dynamic, .capture_tap => .bf16,
        };
        self.out = try self.rig.output(self.count() * c.size(elem));
        if (self.kind == .capture_tap) {
            const untouched = std.mem.bytesAsSlice(u16, self.rig.bytes(self.out, self.count() * 2));
            @memset(untouched, c.toBf16(1));
        }
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
            .ar => host.ar(pl, e, at(R[0]), at(R[1]), at(R[2]), y, r, 2, .{ .rows = ar_batch, .blocks_stride = ar_blocks, .partial_stride = p.hidden, .out_stride = 2 * p.hidden }),
            .conv_dynamic => host.convDynamic(pl, e, at(R[0]), at(R[1]), at(R[2]), at(R[3]), y, r, n),
            .router => host.router(pl, e, at(R[0]), at(R[1]), at(R[2]), at(self.route_ids), y, r),
            .logits => host.logits(pl, e, at(R[0]), at(R[1]), y, r, n),
            .attention_partial => host.attentionPartial(pl, e, at(R[0]), at(R[1]), at(self.blocks), y, self.shape.rows, self.shape.k),
            .capture_tap => host.captureTap(pl, e, at(R[0]), y, self.shape.rows, self.shape.k),
        }
        try self.rig.finish(pass);
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        const r = self.shape.rows;
        const n = self.shape.n;
        switch (self.kind) {
            .embed_norm => p.embedNormRef(self.ids, self.f[0], self.f[1], want),
            .ar => {
                for (0..ar_batch) |b| {
                    const blks = arBlocks(self.f[0], b);
                    const part = arPartial(self.f[1], b);
                    p.arRef(blks, part, self.f[2], r, 2, want[b * 2 * p.hidden ..][0 .. 2 * p.hidden]);
                    var inv: [p.max_sources]f64 = undefined;
                    for (0..r) |s| {
                        var ss: f64 = 0;
                        for (0..p.hidden) |d| {
                            const v: f64 = if (s + 1 == r) part[d] else blks[s * p.hidden + d];
                            ss += v * v;
                        }
                        inv[s] = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(p.hidden)) + p.eps);
                    }
                    for (0..2) |q| for (0..p.hidden) |d| {
                        var term_mag: f64 = 0;
                        for (0..r) |s| {
                            const v: f64 = if (s + 1 == r) part[d] else blks[s * p.hidden + d];
                            term_mag += @abs(v * inv[s]);
                        }
                        const idx = (b * 2 + q) * p.hidden + d;
                        mag[idx] = @max(10 * @abs(want[idx]) + 1e-5, term_mag / @as(f64, @floatFromInt(r)));
                    };
                }
                return;
            },
            .conv_dynamic => p.convDynamicRef(self.f[0], self.f[1], self.f[2], self.f[3], r, n, want),
            .router => {
                const ids_tmp = try self.a.alloc(u32, r * p.topk);
                defer self.a.free(ids_tmp);
                p.routerRef(self.f[0], self.f[1], self.f[2], r, ids_tmp, want);
            },
            .logits => p.logitsRef(self.f[0], self.f[1], r, n, want),
            .attention_partial => {
                for (want, mag, 0..) |*w, *m, i| {
                    w.* = p.attentionPartialRef(self.f[0][i], self.f[1][i], self.shape.k);
                    m.* = @abs(@as(f64, self.f[0][i])) + @abs(@as(f64, self.f[1][i]));
                }
                return;
            },
            .capture_tap => {
                for (want, mag, 0..) |*w, *m, i| {
                    const row = i / (p.tap_channels * p.hidden);
                    const ch = (i / p.hidden) % p.tap_channels;
                    const x = self.f[0][row * p.hidden + i % p.hidden];
                    w.* = if (ch == self.shape.k) x else 1;
                    m.* = @abs(@as(f64, x));
                }
                return;
            },
        }
        for (mag, want) |*m, w| m.* = 10 * @abs(w) + 1e-5;
    }

    pub fn simulate(self: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.kind == .ar) {
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
            for (0..ar_batch) |b| p.arIn(arBlocks(self.f[0], b), arPartial(self.f[1], b), self.f[2], self.shape.rows, 2, sim, out[b * 2 * p.hidden ..][0 .. 2 * p.hidden]);
            return;
        }
        const mag = try self.a.alloc(f64, out.len);
        defer self.a.free(mag);
        return self.reference(out, mag);
    }

    pub fn check(self: *Probe, gate: g.Gate) !?[]const u8 {
        if (self.kind == .attention_partial) return self.checkAttentionPartial(gate);
        if (self.kind != .router) return "numeric hooks must be used";
        const bytes = try self.rig.outputBytes();
        const want = try self.a.alloc(f64, self.count());
        defer self.a.free(want);
        const ids_want = try self.a.alloc(u32, self.shape.rows * p.topk);
        defer self.a.free(ids_want);
        p.routerRef(self.f[0], self.f[1], self.f[2], self.shape.rows, ids_want, want);
        if (gate == .nonfinite) {
            for (want, 0..) |w, i| if (std.math.isNan(c.read(.f32, bytes, i)) != std.math.isNan(w)) return "NaN normalization rule differs";
            return null;
        }
        const got_ids = std.mem.bytesAsSlice(u32, self.rig.bytes(self.route_ids, self.shape.rows * p.topk * 4));
        for (0..self.shape.rows * p.topk) |i| {
            if (got_ids[i] != ids_want[i]) return "selected IDs differ";
            const got = c.read(.f32, bytes, i);
            if (@abs(got - want[i]) > 1e-5 * @max(1, @abs(want[i]))) return "router weight differs";
        }
        if (!self.nan_keys_run) {
            self.nan_keys_run = true;
            return self.nanKeys();
        }
        return null;
    }

    fn checkAttentionPartial(self: *Probe, gate: g.Gate) !?[]const u8 {
        if (gate != .shapes and gate != .nonfinite) return "numeric hooks must be used";
        const want = try self.a.alloc(f64, self.count());
        defer self.a.free(want);
        const mag = try self.a.alloc(f64, self.count());
        defer self.a.free(mag);
        try self.reference(want, mag);
        const bytes = try self.rig.outputBytes();
        var planted: usize = 0;
        for (want, 0..) |w, i| {
            const got = c.read(.f32, bytes, i);
            if (c.classOf(got) != c.classOf(w)) return "partial's floating-point class differs";
            if (c.classOf(w) != .finite) planted += 1;
            if (gate == .shapes and std.math.isFinite(w) and got != w) return "FP32 partial differs from FP64 reference";
        }
        if (gate == .nonfinite and (planted == 0 or planted == want.len)) return "planted NaN does not discriminate";
        const got_blocks = std.mem.bytesAsSlice(u16, self.rig.bytes(self.blocks, std.mem.sliceAsBytes(self.blocks_before).len));
        for (0..self.shape.rows) |row| for (0..p.block_slots) |slot| for (0..p.hidden) |d| {
            const idx = (row * p.block_slots + slot) * p.hidden + d;
            const expected = if (self.shape.k != 0 and slot == self.shape.k - 1)
                c.toBf16(self.f[0][row * p.hidden + d])
            else
                self.blocks_before[idx];
            if (got_blocks[idx] != expected) return "completed-block row/slot layout differs";
        };
        return null;
    }

    /// Six rows with NaN keys across 384 experts, 25 launches, each matching the serial scan's IDs and weights.
    fn nanKeys(self: *Probe) !?[]const u8 {
        const sets = [_][]const u16{ &.{0}, &.{5}, &.{127}, &.{383}, &.{ 7, 290 }, &.{ 1, 2, 3, 382 } };
        const r = sets.len;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        var prng = std.Random.DefaultPrng.init(0x5094);
        const rng = prng.random();
        const x = try self.floats(r * p.hidden, rng);
        const w = try self.floats(p.expert_count * p.hidden, rng);
        const bias = try self.arena.allocator().alloc(f32, p.expert_count);
        for (bias, 0..) |*b, e| b.* = @as(f32, @floatFromInt(e * 37 % 384)) / 3072 - 0.0625;
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
        const ids = try self.rig.scratch(r * p.topk * 4);
        const weights = try self.rig.scratch(r * p.topk * 4);
        var want_ids: [r * p.topk]u32 = undefined;
        var want_w: [r * p.topk]f64 = undefined;
        p.routerRef(x, w, bias, r, &want_ids, &want_w);
        var first: [r * p.topk * 4]u8 = undefined;
        for (0..25) |launch| {
            const pass = self.rig.begin();
            host.router(&self.pipes, pass.e, at(xs), at(ws), at(bs), at(ids), at(weights), r);
            try self.rig.finish(pass);
            const got_ids = std.mem.bytesAsSlice(u32, self.rig.bytes(ids, r * p.topk * 4));
            const got = self.rig.bytes(weights, r * p.topk * 4);
            for (0..r * p.topk) |i| {
                if (got_ids[i] != want_ids[i]) return "NaN-key selected IDs differ from serial scan";
                const v = c.read(.f32, got, i);
                if (std.math.isNan(v) != std.math.isNan(want_w[i])) return "NaN-key weight class differs";
                if (!std.math.isNan(want_w[i]) and @abs(v - want_w[i]) > 1e-5 * @max(1, @abs(want_w[i]))) return "NaN-key router weight differs";
            }
            if (launch == 0) {
                @memcpy(&first, got);
            } else if (!std.mem.eql(u8, &first, got)) return "repeated NaN-key launch differs";
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
            const text = try controls.source(s.a, sources.small_ops, kind);
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

    fn addNanControl(s: *Suite, control: controls.NanControl, only: ?[]const u8) !void {
        const name = controls.nanName(control);
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        const text = try controls.nanSource(s.a, sources.small_ops, control);
        defer s.a.free(text);
        probe.* = try Probe.initSource(s.a, s.device, s.queue, .router, text);
        errdefer probe.deinit();
        try s.probes.append(s.a, probe);
        var k = probe.kernel();
        k.spec.name = name;
        k.control = .shapes;
        try s.kernels.append(s.a, k);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(p.Kind)) |kind| {
            try s.add(kind, false, only);
            try s.add(kind, true, only);
        }
        for (std.enums.values(controls.NanControl)) |control| try s.addNanControl(control, only);
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
    }
};
