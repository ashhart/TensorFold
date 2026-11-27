//! Byte-exact contract probes for Inkling 2 Mini's cached and fused AR glue kernels.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.glue;
const p = host.plan;
const sources = @import("inkling2_metal");

pub fn depth(s: c.Shape) usize {
    return s.k + 40;
}
pub fn value(rng: std.Random) f32 {
    return rng.float(f32) * 0.5 - 0.25;
}
pub fn round(values: []f32) void {
    for (values) |*v| v.* = c.fromBf16(c.toBf16(v.*));
}
pub fn writeF32s(bytes: []u8, v: f32) void {
    const bits: u32 = @bitCast(v);
    var i: usize = 0;
    while (i < bytes.len) : (i += 4) std.mem.writeInt(u32, bytes[i..][0..4], bits, .little);
}
pub fn fillStressQueries(queries: []f32, q_count: usize, rng: std.Random) void {
    for (0..q_count) |q| {
        const row = queries[q * p.hidden ..][0..p.hidden];
        for (0..p.hidden / 2) |d| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
            const nudge: f32 = if (d < 8) 1 else 0;
            row[d] = v;
            row[p.hidden / 2 + d] = nudge - v;
        }
    }
}
pub fn arSim(acc: c.Accum, order: g.Order) p.ArSum {
    return .{
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
}
const RuntimeArgs = extern struct { count: u32, boundary: u32 };
fn dispatchRt(pipe: mtl.Pipeline, e: mtl.ComputeEncoder, refs: []const host.Ref, count_val: usize, boundary: usize) void {
    e.setPipeline(pipe);
    for (refs, 0..) |r_, i| e.setBuffer(r_.buf, r_.off, i);
    e.setValue(RuntimeArgs{ .count = @intCast(count_val), .boundary = @intCast(boundary) }, refs.len);
    e.dispatchThreads(mtl.Size.of(count_val, 1, 1), mtl.Size.of(256, 1, 1));
}
pub fn at(ref: rig_.Ref) host.Ref {
    return .{ .buf = ref.buf, .off = ref.off };
}
fn atOff(ref: rig_.Ref, byte_off: usize) host.Ref {
    return .{ .buf = ref.buf, .off = ref.off + byte_off };
}

const cached_served = [_]c.Shape{
    .{ .rows = 2, .n = 2 * p.hidden, .k = p.hidden }, .{ .rows = 1, .n = p.hidden, .k = p.hidden },
    .{ .rows = 1, .n = 3 * p.hidden, .k = p.hidden }, .{ .rows = 2, .n = p.hidden, .k = p.hidden },
    .{ .rows = 3, .n = 3 * p.hidden, .k = p.hidden }, .{ .rows = 4, .n = p.hidden, .k = p.hidden },
    .{ .rows = 5, .n = p.hidden, .k = p.hidden },     .{ .rows = 5, .n = 3 * p.hidden, .k = p.hidden },
};
const inv_rms_served = [_]c.Shape{
    .{ .rows = 4, .n = 1, .k = p.hidden }, .{ .rows = 5, .n = 2, .k = p.hidden },
    .{ .rows = 7, .n = 3, .k = p.hidden }, .{ .rows = 32, .n = 4, .k = p.hidden },
};
const attn_served = blk: {
    var out: [25]c.Shape = undefined;
    out[0] = .{ .rows = 4, .n = 5 * p.hidden, .k = 4 };
    var idx: usize = 1;
    for (1..6) |s| for (0..5) |b| {
        if (s == 5 and b == 4) continue;
        out[idx] = .{ .rows = ((idx - 1) % 8) + 1, .n = s * p.hidden, .k = b };
        idx += 1;
    };
    break :blk out;
};
const mlp_served = blk: {
    var out: [15]c.Shape = undefined;
    out[0] = .{ .rows = 4, .n = 3 * p.hidden, .k = 5 };
    var idx: usize = 1;
    for (1..6) |s| for (1..4) |q| {
        if (s == 5 and q == 3) continue;
        out[idx] = .{ .rows = ((idx - 1) % 8) + 1, .n = q * p.hidden, .k = s };
        idx += 1;
    };
    break :blk out;
};

const RtPipes = struct {
    attn_partial: mtl.Pipeline,
    mlp_partial: mtl.Pipeline,
    widen: mtl.Pipeline,

    fn load(device: mtl.Device) !RtPipes {
        const lib = try mtl.Library.fromSource(device, sources.runtime, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var attn = try mtl.Pipeline.init(device, lib, "tf_ink2_attention_partial", false);
        errdefer attn.deinit();
        var mlp = try mtl.Pipeline.init(device, lib, "tf_ink2_mlp_partial", false);
        errdefer mlp.deinit();
        return .{ .attn_partial = attn, .mlp_partial = mlp, .widen = try mtl.Pipeline.init(device, lib, "tf_ink2_widen", false) };
    }
    fn deinit(r: *RtPipes) void {
        r.widen.deinit();
        r.mlp_partial.deinit();
        r.attn_partial.deinit();
    }
};

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: p.Kind,
    pipes: host.Pipelines,
    base: ?host.Pipelines = null,
    rt: ?RtPipes = null,
    shape: c.Shape = undefined,
    p_vals: []f32 = &.{},
    x_vals: []f32 = &.{},
    blocks_vals: []f32 = &.{},
    queries: []f32 = &.{},
    blocks_init_u16: []u16 = &.{},
    inv_rms_init: []f32 = &.{},
    refs: [5]rig_.Ref = undefined,
    blocks: rig_.Ref = undefined,
    p_out: rig_.Ref = undefined,
    out_partial: rig_.Ref = undefined,
    out_ar: rig_.Ref = undefined,
    out_inv_rms: rig_.Ref = undefined,
    base_p: rig_.Ref = undefined,
    base_partial: rig_.Ref = undefined,
    base_ar: rig_.Ref = undefined,
    base_inv_rms: rig_.Ref = undefined,
    out_bytes: []u8 = &.{},
    base_bytes: []u8 = &.{},

    pub fn initSource(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: p.Kind, text: []const u8) !Probe {
        var pipes = try host.Pipelines.loadSource(device, text);
        errdefer pipes.deinit();
        var base: ?host.Pipelines = if (kind != .block_inv_rms) try host.Pipelines.load(device) else null;
        errdefer if (base) |*b| b.deinit();
        const rt: ?RtPipes = if (kind == .attn_partial_ar or kind == .mlp_partial_ar) try RtPipes.load(device) else null;
        return .{
            .a = a,
            .arena = .init(a),
            .rig = .{ .a = a, .device = device, .queue = queue },
            .kind = kind,
            .pipes = pipes,
            .base = base,
            .rt = rt,
        };
    }

    pub fn deinit(self: *Probe) void {
        self.rig.deinit();
        self.arena.deinit();
        if (self.rt) |*rt| rt.deinit();
        if (self.base) |*base| base.deinit();
        self.pipes.deinit();
    }

    const vt_replaced = blk: {
        var hooks = g.hooks(Probe);
        hooks.output = outputHook;
        hooks.check = null;
        break :blk hooks;
    };
    const vt_plain = blk: {
        var hooks = vt_replaced;
        hooks.replaced = null;
        break :blk hooks;
    };
    fn outputHook(ctx: *anyopaque) anyerror![]const u8 {
        const self: *Probe = @ptrCast(@alignCast(ctx));
        return self.output();
    }

    pub fn kernel(self: *Probe) g.Kernel {
        const replaces: ?[]const u8 = switch (self.kind) {
            .cached_ar => "tf_inkling2_ar",
            .block_inv_rms => null,
            .attn_partial_ar => "tf_ink2_attention_partial + tf_inkling2_block_inv_rms + tf_inkling2_cached_ar",
            .mlp_partial_ar => "tf_ink2_mlp_partial + tf_ink2_widen + tf_inkling2_cached_ar",
            else => unreachable,
        };
        const served: []const c.Shape = switch (self.kind) {
            .cached_ar => &cached_served,
            .block_inv_rms => &inv_rms_served,
            .attn_partial_ar => &attn_served,
            .mlp_partial_ar => &mlp_served,
            else => unreachable,
        };
        return .{
            .ctx = self,
            .vt = if (self.kind == .block_inv_rms) &vt_plain else &vt_replaced,
            .spec = .{
                .name = self.kind.label(),
                .accum = .f32,
                .depth = depth,
                .ops = if (self.kind == .block_inv_rms) 8 else 96,
                .out = .f32,
                .replaces = replaces,
                .served = served,
                .sums = .{},
            },
        };
    }

    pub fn count(self: *Probe) usize {
        if (self.kind == .cached_ar) return self.shape.n;
        return p.count(self.kind, .{ .rows = self.shape.rows, .n = self.shape.n, .k = self.shape.k });
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

    fn fillStressBlocks(blocks: []f32, rows: usize) void {
        for (0..rows) |row| for (0..4) |s| {
            @memset(blocks[(row * 4 + s) * p.hidden ..][0..p.hidden], if (s % 2 == 0) 256 else -256);
        };
    }

    fn computeInvRms(self: *Probe, blocks_ref: rig_.Ref, r: usize) ![]f32 {
        const base = &(self.base orelse return error.NoBaseline);
        const pass = self.rig.begin();
        for (0..4) |slot| host.blockInvRms(base, pass.e, at(blocks_ref), at(self.base_inv_rms), @intCast(r), @intCast(slot));
        try self.rig.finish(pass);
        const inv = try self.arena.allocator().alloc(f32, r * 4);
        @memcpy(inv, std.mem.bytesAsSlice(f32, self.rig.bytes(self.base_inv_rms, r * 4 * 4)));
        return inv;
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!p.valid(self.kind, .{ .rows = shape.rows, .n = shape.n, .k = shape.k })) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const r = shape.rows;
        const n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (self.kind == .block_inv_rms) {
            const slot = n - 1;
            self.blocks_vals = try self.floats(r * 4 * p.hidden, rng);
            if (fill == .stress) {
                for (0..r) |row| {
                    const block = self.blocks_vals[(row * 4 + slot) * p.hidden ..][0..p.hidden];
                    @memset(block, 1);
                    block[0] = 1024;
                }
            }
            if (fill == .nonfinite) self.blocks_vals[slot * p.hidden] = std.math.nan(f32);
            round(self.blocks_vals);
            self.refs[0] = try self.input(self.blocks_vals, true);
            self.out_ar = try self.rig.output(self.count() * 4);
            writeF32s(self.rig.bytes(self.out_ar, self.count() * 4), 1);
            return true;
        }
        if (self.kind == .cached_ar) {
            const q_count = n / p.hidden;
            self.blocks_vals = try self.floats((r - 1) * p.hidden, rng);
            self.p_vals = try self.floats(p.hidden, rng);
            self.queries = try self.floats(q_count * p.hidden, rng);
            if (fill == .stress) {
                for (0..r) |s| @memset(if (s + 1 == r) self.p_vals else self.blocks_vals[s * p.hidden ..][0..p.hidden], if (s % 2 == 0) 256 else -256);
                fillStressQueries(self.queries, q_count, rng);
            } else if (fill == .random and q_count == 3 and (r == 3 or r == 5)) {
                if (r == 3) {
                    @memset(self.blocks_vals[0..p.hidden], 0);
                    self.blocks_vals[1] = -0.0;
                    @memset(self.blocks_vals[p.hidden .. 2 * p.hidden], 1e-18);
                    for (self.p_vals, 0..) |*v, d| v.* = if (d % 3 == 0) 1e6 else if (d % 3 == 1) -1e6 else 1e-5;
                } else {
                    @memset(self.blocks_vals[0..p.hidden], 256);
                    @memset(self.blocks_vals[p.hidden .. 2 * p.hidden], -0.0078125);
                    @memset(self.blocks_vals[2 * p.hidden .. 3 * p.hidden], 1e8);
                    @memset(self.blocks_vals[3 * p.hidden .. 4 * p.hidden], 1e-20);
                    for (self.p_vals, 0..) |*v, d| v.* = if (d % 3 == 0) 256 else if (d % 3 == 1) -256 else 0.5;
                }
            }
            if (fill == .nonfinite) self.queries[0] = std.math.inf(f32);
            round(self.blocks_vals);
            const blocks_bits = try self.arena.allocator().alloc(u16, 4 * p.hidden);
            @memset(blocks_bits, c.toBf16(123));
            for (self.blocks_vals, 0..) |x, i| blocks_bits[i] = c.toBf16(x);
            self.refs[0] = try self.rig.input(std.mem.sliceAsBytes(blocks_bits));
            self.refs[1] = try self.input(self.p_vals, false);
            self.refs[2] = try self.input(self.queries, false);
            self.base_inv_rms = try self.rig.scratch(4 * 4);
            writeF32s(self.rig.bytes(self.base_inv_rms, 4 * 4), 999);
            self.base_ar = try self.rig.scratch(q_count * p.hidden * 4);
            self.out_ar = try self.rig.output(q_count * p.hidden * 4);
            return true;
        }
        self.p_vals = try self.floats(r * p.hidden, rng);
        self.x_vals = try self.floats(r * p.hidden, rng);
        self.blocks_vals = try self.floats(r * 4 * p.hidden, rng);
        if (self.kind == .attn_partial_ar) {
            const srcs = n / p.hidden;
            const boundary = shape.k;
            self.queries = try self.floats(p.hidden, rng);
            if (fill == .stress) {
                fillStressBlocks(self.blocks_vals, r);
                fillStressQueries(self.queries, 1, rng);
                const p_sign: f32 = if (boundary != 0)
                    (if ((boundary - 1) % 2 == 0) 256 else -256)
                else
                    (if ((srcs - 1) % 2 == 0) 128 else -128);
                const x_sign: f32 = if (boundary != 0)
                    (if ((srcs - 1) % 2 == 0) 256 else -256)
                else
                    (if ((srcs - 1) % 2 == 0) 128 else -128);
                @memset(self.p_vals, p_sign);
                @memset(self.x_vals, x_sign);
            }
            round(self.p_vals);
            round(self.x_vals);
            round(self.blocks_vals);
            if (fill == .nonfinite) {
                self.x_vals[0] = std.math.nan(f32);
                if (boundary != 0) self.p_vals[1] = std.math.nan(f32);
            }
            self.refs[0] = try self.input(self.p_vals, true);
            self.refs[1] = try self.input(self.x_vals, true);
            self.refs[2] = try self.input(self.queries, false);
            self.blocks_init_u16 = try self.arena.allocator().alloc(u16, r * 4 * p.hidden);
            for (self.blocks_init_u16, self.blocks_vals) |*b, x| b.* = c.toBf16(x);
            self.blocks = try self.rig.scratch(r * 4 * p.hidden * 2);
            @memcpy(self.rig.bytes(self.blocks, r * 4 * p.hidden * 2), std.mem.sliceAsBytes(self.blocks_init_u16));
            self.base_partial = try self.rig.scratch(r * p.hidden * 4);
            self.base_ar = try self.rig.scratch(r * p.hidden * 4);
            self.base_inv_rms = try self.rig.scratch(r * 4 * 4);
            const bf_len = r * 4 * p.hidden * 2;
            self.out_bytes = try self.arena.allocator().alloc(u8, self.count() * 4 + bf_len);
            self.base_bytes = try self.arena.allocator().alloc(u8, self.count() * 4 + bf_len);
            self.inv_rms_init = try self.computeInvRms(self.blocks, r);
            if (boundary != 0) for (0..r) |row| {
                self.inv_rms_init[row * 4 + boundary - 1] = 999;
            };
            self.out_partial = try self.rig.output(r * p.hidden * 4);
            self.out_ar = try self.rig.output(r * p.hidden * 4);
            self.out_inv_rms = try self.rig.output(r * 4 * 4);
            @memcpy(self.rig.bytes(self.out_inv_rms, r * 4 * 4), std.mem.sliceAsBytes(self.inv_rms_init));
        } else {
            const q_count = n / p.hidden;
            const srcs = shape.k;
            self.queries = try self.floats(q_count * p.hidden, rng);
            if (fill == .stress) {
                fillStressBlocks(self.blocks_vals, r);
                fillStressQueries(self.queries, q_count, rng);
                const sign: f32 = if ((srcs - 1) % 2 == 0) 128 else -128;
                @memset(self.p_vals, sign);
                @memset(self.x_vals, sign);
            }
            round(self.x_vals);
            round(self.blocks_vals);
            if (fill == .nonfinite) self.x_vals[0] = std.math.nan(f32);
            self.refs[0] = try self.input(self.p_vals, false);
            self.refs[1] = try self.input(self.x_vals, true);
            self.refs[2] = try self.input(self.blocks_vals, true);
            self.base_inv_rms = try self.rig.scratch(r * 4 * 4);
            self.inv_rms_init = try self.computeInvRms(self.refs[2], r);
            for (0..r) |row| for (srcs - 1..4) |s| {
                self.inv_rms_init[row * 4 + s] = 999;
            };
            self.refs[3] = try self.input(self.inv_rms_init, false);
            self.refs[4] = try self.input(self.queries, false);
            self.p_out = try self.rig.scratch(r * p.hidden * 2);
            self.base_p = try self.rig.scratch(r * p.hidden * 2);
            self.base_partial = try self.rig.scratch(r * p.hidden * 4);
            self.base_ar = try self.rig.scratch(q_count * r * p.hidden * 4);
            const bf_len = r * p.hidden * 2;
            self.out_bytes = try self.arena.allocator().alloc(u8, self.count() * 4 + bf_len);
            self.base_bytes = try self.arena.allocator().alloc(u8, self.count() * 4 + bf_len);
            self.out_partial = try self.rig.output(r * p.hidden * 4);
            self.out_ar = try self.rig.output(q_count * r * p.hidden * 4);
        }
        return true;
    }

    pub fn run(self: *Probe) !void {
        const pass = self.rig.begin();
        const r: u32 = @intCast(self.shape.rows);
        const n: u32 = @intCast(self.shape.n);
        const k: u32 = @intCast(self.shape.k);
        switch (self.kind) {
            .block_inv_rms => host.blockInvRms(&self.pipes, pass.e, at(self.refs[0]), at(self.out_ar), r, n - 1),
            .cached_ar => {
                for (0..r - 1) |s| host.blockInvRms(&self.pipes, pass.e, at(self.refs[0]), at(self.base_inv_rms), 1, @intCast(s));
                host.cachedAr(&self.pipes, pass.e, at(self.refs[0]), at(self.refs[1]), at(self.base_inv_rms), at(self.refs[2]), at(self.out_ar), r, n / p.hidden);
            },
            .attn_partial_ar => host.attnPartialAr(&self.pipes, pass.e, at(self.refs[0]), at(self.refs[1]), at(self.blocks), at(self.out_inv_rms), at(self.out_partial), at(self.refs[2]), at(self.out_ar), r, n / p.hidden, k),
            .mlp_partial_ar => host.mlpPartialAr(&self.pipes, pass.e, at(self.refs[0]), at(self.refs[1]), at(self.refs[2]), at(self.refs[3]), at(self.refs[4]), at(self.p_out), at(self.out_partial), at(self.out_ar), r, k, n / p.hidden),
            else => unreachable,
        }
        try self.rig.finish(pass);
    }

    pub fn output(self: *Probe) ![]const u8 {
        const fp32 = try self.rig.outputBytes();
        if (self.kind == .cached_ar or self.kind == .block_inv_rms) return fp32;
        const r = self.shape.rows;
        const bf_ref = if (self.kind == .attn_partial_ar) self.blocks else self.p_out;
        const bf_len = if (self.kind == .attn_partial_ar) r * 4 * p.hidden * 2 else r * p.hidden * 2;
        @memcpy(self.out_bytes[0..fp32.len], fp32);
        @memcpy(self.out_bytes[fp32.len..][0..bf_len], self.rig.bytes(bf_ref, bf_len));
        return self.out_bytes;
    }

    pub fn replaced(self: *Probe) ![]const u8 {
        const base = &(self.base orelse return error.NoBaseline);
        const r = self.shape.rows;
        const n = self.shape.n;
        const k = self.shape.k;
        if (self.kind == .cached_ar) {
            const len = self.count() * 4;
            @memset(self.rig.bytes(self.base_ar, len), 0xff);
            const pass = self.rig.begin();
            host.ar(base, pass.e, at(self.refs[0]), at(self.refs[1]), at(self.refs[2]), at(self.base_ar), @intCast(r), @intCast(n / p.hidden), .{});
            try self.rig.finish(pass);
            return self.rig.bytes(self.base_ar, len);
        }
        const rt = &(self.rt orelse return error.NoBaseline);
        if (self.kind == .attn_partial_ar) {
            const srcs: u32 = @intCast(n / p.hidden);
            @memcpy(self.rig.bytes(self.blocks, r * 4 * p.hidden * 2), std.mem.sliceAsBytes(self.blocks_init_u16));
            @memcpy(self.rig.bytes(self.base_inv_rms, r * 4 * 4), std.mem.sliceAsBytes(self.inv_rms_init));
            const pass = self.rig.begin();
            dispatchRt(rt.attn_partial, pass.e, &.{ at(self.refs[0]), at(self.refs[1]), at(self.blocks), at(self.base_partial) }, r * p.hidden, k);
            if (k != 0) host.blockInvRms(base, pass.e, at(self.blocks), at(self.base_inv_rms), @intCast(r), @intCast(k - 1));
            for (0..r) |row| {
                host.cachedAr(base, pass.e, atOff(self.blocks, row * 4 * p.hidden * 2), atOff(self.base_partial, row * p.hidden * 4), atOff(self.base_inv_rms, row * 4 * 4), at(self.refs[2]), atOff(self.base_ar, row * p.hidden * 4), srcs, 1);
            }
            try self.rig.finish(pass);
            const part_bytes = r * p.hidden * 4;
            const inv_bytes = r * 4 * 4;
            const bf_bytes = r * 4 * p.hidden * 2;
            const dst = self.base_bytes;
            @memcpy(dst[0..part_bytes], self.rig.bytes(self.base_partial, part_bytes));
            @memcpy(dst[part_bytes..][0..part_bytes], self.rig.bytes(self.base_ar, part_bytes));
            @memcpy(dst[2 * part_bytes ..][0..inv_bytes], self.rig.bytes(self.base_inv_rms, inv_bytes));
            @memcpy(dst[2 * part_bytes + inv_bytes ..][0..bf_bytes], self.rig.bytes(self.blocks, bf_bytes));
            return dst;
        }
        const q_count = n / p.hidden;
        const pass = self.rig.begin();
        dispatchRt(rt.mlp_partial, pass.e, &.{ at(self.refs[0]), at(self.refs[1]), at(self.base_p) }, r * p.hidden, 0);
        dispatchRt(rt.widen, pass.e, &.{ at(self.base_p), at(self.base_partial) }, r * p.hidden, 0);
        for (0..q_count) |q| for (0..r) |row| {
            host.cachedAr(base, pass.e, atOff(self.refs[2], row * 4 * p.hidden * 2), atOff(self.base_partial, row * p.hidden * 4), atOff(self.refs[3], row * 4 * 4), atOff(self.refs[4], q * p.hidden * 4), atOff(self.base_ar, (q * r + row) * p.hidden * 4), @intCast(k), 1);
        };
        try self.rig.finish(pass);
        const part_bytes = r * p.hidden * 4;
        const ar_bytes = q_count * part_bytes;
        const bf_bytes = r * p.hidden * 2;
        const dst = self.base_bytes;
        @memcpy(dst[0..part_bytes], self.rig.bytes(self.base_partial, part_bytes));
        @memcpy(dst[part_bytes..][0..ar_bytes], self.rig.bytes(self.base_ar, ar_bytes));
        @memcpy(dst[part_bytes + ar_bytes ..][0..bf_bytes], self.rig.bytes(self.base_p, bf_bytes));
        return dst;
    }

    fn evaluate(self: *Probe, sim: ?p.ArSum, want: []f64) !void {
        const r = self.shape.rows;
        const n = self.shape.n;
        const k = self.shape.k;
        if (self.kind == .block_inv_rms) {
            const slot = n - 1;
            for (0..r) |row| for (0..4) |s| {
                want[row * 4 + s] = if (s == slot) p.blockInvRmsIn(self.blocks_vals[(row * 4 + s) * p.hidden ..][0..p.hidden], sim) else 1;
            };
            return;
        }
        if (self.kind == .cached_ar) {
            for (0..n / p.hidden) |q| p.blockResidualIn(self.blocks_vals, self.p_vals, self.queries, r, q, sim, want[q * p.hidden ..][0..p.hidden]);
            return;
        }
        const part_n = r * p.hidden;
        if (self.kind == .attn_partial_ar) {
            const srcs = n / p.hidden;
            const row_blocks = try self.a.alloc(f32, 4 * p.hidden);
            defer self.a.free(row_blocks);
            for (0..r) |row| {
                @memcpy(row_blocks, self.blocks_vals[row * 4 * p.hidden ..][0 .. 4 * p.hidden]);
                if (k != 0) @memcpy(row_blocks[(k - 1) * p.hidden ..][0..p.hidden], self.p_vals[row * p.hidden ..][0..p.hidden]);
                var partial_f32: [p.hidden]f32 = undefined;
                for (0..p.hidden) |d| {
                    const pv = self.p_vals[row * p.hidden + d];
                    const xv = self.x_vals[row * p.hidden + d];
                    const val: f32 = if (k != 0) xv else pv + xv;
                    partial_f32[d] = val;
                    want[row * p.hidden + d] = val;
                }
                p.blockResidualIn(row_blocks, &partial_f32, self.queries, srcs, 0, sim, want[part_n + row * p.hidden ..][0..p.hidden]);
                for (0..4) |s| want[2 * part_n + row * 4 + s] = if (k != 0 and s == k - 1)
                    p.blockInvRmsIn(self.p_vals[row * p.hidden ..][0..p.hidden], sim)
                else
                    self.inv_rms_init[row * 4 + s];
            }
            return;
        }
        const q_count = n / p.hidden;
        for (0..r) |row| {
            var partial_f32: [p.hidden]f32 = undefined;
            for (0..p.hidden) |d| {
                const v: f32 = @floatCast(p.roundBf16(@as(f32, self.p_vals[row * p.hidden + d] + self.x_vals[row * p.hidden + d])));
                partial_f32[d] = v;
                want[row * p.hidden + d] = v;
            }
            const row_blocks = self.blocks_vals[row * 4 * p.hidden ..][0 .. 4 * p.hidden];
            for (0..q_count) |q| {
                p.blockResidualIn(row_blocks, &partial_f32, self.queries, k, q, sim, want[part_n + (q * r + row) * p.hidden ..][0..p.hidden]);
            }
        }
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        try self.evaluate(null, want);
        for (mag, want) |*m, w| m.* = 10 * @abs(w) + 1e-5;
    }

    pub fn simulate(self: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.evaluate(arSim(acc, order), out);
    }
};
