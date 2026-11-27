//! Byte-exact contract probes for Inkling 2 Mini's BF16 RMSNorm and fused AR-Norm glue kernels.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const fused = @import("metal_inkling2_glue_fused.zig");
const host = @import("tensorfold").inkling2_metal_host.glue;
const p = host.plan;
const sources = @import("inkling2_metal");
const kernel_sources = @import("kernel_sources");

const at = fused.at;
const value = fused.value;
const round = fused.round;

const RuntimeArgs = extern struct { count: u32, boundary: u32 = 0 };
const NormArgs = extern struct { rows: u32, width: u32, heads: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 };

fn dispatchWiden(pipe: mtl.Pipeline, e: mtl.ComputeEncoder, src: host.Ref, dst: host.Ref, count_val: usize) void {
    e.setPipeline(pipe);
    e.setBuffer(src.buf, src.off, 0);
    e.setBuffer(dst.buf, dst.off, 1);
    e.setValue(RuntimeArgs{ .count = @intCast(count_val) }, 2);
    e.dispatchThreads(mtl.Size.of(count_val, 1, 1), mtl.Size.of(256, 1, 1));
}

fn dispatchSpNorm(pipe: mtl.Pipeline, e: mtl.ComputeEncoder, src: host.Ref, w: host.Ref, dst: host.Ref, rows: usize) void {
    e.setPipeline(pipe);
    e.setBuffer(src.buf, src.off, 0);
    e.setBuffer(w.buf, w.off, 1);
    e.setValue(NormArgs{ .rows = @intCast(rows), .width = p.hidden, .eps = 1e-6 }, 2);
    e.setBuffer(dst.buf, dst.off, 3);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

const norm_served = [_]c.Shape{
    .{ .rows = 1, .n = p.hidden, .k = p.hidden },
    .{ .rows = 5, .n = p.hidden, .k = p.hidden },
    .{ .rows = 7, .n = p.hidden, .k = p.hidden },
};
const attn_norm_served = blk: {
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
const mlp_norm_served = [_]c.Shape{
    .{ .rows = 4, .n = p.hidden, .k = 5 },
    .{ .rows = 1, .n = p.hidden, .k = 1 },
    .{ .rows = 5, .n = p.hidden, .k = 2 },
    .{ .rows = 7, .n = p.hidden, .k = 3 },
    .{ .rows = 8, .n = p.hidden, .k = 4 },
};

const BasePipes = struct {
    glue: host.Pipelines,
    widen: mtl.Pipeline,
    sp_norm: mtl.Pipeline,

    fn load(device: mtl.Device) !BasePipes {
        var gp = try host.Pipelines.load(device);
        errdefer gp.deinit();
        const rt_lib = try mtl.Library.fromSource(device, sources.runtime, mtl.CompileOptions.mlx());
        defer rt_lib.deinit();
        var widen = try mtl.Pipeline.init(device, rt_lib, "tf_ink2_widen", false);
        errdefer widen.deinit();
        const sp_lib = try mtl.Library.fromSource(device, kernel_sources.spark25_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer sp_lib.deinit();
        return .{ .glue = gp, .widen = widen, .sp_norm = try mtl.Pipeline.init(device, sp_lib, "sp_norm", false) };
    }
    fn deinit(b: *BasePipes) void {
        b.sp_norm.deinit();
        b.widen.deinit();
        b.glue.deinit();
    }
};

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: p.Kind,
    pipes: host.Pipelines,
    base: BasePipes,
    shape: c.Shape = undefined,
    p_vals: []f32 = &.{},
    x_vals: []f32 = &.{},
    blocks_vals: []f32 = &.{},
    queries: []f32 = &.{},
    norm_w: []f32 = &.{},
    blocks_init_u16: []u16 = &.{},
    inv_rms_init: []f32 = &.{},
    refs: [6]rig_.Ref = undefined,
    blocks: rig_.Ref = undefined,
    p_out: rig_.Ref = undefined,
    out_norm: rig_.Ref = undefined,
    out_partial: rig_.Ref = undefined,
    out_inv_rms: rig_.Ref = undefined,
    base_p: rig_.Ref = undefined,
    base_partial: rig_.Ref = undefined,
    base_ar: rig_.Ref = undefined,
    base_inv_rms: rig_.Ref = undefined,
    base_norm: rig_.Ref = undefined,
    out_bytes: []u8 = &.{},
    base_bytes: []u8 = &.{},

    pub fn initSource(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: p.Kind, text: []const u8) !Probe {
        var pipes = try host.Pipelines.loadSource(device, text);
        errdefer pipes.deinit();
        var base = try BasePipes.load(device);
        errdefer base.deinit();
        return .{
            .a = a,
            .arena = .init(a),
            .rig = .{ .a = a, .device = device, .queue = queue },
            .kind = kind,
            .pipes = pipes,
            .base = base,
        };
    }

    pub fn deinit(self: *Probe) void {
        self.rig.deinit();
        self.arena.deinit();
        self.base.deinit();
        self.pipes.deinit();
    }

    const vt = blk: {
        var hooks = g.hooks(Probe);
        hooks.output = outputHook;
        hooks.check = null;
        break :blk hooks;
    };
    fn outputHook(ctx: *anyopaque) anyerror![]const u8 {
        const self: *Probe = @ptrCast(@alignCast(ctx));
        return self.output();
    }

    pub fn kernel(self: *Probe) g.Kernel {
        const replaces: []const u8 = switch (self.kind) {
            .norm_bf16 => "tf_ink2_widen + sp_norm",
            .attn_partial_ar_norm => "tf_inkling2_attn_partial_ar + sp_norm",
            .mlp_partial_ar_norm => "tf_inkling2_mlp_partial_ar + sp_norm",
            else => unreachable,
        };
        const served: []const c.Shape = switch (self.kind) {
            .norm_bf16 => &norm_served,
            .attn_partial_ar_norm => &attn_norm_served,
            .mlp_partial_ar_norm => &mlp_norm_served,
            else => unreachable,
        };
        return .{
            .ctx = self,
            .vt = &vt,
            .spec = .{
                .name = self.kind.label(),
                .accum = .f32,
                .depth = fused.depth,
                .ops = 96,
                .out = .bf16,
                .replaces = replaces,
                .served = served,
                .sums = .{ .f16 = 0, .bf16 = 0, .why = "BF16 RMS norm endpoint has no uniform lower-precision separability factor" },
            },
        };
    }

    pub fn count(self: *Probe) usize {
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
        const pass = self.rig.begin();
        for (0..4) |slot| host.blockInvRms(&self.base.glue, pass.e, at(blocks_ref), at(self.base_inv_rms), @intCast(r), @intCast(slot));
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
        self.norm_w = try self.floats(p.hidden, rng);
        round(self.norm_w);
        if (self.kind == .norm_bf16) {
            self.x_vals = try self.floats(r * p.hidden, rng);
            if (fill == .stress) @memset(self.x_vals, 256);
            if (fill == .nonfinite) self.x_vals[0] = std.math.nan(f32);
            round(self.x_vals);
            self.refs[0] = try self.input(self.x_vals, true);
            self.refs[1] = try self.input(self.norm_w, true);
            self.base_ar = try self.rig.scratch(r * p.hidden * 4);
            self.base_norm = try self.rig.scratch(r * p.hidden * 2);
            self.out_norm = try self.rig.output(r * p.hidden * 2);
            return true;
        }
        self.p_vals = try self.floats(r * p.hidden, rng);
        self.x_vals = try self.floats(r * p.hidden, rng);
        self.blocks_vals = try self.floats(r * 4 * p.hidden, rng);
        self.queries = try self.floats(p.hidden, rng);
        if (self.kind == .attn_partial_ar_norm) {
            const srcs = n / p.hidden;
            const boundary = shape.k;
            if (fill == .stress) {
                fillStressBlocks(self.blocks_vals, r);
                fused.fillStressQueries(self.queries, 1, rng);
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
            self.refs[3] = try self.input(self.norm_w, true);
            self.blocks_init_u16 = try self.arena.allocator().alloc(u16, r * 4 * p.hidden);
            for (self.blocks_init_u16, self.blocks_vals) |*b, x| b.* = c.toBf16(x);
            self.blocks = try self.rig.scratch(r * 4 * p.hidden * 2);
            @memcpy(self.rig.bytes(self.blocks, r * 4 * p.hidden * 2), std.mem.sliceAsBytes(self.blocks_init_u16));
            self.base_partial = try self.rig.scratch(r * p.hidden * 4);
            self.base_ar = try self.rig.scratch(r * p.hidden * 4);
            self.base_inv_rms = try self.rig.scratch(r * 4 * 4);
            self.base_norm = try self.rig.scratch(r * p.hidden * 2);
            self.inv_rms_init = try self.computeInvRms(self.blocks, r);
            if (boundary != 0) for (0..r) |row| {
                self.inv_rms_init[row * 4 + boundary - 1] = 999;
            };
            self.out_norm = try self.rig.output(r * p.hidden * 2);
            self.out_partial = try self.rig.output(r * p.hidden * 4);
            self.out_inv_rms = try self.rig.output(r * 4 * 4);
            @memcpy(self.rig.bytes(self.out_inv_rms, r * 4 * 4), std.mem.sliceAsBytes(self.inv_rms_init));
            const total_bytes = r * p.hidden * 2 + r * p.hidden * 4 + r * 4 * 4 + r * 4 * p.hidden * 2;
            self.out_bytes = try self.arena.allocator().alloc(u8, total_bytes);
            self.base_bytes = try self.arena.allocator().alloc(u8, total_bytes);
            return true;
        }
        const srcs = shape.k;
        if (fill == .stress) {
            fillStressBlocks(self.blocks_vals, r);
            fused.fillStressQueries(self.queries, 1, rng);
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
        self.refs[5] = try self.input(self.norm_w, true);
        self.p_out = try self.rig.scratch(r * p.hidden * 2);
        self.base_p = try self.rig.scratch(r * p.hidden * 2);
        self.base_partial = try self.rig.scratch(r * p.hidden * 4);
        self.base_ar = try self.rig.scratch(r * p.hidden * 4);
        self.base_norm = try self.rig.scratch(r * p.hidden * 2);
        self.out_norm = try self.rig.output(r * p.hidden * 2);
        self.out_partial = try self.rig.output(r * p.hidden * 4);
        const total_bytes = r * p.hidden * 2 + r * p.hidden * 4 + r * p.hidden * 2;
        self.out_bytes = try self.arena.allocator().alloc(u8, total_bytes);
        self.base_bytes = try self.arena.allocator().alloc(u8, total_bytes);
        return true;
    }

    pub fn run(self: *Probe) !void {
        const pass = self.rig.begin();
        const r: u32 = @intCast(self.shape.rows);
        const n: u32 = @intCast(self.shape.n);
        const k: u32 = @intCast(self.shape.k);
        switch (self.kind) {
            .norm_bf16 => host.normBf16(&self.pipes, pass.e, at(self.refs[0]), at(self.refs[1]), at(self.out_norm), r),
            .attn_partial_ar_norm => host.attnPartialArNorm(&self.pipes, pass.e, at(self.refs[0]), at(self.refs[1]), at(self.blocks), at(self.out_inv_rms), at(self.out_partial), at(self.refs[2]), at(self.refs[3]), at(self.out_norm), r, n / p.hidden, k),
            .mlp_partial_ar_norm => host.mlpPartialArNorm(&self.pipes, pass.e, at(self.refs[0]), at(self.refs[1]), at(self.refs[2]), at(self.refs[3]), at(self.refs[4]), at(self.refs[5]), at(self.p_out), at(self.out_partial), at(self.out_norm), r, k),
            else => unreachable,
        }
        try self.rig.finish(pass);
    }

    pub fn output(self: *Probe) ![]const u8 {
        const outs = try self.rig.outputBytes();
        if (self.kind == .norm_bf16) return outs;
        const r = self.shape.rows;
        const bf_ref = if (self.kind == .attn_partial_ar_norm) self.blocks else self.p_out;
        const bf_len = if (self.kind == .attn_partial_ar_norm) r * 4 * p.hidden * 2 else r * p.hidden * 2;
        @memcpy(self.out_bytes[0..outs.len], outs);
        @memcpy(self.out_bytes[outs.len..][0..bf_len], self.rig.bytes(bf_ref, bf_len));
        return self.out_bytes;
    }

    pub fn replaced(self: *Probe) ![]const u8 {
        const r = self.shape.rows;
        const n = self.shape.n;
        const k = self.shape.k;
        if (self.kind == .norm_bf16) {
            const pass = self.rig.begin();
            dispatchWiden(self.base.widen, pass.e, at(self.refs[0]), at(self.base_ar), r * p.hidden);
            dispatchSpNorm(self.base.sp_norm, pass.e, at(self.base_ar), at(self.refs[1]), at(self.base_norm), r);
            try self.rig.finish(pass);
            return self.rig.bytes(self.base_norm, r * p.hidden * 2);
        }
        if (self.kind == .attn_partial_ar_norm) {
            const srcs: u32 = @intCast(n / p.hidden);
            @memcpy(self.rig.bytes(self.blocks, r * 4 * p.hidden * 2), std.mem.sliceAsBytes(self.blocks_init_u16));
            @memcpy(self.rig.bytes(self.base_inv_rms, r * 4 * 4), std.mem.sliceAsBytes(self.inv_rms_init));
            const pass = self.rig.begin();
            host.attnPartialAr(&self.base.glue, pass.e, at(self.refs[0]), at(self.refs[1]), at(self.blocks), at(self.base_inv_rms), at(self.base_partial), at(self.refs[2]), at(self.base_ar), @intCast(r), srcs, @intCast(k));
            dispatchSpNorm(self.base.sp_norm, pass.e, at(self.base_ar), at(self.refs[3]), at(self.base_norm), r);
            try self.rig.finish(pass);
            const norm_bytes = r * p.hidden * 2;
            const part_bytes = r * p.hidden * 4;
            const inv_bytes = r * 4 * 4;
            const bf_bytes = r * 4 * p.hidden * 2;
            const dst = self.base_bytes;
            @memcpy(dst[0..norm_bytes], self.rig.bytes(self.base_norm, norm_bytes));
            @memcpy(dst[norm_bytes..][0..part_bytes], self.rig.bytes(self.base_partial, part_bytes));
            @memcpy(dst[norm_bytes + part_bytes ..][0..inv_bytes], self.rig.bytes(self.base_inv_rms, inv_bytes));
            @memcpy(dst[norm_bytes + part_bytes + inv_bytes ..][0..bf_bytes], self.rig.bytes(self.blocks, bf_bytes));
            return dst;
        }
        const pass = self.rig.begin();
        host.mlpPartialAr(&self.base.glue, pass.e, at(self.refs[0]), at(self.refs[1]), at(self.refs[2]), at(self.refs[3]), at(self.refs[4]), at(self.base_p), at(self.base_partial), at(self.base_ar), @intCast(r), @intCast(k), 1);
        dispatchSpNorm(self.base.sp_norm, pass.e, at(self.base_ar), at(self.refs[5]), at(self.base_norm), r);
        try self.rig.finish(pass);
        const norm_bytes = r * p.hidden * 2;
        const part_bytes = r * p.hidden * 4;
        const bf_bytes = r * p.hidden * 2;
        const dst = self.base_bytes;
        @memcpy(dst[0..norm_bytes], self.rig.bytes(self.base_norm, norm_bytes));
        @memcpy(dst[norm_bytes..][0..part_bytes], self.rig.bytes(self.base_partial, part_bytes));
        @memcpy(dst[norm_bytes + part_bytes ..][0..bf_bytes], self.rig.bytes(self.base_p, bf_bytes));
        return dst;
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        const r = self.shape.rows;
        const n = self.shape.n;
        const k = self.shape.k;
        var ar_row: [p.hidden]f64 = undefined;
        if (self.kind == .norm_bf16) {
            for (0..r) |row| {
                for (0..p.hidden) |d| ar_row[d] = self.x_vals[row * p.hidden + d];
                p.rmsNormRow(&ar_row, self.norm_w, want[row * p.hidden ..][0..p.hidden]);
            }
        } else if (self.kind == .attn_partial_ar_norm) {
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
                    partial_f32[d] = if (k != 0) xv else pv + xv;
                }
                p.blockResidual(row_blocks, &partial_f32, self.queries, srcs, 0, &ar_row);
                p.rmsNormRow(&ar_row, self.norm_w, want[row * p.hidden ..][0..p.hidden]);
            }
        } else {
            for (0..r) |row| {
                var partial_f32: [p.hidden]f32 = undefined;
                for (0..p.hidden) |d| {
                    partial_f32[d] = @floatCast(p.roundBf16(@as(f32, self.p_vals[row * p.hidden + d] + self.x_vals[row * p.hidden + d])));
                }
                const row_blocks = self.blocks_vals[row * 4 * p.hidden ..][0 .. 4 * p.hidden];
                p.blockResidual(row_blocks, &partial_f32, self.queries, k, 0, &ar_row);
                p.rmsNormRow(&ar_row, self.norm_w, want[row * p.hidden ..][0..p.hidden]);
            }
        }
        for (mag, want) |*m, w| m.* = 10 * @abs(w) + 1e-5;
    }

    pub fn simulate(self: *Probe, _: c.Accum, _: g.Order, out: []f64) !void {
        const mag = try self.a.alloc(f64, out.len);
        defer self.a.free(mag);
        return self.reference(out, mag);
    }
};
