//! The mla image (mla.cu) under the contract: its seven launches through gates 1-4, and a planted control for each.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const proj_k = @import("contract_mla_proj.zig");
const chunks_k = @import("contract_mla_chunks.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

pub const Kind = enum { gate, rope };

/// A planted fault, each a control red at gate 3: the next head's gate column, the next position, no YaRN factor.
pub const Fault = enum { none, next_column, next_position, no_factor };

pub const Entry = struct { name: []const u8, kind: Kind, scale: f32 = 1, pos: i32 = 0, fault: Fault = .none };

const yarn: f32 = 1.138629436111989; // 0.1 ln(4) + 1: YaRN's attention factor at a factor of 4

pub const entries = [_]Entry{
    .{ .name = "cuda mla tf_mla_gate (head-wise sigmoid gate, in place)", .kind = .gate },
    .{ .name = "cuda mla tf_mla_rope (plain, positions from 1,000)", .kind = .rope, .pos = 1000 },
    .{ .name = "cuda mla tf_mla_rope (YaRN's attention factor, positions from 900,000)", .kind = .rope, .scale = yarn, .pos = 900000 },
    .{ .name = "control: mla tf_mla_gate gates each head by the next head's column", .kind = .gate, .fault = .next_column },
    .{ .name = "control: mla tf_mla_rope (plain) turns each row by the next position's angle", .kind = .rope, .pos = 1000, .fault = .next_position },
    .{ .name = "control: mla tf_mla_rope (YaRN) leaves the attention factor out", .kind = .rope, .scale = yarn, .pos = 900000, .fault = .no_factor },
};

const head_w = 192; // rope: q heads 192 wide, the 64 rope columns from column 128
const rope_at = 128;
const gate_shapes = [_]c.Shape{ .{ .rows = 1, .n = 32, .k = 128 }, .{ .rows = 40, .n = 64, .k = 128 } };
const rope_shapes = [_]c.Shape{ .{ .rows = 1, .n = 32, .k = 64 }, .{ .rows = 40, .n = 64, .k = 64 } };

fn depthNone(_: c.Shape) usize {
    return 0;
}

pub fn spec(e: Entry) c.Spec {
    return switch (e.kind) {
        // exp (2 ulp), the add and the division: sigmoid within 8 units; bf16(sigmoid) is the mid rounding
        .gate => .{ .name = e.name, .accum = .none, .depth = depthNone, .ops = 8, .mids = 1, .out = .bf16, .in_place = true, .served = &gate_shapes },
        .rope => .{ .name = e.name, .accum = .none, .depth = depthNone, .out = .bf16, .served = &rope_shapes },
    };
}

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

/// tf_mla_gate and tf_mla_rope as cuda.mla.Mla.gate and .rope launch them.
pub const Rows = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    shape: c.Shape = undefined,
    a16: []u16 = &.{}, // gate g, rope x
    b16: []u16 = &.{}, // gate o (start)
    freqs: [32]f32 = undefined, // Ling-3.1's inv_freq, theta 6e6
    extra: []f64 = &.{}, // per output: the allowance beyond the spec's bound
    out: u64 = 0,
    in: [3]u64 = @splat(0),

    pub fn deinit(self: *Rows) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Rows) void {
        const a = self.rig.gpu.gpa;
        a.free(self.a16);
        a.free(self.b16);
        a.free(self.extra);
        self.a16 = &.{};
        self.b16 = &.{};
        self.extra = &.{};
    }

    pub fn kernel(self: *Rows) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.fault != .none) .shapes else null };
    }

    const vt = g.hooks(Rows);

    pub fn count(self: *const Rows) usize {
        const s = self.shape;
        return if (self.entry.kind == .gate) s.rows * s.n * s.k else s.rows * s.n * 64;
    }

    pub fn prepare(self: *Rows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        const n = shape.n;
        const sym = struct {
            fn v(r: std.Random, w: f32) u16 {
                return c.toBf16((r.float(f32) * 2 - 1) * w);
            }
        };
        switch (self.entry.kind) {
            .gate => {
                const gs = n + 3;
                self.a16 = try a.alloc(u16, rows * gs);
                self.b16 = try a.alloc(u16, rows * n * shape.k);
                for (self.a16) |*x| x.* = sym.v(rng, 6);
                for (self.b16) |*x| x.* = sym.v(rng, 3);
                if (fill == .nonfinite) {
                    self.b16[5] = c.toBf16(std.math.nan(f32));
                    self.b16[n * shape.k + 7] = c.toBf16(std.math.inf(f32));
                    self.a16[2 * gs + 1] = c.toBf16(-std.math.inf(f32));
                    self.a16[3 * gs + 2] = c.toBf16(std.math.nan(f32));
                }
                self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.a16));
                self.out = try self.rig.output(self.b16.len * 2);
            },
            .rope => {
                self.a16 = try a.alloc(u16, rows * n * head_w);
                for (self.a16) |*x| x.* = sym.v(rng, 4);
                if (fill == .nonfinite) {
                    self.a16[rope_at + 4] = c.toBf16(std.math.nan(f32));
                    self.a16[n * head_w + rope_at + 9] = c.toBf16(std.math.inf(f32));
                }
                self.freqs = cuda.mla.ropeFreqs(6.0e6);
                const pos: i32 = self.entry.pos + @intFromBool(self.entry.fault == .next_position);
                self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.a16));
                self.in[1] = try self.rig.input(std.mem.asBytes(&pos));
                self.in[2] = try self.rig.input(std.mem.sliceAsBytes(&self.freqs));
                self.out = try self.rig.output(rows * n * 64 * 2);
            },
        }
        return true;
    }

    pub fn run(self: *Rows) !void {
        const s = self.shape;
        const e = self.entry;
        var a: cuda.Args = .{};
        const st = self.rig.s;
        switch (e.kind) {
            .gate => {
                try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.out, .len = self.b16.len * 2 }).upload(0, std.mem.sliceAsBytes(self.b16));
                const g_at = self.in[0] + 2 * @as(u64, @intFromBool(e.fault == .next_column));
                inline for (.{ self.out, @as(i64, @intCast(s.n)), g_at, @as(i64, @intCast(s.n + 3)) }) |v| a.add(v);
                try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(s.rows), .y = @intCast(s.n) }, .block = .{ .x = @intCast(s.k) } }, st, &a);
            },
            .rope => {
                const scale: f32 = if (e.fault == .no_factor) 1 else e.scale;
                inline for (.{ self.in[0] + rope_at * 2, @as(i64, @intCast(s.n * head_w)), @as(i64, head_w), self.out, @as(i64, @intCast(s.n * 64)), @as(i64, 64), self.in[1], self.in[2], scale }) |v| a.add(v);
                try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(s.rows), .y = @intCast(s.n) }, .block = .{ .x = 32 } }, st, &a);
            },
        }
        try st.synchronize();
    }

    pub fn reference(self: *Rows, want: []f64, mag: []f64) !void {
        const s = self.shape;
        const a = self.rig.gpu.gpa;
        a.free(self.extra);
        self.extra = try a.alloc(f64, want.len);
        @memset(self.extra, 0);
        @memset(mag, 0);
        switch (self.entry.kind) {
            .gate => for (want, 0..) |*w, i| {
                const r = i / (s.n * s.k);
                const h = (i / s.k) % s.n;
                const z = bf(self.a16[r * (s.n + 3) + h]);
                w.* = bf(self.b16[i]) * (1 / (1 + @exp(-z)));
            },
            .rope => for (want, mag, self.extra, 0..) |*w, *m, *x, o| {
                const r = o / (s.n * 64);
                const h = (o / 64) % s.n;
                const col = o % 64;
                const i = col / 2;
                const src = (r * s.n + h) * head_w + rope_at + 2 * i;
                const x0 = bf(self.a16[src]);
                const x1 = bf(self.a16[src + 1]);
                const p: f32 = @floatFromInt(@as(i64, self.entry.pos) + @as(i64, @intCast(r)));
                const angle: f64 = self.freqs[i] * p; // the fp32 product, as the kernel and transformers form it
                const sc: f64 = self.entry.scale;
                const co = @cos(angle) * sc;
                const si = @sin(angle) * sc;
                const p0 = if (col % 2 == 0) x0 * co else x1 * co;
                const p1 = if (col % 2 == 0) -x1 * si else x0 * si;
                w.* = p0 + p1;
                m.* = @abs(p0) + @abs(p1);
                // sincosf's 2 ulp, the scale, two products and the add: within 8 units of the terms
                x.* = 8 * 0x1p-24 * m.*;
            },
        }
    }

    pub fn bounds(self: *Rows, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = want;
        _ = mag;
        for (allow, self.extra) |*l, x| l.* += x * (1 + 0x1p-7);
    }

    pub fn simulate(_: *Rows, _: c.Accum, _: g.Order, _: []f64) !void {
        return error.NoSums;
    }
};

/// Every registration and control on the heap (the gates hold pointers to them), the mla image loaded once.
pub const Group = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    rows: [entries.len]Rows = undefined,
    projs: [proj_k.entries.len]proj_k.Proj = undefined,
    chunks: [chunks_k.forms.len]chunks_k.Chunks = undefined,

    /// `merge`: glm_glue's tf_glm_merge, which turns the chunk partials into the rows the reference checks.
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, merge: cuda.Function, kernels: *std.ArrayList(g.Kernel)) !*Group {
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.mla);
        errdefer mod.unload();
        const fs = [_]cuda.Function{
            try mod.function("tf_mla_gate"),
            try mod.function("tf_mla_rope"),
            try mod.function("tf_mla_absorb_q_128"),
            try mod.function("tf_mla_expand_v_128"),
            try mod.function("tf_mla_chunks_r16"),
            try mod.function("tf_mla_chunks_split_r16"),
            try mod.function("tf_mla_merge_part"),
        };
        for (fs[4..6]) |f| try f.allowDynamicShared(cuda.mla.chunksSmem(cuda.mla.heads_tile));
        const self = try a.create(Group);
        self.* = .{ .a = a, .mod = mod };
        for (&self.rows, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = fs[@intFromBool(e.kind == .rope)] };
        for (&self.projs, proj_k.entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = fs[if (e.kind == .absorb) 2 else 3] };
        for (&self.chunks, chunks_k.forms) |*x, form| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = fs[4], .merge = merge, .fault = form.fault, .split = if (form.split) .{ fs[5], fs[6] } else null };
        for (&self.rows) |*x| try kernels.append(a, x.kernel());
        for (&self.projs) |*x| try kernels.append(a, x.kernel());
        for (&self.chunks) |*x| try kernels.append(a, x.kernel());
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (&self.rows) |*x| x.deinit();
        for (&self.projs) |*x| x.deinit();
        for (&self.chunks) |*x| x.deinit();
        self.mod.unload();
        self.a.destroy(self);
    }
};
