//! The ModelOpt products under the contract: NVFP4's e2m1 codes with a group scale, and FP8's E4M3 codes with one.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const quant = core.quant;

pub const Which = enum { nvfp4, fp8 };

/// The kernel's own chain: k fmaf into the sum, a group's fmaf into the accumulator (NVFP4), then the scale.
fn depthNv(s: c.Shape) usize {
    return s.k + s.k / quant.nvfp4.group + 1;
}

/// FP8 has no groups: k fmaf into the sum, then the scale.
fn depthFp8(s: c.Shape) usize {
    return s.k + 1;
}

const served = [_]c.Shape{
    .{ .rows = 1, .n = 256, .k = 4096 },
    .{ .rows = 7, .n = 1024, .k = 1024 },
    .{ .rows = 129, .n = 130, .k = 2048 },
};

pub fn spec(w: Which) c.Spec {
    return .{
        .name = switch (w) {
            .nvfp4 => "hip ModelOpt NVFP4 product",
            .fp8 => "hip ModelOpt FP8 product",
        },
        .accum = .f32,
        .depth = switch (w) {
            .nvfp4 => depthNv,
            .fp8 => depthFp8,
        },
        .out = .f32,
        .served = &served,
        .tiles = .{ .{ .size = 256 }, .{}, .{} },
    };
}

pub const Product = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    codes: []u8 = &.{},
    table: []u16 = &.{},
    global: f32 = 1,
    dx: u64 = 0,
    dc: u64 = 0,
    dt: u64 = 0,
    ds: u64 = 0,
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, w: Which) Product {
        return .{ .rig = .{ .a = a, .r = r, .s = s }, .l = l, .which = w, .arena = .init(a) };
    }

    pub fn deinit(t: *Product) void {
        t.arena.deinit();
        t.rig.deinit();
    }

    const vt = g.hooks(Product);

    pub fn kernel(t: *Product) g.Kernel {
        return .{ .spec = spec(t.which), .ctx = t, .vt = &vt };
    }

    pub fn prepare(t: *Product, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.n == 0 or s.k == 0 or s.k % 32 != 0) return false;
        t.rig.reset();
        _ = t.arena.reset(.retain_capacity);
        const a = t.arena.allocator();
        t.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try a.alloc(u16, s.rows * s.k);
        for (t.x) |*v| v.* = if (fill == .stress)
            c.toBf16(@as(f32, @floatFromInt(@as(i8, @intCast(rng.uintLessThan(u8, 6))) - 3)) / 2)
        else
            c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .nonfinite and t.x.len > 0) t.x[0] = c.toBf16(std.math.inf(f32));
        t.dx = try t.rig.input(std.mem.sliceAsBytes(t.x));

        switch (t.which) {
            .nvfp4 => {
                t.codes = try a.alloc(u8, s.n * s.k / 2);
                // stress: e2m1's largest level in both nibbles, so an fp16 sum would round
                for (t.codes) |*b| b.* = if (fill == .stress) 0x77 else @intCast(rng.uintLessThan(u16, 256));
                t.table = try a.alloc(u16, s.n * (s.k / quant.nvfp4.group));
                // E4M3 in [0.5, 1.875) widened to BF16, exactly what the reader uploads
                for (t.table) |*v| {
                    // stress: one exact scale, so a group's product is exact too
                    const byte: u8 = if (fill == .stress) 0x38 else @intCast(0x30 + rng.uintLessThan(u8, 0x10));
                    const scale: f32 = @floatCast(quant.fp8.e4m3(byte));
                    v.* = @truncate(@as(u32, @bitCast(scale)) >> 16);
                }
                t.global = if (fill == .stress) 0.5 else 0.3;
                t.dc = try t.rig.input(t.codes);
                t.dt = try t.rig.input(std.mem.sliceAsBytes(t.table));
                const one = [_]f32{t.global};
                t.ds = try t.rig.input(std.mem.sliceAsBytes(&one));
            },
            .fp8 => {
                t.codes = try a.alloc(u8, s.n * s.k);
                for (t.codes) |*b| {
                    // stress: E4M3 values in [1, 1.875), each an exact multiple of 0.125
                    var v: u8 = if (fill == .stress) @intCast(0x38 + rng.uintLessThan(u8, 8)) else @intCast(rng.uintLessThan(u16, 256));
                    if (v & 0x7f == 0x7f) v &= 0xfe; // E4M3's NaN codes are refused at read
                    b.* = v;
                }
                t.global = if (fill == .stress) 0.5 else 0.01171875;
                t.dc = try t.rig.input(t.codes);
                const one = [_]f32{t.global};
                t.ds = try t.rig.input(std.mem.sliceAsBytes(&one));
            },
        }
        t.out = try t.rig.output(s.rows * s.n * 4);
        return true;
    }

    pub fn run(t: *Product) !void {
        const s = t.shape;
        switch (t.which) {
            .nvfp4 => {
                const w: hip.nvfp4.Weight = .{
                    .x = t.dx,
                    .words = t.dc,
                    .scale = .{ .p = t.dt, .kind = @backingInt(quant.Tables.bf16) },
                    .out = t.out,
                    .m = @intCast(s.rows),
                    .n = @intCast(s.n),
                    .k = @intCast(s.k),
                    .global = t.global,
                };
                try hip.nvfp4.Kernels.gemmGo(&t.l.nvfp4, t.l.r, w, t.rig.s.handle);
            },
            .fp8 => {
                const w: hip.fp8.Weight = .{
                    .x = t.dx,
                    .codes = t.dc,
                    .scale = t.ds,
                    .out = t.out,
                    .m = @intCast(s.rows),
                    .n = @intCast(s.n),
                    .k = @intCast(s.k),
                };
                try hip.fp8.Kernels.gemmGo(&t.l.fp8, t.l.r, w, t.rig.s.handle);
            },
        }
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Product) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Product, want: []f64, mag: []f64) !void {
        const s = t.shape;
        for (0..s.rows) |r| for (0..s.n) |col| {
            var acc: f64 = 0;
            var abs: f64 = 0;
            for (0..s.k) |j| {
                const code: f64 = switch (t.which) {
                    .nvfp4 => quant.nvfp4.e2m1(if ((j & 1) == 0) t.codes[col * (s.k / 2) + j / 2] & 0xf else t.codes[col * (s.k / 2) + j / 2] >> 4),
                    .fp8 => quant.fp8.e4m3(t.codes[col * s.k + j]),
                };
                const scale: f64 = switch (t.which) {
                    .nvfp4 => @as(f64, @floatCast(@as(f32, @bitCast(@as(u32, t.table[col * (s.k / quant.nvfp4.group) + j / quant.nvfp4.group]) << 16)))) * @as(f64, t.global),
                    .fp8 => @as(f64, t.global),
                };
                const term = c.fromBf16(t.x[r * s.k + j]) * code * scale;
                acc += term;
                abs += @abs(term);
            }
            want[r * s.n + col] = acc;
            mag[r * s.n + col] = abs;
        };
    }

    /// The same sums in an accumulator, for gate 4: the kernel's k-long chain, terms scaled as the reference has them.
    pub fn simulate(t: *Product, acc: c.Accum, order: g.Order, out: []f64) !void {
        _ = order; // the kernel sums one chain in k order
        const s = t.shape;
        for (0..s.rows) |r| {
            for (0..s.n) |col| {
                var exact: f64 = 0;
                var a: f32 = 0;
                for (0..s.k) |j| {
                    const code: f64 = switch (t.which) {
                        .nvfp4 => quant.nvfp4.e2m1(if ((j & 1) == 0) t.codes[col * (s.k / 2) + j / 2] & 0xf else t.codes[col * (s.k / 2) + j / 2] >> 4),
                        .fp8 => quant.fp8.e4m3(t.codes[col * s.k + j]),
                    };
                    const scale: f64 = switch (t.which) {
                        .nvfp4 => @as(f64, @floatCast(@as(f32, @bitCast(@as(u32, t.table[col * (s.k / quant.nvfp4.group) + j / quant.nvfp4.group]) << 16)))) * @as(f64, t.global),
                        .fp8 => @as(f64, t.global),
                    };
                    const term = c.fromBf16(t.x[r * s.k + j]) * code * scale;
                    switch (acc) {
                        // exact fp32 sums: no intermediate rounding, one store; the gates that need
                        // a rounded accumulator are fp16 and bf16, below
                        .f32 => exact += term,
                        .bf16 => a = @floatCast(core.bf16_round.bf16(@as(f64, a) + term)),
                        .f16 => a = @floatCast(@as(f16, @floatCast(@as(f64, a) + term))),
                        .none => unreachable,
                    }
                }
                out[r * s.n + col] = switch (acc) {
                    .f32 => exact,
                    else => @floatCast(a),
                };
            }
        }
    }
};
