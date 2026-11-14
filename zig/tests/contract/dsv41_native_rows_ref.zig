//! Independent row-position, BF16 rotation and packed-cache formulas; mutable ring images retain untouched bytes.
const std = @import("std");
const contract = @import("contract");
const math = @import("exl3_chain_ref.zig");
const Kind = @import("dsv41_native_rows_source.zig").Kind;
const c = contract.spec;
const g = contract.gates;

pub const ring_offset = 2 * 128 * 528;
pub const live_bytes = 3 * 128 * 528;
const positions = [_]u32{ 0, 5, 127, 128, 130, 700, 16700, 1048575 };

pub const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    kind: Kind,
    shape: c.Shape,
    inverse: bool,
    stride: usize,
    x: []u16,
    weight: []u16,
    factors: []f32,
    pos: []u32,
    chunk: []u8,
    images: []u8,

    pub fn init(a: std.mem.Allocator, kind: Kind, shape: c.Shape, inverse: bool, fill: g.Fill, seed: u64) !Fixture {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const host = arena.allocator();
        const n = shape.rows;
        const stride = (if (kind == .rope) shape.n else 512) + 16;
        const capacity = if (kind == .publish) 0 else @as(usize, positions[n - 1]) + 1;
        var f = Fixture{
            .arena = arena,
            .kind = kind,
            .shape = shape,
            .inverse = inverse,
            .stride = stride,
            .x = try host.alloc(u16, if (kind == .publish) 0 else n * stride),
            .weight = try host.alloc(u16, if (kind == .glue) 512 else 0),
            .factors = try host.alloc(f32, capacity * 64),
            .pos = try host.dupe(u32, positions[0..n]),
            .chunk = try host.alloc(u8, if (kind == .publish) n * 528 else 0),
            .images = try host.alloc(u8, if (kind == .publish) n * live_bytes else 0),
        };
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (f.x) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (f.weight) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        rng.bytes(f.chunk);
        rng.bytes(f.images);
        if (kind == .publish) f.chunk[0] = 0x31;
        @memset(f.factors, 0);
        if (kind != .publish) for (f.pos) |pos| {
            for (0..32) |pair| {
                const angle = @as(f32, @floatFromInt(pos % 100003 + pair * 71 + 1)) / 97;
                f.factors[@as(usize, pos) * 64 + 2 * pair] = @cos(angle);
                f.factors[@as(usize, pos) * 64 + 2 * pair + 1] = @sin(angle);
            }
        };
        if (fill == .nonfinite) {
            if (kind == .publish) {
                f.chunk[0] = 0x7f;
                f.chunk[1] = 0xff;
            } else {
                const at: usize = if (kind == .rope) 512 - 64 else 171;
                f.x[at] = c.toBf16(std.math.nan(f32));
                if (n > 1) f.x[stride + 511] = c.toBf16(std.math.inf(f32));
                if (n > 2) f.x[2 * stride + 510] = c.toBf16(-std.math.inf(f32));
            }
        }
        if (fill == .stress) {
            @memset(f.x, c.toBf16(256));
            @memset(f.weight, c.toBf16(1));
            for (0..f.factors.len / 2) |pair| {
                f.factors[2 * pair] = 1;
                f.factors[2 * pair + 1] = 0;
            }
        }
        f.arena = arena;
        return f;
    }

    pub fn deinit(f: *Fixture) void {
        f.arena.deinit();
    }

    pub fn count(f: Fixture) usize {
        return f.shape.rows * switch (f.kind) {
            .rope => f.shape.n,
            .glue => 528,
            .publish => live_bytes,
        };
    }

    pub fn evaluate(f: Fixture, acc: c.Accum, order: g.Order, out: []f64) void {
        const n = f.shape.rows;
        if (f.kind == .publish) {
            for (out, f.images) |*v, byte| v.* = @floatFromInt(byte);
            for (0..n) |row| {
                const slot = n - row - 1;
                const first = slot * live_bytes + ring_offset + (f.pos[row] % 128) * 528;
                for (out[first..][0..528], f.chunk[row * 528 ..][0..528]) |*v, byte| v.* = @floatFromInt(byte);
            }
            return;
        }
        var x: [32768]f64 = undefined;
        var normal: [512]f64 = undefined;
        var weight: [512]f64 = undefined;
        var rotated: [512]f64 = undefined;
        if (f.kind == .glue) for (&weight, f.weight) |*v, bits| {
            v.* = c.fromBf16(bits);
        };
        for (0..n) |row| {
            const width = if (f.kind == .rope) f.shape.n else 512;
            for (x[0..width], f.x[row * f.stride ..][0..width]) |*v, bits| v.* = c.fromBf16(bits);
            const factors = f.factors[@as(usize, f.pos[row]) * 64 ..][0..64];
            if (f.kind == .rope) {
                math.rope(x[0..width], 512, 64, factors, f.inverse, out[row * width ..][0..width]);
            } else {
                math.rms(x[0..512], &weight, @as(f32, 1e-6), &normal, acc, order);
                math.rope(&normal, 512, 64, factors, false, &rotated);
                var key_pack: [528]u8 = undefined;
                math.packKey(&rotated, &key_pack);
                for (out[row * 528 ..][0..528], key_pack) |*v, byte| v.* = @floatFromInt(byte);
            }
        }
    }
};
