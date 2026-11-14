//! Native packed attention's visible-key map, scalar format decoder and exact-score value cancellation plant.
const std = @import("std");
const contract = @import("contract");
const quant = @import("dsv41_glue_ref.zig");
const math = @import("exl3_chain_ref.zig");
const c = contract.spec;
const g = contract.gates;
pub const dense_rows = 641;
pub const ring_offset = 2 * 128 * 528;
pub const live_bytes = 3 * 128 * 528;
pub const Config = struct { ratio: u32, splits: u32, owner: u32, merge_split: bool = true };
pub const configs = [_]Config{
    .{ .ratio = 0, .splits = 1, .owner = 0 },
    .{ .ratio = 1, .splits = 8, .owner = 1 },
    .{ .ratio = 2, .splits = 32, .owner = 3 },
    .{ .ratio = 2, .splits = 8, .owner = 2, .merge_split = false },
};
const positions = [_]u32{ 0, 5, 127, 128, 130, 700, 16700, 1048575 };

fn e4(code: u8) f64 {
    const mag = code & 127;
    const v = if (mag == 127) std.math.nan(f64) else quant.e4m3[mag];
    return if (code & 128 != 0) -v else v;
}

pub fn decode(bytes: []const u8, compressed: bool, col: usize) f64 {
    const value = if (compressed) blk: {
        const bits: u4 = @truncate(bytes[col / 2] >> @as(u3, @intCast(4 * (col % 2))));
        const magnitude = quant.e2m1[bits & 7];
        break :blk (if (bits & 8 != 0) -magnitude else magnitude) * e4(bytes[256 + col / 16]);
    } else blk: {
        const scale: f32 = @bitCast(@as(u32, bytes[512 + col / 32]) << 23);
        break :blk e4(bytes[col]) * @as(f64, scale);
    };
    return math.bf(value);
}

fn windowFill(bytes: []u8, rng: std.Random) void {
    rng.bytes(bytes);
    for (0..bytes.len / 528) |row| {
        const one = bytes[row * 528 ..][0..528];
        for (one[0..512]) |*code| if (code.* & 127 == 127) {
            code.* -= 1;
        };
        for (one[512..]) |*scale| scale.* = 122 + rng.uintLessThan(u8, 3);
    }
}

fn mainFill(bytes: []u8, rng: std.Random) void {
    rng.bytes(bytes);
    for (0..bytes.len / 288) |row| {
        for (bytes[row * 288 + 256 ..][0..32]) |*scale| scale.* = 32 + rng.uintLessThan(u8, 25);
    }
}

pub const Key = struct { bytes: []const u8, compressed: bool = false };
pub const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    shape: c.Shape,
    config: Config,
    fill: g.Fill,
    q_stride: usize,
    q: []u16,
    sink: []f32,
    pos: []u32,
    rings: []u8,
    chunk: []u8,
    mains: [][]u8,
    decoy: []u8,
    picks: []i32,

    pub fn init(a: std.mem.Allocator, shape: c.Shape, config: Config, fill: g.Fill, seed: u64, dense: bool) !Fixture {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const host = arena.allocator();
        const stride = shape.n + 16;
        const n = shape.rows;
        var f = Fixture{
            .arena = arena,
            .shape = shape,
            .config = config,
            .fill = fill,
            .q_stride = stride,
            .q = try host.alloc(u16, if (dense) 0 else n * stride),
            .sink = try host.alloc(f32, if (dense) 0 else shape.n / 512),
            .pos = try host.dupe(u32, positions[0..n]),
            .rings = try host.alloc(u8, n * live_bytes),
            .chunk = try host.alloc(u8, n * 528),
            .mains = try host.alloc([]u8, n),
            .decoy = &.{},
            .picks = try host.alloc(i32, n * 512),
        };
        if (fill == .stress) @memset(f.pos, 127);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (f.q) |*v| v.* = c.toBf16((rng.float(f32) - 0.5) / 16);
        for (f.sink) |*v| v.* = rng.float(f32) - 0.5;
        windowFill(f.rings, rng);
        windowFill(f.chunk, rng);
        f.chunk[0] = 0x38;
        var maximum: usize = 1;
        for (f.mains, 0..) |*main, slot| {
            const count = f.compressedRows(slot);
            main.* = try host.alloc(u8, @max(count, 1) * 288);
            mainFill(main.*, rng);
            maximum = @max(maximum, count);
            for (f.picks[slot * 512 ..][0..512], 0..) |*pick, i| {
                pick.* = if (count == 0) -1 else switch (i % 7) {
                    0 => @intCast(count - 1),
                    1 => -1,
                    2 => @intCast(count),
                    else => @intCast((i * 29 + slot) % count),
                };
            }
        }
        f.decoy = try host.alloc(u8, maximum * 288);
        @memset(f.decoy, 0x11);
        for (0..maximum) |row| @memset(f.decoy[row * 288 + 256 ..][0..32], 0x38);
        if (fill == .nonfinite) {
            if (dense) {
                f.chunk[0] = 0x7f;
                f.chunk[1] = 0xff;
                f.chunk[32] = 0x38;
                f.chunk[33] = 0xb8;
                f.chunk[513] = 255;
            } else {
                // Finite, positive K/V makes +Inf scores NaN probabilities; NaN and -Inf scores are masked.
                for (0..n) |slot| {
                    const ring = f.rings[(n - slot - 1) * live_bytes ..][0..live_bytes];
                    for (0..live_bytes / 528) |row| {
                        @memset(ring[row * 528 ..][0..512], 0x38);
                        @memset(ring[row * 528 + 512 ..][0..16], 127);
                    }
                    @memset(f.chunk[slot * 528 ..][0..512], 0x38);
                    @memset(f.chunk[slot * 528 + 512 ..][0..16], 127);
                    @memset(f.mains[slot], 0x22);
                    for (0..f.mains[slot].len / 288) |row| @memset(f.mains[slot][row * 288 + 256 ..][0..32], 0x38);
                    const q = f.q[slot * stride ..][0..shape.n];
                    @memset(q, c.toBf16(0));
                    @memset(q[0..512], c.toBf16(std.math.nan(f32)));
                    q[512] = c.toBf16(std.math.inf(f32));
                    q[1024] = c.toBf16(-std.math.inf(f32));
                }
            }
        }
        if (fill == .stress) {
            @memset(f.q, c.toBf16(0));
            @memset(f.sink, 0);
            @memset(f.chunk, 0);
            for (0..n) |slot| {
                const ring = f.rings[(n - slot - 1) * live_bytes ..][0..live_bytes];
                @memset(ring, 0);
                @memset(f.mains[slot], 0);
                for (f.picks[slot * 512 ..][0..512], 0..) |*pick, i| pick.* = if (f.compressedRows(slot) == 0) -1 else @intCast(i % f.compressedRows(slot));
                for ([_]usize{ 0, 32, 64, 96 }, [_]f64{ 4096, 1, -4096, -0.5 }) |row, value| {
                    var values: [512]f64 = @splat(value);
                    math.packKey(&values, ring[ring_offset + row * 528 ..][0..528]);
                }
            }
        }
        f.arena = arena;
        return f;
    }

    pub fn deinit(f: *Fixture) void {
        f.arena.deinit();
    }
    pub fn compressedRows(f: Fixture, slot: usize) usize {
        return if (f.config.ratio == 0) 0 else (f.pos[slot] + 1) / f.config.ratio;
    }
    pub fn key(f: Fixture, slot: usize, index: usize) ?Key {
        const count = f.compressedRows(slot);
        if (index < 128) {
            if (index >= @min(f.pos[slot], 128)) return null;
            const offset = (f.shape.rows - slot - 1) * live_bytes + ring_offset + index * 528;
            return .{ .bytes = f.rings[offset..][0..528] };
        }
        if (index == 128) return .{ .bytes = f.chunk[slot * 528 ..][0..528] };
        const i = index - 129;
        if (i >= @min(count, 512)) return null;
        const pick = f.picks[slot * 512 + i];
        if (pick < 0 or pick >= count) return null;
        return .{ .bytes = f.mains[slot][@as(usize, @intCast(pick)) * 288 ..][0..288], .compressed = true };
    }
    pub fn denseReference(f: Fixture, out: []f64) void {
        for (0..f.shape.rows) |slot| for (0..dense_rows) |index| {
            const row = out[(slot * dense_rows + index) * 512 ..][0..512];
            if (f.key(slot, index)) |k| {
                for (row, 0..) |*v, col| v.* = decode(k.bytes, k.compressed, col);
            } else @memset(row, std.math.nan(f64));
        };
    }
};

pub fn stressValue(keys: usize, acc: c.Accum, order: g.Order) f64 {
    var terms: [640]f64 = @splat(0);
    for ([_]usize{ 0, 32, 64, 96 }, [_]f64{ 4096, 1, -4096, -0.5 }) |i, value| terms[i] = value;
    return contract.rows.sumIn(acc, order, terms[0..keys]) / @as(f64, @floatFromInt(keys + 1));
}

test "native attention value cancellation survives FP32 but separates both narrow sums in every order" {
    const spec = c.Spec{ .name = "native-value", .accum = .f32, .depth = struct {
        fn get(_: c.Shape) usize {
            return 1320;
        }
    }.get, .ops = 12, .out = .bf16, .served = &.{.{ .rows = 1, .n = 8192, .k = 512 }} };
    for ([_]usize{ 128, 192, 256, 640 }) |keys| {
        const want = 0.5 / @as(f64, @floatFromInt(keys + 1));
        for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
            try std.testing.expectEqual(want, stressValue(keys, .f32, order));
            for ([_]c.Accum{ .f16, .bf16 }) |acc| try std.testing.expect(@abs(stressValue(keys, acc, order) - want) > 100 * c.boundExact(spec, want));
        }
    }
}

test "native packed decoder preserves signed zero, non-finite scale classes and both nibble signs" {
    var window: [528]u8 = @splat(0);
    @memset(window[512..], 127);
    window[0] = 0x80;
    window[1] = 0x7f;
    window[32] = 0x38;
    window[33] = 0xb8;
    window[513] = 255;
    try std.testing.expect(std.math.signbit(decode(&window, false, 0)));
    try std.testing.expectEqual(c.Class.nan, c.classOf(decode(&window, false, 1)));
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(decode(&window, false, 32)));
    try std.testing.expectEqual(c.Class.neg_inf, c.classOf(decode(&window, false, 33)));
    var main: [288]u8 = @splat(0);
    @memset(main[256..], 0x38);
    main[0] = 0xf7;
    try std.testing.expectEqual(@as(f64, 6), decode(&main, true, 0));
    try std.testing.expectEqual(@as(f64, -6), decode(&main, true, 1));
}
