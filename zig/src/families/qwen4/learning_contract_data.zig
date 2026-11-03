//! Typed native primitive fixtures retain metadata separately from arithmetic and expose binary64 precision intervals.
const std = @import("std");
const core = @import("core");
const rows = core.living_rows;
const fixture = @import("learning_fixture.zig");
pub const Kind = enum(u32) { f32, bf16, integer };
pub const Value = extern struct { value: f64, err: f64, depth: u32, pending: ?*anyopaque };
extern fn tf_living_reference(c_uint, c_uint, rows.Params, [*][*]u8, [*]const u32, [*]const u32, [*][*]Value, c_uint, c_uint) bool;
pub fn kind(comptime spec: type, op: spec.Op, p: rows.Params, index: usize) Kind {
    if (spec == core.living_rows) return switch (op) {
        .linear => if (index == 1) (if (p.u[3] == 1) .integer else .bf16) else if (index == 2 or index == 3) .bf16 else .f32,
        .rms => if (index == 0) .bf16 else .f32,
        .hc_act => if (index == 0) .bf16 else .f32,
        .hc_mix => if (index < 2) .bf16 else .f32,
        .hc_branch => if (index == 1 or index == 2) .bf16 else .f32,
        .swiglu => if (index < 2) .bf16 else .f32,
        .loss => if (index == 0) .bf16 else if (index == 1) .integer else .f32,
        .nudge => if (index == 0 or index == 8) .bf16 else .f32,
        .gate => if (index < 2) .bf16 else .f32,
        .rope_rms => if (index == 0) .bf16 else .f32,
        .pocket_act => if (index == 0) .bf16 else .f32,
        .pocket_add => if (index == 2 or index == 8) .bf16 else .f32,
        .pocket_back => .f32,
        .pocket_coeff => if (index == 0) .bf16 else .f32,
    };
    if (spec == core.living_gdn) return switch (op) {
        .front => if (index == 0 or index == 8) .bf16 else .f32,
        .norm => if (index == 0) .bf16 else .f32,
        .readout => if (index == 0) .bf16 else .f32,
        .scan => if (index == 0 or index == 4) .bf16 else .f32,
        .qk => if (index == 0) .bf16 else .f32,
        .gates => if (index == 1) .bf16 else .f32,
        .conv => if (index == 0) .bf16 else .f32,
    };
    return switch (op) {
        .scores => if (index < 3) .bf16 else if (index >= 4 and index <= 6) .integer else .f32,
        .softmax => if (index >= 2 and index <= 4) .integer else .f32,
        .query => if (index == 1) .bf16 else if (index >= 2 and index <= 4) .integer else .f32,
        .key_value => if (index == 2) .bf16 else if (index >= 4 and index <= 6) .integer else .f32,
    };
}
pub fn params(comptime spec: type, op: spec.Op, count: usize, width: usize) rows.Params {
    var p: rows.Params = .{};
    p.u[0] = @intCast(count);
    if (spec == core.living_rows) {
        p.u[1] = @intCast(width);
        p.u[2] = 2;
        p.u[4] = 64;
        p.f[0] = 1e-6;
        switch (op) {
            .linear => {
                p.u[1] = 3;
                p.u[2] = @intCast(width);
            },
            .gate => {
                p.u[2] = 32;
                p.u[3] = @intCast(2 * width);
            },
            .rope_rms => {
                p.u[1] = 2;
                p.u[2] = @intCast(width);
                p.u[3] = 16;
                p.u[4] = @intCast(4 * width);
                p.u[6] = @intCast(2 * width);
            },
            .pocket_act, .pocket_add, .pocket_back, .pocket_coeff => {
                p.f[0] = 1;
            },
            else => {},
        }
    } else if (spec == core.living_gdn) {
        p.u[1] = 1;
        p.u[2] = 2;
        p.u[3] = 772;
        p.f[0] = 1e-6;
    } else {
        p.u[1] = 4;
        p.u[2] = 2;
        p.u[3] = @intCast(width);
        p.u[4] = @intCast(count);
        p.f[0] = 1 / @sqrt(@as(f32, @floatFromInt(width)));
    }
    return p;
}
pub fn reduces(comptime spec: type, op: spec.Op) bool {
    if (spec == core.living_rows) return switch (op) {
        .hc_act, .hc_mix, .swiglu, .nudge, .gate => false,
        else => true,
    };
    if (spec == core.living_gdn) return op != .gates;
    return true;
}
pub fn Fixture(comptime spec: type) type {
    return struct {
        const Self = @This();
        pub const Raw = fixture.Case(spec);
        raw: Raw,
        kinds: [11]u32 = undefined,
        counts: [11]u32 = undefined,
        values: [3][]Value = undefined,
        owned: usize = 0,
        a: std.mem.Allocator,
        pub fn init(a: std.mem.Allocator, op: spec.Op, p: rows.Params, seed: u64, nonfinite: bool) !Self {
            var out: Self = .{ .raw = try Raw.initWithAllocator(a, op, p), .a = a };
            errdefer out.deinit();
            const n = try spec.needs(op, p);
            var random: std.Random.DefaultPrng = .init(seed);
            for (out.raw.data, n.bytes, 0..) |data, size, i| {
                const k = kind(spec, op, p, i);
                out.kinds[i] = @backingInt(k);
                out.counts[i] = @intCast(size / @as(usize, if (k == .bf16) 2 else 4));
                if (i >= 8) continue;
                if (k == .integer) {
                    @memset(data, 0);
                    continue;
                }
                for (0..out.counts[i]) |j| {
                    const value = (random.random().float(f32) - 0.5) * 0.25;
                    if (k == .bf16) std.mem.writeInt(u16, data[j * 2 ..][0..2], core.quant.convert.bf16(value), .little) else std.mem.writeInt(u32, data[j * 4 ..][0..4], @bitCast(value), .little);
                }
            }
            for (&out.values, 8..) |*v, i| {
                v.* = try a.alloc(Value, out.counts[i]);
                out.owned += 1;
            }
            if (spec == core.living_rows and op == .loss) {
                const targets = std.mem.bytesAsSlice(u32, out.raw.data[1]);
                for (0..p.u[0]) |i| {
                    targets[i] = @intCast(i % p.u[1]);
                    out.raw.floats(2)[i] = 1 / @as(f32, @floatFromInt(p.u[0]));
                }
            }
            if (spec == core.living_qsa) {
                const first: usize = if (op == .scores or op == .key_value) 4 else 2;
                const ids = std.mem.bytesAsSlice(i32, out.raw.data[first]);
                const counts = std.mem.bytesAsSlice(i32, out.raw.data[first + 1]);
                const flags = std.mem.bytesAsSlice(i32, out.raw.data[first + 2]);
                for (0..p.u[0]) |row| {
                    flags[row] = @intCast(row % 2);
                    counts[row] = @intCast(row / 2 + 1);
                    for (0..row / 2 + 1) |j| ids[row * p.u[4] + j] = @intCast(2 * j);
                }
            }
            if (nonfinite) {
                const k = kind(spec, op, p, 0);
                if (k != .integer) {
                    const choices = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
                    const stride = out.counts[0] / p.u[0];
                    const offset: usize = if (spec == core.living_gdn) switch (op) {
                        .readout => (2 * p.u[1] + p.u[2]) * 128,
                        .scan => 2 * p.u[1] * 128,
                        .gates => 384,
                        else => 0,
                    } else 0;
                    for (0..@min(p.u[0], choices.len)) |row| {
                        const i = row * stride + offset;
                        if (k == .bf16) std.mem.writeInt(u16, out.raw.data[0][i * 2 ..][0..2], core.quant.convert.bf16(choices[row]), .little) else out.raw.floats(0)[i] = choices[row];
                    }
                    if (p.u[0] >= 4 and stride >= 2) for ([_]f32{ std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, j| {
                        const i = 3 * stride + offset + j;
                        if (k == .bf16) std.mem.writeInt(u16, out.raw.data[0][i * 2 ..][0..2], core.quant.convert.bf16(v), .little) else out.raw.floats(0)[i] = v;
                    };
                }
            }
            return out;
        }
        pub fn deinit(f: *Self) void {
            for (f.values[0..f.owned]) |v| f.a.free(v);
            f.raw.deinit();
        }
        pub fn stress(f: *Self) void {
            if (spec == core.living_rows) switch (f.raw.op) {
                .linear => {
                    for (0..f.counts[0]) |i| f.raw.floats(0)[i] = ([_]f32{ 65537, 65536, -131072, 0 })[i % f.raw.p.u[1] % 4];
                    const weights = std.mem.bytesAsSlice(u16, f.raw.data[1]);
                    @memset(weights, core.quant.convert.bf16(1));
                },
                .rms, .rope_rms => {
                    f.raw.p.f[0] = 3;
                    @memset(std.mem.bytesAsSlice(u16, f.raw.data[0]), core.quant.convert.bf16(1));
                    @memset(f.raw.floats(1), 1);
                    @memset(f.raw.floats(2), 0);
                    const width: usize = if (f.raw.op == .rms) f.raw.p.u[1] else f.raw.p.u[2];
                    for (0..f.counts[2]) |i| if (i % width < 3) {
                        f.raw.floats(2)[i] = ([_]f32{ 65537, 65536, -131072 })[i % width];
                    };
                    if (f.raw.op == .rope_rms) @memset(f.raw.floats(3), 0);
                },
                .loss => {
                    @memset(std.mem.bytesAsSlice(u16, f.raw.data[0]), core.quant.convert.bf16(-0.000244140625));
                    for (0..f.raw.p.u[0]) |r| std.mem.writeInt(u16, f.raw.data[0][r * f.raw.p.u[1] * 2 ..][0..2], 0, .little);
                },
                else => {},
            };
            if (spec == core.living_qsa) {
                f.raw.p.f[0] = 0.0625;
                for (f.raw.data, 0..) |bytes, i| {
                    if (i >= 8 or f.kinds[i] == @backingInt(Kind.integer)) continue;
                    for (0..f.counts[i]) |j| {
                        const v = @as(f32, @floatFromInt(@as(i32, @intCast((j * 13 + i * 17) % 31)) - 15)) * 0.125;
                        if (f.kinds[i] == @backingInt(Kind.bf16)) std.mem.writeInt(u16, bytes[j * 2 ..][0..2], core.quant.convert.bf16(v), .little) else f.raw.floats(i)[j] = v;
                    }
                }
            }
            if (spec == core.living_qsa and (f.raw.op == .query or f.raw.op == .key_value)) {
                const r: usize = f.raw.p.u[0];
                const heads: usize = f.raw.p.u[1];
                for (0..r) |row| for (0..heads) |head| for (0..r) |key| {
                    const j = if (f.raw.op == .query) key else row;
                    f.raw.floats(0)[(row * heads + head) * r + key] = ([_]f32{ 65537, 65536, -131072, 0 })[j % 4];
                };
                const index: usize = if (f.raw.op == .query) 1 else 2;
                @memset(std.mem.bytesAsSlice(u16, f.raw.data[index]), core.quant.convert.bf16(1));
            }
            if (spec == core.living_qsa and f.raw.op == .softmax) {
                @memset(f.raw.floats(0), 0);
                @memset(f.raw.floats(1), 0);
                const r: usize = f.raw.p.u[0];
                const heads: usize = f.raw.p.u[1];
                if (r >= 4) for (0..heads) |h| for ([_]f32{ 65537, 65536, -131072, 0 }, 0..) |v, j| {
                    f.raw.floats(1)[(3 * heads + h) * r + j] = v;
                };
            }
        }
        pub fn reference(f: *Self) !void {
            var ptrs: [11][*]u8 = undefined;
            for (f.raw.data, &ptrs) |bytes, *ptr| ptr.* = bytes.ptr;
            var outputs: [3][*]Value = undefined;
            for (f.values, &outputs) |value, *ptr| ptr.* = value.ptr;
            const tag = if (spec == core.living_rows) 0 else if (spec == core.living_gdn) 1 else 2;
            if (!tf_living_reference(tag, @intCast(@backingInt(f.raw.op)), f.raw.p, &ptrs, &f.kinds, &f.counts, &outputs, 0, 0)) return error.OutOfMemory;
        }
        pub fn simulate(f: *Self, acc: core.contract.spec.Accum, order: core.contract.gates.Order, out: []f64) !void {
            var temp: [3][]Value = undefined;
            var made: usize = 0;
            defer for (temp[0..made]) |v| f.a.free(v);
            for (&temp, f.values) |*v, base| {
                v.* = try f.a.alloc(Value, base.len);
                made += 1;
            }
            var ptrs: [11][*]u8 = undefined;
            for (f.raw.data, &ptrs) |bytes, *ptr| ptr.* = bytes.ptr;
            var outputs: [3][*]Value = undefined;
            for (temp, &outputs) |v, *ptr| ptr.* = v.ptr;
            const tag = if (spec == core.living_rows) 0 else if (spec == core.living_gdn) 1 else 2;
            const mode: c_uint = switch (acc) {
                .none => 0,
                .f32 => 1,
                .f16 => 2,
                .bf16 => 3,
            };
            if (!tf_living_reference(tag, @intCast(@backingInt(f.raw.op)), f.raw.p, &ptrs, &f.kinds, &f.counts, &outputs, mode, @intCast(@backingInt(order)))) return error.OutOfMemory;
            var at: usize = 0;
            for (temp) |values| for (values) |v| {
                out[at] = v.value;
                at += 1;
            };
        }
        pub fn output(f: *Self, index: usize, cell: usize) f64 {
            return if (f.kinds[index] == @backingInt(Kind.bf16)) @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, f.raw.data[index][cell * 2 ..][0..2], .little)) << 16)) else f.raw.floats(index)[cell];
        }
    };
}
