//! Host-owned chain inputs and independent float64 outputs, including every intermediate activation plane.
const std = @import("std");
const contract = @import("contract");
const math = @import("exl3_chain_ref.zig");
const Kind = @import("exl3_chain_source.zig").Kind;
const c = contract.spec;
const g = contract.gates;

pub const Fixture = struct {
    pub const max_width: usize = 32768;
    arena: std.heap.ArenaAllocator,
    kind: Kind,
    shape: c.Shape,
    stress: bool,
    sums: []f32,
    x: []u16,
    up: []u16,
    su: []f16,
    sv: []f16,
    bias: []f16,
    weights: []f32,
    norm_weight: []f32,
    factors: []f32,
    routes: []u32,
    ids: []u32,
    destinations: []u32,
    signs: bool,
    inverse: bool,
    weighted: bool,
    stride: usize,
    head_width: usize,
    position: usize,
    descriptors: usize,
    expert_count: usize,
    physical_rows: usize,
    picks: usize = 6,
    rotary: usize = 64,
    eps: f32 = 1e-6,

    pub fn init(a: std.mem.Allocator, kind: Kind, shape: c.Shape, fill: g.Fill, seed: u64) !Fixture {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const host = arena.allocator();
        const k = shape.k;
        const n = shape.rows;
        const head_width: usize = if (k % 512 == 0 and seed % 2 == 0) 512 else 128;
        const position: usize = if (fill == .stress) 0 else ([_]usize{ 0, 7, 4095, 16384 })[seed % 4];
        const stride = k + (if (kind == .rope_in) @as(usize, 0) else 16);
        const expert_count: usize = switch (kind) {
            .swiglu, .gate_up, .down => 384,
            .qkv => 2,
            else => 8,
        };
        const descriptors: usize = switch (kind) {
            .gate_up => 2 * expert_count + 2,
            .down => expert_count + 1,
            else => expert_count,
        };
        const physical_rows = switch (kind) {
            .gate_up => 14 * n,
            .down => 7 * n,
            .qkv => 2 * n,
            else => n,
        };
        var f = Fixture{
            .arena = arena,
            .kind = kind,
            .shape = shape,
            .stress = fill == .stress,
            .sums = try host.alloc(f32, physical_rows * k),
            .x = try host.alloc(u16, n * stride),
            .up = try host.alloc(u16, n * stride),
            .su = try host.alloc(f16, @max(descriptors, if (kind == .out_in) n else 0) * k),
            .sv = try host.alloc(f16, descriptors * k),
            .bias = try host.alloc(f16, descriptors * k),
            .weights = try host.alloc(f32, @max(physical_rows, n)),
            .norm_weight = try host.alloc(f32, k),
            .factors = try host.alloc(f32, (position + n) * 64),
            .routes = try host.alloc(u32, physical_rows),
            .ids = try host.alloc(u32, physical_rows),
            .destinations = try host.alloc(u32, n),
            .stride = stride,
            .head_width = head_width,
            .position = position,
            .descriptors = descriptors,
            .expert_count = expert_count,
            .physical_rows = physical_rows,
            .signs = fill != .stress and seed % 2 == 0,
            .inverse = seed % 2 != 0,
            .weighted = seed % 3 != 0,
        };
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (f.sums) |*v| v.* = (rng.float(f32) * 2 - 1) * 3;
        for (f.x) |*v| v.* = c.toBf16(rng.float(f32) * 32 - 16);
        for (f.up) |*v| v.* = c.toBf16(rng.float(f32) * 32 - 16);
        for (f.su) |*v| v.* = if (rng.boolean()) 1.25 else -0.75;
        for (f.sv) |*v| v.* = if (rng.boolean()) 0.5 else -1.5;
        for (f.bias) |*v| v.* = @floatCast((rng.float(f32) - 0.5) * 0.25);
        for (f.weights) |*v| v.* = 0.25 + rng.float(f32) * 1.5;
        for (f.norm_weight) |*v| v.* = 0.5 + rng.float(f32);
        for (f.destinations, 0..) |*dest, row| dest.* = @intCast(n - row - 1);
        for (0..position + n) |row| {
            for (0..32) |pair| {
                const angle = @as(f32, @floatFromInt(row * 17 + pair + 1)) / 37;
                f.factors[row * 64 + pair * 2] = @cos(angle);
                f.factors[row * 64 + pair * 2 + 1] = @sin(angle);
            }
        }
        for (f.routes, 0..) |*v, i| v.* = @intCast((i * 67 + expert_count - 1) % expert_count);
        for (f.ids, 0..) |*v, i| v.* = @intCast(i % descriptors);
        if (kind == .gate_up) {
            for (0..6 * n) |row| {
                f.ids[row] = f.routes[row];
                f.ids[6 * n + row] = @intCast(expert_count + f.routes[row]);
            }
            for (0..n) |row| {
                f.ids[12 * n + row] = @intCast(2 * expert_count);
                f.ids[13 * n + row] = @intCast(2 * expert_count + 1);
            }
        }
        if (kind == .down) {
            for (0..n) |row| @memcpy(f.routes[row * 6 ..][0..6], &[_]u32{ 383, 0, 1, 0, 382, 1 });
        }
        if (fill == .nonfinite) {
            if (kind == .swiglu) {
                f.x[0] = c.toBf16(-std.math.inf(f32));
                if (n > 1) f.weights[1] = std.math.inf(f32);
                if (n > 2) f.weights[2] = std.math.nan(f32);
            } else if (kind == .rms or kind == .rope_in) {
                f.x[0] = c.toBf16(std.math.nan(f32));
                if (n > 1) f.x[f.stride + k - 1] = c.toBf16(std.math.inf(f32));
                if (n > 2) f.x[2 * f.stride + k / 2] = c.toBf16(-std.math.inf(f32));
            } else if (kind == .qkv) {
                for (0..n) |row| f.sums[2 * row * k] = switch (row % 3) {
                    0 => std.math.nan(f32),
                    1 => std.math.inf(f32),
                    else => -std.math.inf(f32),
                };
            } else {
                f.sums[0] = std.math.nan(f32);
                if (n > 1) f.sums[k] = std.math.inf(f32);
                if (n > 2) f.sums[2 * k + k / 2] = -std.math.inf(f32);
            }
        }
        if (fill == .stress) f.precisionPlant();
        f.arena = arena;
        return f;
    }

    pub fn deinit(f: *Fixture) void {
        f.arena.deinit();
    }

    fn precisionPlant(f: *Fixture) void {
        @memset(f.x, 0);
        @memset(f.up, 0);
        @memset(f.sums, 0);
        @memset(f.su, 1);
        @memset(f.sv, 1);
        @memset(f.bias, 0);
        @memset(f.weights, 1);
        @memset(f.norm_weight, 1);
        f.signs = false;
        f.weighted = true;
        for (0..f.factors.len / 2) |pair| {
            f.factors[2 * pair] = 1;
            f.factors[2 * pair + 1] = 0;
        }
        for (0..f.shape.rows) |row| {
            for (0..f.shape.k / 128) |block| {
                const at = row * f.stride + block * 128;
                for (math.plant, 0..) |v, j| f.x[at + j] = c.toBf16(@floatCast(v));
                if (f.kind == .swiglu) {
                    for (0..128) |j| f.x[at + j] = c.toBf16(10);
                    const ups = [_]f32{ 0.80078125, 0.00019550323486328125, -0.80078125, -0.000097751617431640625 };
                    for (ups, 0..) |v, j| f.up[at + j] = c.toBf16(v);
                }
                if (f.kind == .out_rope) for (math.plant, 0..) |v, j| {
                    f.sums[row * f.shape.k + block * 128 + j] = @floatCast(v);
                };
            }
        }
        if (f.kind == .out_in or f.kind == .qkv) {
            for (0..f.descriptors) |id| for (0..f.shape.k / 128) |block| {
                for (math.plant, 0..) |v, j| f.bias[id * f.shape.k + block * 128 + j] = @floatCast(v);
            };
            if (f.kind == .qkv) @memset(f.bias[f.shape.k..], 1);
        }
        if (f.kind == .gate_up) {
            for (0..f.descriptors) |id| {
                if (id < f.expert_count or id == 2 * f.expert_count) {
                    @memset(f.bias[id * f.shape.k ..][0..f.shape.k], 10);
                } else {
                    for (0..f.shape.k / 128) |block| {
                        const ups = [_]f16{ 0.80078125, 0.00019550323486328125, -0.80078125, -0.000097751617431640625 };
                        @memcpy(f.bias[id * f.shape.k + block * 128 ..][0..4], &ups);
                    }
                }
            }
        }
        if (f.kind == .down) {
            const values = [_]f16{ 4096, 1, -8192, -0.5 };
            const ids = [_]usize{ 0, 1, 382, 383 };
            for (values, ids) |v, id| @memset(f.bias[id * f.shape.k ..][0..f.shape.k], v);
        }
    }

    pub fn outputCount(f: Fixture) usize {
        const k = f.shape.k;
        return f.shape.rows * switch (f.kind) {
            .swiglu => k,
            .out_rope, .down => k,
            .out_in, .rms, .rope_in => 2 * k,
            .gate_up => 7 * k + 1,
            .qkv => 2 * k + 1040,
        };
    }

    fn scales(f: Fixture, src: []const f16, id: usize, out: []f64) void {
        if (f.signs) {
            const words = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(src));
            for (out, 0..) |*v, i| v.* = if ((words[id * f.shape.k + i / 16] >> @as(u4, @intCast(i % 16))) & 1 != 0) -1 else 1;
        } else for (out, src[id * f.shape.k ..][0..out.len]) |*v, s| v.* = s;
    }

    fn projected(f: Fixture, row: usize, id: usize, out: []f64, acc: c.Accum, order: g.Order) void {
        var sums: [max_width]f64 = undefined;
        var sv: [max_width]f64 = undefined;
        var bias: [max_width]f64 = undefined;
        const k = f.shape.k;
        for (sums[0..k], f.sums[row * k ..][0..k]) |*v, s| v.* = s;
        for (bias[0..k], f.bias[id * k ..][0..k]) |*v, s| v.* = s;
        f.scales(f.sv, id, sv[0..k]);
        math.rotateOut(sums[0..k], sv[0..k], bias[0..k], out, acc, order);
    }

    pub fn evaluate(f: Fixture, acc: c.Accum, order: g.Order, out: []f64) !void {
        const k = f.shape.k;
        const n = f.shape.rows;
        var a: [max_width]f64 = undefined;
        var b: [max_width]f64 = undefined;
        var d: [max_width]f64 = undefined;
        var su: [max_width]f64 = undefined;
        var weight: [max_width]f64 = undefined;
        for (weight[0..k], f.norm_weight) |*v, w| v.* = w;
        for (0..n) |row| {
            const dst = f.destinations[row];
            const output = out[row * (f.outputCount() / n) ..][0 .. f.outputCount() / n];
            const id = if (f.kind == .qkv) @as(u32, 0) else f.routes[row];
            f.scales(f.su, id, su[0..k]);
            switch (f.kind) {
                .swiglu => {
                    for (a[0..k], 0..) |*v, i| v.* = math.swiglu(c.fromBf16(f.x[row * f.stride + i]), c.fromBf16(f.up[row * f.stride + i]), if (f.weighted) f.weights[row] else 1);
                    math.rotateIn(a[0..k], su[0..k], out[@as(usize, dst) * k ..][0..k], acc, order);
                },
                .out_in => {
                    f.projected(row, id, a[0..k], acc, order);
                    @memcpy(out[@as(usize, dst) * k ..][0..k], a[0..k]);
                    const words = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(f.su));
                    for (su[0..k], 0..) |*v, i| {
                        const at = @as(usize, dst) * k + i;
                        v.* = if (f.signs) (if ((words[at / 16] >> @as(u4, @intCast(at % 16))) & 1 != 0) @as(f64, -1) else 1) else f.su[at];
                    }
                    math.rotateIn(a[0..k], su[0..k], out[n * k + @as(usize, dst) * k ..][0..k], acc, order);
                },
                .rms => {
                    for (a[0..k], f.x[row * f.stride ..][0..k]) |*v, bits| v.* = c.fromBf16(bits);
                    math.rms(a[0..k], weight[0..k], f.eps, b[0..k], acc, order);
                    @memcpy(out[row * k ..][0..k], b[0..k]);
                    f.scales(f.su, 0, su[0..k]);
                    math.rotateIn(b[0..k], su[0..k], out[n * k + row * k ..][0..k], acc, order);
                },
                .out_rope => {
                    f.projected(row, 0, a[0..k], acc, order);
                    math.rope(a[0..k], f.head_width, f.rotary, f.factors[f.factorOffset(row)..][0..64], f.inverse, output);
                },
                .rope_in => {
                    for (a[0..k], f.x[row * f.stride ..][0..k]) |*v, bits| v.* = c.fromBf16(bits);
                    math.rope(a[0..k], f.head_width, f.rotary, f.factors[f.factorOffset(row)..][0..64], f.inverse, b[0..k]);
                    @memcpy(out[row * k ..][0..k], b[0..k]);
                    math.rotateIn(b[0..k], su[0..k], out[n * k + row * k ..][0..k], acc, order);
                },
                .gate_up => {
                    for (0..7) |slot| {
                        const routed = slot < 6;
                        const picked = row * 6 + slot;
                        const expert = if (routed) f.routes[picked] else f.expert_count;
                        const gate_row = if (routed) picked else 12 * n + row;
                        const up_row = if (routed) 6 * n + picked else 13 * n + row;
                        f.projected(gate_row, if (routed) expert else 2 * f.expert_count, a[0..k], acc, order);
                        f.projected(up_row, if (routed) f.expert_count + expert else 2 * f.expert_count + 1, b[0..k], acc, order);
                        for (a[0..k], b[0..k], d[0..k]) |gate, up, *v| v.* = math.swiglu(gate, up, if (routed and f.weighted) f.weights[picked] else 1);
                        f.scales(f.su, expert, su[0..k]);
                        const dest_row = if (routed) picked else 6 * n + row;
                        math.rotateIn(d[0..k], su[0..k], out[dest_row * k ..][0..k], acc, order);
                    }
                    out[7 * n * k + row] = @floatFromInt(f.expert_count);
                },
                .down => {
                    var slots: [6]usize = .{ 0, 1, 2, 3, 4, 5 };
                    for (1..6) |i| {
                        const slot = slots[i];
                        var j = i;
                        while (j > 0 and f.routes[row * 6 + slots[j - 1]] > f.routes[row * 6 + slot]) : (j -= 1) slots[j] = slots[j - 1];
                        slots[j] = slot;
                    }
                    var values: [7][max_width]f64 = undefined;
                    for (slots, 0..) |slot, rank| f.projected(row * 6 + slot, f.routes[row * 6 + slot], values[rank][0..k], acc, order);
                    f.projected(6 * n + row, f.expert_count, values[6][0..k], acc, order);
                    for (output, 0..) |*v, i| {
                        var terms: [7]f64 = undefined;
                        for (&terms, 0..) |*t, rank| t.* = values[rank][i];
                        v.* = math.bf(contract.rows.sumIn(acc, order, &terms));
                    }
                },
                .qkv => {
                    f.projected(2 * row, 0, a[0..k], acc, order);
                    math.rms(a[0..k], weight[0..k], f.eps, b[0..k], acc, order);
                    @memcpy(output[0..k], b[0..k]);
                    f.scales(f.su, 0, su[0..k]);
                    math.rotateIn(b[0..k], su[0..k], output[k..][0..k], acc, order);
                    f.projected(2 * row + 1, 1, a[0..k], acc, order);
                    @memcpy(output[2 * k ..][0..512], a[0..512]);
                    @memset(su[0..512], 1);
                    math.rms(a[0..512], su[0..512], f.eps, b[0..512], acc, order);
                    math.rope(b[0..512], 512, f.rotary, f.factors[f.factorOffset(row)..][0..64], false, d[0..512]);
                    var key_pack: [528]u8 = undefined;
                    math.packKey(d[0..512], &key_pack);
                    for (output[2 * k + 512 ..], key_pack) |*v, byte| v.* = @floatFromInt(byte);
                },
            }
        }
    }
    pub fn factorOffset(f: Fixture, row: usize) usize {
        return (f.position + (if (f.kind == .rope_in) row / 8 else row)) * f.rotary;
    }
};
