//! Qwen image kernels and text M-RoPE: guarded production launches against separate FP64 formulas.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const vision = @import("qwen_hip_vision");
const kit = @import("contract_kit.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const sums = core.contract.rows;
const rope = @import("contract_qwen_rope_ref.zig");
const Kind = enum { linear, embed, norm, qkv, attention, gelu_tanh, gelu_erf, residual, cast, text_rope, leaky_attention };
pub const Profile = enum { full, smoke };
const kinds = std.enums.values(Kind);
const control_symbols = [_][:0]const u8{ "tf_ctl_qvh_qkv_unrounded", "tf_ctl_qvh_text_trig_unrounded", "tf_ctl_qvh_text_products_unrounded" };

fn depth(s: c.Shape) usize {
    return s.k + s.rows * 2 + 32;
}
const dot_smoke = [_]c.Shape{.{ .rows = 3, .n = 17, .k = 97 }};
const dot_full = [_]c.Shape{ .{ .rows = 4, .n = 768, .k = 1536 }, .{ .rows = 4, .n = 2560, .k = 4096 } };
const row_smoke = [_]c.Shape{.{ .rows = 3, .n = 65, .k = 65 }};
const row_full = [_]c.Shape{ .{ .rows = 7, .n = 768, .k = 768 }, .{ .rows = 4, .n = 1024, .k = 1024 }, .{ .rows = 3, .n = 4096, .k = 4096 } };
const qkv_smoke = [_]c.Shape{.{ .rows = 5, .n = 2, .k = 64 }};
const qkv_full = [_]c.Shape{ .{ .rows = 196, .n = 12, .k = 64 }, .{ .rows = 256, .n = 16, .k = 64 }, .{ .rows = 16384, .n = 16, .k = 64 } };
const att_smoke = [_]c.Shape{.{ .rows = 65, .n = 2, .k = 64 }};
const att_full = [_]c.Shape{ .{ .rows = 196, .n = 12, .k = 64 }, .{ .rows = 1024, .n = 16, .k = 64 }, .{ .rows = 16384, .n = 16, .k = 64 } };
const text_smoke = [_]c.Shape{.{ .rows = 3, .n = 2, .k = 256 }};
const text_full = [_]c.Shape{ .{ .rows = 7, .n = 8, .k = 256 }, .{ .rows = 256, .n = 16, .k = 256 } };

fn spec(k: Kind, p: Profile) c.Spec {
    const served: []const c.Shape = switch (k) {
        .linear => if (p == .smoke) &dot_smoke else &dot_full,
        .qkv => if (p == .smoke) &qkv_smoke else &qkv_full,
        .attention, .leaky_attention => if (p == .smoke) &att_smoke else &att_full,
        .text_rope => if (p == .smoke) &text_smoke else &text_full,
        else => if (p == .smoke) &row_smoke else &row_full,
    };
    const names = comptime blk: {
        var ns: [kinds.len][]const u8 = undefined;
        for (kinds, 0..) |value, i| ns[i] = "HIP Qwen vision " ++ @tagName(value);
        break :blk ns;
    };
    return .{
        .name = names[@backingInt(k)],
        .accum = switch (k) {
            .linear, .norm, .attention, .leaky_attention => .f32,
            else => .none,
        },
        .depth = depth,
        .ops = 32,
        .mids = 2,
        .out = if (k == .linear) .f32 else .bf16,
        .served = served,
        .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "BF16 branch rounding and nonlinear normalization need the FP64 formula and accumulator-order checks" },
    };
}

pub const Bundle = struct {
    a: std.mem.Allocator,
    ops: vision.Kernels,
    cases: [kinds.len + control_symbols.len]Case,
    pub fn deinit(b: *Bundle) void {
        for (&b.cases) |*x| x.deinit();
        b.ops.deinit();
        b.a.destroy(b);
    }
};

/// The `qwen_vision` contract entry (contract_families.zig): every case at `profile`'s shapes through gates 1-4.
pub fn run(a: std.mem.Allocator, _: std.Io, profile: @import("contract_entry.zig").Profile) anyerror!g.Tally {
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var l = try hip.Launcher.load(&r, .{}, hip.kernels.images);
    defer l.unload();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    const bundle = try createProfile(a, &r, s, &l, &kernels, if (profile == .smoke) .smoke else .full);
    defer bundle.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, kernels.items, 0x6b63, &t);
    return t;
}

pub fn create(a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, list: *std.ArrayList(g.Kernel)) !*Bundle {
    return createProfile(a, r, s, l, list, .full);
}

pub fn createProfile(a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, list: *std.ArrayList(g.Kernel), profile: Profile) !*Bundle {
    _ = l;
    const b = try a.create(Bundle);
    errdefer a.destroy(b);
    b.a = a;
    b.ops = try vision.Kernels.load(r, hip.kernels.images[@backingInt(hip.kernels.Group.qwen_vision)]);
    errdefer b.ops.deinit();
    var made: usize = 0;
    errdefer for (b.cases[0..made]) |*x| x.deinit();
    for (b.cases[0..kinds.len], kinds) |*x, k| {
        x.* = .{ .rig = .{ .a = a, .r = r, .s = s }, .arena = .init(a), .ops = &b.ops, .kind = k, .profile = profile };
        made += 1;
    }
    for (b.cases[kinds.len..], control_symbols, 0..) |*x, symbol, i| {
        x.* = .{ .rig = .{ .a = a, .r = r, .s = s }, .arena = .init(a), .ops = &b.ops, .kind = if (i == 0) .qkv else .text_rope, .profile = profile, .control_fn = try b.ops.mod.function(symbol), .control_name = symbol };
        made += 1;
    }
    const before = list.items.len;
    errdefer list.shrinkRetainingCapacity(before);
    for (&b.cases) |*x| {
        var registration = spec(x.kind, profile);
        if (x.control_name) |name| registration.name = name;
        try list.append(a, .{ .spec = registration, .ctx = x, .vt = if (x.kind == .qkv or x.kind == .text_rope) &Case.rot_vt else &Case.vt, .control = if (x.control_fn != null) .shapes else if (x.kind == .leaky_attention) .nonfinite else null });
    }
    return b;
}

fn b16(x: f64) f64 {
    return kit.round(.bf16, x);
}

pub const Case = struct {
    rig: Rig,
    arena: std.heap.ArenaAllocator,
    ops: *const vision.Kernels,
    kind: Kind,
    profile: Profile,
    shape: c.Shape = undefined,
    x: []f64 = &.{},
    w: []f64 = &.{},
    bias: []f64 = &.{},
    y: []f64 = &.{},
    z: []f64 = &.{},
    pos: []i32 = &.{},
    idx: []i32 = &.{},
    weights: []f64 = &.{},
    inp: [8]u64 = @splat(0),
    out: [3]u64 = @splat(0),
    initial: []u8 = &.{},
    rotations: []rope.Rotation = &.{},
    control_fn: ?hip.Function = null,
    control_name: ?[]const u8 = null,
    const rot_vt = g.hooks(Case);
    const vt = blk: {
        var hooks = g.hooks(Case);
        hooks.intervals = null;
        hooks.bounds = null;
        break :blk hooks;
    };

    pub fn deinit(t: *Case) void {
        t.rig.deinit();
        t.arena.deinit();
    }
    fn array(t: *Case, n: usize, rng: std.Random, scale: f64, bf: bool) ![]f64 {
        const v = try t.arena.allocator().alloc(f64, n);
        for (v) |*x| {
            const r = (rng.float(f64) * 2 - 1) * scale;
            x.* = if (bf) b16(r) else @as(f32, @floatCast(r));
        }
        return v;
    }
    fn input(t: *Case, values: []const f64, bf: bool) !u64 {
        const bytes = try t.arena.allocator().alloc(u8, values.len * @as(usize, if (bf) 2 else 4));
        for (values, 0..) |x, i| kit.put(if (bf) .bf16 else .f32, bytes, i, x);
        return t.rig.input(bytes);
    }

    fn split(t: *Case) usize {
        return if (t.shape.rows == 16384) t.shape.rows else @min(@as(usize, 3), t.shape.rows);
    }

    pub fn prepare(t: *Case, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.rows > 16384 or s.n == 0 or s.n > 8192 or s.k == 0 or s.k > 8192) return false;
        if ((t.kind == .qkv or t.kind == .attention or t.kind == .leaky_attention) and (s.k != 64 or s.n > 32)) return false;
        if (t.kind == .text_rope and (s.k != 256 or s.n > 32)) return false;
        if (t.kind != .linear and t.kind != .qkv and t.kind != .attention and t.kind != .leaky_attention and t.kind != .text_rope and s.n != s.k) return false;
        t.rig.reset();
        _ = t.arena.reset(.retain_capacity);
        t.shape = s;
        t.inp = @splat(0);
        t.out = @splat(0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const a = t.arena.allocator();
        const heads = s.n;
        const width = if (t.kind == .qkv or t.kind == .attention or t.kind == .leaky_attention or t.kind == .text_rope) heads * s.k else s.n;
        const n = s.rows * width;
        t.x = try t.array(if (t.kind == .linear) s.rows * s.k else if (t.kind == .qkv) 3 * n else n, rng, 0.75, t.kind == .linear or t.kind == .norm or t.kind == .text_rope);
        t.bias = try t.array(if (t.kind == .qkv) 3 * width else width, rng, 0.125, true);
        t.w = try t.array(if (t.kind == .linear) s.n * s.k else if (t.kind == .embed) 4 * width else width, rng, if (t.kind == .linear) 0.125 else 1, true);
        t.y = try t.array(n, rng, 0.5, true);
        t.z = try t.array(n, rng, 0.5, true);
        if (fill == .nonfinite) {
            const special = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (special, 0..) |value, i| if (i * width < t.x.len) {
                t.x[i * width] = value;
            };
        }
        if (t.kind == .attention or t.kind == .leaky_attention) {
            for (t.x, t.y, t.z, 0..) |*q, *k, *v, i| {
                const row = (i / 64) % s.rows;
                const h = i / (64 * s.rows);
                const d = i % 64;
                const image: usize = @intFromBool(row >= t.split());
                const local = if (image == 0) row else row - t.split();
                q.* = b16(@as(f64, @floatFromInt((local * 11 + d * 3 + h * 7) % 31)) / 32 - 0.5 + @as(f64, @floatFromInt(image)));
                k.* = b16(@as(f64, @floatFromInt((local % 8 * 5 + d * 7 + h * 3) % 29)) / 32 - 0.25);
                v.* = b16(@as(f64, @floatFromInt((local % 8 * 7 + d * 11 + h) % 23)) / 32 + @as(f64, @floatFromInt(image)) * 2);
            }
            if (fill == .nonfinite) t.y[0] = std.math.nan(f64);
            t.inp[0] = try t.input(t.x, true);
            t.inp[1] = try t.input(t.y, true);
            t.inp[2] = try t.input(t.z, true);
        } else if (t.kind == .embed) {
            t.idx = try a.alloc(i32, s.rows * 4);
            t.weights = try a.alloc(f64, s.rows * 4);
            for (t.idx, t.weights, 0..) |*index, *w, i| {
                index.* = @intCast((i + i / 4) % 4);
                w.* = (&[_]f64{ 0.125, 0.25, 0.375, 0.25 })[i % 4];
            }
            t.inp[0] = try t.input(t.x, false);
            t.inp[1] = try t.input(t.bias, false);
            t.inp[2] = try t.input(t.w, false);
            t.inp[3] = try t.rig.input(std.mem.sliceAsBytes(t.idx));
            t.inp[4] = try t.input(t.weights, false);
        } else if (t.kind == .qkv) {
            t.pos = try a.alloc(i32, s.rows * 2);
            for (t.pos, 0..) |*p, i| p.* = @intCast((i * 7 + i / 2) % 53);
            t.weights = try a.alloc(f64, 16);
            for (t.weights, 0..) |*v, i| v.* = @as(f32, @floatCast(1 / std.math.pow(f64, 10000, @as(f64, @floatFromInt(i)) / 16)));
            if (fill != .nonfinite) {
                t.x[0] = @as(f32, 0.1684);
                t.x[32] = 0.40625;
                t.bias[0] = 0;
                t.bias[32] = 0;
                t.pos[0] = 1;
            }
            t.inp[0] = try t.input(t.x, false);
            t.inp[1] = try t.input(t.bias, false);
            t.inp[2] = try t.rig.input(std.mem.sliceAsBytes(t.pos));
            t.inp[3] = try t.input(t.weights, false);
        } else if (t.kind == .text_rope) {
            t.pos = try a.alloc(i32, s.rows * 3);
            for (t.pos, 0..) |*p, i| p.* = @as(i32, @intCast((i / 3) * 11 + (i % 3) * 23)) - 7;
            if (fill != .nonfinite) {
                t.x[0] = 0.16796875;
                t.x[32] = 0.40625;
                t.pos[0] = 1;
            }
            t.inp[0] = try t.input(t.x, true);
            t.inp[1] = try t.rig.input(std.mem.sliceAsBytes(t.pos));
        } else {
            t.inp[0] = try t.input(t.x, t.kind == .linear or t.kind == .norm);
            t.inp[1] = try t.input(t.w, false);
            t.inp[2] = try t.input(t.bias, false);
        }
        if (t.kind == .qkv or t.kind == .text_rope) t.rotations = try rope.tables(a, t);
        t.out[0] = try t.rig.output(t.count() * c.size(spec(t.kind, t.profile).out) / @as(usize, if (t.kind == .qkv) 3 else 1));
        if (t.kind == .qkv) {
            t.out[1] = try t.rig.output(n * 2);
            t.out[2] = try t.rig.output(n * 2);
        }
        if (t.kind == .residual) {
            t.initial = try a.alloc(u8, n * 2);
            for (t.y, 0..) |value, i| kit.put(.bf16, t.initial, i, value);
        }
        return true;
    }

    pub fn count(t: *Case) usize {
        const s = t.shape;
        return s.rows * s.n * @as(usize, (if (t.kind == .qkv) 192 else if (t.kind == .attention or t.kind == .leaky_attention) 64 else if (t.kind == .text_rope) 256 else 1));
    }

    pub fn run(t: *Case) !void {
        const s = t.shape;
        const p = t.inp;
        const o = t.out;
        const k = t.ops;
        const stream = t.rig.s;
        if (t.control_fn) |f| {
            var args: hip.launch.Args = .{};
            if (t.kind == .qkv) {
                for ([_]u64{ p[0], p[1], p[2], p[3], o[0], o[1], o[2] }) |v| try args.add(v);
                try args.add(@as(c_int, @intCast(s.rows)));
                try args.add(@as(c_int, @intCast(s.n)));
            } else {
                inline for (.{ p[0], @as(c_int, 2), o[0], @as(c_int, 2), @as(i64, @intCast(s.rows * 256)), @as(i64, 256), @as(c_int, @intCast(s.rows)), @as(c_int, @intCast(s.n)), @as(c_int, 256), @as(c_int, 64), p[1], @as(f32, 10000000) }) |v| try args.add(v);
            }
            const cfg: hip.launch.Config = if (t.kind == .qkv) .{ .grid = .{ .x = @intCast(s.rows), .y = @intCast(s.n) }, .block = .{ .x = 64 } } else .{ .grid = .{ .x = @intCast(s.rows * s.n) }, .block = .{ .x = 256 } };
            try hip.launch.launch(f, cfg, stream, &args);
            try stream.synchronize();
            return;
        }
        switch (t.kind) {
            .linear => try k.linear(stream, p[0], p[1], o[0], s.rows, s.n, s.k),
            .embed => try k.embed(stream, p[0], p[1], p[2], p[3], p[4], o[0], s.rows, s.n),
            .norm => try k.layernorm(stream, p[0], p[1], p[2], o[0], s.rows, s.n, 1e-6),
            .qkv => try k.qkv(stream, p[0], p[1], p[2], p[3], o[0], o[1], o[2], s.rows, s.n),
            .attention => {
                const first = t.split();
                try k.attention(stream, p[0], p[1], p[2], o[0], s.rows, s.n, 0, first);
                if (first < s.rows) try k.attention(stream, p[0], p[1], p[2], o[0], s.rows, s.n, first, s.rows - first);
            },
            .leaky_attention => try k.attention(stream, p[0], p[1], p[2], o[0], s.rows, s.n, 0, s.rows),
            .gelu_tanh, .gelu_erf => try k.gelu(stream, p[0], p[2], o[0], s.rows, s.n, t.kind == .gelu_tanh),
            .cast => try k.cast(stream, p[0], p[2], o[0], s.rows, s.n),
            .residual => {
                try t.rig.put(o[0], t.initial);
                try k.residual(stream, p[0], p[2], o[0], s.rows, s.n);
            },
            .text_rope => try k.textRope(stream, .{ .ptr = p[0], .kind = .bf16 }, .{ .ptr = o[0], .kind = .bf16 }, s.rows * 256, 256, s.rows, s.n, 256, 64, p[1], 10000000),
        }
        try stream.synchronize();
    }

    pub fn reference(t: *Case, want: []f64, mag: []f64) !void {
        try t.evaluate(want, mag, .none, .sequential);
    }
    pub fn bounds(t: *Case, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = mag;
        rope.outputs(t, null, null, want, allow);
    }
    pub fn intervals(t: *Case, low: []f64, high: []f64) !void {
        rope.outputs(t, low, high, null, null);
    }
    pub fn simulate(t: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.arena.allocator().alloc(f64, out.len);
        try t.evaluate(out, mag, acc, order);
    }

    fn evaluate(t: *Case, want: []f64, mag: []f64, acc: c.Accum, order: g.Order) !void {
        const s = t.shape;
        const width = s.n;
        @memset(mag, 0);
        const terms = try t.arena.allocator().alloc(f64, @max(s.k, s.rows));
        switch (t.kind) {
            .linear => for (0..s.rows) |r| {
                for (0..s.n) |d| {
                    for (terms[0..s.k], 0..) |*v, j| v.* = t.x[r * s.k + j] * t.w[d * s.k + j];
                    const at = r * s.n + d;
                    want[at] = total(terms[0..s.k], acc, order);
                    for (terms[0..s.k]) |v| mag[at] += @abs(v);
                }
            },
            .norm => for (0..s.rows) |r| {
                const row = t.x[r * width ..][0..width];
                const mean = total(row, acc, order) / @as(f64, @floatFromInt(width));
                for (terms[0..width], row) |*v, x| v.* = (x - mean) * (x - mean);
                const inv = 1 / @sqrt(total(terms[0..width], acc, order) / @as(f64, @floatFromInt(width)) + 1e-6);
                for (row, 0..) |x, d| {
                    const at = r * width + d;
                    want[at] = (x - mean) * inv * t.w[d] + t.bias[d];
                    mag[at] = @abs(want[at]) + @abs(t.bias[d]);
                }
            },
            .embed => for (want, 0..) |*out, i| {
                const r = i / width;
                const d = i % width;
                var p: f64 = 0;
                for (0..4) |j| p = b16(p + b16(t.w[@as(usize, @intCast(t.idx[r * 4 + j])) * width + d] * b16(t.weights[r * 4 + j])));
                out.* = b16(t.x[i] + t.bias[d]) + p;
            },
            .qkv => for (0..3) |which| {
                for (0..s.n) |h| {
                    for (0..s.rows) |r| {
                        for (0..64) |d| {
                            const src = r * 3 * s.n * 64 + which * s.n * 64 + h * 64;
                            const j = d % 32;
                            const dst = ((which * s.n + h) * s.rows + r) * 64 + d;
                            if (which == 2) {
                                want[dst] = t.x[src + d] + t.bias[which * s.n * 64 + h * 64 + d];
                                continue;
                            }
                            const angle = @as(f32, @floatFromInt(t.pos[r * 2 + j / 16])) * @as(f32, @floatCast(t.weights[j % 16]));
                            const a = b16(t.x[src + d] + t.bias[which * s.n * 64 + h * 64 + d]);
                            const other = if (d < 32) d + 32 else d - 32;
                            const b = b16(t.x[src + other] + t.bias[which * s.n * 64 + h * 64 + other]);
                            want[dst] = a * @cos(@as(f64, angle)) + (if (d < 32) -b else b) * @sin(@as(f64, angle));
                        }
                    }
                }
            },
            .attention, .leaky_attention => try t.attentionRef(want, mag, acc, order, terms),
            .text_rope => for (want, 0..) |*out, i| {
                const d = i % 256;
                const r = (i / 256) % s.rows;
                const h = i / (256 * s.rows);
                const base = (r * s.n + h) * 256;
                if (d >= 64) {
                    out.* = t.x[base + d];
                    continue;
                }
                const j = d % 32;
                const axis: usize = if (j % 3 == 1 and j < 33) 1 else if (j % 3 == 2 and j < 30) 2 else 0;
                const inverse: f32 = @floatCast(1 / std.math.pow(f64, 10000000, @as(f64, @floatFromInt(j)) / 32));
                const angle: f32 = @as(f32, @floatFromInt(t.pos[r * 3 + axis])) * inverse;
                const cs = b16(@cos(angle));
                const sn = b16(@sin(angle));
                out.* = if (d < 32) b16(t.x[base + j] * cs) - b16(t.x[base + j + 32] * sn) else b16(t.x[base + j + 32] * cs) + b16(t.x[base + j] * sn);
            },
            else => for (want, 0..) |*out, i| {
                const x = b16(t.x[i] + t.bias[i % width]);
                out.* = switch (t.kind) {
                    .cast => t.x[i] + t.bias[i % width],
                    .residual => t.y[i] + x,
                    .gelu_tanh => 0.5 * x * (1 + std.math.tanh(@as(f64, 0.7978845608028654) * (x + 0.044715 * x * x * x))),
                    .gelu_erf => 0.5 * x * (1 + erf(x / @sqrt(@as(f64, 2)))),
                    else => unreachable,
                };
            },
        }
    }

    fn attentionRef(t: *Case, want: []f64, mag: []f64, acc: c.Accum, order: g.Order, terms: []f64) !void {
        const s = t.shape;
        const boundary = t.split();
        for (0..s.n) |h| {
            for (0..s.rows) |r| {
                const start: usize = if (r < boundary) 0 else boundary;
                const span = if (r < boundary) boundary else s.rows - boundary;
                var scores: [8]f64 = undefined;
                var maximum: f64 = -std.math.inf(f64);
                var bad = false;
                for (0..@min(span, 8)) |j| {
                    for (terms[0..64], 0..) |*v, d| v.* = t.x[(h * s.rows + r) * 64 + d] * t.y[(h * s.rows + start + j) * 64 + d];
                    scores[j] = b16(b16(total(terms[0..64], acc, order)) * 0.125);
                    bad = bad or !std.math.isFinite(scores[j]);
                    maximum = @max(maximum, scores[j]);
                }
                var denominator: f64 = 0;
                for (0..@min(span, 8)) |j| denominator += @exp(scores[j] - maximum) * @as(f64, @floatFromInt((span + 7 - j) / 8));
                if (acc != .none) {
                    for (terms[0..span], 0..) |*v, j| v.* = @exp(scores[j % 8] - maximum);
                    denominator = total(terms[0..span], acc, order);
                }
                for (0..64) |d| {
                    const dst = (r * s.n + h) * 64 + d;
                    var value: f64 = 0;
                    var magnitude: f64 = 0;
                    for (0..@min(span, 8)) |j| {
                        const p = b16(@exp(scores[j] - maximum) / denominator);
                        const term = p * t.z[(h * s.rows + start + j) * 64 + d];
                        value += term * @as(f64, @floatFromInt((span + 7 - j) / 8));
                        magnitude += @abs(term) * @as(f64, @floatFromInt((span + 7 - j) / 8));
                    }
                    if (acc != .none) {
                        for (terms[0..span], 0..) |*v, j| {
                            const p = b16(@exp(scores[j % 8] - maximum) / denominator);
                            v.* = p * t.z[(h * s.rows + start + j % 8) * 64 + d];
                        }
                        value = total(terms[0..span], acc, order);
                    }
                    want[dst] = if (bad) std.math.nan(f64) else value;
                    mag[dst] = magnitude;
                }
            }
        }
    }
};

fn total(values: []const f64, acc: c.Accum, order: g.Order) f64 {
    if (acc != .none) return sums.sumIn(acc, order, values);
    var sum: f64 = 0;
    for (values) |v| sum += v;
    return sum;
}

extern "c" fn erf(f64) f64;

test "Qwen contract text reference keeps all axes and leaves unrotated channels unchanged" {
    var t: Case = .{ .rig = undefined, .arena = .init(std.testing.allocator), .ops = undefined, .kind = .text_rope, .profile = .smoke };
    defer t.arena.deinit();
    t.shape = .{ .rows = 1, .n = 1, .k = 256 };
    t.x = try t.arena.allocator().alloc(f64, 256);
    @memset(t.x, 1);
    var positions = [_]i32{ 0, 11, -17 };
    t.pos = &positions;
    var want: [256]f64 = undefined;
    var mag: [256]f64 = undefined;
    try t.reference(&want, &mag);
    try std.testing.expectEqual(@as(f64, 1), want[0]);
    try std.testing.expect(want[1] != want[0]);
    try std.testing.expect(want[2] != want[0]);
    try std.testing.expectEqual(@as(f64, 1), want[62]);
    try std.testing.expectEqual(@as(f64, 1), want[64]);
    try std.testing.expectEqual(@as(f64, 1), want[255]);
}

test "Qwen contract attention reference does not carry a first-image NaN into the next image" {
    var t: Case = .{ .rig = undefined, .arena = .init(std.testing.allocator), .ops = undefined, .kind = .attention, .profile = .smoke };
    defer t.arena.deinit();
    t.shape = .{ .rows = 5, .n = 1, .k = 64 };
    const a = t.arena.allocator();
    t.x = try a.alloc(f64, 320);
    t.y = try a.alloc(f64, 320);
    t.z = try a.alloc(f64, 320);
    @memset(t.x, 0);
    @memset(t.y, 0);
    @memset(t.z, 2);
    t.y[0] = std.math.nan(f64);
    var want: [320]f64 = undefined;
    var mag: [320]f64 = undefined;
    try t.reference(&want, &mag);
    for (want[0 .. 3 * 64]) |v| try std.testing.expect(std.math.isNan(v));
    for (want[3 * 64 ..]) |v| try std.testing.expectEqual(@as(f64, 2), v);
}

test "Qwen contract registration compiles each guarded production launch" {
    _ = &create;
    _ = &Case.prepare;
    _ = &Case.run;
    for (kinds) |k| {
        const s = spec(k, .smoke);
        try std.testing.expect(s.served.len > 0);
        try std.testing.expect(s.name.len > 0);
    }
}
