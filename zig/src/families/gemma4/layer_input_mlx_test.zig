//! The host's per-layer input steps against MLX 0.32.4 at the E4B's shapes: prints, full outputs, float64.
const std = @import("std");
const bf16_round = @import("../../core/bf16_round.zig");
const cfg = @import("config.zig");
const li = @import("layer_input.zig");

const rn = li.rn;
const scalar = li.scalar;
const dot = li.dot;
const rms = li.rms;
const gelu = li.gelu;
const inputs = li.inputs;
const apply = li.apply;
const Model = li.Model;
const Layer = li.Layer;
const Ref = li.Ref;
const spread = li.spread;

const mlx_fixture = "tools/zig/fixtures/e4b_layer_input_mlx.bin"; // optional: the oracle's --full outputs, not committed
const mlx_prints = "tools/zig/fixtures/e4b_layer_input_mlx.json"; // SHA-256 prints of MLX's outputs, committed

/// The oracle's generator: splitmix64 of seed + (i + 1) * golden gamma, the top 24 bits over 2^23 minus one.
fn uniform(seed: u64, i: usize) f32 {
    var z = seed +% (@as(u64, i) +% 1) *% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    z ^= z >> 31;
    return @as(f32, @floatFromInt(@as(u32, @intCast(z >> 40)))) / 8388608.0 - 1.0;
}

/// Stream `id` of `n` bf16 values: uniform * scale (+ shift), as the oracle makes them.
fn seeded(a: std.mem.Allocator, id: u64, n: usize, scale: f32, shift: f32) ![]f32 {
    const out = try a.alloc(f32, n);
    for (out, 0..) |*o, i| {
        const v: f32 = uniform(0xE4B00000 + id, i) * scale;
        o.* = rn(if (shift != 0) v + shift else v);
    }
    return out;
}

/// The oracle's next `n` bf16 values, as floats.
const Reader = struct {
    u16s: []const u16,

    fn take(r: *Reader, a: std.mem.Allocator, n: usize) ![]f32 {
        const out = try a.alloc(f32, n);
        for (out, r.u16s[0..n]) |*o, b| o.* = bf16_round.value(b);
        r.u16s = r.u16s[n..];
        return out;
    }
};

/// How many of `got` differ bit for bit from `want` (both bf16 values), and the largest difference.
const Miss = struct {
    differ: usize = 0,
    total: usize = 0,
    worst: f32 = 0,

    fn add(m: *Miss, got: []const f32, want: []const f32) void {
        for (got, want) |g, w| {
            m.total += 1;
            if (@as(u32, @bitCast(g)) != @as(u32, @bitCast(w)) and !(g == 0 and w == 0)) m.differ += 1;
            m.worst = @max(m.worst, @abs(g - w));
        }
    }
};

const mlx_steps = [_][]const u8{ "projection", "scale", "rms norm", "table scale", "sum", "mix", "gate", "gelu", "times input", "back projection", "post rms norm", "residual", "layer scalar" };

/// The oracle's seeded inputs and weights.
const Data = struct {
    h: []f32,
    model: Model,
    tab: []f32,
    gate_w: []f32,
    back_w: []f32,
    post_w: []f32,
    layer_scalar: []f32,
    residual: []f32,

    fn init(a: std.mem.Allocator, hidden: usize, width: usize, layers: usize, tokens: usize) !Data {
        return .{
            .h = try seeded(a, 1, tokens * hidden, 3.0, 0),
            .model = .{ .project = try seeded(a, 2, layers * width * hidden, 0.03, 0), .norm = try seeded(a, 3, width, 0.5, 1.0) },
            .tab = try seeded(a, 4, tokens * layers * width, 0.08, 0),
            .gate_w = try seeded(a, 5, layers * width * hidden, 0.05, 0),
            .back_w = try seeded(a, 6, layers * hidden * width, 0.1, 0),
            .post_w = try seeded(a, 7, layers * hidden, 0.5, 1.0),
            .layer_scalar = try seeded(a, 8, layers, 0.4, 0.5),
            .residual = try seeded(a, 9, tokens * layers * hidden, 12.0, 0),
        };
    }
};

/// SHA-256 of bf16 values as little-endian u16, in lower-case hex.
fn digest(values: []const f32) [64]u8 {
    var sha: std.crypto.hash.sha2.Sha256 = .init(.{});
    for (values) |v| {
        const b = bf16_round.bits(v);
        sha.update(&.{ @truncate(b), @truncate(b >> 8) });
    }
    var out: [32]u8 = undefined;
    sha.final(&out);
    return std.fmt.bytesToHex(out, .lower);
}

const Prints = struct { digests: struct { gelu_finite: []const u8, steps: []const []const u8, layers: []const []const []const []const u8 } };

test "the host's own steps have the SHA-256 prints of MLX's outputs at the E4B's shapes, and its GELU for every bf16" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = std.Io.Dir.cwd().readFileAlloc(std.testing.io, mlx_prints, a, .limited(1 << 20)) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest, // the oracle script writes it (needs MLX 0.32.4)
        else => return err,
    };
    const prints = (try std.json.parseFromSlice(Prints, a, bytes, .{ .ignore_unknown_fields = true })).value.digests;
    const hidden = 2560;
    const width = 256;
    const layers = prints.layers[0].len;
    const tokens = prints.layers.len;
    try std.testing.expectEqual(@as(usize, 13), prints.steps.len);

    var finite = try a.alloc(f32, 1 << 16);
    var count: usize = 0;
    for (0..1 << 16) |pattern| {
        const x = bf16_round.value(@intCast(pattern));
        if (!std.math.isFinite(x)) continue;
        finite[count] = gelu(x);
        count += 1;
    }
    try std.testing.expectEqualStrings(prints.gelu_finite, &digest(finite[0..count]));

    const d = try Data.init(a, hidden, width, layers, tokens);
    var geo = cfg.e4b;
    geo.layers = layers;
    const scale_h = scalar(1.0 / @sqrt(@as(f64, @floatFromInt(hidden))));
    const token = scalar(@sqrt(@as(f64, @floatFromInt(width))));
    const mix = scalar(std.math.sqrt1_2);
    const own = try a.alloc(f32, 13 * hidden);
    const whole = try a.alloc(f32, layers * width);
    const row = try a.alloc(f32, hidden);
    const scratch = try a.alloc(f32, hidden + width);
    for (0..tokens) |t| {
        // inputs() whole, against the sixth step of every layer
        inputs(geo, d.model, d.h[t * hidden ..][0..hidden], d.tab[t * layers * width ..][0 .. layers * width], whole);
        for (0..layers) |l| {
            // each step fed by the host's own previous step
            const st = struct {
                fn at(buf: []f32, i: usize, n: usize) []f32 {
                    return buf[i * hidden ..][0..n];
                }
            }.at;
            const res = d.residual[(t * layers + l) * hidden ..][0..hidden];
            const layer: Layer = .{ .gate = d.gate_w[l * width * hidden ..][0 .. width * hidden], .project = d.back_w[l * hidden * width ..][0 .. hidden * width], .norm = d.post_w[l * hidden ..][0..hidden], .scalar = d.layer_scalar[l] };
            for (st(own, 0, width), 0..) |*p, j| p.* = rn(dot(d.model.project[(l * width + j) * hidden ..][0..hidden], d.h[t * hidden ..][0..hidden]));
            for (st(own, 1, width), st(own, 0, width)) |*p, v| p.* = rn(v * scale_h);
            rms(st(own, 1, width), d.model.norm, st(own, 2, width));
            for (st(own, 3, width), d.tab[(t * layers + l) * width ..][0..width]) |*p, v| p.* = rn(v * token);
            for (st(own, 4, width), st(own, 2, width), st(own, 3, width)) |*p, x, y| p.* = rn(x + y);
            for (st(own, 5, width), st(own, 4, width)) |*p, v| p.* = rn(v * mix);
            for (st(own, 6, width), 0..) |*p, j| p.* = rn(dot(layer.gate[j * hidden ..][0..hidden], res));
            for (st(own, 7, width), st(own, 6, width)) |*p, v| p.* = gelu(v);
            for (st(own, 8, width), st(own, 7, width), st(own, 5, width)) |*p, x, y| p.* = rn(x * y);
            for (st(own, 9, hidden), 0..) |*p, j| p.* = rn(dot(layer.project[j * width ..][0..width], st(own, 8, width)));
            rms(st(own, 9, hidden), layer.norm, st(own, 10, hidden));
            for (st(own, 11, hidden), res, st(own, 10, hidden)) |*p, x, y| p.* = rn(x + y);
            for (st(own, 12, hidden), st(own, 11, hidden)) |*p, v| p.* = rn(v * layer.scalar);
            const lens = [13]usize{ width, width, width, width, width, width, width, width, width, hidden, hidden, hidden, hidden };
            for (prints.steps, lens, 0..) |name, n, i| {
                const want = prints.layers[t][l][i];
                const got = digest(st(own, i, n));
                if (!std.mem.eql(u8, want, &got)) {
                    std.debug.print("first step that differs from MLX: token {d}, layer {d}, \"{s}\"\n", .{ t, l, name });
                    return error.StepDiffersFromMlx;
                }
            }
            // inputs() and apply() as wholes agree with the chain
            try std.testing.expectEqualStrings(prints.layers[t][l][5], &digest(whole[l * width ..][0..width]));
            @memcpy(row, res);
            apply(geo, layer, st(own, 5, width), row, scratch);
            try std.testing.expectEqualStrings(prints.layers[t][l][12], &digest(row));
        }
    }
}

test "fed MLX's outputs, the host's steps equal them bit for bit at the E4B's shapes, and its GELU for every bf16" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = std.Io.Dir.cwd().readFileAlloc(std.testing.io, mlx_fixture, a, .limited(1 << 24)) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest, // `python tools/zig/gemma4_e4b_layer_input_oracle.py --full` writes it
        else => return err,
    };
    try std.testing.expectEqualStrings("E4BL", bytes[0..4]);
    const head = std.mem.bytesAsSlice(u32, bytes[4..24]);
    const hidden: usize = head[1];
    const width: usize = head[2];
    const layers: usize = head[3];
    const tokens: usize = head[4];
    try std.testing.expectEqual(@as(u32, 1), head[0]);
    try std.testing.expectEqual(@as(usize, 2560), hidden);
    try std.testing.expectEqual(@as(usize, 256), width);
    var reader: Reader = .{ .u16s = @alignCast(std.mem.bytesAsSlice(u16, bytes[24..])) };

    // GELU over every bf16 pattern
    const table = try reader.take(a, 1 << 16);
    var gelu_miss: Miss = .{};
    for (table, 0..) |want, pattern| {
        const x = bf16_round.value(@intCast(pattern));
        if (!std.math.isFinite(x)) continue;
        gelu_miss.add(&.{gelu(x)}, &.{want});
    }
    std.debug.print("gelu: {d} of {d} finite bf16 differ\n", .{ gelu_miss.differ, gelu_miss.total });
    try std.testing.expectEqual(@as(usize, 0), gelu_miss.differ);

    // the oracle's inputs
    const d = try Data.init(a, hidden, width, layers, tokens);
    const h = d.h;
    const model = d.model;
    const tab = d.tab;
    const gate_w = d.gate_w;
    const back_w = d.back_w;
    const post_w = d.post_w;
    const layer_scalar = d.layer_scalar;
    const residual = d.residual;

    var geo = cfg.e4b;
    geo.layers = layers;
    const scale_h = scalar(1.0 / @sqrt(@as(f64, @floatFromInt(hidden))));
    const token = scalar(@sqrt(@as(f64, @floatFromInt(width))));
    const mix = scalar(std.math.sqrt1_2);
    var miss: [mlx_steps.len]Miss = @splat(.{});
    var whole_inputs: Miss = .{};
    var whole_block: Miss = .{};
    const buf = try a.alloc(f32, hidden + width);
    for (0..tokens) |t| {
        // inputs(): this token's per-layer inputs at once, against the last step
        const end_to_end = try a.alloc(f32, layers * width);
        inputs(geo, model, h[t * hidden ..][0..hidden], tab[t * layers * width ..][0 .. layers * width], end_to_end);
        for (0..layers) |l| {
            const proj = try reader.take(a, width);
            const scaled = try reader.take(a, width);
            const normed = try reader.take(a, width);
            const tok = try reader.take(a, width);
            const summed = try reader.take(a, width);
            const final = try reader.take(a, width);
            // each step from MLX's output of the one before
            const mine = buf[0..width];
            for (mine, 0..) |*p, j| p.* = rn(dot(model.project[(l * width + j) * hidden ..][0..hidden], h[t * hidden ..][0..hidden]));
            miss[0].add(mine, proj);
            for (mine, proj) |*p, v| p.* = rn(v * scale_h);
            miss[1].add(mine, scaled);
            rms(scaled, model.norm, mine);
            miss[2].add(mine, normed);
            for (mine, tab[(t * layers + l) * width ..][0..width]) |*p, v| p.* = rn(v * token);
            miss[3].add(mine, tok);
            for (mine, normed, tok) |*p, x, y| p.* = rn(x + y);
            miss[4].add(mine, summed);
            for (mine, summed) |*p, v| p.* = rn(v * mix);
            miss[5].add(mine, final);
            whole_inputs.add(end_to_end[l * width ..][0..width], final);
            // the block of this layer on the oracle's residual row
            const gate = try reader.take(a, width);
            const act = try reader.take(a, width);
            const up = try reader.take(a, width);
            const back = try reader.take(a, hidden);
            const post = try reader.take(a, hidden);
            const added = try reader.take(a, hidden);
            const out = try reader.take(a, hidden);
            const res = residual[(t * layers + l) * hidden ..][0..hidden];
            const layer: Layer = .{ .gate = gate_w[l * width * hidden ..][0 .. width * hidden], .project = back_w[l * hidden * width ..][0 .. hidden * width], .norm = post_w[l * hidden ..][0..hidden], .scalar = layer_scalar[l] };
            for (mine, 0..) |*p, j| p.* = rn(dot(layer.gate[j * hidden ..][0..hidden], res));
            miss[6].add(mine, gate);
            for (mine, gate) |*p, v| p.* = gelu(v);
            miss[7].add(mine, act);
            for (mine, act, final) |*p, x, y| p.* = rn(x * y);
            miss[8].add(mine, up);
            const wide = try a.alloc(f32, hidden);
            for (wide, 0..) |*p, j| p.* = rn(dot(layer.project[j * width ..][0..width], up));
            miss[9].add(wide, back);
            rms(back, layer.norm, wide);
            miss[10].add(wide, post);
            for (wide, res, post) |*p, x, y| p.* = rn(x + y);
            miss[11].add(wide, added);
            for (wide, added) |*p, v| p.* = rn(v * layer.scalar);
            miss[12].add(wide, out);
            // apply() as a whole on the same residual and input
            const row = try a.dupe(f32, res);
            apply(geo, layer, final, row, buf);
            whole_block.add(row, out);
        }
    }
    for (mlx_steps, miss) |name, m| std.debug.print("{s}: {d} of {d} differ (worst {e:.2})\n", .{ name, m.differ, m.total, m.worst });
    std.debug.print("inputs(): {d} of {d} differ; apply(): {d} of {d} differ\n", .{ whole_inputs.differ, whole_inputs.total, whole_block.differ, whole_block.total });
    for (miss) |m| try std.testing.expectEqual(@as(usize, 0), m.differ);
    try std.testing.expectEqual(@as(usize, 0), whole_inputs.differ);
    try std.testing.expectEqual(@as(usize, 0), whole_block.differ);
}

test "the float64 reference holds the real shapes, and the host's distance from it is a stated bound, not equality" {
    // Ref is the same bf16 steps with an exact float64 sum; the host's distance from it is the measured bound.
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const hidden = 2560;
    const width = 256;
    const layers = 3;
    const tokens = 4;
    var geo = cfg.e4b;
    geo.layers = layers;
    const h = try seeded(a, 1, tokens * hidden, 3.0, 0);
    const model: Model = .{ .project = try seeded(a, 2, layers * width * hidden, 0.03, 0), .norm = try seeded(a, 3, width, 0.5, 1.0) };
    const tab = try seeded(a, 4, tokens * layers * width, 0.08, 0);
    const n = layers * width;
    const host = try a.alloc(f32, tokens * n);
    const want = try a.alloc(f64, tokens * n);
    for (0..tokens) |t| {
        const h64 = try a.alloc(f64, hidden);
        for (h64, h[t * hidden ..][0..hidden]) |*d, s| d.* = s;
        inputs(geo, model, h[t * hidden ..][0..hidden], tab[t * n ..][0..n], host[t * n ..][0..n]);
        (Ref{}).inputs(geo, model, h64, tab[t * n ..][0..n], want[t * n ..][0..n]);
    }
    const s = spread(host, want, n);
    std.debug.print("real-shape inputs against float64: worst column {e:.2}, rms {e:.2}, {d} of {d} differ\n", .{ s.worst, s.rms, s.differing, host.len });
    // bound: a flipped rounding moves a value one bf16 step (2^-7 of it) and flips are rare
    try std.testing.expect(s.worst <= 0x1p-7);
    try std.testing.expect(s.rms <= 0x1p-9);
    try std.testing.expect(s.differing * 50 <= host.len);

    const gate_w = try seeded(a, 5, layers * width * hidden, 0.05, 0);
    const back_w = try seeded(a, 6, layers * hidden * width, 0.1, 0);
    const post_w = try seeded(a, 7, layers * hidden, 0.5, 1.0);
    const layer_scalar = try seeded(a, 8, layers, 0.4, 0.5);
    const residual = try seeded(a, 9, tokens * layers * hidden, 12.0, 0);
    const out_host = try a.alloc(f32, tokens * layers * hidden);
    const out_want = try a.alloc(f64, tokens * layers * hidden);
    const scratch = try a.alloc(f32, width + hidden);
    for (0..tokens * layers) |r| {
        const l = r % layers;
        const layer: Layer = .{ .gate = gate_w[l * width * hidden ..][0 .. width * hidden], .project = back_w[l * hidden * width ..][0 .. hidden * width], .norm = post_w[l * hidden ..][0..hidden], .scalar = layer_scalar[l] };
        const input = host[(r / layers) * n + l * width ..][0..width]; // the host's own inputs, in both
        var in64: [width]f64 = undefined;
        for (&in64, input) |*d, v| d.* = v;
        const row = out_host[r * hidden ..][0..hidden];
        @memcpy(row, residual[r * hidden ..][0..hidden]);
        apply(geo, layer, input, row, scratch);
        const ref = out_want[r * hidden ..][0..hidden];
        for (ref, residual[r * hidden ..][0..hidden]) |*d, v| d.* = v;
        Ref.apply(geo, layer, &in64, ref);
    }
    const b = spread(out_host, out_want, hidden);
    std.debug.print("real-shape block against float64: worst column {e:.2}, rms {e:.2}, {d} of {d} differ\n", .{ b.worst, b.rms, b.differing, out_host.len });
    try std.testing.expect(b.worst <= 0x1p-6);
    try std.testing.expect(b.rms <= 0x1p-8);
    try std.testing.expect(b.differing * 20 <= out_host.len);
}
