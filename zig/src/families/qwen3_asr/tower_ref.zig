//! The audio tower in float64, stage by stage as the package's Qwen3ASRAudioEncoder runs it: the check's reference.
const std = @import("std");
const cfg = @import("config.zig");
const chunks = @import("chunks.zig");
const mel = @import("mel.zig");
const ops = @import("ref_ops.zig");
const names = @import("names.zig");

/// The tower's bf16 tensors by name (a mapped checkpoint), widened on use.
pub const Source = struct {
    ctx: *anyopaque,
    get: *const fn (ctx: *anyopaque, name: []const u8) ?[]const u8,

    fn take(s: Source, gpa: std.mem.Allocator, name: []const u8) ![]f64 {
        const bytes = s.get(s.ctx, name) orelse return error.MissingTensor;
        return ops.values(gpa, bytes);
    }
};

/// Where each stage goes: `values` are the stage's elements from `offset` on, in the package's layout.
pub const Sink = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, stage: []const u8, offset: usize, values: []const f64) anyerror!void,
};

/// A conv over one chunk: x [fi][ti][ci] channels last, to y [fo][to][co] with its bias, rounded (GELU follows).
fn conv(gpa: std.mem.Allocator, p: ops.Points, w: []const f64, b: []const f64, x: []const f64, fi: usize, ti: usize, ci: usize, co: usize) ![]f64 {
    const fo = (fi - 1) / 2 + 1;
    const to = (ti - 1) / 2 + 1;
    const k = ci * 9;
    const cols = try gpa.alloc(f64, fo * to * k);
    defer gpa.free(cols);
    for (0..fo) |f| for (0..to) |t| {
        const row = cols[(f * to + t) * k ..][0..k];
        for (0..ci) |c| for (0..3) |kh| for (0..3) |kw| {
            const sf = @as(isize, @intCast(2 * f + kh)) - 1;
            const st = @as(isize, @intCast(2 * t + kw)) - 1;
            const inside = sf >= 0 and st >= 0 and sf < fi and st < ti;
            row[c * 9 + kh * 3 + kw] = if (inside) x[(@as(usize, @intCast(sf)) * ti + @as(usize, @intCast(st))) * ci + c] else 0;
        };
    };
    const y = try gpa.alloc(f64, fo * to * co);
    ops.linear(y, cols, w, b, fo * to, co, k);
    ops.roundAll(p, y);
    return y;
}

fn geluAll(p: ops.Points, y: []f64) void {
    for (y) |*v| v.* = ops.round(p, ops.gelu(v.*));
}

/// A channels-last chunk [f][t][c] in the export's [c][f][t] order.
fn channelsFirst(gpa: std.mem.Allocator, x: []const f64, f: usize, t: usize, c: usize) ![]f64 {
    const out = try gpa.alloc(f64, x.len);
    for (0..f) |i| for (0..t) |j| for (0..c) |k| {
        out[(k * f + i) * t + j] = x[(i * t + j) * c + k];
    };
    return out;
}

/// A conv's output to the sink (the package's hook sees it before the GELU), then the GELU in place.
fn tap(gpa: std.mem.Allocator, sink: Sink, name: []const u8, n: usize, y: []f64, s: [2]usize, co: usize, p: ops.Points) !void {
    const cfirst = try channelsFirst(gpa, y, s[0], s[1], co);
    defer gpa.free(cfirst);
    try sink.put(sink.ctx, name, n * cfirst.len, cfirst);
    geluAll(p, y);
}

/// The sinusoid table's first `rows` rows [rows][d]: sin on the first half, cos on the second, as the package has it.
pub fn positions(gpa: std.mem.Allocator, p: ops.Points, rows: usize, d: usize) ![]f64 {
    const out = try gpa.alloc(f64, rows * d);
    const half = d / 2;
    const inc = @log(10000.0) / @as(f64, @floatFromInt(half - 1));
    for (0..rows) |r| for (0..half) |i| {
        const scaled: f64 = if (p == .torch) blk: {
            const inv: f32 = @exp(@as(f32, @floatCast(-inc * @as(f64, @floatFromInt(i)))));
            break :blk @as(f32, @floatFromInt(r)) * inv;
        } else @as(f64, @floatFromInt(r)) * @exp(-inc * @as(f64, @floatFromInt(i)));
        const s: f64 = if (p == .torch) @sin(@as(f32, @floatCast(scaled))) else @sin(scaled);
        const c: f64 = if (p == .torch) @cos(@as(f32, @floatCast(scaled))) else @cos(scaled);
        out[r * d + i] = ops.round(p, s);
        out[r * d + half + i] = ops.round(p, c);
    };
    return out;
}

/// The convolutions and conv_out over every chunk: the tower's input rows [plan.rows][d], positions added.
fn front(gpa: std.mem.Allocator, a: cfg.Audio, p: ops.Points, src: Source, feats: mel.Features, plan: chunks.Plan, sink: Sink) ![]f64 {
    const t = names.tower;
    const w1 = try src.take(gpa, t ++ "conv2d1.weight");
    defer gpa.free(w1);
    const b1 = try src.take(gpa, t ++ "conv2d1.bias");
    defer gpa.free(b1);
    const w2 = try src.take(gpa, t ++ "conv2d2.weight");
    defer gpa.free(w2);
    const b2 = try src.take(gpa, t ++ "conv2d2.bias");
    defer gpa.free(b2);
    const w3 = try src.take(gpa, t ++ "conv2d3.weight");
    defer gpa.free(w3);
    const b3 = try src.take(gpa, t ++ "conv2d3.bias");
    defer gpa.free(b3);
    const wo = try src.take(gpa, t ++ "conv_out.weight");
    defer gpa.free(wo);
    const cf = a.chunkFrames();
    const bins = a.convBins();
    const crow = chunks.rows(cf);
    const pos = try positions(gpa, p, crow, a.d);
    defer gpa.free(pos);
    const out = try gpa.alloc(f64, plan.rows * a.d);
    errdefer gpa.free(out);
    const input = try gpa.alloc(f64, a.mels * cf);
    defer gpa.free(input);
    const flat = try gpa.alloc(f64, crow * a.conv * bins);
    defer gpa.free(flat);
    const emb = try gpa.alloc(f64, crow * a.d);
    defer gpa.free(emb);
    var row: usize = 0;
    for (0..plan.chunks) |n| {
        for (0..a.mels) |m| for (0..cf) |j| {
            const fr = n * cf + j;
            input[m * cf + j] = if (fr < feats.frames) ops.round(p, feats.at(m, fr)) else 0;
        };
        const s1 = [3]usize{ a.mels, cf, 1 };
        const y1 = try conv(gpa, p, w1, b1, input, s1[0], s1[1], s1[2], a.conv);
        defer gpa.free(y1);
        const s2 = [2]usize{ (s1[0] - 1) / 2 + 1, (s1[1] - 1) / 2 + 1 };
        try tap(gpa, sink, "conv1", n, y1, s2, a.conv, p);
        const y2 = try conv(gpa, p, w2, b2, y1, s2[0], s2[1], a.conv, a.conv);
        defer gpa.free(y2);
        const s3 = [2]usize{ (s2[0] - 1) / 2 + 1, (s2[1] - 1) / 2 + 1 };
        try tap(gpa, sink, "conv2", n, y2, s3, a.conv, p);
        const y3 = try conv(gpa, p, w3, b3, y2, s3[0], s3[1], a.conv, a.conv);
        defer gpa.free(y3);
        const s4 = [2]usize{ (s3[0] - 1) / 2 + 1, (s3[1] - 1) / 2 + 1 };
        try tap(gpa, sink, "conv3", n, y3, s4, a.conv, p);
        if (s4[0] != bins or s4[1] != crow) return error.TowerShape;
        for (0..crow) |tt| for (0..a.conv) |c| for (0..bins) |f| {
            flat[tt * a.conv * bins + c * bins + f] = y3[(f * crow + tt) * a.conv + c];
        };
        ops.linear(emb, flat, wo, null, crow, a.d, a.conv * bins);
        ops.roundAll(p, emb);
        try sink.put(sink.ctx, "conv_out", n * emb.len, emb);
        for (emb, 0..) |*v, i| v.* = ops.round(p, v.* + pos[i]);
        const keep = plan.rowsOf(n);
        @memcpy(out[row * a.d ..][0 .. keep * a.d], emb[0 .. keep * a.d]);
        row += keep;
    }
    return out;
}

/// One encoder layer over the rows x [rows][d], in place, attention within the plan's windows.
fn layer(gpa: std.mem.Allocator, a: cfg.Audio, p: ops.Points, src: Source, i: usize, x: []f64, plan: chunks.Plan) !void {
    var buf: [128]u8 = undefined;
    const W = struct {
        fn get(s: Source, g: std.mem.Allocator, b: []u8, li: usize, n: []const u8) ![]f64 {
            return s.take(g, try std.fmt.bufPrint(b, names.tower ++ "layers.{d}.{s}", .{ li, n }));
        }
    };
    const rows = plan.rows;
    const d = a.d;
    const h = try gpa.alloc(f64, rows * d);
    defer gpa.free(h);
    const qkv = try gpa.alloc(f64, rows * 3 * d);
    defer gpa.free(qkv);
    const att = try gpa.alloc(f64, rows * d);
    defer gpa.free(att);
    const wide = try gpa.alloc(f64, rows * a.ffn);
    defer gpa.free(wide);
    {
        const ln_w = try W.get(src, gpa, &buf, i, "self_attn_layer_norm.weight");
        defer gpa.free(ln_w);
        const ln_b = try W.get(src, gpa, &buf, i, "self_attn_layer_norm.bias");
        defer gpa.free(ln_b);
        @memcpy(h, x);
        ops.layerNorm(h, ln_w, ln_b, d);
        ops.roundAll(p, h);
    }
    const proj = try gpa.alloc(f64, rows * d);
    defer gpa.free(proj);
    for ([_][]const u8{ "q_proj", "k_proj", "v_proj" }, 0..) |n, part| {
        var nb: [48]u8 = undefined;
        const w = try W.get(src, gpa, &buf, i, try std.fmt.bufPrint(&nb, "self_attn.{s}.weight", .{n}));
        defer gpa.free(w);
        const b = try W.get(src, gpa, &buf, i, try std.fmt.bufPrint(&nb, "self_attn.{s}.bias", .{n}));
        defer gpa.free(b);
        ops.linear(proj, h, w, b, rows, d, d);
        for (0..rows) |r| for (0..d) |c| {
            qkv[r * 3 * d + part * d + c] = ops.round(p, proj[r * d + c]);
        };
    }
    for (0..plan.windows()) |wi| {
        const win = plan.window(wi);
        try ops.attention(att, qkv, win.start, win.len, a.heads, a.headDim());
    }
    ops.roundAll(p, att);
    try addLinear(gpa, p, src, &buf, i, "self_attn.out_proj", x, att, rows, d, d);
    {
        const ln_w = try W.get(src, gpa, &buf, i, "final_layer_norm.weight");
        defer gpa.free(ln_w);
        const ln_b = try W.get(src, gpa, &buf, i, "final_layer_norm.bias");
        defer gpa.free(ln_b);
        @memcpy(h, x);
        ops.layerNorm(h, ln_w, ln_b, d);
        ops.roundAll(p, h);
    }
    {
        const w = try W.get(src, gpa, &buf, i, "fc1.weight");
        defer gpa.free(w);
        const b = try W.get(src, gpa, &buf, i, "fc1.bias");
        defer gpa.free(b);
        ops.linear(wide, h, w, b, rows, a.ffn, d);
        for (wide) |*v| v.* = ops.round(p, ops.gelu(ops.round(p, v.*)));
    }
    try addLinear(gpa, p, src, &buf, i, "fc2", x, wide, rows, d, a.ffn);
}

/// x = round(x + round(in . w^T + b)) for layer i's Linear `name`.
fn addLinear(gpa: std.mem.Allocator, p: ops.Points, src: Source, buf: []u8, i: usize, name: []const u8, x: []f64, in: []const f64, rows: usize, n: usize, k: usize) !void {
    const w = try src.take(gpa, try std.fmt.bufPrint(buf, names.tower ++ "layers.{d}.{s}.weight", .{ i, name }));
    defer gpa.free(w);
    const b = try src.take(gpa, try std.fmt.bufPrint(buf, names.tower ++ "layers.{d}.{s}.bias", .{ i, name }));
    defer gpa.free(b);
    const y = try gpa.alloc(f64, rows * n);
    defer gpa.free(y);
    ops.linear(y, in, w, b, rows, n, k);
    for (x, y) |*v, o| v.* = ops.round(p, v.* + ops.round(p, o));
}

/// The tower over `feats`: every stage to `sink`, the audio rows [plan.rows][out] returned.
pub fn run(gpa: std.mem.Allocator, a: cfg.Audio, p: ops.Points, src: Source, feats: mel.Features, sink: Sink) ![]f64 {
    const plan = try chunks.Plan.init(a, feats.frames);
    const x = try front(gpa, a, p, src, feats, plan, sink);
    defer gpa.free(x);
    try sink.put(sink.ctx, "embed", 0, x);
    var name: [16]u8 = undefined;
    for (0..a.layers) |i| {
        try layer(gpa, a, p, src, i, x, plan);
        try sink.put(sink.ctx, try std.fmt.bufPrint(&name, "layer{d}", .{i}), 0, x);
    }
    const t = names.tower;
    const lw = try src.take(gpa, t ++ "ln_post.weight");
    defer gpa.free(lw);
    const lb = try src.take(gpa, t ++ "ln_post.bias");
    defer gpa.free(lb);
    ops.layerNorm(x, lw, lb, a.d);
    ops.roundAll(p, x);
    try sink.put(sink.ctx, "ln_post", 0, x);
    const h = try gpa.alloc(f64, plan.rows * a.d);
    defer gpa.free(h);
    const w1 = try src.take(gpa, t ++ "proj1.weight");
    defer gpa.free(w1);
    const b1 = try src.take(gpa, t ++ "proj1.bias");
    defer gpa.free(b1);
    ops.linear(h, x, w1, b1, plan.rows, a.d, a.d);
    ops.roundAll(p, h);
    try sink.put(sink.ctx, "proj1", 0, h);
    for (h) |*v| v.* = ops.round(p, ops.gelu(v.*));
    const out = try gpa.alloc(f64, plan.rows * a.out);
    errdefer gpa.free(out);
    const w2 = try src.take(gpa, t ++ "proj2.weight");
    defer gpa.free(w2);
    const b2 = try src.take(gpa, t ++ "proj2.bias");
    defer gpa.free(b2);
    ops.linear(out, h, w2, b2, plan.rows, a.out, a.d);
    ops.roundAll(p, out);
    try sink.put(sink.ctx, "audio_rows", 0, out);
    return out;
}
