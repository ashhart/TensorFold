//! Mini's image path: processor, tables, GEMMs, tower vs an f64 reference on a synthetic single-shard checkpoint.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const picture = @import("../../core/picture/picture.zig");
const frags = @import("../../core/frags.zig");
const config = @import("config.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const fxm = @import("metal_draft_fixture.zig");
const gm = @import("metal_vision_gemm.zig");
const image = @import("metal_vision_image.zig");
const tables = @import("metal_vision_tables.zig");
const vision = @import("metal_vision.zig");

fn bf(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn unbf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

fn rb(x: f64) f64 {
    return unbf(bf(@floatCast(x)));
}

fn shape(a: std.mem.Allocator, dims: []const usize) ![]const usize {
    return a.dupe(usize, dims);
}

const t = std.testing;

pub fn rgbImage(a: std.mem.Allocator, w: usize, h: usize) !picture.Rgb {
    const px = try a.alloc(u8, w * h * 3);
    for (0..h) |y| for (0..w) |x| {
        px[(y * w + x) * 3] = @intCast(x * 255 / @max(w - 1, 1));
        px[(y * w + x) * 3 + 1] = @intCast(y * 255 / @max(h - 1, 1));
        px[(y * w + x) * 3 + 2] = @intCast((x + y) % 256);
    };
    return .{ .width = w, .height = h, .pixels = px };
}

test "the processor pads to 40-pixel cells with -1, normalizes to [-1, 1], and orders merge blocks then patches" {
    try t.expectEqual(@as(usize, 2), image.tokensOf(41, 40));
    try t.expectEqual(@as(usize, 1), image.tokensOf(40, 40));
    try t.expectEqual(@as(usize, 6), image.tokensOf(81, 80));
    var rgb = try rgbImage(t.allocator, 50, 30);
    defer rgb.deinit(t.allocator);
    const p = try image.fromRgb(t.allocator, rgb, 7, 16);
    defer t.allocator.free(p.pixels);
    try t.expectEqual(@as(u32, 1), p.gh);
    try t.expectEqual(@as(u32, 2), p.gw);
    try t.expectEqual(@as(usize, 2 * image.token_width), p.pixels.len);
    // token 0, patch 0 (rows 0..19, cols 0..19), pixel (0, 0) channel 0: byte 0 -> -1
    try t.expectEqual(@as(f32, -1.0), p.pixels[0]);
    // token 0, patch 1 = columns 20.., pixel (0, 0): x = 20 of 50 -> byte 20 * 255 / 49
    const byte: f32 = @floatFromInt(20 * 255 / 49);
    try t.expectApproxEqAbs(byte / 255.0 * 2 - 1, p.pixels[image.patch_width], 1e-6);
    // token 0, patch 2 starts at row 20 of a 30-high image: its rows 10.. are padding, (-1 / 255 - 0.5) / 0.5
    const pad: f32 = (-1.0 / 255.0 - 0.5) / 0.5;
    try t.expectApproxEqAbs(pad, p.pixels[2 * image.patch_width + (10 * image.patch + 0) * 3], 1e-6);
    // the same image under a one-token budget is resized to fit, aspect kept
    const small = try image.fromRgb(t.allocator, rgb, 7, 1);
    defer t.allocator.free(small.pixels);
    try t.expectEqual(@as(u32, 1), small.tokens);
    try t.expect(small.hash != p.hash);
    try t.expectError(error.NoImageBudget, image.fromRgb(t.allocator, rgb, 7, 0));
    try t.expect(image.problem(error.NoImageBudget) != null);
}

test "position taps are the half-pixel bilinear grid and the rotary table turns columns on the first frequencies" {
    const same = tables.axis(5, 32, 32);
    try t.expectEqual([2]u32{ 5, 6 }, same.taps);
    try t.expectEqual([2]f32{ 1, 0 }, same.weights);
    const edge = tables.axis(0, 64, 32); // between grid cells -0.25 and 0.25: clamped to the first
    try t.expectEqual([2]u32{ 0, 0 }, edge.taps);
    try t.expectApproxEqAbs(@as(f32, 1.0), edge.weights[0] + edge.weights[1], 1e-6);
    try t.expectEqual([2]usize{ 1, 0 }, tables.place(2, 1, 2)); // third patch of token 0: row 1, column 0
    try t.expectEqual([2]usize{ 0, 2 }, tables.place(4, 2, 2)); // first patch of token 1 of a 2-wide grid
    const hd = 8;
    const n = comptime tables.patches(1, 2, 2);
    var taps: [n * 4]u32 = undefined;
    var wts: [n * 4]f32 = undefined;
    var cs: [(n + 1) * hd]f32 = undefined;
    var sn: [(n + 1) * hd]f32 = undefined;
    tables.fill(1, 2, 2, 32, hd, &taps, &wts, &cs, &sn);
    try t.expectEqual(@as(f32, 1.0), cs[0]);
    try t.expectEqual(@as(f32, 0.0), sn[0]);
    // patch 1 (row 0, column 1): positions (1, 2); pair 0 turns with a column frequency, pair 2 with a row one
    try t.expectApproxEqAbs(@cos(@as(f32, 2.0)), cs[2 * hd], 1e-6);
    try t.expectApproxEqAbs(@cos(@as(f32, 1.0)), cs[2 * hd + 4], 1e-6);
    try t.expectApproxEqAbs(@sin(@as(f32, 2.0)), sn[2 * hd + 1], 1e-6);
    var weight_sum: f32 = 0;
    for (wts[4..8]) |w| weight_sum += w;
    try t.expectApproxEqAbs(@as(f32, 1.0), weight_sum, 1e-5);
}

const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,

    fn init() !Gpu {
        const device = mtl.Device.init() catch return error.SkipZigTest;
        errdefer device.deinit();
        const queue = device.queue() catch return error.SkipZigTest;
        errdefer queue.deinit();
        const src = try frags.source(device, t.allocator, ks.inkling2_vision);
        defer t.allocator.free(src);
        return .{ .device = device, .queue = queue, .lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx()) };
    }
    fn deinit(g: Gpu) void {
        g.lib.deinit();
        g.queue.deinit();
        g.device.deinit();
    }
};

fn gemmCase(g: Gpu, gemm: *const gm.Gemm, kind: gm.Kind, m: u32, n: u32, k: u32, nn: bool) !void {
    const wide = kind == .f32_nt or kind == .f32_nn;
    const out32 = wide or kind == .bf16f_nt;
    const a = try g.device.buffer(@as(usize, m) * k * (if (wide) @as(usize, 4) else 2), mtl.ResourceOptions.shared);
    defer a.deinit();
    const b = try g.device.buffer(@as(usize, n) * k * (if (wide) @as(usize, 4) else 2), mtl.ResourceOptions.shared);
    defer b.deinit();
    const d = try g.device.buffer(@as(usize, m) * n * (if (out32) @as(usize, 4) else 2), mtl.ResourceOptions.shared);
    defer d.deinit();
    var prng = std.Random.DefaultPrng.init(m * 7 + n);
    const rng = prng.random();
    const av = try t.allocator.alloc(f64, @as(usize, m) * k);
    defer t.allocator.free(av);
    const bv = try t.allocator.alloc(f64, @as(usize, n) * k);
    defer t.allocator.free(bv);
    for (av, 0..) |*v, i| {
        const x = rng.float(f32) * 2 - 1;
        if (wide) a.slice(f32, av.len)[i] = x else a.slice(u16, av.len)[i] = bf(x);
        v.* = if (wide) x else unbf(bf(x));
    }
    for (bv, 0..) |*v, i| {
        const x = rng.float(f32) * 2 - 1;
        if (wide) b.slice(f32, bv.len)[i] = x else b.slice(u16, bv.len)[i] = bf(x);
        v.* = if (wide) x else unbf(bf(x));
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    gemm.run(e, kind, .{ .buf = a }, .{ .buf = b }, .{ .buf = d }, .{ .m = m, .n = n, .k = k, .lda = k, .ldb = if (nn) n else k, .ldd = n });
    e.end();
    cb.commit();
    cb.wait();
    try t.expect(cb.failure() == null);
    for (0..m) |r| for (0..n) |c| {
        var s: f64 = 0;
        for (0..k) |i| s += av[r * k + i] * (if (nn) bv[i * n + c] else bv[c * k + i]);
        const got: f64 = if (out32) d.slice(f32, @as(usize, m) * n)[r * n + c] else unbf(d.slice(u16, @as(usize, m) * n)[r * n + c]);
        const want = if (out32) s else rb(s);
        try t.expectApproxEqAbs(want, got, if (out32) 5e-3 + 2e-3 * @abs(want) else 1e-3 + 1e-2 * @abs(want));
    };
}

test "the tensor-unit GEMMs: every kind, whole and ragged tiles, K past and short of 256" {
    const g = try Gpu.init();
    defer g.deinit();
    var gemm = try gm.Gemm.init(g.device, g.lib, &.{ .bf16_nt, .bf16_nn, .bf16f_nt, .f32_nt, .f32_nn });
    defer gemm.deinit();
    for ([_][3]u32{ .{ 64, 128, 256 }, .{ 70, 130, 300 }, .{ 9, 32, 32 }, .{ 3, 200, 128 } }) |s| {
        try gemmCase(g, &gemm, .bf16_nt, s[0], s[1], s[2], false);
        try gemmCase(g, &gemm, .bf16f_nt, s[0], s[1], s[2], false);
        try gemmCase(g, &gemm, .bf16_nn, s[0], s[1], s[2], true);
        try gemmCase(g, &gemm, .f32_nt, s[0], s[1], s[2], false);
        try gemmCase(g, &gemm, .f32_nn, s[0], s[1], s[2], true);
    }
}

// ---- the tower ----

const Named = struct { name: []const u8, dims: []const usize, scale: f32, bias: f32 };

fn writeShard(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, file: []const u8, specs: []const Named, seed: u64) !void {
    var header: std.ArrayList(u8) = .empty;
    var data: std.ArrayList(u8) = .empty;
    try header.append(a, '{');
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (specs, 0..) |s, i| {
        var n: usize = 1;
        for (s.dims) |d| n *= d;
        const v = try a.alloc(u16, n);
        for (v) |*h| h.* = bf(s.bias + s.scale * (rng.float(f32) * 2 - 1));
        if (i != 0) try header.append(a, ',');
        try header.print(a, "\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[", .{s.name});
        for (s.dims, 0..) |d, j| try header.print(a, "{s}{d}", .{ if (j != 0) "," else "", d });
        try header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ data.items.len, data.items.len + n * 2 });
        try data.appendSlice(a, std.mem.sliceAsBytes(v));
    }
    try header.append(a, '}');
    while (header.items.len % 8 != 0) try header.append(a, ' ');
    var out: std.ArrayList(u8) = .empty;
    try out.resize(a, 8);
    std.mem.writeInt(u64, out.items[0..8], header.items.len, .little);
    try out.appendSlice(a, header.items);
    try out.appendSlice(a, data.items);
    try dir.writeFile(io, .{ .sub_path = file, .data = out.items });
}

extern "c" fn erf(x: f64) f64;
fn gelu(x: f64) f64 {
    return 0.5 * x * (1 + erf(x / @sqrt(2.0)));
}

fn layerNorm(x: []const f64, w: []const u16, b: []const u16, eps: f64, out: []f64) void {
    const n: f64 = @floatFromInt(x.len);
    var mean: f64 = 0;
    for (x) |v| mean += v;
    mean /= n;
    var vr: f64 = 0;
    for (x) |v| vr += (v - mean) * (v - mean);
    const inv = 1.0 / @sqrt(vr / n + eps);
    for (x, out, w, b) |v, *o, g, bi| o.* = rb((v - mean) * inv * unbf(g) + unbf(bi));
}

/// out [rows, n] = sum x [rows, k] w^T (+ bias), unrounded: the caller rounds where the model does.
fn lin(x: []const f64, rows: usize, k: usize, w: []const u16, n: usize, bias: ?[]const u16, out: []f64) void {
    for (0..rows) |r| for (0..n) |o| {
        var s: f64 = 0;
        for (0..k) |i| s += x[r * k + i] * unbf(w[o * k + i]);
        out[r * n + o] = s + if (bias) |b| unbf(b[o]) else 0;
    };
}

fn roundAll(v: []f64) void {
    for (v) |*e| e.* = rb(e.*);
}

fn coord(p: usize, pixels: usize, table: usize) f64 {
    const x: f64 = @floatFromInt(p);
    const scale = @as(f64, @floatFromInt(table)) / @as(f64, @floatFromInt(pixels));
    return std.math.clamp((x + 0.5) * scale - 0.5, 0, @as(f64, @floatFromInt(table - 1)));
}

fn oracleTables(gh: usize, gw: usize, merge: usize, side: usize, hd: usize, taps: []u32, wts: []f64, cs: []f64, sn: []f64) void {
    @memset(cs[0..hd], 1);
    @memset(sn[0..hd], 0);
    var p: usize = 0;
    for (0..gh) |ty| for (0..gw) |tx| for (0..merge) |dy| for (0..merge) |dx| {
        const py = ty * merge + dy;
        const px = tx * merge + dx;
        const y = coord(py, gh * merge, side);
        const x = coord(px, gw * merge, side);
        const top: usize = @intFromFloat(@floor(y));
        const left: usize = @intFromFloat(@floor(x));
        const fy = y - @floor(y);
        const fx = x - @floor(x);
        for (0..2) |j| for (0..2) |i| {
            const k = p * 4 + j * 2 + i;
            taps[k] = @intCast(@min(top + j, side - 1) * side + @min(left + i, side - 1));
            wts[k] = (if (j == 0) 1 - fy else fy) * (if (i == 0) 1 - fx else fx);
        };
        const frequencies = hd / 4;
        for (0..hd) |d| {
            const pair = d / 2;
            const axial_pos: f64 = @floatFromInt(if (pair < frequencies) px + 1 else py + 1);
            const exponent = @as(f64, @floatFromInt(pair % frequencies)) / @as(f64, @floatFromInt(frequencies));
            const angle = axial_pos / std.math.pow(f64, 10000, exponent);
            cs[(p + 1) * hd + d] = @cos(angle);
            sn[(p + 1) * hd + d] = @sin(angle);
        }
        p += 1;
    };
}

/// Tower's f64 reference over the checkpoint's `model.visual.*` tensors; each op rounds to bf16 where the model's does.
pub fn reference(a: std.mem.Allocator, w: *const ck.Checkpoint, c: config.Vision, pixels: []const f32, gh: u32, gw: u32) ![]f64 {
    return evaluate(a, w, c, pixels, gh, gw, &.{}, &.{});
}

pub const Error = struct { rms: f64, max_abs: f64 };

fn replay(x: []f64, captured: []const u16, layer: usize, errs: []Error) void {
    if (errs.len == 0) return;
    const expected = captured[layer * x.len ..][0..x.len];
    errs[layer] = compare(expected, x);
    for (x, expected) |*v, b| v.* = unbf(b);
}

pub fn compare(got: []const u16, want: []const f64) Error {
    var abs: f64 = 0;
    var err: f64 = 0;
    var mag: f64 = 0;
    for (got, want) |b, v| {
        const d = @abs(unbf(b) - v);
        if (!std.math.isFinite(d)) return .{ .rms = std.math.inf(f64), .max_abs = std.math.inf(f64) };
        abs = @max(abs, d);
        err += d * d;
        mag += v * v;
    }
    return .{ .rms = @sqrt(err / @max(mag, 0x1p-100)), .max_abs = abs };
}

/// After each stage, record its local error and substitute the GPU's actual BF16 rows for the next stage.
pub fn evaluate(a: std.mem.Allocator, w: *const ck.Checkpoint, c: config.Vision, pixels: []const f32, gh: u32, gw: u32, captured: []const u16, errs: []Error) ![]f64 {
    const h: usize = c.hidden;
    const hd = h / c.heads;
    const P = @as(usize, gh) * gw * c.merge * c.merge;
    const S = P + 1;
    const pre = "model.visual.backbone.";
    const get = struct {
        fn f(m: *const ck.Checkpoint, comptime fmt: []const u8, args: anytype) []const u16 {
            var buf: [128]u8 = undefined;
            return (m.get(std.fmt.bufPrint(&buf, fmt, args) catch unreachable) catch unreachable).host(u16);
        }
    }.f;
    const taps = try a.alloc(u32, P * 4);
    const wts = try a.alloc(f64, P * 4);
    const cs = try a.alloc(f64, S * hd);
    const sn = try a.alloc(f64, S * hd);
    oracleTables(gh, gw, c.merge, c.grid, hd, taps, wts, cs, sn);
    const pix = try a.alloc(f64, P * image.patch_width);
    for (pix, pixels[0..pix.len]) |*o, v| o.* = unbf(bf(v));
    const pe = try a.alloc(f64, P * h);
    lin(pix, P, image.patch_width, get(w, pre ++ "embeddings.patch_embedding.weight", .{}), h, null, pe);
    roundAll(pe);
    var x = try a.alloc(f64, S * h);
    const table = get(w, pre ++ "embeddings.position_embedding", .{});
    const cls = get(w, pre ++ "embeddings.class_embedding", .{});
    const cls_pos = get(w, pre ++ "embeddings.cls_position_embedding", .{});
    for (0..h) |i| x[i] = rb(unbf(cls[i]) + unbf(cls_pos[i]));
    for (0..P) |p| for (0..h) |i| {
        var s: f64 = 0;
        for (0..4) |k| s += unbf(table[taps[p * 4 + k] * h + i]) * wts[p * 4 + k];
        x[(p + 1) * h + i] = rb(pe[p * h + i] + rb(s));
    };
    const tmp = try a.alloc(f64, S * h);
    for (0..S) |r| layerNorm(x[r * h ..][0..h], get(w, pre ++ "ln_pre.weight", .{}), get(w, pre ++ "ln_pre.bias", .{}), c.eps, tmp[r * h ..][0..h]);
    @memcpy(x, tmp);
    replay(x, captured, 0, errs);
    const q = try a.alloc(f64, S * h);
    const k = try a.alloc(f64, S * h);
    const v = try a.alloc(f64, S * h);
    const att = try a.alloc(f64, S * h);
    const ff = h * c.mlp_ratio;
    const mid = try a.alloc(f64, S * ff);
    const branch = try a.alloc(f64, S * h);
    for (0..c.layers) |l| {
        for (0..S) |r| layerNorm(x[r * h ..][0..h], get(w, pre ++ "layers.{d}.ln_1.weight", .{l}), get(w, pre ++ "layers.{d}.ln_1.bias", .{l}), c.eps, tmp[r * h ..][0..h]);
        lin(tmp, S, h, get(w, pre ++ "layers.{d}.self_attn.q_proj.weight", .{l}), h, get(w, pre ++ "layers.{d}.self_attn.q_proj.bias", .{l}), q);
        lin(tmp, S, h, get(w, pre ++ "layers.{d}.self_attn.k_proj.weight", .{l}), h, get(w, pre ++ "layers.{d}.self_attn.k_proj.bias", .{l}), k);
        lin(tmp, S, h, get(w, pre ++ "layers.{d}.self_attn.v_proj.weight", .{l}), h, get(w, pre ++ "layers.{d}.self_attn.v_proj.bias", .{l}), v);
        roundAll(q);
        roundAll(k);
        roundAll(v);
        for ([_][]f64{ q, k }) |m| {
            const raw = try a.dupe(f64, m);
            for (0..S) |r| for (0..h) |i| {
                const d = i % hd;
                const partner = raw[r * h + (i ^ 1)];
                const cc: f64 = cs[r * hd + d];
                const ss: f64 = sn[r * hd + d];
                m[r * h + i] = rb(if (d % 2 == 0) raw[r * h + i] * cc - partner * ss else raw[r * h + i] * cc + partner * ss);
            };
        }
        const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(hd)));
        for (0..c.heads) |hh| for (0..S) |r| {
            const sc = try a.alloc(f64, S);
            var top: f64 = -std.math.inf(f64);
            for (0..S) |j| {
                var s: f64 = 0;
                for (0..hd) |d| s += q[r * h + hh * hd + d] * k[j * h + hh * hd + d];
                sc[j] = rb(rb(s) * scale);
                top = @max(top, sc[j]);
            }
            var z: f64 = 0;
            for (sc) |*s| {
                s.* = @exp(s.* - top);
                z += s.*;
            }
            for (sc) |*s| s.* = rb(s.* / z);
            for (0..hd) |d| {
                var o: f64 = 0;
                for (0..S) |j| o += sc[j] * v[j * h + hh * hd + d];
                att[r * h + hh * hd + d] = rb(o);
            }
        };
        lin(att, S, h, get(w, pre ++ "layers.{d}.self_attn.out_proj.weight", .{l}), h, get(w, pre ++ "layers.{d}.self_attn.out_proj.bias", .{l}), branch);
        const ls1 = get(w, pre ++ "layers.{d}.ls_1.gamma", .{l});
        for (0..S) |r| for (0..h) |i| {
            x[r * h + i] = rb(x[r * h + i] + rb(unbf(ls1[i]) * rb(branch[r * h + i])));
        };
        for (0..S) |r| layerNorm(x[r * h ..][0..h], get(w, pre ++ "layers.{d}.ln_2.weight", .{l}), get(w, pre ++ "layers.{d}.ln_2.bias", .{l}), c.eps, tmp[r * h ..][0..h]);
        lin(tmp, S, h, get(w, pre ++ "layers.{d}.mlp.c_fc.weight", .{l}), ff, get(w, pre ++ "layers.{d}.mlp.c_fc.bias", .{l}), mid);
        for (mid) |*e| e.* = rb(gelu(rb(e.*)));
        lin(mid, S, ff, get(w, pre ++ "layers.{d}.mlp.c_proj.weight", .{l}), h, get(w, pre ++ "layers.{d}.mlp.c_proj.bias", .{l}), branch);
        const ls2 = get(w, pre ++ "layers.{d}.ls_2.gamma", .{l});
        for (0..S) |r| for (0..h) |i| {
            x[r * h + i] = rb(x[r * h + i] + rb(unbf(ls2[i]) * rb(branch[r * h + i])));
        };
        replay(x, captured, l + 1, errs);
    }
    const tokens = P / (c.merge * c.merge);
    const p1 = try a.alloc(f64, tokens * c.proj_hidden);
    lin(x[h..], tokens, h * c.merge * c.merge, get(w, "model.visual.proj_in.weight", .{}), c.proj_hidden, null, p1);
    for (p1) |*e| e.* = rb(gelu(rb(e.*)));
    const out = try a.alloc(f64, tokens * c.text_hidden);
    lin(p1, tokens, c.proj_hidden, get(w, "model.visual.proj_out.weight", .{}), c.text_hidden, null, out);
    roundAll(out);
    return out;
}

/// Vision tower at test widths: 3 layers, width 128 over 2 heads of 64, MLP 4x, a 160-wide projector to text width.
fn towerConfig(a: std.mem.Allocator) !config.Config {
    var c = try fxm.tinyConfig(a);
    c.vision.hidden = 128;
    c.vision.layers = 3;
    c.vision.heads = 2;
    c.vision.proj_hidden = 160;
    c.vision.text_hidden = c.text.hidden;
    return c;
}

/// A tower's tensors in file order: one text tensor first (model.llm.* before model.visual.*), then the vision ones.
fn towerSpecs(a: std.mem.Allocator, c: config.Vision) ![]const Named {
    const h: usize = c.hidden;
    const ff = h * c.mlp_ratio;
    var specs: std.ArrayList(Named) = .empty;
    const pre = "model.visual.backbone.";
    try specs.append(a, .{ .name = "model.llm.norm.weight", .dims = try shape(a, &.{2304}), .scale = 1, .bias = 0 });
    try specs.append(a, .{ .name = pre ++ "embeddings.class_embedding", .dims = try shape(a, &.{h}), .scale = 0.5, .bias = 0 });
    try specs.append(a, .{ .name = pre ++ "embeddings.cls_position_embedding", .dims = try shape(a, &.{h}), .scale = 0.5, .bias = 0 });
    try specs.append(a, .{ .name = pre ++ "embeddings.patch_embedding.weight", .dims = try shape(a, &.{ h, image.patch_width }), .scale = 0.15, .bias = 0 });
    try specs.append(a, .{ .name = pre ++ "embeddings.position_embedding", .dims = try shape(a, &.{ c.grid * c.grid, h }), .scale = 0.5, .bias = 0 });
    try specs.append(a, .{ .name = pre ++ "ln_pre.weight", .dims = try shape(a, &.{h}), .scale = 0.1, .bias = 1 });
    try specs.append(a, .{ .name = pre ++ "ln_pre.bias", .dims = try shape(a, &.{h}), .scale = 0.1, .bias = 0 });
    try specs.append(a, .{ .name = "model.visual.proj_in.weight", .dims = try shape(a, &.{ c.proj_hidden, h * 4 }), .scale = 0.2, .bias = 0 });
    try specs.append(a, .{ .name = "model.visual.proj_out.weight", .dims = try shape(a, &.{ c.text_hidden, c.proj_hidden }), .scale = 0.3, .bias = 0 });
    const per_layer = [_]struct { []const u8, usize, usize, f32, f32 }{
        .{ "ln_1.weight", h, 0, 0.1, 1 },             .{ "ln_1.bias", h, 0, 0.1, 0 },                 .{ "ln_2.weight", h, 0, 0.1, 1 },             .{ "ln_2.bias", h, 0, 0.1, 0 },
        .{ "ls_1.gamma", h, 0, 0.3, 0.6 },            .{ "ls_2.gamma", h, 0, 0.3, 0.6 },              .{ "self_attn.q_proj.weight", h, h, 0.3, 0 }, .{ "self_attn.k_proj.weight", h, h, 0.3, 0 },
        .{ "self_attn.v_proj.weight", h, h, 0.3, 0 }, .{ "self_attn.out_proj.weight", h, h, 0.3, 0 }, .{ "self_attn.q_proj.bias", h, 0, 0.1, 0 },   .{ "self_attn.k_proj.bias", h, 0, 0.1, 0 },
        .{ "self_attn.v_proj.bias", h, 0, 0.1, 0 },   .{ "self_attn.out_proj.bias", h, 0, 0.1, 0 },   .{ "mlp.c_fc.weight", ff, h, 0.3, 0 },        .{ "mlp.c_fc.bias", ff, 0, 0.1, 0 },
        .{ "mlp.c_proj.weight", h, ff, 0.2, 0 },      .{ "mlp.c_proj.bias", h, 0, 0.1, 0 },
    };
    for (0..c.layers) |l| for (per_layer) |p| {
        const dims: []const usize = if (p[2] == 0) try a.dupe(usize, &.{p[1]}) else try a.dupe(usize, &.{ p[1], p[2] });
        try specs.append(a, .{ .name = try std.fmt.allocPrint(a, pre ++ "layers.{d}.{s}", .{ l, p[0] }), .dims = dims, .scale = p[3], .bias = p[4] });
    };
    return specs.items;
}

/// The reference's error against the tower's output rows, relative to the reference's magnitude.
pub fn relative(got: []const u16, want: []const f64) !f64 {
    var err: f64 = 0;
    var mag: f64 = 0;
    for (got, want) |gv, wv| {
        err += (unbf(gv) - wv) * (unbf(gv) - wv);
        mag += wv * wv;
    }
    try t.expect(mag > 0);
    return @sqrt(err / mag);
}

test "Mini's vision config is the tower this engine runs: 23 layers, 16 heads of 64, patch 20, merge 2, 1e-5 norms" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var why: config.Why = .{};
    const c = (try config.parse(arena.allocator(), @embedFile("fixtures/mini_media_config.json"), &why)).vision;
    try t.expectEqual(@as(u32, 1024), c.hidden);
    try t.expectEqual(@as(u32, 23), c.layers);
    try t.expectEqual(@as(u32, 16), c.heads);
    try t.expectEqual(@as(u32, image.patch), c.patch);
    try t.expectEqual(@as(u32, image.merge), c.merge);
    try t.expectEqual(@as(u32, 32), c.grid);
    try t.expectEqual(@as(u32, 4), c.mlp_ratio);
    try t.expectEqual(@as(u32, 4096), c.proj_hidden);
    try t.expectEqual(@as(u32, 2304), c.text_hidden);
    try t.expectEqual(@as(f32, 1e-5), c.eps);
    try t.expectEqual(@as(usize, 1200), image.patch_width); // the checkpoint's patch_embedding.weight [1024, 1200]
}

test "the tower matches the f64 reference on a synthetic single-shard checkpoint read through its index" {
    const g = try Gpu.init();
    defer g.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = try towerConfig(a);
    const c = cfg.vision;
    const specs = try towerSpecs(a, c);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, t.io, tmp.dir, "model-00001-of-00001.safetensors", specs, 0x7157);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors.index.json", .data = try indexFor(a, specs) });
    const path = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    const shard = try a.dupeSentinel(u8, try std.fs.path.join(a, &.{ path, "model-00001-of-00001.safetensors" }), 0);
    var store = ck.Checkpoint.init(t.allocator);
    defer store.deinit();
    try store.addFileSelected(g.device, shard, "", "model.visual.");

    const tower = try vision.Tower.load(t.allocator, t.io, path, g.device, g.queue, &cfg, .{ .max_tokens = 16 });
    defer tower.deinit();
    try t.expect(tower.bytes() > 0);

    for ([_][2]usize{ .{ 80, 40 }, .{ 40, 40 }, .{ 61, 83 } }) |wh| {
        var rgb = try rgbImage(a, wh[0], wh[1]);
        defer rgb.deinit(a);
        const prepared = try image.fromRgb(a, rgb, 1, 16);
        try t.expectEqual(@as(u32, @intCast(image.tokensOf(wh[0], wh[1]))), prepared.tokens);
        const out = try tower.encodeInput(prepared.pixels, prepared.gh, prepared.gw);
        defer out.deinit();
        const want = try reference(a, &store, c, prepared.pixels, prepared.gh, prepared.gw);
        try t.expect(try relative(out.slice(u16, want.len), want) < 0.03);
    }

    // refusals: more tokens than the scratch holds, and pixels that are not that many tokens
    var rgb = try rgbImage(a, 80, 40);
    defer rgb.deinit(a);
    const prepared = try image.fromRgb(a, rgb, 1, 16);
    try t.expectError(error.ImageTooLarge, tower.encodeInput(prepared.pixels, 5, 4));
    try t.expectError(error.ImageGrid, tower.encodeInput(prepared.pixels[1..], prepared.gh, prepared.gw));
}

/// The index of the release: every tensor in the one shard, text tensors included.
fn indexFor(a: std.mem.Allocator, specs: []const Named) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "{\"metadata\":{},\"weight_map\":{");
    for (specs, 0..) |s, i| try out.print(a, "{s}\"{s}\":\"model-00001-of-00001.safetensors\"", .{ if (i == 0) "" else ",", s.name });
    try out.appendSlice(a, "}}");
    return out.items;
}

test "a tower refuses a checkpoint without vision tensors and a head width its kernels cannot run" {
    const g = try Gpu.init();
    defer g.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cfg = try towerConfig(arena.allocator());
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(path);
    cfg.vision.heads = 5; // 128 / 5
    try t.expectError(error.UnsupportedVisionGeometry, vision.Tower.load(t.allocator, t.io, path, g.device, g.queue, &cfg, .{}));
    cfg.vision.heads = 2;
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{}}" });
    try t.expectError(error.NoVisionTensors, vision.Tower.load(t.allocator, t.io, path, g.device, g.queue, &cfg, .{}));
}
