//! Native FP32 layouts corresponding to tools/drama3/codec_convert.py; no FP16 decoder image is reused.
const std = @import("std");
const pth = @import("codec_pth.zig");
const A = std.mem.Allocator;

const Reader = struct {
    a: A,
    archive: *const pth.Archive,

    fn fmt(r: Reader, comptime f: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(r.a, f, args);
    }

    fn raw(r: Reader, name: []const u8) ![]f32 {
        const t = try r.archive.tensor(name);
        var n: u64 = 1;
        for (t.shape) |d| n = std.math.mul(u64, n, d) catch return error.BadTensor;
        if (n > 64 * 1024 * 1024) return error.BadTensor;
        return r.archive.readF32(r.a, name, t.shape);
    }

    fn suff(r: Reader, prefix: []const u8, suffix: []const u8) ![]f32 {
        return r.raw(try r.fmt("{s}.{s}", .{ prefix, suffix }));
    }

    fn weight(r: Reader, prefix: []const u8) !struct { v: []f32, shape: []const u64 } {
        const plain = try r.fmt("{s}.weight", .{prefix});
        if (r.archive.tensors.get(plain)) |t| return .{ .v = try r.raw(plain), .shape = t.shape };
        const modern = try r.fmt("{s}.parametrizations.weight.original1", .{prefix});
        const modern_g = try r.fmt("{s}.parametrizations.weight.original0", .{prefix});
        const vn = if (r.archive.tensors.contains(modern)) modern else try r.fmt("{s}.weight_v", .{prefix});
        const gn = if (r.archive.tensors.contains(modern)) modern_g else try r.fmt("{s}.weight_g", .{prefix});
        const t = try r.archive.tensor(vn);
        const v = try r.raw(vn);
        try fold(v, try r.raw(gn));
        return .{ .v = v, .shape = t.shape };
    }

    fn conv(r: Reader, prefix: []const u8, kind: enum { taps, down, up }, suffix: []const u8, stride: usize) ![]f32 {
        if (std.mem.eql(u8, suffix, "b")) {
            const b = try r.suff(prefix, "bias");
            if (kind != .up) return b;
            const out = try r.a.alloc(f32, b.len * stride);
            for (0..stride) |j| @memcpy(out[j * b.len ..][0..b.len], b);
            return out;
        }
        const w = try r.weight(prefix);
        if (w.shape.len != 3) return error.BadTensor;
        const x: usize = @intCast(w.shape[0]);
        const y: usize = @intCast(w.shape[1]);
        const k: usize = @intCast(w.shape[2]);
        if (kind == .taps) {
            const out = try r.a.alloc(f32, w.v.len);
            for (0..k) |j| for (0..x) |o| for (0..y) |i| {
                out[(j * x + o) * y + i] = w.v[(o * y + i) * k + j];
            };
            return out;
        }
        if (stride == 0 or (k != stride and k != 2 * stride)) return error.BadTensor;
        const offset: usize = if (std.mem.eql(u8, suffix, "wb")) stride else 0;
        if (offset + stride > k) return error.BadTensor;
        const out = try r.a.alloc(f32, x * y * stride);
        if (kind == .up) {
            for (0..stride) |j| for (0..y) |o| for (0..x) |i| {
                out[(j * y + o) * x + i] = w.v[(i * y + o) * k + offset + j];
            };
        } else {
            for (0..x) |o| for (0..stride) |j| for (0..y) |i| {
                out[(o * stride + j) * y + i] = w.v[(o * y + i) * k + offset + j];
            };
        }
        return out;
    }

    fn tf(r: Reader, prefix: []const u8, parts: []const []const u8) ![]f32 {
        if (parts.len == 1 and eq(parts[0], "norm")) return r.suff(prefix, "norm.weight");
        if (parts.len != 2) return error.BadTensor;
        const p = try r.fmt("{s}.layers.{s}", .{ prefix, parts[0] });
        const fields = .{
            .{ "attn_norm", "attention_norm.weight" }, .{ "wqkv", "attention.wqkv.weight" },
            .{ "wo", "attention.wo.weight" },          .{ "ffn_norm", "ffn_norm.weight" },
            .{ "w2", "feed_forward.w2.weight" },       .{ "g_attn", "attention_layer_scale.gamma" },
            .{ "g_ffn", "ffn_layer_scale.gamma" },
        };
        inline for (fields) |f| if (eq(parts[1], f[0])) return r.suff(p, f[1]);
        if (!eq(parts[1], "w13")) return error.BadTensor;
        const w1 = try r.suff(p, "feed_forward.w1.weight");
        const w3 = try r.suff(p, "feed_forward.w3.weight");
        return std.mem.concat(r.a, f32, &.{ w1, w3 });
    }

    fn cn(r: Reader, p: []const u8, field: []const u8) ![]f32 {
        const fields = .{
            .{ "dw", "dwconv.conv.weight" }, .{ "dw_b", "dwconv.conv.bias" },
            .{ "ln_w", "norm.weight" },      .{ "ln_b", "norm.bias" },
            .{ "pw1", "pwconv1.weight" },    .{ "pw1_b", "pwconv1.bias" },
            .{ "pw2", "pwconv2.weight" },    .{ "pw2_b", "pwconv2.bias" },
            .{ "gamma", "gamma" },
        };
        inline for (fields) |f| if (eq(field, f[0])) return r.suff(p, f[1]);
        return error.BadTensor;
    }

    fn vq(r: Reader, field: []const u8) ![]f32 {
        const width: usize = if (eq(field, "table")) 1024 else if (eq(field, "cbn")) 8 else if (eq(field, "in_w")) 8192 else if (eq(field, "in_b")) 8 else return error.BadTensor;
        const books = eq(field, "table") or eq(field, "cbn");
        const out = try r.a.alloc(f32, (if (books) @as(usize, 13312) else 10) * width);
        var at: usize = 0;
        for (0..10) |i| {
            const p = if (i == 0) "quantizer.semantic_quantizer.quantizers.0" else try r.fmt("quantizer.quantizer.quantizers.{d}", .{i - 1});
            if (!books) {
                const values = if (eq(field, "in_w")) (try r.weight(try r.fmt("{s}.in_proj", .{p}))).v else try r.suff(p, "in_proj.bias");
                if (values.len != width) return error.BadTensor;
                @memcpy(out[at..][0..width], values);
                at += width;
                continue;
            }
            const cb = try r.suff(p, "codebook.weight");
            const n: usize = if (i == 0) 4096 else 1024;
            if (cb.len != n * 8) return error.BadTensor;
            if (eq(field, "cbn")) {
                for (0..n) |j| {
                    var norm: f32 = 0;
                    for (cb[j * 8 ..][0..8]) |v| norm += v * v;
                    norm = @max(@sqrt(norm), 1e-12);
                    for (0..8) |d| out[at + j * 8 + d] = cb[j * 8 + d] / norm;
                }
            } else {
                const w = (try r.weight(try r.fmt("{s}.out_proj", .{p}))).v;
                const b = try r.suff(p, "out_proj.bias");
                if (w.len != 1024 * 8 or b.len != 1024) return error.BadTensor;
                for (0..n) |j| for (0..1024) |o| {
                    var sum: f32 = 0;
                    for (0..8) |d| sum = @mulAdd(f32, cb[j * 8 + d], w[o * 8 + d], sum);
                    out[at + j * 1024 + o] = sum + b[o];
                };
            }
            at += n * width;
        }
        return out;
    }

    fn mapped(r: Reader, name: []const u8) ![]f32 {
        var parts: [8][]const u8 = undefined;
        var it = std.mem.splitScalar(u8, name, '.');
        var n: usize = 0;
        while (it.next()) |p| {
            if (n == parts.len) return error.BadTensor;
            parts[n] = p;
            n += 1;
        }
        if (n < 2) return error.BadTensor;
        const p = parts[0..n];
        if (eq(p[0], "vq") and n == 2) return r.vq(p[1]);
        if (eq(p[0], "post") or eq(p[0], "pre")) return r.tf(try r.fmt("quantizer.{s}_module", .{p[0]}), p[1..]);
        if (eq(p[0], "up") or eq(p[0], "down")) {
            const rawp = try r.fmt("quantizer.{s}sample.{s}", .{ p[0], p[1] });
            if (n == 4 and eq(p[2], "cn")) return r.cn(try r.fmt("{s}.1", .{rawp}), p[3]);
            if (n != 3) return error.BadTensor;
            return r.conv(try r.fmt("{s}.0.conv", .{rawp}), if (eq(p[0], "up")) .up else .down, p[2], 2);
        }
        const dec = eq(p[0], "dec");
        if (!dec and !eq(p[0], "enc")) return error.BadTensor;
        const base = if (dec) "decoder.model" else "encoder.block";
        if (n == 2) {
            if (eq(p[1], "final_alpha")) return r.suff(try r.fmt("{s}.5", .{base}), "alpha");
            const final = std.mem.startsWith(u8, p[1], "final");
            const prefix = try r.fmt("{s}.{d}.conv", .{ base, @as(usize, if (final) 6 else 0) });
            return r.conv(prefix, .taps, if (std.mem.endsWith(u8, p[1], "_b")) "b" else "taps", 1);
        }
        const bi = std.fmt.parseInt(usize, p[1], 10) catch return error.BadTensor;
        if (bi >= 4) return error.BadTensor;
        const bp = try r.fmt("{s}.{d}.block", .{ base, bi + 1 });
        if (eq(p[2], "tf")) return r.tf(try r.fmt("{s}.5", .{bp}), p[3..]);
        if (std.mem.startsWith(u8, p[2], "ru") and n == 4) {
            const ri = std.fmt.parseInt(usize, p[2][2..], 10) catch return error.BadTensor;
            if (ri >= 3) return error.BadTensor;
            const rp = try r.fmt("{s}.{d}.block", .{ bp, ri + @as(usize, if (dec) 2 else 0) });
            if (eq(p[3], "a1") or eq(p[3], "a2")) return r.suff(try r.fmt("{s}.{d}", .{ rp, @as(usize, if (eq(p[3], "a1")) 0 else 2) }), "alpha");
            return r.conv(try r.fmt("{s}.{d}.conv", .{ rp, @as(usize, if (std.mem.startsWith(u8, p[3], "c7")) 1 else 3) }), .taps, if (std.mem.endsWith(u8, p[3], "_b")) "b" else "taps", 1);
        }
        if (n != 3) return error.BadTensor;
        if (eq(p[2], "alpha")) return r.suff(try r.fmt("{s}.{d}", .{ bp, @as(usize, if (dec) 0 else 3) }), "alpha");
        const strides = if (dec) [4]usize{ 8, 8, 4, 2 } else [4]usize{ 2, 4, 8, 8 };
        return r.conv(try r.fmt("{s}.{d}.conv", .{ bp, @as(usize, if (dec) 1 else 4) }), if (dec) .up else .down, p[2], strides[bi]);
    }
};

fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn has(a: *const pth.Archive, name: []const u8) bool {
    if (!std.mem.startsWith(u8, name, "enc.") or !std.mem.endsWith(u8, name, ".tf.0.wqkv")) return false;
    var b: [128]u8 = undefined;
    const bi = std.fmt.parseInt(usize, name[4..5], 10) catch return false;
    const raw = std.fmt.bufPrint(&b, "encoder.block.{d}.block.5.layers.0.attention.wqkv.weight", .{bi + 1}) catch return false;
    return a.tensors.contains(raw);
}

pub fn read(gpa: A, a: *const pth.Archive, name: []const u8) ![]f32 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const r: Reader = .{ .a = arena.allocator(), .archive = a };
    return gpa.dupe(f32, try r.mapped(name));
}

fn fold(v: []f32, g: []const f32) !void {
    if (g.len == 0 or v.len % g.len != 0) return error.BadTensor;
    const row = v.len / g.len;
    var width: usize = 1;
    while (width < @min(row, 256)) : (width *= 2) {}
    var partial: [256]f32 = undefined;
    for (g, 0..) |gi, i| {
        const r = v[i * row ..][0..row];
        for (0..width) |lane| {
            var sum: f32 = 0;
            var col = lane;
            while (col < row) : (col += 256) sum = @mulAdd(f32, r[col], r[col], sum);
            partial[lane] = sum;
        }
        var stride = width / 2;
        while (stride != 0) : (stride /= 2) for (0..stride) |lane| {
            partial[lane] += partial[lane + stride];
        };
        const norm = @sqrt(partial[0]);
        if (!(norm > 0) or !std.math.isFinite(norm)) return error.BadTensor;
        for (r) |*x| x.* = gi * x.* * (1 / norm);
    }
}

test "native fold keeps FP32 values and rejects degenerate rows" {
    var v = [_]f32{ 3, 4, 0, 5 };
    try fold(&v, &.{ 10, 2 });
    try std.testing.expectEqualSlices(f32, &.{ 6, 8, 0, 2 }, &v);
    var zero = [_]f32{ 0, 0 };
    try std.testing.expectError(error.BadTensor, fold(&zero, &.{1}));
}
