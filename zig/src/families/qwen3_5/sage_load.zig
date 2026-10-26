//! Sage-A1.5 (qwen3_5_moe) weights on the host, for any backend: fp16 rows of small dense or MLX-quantized tensors,

// and the routed and shared expert tables in the order the MoE kernels read them. No device code here: a backend

// passes its checkpoint (anything with `get(name) !Tensor` and `has(name) bool`) and a sink that copies bytes out.

const std = @import("std");

fn bf16At(b: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, b[2 * i ..][0..2], .little)) << 16);
}

fn tensorName(buf: []u8, comptime f: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, f, args) catch error.NameTooLong;
}

/// fp16 [rows, k] of `prefix`.weight, which is BF16, F16 or MLX affine with groups of 64. The bit width comes from the

// shapes: Sage's MLX router (`mlp.gate`) and shared expert gate are 8 bit while its projections are 4 bit, so the

// router logits need this conversion before an fp16 kernel reads them. The caller owns the result.

pub fn halfRows(gpa: std.mem.Allocator, c: anytype, prefix: []const u8) ![]f16 {
    var nb: [256]u8 = undefined;
    const t = try c.get(try tensorName(&nb, "{s}.weight", .{prefix}));
    if (std.mem.eql(u8, t.dtype, "F16")) {
        const out = try gpa.alloc(f16, t.bytes.len / 2);
        @memcpy(std.mem.sliceAsBytes(out), t.bytes);
        return out;
    }
    if (std.mem.eql(u8, t.dtype, "BF16")) {
        const out = try gpa.alloc(f16, t.bytes.len / 2);
        errdefer gpa.free(out);
        const fp16_max: f32 = 65504.0;
        for (out, 0..) |*o, i| {
            const v = bf16At(t.bytes, i);
            if (!std.math.isFinite(v) or @abs(v) > fp16_max) return error.StoredPrecision;
            o.* = @floatCast(v);
        }
        return out;
    }
    if (!std.mem.eql(u8, t.dtype, "U32")) return error.UnexpectedDtype;
    const sc = try c.get(try tensorName(&nb, "{s}.scales", .{prefix}));
    const bi = try c.get(try tensorName(&nb, "{s}.biases", .{prefix}));
    return mlxRows(gpa, t.bytes, sc.bytes, bi.bytes, @intCast(t.shape[0]), @intCast(t.shape[1]));
}

/// MLX affine dequantization: w = scale * q + bias, q packed low bits first in little-endian u32 words, bf16 scales and

// biases for each group of 64 along a row.

pub fn mlxRows(gpa: std.mem.Allocator, words_b: []const u8, scales: []const u8, biases: []const u8, rows: usize, words: usize) ![]f16 {
    const groups = scales.len / 2 / rows;
    if (groups == 0 or biases.len != scales.len) return error.UnexpectedShape;
    const bits = words * 32 / (groups * 64);
    if (bits != 4 and bits != 8) return error.UnexpectedShape;
    const k = groups * 64;
    const out = try gpa.alloc(f16, rows * k);
    const mask: u32 = (@as(u32, 1) << @intCast(bits)) - 1;
    for (0..rows) |r| for (0..k) |j| {
        const bit = j * bits;
        const w = std.mem.readInt(u32, words_b[4 * (r * words + bit / 32) ..][0..4], .little);
        const q: f32 = @floatFromInt((w >> @intCast(bit % 32)) & mask);
        const g = r * groups + j / 64;
        out[r * k + j] = @floatCast(q * bf16At(scales, g) + bf16At(biases, g));
    };
    return out;
}

/// The six buffers of one expert table (MLX affine 4/64).
pub const Part = enum { gu_w, gu_s, gu_b, dn_w, dn_s, dn_b };

/// Bytes of one expert in each buffer. gate|up: the gate rows then the up rows, 2 ff rows of D inputs. down: D rows of

// ff inputs. One launch then reads both halves of the SwiGLU from one row index.

pub const Sizes = struct {
    gu_w: usize,
    gu_g: usize,
    dn_w: usize,
    dn_g: usize,

    pub fn of(ff: u32, d: u32) Sizes {
        return .{
            .gu_w = 2 * @as(usize, ff) * (d / 8) * 4,
            .gu_g = 2 * @as(usize, ff) * (d / 64) * 2,
            .dn_w = @as(usize, d) * (ff / 8) * 4,
            .dn_g = @as(usize, d) * (ff / 64) * 2,
        };
    }

    pub fn bytes(z: Sizes, p: Part) usize {
        return switch (p) {
            .gu_w => z.gu_w,
            .gu_s, .gu_b => z.gu_g,
            .dn_w => z.dn_w,
            .dn_s, .dn_b => z.dn_g,
        };
    }
};

/// Expert `e`'s bytes of one MLX tensor: from a stacked `switch_mlp` tensor [count, ...] (the MLX export) or from a

// tensor for each expert (`experts.E.`).

fn expertBytes(c: anytype, p: []const u8, part: []const u8, which: []const u8, e: u32, count: u32) ![]const u8 {
    var nb: [256]u8 = undefined;
    const stacked = try tensorName(&nb, "{s}mlp.switch_mlp.{s}.{s}", .{ p, part, which });
    if (c.has(stacked)) {
        const t = try c.get(stacked);
        if (t.shape[0] != count) return error.UnexpectedShape;
        const per = t.bytes.len / count;
        return t.bytes[per * e ..][0..per];
    }
    return (try c.get(try tensorName(&nb, "{s}mlp.experts.{d}.{s}.{s}", .{ p, e, part, which }))).bytes;
}

fn sharedBytes(c: anytype, p: []const u8, part: []const u8, which: []const u8) ![]const u8 {
    var nb: [256]u8 = undefined;
    return (try c.get(try tensorName(&nb, "{s}mlp.shared_expert.{s}.{s}", .{ p, part, which }))).bytes;
}

/// Packs layer `p`'s routed experts (`count` of them) or its shared expert (`shared`, count 1) into the table layout:

// calls `sink.put(part, offset, bytes)` for each piece, with `offset` in bytes from the start of that part's buffer.

pub fn packExperts(c: anytype, p: []const u8, count: u32, shared: bool, ff: u32, d: u32, sink: anytype) !void {
    const z = Sizes.of(ff, d);
    const whiches = [_][]const u8{ "weight", "scales", "biases" };
    const gu = [_]Part{ .gu_w, .gu_s, .gu_b };
    const dn = [_]Part{ .dn_w, .dn_s, .dn_b };
    for (0..count) |ei| {
        const e: u32 = @intCast(ei);
        for (whiches, 0..) |which, wi| {
            const gsz = z.bytes(gu[wi]);
            const dsz = z.bytes(dn[wi]);
            const gate = if (shared) try sharedBytes(c, p, "gate_proj", which) else try expertBytes(c, p, "gate_proj", which, e, count);
            const up = if (shared) try sharedBytes(c, p, "up_proj", which) else try expertBytes(c, p, "up_proj", which, e, count);
            const down = if (shared) try sharedBytes(c, p, "down_proj", which) else try expertBytes(c, p, "down_proj", which, e, count);
            if (gate.len * 2 != gsz or up.len * 2 != gsz or down.len != dsz) return error.UnexpectedShape;
            try sink.put(gu[wi], ei * gsz, gate);
            try sink.put(gu[wi], ei * gsz + gsz / 2, up);
            try sink.put(dn[wi], ei * dsz, down);
        }
    }
}

// ---------------------------------------------------------------- GPTQ (AutoRound) to MLX affine 4/64

/// bf16 bits of `x`, rounded to nearest even.
pub fn toBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// GPTQ symmetric 4 bit (auto_round:auto_gptq, v1 zeros) as MLX affine 4/64, with no requantization.

// qweight I32 [K/8, N] packs 8 values along K, low bits first: the MLX word [N, K/8] is the same word, transposed.

// Each GPTQ group of g (a multiple of 64) along K gives g/64 MLX groups: scale s, bias -s * (qzero + 1). The scale

// goes from fp16 to bf16, so w = bf16(s) * (q - zero) differs from the GPTQ value by the scale rounding only.

pub fn gptqAsMlx(qweight: []const u8, qzeros: []const u8, scales: []const u8, k: usize, n: usize, out_w: []u32, out_s: []u16, out_b: []u16) !void {
    if (k % 64 != 0 or n % 8 != 0 or qweight.len != k / 8 * n * 4) return error.UnexpectedShape;
    const groups = scales.len / 2 / n;
    if (groups == 0 or k % groups != 0 or (k / groups) % 64 != 0 or qzeros.len != groups * (n / 8) * 4) return error.UnexpectedShape;
    if (out_w.len != n * (k / 8) or out_s.len != n * (k / 64) or out_b.len != out_s.len) return error.UnexpectedShape;
    const g = k / groups;
    const words = k / 8;
    for (0..words) |kw| for (0..n) |c| {
        out_w[c * words + kw] = std.mem.readInt(u32, qweight[4 * (kw * n + c) ..][0..4], .little);
    };
    const per = g / 64;
    for (0..groups) |gi| for (0..n) |c| {
        const h: f16 = @bitCast(std.mem.readInt(u16, scales[2 * (gi * n + c) ..][0..2], .little));
        const sb = toBf16(@floatCast(h));
        const s: f32 = @bitCast(@as(u32, sb) << 16);
        const zw = std.mem.readInt(u32, qzeros[4 * (gi * (n / 8) + c / 8) ..][0..4], .little);
        const zero: f32 = @floatFromInt(((zw >> @intCast(4 * (c % 8))) & 15) + 1);
        const bias = toBf16(-s * zero);
        for (0..per) |j| {
            out_s[c * (k / 64) + gi * per + j] = sb;
            out_b[c * (k / 64) + gi * per + j] = bias;
        }
    };
}

/// bf16 rows [r, k] quantized to MLX affine 4/64 by rounding to the nearest level: scale (max - min) / 15 and bias

// min, both rounded to bf16 first, as an MLX export of the same rows.

pub fn quantizeRows(rows: []const u8, r: usize, k: usize, out_w: []u32, out_s: []u16, out_b: []u16) !void {
    if (k % 64 != 0 or rows.len != r * k * 2 or out_w.len != r * (k / 8) or out_s.len != r * (k / 64) or out_b.len != out_s.len) return error.UnexpectedShape;
    for (0..r) |i| for (0..k / 64) |gi| {
        var lo: f32 = std.math.inf(f32);
        var hi: f32 = -std.math.inf(f32);
        for (0..64) |j| {
            const v = bf16At(rows, i * k + gi * 64 + j);
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        const sb = toBf16((hi - lo) / 15.0);
        const bb = toBf16(lo);
        const s: f32 = @bitCast(@as(u32, sb) << 16);
        const b: f32 = @bitCast(@as(u32, bb) << 16);
        out_s[i * (k / 64) + gi] = sb;
        out_b[i * (k / 64) + gi] = bb;
        for (0..8) |wi| {
            var word: u32 = 0;
            for (0..8) |j| {
                const v = bf16At(rows, i * k + gi * 64 + wi * 8 + j);
                const qf = if (s > 0) @round((v - b) / s) else 0;
                const qv: u32 = @intFromFloat(std.math.clamp(qf, 0, 15));
                word |= qv << @intCast(4 * j);
            }
            out_w[i * (k / 8) + gi * 8 + wi] = word;
        }
    };
}

/// bf16 weights plus one: an MLX export stores Sage's RMSNorm weights (layer, q and k norms, the final norm) as 1 + w,

// and the kernels read them that way. The gated GDN norm is stored as is.

pub fn plusOne(in: []const u8, out: []u16) void {
    for (out, 0..) |*o, i| o.* = toBf16(bf16At(in, i) + 1.0);
}

test "GPTQ words transpose to MLX words, groups of 128 split in two, bias is -scale * zero" {
    // K 128, N 8: q(k, c) = (k + c) % 16, scale(c) = 0.5 + c, stored zero 7 (zero 8)
    const k = 128;
    const n = 8;
    var qw: [k / 8 * n]u32 = @splat(0);
    for (0..k) |kk| for (0..n) |c| {
        qw[(kk / 8) * n + c] |= @as(u32, @intCast((kk + c) % 16)) << @intCast(4 * (kk % 8));
    };
    var sc: [n]f16 = undefined;
    for (&sc, 0..) |*x, c| x.* = 0.5 + @as(f16, @floatFromInt(c));
    const qz = [_]u32{0x77777777};
    var w: [n * k / 8]u32 = undefined;
    var s: [n * 2]u16 = undefined;
    var b: [n * 2]u16 = undefined;
    try gptqAsMlx(std.mem.sliceAsBytes(&qw), std.mem.sliceAsBytes(&qz), std.mem.sliceAsBytes(&sc), k, n, &w, &s, &b);
    const ww = try mlxRows(std.testing.allocator, std.mem.sliceAsBytes(&w), std.mem.sliceAsBytes(&s), std.mem.sliceAsBytes(&b), n, k / 8);
    defer std.testing.allocator.free(ww);
    for (0..n) |c| for (0..k) |kk| {
        const want = (0.5 + @as(f32, @floatFromInt(c))) * (@as(f32, @floatFromInt((kk + c) % 16)) - 8);
        try std.testing.expectEqual(want, @as(f32, ww[c * k + kk]));
    };
}

test "quantized rows come back within half a step" {
    const k = 64;
    var rows: [k]u16 = undefined;
    for (&rows, 0..) |*x, j| x.* = toBf16(@as(f32, @floatFromInt(j)) * 0.03 - 1.0);
    var w: [8]u32 = undefined;
    var s: [1]u16 = undefined;
    var b: [1]u16 = undefined;
    try quantizeRows(std.mem.sliceAsBytes(&rows), 1, k, &w, &s, &b);
    const back = try mlxRows(std.testing.allocator, std.mem.sliceAsBytes(&w), std.mem.sliceAsBytes(&s), std.mem.sliceAsBytes(&b), 1, 8);
    defer std.testing.allocator.free(back);
    const step: f32 = @bitCast(@as(u32, s[0]) << 16);
    for (back, 0..) |v, j| try std.testing.expect(@abs(@as(f32, v) - bf16At(std.mem.sliceAsBytes(&rows), j)) <= step * 0.51 + 1e-3);
}

const TestTensor = struct { dtype: []const u8, shape: []const i64, bytes: []const u8 };
const TestCkpt = struct {
    names: []const []const u8,
    tensors: []const TestTensor,
    fn get(c: TestCkpt, n: []const u8) !TestTensor {
        for (c.names, c.tensors) |m, t| if (std.mem.eql(u8, m, n)) return t;
        return error.MissingTensor;
    }
    fn has(c: TestCkpt, n: []const u8) bool {
        for (c.names) |m| if (std.mem.eql(u8, m, n)) return true;
        return false;
    }
};

test "8 bit MLX rows: scale * q + bias, low bits first" {
    // one row of 64 values: q = 0, 1, ..., 63 packed 4 to a word; scale 0.5, bias -1
    var w: [16]u32 = undefined;
    for (&w, 0..) |*x, i| {
        x.* = 0;
        for (0..4) |j| x.* |= @as(u32, @intCast(4 * i + j)) << @intCast(8 * j);
    }
    const sc = [_]u16{0x3f00}; // bf16 0.5
    const bi = [_]u16{0xbf80}; // bf16 -1
    const c: TestCkpt = .{ .names = &.{ "g.weight", "g.scales", "g.biases" }, .tensors = &.{
        .{ .dtype = "U32", .shape = &.{ 1, 16 }, .bytes = std.mem.sliceAsBytes(&w) },
        .{ .dtype = "BF16", .shape = &.{ 1, 1 }, .bytes = std.mem.sliceAsBytes(&sc) },
        .{ .dtype = "BF16", .shape = &.{ 1, 1 }, .bytes = std.mem.sliceAsBytes(&bi) },
    } };
    const out = try halfRows(std.testing.allocator, c, "g");
    defer std.testing.allocator.free(out);
    for (out, 0..) |v, j| try std.testing.expectEqual(@as(f16, @floatFromInt(j)) * 0.5 - 1, v);
}

test "expert tables: gate rows then up rows, down after, for each stacked expert" {
    // ff 64, D 64: gate|up words 2*64*8*4 bytes, groups 2*64*1*2; down words 64*8*4, groups 64*1*2
    const z = Sizes.of(64, 64);
    const names = [_][]const u8{ "mlp.switch_mlp.gate_proj.weight", "mlp.switch_mlp.gate_proj.scales", "mlp.switch_mlp.gate_proj.biases", "mlp.switch_mlp.up_proj.weight", "mlp.switch_mlp.up_proj.scales", "mlp.switch_mlp.up_proj.biases", "mlp.switch_mlp.down_proj.weight", "mlp.switch_mlp.down_proj.scales", "mlp.switch_mlp.down_proj.biases" };
    var store: [9][2 * 2048]u8 = undefined;
    var ts: [9]TestTensor = undefined;
    for (0..9) |i| {
        const which = i % 3;
        const per = if (which == 0) z.gu_w / 2 else z.gu_g / 2; // gate, up and down hold the same bytes at ff = D
        for (0..2) |e| @memset(store[i][e * per ..][0..per], @intCast(16 * (i / 3) + 4 * which + e + 1));
        ts[i] = .{ .dtype = "U32", .shape = &.{ 2, 1 }, .bytes = store[i][0 .. 2 * per] };
    }
    const c: TestCkpt = .{ .names = &names, .tensors = &ts };
    const Sink = struct {
        out: [6][2 * 4096]u8 = undefined,
        fn put(s: *@This(), part: Part, at: usize, b: []const u8) !void {
            @memcpy(s.out[@backingInt(part)][at..][0..b.len], b);
        }
    };
    var s: Sink = .{};
    try packExperts(c, "", 2, false, 64, 64, &s);
    // expert 1's gate|up weight row block: gate bytes (16*0 + 0 + 2) then up bytes (16*1 + 0 + 2)
    try std.testing.expectEqual(@as(u8, 2), s.out[@backingInt(Part.gu_w)][z.gu_w]);
    try std.testing.expectEqual(@as(u8, 18), s.out[@backingInt(Part.gu_w)][z.gu_w + z.gu_w / 2]);
    try std.testing.expectEqual(@as(u8, 33 + 4), s.out[@backingInt(Part.dn_s)][0]);
}
