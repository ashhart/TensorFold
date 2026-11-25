//! AutoAWQ (4-bit, group_size 128, zero_point=true) unpacker and lossless FP32 metadata repacker for M5 and M3.
const std = @import("std");

pub const group_size: usize = 128;
pub const bits: u8 = 4;

/// AutoAWQ packs logical columns [0,2,4,6,1,3,5,7] into nibbles 0..7; column c sits at nibble awq_nibble_order[c].
pub const awq_nibble_order = [8]u5{ 0, 4, 1, 5, 2, 6, 3, 7 };
pub const awq_shifts = [8]u5{ 0, 16, 4, 20, 8, 24, 12, 28 };

/// Lossless group metadata: f32 scale and bias = -z * scale (0-ULP exact for F16/BF16 scales and 4-bit zero).
pub const Float2 = extern struct {
    scale: f32,
    bias: f32,
};

pub const ScaleDType = enum {
    f16,
    bf16,
    f32,

    pub fn byteSize(d: ScaleDType) usize {
        return switch (d) {
            .f16, .bf16 => 2,
            .f32 => 4,
        };
    }
};

pub const Layout = enum { m5_tile, m3_row };

/// Parsed `quantization_config` / `quantization` for AutoAWQ GEMM.
pub const Config = struct {
    bits: u8 = 4,
    group_size: usize = 128,
    zero_point: bool = true,
    modules_to_not_convert: []const []const u8 = &.{},

    pub fn deinit(c: *Config, a: std.mem.Allocator) void {
        for (c.modules_to_not_convert) |m| a.free(m);
        a.free(c.modules_to_not_convert);
        c.* = .{};
    }

    pub fn isQuantized(c: Config, key: []const u8) bool {
        for (c.modules_to_not_convert) |skip| {
            if (std.mem.eql(u8, key, skip)) return false;
            if (std.mem.startsWith(u8, key, skip) and key.len > skip.len and key[skip.len] == '.') return false;
            if (std.mem.endsWith(u8, key, skip) and key.len > skip.len and
                key[key.len - skip.len - 1] == '.') return false;
            var buf: [256]u8 = undefined;
            if (skip.len + 2 <= buf.len) {
                buf[0] = '.';
                @memcpy(buf[1..][0..skip.len], skip);
                buf[1 + skip.len] = '.';
                if (std.mem.indexOf(u8, key, buf[0 .. skip.len + 2]) != null) return false;
            }
        }
        return true;
    }
};

fn findQuantObject(root: std.json.ObjectMap) ?std.json.ObjectMap {
    for ([_][]const u8{ "quantization_config", "quantization" }) |k| {
        if (root.get(k)) |v| if (v == .object) return v.object;
    }
    if (root.get("text_config")) |tc| if (tc == .object) {
        for ([_][]const u8{ "quantization_config", "quantization" }) |k| {
            if (tc.object.get(k)) |v| if (v == .object) return v.object;
        }
    };
    return null;
}

pub fn parseConfig(a: std.mem.Allocator, root: std.json.ObjectMap) !?Config {
    const obj = findQuantObject(root) orelse return null;
    if (obj.get("quant_method")) |qm| {
        if (qm != .string) return error.UnsupportedQuantization;
        if (!std.ascii.eqlIgnoreCase(qm.string, "awq")) {
            return null;
        }
    } else return null;

    if (obj.get("version")) |ver| {
        if (ver != .string or !std.ascii.eqlIgnoreCase(ver.string, "gemm")) return error.UnsupportedQuantization;
    }
    const b_val = obj.get("bits") orelse obj.get("w_bit");
    const b_int: i64 = if (b_val) |v| (if (v == .integer) v.integer else return error.UnsupportedQuantization) else 4;
    if (b_int != 4) return error.UnsupportedQuantization;

    const g_val = obj.get("group_size") orelse obj.get("q_group_size");
    const g_int: i64 = if (g_val) |v| (if (v == .integer) v.integer else return error.UnsupportedQuantization) else 128;
    if (g_int != 128) return error.UnsupportedQuantization;

    const zp_val = obj.get("zero_point");
    const zp = if (zp_val) |v| (if (v == .bool) v.bool else return error.UnsupportedQuantization) else true;
    if (!zp) return error.UnsupportedQuantization;

    var skips: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (skips.items) |s| a.free(s);
        skips.deinit(a);
    }
    if (obj.get("modules_to_not_convert")) |m| {
        if (m != .array) return error.UnsupportedQuantization;
        for (m.array.items) |item| {
            if (item != .string) return error.UnsupportedQuantization;
            try skips.append(a, try a.dupe(u8, item.string));
        }
    }
    return .{ .bits = 4, .group_size = 128, .zero_point = true, .modules_to_not_convert = try skips.toOwnedSlice(a) };
}

pub inline fn unpackNibble(word: u32, col_mod_8: usize) u4 {
    return @truncate(word >> awq_shifts[col_mod_8]);
}

pub inline fn packAwqWord(cols: [8]u4) u32 {
    const order = [8]usize{ 0, 2, 4, 6, 1, 3, 5, 7 };
    var w: u32 = 0;
    for (order, 0..) |c, i| w |= @as(u32, cols[c]) << @intCast(4 * i);
    return w;
}

pub inline fn loadScale(dtype: ScaleDType, bytes: []const u8, idx: usize) f32 {
    return switch (dtype) {
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * idx ..][0..2], .little)))),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * idx ..][0..2], .little)) << 16),
        .f32 => @bitCast(std.mem.readInt(u32, bytes[4 * idx ..][0..4], .little)),
    };
}

pub inline fn losslessMeta(scale: f32, zero: u4) !Float2 {
    if (!std.math.isFinite(scale)) return error.LossyAwqScaleBias;
    if (zero == 0) return .{ .scale = scale, .bias = 0.0 };
    const exact: f64 = -@as(f64, @floatFromInt(zero)) * @as(f64, scale);
    const bias: f32 = @floatCast(exact);
    if (!std.math.isFinite(bias) or @as(f64, bias) != exact) return error.LossyAwqScaleBias;
    return .{ .scale = scale, .bias = bias };
}

/// Raw AutoAWQ tensors for a single projection [N, K]: qweight [K, N/8], qzeros [K/128, N/8], scales [K/128, N].
pub const Part = struct {
    n: usize,
    k: usize,
    qweight: []const u32,
    qzeros: []const u32,
    scales: []const u8,
    scales_dtype: ScaleDType,
    input_scale: ?[]const f32 = null,
};

pub fn validatePart(p: Part) !void {
    if (p.n == 0 or p.n % 8 != 0 or p.k == 0 or p.k % group_size != 0) return error.InvalidAwqShape;
    const kg = p.k / group_size;
    if (p.qweight.len != p.k * (p.n / 8)) return error.InvalidAwqStorage;
    if (p.qzeros.len != kg * (p.n / 8)) return error.InvalidAwqStorage;
    if (p.scales.len != kg * p.n * p.scales_dtype.byteSize()) return error.InvalidAwqStorage;
    if (p.input_scale) |is| if (is.len != p.k) return error.InvalidAwqStorage;
}

fn inputScaleEql(a_opt: ?[]const f32, b_opt: ?[]const f32) bool {
    if (a_opt == null and b_opt == null) return true;
    const a_s = a_opt orelse return false;
    const b_s = b_opt orelse return false;
    return std.mem.eql(u8, std.mem.sliceAsBytes(a_s), std.mem.sliceAsBytes(b_s));
}

/// True when parts share the same K and identical optional input_scale (can stack along N into one projection).
pub fn canStack(parts: []const Part) !bool {
    if (parts.len == 0) return error.EmptyAwqParts;
    for (parts) |p| try validatePart(p);
    const k0 = parts[0].k;
    const is0 = parts[0].input_scale;
    for (parts[1..]) |p| {
        if (p.k != k0) return error.MismatchedAwqK;
        if (!inputScaleEql(is0, p.input_scale)) return false;
    }
    return true;
}

pub fn weightWords(n_total: usize, k: usize, layout: Layout) usize {
    return switch (layout) {
        .m5_tile => ((n_total + 31) / 32) * (k / group_size) * 32 * 16,
        .m3_row => n_total * (k / 8),
    };
}

pub fn metaCount(n_total: usize, k: usize) usize {
    return n_total * (k / group_size);
}

/// Repacks 1 or more AutoAWQ parts (stacked along N) into M5 tile [tile_n][KG][32][16] or M3 row [N][K/8] layout.
pub fn packParts(parts: []const Part, layout: Layout, out_w: []u32, out_sb: []Float2) !void {
    if (parts.len == 0) return error.EmptyAwqParts;
    const k = parts[0].k;
    var n_total: usize = 0;
    for (parts) |p| {
        try validatePart(p);
        if (p.k != k) return error.MismatchedAwqK;
        n_total += p.n;
    }
    if (out_w.len != weightWords(n_total, k, layout) or out_sb.len != metaCount(n_total, k)) {
        return error.InvalidAwqStorage;
    }
    @memset(out_w, 0);
    const kg = k / group_size;
    const kw_row = k / 8;

    var n_base: usize = 0;
    for (parts) |p| {
        const pn8 = p.n / 8;
        for (0..kg) |g| {
            for (0..p.n) |nl| {
                const ng = n_base + nl;
                const sc = loadScale(p.scales_dtype, p.scales, g * p.n + nl);
                const zp = unpackNibble(p.qzeros[g * pn8 + nl / 8], nl % 8);
                const sb = try losslessMeta(sc, zp);
                switch (layout) {
                    .m5_tile => out_sb[g * n_total + ng] = sb,
                    .m3_row => out_sb[ng * kg + g] = sb,
                }
            }
            for (0..pn8) |c8| {
                for (0..16) |w16| {
                    const k0 = g * group_size + w16 * 8;
                    var row_words: [8]u32 = undefined;
                    for (0..8) |j| row_words[j] = p.qweight[(k0 + j) * pn8 + c8];
                    for (0..8) |c| {
                        var packed_k: u32 = 0;
                        inline for (0..8) |j| {
                            packed_k |= @as(u32, unpackNibble(row_words[j], c)) << @intCast(4 * j);
                        }
                        const ng = n_base + c8 * 8 + c;
                        switch (layout) {
                            .m5_tile => {
                                const tile = ng / 32;
                                const col = ng % 32;
                                out_w[((tile * kg + g) * 32 + col) * 16 + w16] = packed_k;
                            },
                            .m3_row => out_w[ng * kw_row + g * 16 + w16] = packed_k,
                        }
                    }
                }
            }
        }
        n_base += p.n;
    }
}

pub const PackedBundle = struct {
    n: usize,
    k: usize,
    layout: Layout,
    w: []u32,
    sb: []Float2,
    input_scale: ?[]f32,

    pub fn deinit(b: *PackedBundle, a: std.mem.Allocator) void {
        a.free(b.w);
        a.free(b.sb);
        if (b.input_scale) |is| a.free(is);
        b.* = undefined;
    }
};

/// Packs parts into 1 stacked bundle when input_scale matches across all parts, else per-part bundles (#1744).
pub fn packBundles(a: std.mem.Allocator, parts: []const Part, layout: Layout) ![]PackedBundle {
    if (try canStack(parts)) {
        const k = parts[0].k;
        var n_total: usize = 0;
        for (parts) |p| n_total += p.n;
        const w = try a.alloc(u32, weightWords(n_total, k, layout));
        errdefer a.free(w);
        const sb = try a.alloc(Float2, metaCount(n_total, k));
        errdefer a.free(sb);
        try packParts(parts, layout, w, sb);
        const is = if (parts[0].input_scale) |s| try a.dupe(f32, s) else null;
        errdefer if (is) |s| a.free(s);
        const out = try a.alloc(PackedBundle, 1);
        out[0] = .{ .n = n_total, .k = k, .layout = layout, .w = w, .sb = sb, .input_scale = is };
        return out;
    }
    const out = try a.alloc(PackedBundle, parts.len);
    var built: usize = 0;
    errdefer {
        for (out[0..built]) |*b| b.deinit(a);
        a.free(out);
    }
    for (parts) |p| {
        const w = try a.alloc(u32, weightWords(p.n, p.k, layout));
        errdefer a.free(w);
        const sb = try a.alloc(Float2, metaCount(p.n, p.k));
        errdefer a.free(sb);
        try packParts(&.{p}, layout, w, sb);
        const is = if (p.input_scale) |s| try a.dupe(f32, s) else null;
        errdefer if (is) |s| a.free(s);
        out[built] = .{ .n = p.n, .k = p.k, .layout = layout, .w = w, .sb = sb, .input_scale = is };
        built += 1;
    }
    return out;
}

/// Float64 reference dequantization [N_total, K] directly from raw AutoAWQ tensors.
pub fn dequantWeightsF64(parts: []const Part, out: []f64) !void {
    if (parts.len == 0) return error.EmptyAwqParts;
    const k = parts[0].k;
    var n_total: usize = 0;
    for (parts) |p| {
        try validatePart(p);
        if (p.k != k) return error.MismatchedAwqK;
        n_total += p.n;
    }
    if (out.len != n_total * k) return error.InvalidAwqStorage;
    var n_base: usize = 0;
    for (parts) |p| {
        const pn8 = p.n / 8;
        for (0..p.n) |nl| {
            const ng = n_base + nl;
            for (0..k) |ki| {
                const g = ki / group_size;
                const q = unpackNibble(p.qweight[ki * pn8 + nl / 8], nl % 8);
                const z = unpackNibble(p.qzeros[g * pn8 + nl / 8], nl % 8);
                const sc = loadScale(p.scales_dtype, p.scales, g * p.n + nl);
                out[ng * k + ki] = (@as(f64, @floatFromInt(q)) - @as(f64, @floatFromInt(z))) * @as(f64, sc);
            }
        }
        n_base += p.n;
    }
}

/// Float64 dequantization [N, K] from repacked M5 or M3 layout via stored f32 scale and bias.
pub fn dequantPackedF64(n: usize, k: usize, layout: Layout, w: []const u32, sb: []const Float2, out: []f64) void {
    const kg = k / group_size;
    const kw_row = k / 8;
    for (0..n) |ni| {
        for (0..k) |ki| {
            const g = ki / group_size;
            const w16 = (ki % group_size) / 8;
            const nib: u5 = @intCast(4 * (ki % 8));
            const word = switch (layout) {
                .m5_tile => w[((ni / 32 * kg + g) * 32 + (ni % 32)) * 16 + w16],
                .m3_row => w[ni * kw_row + g * 16 + w16],
            };
            const q: u4 = @truncate(word >> nib);
            const meta = switch (layout) {
                .m5_tile => sb[g * n + ni],
                .m3_row => sb[ni * kg + g],
            };
            out[ni * k + ki] = @as(f64, @floatFromInt(q)) * @as(f64, meta.scale) + @as(f64, meta.bias);
        }
    }
}

fn fillSyntheticAwq(
    a: std.mem.Allocator,
    n: usize,
    k: usize,
    dtype: ScaleDType,
    seed: u64,
    input_scale: ?[]const f32,
) !struct { p: Part, qw: []u32, qz: []u32, sc: []u8, ref: []f64 } {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const kg = k / group_size;
    const n8 = n / 8;
    const qw = try a.alloc(u32, k * n8);
    const qz = try a.alloc(u32, kg * n8);
    const sc = try a.alloc(u8, kg * n * dtype.byteSize());
    const ref = try a.alloc(f64, n * k);
    const q_plain = try a.alloc(u4, k * n);
    defer a.free(q_plain);
    const z_plain = try a.alloc(u4, kg * n);
    defer a.free(z_plain);

    for (0..kg) |g| {
        for (0..n) |ni| {
            z_plain[g * n + ni] = @intCast(rng.uintLessThan(u32, 16));
            const raw_s: f32 = (rng.float(f32) * 0.05 + 0.002) * (if (rng.boolean()) @as(f32, 1.0) else -1.0);
            switch (dtype) {
                .f16 => {
                    const h: f16 = @floatCast(raw_s);
                    std.mem.writeInt(u16, sc[2 * (g * n + ni) ..][0..2], @bitCast(h), .little);
                },
                .bf16 => {
                    const u: u32 = @bitCast(raw_s);
                    const b: u16 = @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
                    std.mem.writeInt(u16, sc[2 * (g * n + ni) ..][0..2], b, .little);
                },
                .f32 => std.mem.writeInt(u32, sc[4 * (g * n + ni) ..][0..4], @bitCast(raw_s), .little),
            }
        }
        for (0..n8) |c8| {
            var cols: [8]u4 = undefined;
            for (0..8) |c| cols[c] = z_plain[g * n + c8 * 8 + c];
            qz[g * n8 + c8] = packAwqWord(cols);
        }
    }
    for (0..k) |ki| {
        for (0..n8) |c8| {
            var cols: [8]u4 = undefined;
            for (0..8) |c| {
                const q: u4 = @intCast(rng.uintLessThan(u32, 16));
                q_plain[ki * n + c8 * 8 + c] = q;
                cols[c] = q;
            }
            qw[ki * n8 + c8] = packAwqWord(cols);
        }
    }
    for (0..n) |ni| {
        for (0..k) |ki| {
            const g = ki / group_size;
            const sf = loadScale(dtype, sc, g * n + ni);
            const q = q_plain[ki * n + ni];
            const z = z_plain[g * n + ni];
            ref[ni * k + ki] = (@as(f64, @floatFromInt(q)) - @as(f64, @floatFromInt(z))) * @as(f64, sf);
        }
    }
    return .{
        .p = .{
            .n = n,
            .k = k,
            .qweight = qw,
            .qzeros = qz,
            .scales = sc,
            .scales_dtype = dtype,
            .input_scale = input_scale,
        },
        .qw = qw,
        .qz = qz,
        .sc = sc,
        .ref = ref,
    };
}

test "test_awq: bit-exact unpack and repack for [0,2,4,6,1,3,5,7] with F16 and BF16 scales" {
    const a = std.testing.allocator;
    // Verify exact nibble shift inverse on explicit [0,2,4,6,1,3,5,7] packed word.
    const cols = [8]u4{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const word = packAwqWord(cols);
    try std.testing.expectEqual(@as(u32, 0x8642_7531), word);
    for (cols, 0..) |want, c| try std.testing.expectEqual(want, unpackNibble(word, c));

    for ([_]ScaleDType{ .f16, .bf16 }) |dt| {
        for ([_]usize{ 40, 64 }) |n| {
            const k: usize = 256;
            const syn = try fillSyntheticAwq(a, n, k, dt, 0xA11CE + n + @backingInt(dt), null);
            defer {
                a.free(syn.qw);
                a.free(syn.qz);
                a.free(syn.sc);
                a.free(syn.ref);
            }
            const ref_parts = try a.alloc(f64, n * k);
            defer a.free(ref_parts);
            try dequantWeightsF64(&.{syn.p}, ref_parts);
            try std.testing.expectEqualSlices(f64, syn.ref, ref_parts);

            for ([_]Layout{ .m5_tile, .m3_row }) |layout| {
                const w = try a.alloc(u32, weightWords(n, k, layout));
                defer a.free(w);
                const sb = try a.alloc(Float2, metaCount(n, k));
                defer a.free(sb);
                try packParts(&.{syn.p}, layout, w, sb);
                const got = try a.alloc(f64, n * k);
                defer a.free(got);
                dequantPackedF64(n, k, layout, w, sb, got);
                // Every weight (q * s + (-z * s)) matches (q - z) * s within 0 ULP of exact f32 metadata.
                for (got, syn.ref) |gv, wv| try std.testing.expectEqual(wv, gv);
            }
        }
    }
}

test "test_awq: stacking matching input_scale vs per-part split when input_scale differs" {
    const a = std.testing.allocator;
    const k: usize = 128;
    const is_a = try a.alloc(f32, k);
    defer a.free(is_a);
    const is_b = try a.alloc(f32, k);
    defer a.free(is_b);
    for (is_a, is_b, 0..) |*pa, *pb, i| {
        pa.* = 1.0 + @as(f32, @floatFromInt(i % 7)) * 0.125;
        pb.* = pa.* + 0.5;
    }
    const p0 = try fillSyntheticAwq(a, 16, k, .f16, 101, is_a);
    defer {
        a.free(p0.qw);
        a.free(p0.qz);
        a.free(p0.sc);
        a.free(p0.ref);
    }
    const p1_match = try fillSyntheticAwq(a, 24, k, .f16, 202, is_a);
    defer {
        a.free(p1_match.qw);
        a.free(p1_match.qz);
        a.free(p1_match.sc);
        a.free(p1_match.ref);
    }
    const p1_diff = try fillSyntheticAwq(a, 24, k, .bf16, 303, is_b);
    defer {
        a.free(p1_diff.qw);
        a.free(p1_diff.qz);
        a.free(p1_diff.sc);
        a.free(p1_diff.ref);
    }

    const stacked = try packBundles(a, &.{ p0.p, p1_match.p }, .m5_tile);
    defer {
        for (stacked) |*b| b.deinit(a);
        a.free(stacked);
    }
    try std.testing.expectEqual(@as(usize, 1), stacked.len);
    try std.testing.expectEqual(@as(usize, 40), stacked[0].n);

    const ref_stack = try a.alloc(f64, 40 * k);
    defer a.free(ref_stack);
    try dequantWeightsF64(&.{ p0.p, p1_match.p }, ref_stack);
    const got_stack = try a.alloc(f64, 40 * k);
    defer a.free(got_stack);
    dequantPackedF64(40, k, .m5_tile, stacked[0].w, stacked[0].sb, got_stack);
    try std.testing.expectEqualSlices(f64, ref_stack, got_stack);

    const split = try packBundles(a, &.{ p0.p, p1_diff.p }, .m3_row);
    defer {
        for (split) |*b| b.deinit(a);
        a.free(split);
    }
    try std.testing.expectEqual(@as(usize, 2), split.len);
    try std.testing.expectEqual(@as(usize, 16), split[0].n);
    try std.testing.expectEqual(@as(usize, 24), split[1].n);
}

test "test_awq: quantization_config parsing and modules_to_not_convert matching" {
    const a = std.testing.allocator;
    const text =
        \\{"quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128,
        \\ "zero_point": true, "version": "gemm", "modules_to_not_convert": ["lm_head", "visual"]}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    var cfg = (try parseConfig(a, parsed.value.object)).?;
    defer cfg.deinit(a);
    try std.testing.expectEqual(@as(u8, 4), cfg.bits);
    try std.testing.expectEqual(@as(usize, 128), cfg.group_size);
    try std.testing.expect(!cfg.isQuantized("lm_head"));
    try std.testing.expect(!cfg.isQuantized("model.visual.blocks.0.attn.qkv"));
    try std.testing.expect(cfg.isQuantized("model.layers.0.self_attn.q_proj"));
    const legacy = try std.json.parseFromSlice(std.json.Value, a, "{\"loop_awq_config\":{\"quant_method\":\"loop_awq\"}}", .{});
    defer legacy.deinit();
    try std.testing.expect((try parseConfig(a, legacy.value.object)) == null);
}

test "test_awq: lossy F32 scales fail losslessMeta and packParts while exact scales reconstruct zero" {
    const lossy_sc: f32 = @bitCast(@as(u32, 0x3f800001));
    try std.testing.expectError(error.LossyAwqScaleBias, losslessMeta(lossy_sc, 15));
    try std.testing.expectError(error.LossyAwqScaleBias, losslessMeta(std.math.nan(f32), 0));
    try std.testing.expectError(error.LossyAwqScaleBias, losslessMeta(std.math.inf(f32), 1));
    try std.testing.expectEqual(@as(f32, 0.0), (try losslessMeta(lossy_sc, 0)).bias);
    try std.testing.expectEqual(-8.0 * lossy_sc, (try losslessMeta(lossy_sc, 8)).bias);

    const a = std.testing.allocator;
    const n: usize = 8;
    const k: usize = group_size;
    var qw: [k * (n / 8)]u32 = @splat(0xffff_ffff);
    var qz: [1]u32 = .{0xffff_ffff};
    var sc_lossy: [n]u32 = @splat(0x3f800001);
    var sc_exact: [n]u32 = @splat(@bitCast(@as(f32, 0.5)));
    const p_bad: Part = .{ .n = n, .k = k, .qweight = &qw, .qzeros = &qz, .scales = std.mem.sliceAsBytes(&sc_lossy), .scales_dtype = .f32 };
    const p_ok: Part = .{ .n = n, .k = k, .qweight = &qw, .qzeros = &qz, .scales = std.mem.sliceAsBytes(&sc_exact), .scales_dtype = .f32 };
    const sb = try a.alloc(Float2, metaCount(n, k));
    defer a.free(sb);
    for ([_]Layout{ .m5_tile, .m3_row }) |layout| {
        const w_l = try a.alloc(u32, weightWords(n, k, layout));
        defer a.free(w_l);
        try std.testing.expectError(error.LossyAwqScaleBias, packParts(&.{p_bad}, layout, w_l, sb));
        try packParts(&.{p_ok}, layout, w_l, sb);
        const got = try a.alloc(f64, n * k);
        defer a.free(got);
        dequantPackedF64(n, k, layout, w_l, sb, got);
        for (got) |v| try std.testing.expectEqual(@as(f64, 0.0), v);
    }
}
