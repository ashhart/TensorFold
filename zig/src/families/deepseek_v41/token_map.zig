//! The Engram token map: built from the release's tokenizer and config at pull or load, engram.py's algorithm in Zig.
const std = @import("std");
const tok = @import("tokenizer");
const unicode = tok.unicode;
const nfkd_data = tok.nfkd_data;

const magic = "DSENGR01";
/// The release this algorithm and its recorded PCG64 draws come from (DeepSeek AI, MIT).
const source = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/engram.py";

/// Build the map from the release's tokenizer and config, beside the checkpoint where Model.open reads it.
pub fn build(a: std.mem.Allocator, io: std.Io, tokenizer_path: []const u8, config_path: []const u8, output_path: []const u8) !void {
    const tokenizer_bytes = try std.Io.Dir.cwd().readFileAlloc(io, tokenizer_path, a, .limited(64 << 20));
    const config_bytes = try std.Io.Dir.cwd().readFileAlloc(io, config_path, a, .limited(16 << 20));
    var t = try tok.loadTokenizer(io, a, tokenizer_path);
    defer t.deinit();
    var outer = try std.json.parseFromSliceLeaky(std.json.Value, a, config_bytes, .{});
    const config = if (outer == .object and outer.object.get("text_config") != null) outer.object.get("text_config").? else outer;

    // The release's compressed_map: decode each id (raw token on U+FFFD), normalize, dedupe first-seen.
    var key_arena = std.heap.ArenaAllocator.init(a);
    defer key_arena.deinit();
    const ka = key_arena.allocator();
    var keys: std.StringArrayHashMapUnmanaged(u32) = .empty;
    defer keys.deinit(a);
    const raw_vocab: usize = @intCast(try intField(config, "vocab_size"));
    const compressed_expected: usize = @intCast(try intField(config, "engram_compressed_vocab_size"));
    const lookup = try a.alloc(u32, raw_vocab);
    defer a.free(lookup);
    for (0..raw_vocab) |id| {
        const text = try t.decode(a, &.{@intCast(id)}, false);
        defer a.free(text);
        const key = if (std.mem.indexOfScalar(u8, text, 0xEF) != null and std.mem.indexOf(u8, text, "\u{fffd}") != null)
            try ka.dupe(u8, t.id_to_token.get(@intCast(id)) orelse return error.MissingTokenizerEntry)
        else
            try normalizeKey(ka, text);
        const gop = try keys.getOrPut(a, key);
        if (!gop.found_existing) {
            gop.value_ptr.* = @intCast(keys.count() - 1);
            gop.key_ptr.* = key;
        }
        lookup[id] = gop.value_ptr.*;
    }
    if (keys.count() != compressed_expected) {
        std.log.err("tokenizer mismatch: compressed keys {d} != {d}", .{ keys.count(), compressed_expected });
        return error.TokenizerMismatch;
    }

    const order: u64 = @intCast(try intField(config, "engram_max_ngram_size"));
    const heads: u64 = @intCast(try intField(config, "engram_n_heads"));
    if (order != 4 or heads != 8) return error.UnsupportedEngramShape;
    const compressed = keys.count();
    const bound: u64 = @max(1, ((@as(u64, 1) << 63) - 1) / compressed / 2);

    // Per layer: 24 prime buckets never repeated across layers, table offsets, PCG64 multipliers seeded 10007 * layer.
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer seen.deinit(a);
    const layer_ids_v = try arrayField(config, "engram_layer_ids");
    const rows_v = try arrayField(config, "engram_num_embeddings");
    if (layer_ids_v.items.len != rows_v.items.len) return error.ConfigMismatch;
    const layer_count = layer_ids_v.items.len;
    const Layer = struct { id: u64, rows: u64, primes: [24]u32, offsets: [24]u32, multipliers: [4]u64 };
    var layers = try a.alloc(Layer, layer_count);
    defer a.free(layers);
    const engram_vocab_size: u64 = @intCast(try intField(config, "engram_vocab_size"));
    for (layer_ids_v.items, rows_v.items, 0..) |lid_v, rows_v_i, li| {
        const layer_id: u64 = @intCast(lid_v.integer);
        const rows: u64 = @intCast(rows_v_i.integer);
        var primes: [24]u32 = undefined;
        var offsets: [24]u32 = undefined;
        var n: usize = 0;
        var current: u64 = engram_vocab_size - 1;
        while (n < (order - 1) * heads) : (n += 1) {
            current += 1;
            while (true) {
                if (isPrime(current)) {
                    const gop = try seen.getOrPut(a, current);
                    if (!gop.found_existing) break;
                }
                current += 1;
            }
            var sum: u64 = 0;
            for (primes[0..n]) |p| sum += p;
            offsets[n] = @intCast(sum);
            primes[n] = @intCast(current);
        }
        var sum: u64 = 0;
        for (primes) |p| sum += p;
        if (sum != rows) {
            std.log.err("layer {d}: prime buckets sum {d} != table rows {d}", .{ layer_id, sum, rows });
            return error.ConfigMismatch;
        }
        var rng = Pcg64.init(10007 * layer_id);
        var multipliers: [4]u64 = undefined;
        for (&multipliers) |*m| m.* = rng.boundedBelow(bound) * 2 + 1;
        layers[li] = .{ .id = layer_id, .rows = rows, .primes = primes, .offsets = offsets, .multipliers = multipliers };
    }

    const pad_id: usize = @intCast(if (config.object.get("engram_pad_id")) |v| try asInt(v) else try intField(config, "engram_pad_token_id"));
    const pad = lookup[pad_id];

    // LE layout: magic, raw/compressed/pad/order/heads/layers, provenance hashes, per-layer blocks, the lookup array.
    var packed_buf: std.ArrayList(u8) = .empty;
    defer packed_buf.deinit(a);
    try packed_buf.appendSlice(a, magic);
    try appendU32(&packed_buf, a, raw_vocab);
    try appendU32(&packed_buf, a, compressed);
    try appendU32(&packed_buf, a, pad);
    try appendU32(&packed_buf, a, order);
    try appendU32(&packed_buf, a, heads);
    try appendU32(&packed_buf, a, layer_count);
    var tokenizer_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(tokenizer_bytes, &tokenizer_sha, .{});
    var config_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(config_bytes, &config_sha, .{});
    try packed_buf.appendSlice(a, &tokenizer_sha);
    try packed_buf.appendSlice(a, &config_sha);
    for (layers) |layer| {
        try appendU32(&packed_buf, a, layer.id);
        try appendU32(&packed_buf, a, layer.rows);
        for (layer.multipliers) |m| try appendU64(&packed_buf, a, m);
        for (layer.primes) |p| try appendU32(&packed_buf, a, p);
        for (layer.offsets) |o| try appendU32(&packed_buf, a, o);
    }
    for (lookup) |v| try appendU32(&packed_buf, a, v);

    var out_file = try std.Io.Dir.cwd().createFile(io, output_path, .{});
    defer out_file.close(io);
    try out_file.writeStreamingAll(io, packed_buf.items);

    // Manifest beside the map, matching the Python tool's fields.
    var packed_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(packed_buf.items, &packed_sha, .{});
    var m: std.ArrayList(u8) = .empty;
    defer m.deinit(a);
    const W = struct {
        fn f(list: *std.ArrayList(u8), al: std.mem.Allocator, comptime fmt: []const u8, args_: anytype) !void {
            try list.appendSlice(al, try std.fmt.allocPrint(al, fmt, args_));
        }
    }.f;
    const tokenizer_hex = try hexLower(a, &tokenizer_sha);
    const config_hex = try hexLower(a, &config_sha);
    const packed_hex = try hexLower(a, &packed_sha);
    try W(&m, a, "{{\n  \"source\": \"{s}\",\n  \"tokenizer_sha256\": \"{s}\",\n  \"config_sha256\": \"{s}\",\n  \"packed_sha256\": \"{s}\",\n  \"raw_vocab\": {d},\n  \"compressed_vocab\": {d},\n  \"pad\": {d},\n  \"order\": {d},\n  \"heads\": {d},\n  \"layers\": [", .{ source, tokenizer_hex, config_hex, packed_hex, raw_vocab, compressed, pad, order, heads });
    for (layers, 0..) |layer, li| {
        try W(&m, a, "\n    {{\n      \"layer_id\": {d},\n      \"rows\": {d},\n      \"primes\": [{d}, ", .{ layer.id, layer.rows, layer.primes[0] });
        for (layer.primes[1..]) |p| try W(&m, a, "{d}, ", .{p});
        try W(&m, a, "],\n      \"offsets\": [{d}, ", .{layer.offsets[0]});
        for (layer.offsets[1..]) |o| try W(&m, a, "{d}, ", .{o});
        try W(&m, a, "],\n      \"multipliers\": [{d}, ", .{layer.multipliers[0]});
        for (layer.multipliers[1..]) |x| try W(&m, a, "{d}, ", .{x});
        try W(&m, a, "]\n    }}{s}", .{if (li + 1 < layers.len) "," else ""});
    }
    try W(&m, a, "\n  ]\n}}\n", .{});
    const manifest_path = try std.fmt.allocPrint(a, "{s}.json", .{output_path});
    defer a.free(manifest_path);
    var m_file = try std.Io.Dir.cwd().createFile(io, manifest_path, .{});
    defer m_file.close(io);
    try m_file.writeStreamingAll(io, m.items);

    std.debug.print("token map: {d} raw ids -> {d} keys, {d} layers, {d} bytes written to {s}\n", .{ raw_vocab, compressed, layer_count, packed_buf.items.len, output_path });
}

fn appendU32(list: *std.ArrayList(u8), a: std.mem.Allocator, v: u64) !void {
    const tail = try list.addManyAsSlice(a, 4);
    std.mem.writeInt(u32, tail[0..4], @intCast(v), .little);
}

fn appendU64(list: *std.ArrayList(u8), a: std.mem.Allocator, v: u64) !void {
    const tail = try list.addManyAsSlice(a, 8);
    std.mem.writeInt(u64, tail[0..8], v, .little);
}

fn hexLower(a: std.mem.Allocator, bytes: []const u8) ![]u8 {
    const chars = "0123456789abcdef";
    const out = try a.alloc(u8, bytes.len * 2);
    for (bytes, 0..) |b, i| {
        out[i * 2] = chars[b >> 4];
        out[i * 2 + 1] = chars[b & 15];
    }
    return out;
}

fn asInt(v: std.json.Value) !i64 {
    return switch (v) {
        .integer => |n| n,
        .number_string, .string => |s| std.fmt.parseInt(i64, s, 10),
        else => error.ConfigMismatch,
    };
}

fn intField(config: std.json.Value, name: []const u8) !i64 {
    const v = config.object.get(name) orelse return error.ConfigMismatch;
    return asInt(v);
}

fn arrayField(config: std.json.Value, name: []const u8) !std.json.Array {
    const v = config.object.get(name) orelse return error.ConfigMismatch;
    if (v != .array) return error.ConfigMismatch;
    return v.array;
}

fn isPrime(n: u64) bool {
    if (n < 2 or n % 2 == 0) return n == 2;
    var d: u64 = 3;
    while (d * d <= n) : (d += 2) {
        if (n % d == 0) return false;
    }
    return true;
}

// The release's chain NFKC, NFD, StripAccents, Lowercase, replaces and strip composes to NFKD, mark strip, lowercase.

const sentinel: u21 = 0xE000;

fn normalizeKey(a: std.mem.Allocator, text: []const u8) ![]u8 {
    var cps: std.ArrayList(u21) = .empty;
    defer cps.deinit(a);
    var i: usize = 0;
    while (i < text.len) {
        const d = unicode.decodeAt(text, i);
        try decomposeFull(d.cp, &cps, a);
        i += d.len;
    }
    orderCanonical(cps.items);
    var kept: std.ArrayList(u21) = .empty;
    defer kept.deinit(a);
    for (cps.items) |cp| {
        if (!isMark(cp)) try kept.append(a, cp);
    }
    var low: std.ArrayList(u21) = .empty;
    defer low.deinit(a);
    for (kept.items) |cp| {
        if (lowerOf(cp)) |seq| try low.appendSlice(a, seq) else try low.append(a, cp);
    }
    // Replace([ \t\r\n]+, " "): collapse runs of the four regex chars.
    var col: std.ArrayList(u21) = .empty;
    defer col.deinit(a);
    var k: usize = 0;
    while (k < low.items.len) : (k += 1) {
        if (isRegexSpace(low.items[k])) {
            try col.append(a, ' ');
            while (k + 1 < low.items.len and isRegexSpace(low.items[k + 1])) k += 1;
        } else try col.append(a, low.items[k]);
    }
    // Replace(^ $, U+E000): the whole string being a single space.
    if (col.items.len == 1 and col.items[0] == ' ') col.items[0] = sentinel;
    // Strip: trim White_Space at both ends (Rust char::is_whitespace's set).
    var lo: usize = 0;
    var hi: usize = col.items.len;
    while (lo < hi and unicode.isSpace(col.items[lo])) lo += 1;
    while (hi > lo and unicode.isSpace(col.items[hi - 1])) hi -= 1;
    // Replace(U+E000, " ").
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var wrote = false;
    for (col.items[lo..hi]) |cp| {
        try unicode.appendCodepoint(&out, a, if (cp == sentinel) ' ' else cp);
        wrote = true;
    }
    if (!wrote) return a.dupe(u8, text);
    return out.toOwnedSlice(a);
}

fn isRegexSpace(cp: u21) bool {
    return cp == 0x20 or cp == 0x09 or cp == 0x0D or cp == 0x0A;
}

fn decomposeFull(cp: u21, out: *std.ArrayList(u21), a: std.mem.Allocator) !void {
    if (decompOf(cp)) |seq| {
        for (seq) |c| try out.append(a, c);
    } else try out.append(a, cp);
}

fn decompOf(cp: u21) ?[]const u21 {
    var lo: usize = 0;
    var hi: usize = nfkd_data.decompositions.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const e = nfkd_data.decompositions[mid];
        if (e.cp == cp) return nfkd_data.sequences[e.start..][0..e.len];
        if (e.cp < cp) lo = mid + 1 else hi = mid;
    }
    return null;
}

fn orderCanonical(items: []u21) void {
    var start: usize = 0;
    while (start < items.len) : (start += 1) {
        if (combiningClass(items[start]) == 0) continue;
        var end = start;
        while (end < items.len and combiningClass(items[end]) != 0) end += 1;
        std.sort.insertion(u21, items[start..end], {}, struct {
            fn lt(_: void, x: u21, y: u21) bool {
                return combiningClass(x) < combiningClass(y);
            }
        }.lt);
        start = end - 1;
    }
}

fn combiningClass(cp: u21) u8 {
    if (cp < 0x300) return 0;
    var lo: usize = 0;
    var hi: usize = nfkd_data.cccs.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const run = nfkd_data.cccs[mid];
        if (cp < run[0]) hi = mid else if (cp > run[1]) lo = mid + 1 else return @intCast(run[2]);
    }
    return 0;
}

fn isMark(cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = nfkd_data.marks.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const v = nfkd_data.marks[mid];
        if (v == cp) return true;
        if (v < cp) lo = mid + 1 else hi = mid;
    }
    return false;
}

fn lowerOf(cp: u21) ?[]const u21 {
    var lo: usize = 0;
    var hi: usize = nfkd_data.lowercases.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const e = nfkd_data.lowercases[mid];
        if (e.cp == cp) return nfkd_data.lower_sequences[e.start..][0..e.len];
        if (e.cp < cp) lo = mid + 1 else hi = mid;
    }
    return null;
}

// numpy's Generator.integers draw: SeedSequence, PCG64 and Lemire's bounded integers.

const Pcg64 = struct {
    state: u128,
    inc: u128,

    const multiplier: u128 = 0x2360ED051FC65DA44385DF649FCCF645;

    /// pcg64_srandom_r: state = 0, one step, state += initstate, one step; inc = (initseq << 1) | 1 mod 2^128.
    fn init(seed: u64) Pcg64 {
        const w = seedSequenceWords(seed);
        const initstate: u128 = (@as(u128, w[0]) << 64) | @as(u128, w[1]);
        const initseq: u128 = (@as(u128, w[2]) << 64) | @as(u128, w[3]);
        const inc: u128 = (initseq << 1) | 1;
        var state: u128 = 0;
        state = state *% multiplier +% inc;
        state +%= initstate;
        state = state *% multiplier +% inc;
        return .{ .state = state, .inc = inc };
    }

    fn next64(self: *Pcg64) u64 {
        self.state = self.state *% multiplier +% self.inc;
        const hi: u64 = @truncate(self.state >> 64);
        const lo: u64 = @truncate(self.state);
        const rot: u6 = @intCast(hi >> 58);
        return std.math.rotr(u64, lo ^ hi, rot);
    }

    /// Lemire's bounded generator, numpy's masked=false path for Generator.integers.
    fn boundedBelow(self: *Pcg64, bound: u64) u64 {
        const threshold = (0 -% bound) % bound;
        while (true) {
            const x = self.next64();
            const m = @as(u128, x) * @as(u128, bound);
            if (@as(u64, @truncate(m)) >= threshold) return @intCast(m >> 64);
        }
    }
};

// SeedSequence(seed).generate_state(4, uint64): hashmix a 4-word pool, then draw eight words paired little-endian.
fn seedSequenceWords(seed: u64) [4]u64 {
    const entropy = [1]u32{@truncate(seed)};
    var mixer = [4]u32{ 0, 0, 0, 0 };
    var hc: u32 = 0x43b0d7e5;
    for (&mixer, 0..) |*slot, i| {
        slot.* = hashmix(if (i < entropy.len) entropy[i] else 0, &hc);
    }
    for (0..mixer.len) |i_src| {
        for (0..mixer.len) |i_dst| {
            if (i_src != i_dst) mixer[i_dst] = mixWords(mixer[i_dst], hashmix(mixer[i_src], &hc));
        }
    }
    var out: [8]u32 = undefined;
    var hcb: u32 = 0x8b51f9dd;
    for (&out, 0..) |*slot, i| {
        var dv = mixer[i % mixer.len];
        dv ^= hcb;
        hcb = hcb *% 0x58f38ded;
        dv = dv *% hcb;
        dv ^= dv >> 16;
        slot.* = dv;
    }
    var w: [4]u64 = undefined;
    for (0..4) |i| w[i] = @as(u64, out[2 * i]) | (@as(u64, out[2 * i + 1]) << 32);
    return w;
}

fn hashmix(value: u32, hc: *u32) u32 {
    var v = value ^ hc.*;
    hc.* = hc.* *% 0x931e8875;
    v = v *% hc.*;
    v ^= v >> 16;
    return v;
}

fn mixWords(x: u32, y: u32) u32 {
    var result = 0xca01f9dd *% x -% 0x4973f715 *% y;
    result ^= result >> 16;
    return result;
}

test "pcg64 draws match the release's recorded multipliers" {
    // From exl3-2.9bpw/engram-token-map.bin, multipliers (m-1)/2 per layer.
    var rng1 = Pcg64.init(10007);
    try std.testing.expectEqual(@as(u64, 38316048023122), rng1.boundedBelow(46539438283891));
    try std.testing.expectEqual(@as(u64, 2419938046656), rng1.boundedBelow(46539438283891));
    try std.testing.expectEqual(@as(u64, 17979836159674), rng1.boundedBelow(46539438283891));
    try std.testing.expectEqual(@as(u64, 36993668729195), rng1.boundedBelow(46539438283891));
    var rng14 = Pcg64.init(140098);
    try std.testing.expectEqual(@as(u64, 33858405369630), rng14.boundedBelow(46539438283891));
    try std.testing.expectEqual(@as(u64, 25755403400457), rng14.boundedBelow(46539438283891));
    try std.testing.expectEqual(@as(u64, 15460673601360), rng14.boundedBelow(46539438283891));
    try std.testing.expectEqual(@as(u64, 41309613242795), rng14.boundedBelow(46539438283891));
}

test "the normalizer chain matches the release's recorded behavior" {
    const a = std.testing.allocator;
    const cases = [_]struct { in: []const u8, want: []const u8 }{
        .{ .in = "é", .want = "e" },
        .{ .in = "ﬁ", .want = "fi" },
        .{ .in = "İ", .want = "i" },
        // NFKD decomposes Hangul syllables to conjoining jamo; the release's chain never recomposes.
        .{ .in = "가", .want = "\u{1100}\u{1161}" },
        .{ .in = "Äffin", .want = "affin" },
        .{ .in = "ǅ", .want = "dz" },
        .{ .in = "ẞ", .want = "ß" },
        .{ .in = " ", .want = " " },
        .{ .in = " a ", .want = "a" },
        .{ .in = "a\t\tb", .want = "a b" },
        .{ .in = "ﬄ", .want = "ffl" },
        .{ .in = "Ⅷ", .want = "viii" },
        .{ .in = "½", .want = "1⁄2" },
        .{ .in = "㍿", .want = "株式会社" },
        .{ .in = "Ω", .want = "ω" },
        .{ .in = "K", .want = "k" },
        .{ .in = "a\u{0301}\u{0328}", .want = "a" },
        .{ .in = "x\u{200b}\u{00ad}y", .want = "x\u{200b}\u{00ad}y" },
        .{ .in = "😀", .want = "😀" },
    };
    for (cases) |c| {
        const got = try normalizeKey(a, c.in);
        defer a.free(got);
        try std.testing.expectEqualStrings(c.want, got);
    }
}
