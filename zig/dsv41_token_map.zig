//! Builds the canonical Engram token map from the release tokenizer and typed configuration.

// Algorithm: DeepSeek AI inference/engram.py, MIT; the pinned revision is retained in source below.
const std = @import("std");
const tok = @import("src/core/tokenizer/tokenizer.zig");
const unicode = @import("src/core/tokenizer/unicode.zig");
const nfkd_data = @import("src/core/tokenizer/nfkd_data.zig");

const Layer = struct { layer_id: u64, rows: u64, primes: [24]u32, offsets: [24]u32, multipliers: [4]u64 };
const magic = "DSENGR01";
const source = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/engram.py";

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const paths = try parseArgs(args);
    const tokenizer_path: ?[]const u8 = paths.tokenizer;
    const config_path: ?[]const u8 = paths.config;
    const output_path: ?[]const u8 = paths.output;
    const tokenizer_bytes = try std.Io.Dir.cwd().readFileAlloc(io, tokenizer_path.?, a, .limited(64 << 20));
    defer a.free(tokenizer_bytes);
    const config_bytes = try std.Io.Dir.cwd().readFileAlloc(io, config_path.?, a, .limited(16 << 20));
    defer a.free(config_bytes);
    var t = try tok.loadTokenizer(io, a, tokenizer_path.?);
    defer t.deinit();

    const config = try @import("src/families/deepseek_v41/config.zig").Config.parse(a, config_bytes);

    // Decode, fall back to the raw token on U+FFFD, normalize, then deduplicate keys in first-seen order.
    var key_arena = std.heap.ArenaAllocator.init(a);
    defer key_arena.deinit();
    const ka = key_arena.allocator();
    var keys: std.StringArrayHashMapUnmanaged(u32) = .empty;
    defer keys.deinit(a);
    const raw_vocab: usize = config.vocab;
    const compressed_expected: usize = config.engram_compressed_vocab;
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

    const order: u64 = config.engram_max_ngram;
    const heads: u64 = config.engram_heads;
    if (order != 4 or heads != 8) return error.UnsupportedEngramShape;
    const compressed = keys.count();
    const bound: u64 = @max(1, ((@as(u64, 1) << 63) - 1) / compressed / 2);

    // Each layer owns 24 distinct prime buckets; PCG64 multipliers use seed 10007 * layer.
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer seen.deinit(a);
    const layer_count = config.engram_layers.len;
    const layers = try a.alloc(Layer, layer_count);
    defer a.free(layers);
    const engram_vocab_size = config.engram_vocab;
    for (config.engram_layers.items(), config.engram_rows.items(), 0..) |layer_id, rows, li| {
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
        layers[li] = .{ .layer_id = layer_id, .rows = rows, .primes = primes, .offsets = offsets, .multipliers = multipliers };
    }

    const pad_id: usize = config.engram_pad;
    const pad = lookup[pad_id];

    // LE layout: magic, geometry, provenance hashes, layer blocks, then the token lookup array.
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
        try appendU32(&packed_buf, a, layer.layer_id);
        try appendU32(&packed_buf, a, layer.rows);
        for (layer.multipliers) |m| try appendU64(&packed_buf, a, m);
        for (layer.primes) |p| try appendU32(&packed_buf, a, p);
        for (layer.offsets) |o| try appendU32(&packed_buf, a, o);
    }
    for (lookup) |v| try appendU32(&packed_buf, a, v);

    var out_file = try std.Io.Dir.cwd().createFile(io, output_path.?, .{});
    defer out_file.close(io);
    try out_file.writeStreamingAll(io, packed_buf.items);

    // Manifest beside the map, matching the Python tool's fields.
    var packed_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(packed_buf.items, &packed_sha, .{});
    const manifest = try writeManifest(a, tokenizer_sha, config_sha, packed_sha, raw_vocab, compressed, pad, order, heads, layers);
    defer a.free(manifest);
    const manifest_path = try std.fmt.allocPrint(a, "{s}.json", .{output_path.?});
    defer a.free(manifest_path);
    var m_file = try std.Io.Dir.cwd().createFile(io, manifest_path, .{});
    defer m_file.close(io);
    try m_file.writeStreamingAll(io, manifest);

    std.debug.print("token map: {d} raw ids -> {d} keys, {d} layers, {d} bytes written to {s}\n", .{ raw_vocab, compressed, layer_count, packed_buf.items.len, output_path.? });
}

fn writeManifest(a: std.mem.Allocator, tokenizer_sha: [32]u8, config_sha: [32]u8, packed_sha: [32]u8, raw_vocab: usize, compressed: usize, pad: usize, order: u64, heads: u64, layers: []const Layer) ![]u8 {
    return std.json.Stringify.valueAlloc(a, .{
        .source = source,
        .tokenizer_sha256 = std.fmt.bytesToHex(tokenizer_sha, .lower),
        .config_sha256 = std.fmt.bytesToHex(config_sha, .lower),
        .packed_sha256 = std.fmt.bytesToHex(packed_sha, .lower),
        .raw_vocab = raw_vocab,
        .compressed_vocab = compressed,
        .pad = pad,
        .order = order,
        .heads = heads,
        .layers = layers,
    }, .{});
}
const Paths = struct { tokenizer: []const u8, config: []const u8, output: []const u8 };
fn parseArgs(args: []const []const u8) !Paths {
    if (args.len != 7) return error.InvalidArguments;
    var paths: [3]?[]const u8 = @splat(null);
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        const index: usize = if (std.mem.eql(u8, args[i], "--tokenizer")) 0 else if (std.mem.eql(u8, args[i], "--config")) 1 else if (std.mem.eql(u8, args[i], "--output")) 2 else return error.InvalidArguments;
        if (paths[index] != null or args[i + 1].len == 0) return error.InvalidArguments;
        paths[index] = args[i + 1];
    }
    return .{ .tokenizer = paths[0] orelse return error.InvalidArguments, .config = paths[1] orelse return error.InvalidArguments, .output = paths[2] orelse return error.InvalidArguments };
}

fn appendU32(list: *std.ArrayList(u8), a: std.mem.Allocator, v: u64) !void {
    const tail = try list.addManyAsSlice(a, 4);
    std.mem.writeInt(u32, tail[0..4], @intCast(v), .little);
}

fn appendU64(list: *std.ArrayList(u8), a: std.mem.Allocator, v: u64) !void {
    const tail = try list.addManyAsSlice(a, 8);
    std.mem.writeInt(u64, tail[0..8], v, .little);
}

fn isPrime(n: u64) bool {
    if (n < 2 or n % 2 == 0) return n == 2;
    var d: u64 = 3;
    while (d * d <= n) : (d += 2) {
        if (n % d == 0) return false;
    }
    return true;
}

// Rust normalization composes to NFKD, mark stripping, lowercase, whitespace replacement and trimming.

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

// numpy Generator.integers: SeedSequence, PCG64, then Lemire bounded integers.

const Pcg64 = struct {
    state: u128,
    inc: u128,

    const multiplier: u128 = 0x2360ED051FC65DA44385DF649FCCF645;

    /// PCG seeding: step from zero, add initstate, step again; inc = (initseq << 1) | 1 modulo 2^128.
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

/// SeedSequence hashes into four words, draws eight words with a second hash, then pairs them little endian.
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
        .{ .in = "가", .want = "가" },
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

test "export manifest parses strictly with complete arrays and digests" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const layer: Layer = .{ .layer_id = 1, .rows = 24, .primes = @splat(3), .offsets = @splat(7), .multipliers = .{ 11, 13, 17, 19 } };
    const bytes = try writeManifest(a, @splat(1), @splat(2), @splat(3), 129280, 99092, 2, 4, 8, &.{layer});
    const parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer parsed.deinit();
    const top = parsed.value.object;
    const got = top.get("layers").?.array.items[0].object;
    try std.testing.expectEqual(@as(usize, 24), got.get("primes").?.array.items.len);
    try std.testing.expectEqual(@as(i64, 7), got.get("offsets").?.array.items[23].integer);
    try std.testing.expectEqual(@as(i64, 19), got.get("multipliers").?.array.items[3].integer);
    try std.testing.expectEqualStrings(&std.fmt.bytesToHex(@as([32]u8, @splat(1)), .lower), top.get("tokenizer_sha256").?.string);
    try std.testing.expectEqualStrings(&std.fmt.bytesToHex(@as([32]u8, @splat(2)), .lower), top.get("config_sha256").?.string);
    try std.testing.expectEqualStrings(&std.fmt.bytesToHex(@as([32]u8, @splat(3)), .lower), top.get("packed_sha256").?.string);
}
test "export CLI rejects duplicate and missing options before opening files" {
    try std.testing.expectError(error.InvalidArguments, parseArgs(&.{ "tool", "--tokenizer", "A", "--tokenizer", "B", "--output", "C" }));
    try std.testing.expectError(error.InvalidArguments, parseArgs(&.{ "tool", "--tokenizer", "A", "--config", "B", "--config", "C", "--output", "D" }));
    try std.testing.expectError(error.InvalidArguments, parseArgs(&.{ "tool", "--tokenizer", "A" }));
    const paths = try parseArgs(&.{ "tool", "--output", "C", "--config", "B", "--tokenizer", "A" });
    try std.testing.expectEqualStrings("B", paths.config);
}
test "token exporter keeps owned lists and exact integral Config values" {
    const a = std.testing.allocator;
    const fixture = @embedFile("src/families/deepseek_v41/fixtures/config.json");
    const exponent = try std.mem.replaceOwned(u8, a, fixture, "129280", "1.2928e5");
    defer a.free(exponent);
    const fractional = try std.mem.replaceOwned(u8, a, exponent, "\n            14", "\n            14.0");
    defer a.free(fractional);
    const Config = @import("src/families/deepseek_v41/config.zig").Config;
    const want = try Config.parse(a, fixture);
    const got = try Config.parse(a, fractional);
    try std.testing.expectEqual(want.vocab, got.vocab);
    try std.testing.expectEqual(want.engram_layers, got.engram_layers);
    try std.testing.expectEqual(want.engram_rows, got.engram_rows);
    try std.testing.expectEqual(want.engram_pad, got.engram_pad);
}
