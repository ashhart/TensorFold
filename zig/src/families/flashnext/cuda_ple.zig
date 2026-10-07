//! Layer 2's n-gram table for one decode token: hash the row ids, gather those rows, dequant like glue._ple_embed.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const embed = @import("cuda_embed.zig");
const hc = @import("cuda_hc.zig");

const golden: u64 = 0x9E3779B97F4A7C15;
const mix1: u64 = 0xBF58476D1CE4E5B9;
const mix2: u64 = 0x94D049BB133111EB;
const prime_step: u64 = 10007;
const max_n: usize = 8;
const max_heads: usize = 64;
const max_shards: usize = 128;
const layer: usize = 1;

const Tensor = core.safetensors.Tensor;

const Hash = struct {
    n: usize,
    per: usize,
    heads: usize,
    eos: i64,
    dh: usize,
    shards: usize,
    sizes: [max_heads]i64 = undefined,
    offsets: [max_heads]i64 = undefined,
    mult: [max_n]i64 = undefined,
    starts: [max_shards + 1]i64 = undefined,

    fn init(args: struct {
        vocab: u64,
        ngram: usize,
        per: usize,
        base: u64,
        divisor: u64,
        shards: usize,
        seed: u64,
        eos: i64,
        ple_dim: usize,
        ple_index: usize,
    }) !Hash {
        if (args.ngram < 2 or args.ngram > max_n or args.per == 0 or args.shards == 0 or args.shards > max_shards) return error.UnexpectedTensor;
        if (args.base < 2 or args.divisor == 0) return error.UnexpectedTensor;
        const heads = (args.ngram - 1) * args.per;
        if (heads > max_heads or args.ple_dim == 0 or args.ple_dim % heads != 0) return error.UnexpectedTensor;
        var h = Hash{ .n = args.ngram, .per = args.per, .heads = heads, .eos = args.eos, .dh = args.ple_dim / heads, .shards = args.shards };
        var total: i64 = 0;
        for (0..heads) |head| {
            const size: i64 = @intCast(nthPrimeAfter(args.base - 1, args.ple_index * heads + head + 1));
            h.sizes[head] = size;
            h.offsets[head] = total;
            total += size;
        }
        const divisor: i64 = @intCast(args.divisor);
        const shard_n: i64 = @intCast(args.shards);
        const padded = divCeil(total, divisor) * divisor;
        const base: i64 = @divFloor(padded, shard_n);
        const extra: usize = @intCast(@mod(padded, shard_n));
        var acc: i64 = 0;
        for (0..args.shards) |i| {
            h.starts[i] = acc;
            acc += base + @as(i64, @intFromBool(i < extra));
        }
        h.starts[args.shards] = acc;
        const half: u64 = @max(1, (((1 << 63) - 1) / @max(args.vocab, 1)) / 2);
        const seed = args.seed +% prime_step *% @as(u64, @intCast(args.ple_index));
        for (0..args.ngram) |i| {
            const mixed = splitmix64(seed +% golden *% @as(u64, @intCast(i + 1)));
            h.mult[i] = @intCast(2 * (mixed % half) + 1);
        }
        return h;
    }

    fn locate(h: Hash, row: i64) !struct { shard: usize, local: usize } {
        if (row < 0 or row >= h.starts[h.shards]) return error.UnexpectedTensor;
        var lo: usize = 0;
        var hi = h.shards;
        while (lo + 1 < hi) {
            const mid = (lo + hi) / 2;
            if (h.starts[mid] <= row) lo = mid else hi = mid;
        }
        return .{ .shard = lo, .local = @intCast(row - h.starts[lo]) };
    }
};

fn splitmix64(value: u64) u64 {
    var v = value +% golden;
    v = (v ^ (v >> 30)) *% mix1;
    v = (v ^ (v >> 27)) *% mix2;
    return v ^ (v >> 31);
}

fn isPrime(value: u64) bool {
    if (value < 2) return false;
    if (value % 2 == 0) return value == 2;
    var d: u64 = 3;
    while (d <= value / d) : (d += 2) if (value % d == 0) return false;
    return true;
}

fn nthPrimeAfter(start: u64, count: usize) u64 {
    var prime = start;
    for (0..count) |_| {
        prime += 1;
        while (!isPrime(prime)) prime += 1;
    }
    return prime;
}

fn divCeil(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

/// Row ids of one new token after `history` (n - 1 ids). A fresh decode window is n - 1 copies of eos.
fn lastIds(h: Hash, history: []const i64, token: i64, out: []i64) !void {
    if (history.len != h.n - 1 or out.len != h.heads) return error.UnexpectedTensor;
    var seq: [max_n]i64 = undefined;
    for (history, 0..) |v, i| seq[i] = v;
    seq[history.len] = token;
    const width = h.n;
    var eos_at: [max_n]i64 = undefined;
    for (0..width) |i| eos_at[i] = if (seq[i] == h.eos) @intCast(i) else -1;
    var acc: [max_n]i64 = undefined;
    acc[0] = eos_at[0];
    for (1..width) |i| acc[i] = @max(acc[i - 1], eos_at[i]);
    var before: [max_n]i64 = undefined;
    before[0] = -1;
    for (1..width) |i| before[i] = acc[i - 1];
    var shifted: [max_n]i64 = undefined;
    const pos: i64 = @intCast(width - 1);
    const in_segment = pos - (before[width - 1] + 1);
    for (0..h.n) |shift| {
        const source = pos - @as(i64, @intCast(shift));
        const taken = seq[@intCast(@max(source, 0))];
        const keep = in_segment >= @as(i64, @intCast(shift)) and source >= 0;
        shifted[shift] = if (keep) taken else h.eos;
    }
    var at: usize = 0;
    var ngram: usize = 2;
    while (ngram <= h.n) : (ngram += 1) {
        var mixed = shifted[0] *% h.mult[0];
        for (1..ngram) |p| mixed = mixed ^ (shifted[p] *% h.mult[p]);
        for (0..h.per) |_| {
            out[at] = @mod(mixed, h.sizes[at]) + h.offsets[at];
            at += 1;
        }
    }
}

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

fn sumGroups(row: []const u16, out: []f32) void {
    const groups = row.len / 32;
    for (0..groups) |g| {
        var buf: [32]f32 = undefined;
        for (0..32) |i| buf[i] = @bitCast(@as(u32, row[g * 32 + i]) << 16);
        var n: usize = 32;
        while (n > 1) {
            n /= 2;
            for (0..n) |i| buf[i] = buf[2 * i] + buf[2 * i + 1];
        }
        out[g] = buf[0];
    }
}

const Open = struct {
    io: std.Io,
    path: []u8,
    file: std.Io.File,
    map: std.Io.File.MemoryMap,
    parsed: std.json.Parsed(std.json.Value),
    at: usize,

    fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Open {
        const owned = try gpa.dupe(u8, path);
        errdefer gpa.free(owned);
        var file = try std.Io.Dir.cwd().openFile(io, owned, .{});
        errdefer file.close(io);
        const len: usize = @intCast(try file.length(io));
        if (len < 8) return error.BadSafetensors;
        var map = try std.Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        errdefer map.destroy(io);
        const header_len: usize = @intCast(std.mem.readInt(u64, map.memory[0..8], .little));
        if (header_len > len - 8) return error.BadSafetensors;
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, map.memory[8..][0..header_len], .{});
        return .{ .io = io, .path = owned, .file = file, .map = map, .parsed = parsed, .at = 8 + header_len };
    }

    fn close(self: *Open, gpa: std.mem.Allocator) void {
        self.parsed.deinit();
        self.map.destroy(self.io);
        self.file.close(self.io);
        gpa.free(self.path);
        self.* = undefined;
    }

    fn tensor(self: *const Open, name: []const u8, dtype: core.safetensors.DType) !Tensor {
        const o = (self.parsed.value.object.get(name) orelse return error.MissingTensor).object;
        const got = core.safetensors.DType.parse(o.get("dtype").?.string) orelse return error.UnsupportedDType;
        if (got != dtype) return error.UnexpectedTensor;
        const shape = o.get("shape").?.array.items;
        if (shape.len == 0 or shape.len > 4) return error.UnexpectedTensor;
        const offs = o.get("data_offsets").?.array.items;
        const begin: usize = @intCast(offs[0].integer);
        const end: usize = @intCast(offs[1].integer);
        const data = self.map.memory[self.at..];
        if (end < begin or end > data.len) return error.BadSafetensors;
        var t: Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = data[begin..end] };
        for (shape, 0..) |d, i| t.shape[i] = @intCast(d.integer);
        return t;
    }
};

fn readI64(t: Tensor, out: []i64) !void {
    if (t.dtype != .i64 or t.rank != 1 or t.dim(0) != out.len or t.bytes.len != out.len * 8) return error.UnexpectedTensor;
    for (out, 0..) |*o, i| o.* = std.mem.readInt(i64, t.bytes[8 * i ..][0..8], .little);
}

fn weightName(buf: []u8, shard: usize, nested: bool) ![]const u8 {
    return if (nested)
        std.fmt.bufPrint(buf, "language_model.model.layers.{d}.ple.ple_embedding.ngram_embedding.shards.{d}.weight", .{ layer, shard })
    else
        std.fmt.bufPrint(buf, "language_model.model.layers.{d}.ple.ple_embedding.ngram_embedding.shard_{d}.weight", .{ layer, shard });
}

fn partName(buf: []u8, shard: usize, nested: bool, part: []const u8) ![]const u8 {
    return if (nested)
        std.fmt.bufPrint(buf, "language_model.model.layers.{d}.ple.ple_embedding.ngram_embedding.shards.{d}.{s}", .{ layer, shard, part })
    else
        std.fmt.bufPrint(buf, "language_model.model.layers.{d}.ple.ple_embedding.ngram_embedding.shard_{d}.{s}", .{ layer, shard, part });
}

fn findOpen(opened: []Open, path: []const u8) ?*Open {
    for (opened) |*o| if (std.mem.eql(u8, o.path, path)) return o;
    return null;
}

fn copyRow(file: *const Open, shard: usize, nested: bool, local: usize, dh: usize, words: []u8, scales: []u8, biases: []u8) !void {
    var name: [180]u8 = undefined;
    const w = try file.tensor(try partName(&name, shard, nested, "weight"), .u32);
    const s = try file.tensor(try partName(&name, shard, nested, "scales"), .bf16);
    const b = try file.tensor(try partName(&name, shard, nested, "biases"), .bf16);
    const words_w = dh / 8;
    const groups = dh / 32;
    if (w.dim(1) != words_w or s.dim(1) != groups or b.dim(1) != groups or local >= w.dim(0)) return error.UnexpectedTensor;
    const wb = words_w * 4;
    const gb = groups * 2;
    @memcpy(words, w.bytes[local * wb ..][0..wb]);
    @memcpy(scales, s.bytes[local * gb ..][0..gb]);
    @memcpy(biases, b.bytes[local * gb ..][0..gb]);
}

fn tableScale(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, wm: std.json.ObjectMap) !f32 {
    const name = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale";
    const file = wm.get(name) orelse return 1;
    const path = try std.fs.path.join(gpa, &.{ dir, file.string });
    defer gpa.free(path);
    var opened = try Open.load(gpa, io, path);
    defer opened.close(gpa);
    const t = try opened.tensor(name, blk: {
        const o = opened.parsed.value.object.get(name).?.object;
        break :blk core.safetensors.DType.parse(o.get("dtype").?.string) orelse return error.UnsupportedDType;
    });
    if (t.numel() != 1) return error.UnexpectedTensor;
    return switch (t.dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little)),
        .f64 => @floatCast(@as(f64, @bitCast(std.mem.readInt(u64, t.bytes[0..8], .little)))),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[0..2], .little)) << 16),
        else => error.UnexpectedTensor,
    };
}

fn jsonU64(v: std.json.Value) !u64 {
    if (v != .integer or v.integer < 0) return error.UnexpectedTensor;
    return @intCast(v.integer);
}

fn optU64(o: std.json.ObjectMap, name: []const u8, default: u64) !u64 {
    const v = o.get(name) orelse return default;
    if (v == .null) return default;
    return jsonU64(v);
}

const NgramSpec = struct {
    vocab: u64,
    ngram: usize,
    per: usize,
    base: u64,
    divisor: u64,
    shards: usize,
    seed: u64,
    eos: i64,
    ple_dim: usize,
    ple_index: usize,
};

/// `ple_layer_ids` value 2 is decoder layer 1. The eos id is the text config's, which the hash mixes in.
fn readSpec(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !NgramSpec {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const text = if (root.get("text_config")) |v| if (v == .object and v.object.count() > 0) v.object else root else root;
    const layers = try jsonU64(text.get("num_hidden_layers") orelse return error.MissingTensor);
    if (layers <= layer) return error.UnexpectedTensor;
    const hidden = try jsonU64(text.get("hidden_size") orelse return error.MissingTensor);
    const ids = text.get("ple_layer_ids") orelse return error.MissingTensor;
    if (ids != .array) return error.UnexpectedTensor;
    var seen: [512]bool = @splat(false);
    var ple_index: usize = 0;
    var found = false;
    for (ids.array.items) |item| {
        const id = try jsonU64(item);
        if (id == 0 or id > layers or id > seen.len) return error.UnexpectedTensor;
        if (id == layer + 1) found = true;
        if (!seen[id - 1] and id - 1 < layer) ple_index += 1;
        seen[id - 1] = true;
    }
    if (!found) return error.UnexpectedTensor;
    const eos: i64 = if (text.get("eos_token_id")) |v| blk: {
        if (v == .null) break :blk 0;
        if (v == .array) {
            if (v.array.items.len == 0) return error.UnexpectedTensor;
            break :blk @intCast(try jsonU64(v.array.items[0]));
        }
        break :blk @intCast(try jsonU64(v));
    } else 0;
    return .{
        .vocab = try jsonU64(text.get("vocab_size") orelse return error.MissingTensor),
        .ngram = @intCast(try optU64(text, "ngram_size", 3)),
        .per = @intCast(try optU64(text, "heads_per_ngram", 8)),
        .base = try optU64(text, "ngram_vocab_size_base", 20_000_000),
        .divisor = try optU64(text, "make_ngram_vocab_size_divisible_by", 128),
        .shards = @intCast(try optU64(text, "split_ngram_parts", 128)),
        .seed = try optU64(text, "seed", 1234),
        .eos = eos,
        .ple_dim = @intCast(try optU64(text, "ple_embed_dim", hidden)),
        .ple_index = ple_index,
    };
}

/// The text config's eos id, which a fresh n-gram window repeats.
pub fn eosOf(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8) !i64 {
    const cfg = try readSpec(gpa, io, model_dir);
    return cfg.eos;
}

/// Host dequant of one token's sixteen n-gram rows. `history` is the previous n - 1 ids.
pub fn embedding(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, history: []const i64, token: u32, out: []u16, xs: []f32) !void {
    const cfg = try readSpec(gpa, io, model_dir);
    const hash = try Hash.init(.{
        .vocab = cfg.vocab,
        .ngram = cfg.ngram,
        .per = cfg.per,
        .base = cfg.base,
        .divisor = cfg.divisor,
        .shards = cfg.shards,
        .seed = cfg.seed,
        .eos = cfg.eos,
        .ple_dim = cfg.ple_dim,
        .ple_index = cfg.ple_index,
    });
    if (history.len != hash.n - 1 or out.len != hash.heads * hash.dh or xs.len != out.len / 32) return error.UnexpectedTensor;
    var ids: [max_heads]i64 = undefined;
    try lastIds(hash, history, @intCast(token), ids[0..hash.heads]);
    const index_path = try std.fs.path.join(gpa, &.{ model_dir, "model.safetensors.index.json" });
    defer gpa.free(index_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const wm = parsed.value.object.get("weight_map").?.object;
    const scale = try tableScale(gpa, io, model_dir, wm);
    const heads = hash.heads;
    const dh = hash.dh;
    const words = try gpa.alloc(u8, heads * (dh / 8) * 4);
    defer gpa.free(words);
    const scales = try gpa.alloc(u8, heads * (dh / 32) * 2);
    defer gpa.free(scales);
    const biases = try gpa.alloc(u8, scales.len);
    defer gpa.free(biases);
    var opened: [max_heads]Open = undefined;
    var nopen: usize = 0;
    defer for (opened[0..nopen]) |*o| o.close(gpa);
    var key: [180]u8 = undefined;
    for (0..heads) |h| {
        const loc = try hash.locate(ids[h]);
        var nested = false;
        const mapped = if (wm.get(try weightName(&key, loc.shard, false))) |v|
            v.string
        else blk: {
            nested = true;
            break :blk (wm.get(try weightName(&key, loc.shard, true)) orelse return error.MissingTensor).string;
        };
        const path = try std.fs.path.join(gpa, &.{ model_dir, mapped });
        defer gpa.free(path);
        const file = findOpen(opened[0..nopen], path) orelse file: {
            opened[nopen] = try Open.load(gpa, io, path);
            nopen += 1;
            break :file &opened[nopen - 1];
        };
        const wb = (dh / 8) * 4;
        const gb = (dh / 32) * 2;
        try copyRow(file, loc.shard, nested, loc.local, dh, words[h * wb ..][0..wb], scales[h * gb ..][0..gb], biases[h * gb ..][0..gb]);
    }
    for (0..heads) |h| try embed.dequant(words, scales, biases, dh, h, out[h * dh ..][0..dh]);
    if (scale != 1) {
        for (out) |*v| {
            const f: f32 = @bitCast(@as(u32, v.*) << 16);
            v.* = toBf16(f * scale);
        }
    }
    sumGroups(out, xs);
}

/// One decode token. `ple_layer_ids` `[2]` is this layer. Returns 0 when the gathered rows match the host dequant.
pub fn table(comptime Tri: type, gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, token: u32) !u32 {
    const cfg = try readSpec(gpa, io, model_dir);
    const hash = try Hash.init(.{
        .vocab = cfg.vocab,
        .ngram = cfg.ngram,
        .per = cfg.per,
        .base = cfg.base,
        .divisor = cfg.divisor,
        .shards = cfg.shards,
        .seed = cfg.seed,
        .eos = cfg.eos,
        .ple_dim = cfg.ple_dim,
        .ple_index = cfg.ple_index,
    });
    var history: [max_n]i64 = undefined;
    for (0..hash.n - 1) |i| history[i] = hash.eos;
    var ids: [max_heads]i64 = undefined;
    try lastIds(hash, history[0 .. hash.n - 1], token, ids[0..hash.heads]);

    const index_path = try std.fs.path.join(gpa, &.{ model_dir, "model.safetensors.index.json" });
    defer gpa.free(index_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const wm = parsed.value.object.get("weight_map").?.object;

    var cname: [160]u8 = undefined;
    const mult_name = try std.fmt.bufPrint(&cname, "language_model.model.layers.{d}.ple.ple_embedding.layer_multipliers", .{layer});
    const const_file = wm.get(mult_name) orelse return error.MissingTensor;
    const const_path = try std.fs.path.join(gpa, &.{ model_dir, const_file.string });
    defer gpa.free(const_path);
    var constants = try Open.load(gpa, io, const_path);
    defer constants.close(gpa);
    var got_m: [max_n]i64 = undefined;
    var got_o: [max_heads]i64 = undefined;
    var got_s: [max_heads]i64 = undefined;
    try readI64(try constants.tensor(mult_name, .i64), got_m[0..hash.n]);
    const off_name = try std.fmt.bufPrint(&cname, "language_model.model.layers.{d}.ple.ple_embedding.ngram_heads_offsets", .{layer});
    try readI64(try constants.tensor(off_name, .i64), got_o[0..hash.heads]);
    const size_name = try std.fmt.bufPrint(&cname, "language_model.model.layers.{d}.ple.ple_embedding.ngram_heads_vocab_sizes", .{layer});
    try readI64(try constants.tensor(size_name, .i64), got_s[0..hash.heads]);
    if (!std.mem.eql(i64, got_m[0..hash.n], hash.mult[0..hash.n]) or !std.mem.eql(i64, got_o[0..hash.heads], hash.offsets[0..hash.heads]) or !std.mem.eql(i64, got_s[0..hash.heads], hash.sizes[0..hash.heads])) {
        std.debug.print("ngram constants differ\n", .{});
        return 1;
    }
    const scale = try tableScale(gpa, io, model_dir, wm);

    const heads = hash.heads;
    const dh = hash.dh;
    const words = try gpa.alloc(u8, heads * (dh / 8) * 4);
    defer gpa.free(words);
    const scales = try gpa.alloc(u8, heads * (dh / 32) * 2);
    defer gpa.free(scales);
    const biases = try gpa.alloc(u8, scales.len);
    defer gpa.free(biases);
    var opened: [max_heads]Open = undefined;
    var nopen: usize = 0;
    defer for (opened[0..nopen]) |*o| o.close(gpa);
    var key: [180]u8 = undefined;
    for (0..heads) |h| {
        const loc = try hash.locate(ids[h]);
        var nested = false;
        const mapped = if (wm.get(try weightName(&key, loc.shard, false))) |v|
            v.string
        else blk: {
            nested = true;
            break :blk (wm.get(try weightName(&key, loc.shard, true)) orelse return error.MissingTensor).string;
        };
        const path = try std.fs.path.join(gpa, &.{ model_dir, mapped });
        defer gpa.free(path);
        const file = findOpen(opened[0..nopen], path) orelse file: {
            opened[nopen] = try Open.load(gpa, io, path);
            nopen += 1;
            break :file &opened[nopen - 1];
        };
        const wb = (dh / 8) * 4;
        const gb = (dh / 32) * 2;
        try copyRow(file, loc.shard, nested, loc.local, dh, words[h * wb ..][0..wb], scales[h * gb ..][0..gb], biases[h * gb ..][0..gb]);
    }

    const host = try gpa.alloc(u16, heads * dh);
    defer gpa.free(host);
    for (0..heads) |h| try embed.dequant(words, scales, biases, dh, h, host[h * dh ..][0..dh]);
    if (scale != 1) {
        for (host) |*v| {
            const f: f32 = @bitCast(@as(u32, v.*) << 16);
            v.* = toBf16(f * scale);
        }
    }
    const host_xs = try gpa.alloc(f32, heads * dh / 32);
    defer gpa.free(host_xs);
    sumGroups(host, host_xs);

    var w_b = try cuda.DeviceBuffer.fromHost(driver, words);
    defer w_b.free();
    var s_b = try cuda.DeviceBuffer.fromHost(driver, scales);
    defer s_b.free();
    var b_b = try cuda.DeviceBuffer.fromHost(driver, biases);
    defer b_b.free();
    var out_b = try cuda.DeviceBuffer.alloc(driver, host.len * 2);
    defer out_b.free();
    var xs_b = try cuda.DeviceBuffer.alloc(driver, host_xs.len * 4);
    defer xs_b.free();
    try tri.pleEmbed(w_b.ptr, s_b.ptr, b_b.ptr, out_b.ptr, xs_b.ptr, 1, heads, dh, scale);
    const got_bytes = try gpa.alloc(u8, host.len * 2);
    defer gpa.free(got_bytes);
    const xs_bytes = try gpa.alloc(u8, host_xs.len * 4);
    defer gpa.free(xs_bytes);
    try stream.synchronize();
    try out_b.download(0, got_bytes);
    try xs_b.download(0, xs_bytes);
    const gpu = try gpa.alloc(u16, host.len);
    defer gpa.free(gpu);
    for (gpu, 0..) |*o, i| o.* = std.mem.readInt(u16, got_bytes[2 * i ..][0..2], .little);
    var off: usize = 0;
    var steps: u32 = 0;
    var at: usize = 0;
    for (gpu, host, 0..) |a, b, i| {
        const d = hc.mixedSteps(a, b);
        if (d != 0) off += 1;
        if (d > steps) {
            steps = d;
            at = i;
        }
    }
    var xs_ulp: u32 = 0;
    var xs_at: usize = 0;
    for (host_xs, 0..) |want, i| {
        const got: f32 = @bitCast(std.mem.readInt(u32, xs_bytes[4 * i ..][0..4], .little));
        const dist = hc.ulps(got, want);
        if (dist > xs_ulp) {
            xs_ulp = dist;
            xs_at = i;
        }
    }
    std.debug.print("ngram token {d} heads {d} values {d} off {d} max_steps {d} at {d} got {x:0>4} host {x:0>4} xs_ulp {d} xs_at {d} head {x:0>4}\n", .{ token, heads, host.len, off, steps, at, gpu[at], host[at], xs_ulp, xs_at, gpu[0] });
    if (off != 0 or steps != 0 or xs_ulp != 0) return 1;
    return 0;
}

test "a fresh window hashes token 7 into sixteen rows" {
    const h = try Hash.init(.{
        .vocab = 248320,
        .ngram = 3,
        .per = 8,
        .base = 20_000_000,
        .divisor = 128,
        .shards = 128,
        .seed = 1234,
        .eos = 248044,
        .ple_dim = 2560,
        .ple_index = 0,
    });
    try std.testing.expectEqual(@as(i64, 23703573157769), h.mult[0]);
    const history = [_]i64{ 248044, 248044 };
    var ids: [16]i64 = undefined;
    try lastIds(h, &history, 7, &ids);
    const want = [_]i64{ 2927653, 34980843, 54748278, 66612378, 97814964, 109013870, 126560393, 151352333, 167888935, 182235580, 215170017, 237467519, 247510681, 278779700, 296141806, 304994522 };
    try std.testing.expectEqualSlices(i64, &want, &ids);
    const loc = try h.locate(ids[0]);
    try std.testing.expectEqual(@as(usize, 1), loc.shard);
    try std.testing.expectEqual(@as(usize, 427641), loc.local);
}
