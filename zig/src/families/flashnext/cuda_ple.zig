//! Layer 2's n-gram table: the hash of each token's row ids, and the mapped table a pass gathers its rows from.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const embed = @import("cuda_embed.zig");
const hc = @import("cuda_hc.zig");

const golden: u64 = 0x9E3779B97F4A7C15;
const mix1: u64 = 0xBF58476D1CE4E5B9;
const mix2: u64 = 0x94D049BB133111EB;
const prime_step: u64 = 10007;
pub const max_n: usize = 8;
pub const max_heads: usize = 64;
const max_shards: usize = 128;
pub const layer: usize = 1;

const Tensor = core.safetensors.Tensor;

pub const Hash = struct {
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

    pub fn init(args: struct {
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

    pub fn locate(h: Hash, row: i64) !struct { shard: usize, local: usize } {
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
pub fn lastIds(h: Hash, history: []const i64, token: i64, out: []i64) !void {
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

pub fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

pub fn sumGroups(row: []const u16, out: []f32) void {
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

/// One mapped shard (core.safetensors.File) with a typed lookup.
pub const Open = struct {
    file: core.safetensors.File,
    io: std.Io,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Open {
        return .{ .file = try core.safetensors.File.open(gpa, io, path), .io = io };
    }

    pub fn close(self: *Open, gpa: std.mem.Allocator) void {
        _ = gpa;
        self.file.close(self.io);
    }

    pub fn tensor(self: *const Open, name: []const u8, dtype: core.safetensors.DType) !Tensor {
        const t = self.file.get(name) orelse return error.MissingTensor;
        if (t.dtype != dtype) return error.UnexpectedTensor;
        return t;
    }
};

pub fn readI64(t: Tensor, out: []i64) !void {
    if (t.dtype != .i64 or t.rank != 1 or t.dim(0) != out.len or t.bytes.len != out.len * 8) return error.UnexpectedTensor;
    for (out, 0..) |*o, i| o.* = std.mem.readInt(i64, t.bytes[8 * i ..][0..8], .little);
}

pub fn weightName(buf: []u8, shard: usize, nested: bool) ![]const u8 {
    return if (nested)
        std.fmt.bufPrint(buf, "language_model.model.layers.{d}.ple.ple_embedding.ngram_embedding.shards.{d}.weight", .{ layer, shard })
    else
        std.fmt.bufPrint(buf, "language_model.model.layers.{d}.ple.ple_embedding.ngram_embedding.shard_{d}.weight", .{ layer, shard });
}

pub fn partName(buf: []u8, shard: usize, nested: bool, part: []const u8) ![]const u8 {
    return partNameFor(buf, shard, nested, part, false);
}

fn partNameFor(buf: []u8, shard: usize, nested: bool, part: []const u8, modelopt: bool) ![]const u8 {
    const prefix = if (modelopt) "model.language_model" else "language_model.model";
    return if (nested)
        std.fmt.bufPrint(buf, "{s}.layers.{d}.ple.ple_embedding.ngram_embedding.shards.{d}.{s}", .{ prefix, layer, shard, part })
    else
        std.fmt.bufPrint(buf, "{s}.layers.{d}.ple.ple_embedding.ngram_embedding.shard_{d}.{s}", .{ prefix, layer, shard, part });
}

pub fn findOpen(opened: []Open, path: []const u8) ?*Open {
    for (opened) |*o| if (std.mem.eql(u8, o.file.path, path)) return o;
    return null;
}

pub fn copyRow(file: *const Open, shard: usize, nested: bool, local: usize, dh: usize, words: []u8, scales: []u8, biases: []u8) !void {
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

pub fn tableScale(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, wm: std.json.ObjectMap) !f32 {
    return readScale(gpa, io, dir, wm, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale", false);
}

fn readScale(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, wm: std.json.ObjectMap, name: []const u8, required: bool) !f32 {
    const file = wm.get(name) orelse return if (required) error.MissingTensor else 1;
    const path = try std.fs.path.join(gpa, &.{ dir, file.string });
    defer gpa.free(path);
    var opened = try Open.load(gpa, io, path);
    defer opened.close(gpa);
    const t = opened.file.get(name) orelse return error.MissingTensor;
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
pub fn readSpec(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !NgramSpec {
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

/// The n-gram table held open for an engine's life: the hash, the scale, and each shard's slices of its mapped file.
pub const Ngram = struct {
    gpa: std.mem.Allocator,
    hash: Hash,
    scale: f32,
    files: std.ArrayList(Open) = .empty,
    w: [max_shards][]const u8 = undefined,
    s: [max_shards][]const u8 = undefined,
    b: [max_shards][]const u8 = undefined,
    format: enum { mlx, nvfp4 } = .mlx,
    tables: [max_shards]core.nvfp4_rows.Table = undefined,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8) !Ngram {
        const cfg = try readSpec(gpa, io, model_dir);
        var g: Ngram = .{ .gpa = gpa, .hash = try Hash.init(.{
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
        }), .scale = 1 };
        errdefer g.deinit();
        const index_path = try std.fs.path.join(gpa, &.{ model_dir, "model.safetensors.index.json" });
        defer gpa.free(index_path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, gpa, .limited(1 << 26));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        const wm = try core.safetensors.index.weightMap(parsed.value);
        const first = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight";
        const modelopt = wm.contains(first);
        g.format = if (modelopt) .nvfp4 else .mlx;
        g.scale = if (modelopt)
            try readScale(gpa, io, model_dir, wm, "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2", true)
        else
            try tableScale(gpa, io, model_dir, wm);
        const dh = g.hash.dh;
        var key: [180]u8 = undefined;
        for (0..g.hash.shards) |shard| {
            var nested = false;
            const mapped = if (wm.get(try partNameFor(&key, shard, false, "weight", modelopt))) |v|
                v.string
            else blk: {
                nested = true;
                break :blk (wm.get(try partNameFor(&key, shard, true, "weight", modelopt)) orelse return error.MissingTensor).string;
            };
            const path = try std.fs.path.join(gpa, &.{ model_dir, mapped });
            defer gpa.free(path);
            const file = findOpen(g.files.items, path) orelse file: {
                var o = try Open.load(gpa, io, path);
                errdefer o.close(gpa);
                if (@import("builtin").os.tag != .windows) std.posix.madvise(o.file.map.memory.ptr, o.file.map.memory.len, std.posix.MADV.RANDOM) catch {};
                try g.files.append(gpa, o);
                break :file &g.files.items[g.files.items.len - 1];
            };
            const w = try file.tensor(try partNameFor(&key, shard, nested, "weight", modelopt), if (modelopt) .u8 else .u32);
            const expected_rows: usize = @intCast(g.hash.starts[shard + 1] - g.hash.starts[shard]);
            if (w.rank != 2 or w.dim(0) != expected_rows) return error.UnexpectedTensor;
            if (modelopt) {
                const sc = try file.tensor(try partNameFor(&key, shard, nested, "weight_scale", true), .f8_e4m3);
                const table = try core.nvfp4_rows.Table.init(w, sc, g.scale);
                if (table.width != dh) return error.UnexpectedTensor;
                g.tables[shard] = table;
            } else {
                const sc = try file.tensor(try partNameFor(&key, shard, nested, "scales", false), .bf16);
                const bias = try file.tensor(try partNameFor(&key, shard, nested, "biases", false), .bf16);
                if (!w.is(.u32, &.{ expected_rows, dh / 8 }) or !sc.is(.bf16, &.{ expected_rows, dh / 32 }) or
                    !bias.is(.bf16, &.{ expected_rows, dh / 32 })) return error.UnexpectedTensor;
                g.w[shard] = w.bytes;
                g.s[shard] = sc.bytes;
                g.b[shard] = bias.bytes;
            }
        }
        return g;
    }

    pub fn deinit(g: *Ngram) void {
        for (g.files.items) |*o| o.close(g.gpa);
        g.files.deinit(g.gpa);
    }

    /// Values a row's embedding holds (heads x head width).
    pub fn width(g: *const Ngram) usize {
        return g.hash.heads * g.hash.dh;
    }

    /// Rows [lo, hi) of a chunk's token-major row ids `ids`, each dequantized into its head's slice of `out`.
    fn fill(g: *const Ngram, ids: []const i64, out: []u16, lo: usize, hi: usize) !void {
        const dh = g.hash.dh;
        for (lo..hi) |i| {
            const loc = try g.hash.locate(ids[i]);
            const row = out[i * dh ..][0..dh];
            switch (g.format) {
                .mlx => try embed.dequant(g.w[loc.shard], g.s[loc.shard], g.b[loc.shard], dh, loc.local, row),
                .nvfp4 => try g.tables[loc.shard].gather(&.{loc.local}, row),
            }
        }
    }

    fn worker(g: *const Ngram, ids: []const i64, out: []u16, lo: usize, hi: usize, failed: *?anyerror) void {
        g.fill(ids, out, lo, hi) catch |err| {
            failed.* = err;
        };
    }

    /// `embedding` for each token after `history` (moved past the chunk) into out and xs, read by up to 128 threads.
    pub fn rows(g: *const Ngram, history: []i64, tokens: []const u32, out: []u16, xs: []f32) !void {
        const heads = g.hash.heads;
        const w = g.width();
        if (history.len != g.hash.n - 1 or out.len != tokens.len * w or xs.len != out.len / 32) return error.UnexpectedTensor;
        const items = tokens.len * heads;
        const ids = try g.gpa.alloc(i64, items);
        defer g.gpa.free(ids);
        var hist: [max_n]i64 = undefined;
        @memcpy(hist[0..history.len], history);
        for (tokens, 0..) |token, t| {
            try lastIds(g.hash, hist[0..history.len], @intCast(token), ids[t * heads ..][0..heads]);
            for (0..history.len - 1) |i| hist[i] = hist[i + 1];
            hist[history.len - 1] = token;
        }
        const max_threads = 128;
        const n_threads = @min(max_threads, items / 4);
        if (n_threads <= 1) {
            try g.fill(ids, out, 0, items);
        } else {
            var threads: [max_threads]std.Thread = undefined;
            var failed: [max_threads]?anyerror = @splat(null);
            var spawned: usize = 0;
            defer for (threads[0..spawned]) |th| th.join();
            const per = (items + n_threads - 1) / n_threads;
            for (0..n_threads) |i| {
                const lo = @min(items, i * per);
                const hi = @min(items, lo + per);
                threads[i] = try std.Thread.spawn(.{}, worker, .{ g, ids, out, lo, hi, &failed[i] });
                spawned += 1;
            }
            for (threads[0..spawned]) |th| th.join();
            spawned = 0;
            for (failed[0..n_threads]) |f| if (f) |err| return err;
        }
        for (0..tokens.len) |t| {
            const row = out[t * w ..][0..w];
            if (g.format == .mlx and g.scale != 1) {
                for (row) |*v| {
                    const f: f32 = @bitCast(@as(u32, v.*) << 16);
                    v.* = toBf16(f * g.scale);
                }
            }
            sumGroups(row, xs[t * (w / 32) ..][0 .. w / 32]);
        }
        @memcpy(history, hist[0..history.len]);
    }
};

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

test {
    _ = @import("cuda_ple_nvfp4_test.zig");
}
