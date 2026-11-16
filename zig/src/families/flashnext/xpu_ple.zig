//! Layer 1's Per-Layer N-Gram Embedding (PLE) table for XPU: host RAM mmap + multi-threaded gather.
const std = @import("std");
const core = @import("core");
const host_pool = @import("host_pool.zig");

const golden: u64 = 0x9E3779B97F4A7C15;
const mix1: u64 = 0xBF58476D1CE4E5B9;
const mix2: u64 = 0x94D049BB133111EB;
const prime_step: u64 = 10007;
pub const max_n: usize = 8;
pub const max_heads: usize = 64;
pub const max_shards: usize = 128;
pub const layer: usize = 1;

const Tensor = core.safetensors.Tensor;
const toBf16 = core.bf16_round.bits;

fn f32FromBf16(b: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, b[2 * i ..][0..2], .little)) << 16);
}

/// Dequantize one MLX affine 4-bit group-32 row (`token`) of width `dims` into `out` (bf16 bits).
pub fn dequantRow(weight: []const u8, scales: []const u8, biases: []const u8, dims: usize, token: usize, out: []u16) !void {
    if (dims == 0 or dims % 32 != 0 or out.len != dims) return error.UnexpectedTensor;
    const groups = dims / 32;
    const words = dims / 8;
    const woff = token * words * 4;
    const soff = token * groups * 2;
    if (woff + words * 4 > weight.len or soff + groups * 2 > scales.len or soff + groups * 2 > biases.len) return error.UnexpectedTensor;
    for (0..groups) |g| {
        const scale = f32FromBf16(scales[soff..], g);
        const bias = f32FromBf16(biases[soff..], g);
        for (0..4) |i| {
            const word = std.mem.readInt(u32, weight[woff + (g * 4 + i) * 4 ..][0..4], .little);
            for (0..8) |j| {
                const q: f32 = @floatFromInt((word >> @intCast(j * 4)) & 0xF);
                out[g * 32 + i * 8 + j] = toBf16(q * scale + bias);
            }
        }
    }
}

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

pub const Open = struct {
    file: core.safetensors.File,
    io: std.Io,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Open {
        return .{ .file = try core.safetensors.File.open(gpa, io, path), .io = io };
    }

    pub fn close(self: *Open) void {
        self.file.close(self.io);
    }

    pub fn tensor(self: *const Open, name: []const u8, dtype: core.safetensors.DType) !Tensor {
        const t = self.file.get(name) orelse return error.MissingTensor;
        if (t.dtype != dtype) return error.UnexpectedTensor;
        return t;
    }
};

fn partNameFor(buf: []u8, shard: usize, nested: bool, part: []const u8, modelopt: bool) ![]const u8 {
    const prefix = if (modelopt) "model.language_model" else "language_model.model";
    return if (nested)
        std.fmt.bufPrint(buf, "{s}.layers.{d}.ple.ple_embedding.ngram_embedding.shards.{d}.{s}", .{ prefix, layer, shard, part })
    else
        std.fmt.bufPrint(buf, "{s}.layers.{d}.ple.ple_embedding.ngram_embedding.shard_{d}.{s}", .{ prefix, layer, shard, part });
}

fn findOpen(opened: []Open, path: []const u8) ?*Open {
    for (opened) |*o| if (std.mem.eql(u8, o.file.path, path)) return o;
    return null;
}

fn readScale(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, wm: std.json.ObjectMap, name: []const u8) !f32 {
    const file = wm.get(name) orelse return 1.0;
    const path = try std.fs.path.join(gpa, &.{ dir, file.string });
    defer gpa.free(path);
    var opened = try Open.load(gpa, io, path);
    defer opened.close();
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

pub const NgramSpec = struct {
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

fn hostRamAvailableBytes(io: std.Io, gpa: std.mem.Allocator) u64 {
    if (@import("builtin").os.tag != .linux) return 0;
    const text = std.Io.Dir.cwd().readFileAlloc(io, "/proc/meminfo", gpa, .limited(64 << 10)) catch return 0;
    defer gpa.free(text);
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, "MemAvailable:")) {
            var tok = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \t");
            const kb = std.fmt.parseInt(u64, tok.next() orelse return 0, 10) catch return 0;
            return kb * 1024;
        }
    }
    return 0;
}

pub const Ngram = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    hash: Hash,
    scale: f32,
    files: std.ArrayList(Open) = .empty,
    w: [max_shards][]const u8 = undefined,
    s: [max_shards][]const u8 = undefined,
    b: [max_shards][]const u8 = undefined,
    pool: ?*host_pool.Pool = null,
    warmer: ?std.Thread = null,
    total_bytes: usize = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8) !Ngram {
        const cfg = try readSpec(gpa, io, model_dir);
        var g: Ngram = .{
            .gpa = gpa,
            .io = io,
            .hash = try Hash.init(.{
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
            }),
            .scale = 1.0,
        };
        errdefer g.deinit();
        const index_path = try std.fs.path.join(gpa, &.{ model_dir, "model.safetensors.index.json" });
        defer gpa.free(index_path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, gpa, .limited(1 << 26));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        const wm = try core.safetensors.index.weightMap(parsed.value);
        g.scale = try readScale(gpa, io, model_dir, wm, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale");
        const dh = g.hash.dh;
        var key: [180]u8 = undefined;
        for (0..g.hash.shards) |shard| {
            var nested = false;
            const mapped = if (wm.get(try partNameFor(&key, shard, false, "weight", false))) |v|
                v.string
            else blk: {
                nested = true;
                break :blk (wm.get(try partNameFor(&key, shard, true, "weight", false)) orelse return error.MissingTensor).string;
            };
            const path = try std.fs.path.join(gpa, &.{ model_dir, mapped });
            defer gpa.free(path);
            const file = findOpen(g.files.items, path) orelse file: {
                var o = try Open.load(gpa, io, path);
                errdefer o.close();
                if (@import("builtin").os.tag != .windows) {
                    std.posix.madvise(o.file.map.memory.ptr, o.file.map.memory.len, std.posix.MADV.RANDOM) catch {};
                }
                try g.files.append(gpa, o);
                break :file &g.files.items[g.files.items.len - 1];
            };
            const w = try file.tensor(try partNameFor(&key, shard, nested, "weight", false), .u32);
            const sc = try file.tensor(try partNameFor(&key, shard, nested, "scales", false), .bf16);
            const bias = try file.tensor(try partNameFor(&key, shard, nested, "biases", false), .bf16);
            const expected_rows: usize = @intCast(g.hash.starts[shard + 1] - g.hash.starts[shard]);
            if (!w.is(.u32, &.{ expected_rows, dh / 8 }) or
                !sc.is(.bf16, &.{ expected_rows, dh / 32 }) or
                !bias.is(.bf16, &.{ expected_rows, dh / 32 })) return error.UnexpectedTensor;
            g.w[shard] = w.bytes;
            g.s[shard] = sc.bytes;
            g.b[shard] = bias.bytes;
            g.total_bytes += w.bytes.len + sc.bytes.len + bias.bytes.len;
        }
        g.pool = try host_pool.Pool.create(gpa, io);
        return g;
    }

    pub fn deinit(g: *Ngram) void {
        g.waitWarm();
        if (g.pool) |p| p.destroy();
        for (g.files.items) |*o| o.close();
        g.files.deinit(g.gpa);
    }

    pub fn width(g: *const Ngram) usize {
        return g.hash.heads * g.hash.dh;
    }

    /// Warm the n-gram table into host RAM page cache (default on when >= 42 GiB is free, or XPU_FLASHNEXT_WARM=1).
    pub fn warm(g: *Ngram) void {
        if (@import("builtin").os.tag != .linux) return;
        const explicit: ?bool = @import("sys").environment.read(?bool, "XPU_FLASHNEXT_WARM", null) catch null;
        const on = explicit orelse (hostRamAvailableBytes(g.io, g.gpa) >= (42 << 30));
        if (!on) return;
        g.warmer = std.Thread.spawn(.{}, warmMain, .{g.*}) catch null;
    }

    pub fn waitWarm(g: *Ngram) void {
        if (g.warmer) |t| t.join();
        g.warmer = null;
    }

    fn warmMain(g: Ngram) void {
        var bytes: usize = 0;
        for (0..g.hash.shards) |shard| {
            bytes += advise(g.w[shard]) + advise(g.s[shard]) + advise(g.b[shard]);
        }
        std.log.info("flashnext-xpu: {d:.1} GiB PLE n-gram table warmed into host RAM page cache", .{@as(f64, @floatFromInt(bytes)) / (1 << 30)});
    }

    fn advise(bytes: []const u8) usize {
        if (bytes.len == 0) return 0;
        const page = std.heap.pageSize();
        const step = 128 << 10;
        const lo = std.mem.alignBackward(usize, @intFromPtr(bytes.ptr), page);
        const hi = std.mem.alignForward(usize, @intFromPtr(bytes.ptr) + bytes.len, page);
        var at = lo;
        while (at < hi) : (at += step) _ = std.os.linux.madvise(@ptrFromInt(at), @min(step, hi - at), std.os.linux.MADV.WILLNEED);
        return bytes.len;
    }

    pub fn fill(g: *const Ngram, ids: []const i64, out: []u16, lo: usize, hi: usize) !void {
        const dh = g.hash.dh;
        for (lo..hi) |i| {
            const loc = try g.hash.locate(ids[i]);
            const row = out[i * dh ..][0..dh];
            try dequantRow(g.w[loc.shard], g.s[loc.shard], g.b[loc.shard], dh, loc.local, row);
        }
    }

    const Split = struct {
        g: *const Ngram,
        ids: []const i64,
        out: []u16,
        per: usize,

        fn part(ctx: *const anyopaque, i: usize) anyerror!void {
            const s: *const Split = @ptrCast(@alignCast(ctx));
            const lo = @min(s.ids.len, i * s.per);
            try s.g.fill(s.ids, s.out, lo, @min(s.ids.len, lo + s.per));
        }
    };

    /// Gather `[tokens.len, 2560]` bf16 PLE embeddings from host RAM/mmap and advance `history`.
    pub fn rows(g: *const Ngram, history: []i64, tokens: []const u32, out: []u16) !void {
        const heads = g.hash.heads;
        const w = g.width();
        if (history.len != g.hash.n - 1 or out.len != tokens.len * w) return error.UnexpectedTensor;
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
        const n_threads = @min(host_pool.max_parts, items / 4);
        if (n_threads <= 1 or g.pool == null) {
            try g.fill(ids, out, 0, items);
        } else {
            const split: Split = .{ .g = g, .ids = ids, .out = out, .per = (items + n_threads - 1) / n_threads };
            try g.pool.?.run(n_threads, .{ .ctx = &split, .run = Split.part });
        }
        if (g.scale != 1.0) {
            for (out) |*v| {
                const f: f32 = @bitCast(@as(u32, v.*) << 16);
                v.* = toBf16(f * g.scale);
            }
        }
        @memcpy(history, hist[0..history.len]);
    }
};
