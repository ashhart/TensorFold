//! An EXL3 pack's n-gram table (ExLlamaV3 `ngram_embedding.safetensors`) behind `cuda_ple.Ngram` (`format = .exl3`).

const std = @import("std");
const core = @import("core");
const ple = @import("../../cuda_ple.zig");
const fmt = core.exl3_format;

pub const file_name = "ngram_embedding.safetensors";
/// Values a row holds: one head's slice of the 2,560-wide embedding.
pub const row_values: usize = 160;
pub const hot_slots: usize = 4096;
/// Bytes a hot slot holds (a row at 8 bits is 162).
pub const hot_cap: usize = 176;

pub const Read = enum { map, pread };
pub const Emb = enum { f16, bf16 };
pub const Options = struct { read: Read = .map, emb: Emb = .f16 };

/// TF_FLASHNEXT_PLE_READ (map | pread); anything else keeps the default. The bf16 rows are a test option (`emb`).
pub fn optionsFromEnv() Options {
    var o: Options = .{};
    if (std.c.getenv("TF_FLASHNEXT_PLE_READ")) |v| {
        if (std.mem.eql(u8, std.mem.span(v), "pread")) o.read = .pread;
    }
    return o;
}

// ---- the row codec (ported from exl3-flashnext-window-wip exl3_ngram.zig) ------------------------------------------

/// Int16 words in one row at `bits`: the fp16 scale, then 160 values. `bits` outside 2..8 is `NgramWidth`.
pub fn wordsPerRow(bits: usize) error{NgramWidth}!usize {
    if (bits < 2 or bits > 8) return error.NgramWidth;
    return 1 + row_values * bits / 16;
}

/// Mul1 state of value `index`: bit m comes from position `index - m / bits` (plus 160 when negative), bit `m % bits`.
pub fn state(words: []const i16, bits: usize, index: usize) u16 {
    var acc: u16 = 0;
    for (0..16) |m| {
        var pos: isize = @as(isize, @intCast(index)) - @as(isize, @intCast(m / bits));
        if (pos < 0) pos += @as(isize, @intCast(row_values));
        const sb: usize = @as(usize, @intCast(pos)) * bits + (m % bits);
        const word: u16 = @bitCast(words[1 + (sb >> 4)]);
        acc |= ((word >> @as(u4, @intCast(sb & 15))) & 1) << @as(u4, @intCast(m));
    }
    return acc;
}

/// One row into 160 fp16 values: one rounding of `f32(codebook) * f32(scale) + f32(bias[i])`.
pub fn decodeRow(words: []const i16, bits: usize, bias: []const f16, out: []f16) error{NgramWidth}!void {
    const width = try wordsPerRow(bits);
    if (words.len != width or bias.len != row_values or out.len != row_values) return error.NgramWidth;
    const scale: f32 = @floatCast(@as(f16, @bitCast(words[0])));
    // `state(words, bits, i)` for every i at once: value j is stream bits [j * bits, (j + 1) * bits); state i is
    var vals: [row_values]u32 = undefined;
    const mask: u32 = (@as(u32, 1) << @intCast(bits)) - 1;
    for (&vals, 0..) |*v, j| {
        const sb = j * bits;
        const wi = 1 + (sb >> 4);
        const lo: u32 = @as(u16, @bitCast(words[wi]));
        const hi: u32 = if (wi + 1 < words.len) @as(u16, @bitCast(words[wi + 1])) else 0;
        v.* = ((lo | (hi << 16)) >> @intCast(sb & 15)) & mask;
    }
    for (out, bias, 0..) |*o, b, i| {
        var acc: u32 = 0;
        var k: usize = 0;
        while (k * bits < 16) : (k += 1) acc |= vals[(i + row_values - k) % row_values] << @intCast(k * bits);
        const cb: f32 = @floatCast(fmt.codebookValue(.mul1, @truncate(acc)));
        o.* = @floatCast(cb * scale + @as(f32, @floatCast(b)));
    }
}

test "exl3 ple: the packed decode's states equal `state` for every value at every width" {
    var prng = std.Random.DefaultPrng.init(11);
    for (2..9) |bits| {
        const n = try wordsPerRow(bits);
        var words: [1 + row_values * 8 / 16]i16 = undefined;
        for (words[0..n]) |*w| w.* = @bitCast(prng.random().int(u16));
        words[0] = @bitCast(@as(f16, 0.5));
        var bias: [row_values]f16 = undefined;
        for (&bias) |*b| b.* = @floatCast(prng.random().float(f32) - 0.5);
        var got: [row_values]f16 = undefined;
        try decodeRow(words[0..n], bits, &bias, &got);
        const scale: f32 = @floatCast(@as(f16, @bitCast(words[0])));
        for (got, bias, 0..) |g, b, i| {
            const cb: f32 = @floatCast(fmt.codebookValue(.mul1, state(words[0..n], bits, i)));
            const want: f16 = @floatCast(cb * scale + @as(f32, @floatCast(b)));
            try std.testing.expectEqual(@as(u16, @bitCast(want)), @as(u16, @bitCast(g)));
        }
    }
}

fn bf16Of(v: f32) u16 {
    return ple.toBf16(v);
}

/// Group sums of 32 fp16 values (the tree `cuda_ple.sumGroups` adds bf16 values in), for an fp16 row.
pub fn sumGroupsF16(row: []const u16, out: []f32) void {
    for (0..row.len / 32) |g| {
        var buf: [32]f32 = undefined;
        for (0..32) |i| buf[i] = @floatCast(@as(f16, @bitCast(row[g * 32 + i])));
        var n: usize = 32;
        while (n > 1) {
            n /= 2;
            for (0..n) |i| buf[i] = buf[2 * i] + buf[2 * i + 1];
        }
        out[g] = buf[0];
    }
}

// ---- the table ----------------------------------------------------------------------------------------------------

const c = std.c;
const is_windows = @import("builtin").os.tag == .windows;

/// The open table. Heap-allocated (`open`/`destroy`): the pread reader's hot cache is 512 KiB.
pub const Table = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    opts: Options,
    heads: usize,
    shards: usize,
    bits: usize,
    words: usize,
    /// File byte offset of each shard's row 0, and its row count.
    offset: [128]u64 = undefined,
    rows: [128]u64 = undefined,
    bias: []f16,
    /// `.map`: the whole file's mapping (header parsed, nothing populated).
    file: ?core.safetensors.File = null,
    /// `.pread`: the descriptor; the file is never mapped past `open`.
    fd: c.fd_t = if (is_windows) undefined else -1,
    hot: ?*Hot = null,

    pub const Hot = struct {
        lock: std.atomic.Value(u32) = .init(0),
        lookups: u64 = 0,
        misses: u64 = 0,
        tags: [hot_slots]u64 = @splat(std.math.maxInt(u64)),
        stored: [hot_slots][hot_cap]u8 = undefined,

        fn acquire(h: *Hot) void {
            while (h.lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
        }
        fn release(h: *Hot) void {
            h.lock.store(0, .release);
        }
    };

    fn prefix(buf: []u8, part: []const u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "model.language_model.layers.{d}.ple.ple_embedding.ngram_embedding.{s}", .{ ple.layer, part });
    }

    /// The pack's table when `dir` holds an EXL3 n-gram file whose constants equal `hash`'s; null.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, hash: ple.Hash, opts: Options) !?*Table {
        const path = try std.fs.path.join(gpa, &.{ dir, file_name });
        defer gpa.free(path);
        var file = core.safetensors.File.open(gpa, io, path) catch |err| switch (err) {
            error.FileNotFound => return null,
            else => return err,
        };
        var keep_file = false;
        defer if (!keep_file) file.close(io);
        var nb: [200]u8 = undefined;
        const first = file.get(try prefix(&nb, "shard_0.trellis")) orelse return null;
        if (first.dtype != .i16 or first.rank != 2) return error.UnexpectedTensor;
        if (hash.shards > 128 or hash.dh != row_values) return error.UnexpectedTensor;
        // the hash constants the pack ships against the ones the config derives
        {
            var got: [ple.max_heads]i64 = undefined;
            try ple.readI64(file.get(try prefix(&nb, "layer_multipliers")) orelse return error.MissingTensor, got[0..hash.n]);
            if (!std.mem.eql(i64, got[0..hash.n], hash.mult[0..hash.n])) return error.NgramConstants;
            try ple.readI64(file.get(try prefix(&nb, "head_vocab_sizes")) orelse return error.MissingTensor, got[0..hash.heads]);
            if (!std.mem.eql(i64, got[0..hash.heads], hash.sizes[0..hash.heads])) return error.NgramConstants;
            try ple.readI64(file.get(try prefix(&nb, "head_offsets")) orelse return error.MissingTensor, got[0..hash.heads]);
            if (!std.mem.eql(i64, got[0..hash.heads], hash.offsets[0..hash.heads])) return error.NgramConstants;
        }
        // the reference's EXL3 path applies no table scale: refuse a file that carries one
        if (file.get(try prefix(&nb, "weight_scale")) != null) return error.UnexpectedTensor;
        const bias_t = file.get(try prefix(&nb, "head_bias")) orelse return error.MissingTensor;
        if (!bias_t.is(.f16, &.{ hash.heads, row_values })) return error.UnexpectedTensor;
        const words = first.dim(1);
        const bits = (words - 1) * 16 / row_values;
        if ((wordsPerRow(bits) catch return error.UnexpectedTensor) != words) return error.UnexpectedTensor;

        const t = try gpa.create(Table);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .io = io, .opts = opts, .heads = hash.heads, .shards = hash.shards, .bits = bits, .words = words, .bias = try gpa.alloc(f16, hash.heads * row_values) };
        errdefer gpa.free(t.bias);
        @memcpy(std.mem.sliceAsBytes(t.bias), bias_t.bytes);
        for (0..hash.shards) |i| {
            var part: [32]u8 = undefined;
            const e = file.names.get(try prefix(&nb, try std.fmt.bufPrint(&part, "shard_{d}.trellis", .{i}))) orelse return error.MissingTensor;
            const want: usize = @intCast(hash.starts[i + 1] - hash.starts[i]);
            if (e.dtype != .i16 or e.rank != 2 or e.shape[0] != want or e.shape[1] != words) return error.UnexpectedTensor;
            t.offset[i] = file.data + e.begin;
            t.rows[i] = want;
        }
        switch (opts.read) {
            .map => {
                if (@import("builtin").os.tag == .linux) {
                    if (file.map.section != null) _ = std.os.linux.madvise(file.map.memory.ptr, file.map.memory.len, std.os.linux.MADV.RANDOM);
                }
                t.file = file;
                keep_file = true;
            },
            .pread => {
                if (comptime is_windows) return error.UnsupportedPlatform;
                const z = try gpa.dupeSentinel(u8, path, 0);
                defer gpa.free(z);
                const fd = c.open(z.ptr, .{ .ACCMODE = .RDONLY });
                if (fd < 0) return error.OpenFailed;
                t.fd = fd;
                errdefer _ = c.close(fd);
                if (@import("builtin").os.tag == .linux) _ = std.os.linux.fadvise(fd, 0, 0, std.os.linux.POSIX_FADV.RANDOM);
                const hot = try gpa.create(Hot);
                hot.* = .{};
                t.hot = hot;
            },
        }
        return t;
    }

    pub fn destroy(t: *Table) void {
        if (t.file) |*f| f.close(t.io);
        if (comptime !is_windows) {
            if (t.fd >= 0) _ = c.close(t.fd);
        }
        if (t.hot) |h| t.gpa.destroy(h);
        t.gpa.free(t.bias);
        t.gpa.destroy(t);
    }

    pub fn rowBytes(t: *const Table) usize {
        return t.words * 2;
    }

    fn preadAll(t: *const Table, dest: []u8, at: u64) !void {
        if (comptime is_windows) return error.UnsupportedPlatform;
        var done: usize = 0;
        while (done < dest.len) {
            const n = c.pread(t.fd, dest.ptr + done, dest.len - done, @intCast(at + done));
            if (n <= 0) return error.ReadFailed;
            done += @intCast(n);
        }
    }

    /// The stored bytes of row `local` of shard `shard` (`rowBytes` long), through the mapping or the hot cache.
    pub fn raw(t: *const Table, shard: usize, local: usize, dest: []u8) !void {
        if (shard >= t.shards or local >= t.rows[shard] or dest.len != t.rowBytes()) return error.UnexpectedTensor;
        const at = t.offset[shard] + @as(u64, local) * t.rowBytes();
        if (t.file) |f| {
            @memcpy(dest, f.map.memory[at..][0..dest.len]);
            return;
        }
        const h = t.hot.?;
        const slot: usize = @intCast(((at *% 0x9E3779B97F4A7C15) >> 20) % hot_slots);
        h.acquire();
        h.lookups += 1;
        if (h.tags[slot] == at) {
            @memcpy(dest, h.stored[slot][0..dest.len]);
            h.release();
            return;
        }
        h.release();
        try t.preadAll(dest, at);
        h.acquire();
        h.misses += 1;
        h.tags[slot] = at;
        @memcpy(h.stored[slot][0..dest.len], dest);
        h.release();
    }

    /// Row `local` of shard `shard`, decoded with head `head`'s bias into 160 fp16 (or bf16) bit patterns.
    pub fn row(t: *const Table, shard: usize, local: usize, head: usize, out: []u16) !void {
        if (head >= t.heads or out.len != row_values) return error.UnexpectedTensor;
        var bytes: [hot_cap]u8 align(2) = undefined;
        const rb = t.rowBytes();
        try t.raw(shard, local, bytes[0..rb]);
        const words: []const i16 = @as([*]const i16, @ptrCast(&bytes))[0..t.words];
        var vals: [row_values]f16 = undefined;
        try decodeRow(words, t.bits, t.bias[head * row_values ..][0..row_values], &vals);
        switch (t.opts.emb) {
            .f16 => for (out, vals) |*o, v| {
                o.* = @bitCast(v);
            },
            .bf16 => for (out, vals) |*o, v| {
                o.* = bf16Of(@floatCast(v));
            },
        }
    }

    /// Group sums of one decoded row in the table's element type.
    pub fn sums(t: *const Table, row_bits: []const u16, out: []f32) void {
        switch (t.opts.emb) {
            .f16 => sumGroupsF16(row_bits, out),
            .bf16 => ple.sumGroups(row_bits, out),
        }
    }

    pub fn hotStats(t: *const Table) struct { lookups: u64, misses: u64 } {
        const h = t.hot orelse return .{ .lookups = 0, .misses = 0 };
        return .{ .lookups = h.lookups, .misses = h.misses };
    }
};

// ---- tests --------------------------------------------------------------------------------------------------------

/// The Triton `_ple_rows` formulation, a second decoder that shares only the three fp16 constants with `decodeRow`.
fn decodeRowRef(words: []const i16, bits: usize, bias: []const f16, out: []f16) void {
    const k_inv: f32 = @floatCast(@as(f16, @bitCast(fmt.Mul1ScaleBits)));
    const k_bias: f32 = @floatCast(@as(f16, @bitCast(fmt.Mul1BiasBits)));
    const scale: f32 = @floatCast(@as(f16, @bitCast(words[0])));
    for (0..out.len) |i| {
        var st: u64 = 0;
        for (0..16) |m| {
            var pos: isize = @as(isize, @intCast(i)) - @as(isize, @intCast(m / bits));
            if (pos < 0) pos += @as(isize, @intCast(out.len));
            const sb: usize = @as(usize, @intCast(pos)) * bits + (m % bits);
            const word: u64 = @as(u16, @bitCast(words[1 + (sb >> 4)]));
            st |= ((word >> @intCast(sb & 15)) & 1) << @intCast(m);
        }
        const prod: u64 = (st * 0x83DCD12D) & 0xFFFFFFFF;
        const hs: u64 = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255);
        const cb: f16 = @floatCast(@as(f32, @floatFromInt(1024 + hs)) * k_inv + k_bias);
        out[i] = @floatCast(@as(f32, @floatCast(cb)) * scale + @as(f32, @floatCast(bias[i])));
    }
}

test "exl3 ngram: wordsPerRow is one scale word plus 160 K-bit values" {
    try std.testing.expectEqual(@as(usize, 21), try wordsPerRow(2));
    try std.testing.expectEqual(@as(usize, 31), try wordsPerRow(3));
    try std.testing.expectEqual(@as(usize, 51), try wordsPerRow(5));
    try std.testing.expectEqual(@as(usize, 81), try wordsPerRow(8));
    try std.testing.expectError(error.NgramWidth, wordsPerRow(1));
    try std.testing.expectError(error.NgramWidth, wordsPerRow(9));
}

test "exl3 ngram: patterned rows read tail-biting states" {
    var storage: [81]i16 = undefined;
    for (&storage, 0..) |*w, i| w.* = @bitCast(@as(u16, @intCast(i)));
    try std.testing.expectEqual(@as(u16, 0x5001), state(storage[0..21], 2, 0));
    try std.testing.expectEqual(@as(u16, 0x9001), state(storage[0..31], 3, 0));
    try std.testing.expectEqual(@as(u16, 0x8001), state(storage[0..51], 5, 0));
    try std.testing.expectEqual(@as(u16, 0x0020), state(storage[0..51], 5, 1));
    try std.testing.expectEqual(@as(u16, 0x0400), state(storage[0..51], 5, 2));
    try std.testing.expectEqual(@as(u16, 0x0001), state(storage[0..81], 8, 0));
    // value 0 at 5 bits: bit 5 wraps to position 159, word 50 bit 11
    var words: [51]i16 = @splat(0);
    words[50] = @bitCast(@as(u16, 1) << 11);
    try std.testing.expectEqual(@as(u16, 1) << 5, state(&words, 5, 0));
}

test "exl3 ngram: scale and bias round in f32, then once to f16" {
    var words: [51]i16 = @splat(0);
    words[0] = @bitCast(@as(u16, 0x3C01));
    var bias: [160]f16 = @splat(0);
    bias[0] = @bitCast(@as(u16, 0x1400));
    var out: [160]f16 = undefined;
    try decodeRow(&words, 5, &bias, &out);
    // codebook(0) is 0xC2E8; times 0x3C01 plus 0x1400 is 0xC2E9 (rounding the product first would give 0xC2EA)
    try std.testing.expectEqual(@as(u16, 0xC2E8), @as(u16, @bitCast(fmt.codebookValue(.mul1, 0))));
    try std.testing.expectEqual(@as(u16, 0xC2E9), @as(u16, @bitCast(out[0])));
    try std.testing.expectError(error.NgramWidth, decodeRow(words[0..50], 5, &bias, &out));
    try std.testing.expectError(error.NgramWidth, decodeRow(&words, 2, &bias, &out));
}

test "exl3 ngram: decodeRow equals the Triton formulation on random rows at every width" {
    var rng = std.Random.DefaultPrng.init(0x91e);
    const r = rng.random();
    var words: [81]i16 = undefined;
    var bias: [160]f16 = undefined;
    var a: [160]f16 = undefined;
    var b: [160]f16 = undefined;
    for ([_]usize{ 2, 3, 4, 5, 6, 8 }) |bits| {
        const n = try wordsPerRow(bits);
        for (0..300) |_| {
            for (words[0..n]) |*x| x.* = @bitCast(r.int(u16));
            words[0] = @bitCast(@as(f16, @floatCast((r.float(f32) - 0.5) * 4)));
            for (&bias) |*x| x.* = @floatCast((r.float(f32) - 0.5) * 0.25);
            try decodeRow(words[0..n], bits, &bias, &a);
            decodeRowRef(words[0..n], bits, &bias, &b);
            try std.testing.expectEqualSlices(u16, @ptrCast(&b), @ptrCast(&a));
        }
    }
}

/// A small table file plus its config: 2 shards, 4 heads of 160 (n-gram 3, 2 heads an n-gram), 5-bit rows.
const Synth = struct {
    const vocab = 100;
    const eos = 99;
    const base = 11;

    fn spec() ple.Hash {
        return ple.Hash.init(.{ .vocab = vocab, .ngram = 3, .per = 2, .base = base, .divisor = 4, .shards = 2, .seed = 1234, .eos = eos, .ple_dim = 4 * row_values, .ple_index = 0 }) catch unreachable;
    }

    fn put(list: *std.ArrayList(u8), gpa: std.mem.Allocator, bytes: []const u8) !void {
        try list.appendSlice(gpa, bytes);
    }

    /// Writes config.json, an empty index and ngram_embedding.safetensors under `dir`; row r's words are a hash of r.
    fn write(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !void {
        const h = spec();
        var data: std.ArrayList(u8) = .empty;
        defer data.deinit(gpa);
        var header: std.ArrayList(u8) = .empty;
        defer header.deinit(gpa);
        try header.appendSlice(gpa, "{");
        const p = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.";
        var first = true;
        const Entry = struct {
            fn add(hd: *std.ArrayList(u8), a: std.mem.Allocator, f: *bool, name: []const u8, dtype: []const u8, shape: []const usize, lo: usize, hi: usize) !void {
                if (!f.*) try hd.appendSlice(a, ",");
                f.* = false;
                var buf: [400]u8 = undefined;
                var sb: [64]u8 = undefined;
                var sl: usize = 0;
                for (shape, 0..) |d, i| {
                    const s = try std.fmt.bufPrint(sb[sl..], "{s}{d}", .{ if (i == 0) "" else ",", d });
                    sl += s.len;
                }
                try hd.appendSlice(a, try std.fmt.bufPrint(&buf, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{s}],\"data_offsets\":[{d},{d}]}}", .{ name, dtype, sb[0..sl], lo, hi }));
            }
        };
        var nb: [200]u8 = undefined;
        for (0..h.shards) |s| {
            const rows: usize = @intCast(h.starts[s + 1] - h.starts[s]);
            const lo = data.items.len;
            for (0..rows) |r| {
                const global: u64 = @intCast(h.starts[s] + @as(i64, @intCast(r)));
                var prng = std.Random.DefaultPrng.init(global + 1);
                for (0..51) |w| {
                    var v: u16 = prng.random().int(u16);
                    if (w == 0) v = @bitCast(@as(f16, @floatCast(0.25 + 0.01 * @as(f32, @floatFromInt(global)))));
                    try data.appendSlice(gpa, &std.mem.toBytes(v));
                }
            }
            try Entry.add(&header, gpa, &first, try std.fmt.bufPrint(&nb, "{s}shard_{d}.trellis", .{ p, s }), "I16", &.{ rows, 51 }, lo, data.items.len);
        }
        var lo = data.items.len;
        for (0..h.heads * row_values) |i| try data.appendSlice(gpa, &std.mem.toBytes(@as(f16, @floatCast(@as(f32, @floatFromInt(i % 7)) * 0.01))));
        try Entry.add(&header, gpa, &first, try std.fmt.bufPrint(&nb, "{s}head_bias", .{p}), "F16", &.{ h.heads, row_values }, lo, data.items.len);
        lo = data.items.len;
        for (h.mult[0..h.n]) |v| try data.appendSlice(gpa, &std.mem.toBytes(v));
        try Entry.add(&header, gpa, &first, try std.fmt.bufPrint(&nb, "{s}layer_multipliers", .{p}), "I64", &.{h.n}, lo, data.items.len);
        lo = data.items.len;
        for (h.sizes[0..h.heads]) |v| try data.appendSlice(gpa, &std.mem.toBytes(v));
        try Entry.add(&header, gpa, &first, try std.fmt.bufPrint(&nb, "{s}head_vocab_sizes", .{p}), "I64", &.{h.heads}, lo, data.items.len);
        lo = data.items.len;
        for (h.offsets[0..h.heads]) |v| try data.appendSlice(gpa, &std.mem.toBytes(v));
        try Entry.add(&header, gpa, &first, try std.fmt.bufPrint(&nb, "{s}head_offsets", .{p}), "I64", &.{h.heads}, lo, data.items.len);
        try header.appendSlice(gpa, "}");
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        try out.appendSlice(gpa, &std.mem.toBytes(@as(u64, header.items.len)));
        try out.appendSlice(gpa, header.items);
        try out.appendSlice(gpa, data.items);
        try dir.writeFile(io, .{ .sub_path = file_name, .data = out.items });
        try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{}}" });
        try dir.writeFile(io, .{ .sub_path = "config.json", .data =
            \\{"text_config":{"vocab_size":100,"eos_token_id":99,"num_hidden_layers":4,"hidden_size":640,"ple_layer_ids":[2],
            \\"ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":11,"make_ngram_vocab_size_divisible_by":4,
            \\"split_ngram_parts":2,"ple_embed_dim":640}}
        });
    }
};

test "exl3 ngram: Ngram.rows on a synthetic EXL3 table: hook, map and pread readers, hot cache" {
    if (comptime is_windows) return error.SkipZigTest; // the pread reader is POSIX-only
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try Synth.write(gpa, io, tmp.dir);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);

    var g = try ple.Ngram.init(gpa, io, dir);
    defer g.deinit();
    try std.testing.expect(g.format == .exl3);
    const h = g.hash;
    try std.testing.expectEqual(@as(usize, 4), h.heads);
    const tokens = [_]u32{ 5, 7, Synth.eos, 7, 7, 3 };
    const w = g.width();
    var out: [tokens.len * 4 * row_values]u16 = undefined;
    var xs: [tokens.len * 4 * row_values / 32]f32 = undefined;
    var history = [_]i64{ Synth.eos, Synth.eos };
    try g.rows(&history, &tokens, &out, &xs);
    try std.testing.expectEqualSlices(i64, &.{ 7, 3 }, &history);

    // the same rows by hand: upstream's ids, the shard's bytes, decodeRow
    var pread_t = (try Table.open(gpa, io, dir, h, .{ .read = .pread })).?;
    defer pread_t.destroy();
    var hist = [_]i64{ Synth.eos, Synth.eos };
    for (tokens, 0..) |tok, t| {
        var ids: [4]i64 = undefined;
        try ple.lastIds(h, &hist, tok, &ids);
        hist = .{ hist[1], tok };
        for (ids, 0..) |id, head| {
            const loc = try h.locate(id);
            var bytes: [102]u8 align(2) = undefined;
            try pread_t.raw(loc.shard, loc.local, &bytes);
            var mapped: [102]u8 = undefined;
            try g.x3.?.raw(loc.shard, loc.local, &mapped);
            try std.testing.expectEqualSlices(u8, &mapped, &bytes);
            var vals: [row_values]f16 = undefined;
            try decodeRow(@as([*]const i16, @ptrCast(&bytes))[0..51], 5, pread_t.bias[head * row_values ..][0..row_values], &vals);
            const got = out[t * w + head * row_values ..][0..row_values];
            try std.testing.expectEqualSlices(u16, @ptrCast(&vals), got);
            var gs: [5]f32 = undefined;
            sumGroupsF16(got, &gs);
            try std.testing.expectEqualSlices(f32, &gs, xs[(t * w + head * row_values) / 32 ..][0..5]);
        }
    }
    // tokens 2 and 3 after an eos: token 7 twice in a fresh segment reads the same rows; the cache serves the repeat
    const before = pread_t.hotStats();
    var again: [102]u8 = undefined;
    const loc = try h.locate(h.offsets[0]);
    try pread_t.raw(loc.shard, loc.local, &again);
    try pread_t.raw(loc.shard, loc.local, &again);
    const after = pread_t.hotStats();
    try std.testing.expectEqual(before.lookups + 2, after.lookups);
    try std.testing.expect(after.misses <= before.misses + 1);
    // bf16 elements: the fp16 values rounded once more
    var bf_t = (try Table.open(gpa, io, dir, h, .{ .emb = .bf16 })).?;
    defer bf_t.destroy();
    var r16: [row_values]u16 = undefined;
    var rbf: [row_values]u16 = undefined;
    try g.x3.?.row(loc.shard, loc.local, 0, &r16);
    try bf_t.row(loc.shard, loc.local, 0, &rbf);
    for (r16, rbf) |a, b| try std.testing.expectEqual(ple.toBf16(@floatCast(@as(f16, @bitCast(a)))), b);
}

test "exl3 ngram: a directory without the n-gram file is not an EXL3 table" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    try std.testing.expect((try Table.open(gpa, io, dir, Synth.spec(), .{})) == null);
}
