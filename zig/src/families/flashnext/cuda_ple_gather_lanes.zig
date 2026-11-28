//! The shared pass's n-gram gather: every lane's rows in one pool call, so the dedup set and the readers are shared.

const std = @import("std");
const ple = @import("cuda_ple.zig");
const ple_pool = @import("cuda_ple_pool.zig");
const wt = @import("cuda_prompt_wt.zig");
const batch_state = @import("cuda_batch_state.zig");

const dims = wt.dims;

/// The most lanes one pooled call takes: its parts and their histories sit on the gather's own stack.
pub const max_lanes: usize = 64;

/// TF_FLASHNEXT_GATHER_EARLY=1: a shared pass gathers every lane's rows in one pool call.
pub fn gatherEarly() bool {
    // the startup check refused every malformed value already, so a plain default stands here
    return @import("sys").environment.read(bool, "TF_FLASHNEXT_GATHER_EARLY", false) catch false;
}

/// The switch and the lane bound: one lane, or 65 and more, keeps the pass's own `rows` call a stream.
pub fn earlyLanes(on: bool, lanes: usize) bool {
    return on and lanes >= 1 and lanes <= max_lanes;
}

/// One stream's rows in a shared gather: its history (moved past the rows), its tokens, and its slices of out and xs.
pub const Part = struct { history: []i64, tokens: []const u32, out: []u16, xs: []f32 };

/// A shared pass's lanes gathered in one pool call when the switch says so; sets failed.
pub fn early(g: *batch_state.Gather) bool {
    return earlyOn(g, gatherEarly());
}

/// `early` with the switch given, so a test drives the lanes without the process's environment.
pub fn earlyOn(g: *batch_state.Gather, on: bool) bool {
    if (!earlyLanes(on, g.lanes.len)) return false;
    var parts: [max_lanes]Part = undefined;
    var at: usize = 0;
    // each stream's ids come from its own history, so every row holds the bits its own `rows` call writes
    for (g.lanes, 0..) |l, k| {
        parts[k] = .{ .history = l.seq.history[0..], .tokens = g.ids[at..][0..l.rows], .out = g.emb[at * dims ..][0 .. l.rows * dims], .xs = g.xs[at * (dims / 32) ..][0 .. l.rows * (dims / 32)] };
        at += l.rows;
    }
    rowsMany(g.ngram, parts[0..g.lanes.len]) catch |err| {
        g.failed = err;
    };
    return true;
}

/// `Ngram.rows` over several streams' rows in one pool call; each stream's rows follow the one before it.
pub fn rowsMany(g: *const ple.Ngram, parts: []const Part) !void {
    const heads = g.hash.heads;
    const w = g.width();
    if (parts.len == 0 or parts.len > max_lanes) return error.UnexpectedTensor;
    var total: usize = 0;
    for (parts) |q| {
        if (q.history.len != g.hash.n - 1 or q.out.len != q.tokens.len * w or q.xs.len != q.out.len / 32) return error.UnexpectedTensor;
        total += q.tokens.len;
    }
    // the parts' outputs are one buffer's consecutive slices, so one fill covers all of them
    for (parts[1..], 0..) |q, k| if (q.out.ptr != parts[k].out.ptr + parts[k].out.len) return error.UnexpectedTensor;
    var hists: [max_lanes][ple.max_n]i64 = undefined;
    const items = total * heads;
    const ids = try g.gpa.alloc(i64, items);
    defer g.gpa.free(ids);
    var at: usize = 0;
    for (parts, 0..) |q, k| {
        const hist = &hists[k];
        @memcpy(hist[0..q.history.len], q.history);
        for (q.tokens) |token| {
            try ple.lastIds(g.hash, hist[0..q.history.len], @intCast(token), ids[at * heads ..][0..heads]);
            for (0..q.history.len - 1) |i| hist[i] = hist[i + 1];
            hist[q.history.len - 1] = token;
            at += 1;
        }
    }
    var out = parts[0].out;
    out.len = total * w;
    const n_threads = @min(ple_pool.max_parts, items / 4);
    if (g.format != .mlx and items > 1) {
        try @import("cuda_ple_gather.zig").run(g, ids, out);
    } else if (n_threads <= 1 or g.pool == null) {
        try g.fill(ids, out, 0, items);
    } else {
        const split: ple.Ngram.Split = .{ .g = g, .ids = ids, .out = out, .per = (items + n_threads - 1) / n_threads };
        try g.pool.?.run(n_threads, .{ .ctx = &split, .run = ple.Ngram.Split.part });
    }
    for (parts) |q| {
        for (0..q.tokens.len) |t| {
            const row = q.out[t * w ..][0..w];
            if (g.format == .mlx and g.scale != 1) {
                for (row) |*v| {
                    const f: f32 = @bitCast(@as(u32, v.*) << 16);
                    v.* = ple.toBf16(f * g.scale);
                }
            }
            ple.sumGroups(row, q.xs[t * (w / 32) ..][0 .. w / 32]);
        }
    }
    for (parts, 0..) |q, k| @memcpy(q.history, hists[k][0..q.history.len]);
}

const a = std.testing.allocator;
const io = std.testing.io;
const base = "layers.1.ple.ple_embedding.ngram_embedding";
const config =
    \\{"text_config":{"num_hidden_layers":2,"hidden_size":32,"vocab_size":100,"ple_layer_ids":[2],
    \\ "ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":13,"make_ngram_vocab_size_divisible_by":2,
    \\ "split_ngram_parts":2,"ple_embed_dim":2560,"eos_token_id":99}}
;

const Kind = enum { nvfp4, mlx };

const Fill = union(enum) { nibbles: usize, byte: u8, words: usize, bf16: u16 };

/// A two-shard table of the served shape (2,560 wide, 3-gram, 2 heads a gram), NVFP4 or the same rows as MLX 4-bit.
fn fixture(dir: std.Io.Dir, kind: Kind) !void {
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
    const h = try ple.Hash.init(.{ .vocab = 100, .ngram = 3, .per = 2, .base = 13, .divisor = 2, .shards = 2, .seed = 1234, .eos = 99, .ple_dim = dims, .ple_index = 0 });
    var header: std.Io.Writer.Allocating = .init(a);
    defer header.deinit();
    var index: std.Io.Writer.Allocating = .init(a);
    defer index.deinit();
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(a);
    try header.writer.writeByte('{');
    try index.writer.writeAll("{\"weight_map\":{");
    var first = true;
    for (0..2) |shard| {
        const rows: usize = @intCast(h.starts[shard + 1] - h.starts[shard]);
        const start: usize = @intCast(h.starts[shard]);
        switch (kind) {
            .nvfp4 => {
                try tensor(&header, &index, &data, &first, "model.language_model", shard, "weight", "U8", rows, dims / 8, .{ .nibbles = start });
                try tensor(&header, &index, &data, &first, "model.language_model", shard, "weight_scale", "F8_E4M3", rows, dims / 64, .{ .byte = 0x38 });
            },
            .mlx => {
                try tensor(&header, &index, &data, &first, "language_model.model", shard, "weight", "U32", rows, dims / 32, .{ .words = start });
                try tensor(&header, &index, &data, &first, "language_model.model", shard, "scales", "BF16", rows, dims / 128, .{ .bf16 = 0x3f00 });
                try tensor(&header, &index, &data, &first, "language_model.model", shard, "biases", "BF16", rows, dims / 128, .{ .bf16 = 0x3d00 });
            },
        }
    }
    // the table's own scale, which the MLX rows multiply and the NVFP4 rows require
    const scale_name = if (kind == .mlx) "language_model.model." ++ base ++ ".weight_scale" else "model.language_model." ++ base ++ ".weight_scale_2";
    const begin = data.items.len;
    try put(&data, u32, @bitCast(@as(f32, if (kind == .mlx) 0.75 else 0.375)));
    if (!first) {
        try header.writer.writeByte(',');
        try index.writer.writeByte(',');
    }
    try header.writer.print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[{d},{d}]}}", .{ scale_name, begin, data.items.len });
    try index.writer.print("\"{s}\":\"table.safetensors\"", .{scale_name});
    try header.writer.writeByte('}');
    try index.writer.writeAll("}}");
    var image: std.Io.Writer.Allocating = .init(a);
    defer image.deinit();
    const size = std.mem.toBytes(std.mem.nativeToLittle(u64, header.written().len));
    try image.writer.writeAll(&size);
    try image.writer.writeAll(header.written());
    try image.writer.writeAll(data.items);
    try dir.writeFile(io, .{ .sub_path = "table.safetensors", .data = image.written() });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });
}

fn tensor(header: *std.Io.Writer.Allocating, index: *std.Io.Writer.Allocating, data: *std.ArrayList(u8), first: *bool, pre: []const u8, shard: usize, part: []const u8, dtype: []const u8, rows: usize, cols: usize, fill: Fill) !void {
    const begin = data.items.len;
    for (0..rows) |row| {
        for (0..cols) |col| switch (fill) {
            .nibbles => |start| try data.append(a, @intCast((start + row + col) % 16 | (((start + row) * 3 + col) % 16) << 4)),
            .byte => |v| try data.append(a, v),
            .words => |start| try put(data, u32, @truncate((start + row) *% 2654435761 +% col *% 0x11111111)),
            .bf16 => |v| try put(data, u16, v),
        };
    }
    if (first.*) first.* = false else {
        try header.writer.writeByte(',');
        try index.writer.writeByte(',');
    }
    var name: [220]u8 = undefined;
    const full = try std.fmt.bufPrint(&name, "{s}.{s}.shard_{d}.{s}", .{ pre, base, shard, part });
    try header.writer.print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ full, dtype, rows, cols, begin, data.items.len });
    try index.writer.print("\"{s}\":\"table.safetensors\"", .{full});
}

fn put(data: *std.ArrayList(u8), comptime T: type, v: T) !void {
    var b: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &b, v, .little);
    try data.appendSlice(a, &b);
}

fn load(tmp: *std.testing.TmpDir, kind: Kind) !ple.Ngram {
    try fixture(tmp.dir, kind);
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(path);
    return ple.Ngram.init(a, io, path);
}

test "the one pooled call equals each stream's own rows call, byte for byte, on both table formats" {
    const Case = struct {
        history: [3][2]i64,
        tokens: [3][8]u32,
        rows: [3]usize,
        lanes: usize,
    };
    const cases = [_]Case{
        // two lanes of the same rows: every id is asked for twice in the one call
        .{ .history = .{ .{ 99, 99 }, .{ 99, 99 }, .{ 0, 0 } }, .tokens = .{ .{ 7, 12, 7, 0, 0, 0, 0, 0 }, .{ 7, 12, 7, 0, 0, 0, 0, 0 }, .{ 0, 0, 0, 0, 0, 0, 0, 0 } }, .rows = .{ 3, 3, 0 }, .lanes = 2 },
        // three lanes over different histories: the same token hashes elsewhere, and some ids still collide
        .{ .history = .{ .{ 99, 99 }, .{ 12, 99 }, .{ 40, 5 } }, .tokens = .{ .{ 7, 12, 3, 5, 40, 1, 9, 2 }, .{ 12, 7, 3, 9, 1, 40, 5, 2 }, .{ 5, 40, 5, 0, 0, 0, 0, 0 } }, .rows = .{ 8, 8, 3 }, .lanes = 3 },
        // one lane: the same one call, over one stream
        .{ .history = .{ .{ 99, 99 }, .{ 0, 0 }, .{ 0, 0 } }, .tokens = .{ .{ 7, 12, 7, 12, 7, 12, 7, 12 }, .{ 0, 0, 0, 0, 0, 0, 0, 0 }, .{ 0, 0, 0, 0, 0, 0, 0, 0 } }, .rows = .{ 8, 0, 0 }, .lanes = 1 },
    };
    for ([_]Kind{ .nvfp4, .mlx }) |kind| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var g = try load(&tmp, kind);
        defer g.deinit();
        const w = g.width();
        try std.testing.expectEqual(dims, w);
        for (cases) |c| {
            var total: usize = 0;
            for (c.rows[0..c.lanes]) |n| total += n;
            const want = try a.alloc(u16, total * w);
            defer a.free(want);
            const got = try a.alloc(u16, total * w);
            defer a.free(got);
            const want_xs = try a.alloc(f32, total * (w / 32));
            defer a.free(want_xs);
            const got_xs = try a.alloc(f32, total * (w / 32));
            defer a.free(got_xs);
            var want_hist = c.history;
            var got_hist = c.history;
            var at: usize = 0;
            for (0..c.lanes) |k| {
                const n = c.rows[k];
                try g.rows(want_hist[k][0 .. g.hash.n - 1], c.tokens[k][0..n], want[at * w ..][0 .. n * w], want_xs[at * (w / 32) ..][0 .. n * (w / 32)]);
                at += n;
            }
            const ids = try a.alloc(u32, total);
            defer a.free(ids);
            at = 0;
            for (0..c.lanes) |k| {
                @memcpy(ids[at..][0..c.rows[k]], c.tokens[k][0..c.rows[k]]);
                at += c.rows[k];
            }
            var lanes: [3]batch_state.Lane = undefined;
            for (0..c.lanes) |k| lanes[k] = .{ .seq = .{ .conv = &.{}, .state = &.{}, .kc = &.{}, .vc = &.{}, .ikc = &.{}, .pooled = &.{}, .tail = undefined, .history = &got_hist[k], .cap = 0, .scores = 0 }, .pos = 0, .rows = c.rows[k] };
            @memset(got, 0xdead);
            @memset(got_xs, @bitCast(@as(u32, 0xdeadbeef)));
            var gather: batch_state.Gather = .{ .ngram = &g, .history = &.{}, .ids = ids, .emb = got, .xs = got_xs, .io = io, .lanes = lanes[0..c.lanes] };
            try std.testing.expect(!earlyOn(&gather, false));
            try std.testing.expect(gather.failed == null);
            try std.testing.expectEqual(@as(u16, 0xdead), got[0]);
            try std.testing.expect(earlyOn(&gather, true));
            try std.testing.expect(gather.failed == null);
            try std.testing.expectEqualSlices(u16, want, got);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_xs), std.mem.sliceAsBytes(got_xs));
            for (0..c.lanes) |k| try std.testing.expectEqualSlices(i64, want_hist[k][0 .. g.hash.n - 1], got_hist[k][0 .. g.hash.n - 1]);
        }
    }
}

test "a pooled gather refuses parts whose outputs are not one buffer's consecutive slices" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var g = try load(&tmp, .nvfp4);
    defer g.deinit();
    const w = g.width();
    const tokens = [_]u32{ 7, 12 };
    const out = try a.alloc(u16, 8 * w);
    defer a.free(out);
    const xs = try a.alloc(f32, 8 * (w / 32));
    defer a.free(xs);
    var hist = [_]i64{ 99, 99 };
    const good: Part = .{ .history = &hist, .tokens = &tokens, .out = out[0 .. 2 * w], .xs = xs[0 .. 2 * (w / 32)] };
    try rowsMany(&g, &.{good});
    var gap_hist = [_]i64{ 99, 99 };
    const gap: Part = .{ .history = &gap_hist, .tokens = &tokens, .out = out[4 * w ..][0 .. 2 * w], .xs = xs[2 * (w / 32) ..][0 .. 2 * (w / 32)] };
    try std.testing.expectError(error.UnexpectedTensor, rowsMany(&g, &.{ good, gap }));
    var short_hist = [_]i64{99};
    const bad_history: Part = .{ .history = &short_hist, .tokens = &tokens, .out = out[0 .. 2 * w], .xs = xs[0 .. 2 * (w / 32)] };
    try std.testing.expectError(error.UnexpectedTensor, rowsMany(&g, &.{bad_history}));
    try std.testing.expectError(error.UnexpectedTensor, rowsMany(&g, &.{}));
}

test "a pooled pass takes 1 to 64 lanes, and the switch decides the rest" {
    try std.testing.expect(earlyLanes(true, 1));
    try std.testing.expect(earlyLanes(true, 64));
    try std.testing.expect(!earlyLanes(true, 65));
    try std.testing.expect(!earlyLanes(true, 0));
    try std.testing.expect(!earlyLanes(false, 4));
    try std.testing.expectEqual(@as(usize, 64), max_lanes);
}

test "the shared pass takes the pooled gather before its own rows call a stream" {
    const body = @embedFile("cuda_batch_gather.zig");
    const main = std.mem.indexOf(u8, body, "pub fn main(g: *Gather) void {") orelse return error.TestUnexpectedResult;
    const call = std.mem.indexOfPos(u8, body, main, "if (gather_lanes.early(g)) return;") orelse return error.TestUnexpectedResult;
    const per_lane = std.mem.indexOfPos(u8, body, main, "for (g.lanes) |l| {") orelse return error.TestUnexpectedResult;
    try std.testing.expect(call < per_lane);
}
