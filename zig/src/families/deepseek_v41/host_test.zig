//! Host tests of the loader side: release config, synthetic pack, 2.8 bpw widths, header dumps (TF_DSV41_HEADERS).

const std = @import("std");
const testing = std.testing;
const config = @import("config.zig");
const Config = config.Config;
const exl3 = @import("core").exl3_layout;
const Pack = @import("pack.zig").Pack;
const plan = @import("plan.zig");
const named = @import("named.zig");

test "the release config.json: topology, routing and rope" {
    const c = try Config.parse(testing.allocator, @embedFile("fixtures/config.json"));
    try testing.expectEqual(@as(u32, 129280), c.vocab);
    try testing.expectEqual(@as(u32, 43), c.allLayers());
    try testing.expectEqual(@as(f32, 1e-20), c.eps);
    try testing.expect(c.rope.yarn and c.rope.factor == 16 and c.rope.original_max_positions == 65536);
    try testing.expectEqual(@as(f64, 160000), c.rope.compress_theta);
    try testing.expectEqual(config.Mode.swa, c.mode(1));
    try testing.expectEqual(config.Mode.full, c.mode(2));
    try testing.expectEqual(config.Mode.reuse, c.mode(3));
    try testing.expectEqual(config.Mode.full, c.mode(20));
    try testing.expectEqual(config.Mode.reindex, c.mode(24));
    try testing.expectEqual(config.Mode.swa, c.mode(41));
    try testing.expectEqual(@as(?u32, 14), c.kvSource(19));
    try testing.expectEqual(@as(?u32, 20), c.kvSource(39));
    try testing.expectEqual(@as(?u32, 36), c.indexSource(39));
    try testing.expect(c.isCandidateSource(20) and c.usesCandidates(24) and !c.usesCandidates(20));
    try testing.expectEqual(@as(u32, 128), c.expertsOf(40).count);
    try testing.expectEqual(@as(u32, 3), c.expertsOf(40).topk);
    try testing.expectEqual(@as(u32, 24), c.hashCols());
    try testing.expect(c.isEngram(14) and !c.isEngram(2));
    try testing.expectEqual(@as(u32, 1), c.eos);
    // the release's values equal the Config defaults
    const d: Config = .{};
    try testing.expectEqualSlices(u32, d.compress_ratios.items(), c.compress_ratios.items());
    try testing.expectEqualSlices(u64, d.engram_rows.items(), c.engram_rows.items());
    try testing.expectError(error.UnsupportedScoring, Config.parse(testing.allocator, "{\"scoring_func\": \"sigmoid\"}"));
    try testing.expectError(error.BadConfig, Config.parse(testing.allocator, "{\"compress_ratios\": [0, 3]}"));
    try config.tiny().validate();
}

/// Header JSON of a pack for `c`, named as the release packs are (their index, pattern by pattern), header only.
fn tinyPack(a: std.mem.Allocator, c: *const Config, extra: []const u8, skip: []const u8) ![]u8 {
    var n: u64 = 0;
    return tinyPackLen(a, c, extra, skip, &n);
}

/// `tinyPack`, and the data bytes its header describes.
fn tinyPackLen(a: std.mem.Allocator, c: *const Config, extra: []const u8, skip: []const u8, data_len: *u64) ![]u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    errdefer w.deinit();
    var at: u64 = 0;
    var first = true;
    const T = struct {
        fn put(wr: *std.Io.Writer, pos: *u64, f: *bool, sk: []const u8, name: []const u8, dtype: []const u8, shape: []const u64) !void {
            if (sk.len > 0 and std.mem.eql(u8, name, sk)) return;
            var n: u64 = if (std.mem.eql(u8, dtype, "F32") or std.mem.eql(u8, dtype, "I32")) 4 else 2;
            for (shape) |d| n *= d;
            try wr.print("{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (f.*) "" else ",", name, dtype });
            for (shape, 0..) |d, i| try wr.print("{s}{d}", .{ if (i == 0) "" else ",", d });
            try wr.print("],\"data_offsets\":[{d},{d}]}}", .{ pos.*, pos.* + n });
            pos.* += n;
            f.* = false;
        }
        fn group(wr: *std.Io.Writer, pos: *u64, f: *bool, sk: []const u8, p: []const u8, k: u64, n: u64, words: u64) !void {
            var b: [160]u8 = undefined;
            try put(wr, pos, f, sk, try std.fmt.bufPrint(&b, "{s}.trellis", .{p}), "I16", &.{ k / 16, n / 16, words });
            try put(wr, pos, f, sk, try std.fmt.bufPrint(&b, "{s}.suh", .{p}), "F16", &.{k});
            try put(wr, pos, f, sk, try std.fmt.bufPrint(&b, "{s}.svh", .{p}), "F16", &.{n});
            try put(wr, pos, f, sk, try std.fmt.bufPrint(&b, "{s}.mul1", .{p}), "I32", &.{});
        }
    };
    const wr = &w.writer;
    try wr.writeAll("{");
    const D: u64 = c.hidden;
    var nb: [160]u8 = undefined;
    var l: u32 = 0;
    while (l < c.allLayers()) : (l += 1) {
        var pb: [32]u8 = undefined;
        const p = if (l < c.layers) try std.fmt.bufPrint(&pb, "layers.{d}", .{l}) else try std.fmt.bufPrint(&pb, "mtp.{d}", .{l - c.layers});
        const E: u64 = if (l < c.layers) c.experts else c.dspark_experts;
        inline for (.{ "attn_norm.weight", "ffn_norm.weight" }) |s| try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}." ++ s, .{p}), "BF16", &.{D});
        inline for (.{ "attn", "ffn" }) |s| {
            try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.hc_" ++ s ++ "_fn", .{p}), "F32", &.{ 24, 4 * D });
            try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.hc_" ++ s ++ "_base", .{p}), "F32", &.{24});
            try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.hc_" ++ s ++ "_scale", .{p}), "F32", &.{3});
        }
        try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.q_norm.weight", .{p}), "BF16", &.{c.q_lora});
        try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.kv_norm.weight", .{p}), "BF16", &.{c.head_dim});
        try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.attn_sink", .{p}), "F32", &.{c.heads});
        try T.group(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.wq_a", .{p}), D, c.q_lora, 64);
        try T.group(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.wkv", .{p}), D, c.head_dim, 80);
        try T.group(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.wq_b", .{p}), c.q_lora, c.heads * c.head_dim, 64);
        var g: u32 = 0;
        while (g < c.o_groups) : (g += 1) try T.group(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.wo_a.slice.{d}", .{ p, g }), c.heads / c.o_groups * c.head_dim, c.o_lora, 64);
        try T.group(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.wo_b", .{p}), c.o_groups * c.o_lora, D, 64);
        if (l < c.layers and c.kv_sources.contains(l)) {
            try T.group(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.compressor.wkv", .{p}), D, c.head_dim, 80);
            if (c.compressRatio(l) == 2) {
                try T.group(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.compressor.wgate", .{p}), D, c.head_dim, 80);
            }
            try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.compressor.norm.weight", .{p}), "BF16", &.{c.head_dim});
            try T.group(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.indexer.wk", .{p}), c.head_dim, c.index_dim, 128);
            try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.indexer.k_norm.weight", .{p}), "BF16", &.{c.index_dim});
        }
        if (l < c.layers and c.index_sources.contains(l)) {
            try T.group(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.indexer.wq_b", .{p}), c.q_lora, c.index_heads * c.index_dim, 80);
            try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.attn.indexer.weights_proj.weight", .{p}), "BF16", &.{ c.index_heads, D });
        }
        if (c.engram_layers.contains(l)) {
            try T.group(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.engram.wkv", .{p}), c.hashCols() * c.engram_head_dim, (c.engram_max_ngram + 1) * D, 64);
            try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.engram.q_weight", .{p}), "BF16", &.{ c.hc_mult, D });
            try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.engram.k_weight", .{p}), "BF16", &.{ c.hc_mult, D });
        }
        try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.ffn.gate.weight", .{p}), "BF16", &.{ E, D });
        try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.ffn.gate.bias", .{p}), "F32", &.{E});
        try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.ffn.gate.bias_vl", .{p}), "F32", &.{E});
        inline for (.{ "w1", "w2", "w3" }) |m| {
            const kn: [2]u64 = if (comptime std.mem.eql(u8, m, "w2")) .{ c.expert_width, D } else .{ D, c.expert_width };
            try T.group(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.ffn.shared_experts." ++ m, .{p}), kn[0], kn[1], 80);
            var e: u64 = 0;
            while (e < E) : (e += 1) {
                // block 2: expert 1's down projection at 3 bits, expert 3's gate at 3 bits (a gate/up width mismatch)
                const words: u64 = if (l == 2 and ((e == 1 and comptime std.mem.eql(u8, m, "w2")) or (e == 3 and comptime std.mem.eql(u8, m, "w1")))) 48 else 32;
                try T.group(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "{s}.ffn.experts.{d}." ++ m, .{ p, e }), kn[0], kn[1], words);
            }
        }
    }
    try T.put(wr, &at, &first, skip, "embed.weight", "BF16", &.{ c.vocab, D });
    try T.put(wr, &at, &first, skip, "norm.weight", "BF16", &.{D});
    try T.group(wr, &at, &first, skip, "head", D, c.vocab, 96);
    const last = c.mtp_layers - 1;
    try T.group(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "mtp.{d}.main_proj", .{last}), 3 * D, D, 64);
    inline for (.{ "main_norm", "norm" }) |s| try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "mtp.{d}." ++ s ++ ".weight", .{last}), "BF16", &.{D});
    inline for (.{ "embed", "head" }) |s| try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "mtp.{d}.markov_head." ++ s ++ ".weight", .{last}), "BF16", &.{ c.vocab, c.dspark_markov_rank });
    try T.put(wr, &at, &first, skip, try std.fmt.bufPrint(&nb, "mtp.{d}.confidence_head.proj.weight", .{last}), "F32", &.{ 1, D + c.dspark_markov_rank });
    try T.put(wr, &at, &first, skip, "vision.norm.weight", "BF16", &.{D});
    try T.put(wr, &at, &first, skip, "image_start", "BF16", &.{D});
    if (extra.len > 0) try T.put(wr, &at, &first, skip, extra, "BF16", &.{4});
    try wr.writeAll("}");
    data_len.* = at;
    return w.toOwnedSlice();
}

fn tinyPlan(a: std.mem.Allocator, c: *const Config, extra: []const u8, skip: []const u8, rank: u32) !plan.Plan {
    return tinyPlanOf(a, c, c, extra, skip, rank);
}

/// A pack written for `stored`, planned with `c`.
fn tinyPlanOf(a: std.mem.Allocator, c: *const Config, stored: *const Config, extra: []const u8, skip: []const u8, rank: u32) !plan.Plan {
    const json = try tinyPack(a, stored, extra, skip);
    defer a.free(json);
    var p = Pack.init(a);
    defer p.deinit();
    try p.addHeader("model.safetensors", json, std.math.maxInt(u32));
    return plan.build(a, c, &p, .{ .rank = rank, .world = 2 });
}

test "a synthetic pack: both ranks plan every block, the parts tile the full tensors, nothing is left" {
    const a = testing.allocator;
    const c = config.tiny();
    var r0 = try tinyPlan(a, &c, "", "", 0);
    defer r0.deinit();
    var r1 = try tinyPlan(a, &c, "", "", 1);
    defer r1.deinit();
    try testing.expectEqual(@as(usize, 0), r0.leftovers.len);
    try testing.expectEqual(@as(usize, 10), r0.layers.len);
    const l2 = &r0.layers[2];
    try testing.expectEqual(config.Mode.full, l2.mode);
    try testing.expect(l2.experts[1].layout.kind == .ragged and l2.experts[0].layout.kind == .ragged and l2.experts[2].layout.kind == .stack);
    try testing.expectEqual(@as(u32, 1), l2.gate_up_mismatch);
    try testing.expectEqual(@as(u32, 2), l2.experts[1].layout.widths());
    try testing.expect(r0.layers[3].experts[1].layout.kind == .stack);
    try testing.expectEqual(@as(usize, 4), r0.layers[8].experts[0].layout.count());
    // every split part on the two ranks adds up to the stored tensor
    for (r0.layers, r1.layers) |x, y| for (x.groups, y.groups) |g0, g1| {
        if (g0.split == .whole) {
            try testing.expectEqual(g0.full.trellisBytes(), g0.part.trellis.bytes());
        } else try testing.expectEqual(g0.full.trellisBytes(), g0.part.trellis.bytes() + g1.part.trellis.bytes());
    };
    try testing.expectEqual(r0.head.?.full.trellisBytes(), r0.head.?.part.trellis.bytes() + r1.head.?.part.trellis.bytes());
    try testing.expectEqualStrings("mtp.1.main_proj", r0.dspark.?.main_proj.prefix);
    try testing.expect(r0.dspark.?.confidence != null);
    try testing.expectEqual(r0.bytes(), r1.bytes());
    // the embedding: half the vocabulary rows on each rank
    try testing.expectEqual(@as(u64, 128 * 256 * 2), r0.top[0].bytes);
}

test "a synthetic pack: unknown names are leftovers, missing ones and wrong shapes are refused" {
    const a = testing.allocator;
    const c = config.tiny();
    var p = try tinyPlan(a, &c, "layers.3.attn.mystery", "", 0);
    defer p.deinit();
    try testing.expectEqual(@as(usize, 1), p.leftovers.len);
    try testing.expectEqualStrings("layers.3.attn.mystery", p.leftovers[0]);
    try testing.expectError(error.MissingTensor, tinyPlan(a, &c, "", "layers.5.ffn.experts.6.w3.svh", 0));
    try testing.expectError(error.MissingTensor, tinyPlan(a, &c, "", "layers.4.attn.indexer.k_norm.weight", 1));
    var wide = c;
    wide.q_lora = 48;
    try testing.expectError(error.UnexpectedShape, tinyPlanOf(a, &wide, &c, "", "", 0));
}

test "a synthetic pack on disk: every planned tensor named in the reference loader's form" {
    const a = testing.allocator;
    const io = testing.io;
    const c = config.tiny();
    var n: u64 = 0;
    const json = try tinyPackLen(a, &c, "", "", &n);
    defer a.free(json);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    {
        var f = try tmp.dir.createFile(io, "model.safetensors", .{});
        defer f.close(io);
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, json.len, .little);
        try f.writePositionalAll(io, &len, 0);
        try f.writePositionalAll(io, json, 8);
        try f.setLength(io, 8 + json.len + n);
    }
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var p = try Pack.open(a, io, dir);
    defer p.deinit();
    var pl = try plan.build(a, &c, &p, .{ .rank = 1, .world = 2 });
    defer pl.deinit();
    var b: named.Builder = .{ .gpa = a, .io = io, .pack = &p };
    defer b.deinit();
    try b.all(&pl, c.o_groups);
    try testing.expectEqual(named.Kind.i16, b.find("L2.moe.w2.data").?.kind);
    try testing.expect(b.find("L3.moe.w2.trellis") != null and b.find("L3.moe.w2.data") == null);
    var schema: named.Builder = .{ .gpa = a, .io = io, .pack = &p, .metadata_only = true };
    defer schema.deinit();
    try schema.all(&pl, c.o_groups);
    try testing.expectEqual(b.out.items.len, schema.out.items.len);
    for (b.out.items, schema.out.items) |actual, expected| {
        try testing.expectEqualStrings(actual.name, expected.name);
        try testing.expectEqual(actual.kind, expected.kind);
        try testing.expectEqual(actual.rank, expected.rank);
        try testing.expectEqual(actual.shape, expected.shape);
        try testing.expectEqual(@as(usize, 0), expected.bytes.len);
    }
    try testing.expectEqualSlices(usize, &.{ 8, 8, 16, 32 }, b.find("L3.moe.w2.trellis").?.shape[0..4]);
    const qk = b.find("L1.engram.qk").?;
    try testing.expect(qk.kind == .f32 and qk.bytes.len == 4 * c.hc_mult * c.hidden);
    try testing.expectEqual(named.Kind.bf16, b.find("L0.moe.gate").?.kind);
    try testing.expectEqual(named.Kind.f32, b.find("L2.attn.ix_wp").?.kind);
    // rank 1 of 2: half the heads' sinks, half the vocabulary rows, wo_a's one local group
    try testing.expectEqual(@as(usize, 4 * c.heads / 2), b.find("L4.attn.sink").?.bytes.len);
    try testing.expectEqual(@as(usize, 2 * c.vocab / 2 * c.hidden), b.find("embed").?.bytes.len);
    try testing.expect(b.find("L4.attn.wo_a.0.svh") != null and b.find("L4.attn.wo_a.1.svh") == null);
    try testing.expect(b.find("head.trellis") != null and b.find("norm") != null);
}

/// One fixture histogram ("4:383,6:1") as rank-0 parts of a TP=2 projection with full shape `kn`.
fn parts(a: std.mem.Allocator, text: []const u8, kn: [2]u32, split: exl3.Split) ![]exl3.Part {
    var out: std.ArrayList(exl3.Part) = .empty;
    errdefer out.deinit(a);
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |item| {
        const colon = std.mem.indexOfScalar(u8, item, ':') orelse return error.BadFixture;
        const k2 = try std.fmt.parseInt(u32, item[0..colon], 10);
        const n = try std.fmt.parseInt(u32, item[colon + 1 ..], 10);
        const g = try exl3.Group.fromShape(&.{ kn[0] / 16, kn[1] / 16, 8 * k2 });
        try out.appendNTimes(a, try g.part(split, 0, 2), n);
    }
    return out.toOwnedSlice(a);
}

test "the 2.8 bpw EXL3 pack: rank 0's routed + shared trellis bytes of every block match the reference loader's listing" {
    const a = testing.allocator;
    const c: Config = .{};
    var lines = std.mem.tokenizeScalar(u8, @embedFile("fixtures/widths-exl3-2p8.txt"), '\n');
    var blocks: u32 = 0;
    var ragged: u32 = 0;
    while (lines.next()) |line| {
        if (line[0] == '#') continue;
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const layer = try std.fmt.parseInt(u32, f.next().?, 10);
        var total: u64 = 0;
        const projs = [_]struct { kn: [2]u32, split: exl3.Split }{ .{ .kn = .{ c.hidden, c.expert_width }, .split = .col }, .{ .kn = .{ c.expert_width, c.hidden }, .split = .row }, .{ .kn = .{ c.hidden, c.expert_width }, .split = .col } };
        for (projs) |pj| {
            const ps = try parts(a, f.next().?, pj.kn, pj.split);
            defer a.free(ps);
            try testing.expectEqual(@as(usize, c.expertsOf(layer).count), ps.len);
            var lay = try exl3.Layout.of(a, ps);
            defer lay.deinit(a);
            ragged += @intFromBool(lay.kind == .ragged);
            total += lay.trellisBytes();
        }
        var sh = std.mem.tokenizeScalar(u8, f.next().?, ',');
        for (projs) |pj| {
            const g = try exl3.Group.fromShape(&.{ pj.kn[0] / 16, pj.kn[1] / 16, 8 * try std.fmt.parseInt(u32, sh.next().?, 10) });
            total += (try g.part(pj.split, 0, 2)).trellis.bytes();
        }
        try testing.expectEqual(try std.fmt.parseInt(u64, f.next().?, 10), total);
        blocks += 1;
    }
    try testing.expectEqual(@as(u32, 43), blocks);
    try testing.expect(ragged > 40);
}

test "real pack headers (TF_DSV41_HEADERS: model.safetensors.index.json and <shard>.hdr dumps): planned blocks" {
    const dir = testing.environ.getPosix("TF_DSV41_HEADERS") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = testing.io;
    var p = Pack.init(a);
    defer p.deinit();
    const cwd = std.Io.Dir.cwd();
    const index = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    defer a.free(index);
    const text = try cwd.readFileAlloc(io, index, a, .limited(1 << 28));
    defer a.free(text);
    try p.useIndex(text);
    var d = try cwd.openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var it = d.iterate();
    var blocks: std.ArrayList(u32) = .empty;
    defer blocks.deinit(a);
    while (try it.next(io)) |e| {
        if (!std.mem.endsWith(u8, e.name, ".hdr")) continue;
        const bytes = try d.readFileAlloc(io, e.name, a, .limited(1 << 28));
        defer a.free(bytes);
        const shard = e.name[0 .. e.name.len - 4];
        try p.addHeaderDump(shard, bytes);
        if (std.mem.startsWith(u8, shard, "layer-")) try blocks.append(a, try std.fmt.parseInt(u32, shard[6..8], 10));
    }
    const c: Config = .{};
    for ([_]u32{ 0, 1 }) |rank| {
        var pl = try plan.build(a, &c, &p, .{ .rank = rank, .world = 2, .blocks = blocks.items });
        defer pl.deinit();
        for (pl.layers) |*l| {
            const h = l.experts[1].layout.histogram();
            std.debug.print("rank {d} {s} {t}: down {t} K2 4/6/8 = {d}/{d}/{d}, gate/up mismatches {d}, experts {d} B, block {d} B\n", .{ rank, l.prefix, l.mode, l.experts[1].layout.kind, h[4], h[6], h[8], l.gate_up_mismatch, l.expertBytes(), l.bytes() });
        }
    }
}
