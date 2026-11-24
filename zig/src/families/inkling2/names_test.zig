//! Weight maps and layer plans against safetensors headers (fixtures fold them).

const std = @import("std");
const config = @import("config.zig");
const config_test = @import("config_test.zig");
const st = @import("core").safetensors;
const names = @import("names.zig");
const plan = @import("plan.zig");

const t = std.testing;
const model_fixture = @embedFile("fixtures/tiny_model_header.txt");
const mtp_fixture = @embedFile("fixtures/tiny_mtp_header.txt");
const mini_model_fixture = @embedFile("fixtures/mini_model_header.txt");
const mini_mtp_fixture = @embedFile("fixtures/mini_mtp_header.txt");

fn byName(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// The tag safetensors writes for the four dtypes the fixtures hold.
fn dtypeTag(d: st.DType) []const u8 {
    return switch (d) {
        .bf16 => "BF16",
        .f32 => "F32",
        .f8_e4m3 => "F8_E4M3",
        .u8 => "U8",
        else => unreachable,
    };
}

/// Each `{L}` line over its layer ranges; the header line holds the tag, the tensor count and the sha256 of the lines.
pub fn unfold(arena: std.mem.Allocator, text: []const u8, tag: []const u8) !st.Header {
    var out: st.Header = .empty;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var head = std.mem.tokenizeScalar(u8, lines.next().?, ' ');
    if (!std.mem.eql(u8, head.next().?, tag)) return error.BadFixture;
    const count = try std.fmt.parseInt(usize, head.next().?, 10);
    const digest = head.next().?;
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const pat = f.next().?;
        const dtype = st.DType.parse(f.next().?).?;
        var e: st.Entry = .{ .dtype = dtype, .rank = 0, .shape = @splat(1), .begin = 0, .end = 0 };
        var dims = std.mem.tokenizeScalar(u8, f.next().?, 'x');
        while (dims.next()) |d| : (e.rank += 1) e.shape[e.rank] = try std.fmt.parseInt(usize, d, 10);
        const layers = f.next() orelse {
            try out.put(arena, pat, e);
            continue;
        };
        var spans = std.mem.tokenizeScalar(u8, layers, ',');
        while (spans.next()) |span| {
            const dash = std.mem.indexOfScalar(u8, span, '-');
            const lo = try std.fmt.parseInt(u32, if (dash) |d| span[0..d] else span, 10);
            const hi = if (dash) |d| try std.fmt.parseInt(u32, span[d + 1 ..], 10) else lo;
            for (lo..hi + 1) |l| {
                var b: [8]u8 = undefined;
                const named = try std.mem.replaceOwned(u8, arena, pat, "{L}", try std.fmt.bufPrint(&b, "{d}", .{l}));
                try out.put(arena, named, e);
            }
        }
    }
    if (out.count() != count) return error.BadFixture;
    const sorted = try arena.dupe([]const u8, out.keys());
    std.mem.sort([]const u8, sorted, {}, byName);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (sorted, 0..) |name, i| {
        if (i > 0) h.update("\n");
        const e = out.get(name).?;
        var shape: std.ArrayList(u8) = .empty;
        for (e.shape[0..e.rank], 0..) |d, k| try shape.print(arena, "{s}{d}", .{ if (k > 0) "x" else "", d });
        h.update(try std.fmt.allocPrint(arena, "{s} {s} {s}", .{ name, dtypeTag(e.dtype), shape.items }));
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(d, .lower), digest)) return error.BadFixture;
    return out;
}

fn load(a: std.mem.Allocator) !config.Config {
    var why: config.Why = .{};
    return config.parse(a, config_test.tiny, &why);
}

fn loadMini(a: std.mem.Allocator) !config.Config {
    var why: config.Why = .{};
    return config.parse(a, config_test.mini, &why);
}

test "the folded fixtures unfold to the 404 and 84 tensors of tiny's two files, digests equal" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try t.expectEqual(@as(usize, 404), (try unfold(arena.allocator(), model_fixture, "inkling2-tiny-model")).count());
    try t.expectEqual(@as(usize, 84), (try unfold(arena.allocator(), mtp_fixture, "inkling2-tiny-mtp")).count());
}

test "Mini's folded fixtures unfold to the 1244 and 54 tensors of its two files, digests equal" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try t.expectEqual(@as(usize, 1244), (try unfold(arena.allocator(), mini_model_fixture, "inkling2-mini-model")).count());
    try t.expectEqual(@as(usize, 54), (try unfold(arena.allocator(), mini_mtp_fixture, "inkling2-mini-mtp")).count());
}

test "the map names exactly tiny's backbone and vision tensors, with their dtypes and shapes" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try load(a);
    const entries = try unfold(a, model_fixture, "inkling2-tiny-model");
    const r = try names.check(a, &c, entries, .model);
    try t.expectEqual(@as(usize, 404), r.expected);
    try t.expectEqual(@as(usize, 0), r.missing.items.len);
    try t.expectEqual(@as(usize, 0), r.wrong.items.len);
    try t.expectEqual(@as(usize, 0), r.unknown.items.len);
}

test "the map names exactly tiny's draft head tensors" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try load(a);
    const r = try names.check(a, &c, try unfold(a, mtp_fixture, "inkling2-tiny-mtp"), .mtp);
    try t.expectEqual(@as(usize, 84), r.expected);
    try t.expect(r.clean());
}

test "Mini's config parses and the map names exactly its model and draft tensors" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try loadMini(a);
    const model = try names.check(a, &c, try unfold(a, mini_model_fixture, "inkling2-mini-model"), .model);
    try t.expectEqual(@as(usize, 1244), model.expected);
    try t.expect(model.clean());
    const mtp = try names.check(a, &c, try unfold(a, mini_mtp_fixture, "inkling2-mini-mtp"), .mtp);
    try t.expectEqual(@as(usize, 54), mtp.expected);
    try t.expect(mtp.clean());
}

test "a missing tensor, a wrong shape, a wrong dtype and an unknown tensor are each reported by name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try load(a);
    const all = try unfold(a, model_fixture, "inkling2-tiny-model");
    var kept: st.Header = .empty;
    for (all.keys(), all.values()) |name, e| {
        var x = e;
        if (std.mem.eql(u8, name, "model.llm.layers.3.attn.wq_du.weight")) continue; // missing
        if (std.mem.eql(u8, name, "model.llm.layers.4.attn.wk_dv.weight")) x.shape[0] += 1; // shape
        if (std.mem.eql(u8, name, "model.llm.layers.2.mlp.gate.bias")) x.dtype = .bf16; // dtype
        try kept.put(a, name, x);
    }
    try kept.put(a, "model.llm.layers.0.attn.rope_freqs", kept.values()[0]);
    const r = try names.check(a, &c, kept, .model);
    try t.expectEqualStrings("model.llm.layers.3.attn.wq_du.weight", r.missing.items[0]);
    try t.expectEqual(@as(usize, 1), r.missing.items.len);
    try t.expectEqualStrings("model.llm.layers.2.mlp.gate.bias", r.wrong.items[0]);
    try t.expectEqualStrings("model.llm.layers.4.attn.wk_dv.weight", r.wrong.items[1]);
    try t.expectEqualStrings("model.llm.layers.0.attn.rope_freqs", r.unknown.items[0]);
    try t.expect(!r.clean());
}

test "core's header reader feeds the map: a wrong dtype, a wrong rank and an extra tensor are reported by name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try load(a);
    const json =
        \\{"__metadata__":{"format":"pt"},"mask_embed":{"dtype":"F16","shape":[128],"data_offsets":[0,256]},
        \\"norm.weight":{"dtype":"BF16","shape":[128,1],"data_offsets":[256,512]},"extra":{"dtype":"U8","shape":[],"data_offsets":[512,513]}}
    ;
    const r = try names.check(a, &c, try st.parseHeader(a, json, 513), .mtp);
    try t.expectEqual(@as(usize, 84), r.expected);
    try t.expectEqual(@as(usize, 82), r.missing.items.len);
    try t.expectEqualStrings("mask_embed", r.wrong.items[0]);
    try t.expectEqualStrings("norm.weight", r.wrong.items[1]);
    try t.expectEqual(@as(usize, 2), r.wrong.items.len);
    try t.expectEqualStrings("extra", r.unknown.items[0]);
    try t.expectEqual(@as(usize, 1), r.unknown.items.len);
}

test "merging the headers of several files keeps every tensor and refuses a name held twice" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const one = try st.parseHeader(a, "{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1);
    const two = try st.parseHeader(a, "{\"b\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1);
    var all: st.Header = .empty;
    try names.merge(a, &all, one);
    try names.merge(a, &all, two);
    try t.expectEqual(@as(usize, 2), all.count());
    try t.expectError(error.DuplicateName, names.merge(a, &all, one));
}

test "expert precision follows each layer's tensors: NVFP4 on layers 0 to 5, MXFP8 on 6 to 11" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const entries = try unfold(a, model_fixture, "inkling2-tiny-model");
    var prec: [12]plan.Precision = undefined;
    try names.precisions(entries, 12, &prec);
    for (prec, 0..) |p, i| try t.expectEqual(if (i < 6) plan.Precision.nvfp4 else plan.Precision.mxfp8, p);
    var bad = try entries.clone(a);
    bad.getPtr("model.llm.layers.7.mlp.experts.w13_weight").?.dtype = .bf16;
    try t.expectError(error.ExpertFormat, names.precisions(bad, 12, &prec));
}

test "Mini's expert precision is NVFP4 on layers 0 to 19 and MXFP8 on 20 to 23" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const entries = try unfold(arena.allocator(), mini_model_fixture, "inkling2-mini-model");
    var prec: [24]plan.Precision = undefined;
    try names.precisions(entries, 24, &prec);
    for (prec, 0..) |p, i| try t.expectEqual(if (i < 20) plan.Precision.nvfp4 else plan.Precision.mxfp8, p);
}

test "the plan: global layers 5 and 11, blocks of six, three attention queries before each global layer" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try load(a);
    var prec: [12]plan.Precision = undefined;
    try names.precisions(try unfold(a, model_fixture, "inkling2-tiny-model"), 12, &prec);
    const p = try plan.build(&c.text, &prec);
    try t.expectEqual(@as(u32, 2), p.count(.global));
    try t.expectEqual(@as(u32, 10), p.count(.local));
    for (0..12) |i| try t.expectEqual(i == 5 or i == 11, p.at(@intCast(i)).kind == .global);
    try t.expect(p.at(0).boundary and p.at(6).boundary and !p.at(5).boundary);
    try t.expectEqual(@as(u8, 0), p.at(0).attn_queries);
    try t.expectEqual(@as(u8, 3), p.at(5).attn_queries);
    try t.expectEqual(@as(u8, 1), p.at(6).attn_queries);
    // layer 0: no attention input, the MLP input sees its own input as a block and the partial output
    try t.expectEqual(@as(u32, 2), p.at(0).mlp_sources);
    try t.expectEqual(@as(u32, 2), p.at(1).attn_sources);
    try t.expectEqual(@as(u32, 3), p.at(6).mlp_sources);
    try t.expectEqual(@as(u32, 3), p.at(11).attn_sources);
    try t.expectEqual(@as(u32, 3), p.final_sources);
    try t.expectError(error.PrecisionCount, plan.build(&c.text, prec[0..11]));
    try t.expectEqual(@as(u32, 12), p.n_visits);
    for (0..p.n_visits) |i| {
        const v = p.visit(@intCast(i));
        try t.expectEqual(@as(u32, @intCast(i)), v.layer);
        try t.expectEqual(@as(u32, @intCast(i)), v.cache_slot);
        try t.expect(!v.hasMessages());
    }
}

test "Mini's visit plan repeats only the core, gives all 48 visits private state and banks the previous pass" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try loadMini(a);
    var prec: [24]plan.Precision = undefined;
    try names.precisions(try unfold(a, mini_model_fixture, "inkling2-mini-model"), 24, &prec);
    const p = try plan.build(&c.text, &prec);
    try t.expectEqual(@as(u32, 48), p.n_visits);
    try t.expectEqual(@as(u32, 8), p.visitCount(.global));
    try t.expectEqual(@as(u32, 40), p.visitCount(.local));
    for (0..p.n_visits) |i| {
        const v = p.visit(@intCast(i));
        const layer: u32 = if (i < 6) @intCast(i) else if (i < 42) 6 + @as(u32, @intCast((i - 6) % 12)) else @intCast(i - 24);
        const pass: u32 = if (i >= 6 and i < 42) 1 + @as(u32, @intCast((i - 6) / 12)) else 0;
        try t.expectEqual(layer, v.layer);
        try t.expectEqual(pass, v.pass);
        try t.expectEqual(@as(u32, @intCast(i)), v.cache_slot);
        try t.expectEqual(pass >= 2, v.hasMessages());
        const previous: ?u32 = if (pass == 2) 6 else if (pass == 3) 18 else null;
        try t.expectEqual(previous, v.previous_core_start);
    }
    try t.expectEqual(@as(u32, 5), p.final_sources);
}

/// A file's header through the core reader; the files may hold the header alone, so no data length is checked.
fn headerOf(a: std.mem.Allocator, dir: []const u8, file: []const u8) !st.Header {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, file });
    const f = try std.Io.Dir.cwd().openFile(t.io, path, .{});
    defer f.close(t.io);
    var prefix: [8]u8 = undefined;
    if (try f.readPositionalAll(t.io, &prefix, 0) != 8) return error.BadSafetensors;
    const size = std.math.cast(usize, try f.length(t.io)) orelse return error.BadSafetensors;
    const json = try a.alloc(u8, try st.headerSize(std.mem.readInt(u64, &prefix, .little), size));
    if (try f.readPositionalAll(t.io, json, 8) != json.len) return error.BadSafetensors;
    return st.parseHeader(a, json, std.math.maxInt(usize));
}

test "tiny-random's real model.safetensors and mtp.safetensors headers match the map (TF_INKLING2_TINY=<dir>)" {
    const dir = std.mem.span(std.c.getenv("TF_INKLING2_TINY") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, config_test.tiny, &why);
    const model = try names.check(a, &c, try headerOf(a, dir, "model.safetensors"), .model);
    try t.expectEqual(@as(usize, 404), model.expected);
    try t.expect(model.clean());
    const mtp = try names.check(a, &c, try headerOf(a, dir, "mtp.safetensors"), .mtp);
    try t.expectEqual(@as(usize, 84), mtp.expected);
    try t.expect(mtp.clean());
}

test "the real checkpoint's 24 shard headers and mtp.safetensors match the map (TF_INKLING2_REAL=<dir>)" {
    const dir = std.mem.span(std.c.getenv("TF_INKLING2_REAL") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, config_test.real, &why);
    var all: st.Header = .empty;
    for (1..25) |i| {
        const file = try std.fmt.allocPrint(a, "model-{d:0>5}-of-00024.safetensors", .{i});
        try names.merge(a, &all, try headerOf(a, dir, file));
    }
    const model = try names.check(a, &c, all, .model);
    try t.expectEqual(@as(usize, 1770), model.expected);
    try t.expect(model.clean());
    var prec: [config.max_layers]plan.Precision = undefined;
    try names.precisions(all, c.text.layers, &prec);
    for (prec[0..48], 0..) |p, i| try t.expectEqual(if (i < 40) plan.Precision.nvfp4 else plan.Precision.mxfp8, p);
    const mtp = try names.check(a, &c, try headerOf(a, dir, "mtp.safetensors"), .mtp);
    try t.expectEqual(@as(usize, 84), mtp.expected);
    try t.expect(mtp.clean());
}
