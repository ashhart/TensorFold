//! The weight map and layer plan against the tiny-random checkpoint's safetensors headers (fixtures fold them).
const std = @import("std");
const config = @import("config.zig");
const config_test = @import("config_test.zig");
const header = @import("header.zig");
const names = @import("names.zig");
const plan = @import("plan.zig");

const t = std.testing;
const model_fixture = @embedFile("fixtures/tiny_model_header.txt");
const mtp_fixture = @embedFile("fixtures/tiny_mtp_header.txt");

/// Each `{L}` line over its layer ranges; the header line holds the tag, the tensor count and the sha256 of the lines.
pub fn unfold(arena: std.mem.Allocator, text: []const u8, tag: []const u8) ![]header.Entry {
    var out: std.ArrayList(header.Entry) = .empty;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var head = std.mem.tokenizeScalar(u8, lines.next().?, ' ');
    if (!std.mem.eql(u8, head.next().?, tag)) return error.BadFixture;
    const count = try std.fmt.parseInt(usize, head.next().?, 10);
    const digest = head.next().?;
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const pat = f.next().?;
        const dtype = header.Dtype.of(f.next().?).?;
        var e: header.Entry = .{ .name = pat, .dtype = dtype, .rank = 0, .dims = @splat(0) };
        var dims = std.mem.tokenizeScalar(u8, f.next().?, 'x');
        while (dims.next()) |d| : (e.rank += 1) e.dims[e.rank] = try std.fmt.parseInt(u64, d, 10);
        const layers = f.next() orelse {
            try out.append(arena, e);
            continue;
        };
        var spans = std.mem.tokenizeScalar(u8, layers, ',');
        while (spans.next()) |span| {
            const dash = std.mem.indexOfScalar(u8, span, '-');
            const lo = try std.fmt.parseInt(u32, if (dash) |d| span[0..d] else span, 10);
            const hi = if (dash) |d| try std.fmt.parseInt(u32, span[d + 1 ..], 10) else lo;
            for (lo..hi + 1) |l| {
                var b: [8]u8 = undefined;
                var named = e;
                named.name = try std.mem.replaceOwned(u8, arena, pat, "{L}", try std.fmt.bufPrint(&b, "{d}", .{l}));
                try out.append(arena, named);
            }
        }
    }
    header.sortEntries(out.items);
    if (out.items.len != count) return error.BadFixture;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (out.items, 0..) |e, i| {
        if (i > 0) h.update("\n");
        var shape: std.ArrayList(u8) = .empty;
        for (e.shape(), 0..) |d, k| try shape.print(arena, "{s}{d}", .{ if (k > 0) "x" else "", d });
        h.update(try std.fmt.allocPrint(arena, "{s} {s} {s}", .{ e.name, e.dtype.tag(), shape.items }));
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(d, .lower), digest)) return error.BadFixture;
    return out.items;
}

fn load(a: std.mem.Allocator) !config.Config {
    var why: config.Why = .{};
    return config.parse(a, config_test.tiny, &why);
}

test "the folded fixtures unfold to the 404 and 84 tensors of tiny's two files, digests equal" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try t.expectEqual(@as(usize, 404), (try unfold(arena.allocator(), model_fixture, "inkling2-tiny-model")).len);
    try t.expectEqual(@as(usize, 84), (try unfold(arena.allocator(), mtp_fixture, "inkling2-tiny-mtp")).len);
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

test "a missing tensor, a wrong shape, a wrong dtype and an unknown tensor are each reported by name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try load(a);
    const all = try unfold(a, model_fixture, "inkling2-tiny-model");
    var kept: std.ArrayList(header.Entry) = .empty;
    for (all) |e| {
        var x = e;
        if (std.mem.eql(u8, e.name, "model.llm.layers.3.attn.wq_du.weight")) continue; // missing
        if (std.mem.eql(u8, e.name, "model.llm.layers.4.attn.wk_dv.weight")) x.dims[0] += 1; // shape
        if (std.mem.eql(u8, e.name, "model.llm.layers.2.mlp.gate.bias")) x.dtype = .bf16; // dtype
        try kept.append(a, x);
    }
    var extra = kept.items[0];
    extra.name = "model.llm.layers.0.attn.rope_freqs";
    try kept.append(a, extra);
    const r = try names.check(a, &c, kept.items, .model);
    try t.expectEqualStrings("model.llm.layers.3.attn.wq_du.weight", r.missing.items[0]);
    try t.expectEqual(@as(usize, 1), r.missing.items.len);
    try t.expectEqualStrings("model.llm.layers.2.mlp.gate.bias", r.wrong.items[0]);
    try t.expectEqualStrings("model.llm.layers.4.attn.wk_dv.weight", r.wrong.items[1]);
    try t.expectEqualStrings("model.llm.layers.0.attn.rope_freqs", r.unknown.items[0]);
    try t.expect(!r.clean());
}

test "expert precision follows each layer's tensors: NVFP4 on layers 0 to 5, MXFP8 on 6 to 11" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const entries = try unfold(a, model_fixture, "inkling2-tiny-model");
    var prec: [12]plan.Precision = undefined;
    try names.precisions(entries, 12, &prec);
    for (prec, 0..) |p, i| try t.expectEqual(if (i < 6) plan.Precision.nvfp4 else plan.Precision.mxfp8, p);
    const bad = try a.dupe(header.Entry, entries);
    for (bad) |*e| if (std.mem.eql(u8, e.name, "model.llm.layers.7.mlp.experts.w13_weight")) {
        e.dtype = .bf16;
    };
    try t.expectError(error.ExpertFormat, names.precisions(bad, 12, &prec));
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
}

fn headerOf(a: std.mem.Allocator, dir: []const u8, file: []const u8) ![]header.Entry {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, file });
    const f = try std.Io.Dir.cwd().openFile(t.io, path, .{});
    defer f.close(t.io);
    var prefix: [8]u8 = undefined;
    if (try f.readPositionalAll(t.io, &prefix, 0) != 8) return error.BadHeader;
    const n = std.mem.readInt(u64, &prefix, .little);
    if (n > 1 << 24) return error.BadHeader;
    const json = try a.alloc(u8, @intCast(n));
    if (try f.readPositionalAll(t.io, json, 8) != n) return error.BadHeader;
    return header.parse(a, json);
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
    var all: std.ArrayList(header.Entry) = .empty;
    for (1..25) |i| {
        const file = try std.fmt.allocPrint(a, "model-{d:0>5}-of-00024.safetensors", .{i});
        try all.appendSlice(a, try headerOf(a, dir, file));
    }
    header.sortEntries(all.items);
    const model = try names.check(a, &c, all.items, .model);
    try t.expectEqual(@as(usize, 1770), model.expected);
    try t.expect(model.clean());
    var prec: [config.max_layers]plan.Precision = undefined;
    try names.precisions(all.items, c.text.layers, &prec);
    for (prec[0..48], 0..) |p, i| try t.expectEqual(if (i < 40) plan.Precision.nvfp4 else plan.Precision.mxfp8, p);
    const mtp = try names.check(a, &c, try headerOf(a, dir, "mtp.safetensors"), .mtp);
    try t.expectEqual(@as(usize, 84), mtp.expected);
    try t.expect(mtp.clean());
}
