//! Small published release folders (EAGLE-3 and block shapes) open, or are refused naming what is wrong.

const std = @import("std");
const release = @import("drafter_release.zig");
const layout = @import("drafter_layout.zig");
const published = @import("drafter_published.zig");
const Why = release.Why;
const gpa = std.testing.allocator;
const io = std.testing.io;

const adapter: release.Adapter = .{ .layers = 6, .hidden = 8, .vocab = 16, .embed_scale = 1.0, .tied = false, .softcap = null, .head_bias = false };
const target: published.Target = .{ .repo = "test/small", .adapter = adapter };

const eagle =
    \\{"architecture": "kolibri1-eagle3-drafter", "target": "test/small",
    \\ "drafter": {"hidden": 8, "heads": 2, "head_dim": 4, "ffn": 16, "eps": 1e-06, "theta": 10000.0, "layers": 1, "taps": 3},
    \\ "taps": [2, 4, 5], "draft_vocab": "draft_vocab.json", "draft_vocab_size": 5,
    \\ "training_tokens": 10, "training_steps": 2}
;

const block =
    \\{"architecture": "kolibri1-block-drafter", "target": "test/small",
    \\ "drafter": {"hidden": 8, "heads": 2, "head_dim": 4, "ffn": 16, "layers": 2, "taps": 3, "block": 4,
    \\  "eps": 1e-06, "theta": 10000.0, "chain_ctx": true, "row0_chain": true, "pred_rank": 2, "spine_rank": 2},
    \\ "taps": [2, 4, 5], "draft_vocab": "draft_vocab.json", "draft_vocab_size": 5,
    \\ "serving": {"depth": 2, "quant": "fp8", "head": "nvfp4"}, "training_tokens": 10, "training_steps": 2}
;

const Case = struct { config: []const u8 = eagle, vocab: []const u8 = "[3, 1, 4, 15, 9]", skip: ?[]const u8 = null, extra: ?[]const u8 = null, f32_norms: bool = false };

/// A safetensors image of every tensor `config` implies under its published names (bf16 zeros; norms f32 if asked).
fn image(a: std.mem.Allocator, case: Case) ![]u8 {
    var why: Why = .{};
    const got = try published.parse(a, case.config, target, &why);
    var h: std.Io.Writer.Allocating = .init(a);
    try h.writer.writeByte('{');
    var at: usize = 0;
    var first = true;
    var names: std.ArrayList([]const u8) = .empty;
    var shapes: std.ArrayList([2]usize) = .empty;
    var ranks: std.ArrayList(u8) = .empty;
    for (try layout.expected(a, &got.config)) |w| {
        if (std.mem.startsWith(u8, w.name, "target.")) continue;
        const name = try got.names.of(a, w.name);
        if (case.skip != null and std.mem.eql(u8, name, case.skip.?)) continue;
        try names.append(a, name);
        try shapes.append(a, w.shape);
        try ranks.append(a, w.rank);
    }
    if (case.extra) |e| {
        try names.append(a, e);
        try shapes.append(a, .{ 8, 8 });
        try ranks.append(a, 2);
    }
    for (names.items, shapes.items, ranks.items) |name, sh, rank| {
        const wide = case.f32_norms and rank == 1;
        const n = (if (wide) @as(usize, 4) else 2) * sh[0] * (if (rank == 2) sh[1] else 1);
        if (!first) try h.writer.writeByte(',');
        first = false;
        const dt = if (wide) "F32" else "BF16";
        if (rank == 2) {
            try h.writer.print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ name, dt, sh[0], sh[1], at, at + n });
        } else try h.writer.print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ name, dt, sh[0], at, at + n });
        at += n;
    }
    try h.writer.writeAll(",\"__metadata__\":{\"format\":\"pt\"}}");
    const out = try a.alloc(u8, 8 + h.written().len + at);
    @memset(out, 0);
    std.mem.writeInt(u64, out[0..8], h.written().len, .little);
    @memcpy(out[8..][0..h.written().len], h.written());
    return out;
}

fn openCase(case: Case, why: *Why) !published.Release {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const bytes = image(arena.allocator(), case) catch try image(arena.allocator(), .{}); // a refused config: any image
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = case.config });
    try tmp.dir.writeFile(io, .{ .sub_path = "draft_vocab.json", .data = case.vocab });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes });
    return published.Release.open(gpa, io, tmp.dir, target, why);
}

fn expectRefused(case: Case, words: []const []const u8) !void {
    var why: Why = .{};
    if (openCase(case, &why)) |r| {
        var rr = r;
        rr.deinit(gpa);
        return error.TestUnexpectedResult;
    } else |e| try std.testing.expect(e == error.BadDrafterRelease or e == error.AdapterMismatch);
    for (words) |word| if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    };
}

/// `base` with `from` replaced by `to` refused, naming each of `words`.
fn expectEdited(base: []const u8, from: []const u8, to: []const u8, words: []const []const u8) !void {
    const text = try std.mem.replaceOwned(u8, gpa, base, from, to);
    defer gpa.free(text);
    try std.testing.expect(!std.mem.eql(u8, text, base));
    try expectRefused(.{ .config = text }, words);
}

test "an EAGLE-3 release opens: taps after layers 3, 5, 6; one row a pass; layer names under blocks; f32 norms" {
    var why: Why = .{};
    var r = try openCase(.{ .f32_norms = true }, &why);
    defer r.deinit(gpa);
    const c = r.manifest.config;
    try std.testing.expectEqual(release.Architecture.eagle3, c.architecture);
    try std.testing.expectEqualSlices(u32, &.{ 3, 5, 6 }, c.taps.ids);
    try std.testing.expectEqual(@as(u32, 3 * 8 * 2), c.taps.row_bytes);
    try std.testing.expectEqual(@as(u32, 1), c.drafter.block);
    try std.testing.expectEqual(published.window, c.drafter.window);
    try std.testing.expectEqual(release.Serving{ .depth = 2, .body = .bf16, .head = .bf16 }, c.serving);
    try std.testing.expectEqualSlices(u32, &.{ 3, 1, 4, 15, 9 }, r.vocab);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    try std.testing.expectEqualStrings("blocks.0.q.weight", try r.names.of(arena.allocator(), "layers.0.q.weight"));
    try std.testing.expect(published.declared(eagle) and published.declared(block));
}

test "a block release opens with its serving defaults, and its unused inp.weight is admitted by name" {
    var why: Why = .{};
    var r = try openCase(.{ .config = block, .extra = "inp.weight" }, &why);
    defer r.deinit(gpa);
    const c = r.manifest.config;
    try std.testing.expectEqual(release.Architecture.@"block-drafter", c.architecture);
    try std.testing.expectEqual(release.Serving{ .depth = 2, .body = .fp8, .head = .nvfp4 }, c.serving);
    try std.testing.expectEqual(@as(u32, 4), c.drafter.block);
    try expectRefused(.{ .extra = "inp.weight" }, &.{ "inp.weight", "does not name" }); // EAGLE-3 ships none
}

test "a published release with the wrong layout is refused, naming what is wrong" {
    try expectEdited(eagle, "kolibri1-eagle3-drafter", "kolibri1-mtp", &.{ "kolibri1-mtp", "not a published drafter release" });
    try expectEdited(eagle, "\"layers\": 1,", "\"layers\": 1, \"rope\": 1,", &.{"does not parse"});
    try expectEdited(eagle, "\"taps\": 3}", "\"taps\": 2}", &.{ "taps 2", "3 layers" });
    try expectEdited(eagle, "\"layers\": 1,", "\"layers\": 1, \"block\": 4,", &.{ "eagle3", "block 1" });
    try expectEdited(block, "\"chain_ctx\": true", "\"chain_ctx\": false", &.{"chain_ctx"});
    try expectEdited(eagle, "[2, 4, 5]", "[2, 4, 6]", &.{ "tap 7", "6 layers" });
    try expectRefused(.{ .skip = "blocks.0.q.weight" }, &.{"has no blocks.0.q.weight"});
    try expectRefused(.{ .config = block, .skip = "layers.1.ck.weight" }, &.{"has no layers.1.ck.weight"});
    try expectRefused(.{ .extra = "target.head" }, &.{ "target.head", "does not name" }); // the head is the target's
}

test "a release for another target, or with a vocabulary that does not fit it, is refused by name" {
    try expectEdited(eagle, "\"target\": \"test/small\"", "\"target\": \"test/other\"", &.{ "test/other", "test/small" });
    try expectEdited(eagle, "\"hidden\": 8", "\"hidden\": 16", &.{ "hidden 16", "target's 8" });
    try expectRefused(.{ .vocab = "[3, 1, 4, 15]" }, &.{ "4 ids", "says 5" });
    try expectRefused(.{ .vocab = "[3, 1, 4, 15, 16]" }, &.{ "16", "vocab of 16" });
    try expectRefused(.{ .vocab = "[3, 1, 4, 3, 9]" }, &.{"3 appears twice"});
    try expectEdited(eagle, "\"draft_vocab_size\": 5", "\"draft_vocab_size\": 17", &.{"17 outside 1..16"});
}

test "a v1 release is not a published one" {
    try std.testing.expect(!published.declared("{\"format\": \"tensorfold-drafter\", \"version\": 1}"));
    try std.testing.expect(!published.declared("not json"));
}
