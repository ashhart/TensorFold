//! A small release folder written and read whole: a tensor missing, extra or misshapen, or a bad vocab, is refused.

const std = @import("std");
const release = @import("drafter_release.zig");
const layout = @import("drafter_layout.zig");
const Why = release.Why;
const gpa = std.testing.allocator;
const io = std.testing.io;

const small =
    \\{"format": "tensorfold-drafter", "version": 1, "architecture": "block-drafter",
    \\ "target": {"repo": "test/small", "revision": "0", "layers": 4, "hidden": 8, "vocab": 16,
    \\  "embed_scale": 2.0, "tied": false, "softcap": 30.0, "head_bias": true},
    \\ "taps": {"ids": [3, 4], "stream": "residual", "normed": true, "dtype": "f32", "row_bytes": 64},
    \\ "drafter": {"hidden": 8, "heads": 2, "head_dim": 4, "ffn": 16, "layers": 2, "block": 3, "window": 8,
    \\  "eps": 1e-6, "theta": 10000.0, "pred_rank": 2, "spine_rank": 2},
    \\ "draft_vocab": {"file": "draft_vocab.json", "size": 5}, "weights": "model.safetensors",
    \\ "serving": {"depth": 2, "body": "bf16", "head": "bf16"}}
;

/// A safetensors image of `want` in bf16, zero bytes; `skip` leaves one out, `extra` adds one.
fn image(want: []const layout.Want, skip: ?[]const u8, extra: ?[]const u8, bend: ?[]const u8) ![]u8 {
    var h: std.Io.Writer.Allocating = .init(gpa);
    defer h.deinit();
    try h.writer.writeByte('{');
    var at: usize = 0;
    var first = true;
    for (want) |w| {
        if (skip != null and std.mem.eql(u8, w.name, skip.?)) continue;
        const rows = if (bend != null and std.mem.eql(u8, w.name, bend.?)) w.shape[0] + 1 else w.shape[0];
        const n = 2 * rows * (if (w.rank == 2) w.shape[1] else 1);
        if (!first) try h.writer.writeByte(',');
        first = false;
        if (w.rank == 2) {
            try h.writer.print("\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ w.name, rows, w.shape[1], at, at + n });
        } else try h.writer.print("\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ w.name, rows, at, at + n });
        at += n;
    }
    if (extra) |name| {
        try h.writer.print(",\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[8],\"data_offsets\":[{d},{d}]}}", .{ name, at, at + 16 });
        at += 16;
    }
    try h.writer.writeByte('}');
    const out = try gpa.alloc(u8, 8 + h.written().len + at);
    @memset(out, 0);
    std.mem.writeInt(u64, out[0..8], h.written().len, .little);
    @memcpy(out[8..][0..h.written().len], h.written());
    return out;
}

const Case = struct { config: []const u8 = small, vocab: []const u8 = "[3, 1, 4, 15, 9]", skip: ?[]const u8 = null, extra: ?[]const u8 = null, bend: ?[]const u8 = null };

fn openCase(case: Case, why: *Why) !layout.Release {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var cfg_why: Why = .{};
    var m = try release.Manifest.parse(gpa, case.config, &cfg_why);
    defer m.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const bytes = try image(try layout.expected(arena.allocator(), &m.config), case.skip, case.extra, case.bend);
    defer gpa.free(bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = case.config });
    try tmp.dir.writeFile(io, .{ .sub_path = "draft_vocab.json", .data = case.vocab });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes });
    return layout.Release.open(gpa, io, tmp.dir, why);
}

fn expectRefused(case: Case, words: []const []const u8) !void {
    var why: Why = .{};
    try std.testing.expectError(error.BadDrafterRelease, openCase(case, &why));
    for (words) |word| if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    };
}

test "a whole release folder opens: config, vocab in order, and every tensor the config implies" {
    var why: Why = .{};
    var r = try openCase(.{}, &why);
    defer r.deinit(gpa);
    try std.testing.expectEqualSlices(u32, &.{ 3, 1, 4, 15, 9 }, r.vocab);
    try std.testing.expectEqual(@as(?f64, 30.0), r.manifest.config.target.softcap);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const want = try layout.expected(arena.allocator(), &r.manifest.config);
    try std.testing.expectEqual(@as(usize, 3 + 3 + 2 * 12 + 1 + 3 + 3), want.len);
}

test "a release folder missing, adding or misshaping a tensor is refused, naming it" {
    try expectRefused(.{ .skip = "target.head_bias" }, &.{"target.head_bias"});
    try expectRefused(.{ .skip = "layers.1.nc.w" }, &.{"layers.1.nc.w"});
    try expectRefused(.{ .skip = "spine_u.0.weight" }, &.{"spine_u.0.weight"});
    try expectRefused(.{ .extra = "inp.weight" }, &.{ "inp.weight", "does not name" });
    try expectRefused(.{ .bend = "fuse.weight" }, &.{ "fuse.weight", "{ 9, 16 }", "{ 8, 16 }" });
    try expectRefused(.{ .bend = "target.head" }, &.{"target.head is { 6, 8 }"});
    try expectRefused(.{ .vocab = "[3, 1, 4, 15, 3]" }, &.{"3 appears twice"});
}

test "a tied head ships no target.head, and one shipped anyway is refused" {
    const tied = try std.mem.replaceOwned(u8, gpa, small, "\"tied\": false", "\"tied\": true");
    defer gpa.free(tied);
    var why: Why = .{};
    var r = try openCase(.{ .config = tied }, &why);
    r.deinit(gpa);
    try expectRefused(.{ .config = tied, .extra = "target.head" }, &.{"target.head"});
}
