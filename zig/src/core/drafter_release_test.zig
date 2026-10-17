//! Drafter release format: Kolibri's and Nemotron's adapters, refusals, round trips, recordings and release folders.

const std = @import("std");
const release = @import("drafter_release.zig");
const layout = @import("drafter_layout.zig");
const Why = release.Why;
const gpa = std.testing.allocator;
const io = std.testing.io;

/// Kolibri-1's block drafter in this format: taps after layers 45, 48 and 50 of 50, the last normed by the final norm.
const kolibri =
    \\{"format": "tensorfold-drafter", "version": 1, "architecture": "block-drafter",
    \\ "target": {"repo": "Aleph-Alpha/Kolibri-1", "revision": "e52eb4627d11516b0c01de49210ab5a4e4061444",
    \\  "layers": 50, "hidden": 2560, "vocab": 128000, "embed_scale": 1.0, "tied": false, "softcap": null, "head_bias": false},
    \\ "taps": {"ids": [45, 48, 50], "stream": "residual", "normed": true, "dtype": "bf16", "row_bytes": 15360},
    \\ "drafter": {"hidden": 2560, "heads": 20, "head_dim": 128, "ffn": 6144, "layers": 4, "block": 4, "window": 2048,
    \\  "eps": 1e-6, "theta": 10000.0, "pred_rank": 256, "spine_rank": 256},
    \\ "draft_vocab": {"file": "draft_vocab.json", "size": 60614}, "weights": "model.safetensors",
    \\ "serving": {"depth": 2, "body": "fp8", "head": "nvfp4"}}
;

/// Nemotron 3.5 Lightning's adapter and its one tap, the final norm's rows; the drafter's shape is a placeholder.
const nemotron =
    \\{"format": "tensorfold-drafter", "version": 1, "architecture": "block-drafter",
    \\ "target": {"repo": "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "revision": "d9d758fb83953437f7263256b0d96157e2a348b8",
    \\  "layers": 52, "hidden": 2688, "vocab": 131072, "embed_scale": 1.0, "tied": false, "softcap": null, "head_bias": false},
    \\ "taps": {"ids": [52], "stream": "residual", "normed": true, "dtype": "bf16", "row_bytes": 5376},
    \\ "drafter": {"hidden": 2688, "heads": 21, "head_dim": 128, "ffn": 5376, "layers": 2, "block": 4, "window": 2048,
    \\  "eps": 1e-5, "theta": 10000.0, "pred_rank": 0, "spine_rank": 0},
    \\ "draft_vocab": {"file": "draft_vocab.json", "size": 32768}, "weights": "model.safetensors",
    \\ "serving": {"depth": 2, "body": "bf16", "head": "bf16"}}
;

/// A recorder `.json` of one Nemotron stream: 163 rows of the final norm's bf16 rows.
const recorded = "{\"rows\":163,\"first_row\":0,\"prompt_len\":36,\"taps\":[52],\"row_bytes\":5376,\"dtype\":\"bf16\",\"top_k\":0,\"gap\":null}";

const nemotron_adapter: release.Adapter = .{ .layers = 52, .hidden = 2688, .vocab = 131072, .embed_scale = 1.0, .tied = false, .softcap = null, .head_bias = false };

fn expectRefused(text: []const u8, err: anyerror, words: []const []const u8) !void {
    var why: Why = .{};
    try std.testing.expectError(err, release.Manifest.parse(gpa, text, &why));
    try expectNames(&why, words);
}

fn expectNames(why: *const Why, words: []const []const u8) !void {
    for (words) |word| if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    };
}

/// `base` with `from` replaced by `to` once; the test fails if `from` is not there.
fn edited(base: []const u8, from: []const u8, to: []const u8) ![]u8 {
    if (std.mem.indexOf(u8, base, from) == null) return error.TestEditNotFound;
    return std.mem.replaceOwned(u8, gpa, base, from, to);
}

fn expectEdit(from: []const u8, to: []const u8, words: []const []const u8) !void {
    const text = try edited(kolibri, from, to);
    defer gpa.free(text);
    try expectRefused(text, error.BadDrafterRelease, words);
}

fn reparse(m: *const release.Manifest) !release.Manifest {
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try release.write(&m.config, &out.writer);
    var why: Why = .{};
    return release.Manifest.parse(gpa, out.written(), &why);
}

test "Kolibri's and Nemotron's releases parse, and written back they read the same" {
    for ([_][]const u8{ kolibri, nemotron }) |text| {
        var why: Why = .{};
        var m = try release.Manifest.parse(gpa, text, &why);
        defer m.deinit();
        var back = try reparse(&m);
        defer back.deinit();
        var again = try reparse(&back);
        defer again.deinit();
        inline for (@typeInfo(release.Adapter).@"struct".field_names) |f| try std.testing.expectEqual(@field(m.config.target, f), @field(again.config.target, f));
        try std.testing.expectEqualSlices(u32, m.config.taps.ids, again.config.taps.ids);
        try std.testing.expectEqual(m.config.taps.row_bytes, again.config.taps.row_bytes);
        try std.testing.expectEqual(m.config.taps.normed, again.config.taps.normed);
        try std.testing.expectEqual(m.config.drafter, again.config.drafter);
        try std.testing.expectEqual(m.config.serving, again.config.serving);
        try std.testing.expectEqualStrings(m.config.target.revision, again.config.target.revision);
        try std.testing.expectEqualStrings(m.config.draft_vocab.file, again.config.draft_vocab.file);
    }
    var why: Why = .{};
    var k = try release.Manifest.parse(gpa, kolibri, &why);
    defer k.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 45, 48, 50 }, k.config.taps.ids);
    try std.testing.expectEqual(@as(?f64, null), k.config.target.softcap);
}

test "a manifest outside the format is refused, naming what is wrong" {
    try expectEdit("\"tensorfold-drafter\"", "\"kolibri1-block-drafter\"", &.{"kolibri1-block-drafter"});
    try expectEdit("\"version\": 1", "\"version\": 2", &.{ "version 2", "version 1" });
    try expectEdit("[45, 48, 50]", "[45, 50, 48]", &.{ "rise", "48 after 50" });
    try expectEdit("[45, 48, 50]", "[45, 48, 51]", &.{ "tap 51", "50 layers" });
    try expectEdit("[45, 48, 50]", "[45, 45, 50]", &.{ "rise", "45 after 45" });
    try expectEdit("\"row_bytes\": 15360", "\"row_bytes\": 7680", &.{ "row_bytes 7680", "15360" });
    try expectEdit("\"dtype\": \"bf16\"", "\"dtype\": \"f32\"", &.{ "row_bytes 15360", "30720" });
    try expectEdit("{\"hidden\": 2560, \"heads\"", "{\"hidden\": 2048, \"heads\"", &.{ "drafter hidden 2048", "2560" });
    try expectEdit("\"head_dim\": 128", "\"head_dim\": 127", &.{"head_dim 127"});
    try expectEdit("\"block\": 4", "\"block\": 1", &.{"block 1"});
    try expectEdit("\"depth\": 2", "\"depth\": 5", &.{ "depth 5", "1..4" });
    try expectEdit("\"softcap\": null", "\"softcap\": 0", &.{"softcap 0"});
    try expectEdit("\"embed_scale\": 1.0", "\"embed_scale\": -1.0", &.{"embed_scale -1"});
    try expectEdit("\"eps\": 1e-6", "\"eps\": 0", &.{"drafter eps 0"});
    try expectEdit("\"theta\": 10000.0", "\"theta\": -1", &.{"drafter theta -1"});
    try expectEdit("\"eps\": 1e-6", "\"eps\": 1e5000", &.{"drafter eps inf"});
    try expectEdit("\"theta\": 10000.0", "\"theta\": 1e5000", &.{"drafter theta inf"});
    try expectEdit("\"theta\": 10000.0", "\"theta\": \"inf\"", &.{"drafter theta inf"});
    try expectEdit("\"size\": 60614", "\"size\": 128001", &.{ "128001", "128000" });
    try expectEdit("\"model.safetensors\"", "\"../model.safetensors\"", &.{"../model.safetensors"});
    try expectEdit("\"draft_vocab.json\"", "\"\"", &.{"is not a file name"});
    try expectEdit("\"ids\": [45, 48, 50]", "\"ids\": []", &.{"no taps"});
    try expectEdit("\"stream\": \"residual\"", "\"stream\": \"lane0\"", &.{"does not parse"});
    try expectEdit("\"head_bias\": false}", "\"head_bias\": false, \"bias\": 1}", &.{"UnknownField"});
    try expectEdit(", \"head_bias\": false", "", &.{"MissingField"});
    try expectEdit("\"tied\": false", "\"tied\": \"no\"", &.{"does not parse"});
}

test "a release is refused for a target whose adapter differs, naming the field" {
    var why: Why = .{};
    var m = try release.Manifest.parse(gpa, nemotron, &why);
    defer m.deinit();
    try release.check(&m.config, nemotron_adapter, &why);
    const cases = .{
        .{ "tied", release.Adapter{ .layers = 52, .hidden = 2688, .vocab = 131072, .embed_scale = 1.0, .tied = true, .softcap = null, .head_bias = false } },
        .{ "softcap", release.Adapter{ .layers = 52, .hidden = 2688, .vocab = 131072, .embed_scale = 1.0, .tied = false, .softcap = 30.0, .head_bias = false } },
        .{ "head_bias", release.Adapter{ .layers = 52, .hidden = 2688, .vocab = 131072, .embed_scale = 1.0, .tied = false, .softcap = null, .head_bias = true } },
        .{ "embed_scale", release.Adapter{ .layers = 52, .hidden = 2688, .vocab = 131072, .embed_scale = 51.84, .tied = false, .softcap = null, .head_bias = false } },
        .{ "layers", release.Adapter{ .layers = 50, .hidden = 2688, .vocab = 131072, .embed_scale = 1.0, .tied = false, .softcap = null, .head_bias = false } },
        .{ "vocab", release.Adapter{ .layers = 52, .hidden = 2688, .vocab = 128000, .embed_scale = 1.0, .tied = false, .softcap = null, .head_bias = false } },
    };
    inline for (cases) |c| {
        why = .{};
        try std.testing.expectError(error.AdapterMismatch, release.check(&m.config, c[1], &why));
        try expectNames(&why, &.{"target " ++ c[0] ++ " ="});
    }
}

test "the recorder's .json and the release agree on taps, row bytes and dtype, or the recording is refused" {
    var why: Why = .{};
    var m = try release.Manifest.parse(gpa, nemotron, &why);
    defer m.deinit();
    try release.checkRecording(&m.config, gpa, recorded, &why);
    try release.checkRecording(&m.config, gpa, "{\"rows\":0,\"first_row\":0,\"prompt_len\":4,\"taps\":[52],\"row_bytes\":0,\"dtype\":\"\",\"top_k\":0,\"gap\":\"a write failed\"}", &why);
    for ([_][3][]const u8{
        .{ "\"taps\":[52]", "\"taps\":[51]", "recorded taps" },
        .{ "\"row_bytes\":5376", "\"row_bytes\":2688", "recorded row_bytes 2688" },
        .{ "\"dtype\":\"bf16\"", "\"dtype\":\"f32\"", "recorded dtype \"f32\"" },
    }) |c| {
        const text = try edited(recorded, c[0], c[1]);
        defer gpa.free(text);
        try std.testing.expectError(error.RecordingMismatch, release.checkRecording(&m.config, gpa, text, &why));
        try expectNames(&why, &.{c[2]});
    }
    try std.testing.expectError(error.BadRecording, release.checkRecording(&m.config, gpa, "{\"rows\":1}", &why));
}

test "the draft vocabulary is distinct target ids, as many as config.json says" {
    var why: Why = .{};
    const text = try edited(nemotron, "\"size\": 32768", "\"size\": 3");
    defer gpa.free(text);
    var m = try release.Manifest.parse(gpa, text, &why);
    defer m.deinit();
    const ids = try release.parseVocab(&m.config, gpa, "[11, 2, 131071]", &why);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 11, 2, 131071 }, ids);
    for ([_][2][]const u8{ .{ "[11, 2]", "has 2 ids" }, .{ "[11, 2, 11]", "11 appears twice" }, .{ "[11, 2, 131072]", "131072 is past" }, .{ "[11, 2, -1]", "not a JSON array" } }) |c| {
        try std.testing.expectError(error.BadDrafterRelease, release.parseVocab(&m.config, gpa, c[0], &why));
        try expectNames(&why, &.{c[1]});
    }
}

test {
    _ = @import("drafter_folder_test.zig");
}
