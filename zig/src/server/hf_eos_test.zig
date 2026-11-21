//! Exercise actual HfText loading, including optional-file and invalid-EOS refusals.
const std = @import("std");
const HfText = @import("hf_text.zig").HfText;
const fixture = @import("hf_eos_fixture.zig");

test "EOS policy: HfText loads both declared EOS ids and missing generation never makes pad an EOS" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    for ([_]?[]const u8{ fixture.generation, null }) |generation| {
        var f = try fixture.Fixture.init(a, io, generation);
        defer f.deinit();
        var problem: []const u8 = "";
        const t = try HfText.load(a, io, f.path, null, a, &problem);
        defer t.deinit();
        try std.testing.expectEqualSlices(u32, if (generation != null) &.{ 127906, 127901 } else &.{127906}, t.text().eosIds());
    }
}

test "EOS policy: HfText names malformed generation and out-of-vocabulary EOS without a panic or fallback" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    for ([_][]const u8{ "{", "false", "null", "{\"eos_token_id\":-1}", "{\"eos_token_id\":128000}", "{\"eos_token_id\":4294967296}", "{\"eos_token_id\":[127906,true]}" }) |generation| {
        var f = try fixture.Fixture.init(a, io, generation);
        defer f.deinit();
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        var problem: []const u8 = "";
        try std.testing.expectError(error.Load, HfText.load(a, io, f.path, null, arena.allocator(), &problem));
        try std.testing.expect(std.mem.indexOf(u8, problem, "EOS") != null);
    }
}

test "EOS policy: HfText validates present model EOS and the model head bound as well as tokenizer ids" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    for ([_][]const u8{ "{", "{\"eos_token_id\":[127906,false]}", "{\"vocab_size\":3,\"eos_token_id\":127906}" }) |model| {
        var f = try fixture.Fixture.init(a, io, fixture.generation);
        defer f.deinit();
        try f.tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = model });
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        var problem: []const u8 = "";
        try std.testing.expectError(error.Load, HfText.load(a, io, f.path, null, arena.allocator(), &problem));
        try std.testing.expect(std.mem.indexOf(u8, problem, "EOS") != null);
    }
}

test "HfText opens a tokenizer holding Inkling 2's markers as the .inkling format and any other as .template" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const markers = @import("inkling_reply.zig").markers;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    const plain = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"H\":0},\"merges\":[]},\"added_tokens\":[]}";
    const inkling = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"H\":0},\"merges\":[]},\"added_tokens\":[" ++
        "{\"id\":1,\"content\":\"" ++ markers.model ++ "\",\"special\":true}," ++
        "{\"id\":2,\"content\":\"" ++ markers.end_sampling ++ "\",\"special\":true}]}";
    for ([_]struct { json: []const u8, want: @import("model_text.zig").Text.Format }{
        .{ .json = plain, .want = .template },
        .{ .json = inkling, .want = .inkling },
    }) |c| {
        try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = c.json });
        var problem: []const u8 = "";
        const t = try HfText.load(a, io, dir, null, a, &problem);
        defer t.deinit();
        try std.testing.expectEqual(c.want, t.text().vtable.format);
    }
}
