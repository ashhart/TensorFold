//! GLM startup refuses served token IDs the target embedding does not declare.
const std = @import("std");
const HfText = @import("hf_text.zig").HfText;

test "GLM embedding: tokenizer and model vocabulary mismatch refuses startup" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data =
        \\{"model":{"type":"BPE","vocab":{"H":0,"I":1},"merges":[]},"added_tokens":[]}
    });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var problem: []const u8 = "";
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"glm5_next","text_config":{"vocab_size":3}}
    });
    if (HfText.load(gpa, io, dir, null, a, &problem)) |text| {
        text.deinit();
        return error.TestUnexpectedResult;
    } else |err| try std.testing.expectEqual(error.Load, err);
    try std.testing.expect(std.mem.indexOf(u8, problem, "tokenizer/model vocabulary mismatch") != null);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"glm5_next","text_config":{"vocab_size":2}}
    });
    const valid = try HfText.load(gpa, io, dir, null, a, &problem);
    defer valid.deinit();
    try std.testing.expectEqual(2, valid.text().vocabSize());
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"glm_moe_dsa","vocab_size":3}
    });
    if (HfText.load(gpa, io, dir, null, a, &problem)) |text| {
        text.deinit();
        return error.TestUnexpectedResult;
    } else |err| try std.testing.expectEqual(error.Load, err);
}
