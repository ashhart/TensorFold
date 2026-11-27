//! GLM startup refuses a tokenizer that can serve token IDs the target embedding does not declare.
const std = @import("std");
const HfText = @import("hf_text.zig").HfText;

const too_many = "more tokens than the model vocabulary";

fn refuses(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, a: std.mem.Allocator, problem: *[]const u8) !void {
    if (HfText.load(gpa, io, dir, null, a, problem)) |text| {
        text.deinit();
        return error.TestUnexpectedResult;
    } else |err| try std.testing.expectEqual(error.Load, err);
    try std.testing.expect(std.mem.indexOf(u8, problem.*, too_many) != null);
}

test "GLM embedding: a tokenizer larger than the model vocabulary refuses startup; padded vocabularies load" {
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
    // Flash reads text_config: two tokens against one embedding row refuse; equal and padded rows load
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"glm5_next","text_config":{"vocab_size":1}}
    });
    try refuses(gpa, io, dir, a, &problem);
    for ([_][]const u8{
        \\{"model_type":"glm5_next","text_config":{"vocab_size":2}}
        ,
        \\{"model_type":"glm5_next","text_config":{"vocab_size":3}}
        ,
    }) |config| {
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
        const valid = try HfText.load(gpa, io, dir, null, a, &problem);
        defer valid.deinit();
        try std.testing.expectEqual(2, valid.text().vocabSize());
    }
    // full GLM sizes its embedding from the root: a nested or empty text_config must not hide a short root
    for ([_][]const u8{
        \\{"model_type":"glm_moe_dsa","vocab_size":1}
        ,
        \\{"model_type":"glm_moe_dsa","vocab_size":1,"text_config":{"vocab_size":2}}
        ,
        \\{"model_type":"glm_moe_dsa","vocab_size":1,"text_config":{}}
        ,
    }) |config| {
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
        try refuses(gpa, io, dir, a, &problem);
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"glm_moe_dsa","vocab_size":3,"text_config":{"vocab_size":1}}
    });
    const full = try HfText.load(gpa, io, dir, null, a, &problem);
    defer full.deinit();
    try std.testing.expectEqual(2, full.text().vocabSize());
}
