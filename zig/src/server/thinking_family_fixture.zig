//! Checkpoint templates with a lossless ASCII vocabulary and test-only whole thinking markers.
const std = @import("std");
const json = @import("json");
const Fixture = @import("hf_eos_fixture.zig").Fixture;

pub fn init(a: std.mem.Allocator, io: std.Io, source: []const u8, config: []const u8, kind: []const u8) !Fixture {
    var f = try Fixture.init(a, io, null);
    errdefer f.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    const vocab = try json.newObject(scratch);
    for ("HOK", 0..) |ch, id| try vocab.put(scratch, try scratch.dupe(u8, &.{ch}), try json.intValue(scratch, id));
    var next: usize = 3;
    for (0..128) |id| {
        const ch: u8 = @intCast(id);
        if (std.mem.indexOfScalar(u8, "HOK", ch) != null) continue;
        try vocab.put(scratch, try scratch.dupe(u8, &.{ch}), try json.intValue(scratch, next));
        next += 1;
    }
    const added = try scratch.alloc(json.Value, markers.len);
    for (markers, 0..) |marker, i| {
        try vocab.put(scratch, marker.text, try json.intValue(scratch, marker.id));
        const entry = try json.newObject(scratch);
        try entry.put(scratch, "id", try json.intValue(scratch, marker.id));
        try entry.put(scratch, "content", .{ .string = marker.text });
        try entry.put(scratch, "special", .{ .bool = true });
        added[i] = .{ .object = entry };
    }
    const model = try json.newObject(scratch);
    try model.put(scratch, "type", .{ .string = "BPE" });
    try model.put(scratch, "vocab", .{ .object = vocab });
    try model.put(scratch, "merges", .{ .array = &.{} });
    const decoder = try json.newObject(scratch);
    try decoder.put(scratch, "type", .{ .string = "Fuse" });
    const tok = try json.newObject(scratch);
    try tok.put(scratch, "model", .{ .object = model });
    try tok.put(scratch, "decoder", .{ .object = decoder });
    try tok.put(scratch, "added_tokens", .{ .array = added });
    const bytes = try json.stringify(scratch, .{ .object = tok }, .{});
    try f.tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = bytes });
    const model_config = try std.fmt.allocPrint(scratch, "{{\"model_type\":\"{s}\",\"eos_token_id\":127906,\"pad_token_id\":127901,\"vocab_size\":128000}}", .{kind});
    try f.tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = model_config });
    try f.tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = config });
    try f.tmp.dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data = source });
    return f;
}

const markers = [_]struct { text: []const u8, id: usize }{
    .{ .text = "<think>", .id = 300 },
    .{ .text = "</think>", .id = 301 },
    .{ .text = "<|startoftext|>", .id = 302 },
    .{ .text = "<|pad|>", .id = 303 },
    .{ .text = "<|endoftext|>", .id = 127901 },
    .{ .text = "<|im_end|>", .id = 127906 },
};
