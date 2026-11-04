//! Prints the repository's own contrast prompt sets as JSON, so a measurement tool reads them here.
const std = @import("std");
const q = @import("qwen27");

const help =
    \\tf-qwen27-steer-prompts [--set derive|holdout|all]
    \\  Prints the repository's own contrast prompt sets as JSON on stdout.
    \\  --set derive is the set the producer derives a control vector from;
    \\  --set holdout is the set written after the vector was derived, which the producer never reads;
    \\  --set all prints both, and is the default.
;

/// One set's pairs as JSON, for whichever pair type the set declares.
fn setJson(comptime Pair: type, a: std.mem.Allocator, pairs: []const Pair) !std.json.Value {
    var items = std.json.Array.init(a);
    for (pairs) |p| {
        var pair = std.json.ObjectMap.empty;
        try pair.put(a, "request", .{ .string = p.request });
        try pair.put(a, "harmless", .{ .string = p.harmless });
        try items.append(.{ .object = pair });
    }
    return .{ .array = items };
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len > 1 and std.mem.eql(u8, args[1], "--help")) {
        try std.Io.File.stdout().writeStreamingAll(io, help ++ "\n");
        return;
    }
    var which: []const u8 = "all";
    if (args.len > 1) {
        if (!std.mem.eql(u8, args[1], "--set")) return error.UnknownOption;
        if (args.len < 3) return error.MissingValue;
        which = args[2];
    }

    var sets = std.json.ObjectMap.empty;
    if (std.mem.eql(u8, which, "derive") or std.mem.eql(u8, which, "all")) {
        try sets.put(a, "derive", try setJson(@TypeOf(q.steer_prompts.pairs[0]), a, &q.steer_prompts.pairs));
    }
    if (std.mem.eql(u8, which, "holdout") or std.mem.eql(u8, which, "all")) {
        try sets.put(a, "holdout", try setJson(@TypeOf(q.steer_prompts_holdout.pairs[0]), a, &q.steer_prompts_holdout.pairs));
    }
    if (sets.count() == 0) return error.UnknownOption;

    var root = std.json.ObjectMap.empty;
    try root.put(a, "kind", .{ .string = "qwen27_steer_prompts" });
    try root.put(a, "sets", .{ .object = sets });
    const receipt = try std.json.Stringify.valueAlloc(a, std.json.Value{ .object = root }, .{});
    try std.Io.File.stdout().writeStreamingAll(io, receipt);
    try std.Io.File.stdout().writeStreamingAll(io, "\n");
}
