//! The qwen3 CUDA open refuses the flags it does not serve before an absent checkpoint or device can be opened.
const std = @import("std");
const api = @import("engine_api");

pub fn flags(comptime open: anytype) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const base: api.Open = .{ .dir = "missing-qwen3-checkpoint", .model_type = "qwen3" };
    var drafter = base;
    drafter.drafter = "d";
    var vision = base;
    vision.vision = .{};
    var kv8 = base;
    kv8.kv8 = true;
    var int8 = base;
    int8.prompt_int8 = true;
    var loops = base;
    loops.loops = 2;
    var ranks = base;
    ranks.tp = 2;
    var spill = base;
    spill.spill = .evicted;
    var learn = base;
    learn.learn_conversations = true;
    var precision = base;
    precision.precision = .full;
    const asked = [_]struct { api.Open, []const u8 }{
        .{ drafter, "--drafter: Qwen3 dense checkpoints have no drafts" },
        .{ vision, "--vision: this CUDA family reads no images" },
        .{ kv8, "--kv-cache fp8: this CUDA family keeps bf16 caches" },
        .{ int8, "--prompt-precision int8: this CUDA family's prompt matmuls take bf16 inputs" },
        .{ loops, "--loops sets a looped model's recurrent passes" },
        .{ ranks, "this CUDA family has no tensor-parallel ranks" },
        .{ spill, "--spill-gib" },
        .{ learn, "--learn-conversations" },
        .{ precision, "--precision: this CUDA family reads no NVFP4 checkpoint" },
    };
    for (asked) |c| {
        var why: []const u8 = "";
        try std.testing.expect((try open(arena.allocator(), std.testing.allocator, std.testing.io, c[0], &why)) == null);
        try std.testing.expectStringStartsWith(why, c[1]);
    }
}
