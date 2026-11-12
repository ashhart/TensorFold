//! The qwen3 CUDA open refuses the flags it does not serve before an absent checkpoint or device can be opened.
const std = @import("std");
const api = @import("engine_api");
const qwen3 = @import("cuda_qwen3.zig");
const env = @import("sys").env;

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
    var record = base;
    record.record = "rows";
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
        .{ record, "--record: qwen3 checkpoints give no rows to record on CUDA yet" },
        .{ precision, "--precision: this CUDA family reads no NVFP4 checkpoint" },
    };
    for (asked) |c| {
        var why: []const u8 = "";
        try std.testing.expect((try open(arena.allocator(), std.testing.allocator, std.testing.io, c[0], &why)) == null);
        try std.testing.expectStringStartsWith(why, c[1]);
    }
}

/// A captured kernel set is refused by name; native, unset and blank are not; a malformed value is refused unshown.
pub fn kernels(comptime open: anytype) !void {
    const a = std.testing.allocator;
    for ([_]?[]const u8{ null, "", "native" }) |raw| try std.testing.expect((try qwen3.kernelsRefusal(a, raw)) == null);
    const why = (try qwen3.kernelsRefusal(a, "/captured/set")).?;
    try std.testing.expectStringStartsWith(why, "Qwen3 uses embedded native images: unset TENSORFOLD_CUDA_KERNELS");
    const bad = (try qwen3.kernelsRefusal(a, "/secret-dir\nset")).?;
    defer a.free(bad);
    try std.testing.expectStringStartsWith(bad, "TENSORFOLD_CUDA_KERNELS: expected");
    try std.testing.expect(std.mem.indexOf(u8, bad, "secret-dir") == null);
    // the open reads the process's setting before any checkpoint or device access
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    _ = env.setenv("TENSORFOLD_CUDA_KERNELS", "/captured/set", 1);
    defer _ = env.unsetenv("TENSORFOLD_CUDA_KERNELS");
    var problem: []const u8 = "";
    const base: api.Open = .{ .dir = "missing-qwen3-checkpoint", .model_type = "qwen3" };
    try std.testing.expect((try open(arena.allocator(), std.testing.allocator, std.testing.io, base, &problem)) == null);
    try std.testing.expectStringStartsWith(problem, "Qwen3 uses embedded native images");
}
