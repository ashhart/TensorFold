//! Confidential CLI options reach production backend and family admission without any device or checkpoint access.
const std = @import("std");
const cli = @import("cli");
const gate = @import("native").refusals;
const metal_spill = @import("native").metal_spill;
const t = std.testing;
const Promptless = struct {
    pub const keeps_no_prompts = true;
};

fn allowed(why: ?[]const u8) !void {
    if (why) |message| std.debug.print("confidential admission refusal: {s}\n", .{message});
    try t.expect(why == null);
}

test "confidential admission: default CLI options pass the real CUDA backend without checkpoint slots" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    for ([_][]const u8{ "nemotron_h", "qwen3_5", "glm5_next" }) |model_type| {
        const open = try cli.request(arena.allocator(), "not-opened", model_type, .{ .confidential = true });
        try t.expectEqual(@as(?f64, 0), open.prompt_cache_gib);
        try allowed(gate.backend(open));
    }
}

test "confidential admission: GLM family receives no synthesized checkpoint-slot option" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const open = try cli.request(arena.allocator(), "not-opened", "glm5_next", .{ .confidential = true });
    try allowed(gate.backend(open));
    try allowed(gate.family(@import("glm_cuda").native, open));
}

test "confidential admission: CLI-disabled cache passes the real promptless-family refusal" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const open = try cli.request(arena.allocator(), "not-opened", "promptless", .{ .confidential = true });
    try allowed(gate.backend(open));
    try allowed(gate.family(Promptless, open));
    const ordinary = try cli.request(arena.allocator(), "not-opened", "promptless", .{});
    try allowed(gate.backend(ordinary));
    try allowed(gate.family(Promptless, ordinary));
    const positive = try cli.request(arena.allocator(), "not-opened", "promptless", .{ .prompt_cache_gib = 1 });
    try t.expectEqualStrings(gate.no_prompts, gate.family(Promptless, positive).?);
}

test "confidential admission: default CLI options pass the real Metal refusal without checkpoint slots" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const open = try cli.request(arena.allocator(), "not-opened", "glm5_next", .{ .confidential = true });
    try allowed(metal_spill.refusal(open));
    var restored = open;
    restored.keep = 0;
    try t.expectStringStartsWith(metal_spill.refusal(restored) orelse "", "--checkpoint-slots sets how many");
}
