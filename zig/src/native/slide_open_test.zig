//! Unsupported learning is refused before an absent checkpoint or device can be opened.
const std = @import("std");

pub fn omni(comptime open: anytype) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    const text =
        \\{"model_type":"nemotron_h_omni","quantization":{"bits":8,"group_size":64},"llm_config":{
        \\"model_type":"nemotron_h","hidden_size":64,"vocab_size":64,"mamba_num_heads":1,"mamba_head_dim":64,
        \\"n_groups":1,"ssm_state_size":128,"conv_kernel":4,"num_attention_heads":1,"num_key_value_heads":1,
        \\"head_dim":64,"n_routed_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":64,
        \\"moe_shared_expert_intermediate_size":128,"hybrid_override_pattern":"E"}}
    ;
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = text });
    var problem: []const u8 = "";
    try std.testing.expect((try open(a, a, io, .{ .dir = dir, .model_type = "nemotron_h_omni", .context = 32, .slide = true }, &problem)) == null);
    try std.testing.expectStringStartsWith(problem, "--slide: nemotron_h_omni learns 4-bit groups of 64");
    try std.testing.expect(std.mem.indexOf(u8, problem, "backbone.embeddings is 8-bit") != null);
}
const api = @import("engine_api");

pub fn check(comptime open: anytype, backend: []const u8, models: []const []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (models) |model| {
        var why: []const u8 = "";
        const o: api.Open = .{ .dir = "missing-slide-checkpoint", .model_type = model, .slide = true };
        try std.testing.expect((try open(a, std.testing.allocator, std.testing.io, o, &why)) == null);
        try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "--slide: {s} checkpoints do not learn on {s} yet", .{ model, backend }), why);
    }
}
