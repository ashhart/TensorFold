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
    for (models) |model| try message(open, model, try std.fmt.allocPrint(a, "--slide: {s} checkpoints do not learn on {s} yet", .{ model, backend }));
}

pub fn message(comptime open: anytype, model: []const u8, want: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var why: []const u8 = "";
    const o: api.Open = .{ .dir = "missing-slide-checkpoint", .model_type = model, .slide = true };
    try std.testing.expect((try open(arena.allocator(), std.testing.allocator, std.testing.io, o, &why)) == null);
    try std.testing.expectEqualStrings(want, why);
}

pub fn qwen27RopeChecks(comptime open: anytype) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var why: []const u8 = "";
    try std.testing.expect((try open(a, std.testing.allocator, io, .{ .dir = "d", .model_type = "qwen3_5", .rope_yarn = 2.0 }, &why)) == null);
    try std.testing.expectEqualStrings("--rope-yarn: the Metal engine has no YaRN rope; serve without --rope-yarn", why);
    try std.testing.expect((try open(a, std.testing.allocator, io, .{ .dir = "d", .model_type = "qwen3_5", .context = 300000 }, &why)) == null);
    try std.testing.expectEqualStrings("--context 300000 is outside this Qwen checkpoint's 1 to 262,144 tokens", why);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg_text =
        \\{"model_type":"sensenova_loop","loop_settings":{"start":0,"end":4,"eval_count":2,"counts":[2]},
        \\"eos_token_id":1,"tie_word_embeddings":false,"text_config":{"hidden_size":64,"intermediate_size":128,
        \\"num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"vocab_size":97,
        \\"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
        \\"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":16,
        \\"linear_value_head_dim":16,"linear_conv_kernel_dim":4,"max_position_embeddings":262144,
        \\"partial_rotary_factor":0.5,"tie_word_embeddings":false,"rope_parameters":{"rope_type":"yarn","factor":1.0,
        \\"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.5,
        \\"mrope_interleaved":true,"mrope_section":[3,3,2]}}}
    ;
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg_text });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    try std.testing.expect((try open(a, std.testing.allocator, io, .{ .dir = dir, .model_type = "sensenova_loop", .context = 300000 }, &why)) == null);
    try std.testing.expectEqualStrings("--context 300000 is outside this SenseNova checkpoint's 1 to 262144 tokens", why);
    why = "";
    if (try open(a, std.testing.allocator, io, .{ .dir = dir, .model_type = "sensenova_loop", .context = 300000, .rope_yarn = 2.0 }, &why)) |*o| o.close(o.ctx);
    try std.testing.expect(std.mem.indexOf(u8, why, "outside this SenseNova checkpoint's") == null);
    try std.testing.expect(std.mem.indexOf(u8, why, "no YaRN rope") == null);
}
