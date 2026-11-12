//! A plain BF16 Qwen3.5 checkpoint on CUDA (Clef-Flash decisions, LightOnOCR-3 chat): its refusals before device work.
const std = @import("std");
const geometry = @import("cuda_geometry.zig");
const tower = @import("qwen_vision");

/// The joint schema head's config beside the weights marks a Clef checkpoint.
pub const head_file = "joint_head_config.json";

/// What `open` asks of a checkpoint, as the flags set it.
pub const Asked = struct { drafter: bool = false, slide: bool = false, vision: bool = false, kv8: bool = false, prompt_int8: bool = false };

/// True: BF16 decisions; false: other formats or image chat; refused: no head or tower, flags, a quantized/tied head.
pub fn check(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, asked: Asked) !bool {
    const g = geometry.Geometry.read(gpa, io, dir) catch return false;
    const head = present(gpa, io, dir);
    if (!g.bf16) return if (head) error.QwenHeadNeedsBf16 else false;
    if (!head) {
        if (!try chat(gpa, io, dir, g.hidden)) return error.QwenBf16DecisionOnly;
        // a BF16 chat checkpoint is qualified on its own drafts, bf16 caches and bf16 prompts only
        if (asked.drafter) return error.QwenBf16ChatDrafter;
        if (asked.slide) return error.QwenBf16ChatSlide;
        if (asked.kv8) return error.QwenBf16ChatKv8;
        if (asked.prompt_int8) return error.QwenBf16ChatPromptInt8;
        return false;
    }
    // the joint schema head is qualified on untied backbones; a tied one would read a head packed from the embedding
    if (g.tied) return error.QwenJointTied;
    if (asked.drafter) return error.QwenBf16Drafter;
    if (asked.slide) return error.QwenBf16Slide;
    if (asked.vision) return error.QwenBf16Vision;
    if (asked.kv8) return error.QwenBf16Kv8;
    if (asked.prompt_int8) return error.QwenBf16PromptInt8;
    return true;
}

/// A headless BF16 checkpoint chats if the CUDA tower reads its tower at the text's width; false without a tower.
pub fn chat(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, hidden: usize) !bool {
    const read = tower.Settings.read(gpa, io, dir) catch |err| return if (err == error.UnsupportedQwenVision) error.QwenBf16ChatTower else err;
    const s = read orelse return false;
    if (s.tower.out_hidden != hidden) return error.QwenBf16ChatTower;
    return true;
}

/// Whether `dir` ships the joint schema head's config.
pub fn present(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(gpa, &.{ dir, head_file }) catch return false;
    defer gpa.free(path);
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

/// The words of each refusal above.
pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.QwenBf16DecisionOnly => "BF16 Qwen3.5 serves as a decision backbone only: this checkpoint has no joint_head_config.json",
        error.QwenHeadNeedsBf16 => "the joint schema head needs the BF16 checkpoint: this one is quantized",
        error.QwenJointTied => "the joint schema head is qualified on untied BF16 checkpoints: this one ties its head to the embedding (tie_word_embeddings)",
        error.QwenBf16Drafter => "--drafter: a BF16 Qwen3.5 decision backbone drafts nothing; serve without --drafter",
        error.QwenBf16Slide => "--slide: a BF16 Qwen3.5 decision backbone does not learn; serve without --slide",
        error.QwenBf16Vision => "--vision: a BF16 Qwen3.5 decision backbone reads text records only; serve without --vision",
        error.QwenBf16Kv8 => "--kv-cache fp8: the joint schema head is qualified on bf16 caches only; serve without it",
        error.QwenBf16PromptInt8 => "--prompt-precision int8: the joint schema head is qualified on bf16 prompts only; serve without it",
        error.QwenBf16ChatTower => "a BF16 Qwen3.5 checkpoint without a joint head serves image chat, and the CUDA tower cannot read its image tower",
        error.QwenBf16ChatDrafter => "--drafter: a BF16 Qwen3.5 image checkpoint drafts from its own replies; serve without --drafter",
        error.QwenBf16ChatSlide => "--slide: a BF16 Qwen3.5 image checkpoint does not learn; serve without --slide",
        error.QwenBf16ChatKv8 => "--kv-cache fp8: a BF16 Qwen3.5 image checkpoint is qualified on bf16 caches only; serve without it",
        error.QwenBf16ChatPromptInt8 => "--prompt-precision int8: a BF16 Qwen3.5 image checkpoint is qualified on bf16 prompts only; serve without it",
        error.QwenJointPages => "TF_QWEN_KV_PAGES: the joint schema head plans a window a stream; unset it",
        else => null,
    };
}

const config_text =
    \\{"model_type":"qwen3_5","tie_word_embeddings":false,QUANT"text_config":{"model_type":"qwen3_5_text",
    \\ "attention_bias":false,"attn_output_gate":true,"hidden_act":"silu","head_dim":256,"linear_key_head_dim":128,
    \\ "linear_value_head_dim":128,"linear_conv_kernel_dim":4,"hidden_size":4096,"intermediate_size":12288,
    \\ "num_hidden_layers":4,"vocab_size":248320,"linear_num_key_heads":16,"linear_num_value_heads":32,
    \\ "num_attention_heads":16,"num_key_value_heads":4,"rms_norm_eps":1e-06,"max_position_embeddings":262144,
    \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
    \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
;

test "a bf16 checkpoint without a joint head is refused as decision-only on CUDA; with one, every flag the head lacks" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    const plain = try std.mem.replaceOwned(u8, a, config_text, "QUANT", "");
    defer a.free(plain);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = plain });
    try std.testing.expectError(error.QwenBf16DecisionOnly, check(a, io, dir, .{}));
    try std.testing.expectStringStartsWith(explain(error.QwenBf16DecisionOnly).?, "BF16 Qwen3.5 serves as a decision backbone only");
    try tmp.dir.writeFile(io, .{ .sub_path = head_file, .data = "{}" });
    try std.testing.expectError(error.QwenBf16Drafter, check(a, io, dir, .{ .drafter = true }));
    try std.testing.expectError(error.QwenBf16Slide, check(a, io, dir, .{ .slide = true }));
    try std.testing.expectError(error.QwenBf16Vision, check(a, io, dir, .{ .vision = true }));
    try std.testing.expectError(error.QwenBf16Kv8, check(a, io, dir, .{ .kv8 = true }));
    try std.testing.expectError(error.QwenBf16PromptInt8, check(a, io, dir, .{ .prompt_int8 = true }));
    try std.testing.expect(try check(a, io, dir, .{}));
    // a quantized checkpoint that ships the head is refused; without the head it opens as before
    const mlx = try std.mem.replaceOwned(u8, a, config_text, "QUANT", "\"quantization\":{\"group_size\":64,\"bits\":4},");
    defer a.free(mlx);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = mlx });
    try std.testing.expectError(error.QwenHeadNeedsBf16, check(a, io, dir, .{}));
    try tmp.dir.deleteFile(io, head_file);
    try std.testing.expect(!try check(a, io, dir, .{ .drafter = true, .kv8 = true }));
    for ([_]anyerror{ error.QwenHeadNeedsBf16, error.QwenBf16Drafter, error.QwenJointPages }) |err| try std.testing.expect(explain(err) != null);
}

test "a bf16 image checkpoint without a joint head opens as chat on CUDA, refusing unqualified flags and a foreign tower" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    const lighton = @embedFile("fixtures/lighton-ocr-0.8b.json");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = lighton });
    try std.testing.expect(!try check(a, io, dir, .{}));
    try std.testing.expect(!try check(a, io, dir, .{ .vision = true }));
    try std.testing.expectError(error.QwenBf16ChatDrafter, check(a, io, dir, .{ .drafter = true }));
    try std.testing.expectError(error.QwenBf16ChatSlide, check(a, io, dir, .{ .slide = true }));
    try std.testing.expectError(error.QwenBf16ChatKv8, check(a, io, dir, .{ .kv8 = true }));
    try std.testing.expectError(error.QwenBf16ChatPromptInt8, check(a, io, dir, .{ .prompt_int8 = true }));
    for ([_]anyerror{ error.QwenBf16ChatDrafter, error.QwenBf16ChatSlide, error.QwenBf16ChatKv8, error.QwenBf16ChatPromptInt8 }) |err|
        try std.testing.expectStringStartsWith(explain(err).?, "--");
    // a tower whose merger is not the text's width is refused before any weights, without or with --vision
    const narrow = try std.mem.replaceOwned(u8, a, lighton, "\"out_hidden_size\": 1024", "\"out_hidden_size\": 768");
    defer a.free(narrow);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = narrow });
    try std.testing.expectError(error.QwenBf16ChatTower, check(a, io, dir, .{}));
    try std.testing.expectError(error.QwenBf16ChatTower, check(a, io, dir, .{ .vision = true }));
    try std.testing.expect(!std.mem.startsWith(u8, explain(error.QwenBf16ChatTower).?, "--"));
    // a tower geometry the CUDA tower does not run (14-pixel patches) at the same width, refused the same way
    const patch14 = try std.mem.replaceOwned(u8, a, lighton, "\"patch_size\": 16", "\"patch_size\": 14");
    defer a.free(patch14);
    try std.testing.expect(!std.mem.eql(u8, patch14, lighton));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = patch14 });
    try std.testing.expectError(error.QwenBf16ChatTower, check(a, io, dir, .{}));
    // a joint head beside a tied backbone is refused by name before any flag: no Clef checkpoint ties its head
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = lighton });
    try tmp.dir.writeFile(io, .{ .sub_path = head_file, .data = "{}" });
    try std.testing.expectError(error.QwenJointTied, check(a, io, dir, .{}));
    try std.testing.expectError(error.QwenJointTied, check(a, io, dir, .{ .vision = true }));
    try std.testing.expectStringStartsWith(explain(error.QwenJointTied).?, "the joint schema head is qualified on untied");
}
