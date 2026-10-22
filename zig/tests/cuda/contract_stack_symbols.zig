//! The stack's new symbols, each with its own numerical spec and served shapes.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;

/// The stack's new symbols, one spec each.
pub const Kind = enum {
    draw_slice_top,
    draw_rows_finish,
    draw_rows,
    prompt_bf16,
    prompt_fp8,
    prompt_chain,
    decode_bf16,
    decode_fp8,
};

/// The symbol each kind names.
pub const symbol = [_][]const u8{
    "tf_draw_slice_top",
    "tf_draw_rows_finish",
    "tf_draw_rows",
    "hpattn_kernel",
    "hpattn8_kernel",
    "chain_kernel",
    "tf_q35_attn_rows",
    "tf_q35_attn_rows8",
};

/// The rows a verify window's draw holds.
const draw_served = [_]c.Shape{
    .{ .rows = 1, .n = 1, .k = 1 },
    .{ .rows = 64, .n = 1, .k = 1 },
    .{ .rows = 128, .n = 1, .k = 1 },
};
/// The rows a prompt chunk's attention holds.
const prompt_served = [_]c.Shape{
    .{ .rows = 17, .n = 129, .k = 256 },
    .{ .rows = 129, .n = 129, .k = 256 },
    .{ .rows = 4096, .n = 1, .k = 64 },
};
/// The rows a decode window's tensor-core attention holds.
const decode_served = [_]c.Shape{
    .{ .rows = 1, .n = 1, .k = 256 },
    .{ .rows = 64, .n = 32, .k = 256 },
    .{ .rows = 128, .n = 32, .k = 256 },
};
/// The draw's rounding chain: the rule's temperature scales the logits once.
fn drawDepth(_: c.Shape) usize {
    return 1;
}
/// The prompt attention's chain: the key tile's group sums, then the tile's products.
fn promptDepth(s: c.Shape) usize {
    return s.k / 64 + 4;
}
/// The decode attention's chain: the key tile's group sums, then the tile's products.
fn decodeDepth(s: c.Shape) usize {
    return s.k / 64 + 4;
}

/// The spec of one symbol: its own name, its own bounds and its own served shapes.
pub fn spec(kind: Kind) c.Spec {
    return switch (kind) {
        .draw_slice_top => .{ .name = "cuda sample tf_draw_slice_top", .accum = .f32, .depth = drawDepth, .out = .raw, .ops = 1, .tiles = .{ .{ .size = 256 }, .{ .size = 8192 }, .{} }, .served = &draw_served },
        .draw_rows_finish => .{ .name = "cuda sample tf_draw_rows_finish", .accum = .f32, .depth = drawDepth, .out = .raw, .ops = 3, .tiles = .{ .{ .size = 1024 }, .{ .size = 8192 }, .{} }, .served = &draw_served },
        .draw_rows => .{ .name = "cuda sample tf_draw_rows", .accum = .f32, .depth = drawDepth, .out = .raw, .ops = 3, .tiles = .{ .{ .size = 1024 }, .{ .size = 1 }, .{} }, .served = &draw_served },
        .prompt_bf16 => .{ .name = "cuda prompt hpattn_kernel", .accum = .f32, .depth = promptDepth, .out = .bf16, .ops = 4, .tiles = .{ .{ .size = 256 }, .{ .size = 64 }, .{ .size = 32, .step = 32 } }, .served = &prompt_served },
        .prompt_fp8 => .{ .name = "cuda prompt hpattn8_kernel", .accum = .f32, .depth = promptDepth, .out = .bf16, .ops = 6, .tiles = .{ .{ .size = 256 }, .{ .size = 64 }, .{ .size = 32, .step = 32 } }, .served = &prompt_served },
        .prompt_chain => .{ .name = "cuda prompt tf_gdn_prefill chain_kernel", .accum = .f32, .depth = promptDepth, .out = .bf16, .ops = 4, .tiles = .{ .{ .size = 128 }, .{ .size = 64 }, .{ .size = 32, .step = 32 } }, .served = &prompt_served },
        .decode_bf16 => .{ .name = "cuda qwen35 tf_q35_attn_rows", .accum = .f32, .depth = decodeDepth, .out = .bf16, .ops = 4, .tiles = .{ .{ .size = 256 }, .{ .size = 32 }, .{ .size = 32, .step = 32 } }, .served = &decode_served },
        .decode_fp8 => .{ .name = "cuda qwen35 tf_q35_attn_rows8", .accum = .f32, .depth = decodeDepth, .out = .bf16, .ops = 6, .tiles = .{ .{ .size = 256 }, .{ .size = 32 }, .{ .size = 32, .step = 32 } }, .served = &decode_served },
    };
}
