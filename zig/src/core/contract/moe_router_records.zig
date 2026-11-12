//! Expert-first router launch resource records.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "qwen_router_bf16 tf_moe_router_expert_first", .symbol = "tf_moe_router_expert_first", .image = "qwen_router_bf16", .threads = 256 }, .usage = @import("gdn_chain_records.zig").gb10(.{ .regs = 63, .static = 16512, .bound = 256 }), .later_pending = true },
};

test {
    _ = @import("qwen_glue_receipts_test.zig");
}
