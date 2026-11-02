//! Expert-first router launch; usage stays pending until compiled on each supported SM.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "moe tf_moe_router_expert_first", .symbol = "tf_moe_router_expert_first", .image = "moe", .threads = 256 }, .usage = registry.none },
};
