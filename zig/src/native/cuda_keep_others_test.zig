//! TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS reaches every CUDA family whose host prompt cache cuda.zig builds, and no other.
const std = @import("std");
const api = @import("engine_api");
const registry = @import("cuda_families.zig").registry;

test "keep others: the registry row names exactly the CUDA families with a host prompt cache" {
    const row = api.profile.entry(.TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS);
    var cached: usize = 0;
    inline for (registry) |F| {
        const keeps = comptime @hasDecl(F, "snap_save");
        try std.testing.expectEqual(keeps, row.applies(F.model_type, "cuda"));
        cached += @intFromBool(keeps);
    }
    try std.testing.expectEqual(row.families.len, cached);
    try std.testing.expectEqual(api.profile.Class.identical, row.class);
    for ([_][]const u8{ "metal", "hip", "xpu" }) |backend| try std.testing.expect(!row.applies("glm5_next", backend));
}
