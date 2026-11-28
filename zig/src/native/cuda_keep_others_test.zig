//! TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS reaches every CUDA family whose host prompt cache cuda.zig builds, and no other.
const std = @import("std");
const api = @import("engine_api");
const registry = @import("cuda_families.zig").registry;

test "keep others: the registry row names exactly the CUDA families with a host prompt cache" {
    const row = api.profile.entry(.TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS);
    var cached: usize = 0;
    inline for (registry, 0..) |F, i| {
        const keeps = comptime @hasDecl(F, "snap_save");
        try std.testing.expectEqual(keeps, row.applies(F.model_type, "cuda"));
        // a model_type two entries serve (Ripple at one rank and at --tp 2) is one family of the row
        const again = comptime for (registry, 0..) |E, j| {
            if (j < i and std.mem.eql(u8, E.model_type, F.model_type)) break true;
        } else false;
        cached += @intFromBool(keeps and !again);
    }
    try std.testing.expectEqual(row.families.len, cached);
    try std.testing.expectEqual(api.profile.Class.identical, row.class);
    for ([_][]const u8{ "metal", "hip", "xpu" }) |backend| try std.testing.expect(!row.applies("glm5_next", backend));
}
