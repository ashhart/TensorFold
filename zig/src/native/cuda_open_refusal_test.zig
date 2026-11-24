//! A flag a registered CUDA family does not serve is refused through the production opener, never ignored.
const std = @import("std");
const api = @import("engine_api");
const cuda = @import("cuda.zig");
const refusals = @import("cuda_refusal.zig");
const family_memory = @import("cuda_family_memory.zig");
const ranked = @import("cuda_open.zig").ranked;

const asks = [_]api.Open{
    .{ .dir = "/missing-refusal-test-checkpoint", .model_type = "", .kv8 = true },
    .{ .dir = "/missing-refusal-test-checkpoint", .model_type = "", .prompt_int8 = true },
    .{ .dir = "/missing-refusal-test-checkpoint", .model_type = "", .loops = 2 },
    .{ .dir = "/missing-refusal-test-checkpoint", .model_type = "", .vision = .{} },
    .{ .dir = "/missing-refusal-test-checkpoint", .model_type = "", .drafter = "/missing-refusal-test-drafter" },
};

/// The registry entry `open` takes for `o` is the first whose model_type matches and whose ranks admit it.
fn takes(comptime i: usize, o: api.Open) bool {
    inline for (cuda.registry, 0..) |G, j| if (j < i and std.mem.eql(u8, o.model_type, G.model_type) and ranked(G, o)) return false;
    return true;
}

test "cuda.open refuses each flag a registered family does not serve, with that family's own reason" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var refused: usize = 0;
    inline for (cuda.registry, 0..) |F, i| for (asks) |ask| {
        var o = ask;
        o.model_type = F.model_type;
        if (comptime @hasDecl(F, "ranks")) o.tp = F.ranks;
        const f = family_memory.familyOpen(F, o);
        if (!takes(i, o) or try refusals.place(F, a, f) != null) continue;
        const why = refusals.family(F, f) orelse continue;
        var problem: []const u8 = "";
        try std.testing.expect(try cuda.open(a, std.testing.allocator, std.testing.io, o, &problem) == null);
        try std.testing.expectEqualStrings(why, problem);
        refused += 1;
    };
    try std.testing.expect(refused >= cuda.registry.len); // every family refuses at least one of these flags
}
