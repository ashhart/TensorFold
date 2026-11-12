//! Compare the real loader's recorded libraries with the source inventory on each device class.
const std = @import("std");
const mtl = @import("metal");
const kernels = @import("src/families/lfm2_moe/kernels.zig");
const inventory = @import("runtime_sources/lfm2.zig");
const Set = @import("runtime_sources/set.zig").Set;

test "LFM2 runtime inventory: library bytes and options match the real loader on M1 through M5" {
    for ([_]u32{ 13, 14, 15, 16, 17 }) |generation| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const device = mtl.Device{ .units = generation >= 17, .gen = generation };
        mtl.Library.allocator = arena.allocator();
        mtl.Library.count = 0;
        var loaded = try kernels.load(device);
        defer loaded.deinit();
        var set = Set{ .a = arena.allocator(), .units = device.units, .generation = generation };
        try inventory.add(&set, device);
        try std.testing.expect(mtl.Library.count > 0);
        try std.testing.expectEqual(mtl.Library.count, set.entries.items.len);
        for (mtl.Library.seen[0..mtl.Library.count], set.entries.items) |seen, entry| {
            try std.testing.expect(std.mem.eql(u8, seen.text, entry.text));
            try std.testing.expectEqualDeep(seen.options, entry.options);
        }
        try std.testing.expectEqual(device.units, loaded.attn_nax != null);
        try std.testing.expectEqual(device.units, loaded.prompt_mask != null);
    }
}
