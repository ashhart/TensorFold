//! The load's layers, each read into the page cache by TF_GLM_LOAD_PREFETCH threads while the loader packs it.

const std = @import("std");
const core = @import("core");

/// Layers `ids` into `out` through `L` (cuda_weights' loader), each layer's checkpoint pages released after it.
pub fn layers(L: anytype, prefix: []const u8, ids: []const u32, out: anytype) !void {
    const count = readers(@import("cuda_ranks.zig").environment("TF_GLM_LOAD_PREFETCH")) catch |err| {
        std.log.err("TF_GLM_LOAD_PREFETCH: 0 to {d} reader threads", .{core.checkpoint.max_readers});
        return err;
    };
    var ahead: u64 = 0;
    for (ids, out) |i, *l| {
        if (i >= L.w.config.layers) return error.BadLayer;
        {
            var name: [96]u8 = undefined;
            var reading = L.ck.prefetch(try std.fmt.bufPrint(&name, "{s}layers.{d}.", .{ prefix, i }), count);
            defer ahead += reading.join();
            l.* = try L.layer(i);
        }
        L.ck.release(); // a rank reads about half the checkpoint: its pages would crowd the GPU's allocations out
    }
    if (count > 0) std.log.info("weights: {d} readers read {d:.1} GiB ahead of the loader", .{ count, @as(f64, @floatFromInt(ahead)) / (1 << 30) });
}

/// Readers when TF_GLM_LOAD_PREFETCH is unset.
pub const default_readers = 8;

/// TF_GLM_LOAD_PREFETCH as a reader count: unset for `default_readers`, 0 for none (the loader faults every page in).
pub fn readers(value: ?[]const u8) !usize {
    const n = std.fmt.parseInt(usize, value orelse return default_readers, 10) catch return error.BadSetting;
    return if (n <= core.checkpoint.max_readers) n else error.BadSetting;
}

test "TF_GLM_LOAD_PREFETCH: 8 when unset, none at 0, a count up to the limit, a refusal past it or for a word" {
    try std.testing.expectEqual(@as(usize, default_readers), try readers(null));
    try std.testing.expectEqual(@as(usize, 0), try readers("0"));
    try std.testing.expectEqual(@as(usize, 8), try readers("8"));
    try std.testing.expectEqual(@as(usize, 16), try readers("16"));
    try std.testing.expectError(error.BadSetting, readers("17"));
    try std.testing.expectError(error.BadSetting, readers("eight"));
}
