//! The load's layers, each read into the page cache by TF_LOAD_PREFETCH threads while the loader packs it.

const std = @import("std");
const core = @import("core");

/// Layers `ids` into `out` through `L` (cuda_weights' loader), each layer's checkpoint pages released after it.
pub fn layers(L: anytype, prefix: []const u8, ids: []const u32, out: anytype) !void {
    const count = configured(@import("cuda_ranks.zig").environment("TF_LOAD_PREFETCH"), @import("cuda_ranks.zig").environment("TF_GLM_LOAD_PREFETCH")) catch |err| {
        std.log.err("TF_LOAD_PREFETCH (legacy TF_GLM_LOAD_PREFETCH): 0 to {d} reader threads", .{core.checkpoint.max_readers});
        return err;
    };
    _ = prefix;
    try layersWithCount(L, ids, out, count);
}

pub fn layersWithCount(L: anytype, ids: []const u32, out: anytype, count: usize) !void {
    var ahead: u64 = 0;
    for (ids, out) |i, *l| {
        if (i >= L.w.config.layers) return error.BadLayer;
        {
            var reading = try @import("cuda_load_prefetch.zig").begin(L.ck, i, count);
            defer reading.deinit();
            l.* = try L.layer(i);
            ahead += try reading.check();
        }
        L.ck.release(); // a rank reads about half the checkpoint: its pages would crowd the GPU's allocations out
    }
    if (count > 0) std.log.info("weights: {d} readers read {d:.1} GiB ahead of the loader", .{ count, @as(f64, @floatFromInt(ahead)) / (1 << 30) });
}

/// Readers when TF_GLM_LOAD_PREFETCH is unset.
pub const default_readers = 16;

/// TF_GLM_LOAD_PREFETCH as a reader count: unset for `default_readers`, 0 for none (the loader faults every page in).
pub fn readers(value: ?[]const u8) !usize {
    const n = std.fmt.parseInt(usize, value orelse return default_readers, 10) catch return error.BadSetting;
    return if (n <= core.checkpoint.max_readers) n else error.BadSetting;
}

test "TF_GLM_LOAD_PREFETCH: 16 when unset, none at 0, a count up to the limit, a refusal past it or for a word" {
    try std.testing.expectEqual(@as(usize, 16), try readers(null));
    try std.testing.expectEqual(@as(usize, 0), try readers("0"));
    try std.testing.expectEqual(@as(usize, 8), try readers("8"));
    try std.testing.expectEqual(@as(usize, 16), try readers("16"));
    try std.testing.expectError(error.BadSetting, readers("17"));
    try std.testing.expectError(error.BadSetting, readers("eight"));
}

/// The universal option takes precedence; an unset option retains the existing GLM reader setting.
pub fn configured(global: ?[]const u8, legacy: ?[]const u8) !usize {
    return if (global) |v| core.checkpoint_read_ahead.readerCount(v) else readers(legacy);
}

test "load prefetch: GLM universal option overrides the legacy setting including zero" {
    try std.testing.expectEqual(@as(usize, 16), try configured(null, null));
    try std.testing.expectEqual(@as(usize, 0), try configured("0", "16"));
    try std.testing.expectEqual(@as(usize, 4), try configured("4", "0"));
    try std.testing.expectEqual(@as(usize, 0), try configured(null, "0"));
    try std.testing.expectError(error.BadLoadPrefetch, configured("17", "0"));
    try std.testing.expectError(error.BadSetting, configured(null, "word"));
}
