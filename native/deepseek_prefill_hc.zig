const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const Model = @import("deepseek.zig").Model;

pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/hc.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Group = struct { checkpoint: []const u8, cases: []const []const u8 };
    const groups = try std.json.parseFromSlice([]const Group, mx.allocator, bytes, .{});
    defer groups.deinit();
    if (groups.value.len == 0) return error.EmptyFixtures;
    var count: usize = 0;
    for (groups.value) |group| {
        var m = try Model.init(io, try std.fmt.bufPrint(&path, "{s}/{s}", .{ dir, group.checkpoint }));
        defer m.deinit();
        try m.loadDraft(io, try std.fmt.bufPrint(&path, "{s}/{s}/drafter", .{ dir, group.checkpoint }));
        if (group.cases.len == 0) return error.EmptyFixtures;
        for (group.cases) |case| {
            errdefer std.debug.print("DeepSeek prefill HC fixture failed: {s}/{s}\n", .{ group.checkpoint, case });
            var store = cp.Store.init(32);
            defer store.deinit();
            try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case }), "", "");
            var s = mx.Scope{};
            defer s.deinit();
            const x = try store.get("input");
            const branch = try store.get("branch");
            try std.testing.expectError(error.InvalidTensorShape, m.hc(&s, 0, "attn", try s.reshape(x, &.{ mx.dim(x, 0), -1 })));
            inline for (.{ false, true }) |mtp| {
                const prefix = if (mtp) "mtp" else "target";
                inline for (.{ "attn", "ffn" }) |kind| {
                    const result = try m.hc(&s, if (mtp) m.cache.len else 0, kind, x);
                    inline for (.{ "collapsed", "post", "comb" }, 0..) |key, i| {
                        errdefer std.debug.print("Mismatch in {s}-{s}-{s}\n", .{ prefix, kind, key });
                        try @import("variant_checks.zig").equalBits(&s, result[i], try store.get(prefix ++ "-" ++ kind ++ "-" ++ key));
                    }
                    try @import("variant_checks.zig").equalBits(&s, try Model.expand(&s, x, branch, result[1], result[2]), try store.get(prefix ++ "-" ++ kind ++ "-expanded"));
                }
                errdefer std.debug.print("Mismatch in {s}-head\n", .{prefix});
                try @import("variant_checks.zig").equalBits(&s, try m.head(&s, x, mtp), try store.get(prefix ++ "-head"));
            }
            count += 1;
        }
    }
    std.debug.print("PASS: {d} DeepSeek batched hyper-connection/head cases, target/MTP, packed parameters and production hidden width\n", .{count});
}
