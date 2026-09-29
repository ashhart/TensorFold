//! Compare compiled graphs with mlx-lm, including signed-zero output bits.
const std = @import("std");
const mx = @import("mlx.zig");
const ops = @import("prefill_ops.zig");
pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var store = @import("checkpoint.zig").Store.init(64);
    defer store.deinit();
    var path: [4096]u8 = undefined;
    try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/arrays.safetensors", .{dir}), "", "");
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/cases.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { key: []const u8, kind: ops.Kind, inputs: usize };
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    var compiled = ops.Ops{};
    defer compiled.deinit();
    // Reuse each cached compiled graph, including after its input shape changes.
    for (0..2) |_| for (cases.value) |case| {
        errdefer std.debug.print("Prefill fixture {s}, {s}\n", .{ case.key, @tagName(case.kind) });
        var s = mx.Scope{};
        defer s.deinit();
        var inputs: [6]mx.Array = undefined;
        if (case.inputs > inputs.len) return error.InvalidFixture;
        for (0..case.inputs) |i| inputs[i] = try store.field(case.key, try std.fmt.bufPrint(&path, "input{d}", .{i}));
        const out = switch (case.kind) {
            inline else => |kind| try compiled.call(&s, kind, inputs[0..case.inputs]),
        };
        try @import("sampling_checks.zig").equal(&s, out, try store.field(case.key, "expected"));
    };
    std.debug.print("PASS: {d} prefill graphs, fresh/cached, including all 65,280 finite BF16 inputs.\n", .{cases.value.len});
}
