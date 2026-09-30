const std = @import("std");
const mx = @import("mlx.zig");
const glm = @import("glm.zig");

pub fn check(io: std.Io, dir: []const u8, output: []const u8, tiles: bool) !void {
    try glm.Model.prepareRuntime();
    try mx.init();
    defer mx.shutdown();
    var m = try glm.Model.init(io, dir);
    defer m.deinit();
    if (tiles) m.kernels.flash_prefill.decision = true;
    try std.Io.Dir.cwd().createDirPath(io, output);
    var buf: [256]u8 = undefined;
    for ([_]usize{ 17, 63, 64, 511, 512, 513, 2048, 17, 1 }, 0..) |count, step| {
        var tokens: [2048]i32 = undefined;
        var next: [2048]i32 = undefined;
        for (tokens[0..count], next[0..count], 0..) |*id, *after, j| {
            const position = m.position + @as(i32, @intCast(j));
            id.* = 1 + @mod(position, 97);
            after.* = 1 + @mod(position + 1, 97);
        }
        m.trace_dir = if (count > 16) output else null;
        var pass = try m.prefill(tokens[0..count]);
        defer pass.deinit();
        const s = &pass.scope;
        try save(s, output, try std.fmt.bufPrint(&buf, "hidden-{d}", .{step}), pass.hidden);
        try save(s, output, try std.fmt.bufPrint(&buf, "logits-{d}", .{step}), try s.slice(pass.logits, 0, mx.dim(pass.logits, 0) - 1, mx.dim(pass.logits, 0)));
        if (pass.prefilled) try std.testing.expectError(error.InvalidCommit, m.commit(&pass, count - 1));
        try std.testing.expectError(error.InvalidCommit, m.commitMtp(&pass, count));
        try m.commit(&pass, count);
        try std.testing.expectError(error.InvalidCommit, m.commit(&pass, count));
        for (m.cache, 0..) |cache, layer| inline for (comptime std.meta.fieldNames(glm.Cache)) |field| if (@field(cache, field).ctx != null) {
            try save(s, output, try std.fmt.bufPrint(&buf, "cache-{d}-{d}-{s}", .{ step, layer, field }), @field(cache, field));
        };
        var head = try m.forwardMtp(pass.hidden, next[0..count]);
        defer head.deinit();
        inline for (.{ "hidden", "logits", "mtp_projection", "mtp_input" }) |field| try save(&head.scope, output, try std.fmt.bufPrint(&buf, "head-{d}-{s}", .{ step, field }), @field(head, field));
        try std.testing.expectError(error.InvalidCommit, m.commit(&head, count));
        const keep = if (step == 7) count - 3 else count;
        try m.commitMtp(&head, keep);
        try std.testing.expectError(error.InvalidCommit, m.commitMtp(&head, 1));
        if (keep < count) {
            var replay = try m.forwardMtp(try s.slice(pass.hidden, 0, @intCast(keep), @intCast(count)), next[keep..count]);
            defer replay.deinit();
            try save(&replay.scope, output, "head-replay", replay.logits);
            try m.commitMtp(&replay, count - keep);
        }
        inline for (comptime std.meta.fieldNames(glm.Cache)) |field| if (@field(m.mtp_cache, field).ctx != null) {
            try save(s, output, try std.fmt.bufPrint(&buf, "head-cache-{d}-{s}", .{ step, field }), @field(m.mtp_cache, field));
        };
        try std.testing.expectEqual(m.position, m.mtp_position);
        std.debug.print("GLM prefill and draft cache commit at {d} tokens.\n", .{m.position});
    }
    m.trace_dir = null;
    for (0..4) |step| {
        var pass = try m.forward(&.{@as(i32, @intCast(step)) + 200});
        defer pass.deinit();
        try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "continuation-{d}", .{step}), pass.logits);
        try m.commit(&pass, 1);
    }
    m.reset();
    var long_prompt: [2051]i32 = undefined;
    for (&long_prompt, 0..) |*token, j| token.* = 1 + @as(i32, @intCast(j % 97));
    try @import("session_checks.zig").checkSyntheticNeuralPrompt(&m, io, &long_prompt);
}
fn save(s: *mx.Scope, dir: []const u8, name: []const u8, value: mx.Array) !void {
    const path = try std.fmt.allocPrintSentinel(mx.allocator, "{s}/{s}.npy", .{ dir, name }, 0);
    defer mx.allocator.free(path);
    const out = try s.cast(value, mx.f32t);
    try mx.eval(out);
    try mx.check(mx.c.mlx_save(path, out));
}
