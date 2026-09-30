const std = @import("std");
const mx = @import("mlx.zig");
const ds = @import("deepseek.zig");

pub fn check(io: std.Io, dir: []const u8, output: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var m = try ds.Model.init(io, dir);
    defer m.deinit();
    var buf: [4096]u8 = undefined;
    try m.loadDraft(io, try std.fmt.bufPrint(&buf, "{s}/drafter", .{dir}));
    try std.Io.Dir.cwd().createDirPath(io, output);
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
        inline for (.{ "hidden", "streams", "logits" }) |field| try save(s, output, try std.fmt.bufPrint(&buf, "{s}-{d}", .{ field, step }), @field(pass, field));
        if (pass.prefilled) try std.testing.expectError(error.InvalidCommit, m.commit(&pass, count - 1));
        try std.testing.expectError(error.InvalidCommit, m.commitMtp(&pass, count));
        try m.commit(&pass, count);
        try std.testing.expectError(error.InvalidCommit, m.commit(&pass, count));
        for (m.cache, 0..) |cache, layer| inline for (comptime std.meta.fieldNames(ds.Cache)) |field| if (@field(cache, field).ctx != null) {
            try save(s, output, try std.fmt.bufPrint(&buf, "cache-{d}-{d}-{s}", .{ step, layer, field }), @field(cache, field));
        };
        var head = try m.forwardMtp(pass.streams, next[0..count]);
        defer head.deinit();
        inline for (.{ "streams", "hidden", "logits" }) |field| try save(&head.scope, output, try std.fmt.bufPrint(&buf, "head-{d}-{s}", .{ step, field }), @field(head, field));
        try std.testing.expectError(error.InvalidCommit, m.commit(&head, count));
        const keep = if (step == 7) count - 3 else count;
        try m.commitMtp(&head, keep);
        try std.testing.expectError(error.InvalidCommit, m.commitMtp(&head, 1));
        if (keep < count) {
            try save(s, output, "head-partial-keys", m.mtp_cache.keys);
            var replay = try m.forwardMtp(try s.slice(pass.streams, 0, @intCast(keep), @intCast(count)), next[keep..count]);
            defer replay.deinit();
            try save(&replay.scope, output, "head-replay", replay.logits);
            try m.commitMtp(&replay, count - keep);
        }
        try save(s, output, try std.fmt.bufPrint(&buf, "head-cache-{d}-keys", .{step}), m.mtp_cache.keys);
        try std.testing.expectEqual(m.position, m.mtp_position);
        std.debug.print("DeepSeek prefill and draft cache commit at {d} tokens.\n", .{m.position});
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

pub fn checkDspark(io: std.Io, dir: []const u8, output: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var m = try ds.Model.init(io, dir);
    defer m.deinit();
    var buf: [4096]u8 = undefined;
    try m.loadDraft(io, try std.fmt.bufPrint(&buf, "{s}/drafter", .{dir}));
    try std.Io.Dir.cwd().createDirPath(io, output);
    for ([_]usize{ 17, 63, 64, 511, 512, 513, 2048, 17, 1 }, 0..) |count, step| {
        var tokens: [2048]i32 = undefined;
        for (tokens[0..count], 0..) |*id, j| id.* = 1 + @mod(m.position + @as(i32, @intCast(j)), 250);
        var pass = try m.prefill(tokens[0..count]);
        defer pass.deinit();
        const s = &pass.scope;
        try save(s, output, try std.fmt.bufPrint(&buf, "target-{d}", .{step}), pass.logits);
        try save(s, output, try std.fmt.bufPrint(&buf, "taps-{d}", .{step}), pass.taps);
        try m.commit(&pass, count);
        const draft = &m.dspark.?;
        try std.testing.expectEqual(m.position, draft.position);
        for (draft.keys, 0..) |keys, i| try save(s, output, try std.fmt.bufPrint(&buf, "keys-{d}-{d}", .{ step, i }), keys);
        const first: i32 = 55 + @as(i32, @intCast(step));
        try save(s, output, try std.fmt.bufPrint(&buf, "draft-logits-{d}", .{step}), try draft.logits(&m, s, first));
        for ([_]f64{ 0, 0.8 }, 0..) |temperature, mode| {
            var drawn: [16]i32 = undefined;
            const rows: usize = @intCast(draft.config.value.dspark_block_size);
            try draft.draw(&m, first, drawn[0..rows], .{ .seed = 1234, .temperature = temperature, .top_k = 20, .top_p = 0.95, .metal = true });
            try save(s, output, try std.fmt.bufPrint(&buf, "draw-{d}-{d}", .{ step, mode }), try s.ints(drawn[0..rows]));
        }
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
