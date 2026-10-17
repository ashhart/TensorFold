//! The issue's config size through the real benchmark command and model receipt, without a GPU.
const std = @import("std");
const api = @import("engine_api");
const benchmark = @import("benchmark.zig");
const model = @import("benchmark_model.zig");
const cli = @import("benchmark_cli_test.zig");
const fake = @import("benchmark_fake.zig");
const size = 8_668_620;
const limit = 64 << 20;
const rev = "82d93699acd3be9e4617c53360fe5b3f5dbeb1b3";
const families = [_]api.Family{.{ .model_type = "nemotron_h_omni", .formats = &.{} }};
const head = "{\"model_type\":\"nemotron_h_omni\",\"quantization_config\":{\"quant_method\":\"modelopt\",\"quant_algo\":\"NVFP4\"}}";
fn large(a: std.mem.Allocator, dir: []const u8) ![]const u8 {
    const bytes = try a.alloc(u8, size);
    @memset(bytes, ' ');
    @memcpy(bytes[0..head.len], head);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, dir);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "config.json" }), .data = bytes });
    return bytes;
}
fn command(r: *cli.Run, args: []const []const u8) !u8 {
    r.out.clearRetainingCapacity();
    r.err.clearRetainingCapacity();
    return benchmark.command(&.{ .a = r.a, .io = std.testing.io, .environ = &r.env, .home = r.home, .out = &r.out.writer, .err = &r.err.writer, .input = .{ .tty = false, .reader = &r.stdin, .fd = null }, .engines = .{ .version = "test", .families = &families }, .self_exe = r.self_exe }, args);
}
test "issue 934: an 8668620 byte Super VL config reaches the managed benchmark and fills its receipt" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try cli.Run.init(a, &tmp);
    const world = try cli.managedWorld(a, false);
    defer world.stop();
    r.self_exe = try cli.standIn(&r, world.port);
    const dir = try r.path("private-checkpoint");
    _ = try large(a, dir);
    const out = try r.path("run.json");
    try std.testing.expectEqual(@as(u8, 0), try command(&r, &.{ dir, "--output", out, "--tokens", "4", "--reps", "1", "--temperatures", "0" }));
    const receipt = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, out, a, .limited(1 << 20));
    try std.testing.expect(std.mem.indexOf(u8, receipt, "\"family\":\"nemotron_h_omni\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, receipt, "\"quantization\":\"modelopt-nvfp4\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, receipt, "\"config_sha256\":null") == null);
}
test "issue 934: describe preserves the large config hash family quantization and cached revision" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fs.path.join(a, &.{ try fake.tmpPath(a, &tmp), "snapshots", rev });
    const bytes = try large(a, dir);
    const facts = try model.describe(a, std.testing.io, "nvidia/super", dir, false, &families, null);
    try std.testing.expect(facts.config_sha256 != null and facts.family != null and facts.quantization != null);
    try std.testing.expectEqualStrings("nemotron_h_omni", facts.family.?);
    try std.testing.expectEqualStrings("modelopt-nvfp4", facts.quantization.?);
    try std.testing.expectEqualStrings(rev, facts.revision.?);
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    try std.testing.expectEqualStrings(&std.fmt.bytesToHex(hash, .lower), facts.config_sha256.?);
}
fn refusal(comptime mode: enum { oversized, unreadable, missing, malformed }) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try cli.Run.init(a, &tmp);
    const path = try r.path("config.json");
    switch (mode) {
        .oversized => {
            const f = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
            defer f.close(std.testing.io);
            try f.setLength(std.testing.io, limit + 1);
        },
        .unreadable => try std.Io.Dir.cwd().createDir(std.testing.io, path, .default_dir),
        .missing => {},
        .malformed => try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = "{" }),
    }
    try std.testing.expectEqual(@as(u8, 1), try command(&r, &.{r.home}));
    if (mode == .missing) {
        try std.testing.expect(r.has("Checkpoint is incomplete"));
        return;
    }
    try std.testing.expect(!r.has("Checkpoint is incomplete"));
    try std.testing.expect(r.has("config.json"));
    try std.testing.expect(r.has("67108864"));
    const stat = try std.Io.Dir.cwd().statFile(std.testing.io, path, .{});
    try std.testing.expect(r.has(try std.fmt.allocPrint(a, "{d}", .{stat.size})));
}
test "issue 934: an oversized config reports its actual size and shared limit" {
    try refusal(.oversized);
}
test "issue 934: an unreadable config reports a config error with size and limit" {
    try refusal(.unreadable);
}
test "issue 934: a missing config still reports an incomplete checkpoint" {
    try refusal(.missing);
}

test "issue 934: a malformed config reports a config error with size and limit" {
    try refusal(.malformed);
}
