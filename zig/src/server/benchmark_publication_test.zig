//! The actual benchmark command must reject local-only v2 before any publication effect.
const std = @import("std");
const json = @import("json");
const command = @import("benchmark_cli_test.zig");
const benchmark = @import("benchmark.zig");
const receipts = @import("benchmark_receipt_test.zig");
const receipt = @import("benchmark_receipt.zig");
const upload = @import("benchmark_upload.zig");
var touched: [3]usize = @splat(0);
var stop_at_credential = false;
fn credential(_: std.mem.Allocator, _: std.Io, _: upload.Options, problem: *[]const u8) anyerror!?[]const u8 {
    touched[0] += 1;
    if (stop_at_credential) {
        problem.* = "inert credential boundary";
        return null;
    }
    return "inert-publication-value";
}
fn verify(_: std.mem.Allocator, _: std.Io, _: upload.Options, _: json.Value, _: *[]const u8) anyerror!bool {
    touched[1] += 1;
    return true;
}
fn submit(_: std.mem.Allocator, _: std.Io, _: upload.Options, _: json.Value, _: *[]const u8) anyerror!?upload.Ack {
    touched[2] += 1;
    return .{ .id = "inert", .status = "pending", .results_url = null };
}
const services: benchmark.Publication = .{ .credential = credential, .verify_public = verify, .upload = submit };

test "benchmark publication: saved v2 refuses every effect while v1 still reaches the services" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try command.Run.init(a, &tmp);
    r.publication = services;
    const out = try r.path("v2.json");
    var why: []const u8 = "";
    try std.testing.expect(try receipt.save(a, std.testing.io, try receipts.valid(a), out, 1767225660000, &why));
    touched = @splat(0);
    stop_at_credential = false;
    const code = try r.command(&.{ "publish", out, "--yes" });
    try std.testing.expectEqual([3]usize{ 0, 0, 0 }, touched);
    try std.testing.expectEqual(@as(u8, 1), code);
    try std.testing.expect(r.has(upload.v2_disabled));
    const legacy = try r.path("v1.json");
    try std.testing.expect(try receipt.save(a, std.testing.io, try receipts.legacy(a), legacy, 1767225660000, &why));
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "publish", legacy, "--yes" }));
    try std.testing.expectEqual([3]usize{ 1, 1, 1 }, touched);
}

test "benchmark publication: run publish refuses before credentials, server work or a saved receipt" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "cuda", "mlx" }) |backend| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var r = try command.Run.init(a, &tmp);
        r.publication = services;
        const out = try r.path("run.json");
        touched = @splat(0);
        stop_at_credential = true;
        try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "example-org/test-model", "--backend", backend, "--publish", "--output", out }));
        try std.testing.expectEqual([3]usize{ 0, 0, 0 }, touched);
        try std.testing.expect(r.has(upload.v2_disabled));
        try std.testing.expect(!r.has("This generates model replies"));
        try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(std.testing.io, out, .{}));
    }
}
