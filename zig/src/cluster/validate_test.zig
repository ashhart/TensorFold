//! `tensorfold config validate FILE` on the host: valid, invalid, missing, and usage, with no probes or planning.
const std = @import("std");
const validate = @import("validate.zig");
const config_test = @import("config_test.zig");

const io = std.testing.io;

const Run = struct { out: []const u8, err: []const u8, code: u8 };

fn run(a: std.mem.Allocator, args: []const []const u8) !Run {
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const code = try validate.run(.{ .io = io, .a = a, .out = &out.writer, .err = &err.writer }, args);
    return .{ .out = out.written(), .err = err.written(), .code = code };
}

fn has(text: []const u8, needle: []const u8) bool {
    if (std.mem.indexOf(u8, text, needle) != null) return true;
    std.debug.print("missing \"{s}\" in:\n{s}\n", .{ needle, text });
    return false;
}

fn writeCluster(a: std.mem.Allocator, tmp: std.testing.TmpDir, data: []const u8) ![]const u8 {
    try tmp.dir.writeFile(io, .{ .sub_path = "cluster.json", .data = data });
    return std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/cluster.json", .{tmp.sub_path});
}

test "config validate: only the config command belongs to this entry" {
    try std.testing.expect(validate.wants(&.{"config"}) and validate.wants(&.{ "config", "validate", "c.json" }));
    try std.testing.expect(!validate.wants(&.{ "cluster", "check" }) and !validate.wants(&.{"serve"}) and !validate.wants(&.{}));
}

test "config validate: the worked example exits 0 and names the path and counts" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try writeCluster(a, tmp, config_test.example);
    const got = try run(a, &.{ "config", "validate", file });
    try std.testing.expectEqual(@as(u8, 0), got.code);
    try std.testing.expect(has(got.out, file));
    try std.testing.expect(has(got.out, "6 nodes"));
    try std.testing.expect(has(got.out, "3 models"));
    try std.testing.expectEqualStrings("", got.err);
}

test "config validate: an unknown key exits 1 and names the key" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try writeCluster(a, tmp, "{\"schema\":\"tensorfold-cluster/1\",\"cluster\":\"x\",\"nodes\":[{\"name\":\"a\",\"address\":\"h\",\"backend\":\"rocm\"}]}");
    const got = try run(a, &.{ "config", "validate", file });
    try std.testing.expectEqual(@as(u8, 1), got.code);
    try std.testing.expect(has(got.err, "backend"));
    try std.testing.expectEqualStrings("", got.out);
}

test "config validate: a wrong type exits 1 and names the field" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try writeCluster(a, tmp, "{\"schema\":\"tensorfold-cluster/1\",\"cluster\":\"x\",\"nodes\":\"nope\"}");
    const got = try run(a, &.{ "config", "validate", file });
    try std.testing.expectEqual(@as(u8, 1), got.code);
    try std.testing.expect(has(got.err, "nodes: must be an array"));
    try std.testing.expectEqualStrings("", got.out);
}

test "config validate: a conflicting layout exits 1 and names the problem" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const text =
        \\{"schema":"tensorfold-cluster/1","cluster":"x",
        \\"nodes":[{"name":"m1","address":"h","backend":"metal"},{"name":"m2","address":"h","backend":"metal"},
        \\{"name":"c1","address":"h","backend":"cuda"},{"name":"c2","address":"h","backend":"cuda"}],
        \\"models":{"k":{"path":"/m","nodes":["m1","m2","c1","c2"],"parallel":{"tensor":3}}}}
    ;
    const file = try writeCluster(a, tmp, text);
    const got = try run(a, &.{ "config", "validate", file });
    try std.testing.expectEqual(@as(u8, 1), got.code);
    try std.testing.expect(has(got.err, "does not divide"));
    try std.testing.expectEqualStrings("", got.out);
}

test "config validate: a missing path exits 1 and includes the path tried" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = "no-such-cluster-9f3a.json";
    const got = try run(a, &.{ "config", "validate", path });
    try std.testing.expectEqual(@as(u8, 1), got.code);
    try std.testing.expect(has(got.err, path));
    try std.testing.expect(has(got.err, "FileNotFound"));
    try std.testing.expectEqualStrings("", got.out);
}

test "config validate: bad usage exits 2" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { args: []const []const u8 };
    const cases = [_]Case{
        .{ .args = &.{"config"} },
        .{ .args = &.{ "config", "validate" } },
        .{ .args = &.{ "config", "check", "c.json" } },
        .{ .args = &.{ "config", "validate", "a.json", "b.json" } },
        .{ .args = &.{ "config", "validate", "--cluster", "c.json" } },
        .{ .args = &.{ "config", "validate", "--help" } },
    };
    for (cases) |k| {
        const got = try run(a, k.args);
        if (got.code != 2) std.debug.print("usage: exit {d}\n{s}{s}", .{ got.code, got.out, got.err });
        try std.testing.expectEqual(@as(u8, 2), got.code);
        try std.testing.expect(has(got.err, "usage: tensorfold config validate FILE"));
        try std.testing.expectEqualStrings("", got.out);
    }
}
