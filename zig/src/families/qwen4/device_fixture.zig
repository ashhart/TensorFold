//! Device gates create only their own nonzero checkpoint and compare raw BF16 outputs with the binary64 fixture.
const std = @import("std");
const cfg = @import("config.zig");
pub const Oracle = struct { schema: u32, oracle: []const u8, tokens: []const u32, logits: [][]f64, picks: []const u32, trace: std.json.Value };
pub const config = @embedFile("fixtures/native-dense.json");
pub const metal_reference = @embedFile("fixtures/native-metal-binary64.json");
pub const cuda_reference = @embedFile("fixtures/native-cuda-binary64.json");
pub const legacy_reference = @embedFile("fixtures/native-binary64.json");
pub const reference = cuda_reference;
pub const Fixture = struct {
    path: []const u8,
    a: std.mem.Allocator,
    io: std.Io,
    pub fn open(a: std.mem.Allocator, io: std.Io) !Fixture {
        const path = try std.fmt.allocPrint(a, ".qwen4-device-{d}", .{std.Io.Clock.real.now(io).toNanoseconds()});
        errdefer a.free(path);
        try std.Io.Dir.cwd().createDir(io, path, .default_dir);
        errdefer std.Io.Dir.cwd().deleteTree(io, path) catch {};
        var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
        defer dir.close(io);
        var problem: cfg.Problem = .{};
        var c = try cfg.parse(a, config, &problem);
        defer c.deinit();
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        try dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
        try @import("reader_fixture.zig").writeDir(dir, io, arena.allocator(), &c, false, true, false);
        return .{ .path = path, .a = a, .io = io };
    }
    pub fn close(f: Fixture) void {
        std.Io.Dir.cwd().deleteTree(f.io, f.path) catch {};
        f.a.free(f.path);
    }
};
pub fn same(comptime T: type, name: []const u8, want: []const T, got: []const T) !void {
    if (std.mem.eql(T, want, got)) return;
    std.debug.print("FAIL qwen4 {s}: expected {d} values, got {d}; first difference {?d}\n", .{ name, want.len, got.len, std.mem.indexOfDiff(T, want, got) });
    return error.Qwen4DeviceExactness;
}
pub fn oracle(want: Oracle, logits: []const u16, picks: []const u32) !void {
    if (want.schema != 1 or want.logits.len != want.tokens.len or want.picks.len != want.tokens.len) return error.Qwen4OracleShape;
    var at: usize = 0;
    var different: usize = 0;
    for (want.logits, 0..) |row, r| for (row, 0..) |value, col| {
        if (at >= logits.len) return error.Qwen4OracleShape;
        const expected = @import("core").quant.convert.bf16(@floatCast(value));
        if (expected != logits[at]) {
            if (different < 8) std.debug.print("FAIL qwen4 binary64 row={d} col={d}: expected 0x{x:0>4}, got 0x{x:0>4}\n", .{ r, col, expected, logits[at] });
            different += 1;
        }
        at += 1;
    };
    if (at != logits.len) return error.Qwen4OracleShape;
    if (different != 0) {
        std.debug.print("FAIL qwen4 binary64: {d}/{d} logits differ\n", .{ different, logits.len });
        return error.Qwen4OracleMismatch;
    }
    try same(u32, "binary64 token picks", want.picks, picks);
}
