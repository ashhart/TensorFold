//! Emit startup receipts after listening succeeds and after a whole OpenAI reply is written.
const std = @import("std");
const timing = @import("core").startup_receipt;
const memory = @import("process_memory.zig");
const json = @import("json");
const log = @import("log.zig");
var mutex: std.Io.Mutex = .init;

pub fn ready(port: u16) void {
    if (timing.offset() == null) return;
    mutex.lockUncancelable(log.io());
    defer mutex.unlock(log.io());
    const receipt = timing.ready() orelse return;
    emit(receipt, port, null);
}

pub fn complete(start: ?u64, payload: json.Value) void {
    if (start == null) return;
    const runtime = payload.get("tensorfold") orelse return;
    const hash = runtime.get("token_sha256") orelse return;
    if (hash != .string or !validHash(hash.string)) return;
    mutex.lockUncancelable(log.io());
    defer mutex.unlock(log.io());
    const receipt = timing.complete(start) orelse return;
    emit(receipt, null, hash.string);
}

fn validHash(hash: []const u8) bool {
    if (hash.len != 64) return false;
    for (hash) |c| if (!(c >= '0' and c <= '9') and !(c >= 'a' and c <= 'f')) return false;
    return true;
}

fn emit(receipt: timing.Receipt, port: ?u16, hash: ?[]const u8) void {
    var bytes: [8192]u8 = undefined;
    var out: std.Io.Writer = .fixed(&bytes);
    format(&out, receipt, port, hash, memory.read()) catch return;
    std.debug.print("startup-receipt {s}\n", .{out.buffered()});
}

fn format(w: *std.Io.Writer, receipt: timing.Receipt, port: ?u16, hash: ?[]const u8, resources: ?memory.Snapshot) !void {
    try std.json.Stringify.value(.{ .startup = receipt, .port = port, .token_sha256 = hash, .resources = resources }, .{}, w);
}

test "startup receipt: JSON carries monotonic spans, kernel resources and only a token hash" {
    var state: timing.State = .{};
    state.target = true;
    state.start(.read_pack, 10);
    state.end(.read_pack, 30);
    state.end(.model_ready, 40);
    const ready_record = state.ready(50).?;
    var buf: [8192]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try format(&w, ready_record, 12345, null, .{ .physical_footprint_bytes = 10, .lifetime_peak_physical_footprint_bytes = 20, .disk_io_bytes_read = 30 });
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, w.buffered(), .{});
    defer parsed.deinit();
    const value = parsed.value;
    try std.testing.expectEqualStrings("ready", value.object.get("startup").?.object.get("event").?.string);
    try std.testing.expectEqual(@as(i64, 30), value.object.get("resources").?.object.get("disk_io_bytes_read").?.integer);
    try std.testing.expect(value.object.get("token_sha256").? == .null);
    const good: [64]u8 = @splat('a');
    const bad: [64]u8 = @splat('G');
    try std.testing.expect(validHash(&good));
    try std.testing.expect(!validHash(good[0..63]));
    try std.testing.expect(!validHash(&bad));
}
