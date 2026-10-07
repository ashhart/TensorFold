//! The compaction note file and the hash of the messages a note already covers.
const std = @import("std");
const json = @import("json.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const Mem = struct { note: []const u8, covered: usize, hash: []const u8 };

pub fn conversationKey(a: Allocator, msgs: []const Value) ![16]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    if (firstRole(msgs, "system")) |m| h.update(try json.stringify(a, m, .{ .compact = true }));
    h.update("\n");
    if (firstRole(msgs, "user")) |m| h.update(try json.stringify(a, m, .{ .compact = true }));
    var d: [32]u8 = undefined;
    h.final(&d);
    const hex = std.fmt.bytesToHex(d, .lower);
    return hex[0..16].*;
}

pub fn hashes(a: Allocator, msgs: []const Value, mem: Mem) bool {
    if (mem.covered == 0 or mem.covered > msgs.len or mem.hash.len == 0) return false;
    const got = prefixHash(a, msgs[0..mem.covered]) catch return false;
    return std.mem.eql(u8, got, mem.hash);
}

pub fn loadMem(srv: *Server, a: Allocator, msgs: []const Value) ?Mem {
    const dir = srv.config.compact_memory orelse return null;
    const key = conversationKey(a, msgs) catch return null;
    const note = readNamed(srv, a, dir, &key, ".md") orelse return null;
    const js = readNamed(srv, a, dir, &key, ".json") orelse return .{ .note = note, .covered = 0, .hash = "" };
    const parsed = json.parse(a, js) catch return .{ .note = note, .covered = 0, .hash = "" };
    if (parsed != .ok) return .{ .note = note, .covered = 0, .hash = "" };
    const covered: usize = if (parsed.ok.get("covered")) |c| blk: {
        const n = c.int64() orelse 0;
        break :blk if (n <= 0) 0 else @intCast(n);
    } else 0;
    const hash = if (parsed.ok.get("hash")) |h| (if (h == .string) h.string else "") else "";
    return .{ .note = note, .covered = covered, .hash = hash };
}

pub fn storeMem(srv: *Server, a: Allocator, msgs: []const Value, covered: usize, note: []const u8) void {
    const dir = srv.config.compact_memory orelse return;
    std.Io.Dir.cwd().createDirPath(srv.io, dir) catch return;
    const key = conversationKey(a, msgs) catch return;
    const n = @min(covered, msgs.len);
    const hash = prefixHash(a, msgs[0..n]) catch return;
    const body = std.fmt.allocPrint(a, "{{\"covered\":{d},\"hash\":\"{s}\"}}", .{ n, hash }) catch return;
    writeNamed(srv, a, dir, &key, ".md", note);
    writeNamed(srv, a, dir, &key, ".json", body);
}

fn readNamed(srv: *Server, a: Allocator, dir: []const u8, key: []const u8, ext: []const u8) ?[]u8 {
    const name = std.fmt.allocPrint(a, "{s}{s}", .{ key, ext }) catch return null;
    const path = std.fs.path.join(a, &.{ dir, name }) catch return null;
    return std.Io.Dir.cwd().readFileAlloc(srv.io, path, a, .limited(1 << 20)) catch null;
}

fn writeNamed(srv: *Server, a: Allocator, dir: []const u8, key: []const u8, ext: []const u8, data: []const u8) void {
    const name = std.fmt.allocPrint(a, "{s}{s}", .{ key, ext }) catch return;
    const tmp = std.fmt.allocPrint(a, "{s}{s}.tmp", .{ key, ext }) catch return;
    const path = std.fs.path.join(a, &.{ dir, name }) catch return;
    const tmp_path = std.fs.path.join(a, &.{ dir, tmp }) catch return;
    std.Io.Dir.cwd().writeFile(srv.io, .{ .sub_path = tmp_path, .data = data }) catch return;
    std.Io.Dir.cwd().rename(tmp_path, std.Io.Dir.cwd(), path, srv.io) catch return;
}

fn prefixHash(a: Allocator, msgs: []const Value) ![]u8 {
    const bytes = try json.stringify(a, .{ .array = @constCast(msgs) }, .{ .compact = true });
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(bytes);
    var d: [32]u8 = undefined;
    h.final(&d);
    const hex = std.fmt.bytesToHex(d, .lower);
    return a.dupe(u8, &hex);
}

fn firstRole(msgs: []const Value, role: []const u8) ?Value {
    for (msgs) |m| if (std.mem.eql(u8, roleOf(m), role)) return m;
    return null;
}

fn roleOf(m: Value) []const u8 {
    const r = m.get("role") orelse return "";
    return if (r == .string) r.string else "";
}
