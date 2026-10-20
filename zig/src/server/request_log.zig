//! TENSORFOLD_REQUEST_LOG: each chat and completion body appended as one JSON line, images redacted, for replays.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const json = @import("json");
const meta = @import("file_meta.zig").meta;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The body with every image part's URL replaced, the rest kept.
fn redact(a: Allocator, v: Value) Allocator.Error!Value {
    switch (v) {
        .array => |items| {
            const out = try a.alloc(Value, items.len);
            for (items, out) |item, *slot| slot.* = try redact(a, item);
            return .{ .array = out };
        },
        .object => |o| {
            if (v.typeIs("image_url")) {
                const copy = try json.copyObject(a, o);
                const url = try json.newObject(a);
                try url.put(a, "url", .{ .string = "<redacted>" });
                try copy.put(a, "image_url", .{ .object = url });
                return .{ .object = copy };
            }
            const out = try json.newObject(a);
            for (o.keys(), o.values()) |k, item| try out.put(a, k, try redact(a, item));
            return .{ .object = out };
        },
        else => return v,
    }
}

/// Appends ``body`` to ``path`` unless it is background work (batch jobs are not client traffic).
pub fn append(a: Allocator, path: []const u8, body: Value) void {
    if (body.strField("priority")) |p| if (std.mem.eql(u8, p, "background")) return;
    const line = json.stringify(a, redact(a, body) catch return, .{}) catch return;
    const pathz = a.dupeSentinel(u8, path, 0) catch return;
    const fd = platform.c.open(pathz, .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return warnOnce(path, "it cannot be opened, or it is a symbolic link");
    defer _ = platform.c.close(fd);
    if (!private(fd)) return warnOnce(path, "it is not a regular file only its owner can read");
    const framed = std.mem.concat(a, u8, &.{ line, "\n" }) catch return;
    _ = platform.c.write(fd, framed.ptr, framed.len);
}

/// A regular file no one but its owner can read or write: a name in a shared directory can be someone else's file.
fn private(fd: platform.c.fd_t) bool {
    if (builtin.os.tag == .windows) return true; // NTFS keeps access in ACLs, not mode bits
    const m = meta(fd, null) orelse return false;
    return m.mode & 0o170000 == 0o100000 and m.mode & 0o077 == 0;
}

var warned = std.atomic.Value(bool).init(false);

fn warnOnce(path: []const u8, why: []const u8) void {
    if (warned.swap(true, .monotonic)) return;
    std.log.warn("TENSORFOLD_REQUEST_LOG {s} gets no requests: {s}", .{ path, why });
}

test "the request log is created owner-only and keeps image URLs out" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/requests.jsonl", .{tmp.sub_path});
    const url = try json.newObject(a);
    try url.put(a, "url", .{ .string = "https://example.test/private.png" });
    const part = try json.newObject(a);
    try part.put(a, "type", .{ .string = "image_url" });
    try part.put(a, "image_url", .{ .object = url });
    const body = try json.newObject(a);
    try body.put(a, "messages", .{ .array = try a.dupe(Value, &.{.{ .object = part }}) });
    append(a, path, .{ .object = body });
    if (comptime @import("builtin").os.tag != .windows) { // Windows files have no mode bits
        const st = try std.Io.Dir.cwd().statFile(std.testing.io, path, .{}); // Linux's libc has no stat for std.c
        try std.testing.expectEqual(@as(u32, 0o600), @as(u32, @intCast(st.permissions.toMode())) & 0o777);
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 16));
    try std.testing.expect(std.mem.indexOf(u8, text, "<redacted>") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "example.test") == null);
}

test "appends from several threads keep every line whole" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try std.process.currentPathAlloc(std.testing.io, a);
    defer a.free(cwd);
    const path = try std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}/log.jsonl", .{ cwd, tmp.sub_path });
    defer a.free(path);
    const threads = 8;
    const lines = 200;
    const head = "{\"line\": \"";
    const Worker = struct {
        fn run(p: []const u8, k: u8) void {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const text = arena.allocator().alloc(u8, 300) catch return;
            @memset(text, 'a' + k);
            for (0..lines) |_| {
                const o = json.newObject(arena.allocator()) catch return;
                o.put(arena.allocator(), "line", .{ .string = text }) catch return;
                append(arena.allocator(), p, .{ .object = o });
            }
        }
    };
    var workers: [threads]std.Thread = undefined;
    for (&workers, 0..) |*w, k| w.* = try std.Thread.spawn(.{}, Worker.run, .{ path, @as(u8, @intCast(k)) });
    for (workers) |w| w.join();
    const got = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 24));
    defer a.free(got);
    var counts: [threads]usize = @splat(0);
    var it = std.mem.splitScalar(u8, got, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        try std.testing.expectEqual(head.len + 300 + 2, line.len);
        const k = line[head.len] - 'a';
        try std.testing.expect(k < threads);
        for (line[head.len..][0..300]) |c| try std.testing.expectEqual('a' + k, c);
        counts[k] += 1;
    }
    for (counts) |n| try std.testing.expectEqual(@as(usize, lines), n);
}

test "a log others can read, or a symbolic link, gets no requests; the owner's own file still does" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // refusal by mode bits and links; NTFS keeps access in ACLs
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    // a name another user created first, readable by others: nothing is appended to it
    try tmp.dir.writeFile(io, .{ .sub_path = "taken.jsonl", .data = "", .flags = .{ .permissions = .fromMode(0o644) } });
    append(a, try std.fmt.allocPrint(a, "{s}/taken.jsonl", .{dir}), .{ .object = body });
    try std.testing.expectEqual(@as(usize, 0), (try tmp.dir.readFileAlloc(io, "taken.jsonl", a, .limited(1 << 10))).len);
    // a symbolic link to another file: the target is never written
    try tmp.dir.writeFile(io, .{ .sub_path = "target", .data = "", .flags = .{ .permissions = .fromMode(0o600) } });
    try tmp.dir.symLink(io, "target", "link.jsonl", .{});
    append(a, try std.fmt.allocPrint(a, "{s}/link.jsonl", .{dir}), .{ .object = body });
    try std.testing.expectEqual(@as(usize, 0), (try tmp.dir.readFileAlloc(io, "target", a, .limited(1 << 10))).len);
    // the owner's own 0600 file keeps its lines
    append(a, try std.fmt.allocPrint(a, "{s}/target", .{dir}), .{ .object = body });
    try std.testing.expect((try tmp.dir.readFileAlloc(io, "target", a, .limited(1 << 10))).len > 0);
}
