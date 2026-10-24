//! The lookup tests' rig: a byte-tokenized store with its journal in a temporary folder, a server, and HTTP requests.
const std = @import("std");
const server = @import("root.zig");
const json = server.json;
const index = @import("notes/index.zig");
const Storage = @import("notes/storage.zig").Storage;
const Echo = @import("notes_storage_http_test.zig").Echo;
const Bytes = @import("notes_storage_bytes_test.zig").Bytes;
const exchange = @import("notes_lookup_test.zig").exchange;
const testing = std.testing;

pub const snapshot = @embedFile("notes/testdata/l6_bytes_snapshot.json");

/// The n-gram index a rig's store builds.
pub const Kind = struct { kind: index.Kind = .postings };

/// LLWZ's posting lists, then the suffix array.
pub const kinds = [_]Kind{ .{}, .{ .kind = .sa } };

pub const Rig = struct {
    tmp: testing.TmpDir,
    arena: std.heap.ArenaAllocator,
    text: Bytes = .{},
    engine: Echo = .{},
    srv: *server.Server,

    /// A server with `snap` as its notes, writes journaled beside it.
    pub fn initWithSnap(gpa: std.mem.Allocator, k: Kind, snap: []const u8) !*Rig {
        const r = try testing.allocator.create(Rig);
        errdefer testing.allocator.destroy(r);
        r.* = .{ .tmp = testing.tmpDir(.{}), .arena = .init(testing.allocator), .srv = undefined };
        errdefer r.arena.deinit();
        errdefer r.tmp.cleanup();
        const a = r.arena.allocator();
        r.srv = try server.Server.init(testing.allocator, testing.io, r.engine.engine(), r.text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
        errdefer r.srv.deinit();
        const dir = try r.tmp.dir.realPathFileAlloc(testing.io, ".", a);
        const path = try std.fs.path.join(a, &.{ dir, "notes.json.journal" });
        var why: []const u8 = "";
        r.srv.notes = try Storage.loadKind(gpa, snap, r.text.text(), path, k.kind, &why);
        try r.srv.notes.?.beginRun(a, .{});
        return r;
    }

    pub fn store(r: *Rig) *Storage {
        return r.srv.notes.?;
    }

    pub fn deinit(r: *Rig) void {
        if (r.srv.notes) |s| s.deinit();
        r.srv.notes = null;
        r.srv.deinit();
        r.tmp.cleanup();
        r.arena.deinit();
        testing.allocator.destroy(r);
    }
};

pub const Reply = struct { status: u16, body: []const u8, value: json.Value };

/// One HTTP request; the reply's body and its JSON live in `a`.
pub fn request(srv: *server.Server, a: std.mem.Allocator, method: []const u8, path: []const u8, body: []const u8) !Reply {
    const got = try exchange(srv, method, path, body);
    defer testing.allocator.free(got.body);
    const kept = try a.dupe(u8, got.body);
    return .{ .status = got.status, .body = kept, .value = (try json.parse(a, kept)).ok };
}

/// The error message of a refused request ("" for none).
pub fn message(r: Reply) []const u8 {
    const e = r.value.get("error") orelse return "";
    return e.strField("message") orelse "";
}
