//! Shard pulls over the cluster's links: a pull names a file range; the holder answers with the bytes and their SHA-256.
const std = @import("std");
const node = @import("node.zig");
const wire = @import("wire.zig");
const transport = @import("transport.zig");
const membership = @import("membership.zig");
const checkpoint = @import("checkpoint.zig");
const loader = @import("loader.zig");

const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Status = enum(u8) { ok = 0, missing = 1, failed = 2 };

/// Bytes of a chunk message besides the payload: the header, id, status and SHA-256.
pub const overhead = wire.header_bytes + 4 + 1 + 32;

/// One node's side of pulls: it serves the files it holds and waits for the chunks it asked for.
pub const Agent = struct {
    io: Io,
    gpa: std.mem.Allocator,
    dir: Io.Dir,
    files: []const checkpoint.File,
    members: *membership.Membership,
    t: transport.Transport,
    /// Advances every node's loop (in tests, all simulated nodes; on a host, this node's own).
    pump: *const fn (ctx: *anyopaque) anyerror!void,
    pump_ctx: *anyopaque,
    sources: []const node.NodeId,
    next_id: u32 = 0,
    want: ?struct { id: u32, out: []u8, sha: [32]u8 = undefined, status: ?Status = null } = null,
    served: u64 = 0,
    out: []u8 = &.{},

    pub fn sink(ag: *Agent) membership.Sink {
        return .{ .ctx = ag, .handle = handle };
    }

    pub fn source(ag: *Agent) loader.Source {
        return .{ .ptr = ag, .pull = pullFn };
    }

    pub fn deinit(ag: *Agent) void {
        ag.gpa.free(ag.out);
    }

    fn handle(ctx: *anyopaque, link: transport.LinkId, h: wire.Header, body: []const u8) void {
        const ag: *Agent = @ptrCast(@alignCast(ctx));
        var in: wire.In = .{ .buf = body };
        switch (h.kind) {
            .pull => ag.serve(link, h.from, &in) catch {},
            .chunk => {
                const id = in.int(u32) catch return;
                const status = std.enums.fromInt(Status, in.int(u8) catch return) orelse return;
                const sha = (in.bytes(32) catch return)[0..32].*;
                const w = &(ag.want orelse return);
                if (w.id != id) return;
                w.status = status;
                w.sha = sha;
                if (status == .ok) {
                    const data = in.bytes(w.out.len) catch {
                        w.status = .failed;
                        return;
                    };
                    @memcpy(w.out, data);
                }
            },
            else => {},
        }
    }

    fn serve(ag: *Agent, link: transport.LinkId, to: node.NodeId, in: *wire.In) !void {
        const id = try in.int(u32);
        const file = try in.int(u32);
        const offset = try in.int(u64);
        const len = try in.int(u32);
        if (ag.out.len < ag.t.limit()) {
            ag.gpa.free(ag.out);
            ag.out = try ag.gpa.alloc(u8, ag.t.limit());
        }
        var o: wire.Out = .{ .buf = ag.out };
        try o.int(u32, id);
        const data_at = o.at + 1 + 32;
        const status: Status = blk: {
            if (file >= ag.files.len or data_at + len > ag.out.len) break :blk .failed;
            const f = ag.files[file];
            const st = ag.dir.statFile(ag.io, f.name, .{}) catch break :blk .missing;
            if (st.size != f.size or offset + len > f.size) break :blk .missing;
            const h = ag.dir.openFile(ag.io, f.name, .{}) catch break :blk .missing;
            defer h.close(ag.io);
            const n = h.readPositionalAll(ag.io, ag.out[data_at..][0..len], offset) catch break :blk .failed;
            break :blk if (n == len) .ok else .failed;
        };
        try o.int(u8, @backingInt(status));
        var sha: [32]u8 = @splat(0);
        if (status == .ok) Sha256.hash(ag.out[data_at..][0..len], &sha, .{});
        try o.bytes(&sha);
        if (status == .ok) o.at += len;
        ag.served += @intFromBool(status == .ok);
        try ag.t.send(link, o.finish(.{ .kind = .chunk, .from = ag.members.me.id, .to = to }));
    }

    /// loader.Source: ask each source node in turn until one holds the file; chunks larger than a message are split.
    fn pullFn(ptr: *anyopaque, file: u32, offset: u64, out: []u8) anyerror![32]u8 {
        const ag: *Agent = @ptrCast(@alignCast(ptr));
        const room = ag.t.limit() - overhead;
        if (out.len > room) return error.ChunkTooLarge;
        for (ag.sources) |src| {
            if (src == ag.members.me.id) continue;
            const m = ag.members.find(src) orelse continue;
            const via = m.route orelse continue;
            ag.next_id += 1;
            ag.want = .{ .id = ag.next_id, .out = out };
            var req: [64]u8 = undefined;
            var o: wire.Out = .{ .buf = &req };
            try o.int(u32, ag.next_id);
            try o.int(u32, file);
            try o.int(u64, offset);
            try o.int(u32, @intCast(out.len));
            ag.t.send(via, o.finish(.{ .kind = .pull, .from = ag.members.me.id, .to = src })) catch continue;
            var spins: u32 = 0;
            while (ag.want.?.status == null and spins < 10_000) : (spins += 1) try ag.pump(ag.pump_ctx);
            const got = ag.want.?;
            ag.want = null;
            if (got.status == .ok) return got.sha;
        }
        return error.MissingShard;
    }
};

test {
    _ = @import("pull_test.zig");
}
