//! Leader-owned operations replayed on the other CUDA rank; sequence and cache identities survive map relocation.
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const snaps = @import("cuda_snap.zig");
const wire = @import("cuda_rank_wire.zig");
const Collective = @import("cuda_collective.zig").Collective;
const wt = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");

pub const Identity = extern struct {
    config: [32]u8,
    index: [32]u8,
    program: [32]u8,
    checkpoint: [32]u8,
    context: u64,
    chunk: u64,
    decode_graph: u64 = 0,
    mtp: u64 = 0,
};

fn fileHash(a: std.mem.Allocator, io: std.Io, path: []const u8) ![32]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28));
    defer a.free(bytes);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}

pub fn identity(a: std.mem.Allocator, io: std.Io, dir: []const u8, context: usize, chunk: usize) !Identity {
    const digest = std.mem.span(std.c.getenv("TENSORFOLD_CHECKPOINT_SHA256") orelse return error.NoCheckpointIdentity);
    if (digest.len != 64) return error.NoCheckpointIdentity;
    var checkpoint: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&checkpoint, digest) catch return error.NoCheckpointIdentity;
    const config = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(config);
    const index = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    defer a.free(index);
    const program = try std.process.executablePathAlloc(io, a);
    defer a.free(program);
    return .{ .config = try fileHash(a, io, config), .index = try fileHash(a, io, index), .program = try fileHash(a, io, program), .checkpoint = checkpoint, .context = context, .chunk = chunk, .decode_graph = @intFromBool(@import("cuda_round_graph.zig").enabled()) };
}

pub const Session = struct {
    gpa: std.mem.Allocator,
    rank: usize,
    link: wire.Link,
    collective: Collective,
    engine: *prompt.Engine = undefined,
    next: u64 = 1,
    next_object: u64 = 1,
    seqs: std.AutoHashMapUnmanaged(u64, *prompt.Seq) = .empty,
    saved: std.AutoHashMapUnmanaged(u64, *snaps.Snap) = .empty,
    windows: std.AutoHashMapUnmanaged(u64, mtp.Draft) = .empty,
    drafting: bool = false,
    stopped: bool = false,
    available: std.atomic.Value(bool) = .init(true),

    pub fn init(a: std.mem.Allocator, io: std.Io, host: []const u8, port: u16, rank: usize, mine: Identity) !*Session {
        const s = try a.create(Session);
        errdefer a.destroy(s);
        const link = try wire.Link.open(io, host, port, rank, 60_000);
        errdefer link.close();
        const until = wire.deadline(60_000);
        var peer: Identity = undefined;
        const hello: wire.Header = .{ .kind = .hello, .arg = rank, .bytes = @sizeOf(Identity) };
        if (rank == 0) {
            try link.sendHeader(hello, until);
            try link.send(std.mem.asBytes(&mine), until);
            const h = try link.receiveHeader(until);
            if (h.kind != .hello or h.sequence != 0 or h.arg != 1 or h.bytes != @sizeOf(Identity) or h.flags != 0 or h.object != 0) return error.RankIdentity;
            try link.receive(std.mem.asBytes(&peer), until);
        } else {
            const h = try link.receiveHeader(until);
            if (h.kind != .hello or h.sequence != 0 or h.arg != 0 or h.bytes != @sizeOf(Identity) or h.flags != 0 or h.object != 0) return error.RankIdentity;
            try link.receive(std.mem.asBytes(&peer), until);
            try link.sendHeader(hello, until);
            try link.send(std.mem.asBytes(&mine), until);
        }
        if (!sameIdentity(mine, peer)) return error.RankIdentity;
        s.* = .{ .gpa = a, .rank = rank, .link = link, .collective = try Collective.init(link, rank, 30_000), .drafting = mine.mtp == 1 };
        return s;
    }

    pub fn healthy(s: *const Session) bool {
        if (!s.available.load(.acquire)) return false;
        s.link.checkPeer() catch return false;
        return true;
    }

    pub fn poison(s: *Session) void {
        s.stopped = true;
        s.available.store(false, .release);
        s.collective.abort();
        _ = std.c.shutdown(s.link.fd, std.posix.SHUT.RDWR);
    }

    pub fn deinit(s: *Session) void {
        if (s.rank == 0 and !s.stopped) {
            const h = s.command(.{ .kind = .stop }, &.{}) catch null;
            if (h) |sent| s.ack(sent) catch s.poison();
        }
        var wi = s.windows.valueIterator();
        while (wi.next()) |d| d.deinit();
        s.windows.deinit(s.gpa);
        s.collective.deinit();
        var qi = s.seqs.valueIterator();
        while (qi.next()) |q| {
            q.*.deinit(s.gpa);
            s.gpa.destroy(q.*);
        }
        s.seqs.deinit(s.gpa);
        var si = s.saved.valueIterator();
        while (si.next()) |saved| snaps.drop(s.gpa, s.engine.stream.*, saved.*);
        s.saved.deinit(s.gpa);
        s.link.close();
        s.gpa.destroy(s);
    }

    /// Every rank's host budgets the same number of streams and cache bytes, using the least local room.
    pub fn room(s: *Session, local: u64) !u64 {
        const until = wire.deadline(600_000);
        errdefer s.poison();
        const mine: wire.Header = .{ .kind = .ready, .arg = local };
        const peer = if (s.rank == 0) peer: {
            try s.link.sendHeader(mine, until);
            break :peer try s.link.receiveHeader(until);
        } else peer: {
            const h = try s.link.receiveHeader(until);
            try s.link.sendHeader(mine, until);
            break :peer h;
        };
        if (peer.kind != .ready or peer.sequence != 0 or peer.object != 0 or peer.bytes != 0 or peer.flags != 0) return error.BadRankFrame;
        return @min(local, peer.arg);
    }

    fn command(s: *Session, input: wire.Header, payload: []const u8) !wire.Header {
        if (s.rank != 0 or s.stopped or s.collective.failed) return error.RankDisconnected;
        if (payload.len > wire.max_payload or input.bytes != payload.len) return error.BadRankFrame;
        errdefer s.poison();
        var h = input;
        h.sequence = s.next;
        s.next = std.math.add(u64, s.next, 1) catch return error.BadRankFrame;
        const until = wire.deadline(30_000);
        try s.link.sendHeader(h, until);
        try s.link.send(payload, until);
        return h;
    }

    fn ack(s: *Session, sent: wire.Header) !void {
        errdefer s.poison();
        const h = try s.link.receiveHeader(wire.deadline(30_000));
        if (h.kind != .ack or h.sequence != sent.sequence or h.object != sent.object or h.bytes != 0 or h.arg != 0) return error.BadRankFrame;
        if (h.flags == 1) return error.OutOfDeviceMemory;
        if (h.flags != 0) return error.RemoteRank;
    }

    pub fn create(s: *Session, cap: usize) !prompt.Seq {
        var q = try s.engine.newSeq(cap);
        errdefer q.deinit(s.gpa);
        const id = s.next_object;
        s.next_object = std.math.add(u64, id, 1) catch return error.BadRankFrame;
        const h = try s.command(.{ .kind = .create, .object = id, .arg = cap }, &.{});
        try s.ack(h);
        q.rank_id = id;
        return q;
    }

    pub fn drop(s: *Session, q: *prompt.Seq) void {
        s.engine.round_graph.clear();
        if (!s.stopped) {
            const h = s.command(.{ .kind = .drop_seq, .object = q.rank_id }, &.{}) catch {
                s.poison();
                q.deinit(s.gpa);
                return;
            };
            s.ack(h) catch s.poison();
        }
        q.deinit(s.gpa);
    }

    pub fn forward(s: *Session, comptime Tri: type, tri: Tri, q: *prompt.Seq, ids: []const u32, hooks: prompt.batch.Hooks) !void {
        if (ids.len == 0 or ids.len > wire.max_payload / 4 or q.fed + ids.len > q.cap or hooks.keep != null or hooks.logits == .all) return error.UnsupportedRankOperation;
        try tokenIds(ids);
        const h = try s.command(.{ .kind = .forward, .object = q.rank_id, .bytes = @intCast(ids.len * 4), .flags = @backingInt(hooks.logits) }, std.mem.sliceAsBytes(ids));
        errdefer s.poison();
        try s.engine.forward(Tri, tri, q, ids, hooks);
        try s.ack(h);
    }

    pub fn round(s: *Session, comptime Tri: type, tri: Tri, seqs: []const *prompt.Seq, ids: []const u32, out: u64) !void {
        if (seqs.len == 0 or seqs.len != ids.len or seqs.len > 32) return error.UnsupportedRankOperation;
        try tokenIds(ids);
        var raw: [32 * 12]u8 = undefined;
        for (seqs, ids, 0..) |q, token, i| {
            if (q.rank_id == 0 or q.fed >= q.cap) return error.UnexpectedTensor;
            for (seqs[0..i]) |prior| if (prior.rank_id == q.rank_id) return error.DuplicateRankSequence;
            std.mem.writeInt(u64, raw[i * 12 ..][0..8], q.rank_id, .little);
            std.mem.writeInt(u32, raw[i * 12 + 8 ..][0..4], token, .little);
        }
        const bytes = raw[0 .. seqs.len * 12];
        const h = try s.command(.{ .kind = .round, .bytes = @intCast(bytes.len) }, bytes);
        errdefer s.poison();
        try s.engine.round(Tri, tri, seqs, ids, out);
        try s.ack(h);
    }

    /// Both ranks keep their own recurrent window; only the leader consumes draft scores and hidden rows.
    pub fn verify(s: *Session, comptime Tri: type, tri: Tri, q: *prompt.Seq, d: *mtp.Draft, ids: []const u32) !void {
        if (!s.drafting or ids.len == 0 or ids.len > mtp.max_rows or q.fed + ids.len > q.cap) return error.UnsupportedRankOperation;
        try tokenIds(ids);
        const h = try s.command(.{ .kind = .verify, .object = q.rank_id, .bytes = @intCast(ids.len * 4) }, std.mem.sliceAsBytes(ids));
        errdefer s.poison();
        const ops: mtp.Ops(Tri) = .{ .e = s.engine, .tri = tri, .h = undefined };
        try ops.verify(q, d, ids);
        try s.ack(h);
    }

    pub fn keep(s: *Session, comptime Tri: type, tri: Tri, q: *prompt.Seq, d: *mtp.Draft, count: usize) !void {
        if (!s.drafting or count == 0 or count > d.rows) return error.CannotRewind;
        const h = try s.command(.{ .kind = .keep, .object = q.rank_id, .arg = count }, &.{});
        errdefer s.poison();
        const ops: mtp.Ops(Tri) = .{ .e = s.engine, .tri = tri, .h = undefined };
        try ops.rollback(q, d, count);
        try s.collective.finish(s.engine.stream.*);
        try s.ack(h);
    }

    fn window(s: *Session, q: *prompt.Seq) !*mtp.Draft {
        const item = try s.windows.getOrPut(s.gpa, q.rank_id);
        if (!item.found_existing) {
            errdefer _ = s.windows.remove(q.rank_id);
            item.value_ptr.* = try mtp.Draft.init(s.gpa, s.engine.driver, s.engine.stream.*, q.cap, 1, s.engine.chunk);
        }
        return item.value_ptr;
    }

    pub fn save(s: *Session, q: *prompt.Seq) !*snaps.Snap {
        const kept = try snaps.save(s.gpa, s.engine.driver, s.engine.stream.*, q, null);
        errdefer snaps.drop(s.gpa, s.engine.stream.*, kept);
        kept.rank_id = s.next_object;
        s.next_object = std.math.add(u64, s.next_object, 1) catch return error.BadRankFrame;
        const h = try s.command(.{ .kind = .save, .object = q.rank_id, .arg = kept.rank_id }, &.{});
        try s.ack(h);
        return kept;
    }

    pub fn restore(s: *Session, q: *prompt.Seq, kept: *const snaps.Snap) !void {
        if (kept.at > q.cap) return error.SequenceTooShort;
        const h = try s.command(.{ .kind = .restore, .object = q.rank_id, .arg = kept.rank_id }, &.{});
        errdefer s.poison();
        try snaps.restore(s.engine.stream.*, q, null, kept);
        try s.ack(h);
    }

    pub fn dropSaved(s: *Session, kept: *snaps.Snap) void {
        if (!s.stopped) {
            const h = s.command(.{ .kind = .drop_snap, .object = kept.rank_id }, &.{}) catch {
                s.poison();
                snaps.drop(s.gpa, s.engine.stream.*, kept);
                return;
            };
            s.ack(h) catch s.poison();
        }
        snaps.drop(s.gpa, s.engine.stream.*, kept);
    }

    pub fn follow(s: *Session, comptime Tri: type, tri: Tri, logits: u64) !void {
        if (s.rank != 1) return error.BadRank;
        errdefer s.poison();
        const payload = try s.gpa.alloc(u8, wire.max_payload);
        defer s.gpa.free(payload);
        while (true) {
            const h = try s.link.receiveHeader(null);
            if (h.sequence != s.next) return error.BadRankFrame;
            s.next = std.math.add(u64, s.next, 1) catch return error.BadRankFrame;
            try s.link.receive(payload[0..h.bytes], wire.deadline(30_000));
            const stop = s.apply(Tri, tri, logits, h, payload[0..h.bytes]) catch |err| {
                s.link.sendHeader(.{ .kind = .ack, .sequence = h.sequence, .object = h.object, .flags = 2 }, wire.deadline(30_000)) catch {};
                return err;
            };
            try s.link.sendHeader(.{ .kind = .ack, .sequence = h.sequence, .object = h.object }, wire.deadline(30_000));
            if (stop) {
                s.stopped = true;
                s.available.store(false, .release);
                return;
            }
        }
    }

    fn apply(s: *Session, comptime Tri: type, tri: Tri, logits: u64, h: wire.Header, payload: []const u8) !bool {
        try shape(h);
        switch (h.kind) {
            .create => {
                if (s.seqs.contains(h.object)) return error.BadRankFrame;
                const q = try s.gpa.create(prompt.Seq);
                errdefer s.gpa.destroy(q);
                q.* = try s.engine.newSeq(@intCast(h.arg));
                errdefer q.deinit(s.gpa);
                q.rank_id = h.object;
                try s.seqs.put(s.gpa, h.object, q);
            },
            .forward => {
                const q = s.seqs.get(h.object) orelse return error.UnknownRankSequence;
                const ids = try s.gpa.alloc(u32, payload.len / 4);
                defer s.gpa.free(ids);
                for (ids, 0..) |*v, i| v.* = std.mem.readInt(u32, payload[i * 4 ..][0..4], .little);
                try tokenIds(ids);
                try s.engine.forward(Tri, tri, q, ids, .{ .logits = @fromBackingInt(@intCast(h.flags)) });
            },
            .round => {
                var seqs: [32]*prompt.Seq = undefined;
                var ids: [32]u32 = undefined;
                const n = payload.len / 12;
                try roundPayload(payload);
                for (0..n) |i| {
                    seqs[i] = s.seqs.get(std.mem.readInt(u64, payload[i * 12 ..][0..8], .little)) orelse return error.UnknownRankSequence;
                    ids[i] = std.mem.readInt(u32, payload[i * 12 + 8 ..][0..4], .little);
                    if (seqs[i].fed >= seqs[i].cap) return error.UnexpectedTensor;
                }
                try s.engine.round(Tri, tri, seqs[0..n], ids[0..n], logits);
            },
            .verify => {
                if (!s.drafting) return error.UnsupportedRankOperation;
                const q = s.seqs.get(h.object) orelse return error.UnknownRankSequence;
                var ids: [mtp.max_rows]u32 = undefined;
                const n = payload.len / 4;
                for (ids[0..n], 0..) |*id, i| id.* = std.mem.readInt(u32, payload[i * 4 ..][0..4], .little);
                try tokenIds(ids[0..n]);
                const ops: mtp.Ops(Tri) = .{ .e = s.engine, .tri = tri, .h = undefined };
                try ops.verify(q, try s.window(q), ids[0..n]);
            },
            .keep => {
                if (!s.drafting) return error.UnsupportedRankOperation;
                const q = s.seqs.get(h.object) orelse return error.UnknownRankSequence;
                const d = s.windows.getPtr(h.object) orelse return error.CannotRewind;
                const ops: mtp.Ops(Tri) = .{ .e = s.engine, .tri = tri, .h = undefined };
                try ops.rollback(q, d, @intCast(h.arg));
                try s.collective.finish(s.engine.stream.*);
            },
            .drop_seq => {
                s.engine.round_graph.clear();
                if (s.windows.fetchRemove(h.object)) |entry| {
                    var d = entry.value;
                    d.deinit();
                }
                const entry = s.seqs.fetchRemove(h.object) orelse return error.UnknownRankSequence;
                entry.value.deinit(s.gpa);
                s.gpa.destroy(entry.value);
            },
            .save => {
                if (s.saved.contains(h.arg)) return error.BadRankFrame;
                const q = s.seqs.get(h.object) orelse return error.UnknownRankSequence;
                const kept = try snaps.save(s.gpa, s.engine.driver, s.engine.stream.*, q, null);
                errdefer snaps.drop(s.gpa, s.engine.stream.*, kept);
                try s.saved.put(s.gpa, h.arg, kept);
            },
            .restore => {
                const q = s.seqs.get(h.object) orelse return error.UnknownRankSequence;
                const kept = s.saved.get(h.arg) orelse return error.UnknownRankSnapshot;
                try snaps.restore(s.engine.stream.*, q, null, kept);
            },
            .drop_snap => {
                const entry = s.saved.fetchRemove(h.object) orelse return error.UnknownRankSnapshot;
                snaps.drop(s.gpa, s.engine.stream.*, entry.value);
            },
            .stop => return true,
            else => return error.BadRankFrame,
        }
        return false;
    }
};

fn tokenIds(ids: []const u32) !void {
    for (ids) |id| if (id >= wt.vocab) return error.InvalidToken;
}

fn roundPayload(payload: []const u8) !void {
    if (payload.len == 0 or payload.len % 12 != 0 or payload.len / 12 > 32) return error.BadRankFrame;
    var seen: [32]u64 = undefined;
    for (0..payload.len / 12) |i| {
        const id = std.mem.readInt(u64, payload[i * 12 ..][0..8], .little);
        const token = std.mem.readInt(u32, payload[i * 12 + 8 ..][0..4], .little);
        if (id == 0) return error.BadRankFrame;
        for (seen[0..i]) |prior| if (prior == id) return error.DuplicateRankSequence;
        try tokenIds(&.{token});
        seen[i] = id;
    }
}

fn sameIdentity(a: Identity, b: Identity) bool {
    return std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
}

/// Refuse malformed operations before looking up an owner or touching the GPU.
pub fn shape(h: wire.Header) !void {
    switch (h.kind) {
        .create => if (h.object == 0 or h.arg == 0 or h.arg > prompt.max_tokens or h.bytes != 0 or h.flags != 0) return error.BadRankFrame,
        .forward => if (h.object == 0 or h.arg != 0 or h.bytes == 0 or h.bytes % 4 != 0 or (h.flags != @backingInt(prompt.batch.Logits.last) and h.flags != @backingInt(prompt.batch.Logits.none))) return error.BadRankFrame,
        .verify => if (h.object == 0 or h.arg != 0 or h.flags != 0 or h.bytes == 0 or h.bytes % 4 != 0 or h.bytes / 4 > mtp.max_rows) return error.BadRankFrame,
        .keep => if (h.object == 0 or h.arg == 0 or h.arg > mtp.max_rows or h.bytes != 0 or h.flags != 0) return error.BadRankFrame,
        .round => if (h.object != 0 or h.arg != 0 or h.flags != 0 or h.bytes == 0 or h.bytes % 12 != 0 or h.bytes / 12 > 32) return error.BadRankFrame,
        .save, .restore => if (h.object == 0 or h.arg == 0 or h.bytes != 0 or h.flags != 0) return error.BadRankFrame,
        .drop_seq, .drop_snap => if (h.object == 0 or h.arg != 0 or h.bytes != 0 or h.flags != 0) return error.BadRankFrame,
        .stop => if (h.object != 0 or h.arg != 0 or h.bytes != 0 or h.flags != 0) return error.BadRankFrame,
        else => return error.BadRankFrame,
    }
}

test "rank identity binds checkpoint, config, index, program and execution settings" {
    const base: Identity = .{ .config = @splat(1), .index = @splat(2), .program = @splat(3), .checkpoint = @splat(4), .context = 4096, .chunk = 64 };
    try std.testing.expect(sameIdentity(base, base));
    inline for (.{ "config", "index", "program", "checkpoint" }) |name| {
        var other = base;
        @field(other, name)[0] ^= 1;
        try std.testing.expect(!sameIdentity(base, other));
    }
    var other = base;
    other.context += 1;
    try std.testing.expect(!sameIdentity(base, other));
    other = base;
    other.chunk += 1;
    try std.testing.expect(!sameIdentity(base, other));
    other = base;
    other.decode_graph = 1;
    try std.testing.expect(!sameIdentity(base, other));
    other = base;
    other.mtp = 1;
    try std.testing.expect(!sameIdentity(base, other));
}

test "rank operations refuse invalid geometry and unsupported hooks before execution" {
    try shape(.{ .kind = .create, .object = 1, .arg = 4096 });
    try shape(.{ .kind = .forward, .object = 1, .bytes = 4 });
    try shape(.{ .kind = .round, .bytes = 24 });
    try shape(.{ .kind = .verify, .object = 1, .bytes = mtp.max_rows * 4 });
    try shape(.{ .kind = .keep, .object = 1, .arg = mtp.max_rows });
    for ([_]wire.Header{
        .{ .kind = .verify, .object = 1, .bytes = (mtp.max_rows + 1) * 4 },
        .{ .kind = .verify, .object = 1, .bytes = 3 },
        .{ .kind = .verify, .bytes = 4 },
        .{ .kind = .verify, .object = 1, .bytes = 4, .flags = 1 },
        .{ .kind = .keep, .object = 1 },
        .{ .kind = .keep, .object = 1, .arg = mtp.max_rows + 1 },
        .{ .kind = .keep, .object = 1, .arg = 1, .bytes = 4 },
    }) |bad| try std.testing.expectError(error.BadRankFrame, shape(bad));
    for ([_]wire.Header{
        .{ .kind = .hello },                             .{ .kind = .create, .arg = 4096 },                          .{ .kind = .create, .object = 1, .arg = prompt.max_tokens + 1 },
        .{ .kind = .forward, .object = 1, .bytes = 3 },  .{ .kind = .forward, .object = 1, .bytes = 4, .flags = 1 }, .{ .kind = .round, .bytes = 33 * 12 },
        .{ .kind = .round, .bytes = 13 },                .{ .kind = .stop, .object = 1 },                            .{ .kind = .save, .object = 1 },
        .{ .kind = .drop_seq, .object = 1, .bytes = 4 },
    }) |h| try std.testing.expectError(error.BadRankFrame, shape(h));
}

test "round payload rejects duplicate owners and out-of-vocabulary tokens before GPU execution" {
    var bytes: [24]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], 9, .little);
    std.mem.writeInt(u64, bytes[12..20], 10, .little);
    try roundPayload(&bytes);
    std.mem.writeInt(u64, bytes[12..20], 9, .little);
    try std.testing.expectError(error.DuplicateRankSequence, roundPayload(&bytes));
    std.mem.writeInt(u64, bytes[12..20], 10, .little);
    std.mem.writeInt(u32, bytes[20..24], wt.vocab, .little);
    try std.testing.expectError(error.InvalidToken, roundPayload(&bytes));
    try std.testing.expectError(error.InvalidToken, tokenIds(&.{ 0, wt.vocab }));
    try std.testing.expectError(error.BadRankFrame, roundPayload(bytes[0..13]));
}
