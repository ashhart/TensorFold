//! TP over processes: rank 0's Leader sends each step (a header, then ids) by NCCL first; the other ranks `follow`.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("engine.zig");
const ops = @import("ops.zig");
const sampler = @import("sampler.zig");
const Engine = engine.Engine;
const image = @import("image.zig");
const vision = @import("vision.zig");
const pictures = @import("pictures.zig");

pub const Kind = enum(u32) { begin = 1, chunk, shared, release, idle, end, save, drop, reserve, images };

/// One step as it crosses: 64 bytes, then `n` u32 ids. NCCL waits as long as it takes, so a follower idles safely.
pub const Header = extern struct {
    kind: u32,
    n: u32 = 0, // ids that follow
    depth: u32 = 0, // shared: its parts; chunk: where it starts
    flags: u32 = 0, // begin: 1 sampled
    a: u64 = 0, // chunk: the prompt's length; save: the state's id; reserve: slots
    seed: u64 = 0, // chunk: a kept state's id to resume from (0: none); save: its position
    temperature: f64 = 0,
    top_p: f64 = 0,
    min_p: f64 = 0,
    top_k: u32 = 0,
    slot: u32 = 0, // the stream's slot the step runs on

    comptime {
        std.debug.assert(@sizeOf(Header) == 64);
    }
};

pub const put_ids = 96 - 16;

pub const Link = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    buf: cuda.DeviceBuffer, // header, then ids
    ids: []u32, // host copy of the last step's ids
    gpa: std.mem.Allocator,
    asleep: bool = false, // rank 0: the other ranks sleep on their lines (cuda.tp_watch) until its next step

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, max_ids: usize) !Link {
        var buf = try cuda.DeviceBuffer.alloc(d, 64 + max_ids * 4);
        errdefer buf.free();
        return .{ .lib = lib, .comm = comm, .stream = stream, .buf = buf, .ids = try gpa.alloc(u32, max_ids), .gpa = gpa };
    }

    pub fn deinit(l: *Link) void {
        // the last step's broadcast (rank 0's end) may still read buf
        cuda.completion.drain(l.stream.d, l.stream, "the step link's last broadcast did not drain before its buffer was freed");
        l.gpa.free(l.ids);
        l.buf.free();
    }

    fn post(l: *Link, bytes: usize) !void {
        try l.lib.check(l.lib.api.ncclBroadcast(l.buf.ptr, l.buf.ptr, bytes, .u8, 0, l.comm, l.stream.handle), "ncclBroadcast");
    }

    /// Rank 0: the step, then its ids, to every rank in two broadcasts; up to `put_ids` ids by a kernel, unwaited.
    pub fn send(l: *Link, k: *const ops.Kernels, h: Header, ids: []const u32) !void {
        if (ids.len > l.ids.len) return error.TooManyIds;
        var hh = h;
        hh.n = @intCast(ids.len);
        if (ids.len <= put_ids) {
            var w: ops.Words = undefined;
            @memcpy(w.w[0..16], std.mem.bytesAsSlice(u32, std.mem.asBytes(&hh)));
            @memcpy(w.w[16..][0..ids.len], ids);
            try ops.putWords(k, l.stream, l.buf.ptr, 16 + ids.len, w);
        } else {
            try l.buf.uploadAsync(0, std.mem.asBytes(&hh), l.stream.handle);
            try l.buf.uploadAsync(64, std.mem.sliceAsBytes(ids), l.stream.handle);
        }
        try l.post(64);
        if (ids.len > 0) try l.post(64 + ids.len * 4);
        if (ids.len > put_ids) try l.stream.synchronize(); // the host ids' upload is done before they change
    }

    /// Other ranks: the next step and its ids (valid until the next receive).
    pub fn receive(l: *Link) !struct { h: Header, ids: []const u32 } {
        try l.post(64);
        try l.stream.synchronize();
        var h: Header = undefined;
        try l.buf.download(0, std.mem.asBytes(&h));
        if (h.n > l.ids.len) return error.TooManyIds;
        if (h.n > 0) {
            try l.post(64 + h.n * 4);
            try l.stream.synchronize();
            try l.buf.download(64, std.mem.sliceAsBytes(l.ids[0..h.n]));
        }
        return .{ .h = h, .ids = l.ids[0..h.n] };
    }
};

/// A kept prompt state: its id (the same on every rank) and the position it stands at.
pub const Kept = struct { id: u64, at: u32 };

/// Rank 0's engine: every step goes to the other ranks first.
pub const Leader = struct {
    e: *Engine,
    link: ?*Link, // null: one rank
    next_id: u64 = 1,
    dropped: std.ArrayList(u64) = .empty, // ids dropped since the other ranks last heard

    pub fn deinit(x: *Leader) void {
        x.dropped.deinit(x.e.gpa);
    }

    fn send(x: *Leader, h: Header, ids: []const u32) !void {
        const l = x.link orelse return;
        if (h.kind != @backingInt(Kind.drop)) try x.flush();
        var hh = h;
        hh.slot = x.e.slot;
        if (l.asleep) {
            cuda.tp_watch.wake();
            l.asleep = false;
        }
        try l.send(&x.e.m.ops, hh, ids);
    }

    /// States rank 0 dropped since the last step, so the other ranks free theirs too.
    fn flush(x: *Leader) anyerror!void {
        const d = &x.dropped;
        while (d.items.len > 0) {
            const n = @min(d.items.len, put_ids / 2);
            var ids: [put_ids]u32 = undefined;
            for (d.items[0..n], 0..) |id, i| ids[2 * i ..][0..2].* = .{ @truncate(id), @truncate(id >> 32) };
            try x.send(.{ .kind = @backingInt(Kind.drop) }, ids[0 .. 2 * n]);
            d.replaceRangeAssumeCapacity(0, n, &.{});
        }
    }

    /// Every rank's caches for `slots` streams.
    pub fn reserve(x: *Leader, slots: usize) !void {
        try x.send(.{ .kind = @backingInt(Kind.reserve), .a = slots }, &.{});
        try x.e.reserve(slots);
    }

    /// The bound slot's state at `at` kept on every rank under one new id.
    pub fn save(x: *Leader, at: usize) !Kept {
        const id = x.next_id;
        x.next_id += 1; // an id is used once, even when this rank's copy then fails
        if (at == 0 or at != x.e.pos()) return error.BadKeep;
        try x.send(.{ .kind = @backingInt(Kind.save), .a = id, .seed = at }, &.{});
        x.e.keepAs(id, at) catch |err| {
            x.drop(id);
            return err;
        };
        return .{ .id = id, .at = @intCast(at) };
    }

    /// A state rank 0's store let go: freed here now, on the other ranks at the next step.
    pub fn drop(x: *Leader, id: u64) void {
        x.e.dropState(id);
        if (x.link != null) x.dropped.append(x.e.gpa, id) catch {};
    }

    /// No stream is left: the other ranks sleep on their lines until the next step instead of spinning in NCCL.
    pub fn idle(x: *Leader) !void {
        const l = x.link orelse return;
        if (l.asleep or !cuda.tp_watch.open()) return;
        try x.send(.{ .kind = @backingInt(Kind.idle) }, &.{});
        l.asleep = true;
    }

    pub fn bind(x: *Leader, slot: u32) !void {
        try x.e.bind(slot);
    }

    pub fn release(x: *Leader, slot: u32) !void {
        try x.e.bind(slot);
        try x.send(.{ .kind = @backingInt(Kind.release) }, &.{});
    }

    pub fn begin(x: *Leader, s: ?sampler.Sampling) !void {
        const c = sampler.check(s);
        var h: Header = .{ .kind = @backingInt(Kind.begin) };
        if (c) |v| {
            h.flags = 1;
            h.seed = v.seed;
            h.temperature = v.temperature;
            h.top_p = v.top_p;
            h.min_p = v.min_p;
            h.top_k = v.top_k;
        }
        try x.send(h, &.{});
        x.e.begin(c);
    }

    /// A prompt chunk on every rank, kept state `from` restored first; the first token once it ends the prompt.
    pub fn chunk(x: *Leader, ids: []const u32, start: usize, from: ?u64, cancel: ?engine.Cancel) !?u32 {
        if (start >= ids.len) return error.BadChunk;
        const part = ids[start..@min(ids.len, start + engine.prompt_rows)];
        // rank 0's cancel decides before the step goes out: every rank runs a chunk it is sent
        if (cancel) |c| if (c.check(c.ptr)) return error.Cancelled;
        if (from) |id| {
            const w = x.e.states.where(id) orelse return error.SnapshotStale;
            if (w.at != start or w.slot != x.e.slot) return error.BadChunk;
        } else if (start % engine.prompt_rows != 0 or (start != 0 and start != x.e.pos())) return error.BadChunk;
        try x.send(.{ .kind = @backingInt(Kind.chunk), .depth = @intCast(start), .a = ids.len, .seed = from orelse 0 }, part);
        if (from) |id| try x.e.restoreState(id);
        return x.e.prefillChunk(ids, start);
    }

    /// Several streams' rows in one forward on every rank: parts' slots and rows, then ids.
    pub fn verifyShared(x: *Leader, parts: []const engine.Part) !void {
        if (parts.len == 0 or parts.len > engine.max_shared) return error.WindowTooWide;
        var buf: [2 * engine.max_shared + engine.shared_rows]u32 = undefined;
        var n: usize = 2 * parts.len;
        for (parts, 0..) |p, i| {
            if (n + p.ids.len > buf.len or p.ids.len == 0 or p.ids.len > engine.window_rows) return error.WindowTooWide;
            for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
            buf[2 * i] = p.slot;
            buf[2 * i + 1] = @intCast(p.ids.len);
            @memcpy(buf[n..][0..p.ids.len], p.ids);
            n += p.ids.len;
        }
        try x.send(.{ .kind = @backingInt(Kind.shared), .depth = @intCast(parts.len) }, buf[0..n]);
        try x.e.verifyShared(parts);
    }

    /// The bound slot's pictures on every rank: their spans go out first, then rank 0 encodes and sends the rows.
    pub fn images(x: *Leader, pics: []const image.Prepared, spans: []const vision.Span) !void {
        var words: [3 * 64]u32 = undefined;
        _ = try pictures.checkSpans(spans, x.e.vision orelse return error.NoVision);
        try x.send(.{ .kind = @backingInt(Kind.images), .depth = @intCast(spans.len) }, try pictures.toWords(spans, &words));
        try x.e.images(pics, spans);
    }

    pub fn end(x: *Leader) !void {
        try x.send(.{ .kind = @backingInt(Kind.end) }, &.{});
    }
};

/// A rank other than 0 runs rank 0's steps until it ends (rank 0 decides cancels before it sends a step).
pub fn follow(e: *Engine, link: *Link) !void {
    cuda.tp_watch.sleep(e.io);
    while (true) {
        const step = try link.receive();
        const h = step.h;
        const kind: Kind = @fromBackingInt(@intCast(h.kind));
        if (kind == .begin or kind == .chunk or kind == .save or kind == .images) try e.bind(h.slot);
        switch (kind) {
            .reserve => try e.reserve(@intCast(h.a)),
            .begin => e.begin(if (h.flags & 1 != 0) .{ .seed = h.seed, .temperature = h.temperature, .top_p = h.top_p, .min_p = h.min_p, .top_k = h.top_k } else null),
            .chunk => {
                // the prompt's ids past this chunk are not needed here: the chunk's own rows carry its positions
                const ids = try e.gpa.alloc(u32, h.a);
                defer e.gpa.free(ids);
                @memset(ids, 0);
                @memcpy(ids[h.depth..][0..step.ids.len], step.ids);
                if (h.seed != 0) try e.restoreState(h.seed);
                _ = try e.prefillChunk(ids, h.depth);
            },
            .shared => {
                const k = h.depth;
                if (k == 0 or k > engine.max_shared or step.ids.len < 2 * k) return error.BadStep;
                var parts: [engine.max_shared]engine.Part = undefined;
                var outs: [engine.shared_rows]u32 = undefined;
                var at: usize = 2 * k;
                var row: usize = 0;
                for (parts[0..k], 0..) |*p, i| {
                    const n = step.ids[2 * i + 1];
                    if (at + n > step.ids.len or row + n > outs.len) return error.BadStep;
                    p.* = .{ .slot = step.ids[2 * i], .ids = step.ids[at..][0..n], .out = outs[row..][0..n] };
                    at += n;
                    row += n;
                }
                try e.verifyShared(parts[0..k]);
            },
            .images => {
                var spans: [64]vision.Span = undefined;
                try e.images(null, try pictures.fromWords(step.ids, &spans));
            },
            .release => {},
            .save => try e.keepAs(h.a, @intCast(h.seed)),
            .drop => {
                var i: usize = 0;
                while (i + 1 < step.ids.len) : (i += 2) e.dropState(@as(u64, step.ids[i]) | @as(u64, step.ids[i + 1]) << 32);
            },
            .idle => cuda.tp_watch.sleep(e.io),
            .end => return cuda.tp_watch.ended(),
        }
    }
}

test "a step's header is 64 bytes and keeps its fields" {
    const h: Header = .{ .kind = @backingInt(Kind.shared), .n = 3, .depth = 2, .flags = 1, .a = 7, .seed = 1 << 62, .temperature = 0.6, .top_p = 0.95, .top_k = 20 };
    var back: Header = undefined;
    @memcpy(std.mem.asBytes(&back), std.mem.asBytes(&h));
    try std.testing.expectEqual(h, back);
}
