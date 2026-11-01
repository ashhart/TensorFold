//! TP over processes: rank 0's Leader sends each step (a header, then ids) over NCCL first; the other ranks `follow`.

const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const kern = @import("cuda_kernels.zig");
const sampler = @import("cuda_sampler.zig");
const Engine = engine.Engine;

pub const Kind = enum(u32) { begin = 1, chunk, verify, shared, keep, release, idle, end, block, save, drop, image, redraw };

/// One step as it crosses: 64 bytes, then `n` u32 ids.
pub const Header = extern struct {
    kind: u32,
    n: u32 = 0, // ids that follow
    depth: u32 = 0, // shared: its parts; chunk: where it starts
    flags: u32 = 0, // begin: 1 sampled
    a: u64 = 0, // keep: rows; chunk: the prompt's length; save: the state's id
    seed: u64 = 0, // chunk: a kept state's id to resume from (0: none); save: its position
    temperature: f64 = 0,
    top_p: f64 = 0,
    min_p: f64 = 0,
    top_k: u32 = 0,
    slot: u32 = 0, // the stream's sequence the step runs on

    comptime {
        std.debug.assert(@sizeOf(Header) == 64);
    }
};

/// tf_mm3_put_words' argument: a step's header (16 words) and up to `put_ids` ids.
pub const Words = extern struct { w: [96]u32 };
pub const put_ids = 96 - 16;

pub const Link = struct {
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    buf: cuda.DeviceBuffer, // header, then ids
    ids: []u32, // host copy of the last step's ids
    gpa: std.mem.Allocator,
    asleep: bool = false, // rank 0: the other ranks sleep on their lines (cuda.tp_watch) until its next step

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, max_ids: usize) !Link {
        const buf = try cuda.DeviceBuffer.alloc(d, 64 + max_ids * 4);
        errdefer {
            var owned = buf;
            owned.free();
        }
        return .{ .d = d, .lib = lib, .comm = comm, .stream = stream, .buf = buf, .ids = try gpa.alloc(u32, max_ids), .gpa = gpa };
    }

    pub fn deinit(l: *Link) void {
        l.gpa.free(l.ids);
        l.buf.free();
    }

    fn post(l: *Link, bytes: usize) !void {
        try l.lib.check(l.lib.api.ncclBroadcast(l.buf.ptr, l.buf.ptr, bytes, .u8, 0, l.comm, l.stream.handle), "ncclBroadcast");
    }

    /// Rank 0: the step, then its ids, to every rank in two broadcasts; up to `put_ids` ids by a kernel, unwaited.
    pub fn send(l: *Link, put: cuda.Function, h: Header, ids: []const u32) !void {
        if (ids.len > l.ids.len) return error.TooManyIds;
        var hh = h;
        hh.n = @intCast(ids.len);
        if (ids.len <= put_ids) {
            var w: Words = undefined;
            @memcpy(w.w[0..16], std.mem.bytesAsSlice(u32, std.mem.asBytes(&hh)));
            @memcpy(w.w[16..][0..ids.len], ids);
            var a: cuda.Args = .{};
            a.add(l.buf.ptr);
            a.add(@as(u32, @intCast(16 + ids.len)));
            a.add(w);
            try cuda.launch.launch(put, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, l.stream, &a);
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

/// Rank 0's engine: every step goes to the other ranks first.
pub const Leader = struct {
    e: *Engine,
    link: ?*Link, // null: one rank

    fn send(x: Leader, h: Header, ids: []const u32) !void {
        const l = x.link orelse return;
        if (h.kind != @backingInt(Kind.drop)) try x.flush();
        var hh = h;
        hh.slot = x.e.slot;
        if (l.asleep) {
            cuda.tp_watch.wake();
            l.asleep = false;
        }
        try l.send(x.e.m.k.put_words, hh, ids);
    }

    /// States rank 0 dropped since the last step, so the other ranks free theirs too.
    fn flush(x: Leader) anyerror!void {
        const d = &x.e.dropped;
        while (d.items.len > 0) {
            const n = @min(d.items.len, put_ids / 2);
            var ids: [put_ids]u32 = undefined;
            for (d.items[0..n], 0..) |id, i| ids[2 * i ..][0..2].* = .{ @truncate(id), @truncate(id >> 32) };
            try x.send(.{ .kind = @backingInt(Kind.drop) }, ids[0 .. 2 * n]);
            d.replaceRangeAssumeCapacity(0, n, &.{});
        }
    }

    /// The prompt pass's images on every rank (each image's gh, gw, at and tokens; its patches from here).
    pub fn images(x: Leader, imgs: []const @import("lanes").stream.Image) !void {
        var spans: [64][4]u32 = undefined;
        var px: [64][]const f32 = undefined;
        if (imgs.len > spans.len) return error.TooManyImages;
        for (imgs, 0..) |im, i| {
            spans[i] = .{ im.gh, im.gw, im.at, im.tokens };
            px[i] = im.pixels;
        }
        const flat: []const u32 = @ptrCast(spans[0..imgs.len]);
        try engine.Engine.imageLimits(x.e.images orelse return error.NoVision, spans[0..imgs.len]);
        try x.send(.{ .kind = @backingInt(Kind.image) }, flat);
        try x.e.encodeImages(spans[0..imgs.len], px[0..imgs.len]);
    }

    /// The bound sequence's state at `at` kept on every rank under one new id.
    pub fn save(x: Leader, at: usize) !engine.Kept {
        const id = x.e.next_snap;
        x.e.next_snap += 1; // an id is used once, even when this rank's copy then fails
        if (at == 0 or at > x.e.pos()) return error.BadKeep;
        try x.send(.{ .kind = @backingInt(Kind.save), .a = id, .seed = at }, &.{});
        x.e.keepAs(id, at) catch |err| {
            x.drop(id); // the other ranks free their copy at the next step
            return err;
        };
        return .{ .id = id, .at = @intCast(at) };
    }

    /// A state rank 0's store let go: freed here now, on the other ranks at the next step.
    pub fn drop(x: Leader, id: u64) void {
        x.e.dropState(id);
        if (x.link != null) x.e.dropped.append(x.e.gpa, id) catch {};
    }

    /// No stream is left: the other ranks sleep on their lines until the next step instead of spinning in NCCL.
    pub fn idle(x: Leader) !void {
        const l = x.link orelse return;
        if (l.asleep or !cuda.tp_watch.open()) return;
        try x.send(.{ .kind = @backingInt(Kind.idle) }, &.{});
        l.asleep = true;
    }

    pub fn bind(x: Leader, slot: u32) !void {
        try x.e.bind(slot);
    }

    pub fn release(x: Leader, slot: u32) !void {
        try x.e.bind(slot);
        try x.send(.{ .kind = @backingInt(Kind.release) }, &.{});
        x.e.release(slot);
    }

    pub fn begin(x: Leader, s: ?sampler.Sampling) !void {
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

    /// A prompt chunk on every rank, `from`'s state restored first; the first token once it ends the prompt.
    pub fn chunk(x: Leader, ids: []const u32, start: usize, from: ?engine.Resume, cancel: ?engine.Cancel) !?u32 {
        if (start >= ids.len) return error.BadChunk;
        const part = ids[start..@min(ids.len, start + engine.prompt_rows)];
        // rank 0's cancel decides before the step goes out: every rank runs a chunk it is sent
        if (cancel) |c| if (c.check(c.ptr)) return error.Cancelled;
        if (start % engine.prompt_rows != 0 or (from == null and start != x.e.pos()) or (from != null and (from.?.at != start or x.e.keptAt(from.?.id) != start))) return error.BadChunk;
        try x.send(.{ .kind = @backingInt(Kind.chunk), .depth = @intCast(start), .a = ids.len, .seed = if (from) |r| r.id else 0 }, part);
        if (from) |r| try x.e.restoreState(r.id);
        return x.e.prefillChunk(ids, start, null);
    }

    /// `bits`: the rows' grammar words (structured output, with Engine.structured), else 0.
    pub fn verify(x: Leader, ids: []const u32, out: []u32, bits: u64) !void {
        try x.send(.{ .kind = @backingInt(Kind.verify), .flags = @intFromBool(x.e.structured) }, ids);
        try x.e.verify(ids, out, bits);
    }

    /// Structured output's first token: every rank gathers the prompt's last row again, rank 0 draws it among `bits`.
    pub fn redraw(x: Leader, len: usize, bits: u64) !u32 {
        try x.send(.{ .kind = @backingInt(Kind.redraw), .a = len }, &.{});
        return x.e.redrawFirst(len, bits);
    }

    /// Several streams' windows in one forward on every rank: parts' slots and rows, then ids.
    pub fn verifyShared(x: Leader, parts: []const engine.Part) !void {
        if (parts.len == 0 or parts.len > engine.max_shared) return error.WindowTooWide;
        var buf: [2 * engine.max_shared + engine.shared_rows]u32 = undefined;
        var n: usize = 2 * parts.len;
        var rows: usize = 0;
        for (parts, 0..) |p, i| {
            rows += p.ids.len;
            if (n + p.ids.len > buf.len or rows > engine.shared_rows or p.ids.len == 0 or p.ids.len > engine.window_rows) return error.WindowTooWide;
            for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
            buf[2 * i] = p.slot;
            buf[2 * i + 1] = @intCast(p.ids.len);
            @memcpy(buf[n..][0..p.ids.len], p.ids);
            n += p.ids.len;
        }
        try x.send(.{ .kind = @backingInt(Kind.shared), .depth = @intCast(parts.len), .flags = @intFromBool(x.e.structured) }, buf[0..n]);
        try x.e.verifyShared(parts);
    }

    /// Every rank's draft round (Engine.draftBlock): each request's slot, kept rows, anchor and drafts wanted.
    pub fn block(x: Leader, reqs: []const engine.Req) !void {
        if (reqs.len == 0 or reqs.len > engine.max_shared) return error.TooManyStreams;
        var buf: [4 * engine.max_shared]u32 = undefined;
        for (reqs, 0..) |q, i| buf[4 * i ..][0..4].* = .{ q.slot, q.n, q.anchor, @intCast(q.out.len) };
        try x.send(.{ .kind = @backingInt(Kind.block), .depth = @intCast(reqs.len) }, buf[0 .. 4 * reqs.len]);
        try x.e.draftBlock(reqs);
    }

    pub fn keep(x: Leader, n: usize) !void {
        try x.send(.{ .kind = @backingInt(Kind.keep), .a = n }, &.{});
        try x.e.keep(n);
    }

    pub fn end(x: Leader) !void {
        try x.send(.{ .kind = @backingInt(Kind.end) }, &.{});
    }
};

/// A rank other than 0 runs rank 0's steps until it ends (rank 0 decides cancels before it sends a step).
pub fn follow(e: *Engine, link: *Link) !void {
    var sampled: [engine.shared_rows]u32 = undefined;
    cuda.tp_watch.sleep(e.io);
    while (true) {
        const step = try link.receive();
        const h = step.h;
        const kind: Kind = @fromBackingInt(@intCast(h.kind));
        e.structured = (kind == .verify or kind == .shared) and h.flags & 1 != 0; // rank 0 alone masks
        if (kind != .end and kind != .shared and kind != .idle and kind != .block and kind != .drop) try e.bind(h.slot);
        switch (kind) {
            .begin => e.begin(if (h.flags & 1 != 0) .{ .seed = h.seed, .temperature = h.temperature, .top_p = h.top_p, .min_p = h.min_p, .top_k = h.top_k } else null),
            .chunk => {
                // the prompt's ids past this chunk are not needed here: the chunk's own rows carry its positions
                const ids = try e.gpa.alloc(u32, h.a);
                defer e.gpa.free(ids);
                @memcpy(ids[h.depth..][0..step.ids.len], step.ids);
                if (h.seed != 0) try e.restoreState(h.seed);
                _ = try e.prefillChunk(ids, h.depth, null);
            },
            .verify => try e.verify(step.ids, sampled[0..step.ids.len], 0),
            .redraw => _ = try e.redrawFirst(@intCast(h.a), 0),
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
            .keep => try e.keep(@intCast(h.a)),
            .block => {
                const k = h.depth;
                if (k == 0 or k > engine.max_shared or step.ids.len != 4 * k) return error.BadStep;
                var reqs: [engine.max_shared]engine.Req = undefined;
                var outs: [engine.max_shared][16]u32 = undefined;
                for (reqs[0..k], 0..) |*q, i| {
                    const n_out = step.ids[4 * i + 3];
                    if (n_out > 16) return error.BadStep;
                    q.* = .{ .slot = step.ids[4 * i], .n = step.ids[4 * i + 1], .anchor = step.ids[4 * i + 2], .out = outs[i][0..n_out] };
                }
                try e.draftBlock(reqs[0..k]);
            },
            .release => e.release(h.slot),
            .save => try e.keepAs(h.a, @intCast(h.seed)),
            .image => {
                if (step.ids.len % 4 != 0) return error.BadStep;
                const spans: []const [4]u32 = @ptrCast(step.ids);
                try e.encodeImages(spans, null);
            },
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
    const h: Header = .{ .kind = @backingInt(Kind.verify), .n = 3, .depth = 2, .flags = 1, .a = 7, .seed = 1 << 62, .temperature = 0.6, .top_p = 0.95, .top_k = 20 };
    const bytes = std.mem.asBytes(&h);
    var back: Header = undefined;
    @memcpy(std.mem.asBytes(&back), bytes);
    try std.testing.expectEqual(h, back);
}
