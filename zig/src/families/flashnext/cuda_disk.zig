//! Flash Next's CUDA kept states on disk (--learn-conversations): their image, device copies out and in, the probe.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const snap = @import("cuda_snap.zig");
const rows = api.prompt_cache.rows;
const sh = snap.shape;
const Allocator = std.mem.Allocator;

const words = 32; // the n-gram history, the position, whether the MTP head's rows follow
const Words = extern struct { history: [2]i64, at: u64, head: u64 };

/// The image's parts at `at`: words, DeltaNet tails and states, each attention piece by layer, the tail, the head's.
pub fn partsOf(at: u32, head: bool, out: *std.ArrayList(rows.image.Part), a: Allocator) !void {
    try out.append(a, .{ .bytes = words });
    try out.append(a, .{ .bytes = sh.linear * sh.conv });
    try out.append(a, .{ .bytes = sh.linear * sh.state });
    try attention(at, sh.attn, out, a);
    try out.append(a, .{ .bytes = sh.tail });
    if (head) try attention(at, 1, out, a);
}

/// As cuda_snap.zig's move lays them: every layer's keys, then values, indexer keys, and pooled rows of `ratio`.
fn attention(at: u32, layers: usize, out: *std.ArrayList(rows.image.Part), a: Allocator) !void {
    for (0..2 * layers) |_| try out.append(a, .{ .bytes = @as(u64, at) * sh.kv_row, .row = sh.kv_row });
    for (0..layers) |_| try out.append(a, .{ .bytes = @as(u64, at) * sh.index_row, .row = sh.index_row });
    for (0..layers) |_| try out.append(a, .{ .bytes = @as(u64, at / sh.ratio) * sh.index_row, .row = sh.index_row, .per = sh.ratio });
}

/// The disk tier of a Flash Next namespace `F` (its Cuda backend is the prompt cache's target).
pub fn Kept(comptime F: type) type {
    return struct {
        pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .whole = whole };
        /// The states are one tensor rank's: --tp refuses --learn-conversations (native/cuda_refusal.zig).
        pub const one_rank = true;

        fn of(ptr: *anyopaque) *F.Cuda {
            return @ptrCast(@alignCast(ptr));
        }

        fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
            try partsOf(at, of(ptr).head != null, out, a);
        }

        fn copy(c: *F.Cuda, st: *snap.Snap, ranges: []const rows.image.Range, host: []u8, out: bool) !void {
            const d = c.engine.driver;
            // a later range's failure leaves earlier copies queued on `host`, which its caller then frees or reuses
            errdefer cuda.completion.drain(d, c.stream, "a failed kept-state copy did not drain");
            var w: Words = .{ .history = st.history, .at = st.at, .head = @intFromBool(st.head) };
            var o: usize = 0;
            for (ranges) |g| {
                var off: usize = @intCast(g.off);
                var len: usize = @intCast(g.len);
                if (off + len > words + st.buf.len) return error.BadKeptRange;
                if (off < words) {
                    const n = @min(len, words - off);
                    if (out) @memcpy(host[o..][0..n], std.mem.asBytes(&w)[off..][0..n]) else @memcpy(std.mem.asBytes(&w)[off..][0..n], host[o..][0..n]);
                    o += n;
                    off += n;
                    len -= n;
                }
                if (len == 0) continue;
                const dev = st.buf.ptr + off - words;
                if (out) {
                    try d.check(d.api.cuMemcpyDtoHAsync_v2(host[o..].ptr, dev, len, c.stream.handle), "cuMemcpyDtoHAsync");
                } else try d.check(d.api.cuMemcpyHtoDAsync_v2(dev, host[o..].ptr, len, c.stream.handle), "cuMemcpyHtoDAsync");
                o += len;
            }
            try c.stream.synchronize(); // `host` is the store's staging, reused by its next piece
            if (out) return;
            // a state read back stands where it was kept, with the head's rows as kept: else it restores nowhere
            if (w.at != st.at or w.head != @intFromBool(st.head)) return error.KeptWordsDiffer;
            st.history = w.history;
        }

        fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
            try copy(of(ptr), @ptrCast(@alignCast(saved)), ranges, out, true);
        }

        fn scatter(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
            try copy(of(ptr), @ptrCast(@alignCast(saved)), ranges, @constCast(in), false);
        }

        /// A device buffer for `at` tokens with the head's rows if this server drafts, filled by `scatter`.
        fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
            const c = of(ptr);
            const head = c.head != null;
            const st = try c.gpa.create(snap.Snap);
            errdefer c.gpa.destroy(st);
            st.* = .{ .buf = try cuda.DeviceBuffer.alloc(c.engine.driver, snap.bytes(at, head)), .at = at, .history = .{ 0, 0 }, .head = head };
            return @ptrCast(st);
        }

        /// A state from a stream that did not draft lacks the head's rows this server's parts name: it stays off disk.
        fn whole(ptr: *anyopaque, saved: api.prompt_cache.Saved) bool {
            const st: *const snap.Snap = @ptrCast(@alignCast(saved));
            return st.head == (of(ptr).head != null) and st.rank_id == 0;
        }

        /// A fixed prompt's state at a mark inside its last chunk, hashed, and the prompt rate it took.
        pub fn probe(a: Allocator, io: std.Io, backend: lanes.backend.Backend, target: *anyopaque) !struct { hash: u64, rate: f64 } {
            const c = of(target);
            var toks: [4096 + 40 + 7]u32 = undefined;
            for (&toks, 0..) |*x, i| x.* = @intCast(1000 + i * 7919 % 50000);
            const Hook = struct {
                c: *F.Cuda,
                saved: ?*anyopaque = null,
                failed: ?anyerror = null,
                fn at(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
                    const p: *@This() = @ptrCast(@alignCast(ptr));
                    p.saved = F.snap_save(p.c, s, mark) catch |e| {
                        p.failed = e;
                        return;
                    };
                }
            };
            var hook: Hook = .{ .c = c };
            defer if (hook.saved) |saved| F.snap_drop(c, saved);
            const marks = [_]u32{4136};
            var s = try lanes.Stream.init(a, .{ .id = "probe", .prompt = &toks, .max_new = 1, .drafts = c.head != null, .reuse = .{ .marks = &marks, .hook = .{ .ptr = &hook, .at = Hook.at } } });
            defer s.deinit(a);
            const ns = pass: {
                defer backend.vtable.release(backend.ptr, &s); // once, whether the pass, its wait or nothing failed
                const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
                try backend.vtable.prefill(backend.ptr, &s);
                try c.stream.synchronize();
                break :pass std.Io.Clock.awake.now(io).toNanoseconds() - t0;
            };
            const saved = hook.saved orelse return hook.failed orelse error.ProbeUnmarked;
            if (!whole(c, saved)) return error.ProbeUnmarked;
            const st: *snap.Snap = @ptrCast(@alignCast(saved));
            const bytes = try a.alloc(u8, words + st.buf.len);
            try copy(c, st, &.{.{ .off = 0, .len = bytes.len }}, bytes, true);
            return .{ .hash = std.hash.Wyhash.hash(0x666e, bytes), .rate = @as(f64, @floatFromInt(toks.len)) / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
        }

        /// What else the states depend on: whether the MTP head's rows are part of them.
        pub fn facts(target: *anyopaque) [1]u64 {
            return .{@intFromBool(of(target).head != null)};
        }
    };
}

test "the image covers a state's device bytes once, its rows split into 256-token blocks, with and without the head" {
    const a = std.testing.allocator;
    for ([_]bool{ false, true }) |head| for ([_]u32{ 1, 3, 256, 1000, 4137 }) |at| {
        var parts: std.ArrayList(rows.image.Part) = .empty;
        defer parts.deinit(a);
        try partsOf(at, head, &parts, a);
        try std.testing.expect(rows.image.valid(parts.items, at, rows.block_tokens));
        try std.testing.expectEqual(words + snap.bytes(at, head), rows.image.imageBytes(parts.items));
    };
}

test "the image's parts are cuda_snap.move's pieces in its order: sizes, offsets and rows, with and without the head" {
    const a = std.testing.allocator;
    for ([_]bool{ false, true }) |head| for ([_]u32{ 0, 1, 3, 4, 255, 256, 1000, 4136, 4137 }) |at| {
        var parts: std.ArrayList(rows.image.Part) = .empty;
        defer parts.deinit(a);
        try partsOf(at, head, &parts, a);
        try std.testing.expectEqual(@as(u64, words), parts.items[0].bytes);
        const o = snap.order(at, head);
        var i: usize = 1;
        var off: u64 = words; // the image's offset; move's buffer offset is this less the words
        var moved: u64 = 0;
        for (o.p[0..o.len]) |p| {
            if (p.row == 0) { // the DeltaNet's pieces and the tail: one part of every layer's bytes
                try std.testing.expectEqual(@as(u64, p.layers * p.n), parts.items[i].bytes);
                try std.testing.expectEqual(@as(u32, 0), parts.items[i].row);
                off += parts.items[i].bytes;
                moved += p.layers * p.n;
                i += 1;
                continue;
            }
            for (0..p.layers) |_| { // each layer's rows: keys, values, indexer keys, pooled blocks
                const x = parts.items[i];
                try std.testing.expectEqual(@as(u64, p.n), x.bytes);
                try std.testing.expectEqual(@as(u64, p.row), x.row);
                try std.testing.expectEqual(@as(u64, p.per), x.per);
                try std.testing.expectEqual(off - words, moved);
                off += x.bytes;
                moved += p.n;
                i += 1;
            }
        }
        try std.testing.expectEqual(parts.items.len, i);
        try std.testing.expectEqual(@as(u64, snap.bytes(at, head)), moved);
        try std.testing.expectEqual(rows.image.imageBytes(parts.items), off);
    };
}

/// A driver whose device copies, out or in, queue the first and refuse the next; `pending`: queued, not yet waited for.
const OneCopy = struct {
    var queued: usize = 0;
    var pending: usize = 0;
    fn queue() cuda.abi.Result {
        if (queued == 1) return cuda.abi.error_not_ready; // check() returns NotReady for it, logging nothing
        queued += 1;
        pending += 1;
        return cuda.abi.success;
    }
    fn toHost(_: ?*anyopaque, _: u64, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return queue();
    }
    fn toDevice(_: u64, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return queue();
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        pending = 0;
        return cuda.abi.success;
    }
};

/// A Flash Next namespace whose Cuda has just the fields its image reads, and no MTP head.
const Bare = struct {
    pub const Cuda = struct { gpa: Allocator, engine: struct { driver: *const cuda.Driver }, head: ?*anyopaque = null, stream: cuda.Stream };
};

test "Flash Next completion: a kept state's copy out or in that fails part way drains the copies it queued" {
    var d: cuda.Driver = undefined;
    d.api.cuMemcpyDtoHAsync_v2 = OneCopy.toHost;
    d.api.cuMemcpyHtoDAsync_v2 = OneCopy.toDevice;
    d.api.cuStreamSynchronize = OneCopy.sync;
    var c: Bare.Cuda = .{ .gpa = std.testing.allocator, .engine = .{ .driver = &d }, .stream = .{ .d = &d, .handle = @ptrFromInt(331) } };
    var st: snap.Snap = .{ .buf = .{ .d = &d, .ptr = 1 << 20, .len = 1024 }, .at = 0, .history = .{ 0, 0 }, .head = false };
    var host: [words + 1024]u8 = @splat(0);
    const K = Kept(Bare);
    // the words and half the state queue a copy; the driver refuses the other half, or that range runs past the state
    for ([_]bool{ true, false }) |refused| for ([_]bool{ true, false }) |out| {
        OneCopy.queued = 0;
        OneCopy.pending = 0;
        const ranges = [_]rows.image.Range{ .{ .off = 0, .len = words + 512 }, .{ .off = words + 512, .len = if (refused) 512 else 1024 } };
        const got = if (out) K.image.gather(&c, &st, &ranges, &host) else K.image.scatter(&c, &st, &ranges, &host);
        try std.testing.expectError(if (refused) error.NotReady else error.BadKeptRange, got);
        try std.testing.expectEqual(@as(usize, 1), OneCopy.queued);
        try std.testing.expectEqual(@as(usize, 0), OneCopy.pending); // the caller's `host` may go: nothing copies on it
    };
}
