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

        fn of(ptr: *anyopaque) *F.Cuda {
            return @ptrCast(@alignCast(ptr));
        }

        fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
            try partsOf(at, of(ptr).head != null, out, a);
        }

        fn copy(c: *F.Cuda, st: *snap.Snap, ranges: []const rows.image.Range, host: []u8, out: bool) !void {
            const d = c.engine.driver;
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
            const marks = [_]u32{4136};
            var s = try lanes.Stream.init(a, .{ .id = "probe", .prompt = &toks, .max_new = 1, .drafts = c.head != null, .reuse = .{ .marks = &marks, .hook = .{ .ptr = &hook, .at = Hook.at } } });
            defer s.deinit(a);
            const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
            backend.vtable.prefill(backend.ptr, &s) catch |e| {
                backend.vtable.release(backend.ptr, &s);
                return e;
            };
            try c.stream.synchronize();
            const ns = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
            backend.vtable.release(backend.ptr, &s);
            const saved = hook.saved orelse return hook.failed orelse error.ProbeUnmarked;
            defer F.snap_drop(c, saved);
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
