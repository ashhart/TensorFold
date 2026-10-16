//! What every HIP family's lane backend answers alike: its streams, the drawn tokens, the first draws, the held drafts.

const std = @import("std");
const platform = @import("sys");
const hip = @import("hip");
const lanes = @import("lanes");
const draw = @import("draw.zig");

const be = lanes.backend;

/// Drawn tokens a handle names, newest last.
pub const ring = 1024;

/// Where a prompt pass stands: the row it has reached, and the cuts it still keeps a state at.
pub const Fill = struct {
    at: usize,
    stops: [16]u32 = undefined,
    count: usize = 0,
    next: usize = 0,

    /// A pass resumed at `from` with the cache's `marks` as its cuts (the first 16).
    pub fn init(from: usize, marks: []const u32) Fill {
        var f: Fill = .{ .at = from, .count = @min(marks.len, 16) };
        @memcpy(f.stops[0..f.count], marks[0..f.count]);
        return f;
    }

    /// The row the next chunk may reach at most: the next cut, or the prompt's end.
    pub fn limit(f: *const Fill, prompt_len: usize) usize {
        return if (f.next < f.count) f.stops[f.next] else prompt_len;
    }

    /// Whether `to` is the next cut, which is then passed.
    pub fn cut(f: *Fill, to: usize) bool {
        if (f.next < f.count and to == f.stops[f.next]) {
            f.next += 1;
            return true;
        }
        return false;
    }
};

/// The last verify's rows until their keep: how many, and the slot a shared round keeps them from.
pub const Pending = struct { rows: usize, slot: usize = 0 };

/// The tokens drawn so far, each named by its handle.
pub const Drawn = struct {
    tokens: [ring]u32 = undefined,
    next: u64 = 0,

    pub fn take(d: *Drawn, token: u32) u64 {
        const at = d.next;
        d.tokens[at % ring] = token;
        d.next += 1;
        return at;
    }

    pub fn read(d: *const Drawn, handle: u64) !u32 {
        if (handle >= d.next or d.next - handle > ring) return error.NoSuchToken;
        return d.tokens[handle % ring];
    }

    /// The last token's handle.
    pub fn last(d: *const Drawn) u64 {
        return d.next - 1;
    }
};

pub fn nowNs() u64 {
    var ts: platform.c.timespec = undefined;
    _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// The context this thread's HIP calls go to.
threadlocal var bound: ?*const hip.Context = null;

/// The calling thread bound to the engine's GPU (the lane thread did not open it).
pub fn bind(ctx: *const hip.Context) void {
    if (bound == ctx) return;
    ctx.makeCurrent() catch |err| std.log.err("hipSetDevice on the lane thread: {s}", .{@errorName(err)});
    bound = ctx;
}

pub fn sampling(s: *const lanes.Stream) ?lanes.Sampling {
    return s.sampling;
}

/// The streams a backend serves, each behind its lane, with the ids the other ranks know them by.
pub fn Streams(comptime Lane: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        map: std.AutoHashMapUnmanaged(*const lanes.Stream, *Lane) = .empty,
        ids: std.AutoHashMapUnmanaged(*const lanes.Stream, u32) = .empty,
        next_id: u32 = 0,
        drawn: Drawn = .{},
        /// Nanoseconds and calls the backend's verifies, keeps and drafts took; the core's own work is the rest.
        spent: [3]u64 = @splat(0),
        calls: [3]u64 = @splat(0),

        pub fn init(gpa: std.mem.Allocator) Self {
            return .{ .gpa = gpa };
        }

        /// The timings logged, the maps freed (every lane freed by the family before).
        pub fn deinit(st: *Self) void {
            const names = [_][]const u8{ "verify", "keep", "draft" };
            for (names, st.spent, st.calls) |name, ns, n| if (n > 0) std.log.info(
                "backend {s}: {d} calls, {d:.0} us each",
                .{ name, n, @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(n)) / 1e3 },
            );
            st.ids.deinit(st.gpa);
            st.map.deinit(st.gpa);
        }

        pub fn get(st: *const Self, s: *const lanes.Stream) !*Lane {
            return st.map.get(s) orelse error.UnknownStream;
        }

        /// A stream's id on every rank.
        pub fn idOf(st: *Self, s: *const lanes.Stream) !u32 {
            const gop = try st.ids.getOrPut(st.gpa, s);
            if (!gop.found_existing) {
                gop.value_ptr.* = st.next_id;
                st.next_id += 1;
            }
            return gop.value_ptr.*;
        }

        /// The stream's lane out of the map (the family frees it), null when it has none.
        pub fn forget(st: *Self, s: *const lanes.Stream) ?*Lane {
            const kv = st.map.fetchRemove(s) orelse return null;
            _ = st.ids.remove(s);
            return kv.value;
        }

        fn timed(st: *Self, which: usize, began: u64) void {
            st.spent[which] += nowNs() - began;
            st.calls[which] += 1;
        }
    };
}

/// The window's tokens into `out`: the pending token, then the drafts `held` or sent; `len` the stream's kept slots.
pub fn windowTokens(w: be.Window, held: []const u32, len: usize, out: []u32) !usize {
    if (w.parents != null) return error.TreesNotBuilt;
    if (w.held > held.len or (w.held > 0 and w.tokens.len > 0)) return error.NoHeldDrafts;
    const n = w.rows();
    if (n > out.len) return error.WindowTooWide;
    for (w.positions, 0..) |p, r| if (p != len + 1 + r) {
        std.log.err("row {d} keyed at {d}, the stream holds {d} slots", .{ r, p, len });
        return error.PositionMismatch;
    };
    out[0] = w.pending;
    if (w.held > 0) @memcpy(out[1..n], held[0..w.held]) else @memcpy(out[1..n], w.tokens);
    return n;
}

/// Each row's draw of the window: its stream's sampling, its position, and its allowed bits (rows x words).
pub fn requests(w: be.Window, out: []draw.Request) void {
    const words = if (w.masks.len > 0) w.masks.len / w.positions.len else 0;
    for (w.positions, 0..) |p, r| out[r] = .{
        .sampling = sampling(w.stream),
        .position = p,
        .mask = w.masks[r * words ..][0..words],
    };
}

/// The prompt's first token a draft request chains from, by handle or value.
pub fn firstToken(drawn: *const Drawn, r: be.DraftRequest) !u32 {
    return switch (r.first orelse return error.NoFirstToken) {
        .handle => |at| drawn.tokens[at % ring],
        .value => |v| v,
    };
}

/// A verify the round loop did not trim keeps every row (the core keeps only on drops).
pub fn settle(h: anytype, lane: anytype) !void {
    if (lane.pending) |p| try h.keepRows(lane, p.rows);
}

/// Before a draft: the rows the request names kept, or every pending row.
pub fn keepFor(h: anytype, lane: anytype, r: be.DraftRequest) !void {
    if (lane.pending == null) return;
    if (r.rows) |kept| try h.keepRows(lane, kept.len) else try settle(h, lane);
}

/// The lane core's backend over family `F`: its `streams`, `tp`, `e.ctx` and `F.Lane`, and the hooks `vtable` calls.
pub fn Backend(comptime F: type) type {
    return struct {
        pub fn backend(h: *F) be.Backend {
            return .{ .ptr = h, .vtable = &vtable };
        }

        const vtable: be.Backend.VTable = .{
            .prefill = prefillFn,
            .prefill_step = prefillStepFn,
            .first = firstFn,
            .first_masked = firstMaskedFn,
            .held = heldTokensFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .tree = heldFn,
            .release = releaseFn,
        };

        /// The backend behind `ptr`, the calling thread bound to its GPU.
        fn of(ptr: *anyopaque) *F {
            const h: *F = @ptrCast(@alignCast(ptr));
            bind(&h.e.ctx);
            return h;
        }

        /// The whole prompt pass, chunk by chunk (a driver of its own, with no rounds between).
        fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            while (!try prefillStepFn(ptr, s)) {}
        }

        /// One chunk of the prompt pass, ending on the next cut or the prompt's end; the last draws the first token.
        fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
            const h = of(ptr);
            if (s.isCancelled()) return error.Cancelled;
            const lane = if (h.streams.map.get(s)) |l| (if (l.fill != null) l else try h.begin(s)) else try h.begin(s);
            const prompt = s.prompt();
            const f = &lane.fill.?;
            const to = try h.reach(lane, f.at, f.limit(prompt.len));
            try h.tp.fill(lane.id, to);
            if (to == prompt.len) {
                // a structured reply keeps the prompt's last row: its first token is drawn again under the grammar
                if (s.grammar != null and lane.first_row == null) lane.first_row = try h.gpa.alloc(
                    u16,
                    h.drawer().vocab,
                );
                const keep = if (s.grammar != null) lane.first_row else null;
                _ = h.streams.drawn.take(try h.finish(
                    lane,
                    prompt,
                    f.at,
                    .{ .sampling = sampling(s), .position = prompt.len, .keep = keep },
                ));
                lane.fill = null;
                return true;
            }
            try h.advance(lane, prompt, f.at, to);
            f.at = to;
            if (f.cut(to)) h.keepCut(lane, prompt, to) catch |err| std.log.warn(
                "prefix not kept: {s}",
                .{@errorName(err)},
            );
            return false;
        }

        fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
            const h = of(ptr);
            if (position != s.prompt_len) {
                std.log.err("first draw at {d}, the prompt has {d} tokens", .{ position, s.prompt_len });
                return error.PositionMismatch;
            }
            return h.streams.drawn.last();
        }

        /// The first token among the grammar's allowed ones, from the prompt row the pass kept.
        fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, bits: []const u32) anyerror!u64 {
            const h = of(ptr);
            if (position != s.prompt_len) return error.PositionMismatch;
            const lane = try h.streams.get(s);
            const kept = lane.first_row orelse return error.StructuresUnsupported;
            const row = try h.gpa.dupe(u16, kept);
            defer h.gpa.free(row);
            return h.streams.drawn.take(try h.drawer().drawHost(
                row,
                .{ .sampling = sampling(s), .position = position, .mask = bits },
            ));
        }

        /// The drafts the head holds for the stream, as host tokens (a grammar walks them).
        fn heldTokensFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
            const lane = try of(ptr).streams.get(s);
            if (out.len > lane.held_n) return error.NoHeldDrafts;
            @memcpy(out, lane.held[0..out.len]);
        }

        fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
            return error.NotPipelined;
        }

        fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
            return of(ptr).streams.drawn.read(handle);
        }

        fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const h = of(ptr);
            const began = nowNs();
            defer h.streams.timed(0, began);
            try h.verify(windows, out);
        }

        /// Keep each stream's accepted prefix: its length, and the state after its last kept row.
        fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            const h = of(ptr);
            const began = nowNs();
            defer h.streams.timed(1, began);
            for (windows, paths) |w, path| {
                const lane = try h.streams.get(w.stream);
                for (path, 0..) |r, j| if (r != j) return error.TreesNotBuilt;
                // the draft request of the same round may have kept these rows already
                if (lane.pending == null) continue;
                try h.keepRows(lane, path.len);
            }
            try h.flush();
        }

        fn draftFn(ptr: *anyopaque, reqs: []const be.DraftRequest) anyerror!void {
            const h = of(ptr);
            const began = nowNs();
            defer h.streams.timed(2, began);
            try h.draft(reqs);
        }

        /// The drafts the head holds for the stream after its last chain: fewer than asked when a draft cut it.
        fn heldFn(ptr: *anyopaque, s: *lanes.Stream, _: std.mem.Allocator) anyerror!?lanes.stream.Held {
            const lane = try of(ptr).streams.get(s);
            return .{ .count = @intCast(lane.held_n) };
        }

        fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
            const h = of(ptr);
            h.release(h.streams.forget(s) orelse return);
        }
    };
}

test "a prompt pass stops at its cuts, then the prompt's end" {
    var f = Fill.init(64, &.{ 128, 256 });
    try std.testing.expectEqual(@as(usize, 128), f.limit(1000));
    try std.testing.expect(!f.cut(100));
    try std.testing.expect(f.cut(128));
    try std.testing.expectEqual(@as(usize, 256), f.limit(1000));
    try std.testing.expect(f.cut(256));
    try std.testing.expectEqual(@as(usize, 1000), f.limit(1000));
}

test "drawn tokens are read back by handle while the ring holds them" {
    var d: Drawn = .{};
    try std.testing.expectEqual(@as(u64, 0), d.take(7));
    try std.testing.expectEqual(@as(u64, 1), d.take(9));
    try std.testing.expectEqual(@as(u32, 9), try d.read(d.last()));
    try std.testing.expectEqual(@as(u32, 7), try d.read(0));
    try std.testing.expectError(error.NoSuchToken, d.read(2));
    for (0..ring) |_| _ = d.take(1);
    try std.testing.expectError(error.NoSuchToken, d.read(0));
}
