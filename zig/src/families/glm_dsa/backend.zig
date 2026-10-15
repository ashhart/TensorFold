//! GLM-5.3 behind the lane core on a pair: a stream a slot, every operation sent to rank 1 before rank 0 runs it.
const std = @import("std");
const lanes = @import("lanes");
const eng = @import("engine.zig");
const fl = @import("follow.zig");
const snap = @import("snap.zig");
const be = lanes.backend;
const Stream = lanes.Stream;
const gpu = @import("gpu.zig");
const R = gpu.max_rows;

const RING = 1024;

pub const Backend = struct {
    gpa: std.mem.Allocator,
    e: *eng.Engine,
    by: std.AutoHashMapUnmanaged(*Stream, u32) = .empty,
    /// Drawn tokens by handle (every read waits for its forward anyway: a pair runs no rounds ahead).
    ring: [RING]u32 = undefined,
    puts: u64 = 0,
    first: []u32,
    wins: std.ArrayList(eng.Win) = .empty,
    reqs: std.ArrayList(eng.Draft) = .empty,
    toks: std.ArrayList(u32) = .empty,
    /// Window ms by rows and one chained head step, timed at load (both Macs in step).
    costs: [R]lanes.config.Cost = undefined,
    head_ms: f64 = 0,
    timed: bool = false,
    next_snap: u32 = 1,

    pub fn init(gpa: std.mem.Allocator, e: *eng.Engine) !Backend {
        return .{ .gpa = gpa, .e = e, .first = try gpa.alloc(u32, e.slots.len) };
    }

    pub fn deinit(b: *Backend) void {
        b.by.deinit(b.gpa);
        b.wins.deinit(b.gpa);
        b.reqs.deinit(b.gpa);
        b.toks.deinit(b.gpa);
        b.gpa.free(b.first);
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefill, .first = firstFn, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true } };
    }

    /// Windows of up to 16 rows (each row's bits whatever shares its forward), MTP chains, streams in one forward.
    pub fn facts(b: *const Backend) lanes.Model {
        return .{
            .exact_width = R,
            .first_copy_rows = 4,
            .mtp = b.e.hasMtp(),
            .speculate = b.e.hasMtp(),
            .speculate_early = false,
            .gpu_tokens = false,
            .plain_guard = true,
            .drafts = eng.max_depth,
            .window_costs = if (b.timed) &b.costs else &.{},
            .mtp_step_ms = b.head_ms,
            .streams_exact = true,
            .hidden_rows = true,
            .batch_rows = R,
            .max_streams = @intCast(b.e.slots.len),
            .draft_streams = true,
        };
    }

    fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }

    fn slotOf(b: *Backend, s: *Stream) !u32 {
        return b.by.get(s) orelse error.UnknownStream;
    }

    fn put(b: *Backend, v: u32) u64 {
        const h = b.puts;
        b.ring[h % RING] = v;
        b.puts += 1;
        return h;
    }

    fn free(b: *const Backend) ?u32 {
        for (b.e.slots, 0..) |s, i| if (!s.used) return @intCast(i);
        return null;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const b = self(ptr);
        const e = b.e;
        if (e.rank() != 0) return error.FollowsPeer;
        if (s.sampling) |p| if (p.temperature > 0) return error.GreedyOnly;
        const prompt = s.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len + s.max_new + eng.max_depth + R + 1 > e.context) return error.ContextFull;
        const i = b.free() orelse return error.NoFreeSlot;
        try b.by.put(b.gpa, s, i);
        try fl.begin(e, i, prompt);
        s.cached = 0;
        var at: u32 = 0;
        if (s.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
            const k: *snap.Snap = @ptrCast(@alignCast(saved));
            if (k.at < prompt.len) {
                try fl.restore(e, i, k.id);
                at = k.at;
                s.cached = at;
            } else s.reuse_failed = true;
        }
        var next: usize = 0;
        var mark: usize = 0;
        while (at < prompt.len) {
            if (s.isCancelled()) return error.Cancelled;
            while (next < s.chunks.len and s.chunks[next] <= at) next += 1;
            while (mark < s.reuse.marks.len and s.reuse.marks[mark] <= at) mark += 1;
            var end: u32 = if (next < s.chunks.len) s.chunks[next] else @intCast(prompt.len);
            if (mark < s.reuse.marks.len) end = @min(end, s.reuse.marks[mark]); // a mark ends a chunk: its state is kept there
            const n = @min(end - at, gpu.prompt_rows);
            if (try fl.chunk(e, i, at, n, false)) |t| b.first[i] = t;
            at += n;
            s.fill_at.store(at, .monotonic); // fl.chunk waits on rank 0: `at` is landed progress
            if (std.mem.indexOfScalar(u32, s.reuse.marks, at) != null) if (s.reuse.hook) |k| k.at(k.ptr, s, at);
        }
    }

    /// The prompt cache's functions: a stream's state at a mark, kept on both Macs by id.
    pub fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        return snap.bytes(self(ptr).e, at);
    }

    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const b = self(ptr);
        const i = try b.slotOf(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        const id = b.next_snap;
        b.next_snap +%= 1;
        return try fl.save(b.e, id, i, at);
    }

    pub fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }

    pub fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
        const k: *snap.Snap = @ptrCast(@alignCast(saved));
        fl.drop(self(ptr).e, k.id);
    }

    fn firstFn(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const i = try b.slotOf(s);
        if (position != b.e.cacheLen(i)) return error.PositionMismatch;
        return b.put(b.first[i]);
    }

    fn queue(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.Unsupported; // a pair's ranks run windows in step
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const b = self(ptr);
        if (handle >= b.puts or b.puts - handle > RING) return error.StaleHandle;
        return b.ring[handle % RING];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        const e = b.e;
        b.wins.clearRetainingCapacity();
        b.toks.clearRetainingCapacity();
        var total: usize = 0;
        for (windows) |w| {
            if (w.parents != null or w.early or w.masks.len > 0) return error.TreeWindowsUnsupported;
            const i = try b.slotOf(w.stream);
            const held = e.held(i);
            if (w.held > held.len) return error.WindowOutOfStep;
            for (w.positions, 0..) |p, r| if (p != e.cacheLen(i) + r + 1) return error.PositionMismatch;
            total += w.rows();
        }
        if (total > R) return error.WindowOutOfStep;
        try b.toks.ensureTotalCapacity(b.gpa, total);
        for (windows) |w| {
            const i = b.by.get(w.stream).?;
            const at = b.toks.items.len;
            b.toks.appendAssumeCapacity(w.pending);
            b.toks.appendSliceAssumeCapacity(e.held(i)[0..w.held]);
            b.toks.appendSliceAssumeCapacity(w.tokens);
            try b.wins.append(b.gpa, .{ .slot = i, .tokens = b.toks.items[at..] });
        }
        var picks: [R]u32 = undefined;
        try fl.window(e, b.wins.items, &picks);
        var row: usize = 0;
        for (windows, out, b.wins.items) |w, o, win| {
            @memcpy(o.sampled, picks[row..][0..w.rows()]);
            @memcpy(o.drafts, win.tokens[1..]);
            row += w.rows();
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        for (windows, paths) |w, path| {
            for (path, 0..) |r, k| if (r != k) return error.TreeWindowsUnsupported;
            try fl.keep(b.e, try b.slotOf(w.stream), @intCast(path.len));
        }
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(ptr);
        if (requests.len > R) return error.DraftOutOfStep;
        var ones: [R]u32 = undefined;
        b.reqs.clearRetainingCapacity();
        for (requests, 0..) |r, k| {
            if (r.early or r.lanes != null or r.ranks) return error.TreeDraftsUnsupported;
            if (r.rows) |path| for (path, 0..) |x, j| if (x != j) return error.TreeDraftsUnsupported;
            const follow: []const u32 = if (r.first) |f| blk: {
                ones[k] = switch (f) {
                    .handle => |h| try read(b, h),
                    .value => |v| v,
                };
                break :blk ones[k .. k + 1];
            } else r.follow;
            try b.reqs.append(b.gpa, .{ .slot = try b.slotOf(r.stream), .prompt = r.rows == null, .follow = follow, .depth = @min(r.depth, eng.max_depth) });
        }
        try fl.draft(b.e, b.reqs.items);
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const b = self(ptr);
        const kv = b.by.fetchRemove(s) orelse return;
        fl.release(b.e, kv.value);
    }

    /// Rank 0 at load: windows of 1-16 rows and head chains timed on a scratch stream (rank 1 runs them in step).
    pub fn time(b: *Backend, io: std.Io) !void {
        const e = b.e;
        var prompt: [64]u32 = undefined;
        for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + 7 * i);
        try fl.begin(e, 0, &prompt);
        var at: u32 = 0;
        var first: u32 = 0;
        while (at < prompt.len) : (at += R) first = (try fl.chunk(e, 0, at, R, false)) orelse first;
        var toks: [R]u32 = @splat(first);
        var picks: [R]u32 = undefined;
        for (0..R) |w| {
            var best: f64 = std.math.inf(f64);
            for (0..3) |_| {
                const t0 = std.Io.Timestamp.now(io, .awake);
                try fl.window(e, &.{.{ .slot = 0, .tokens = toks[0 .. w + 1] }}, &picks);
                best = @min(best, ms(io, t0));
                try fl.keep(e, 0, 1);
            }
            b.costs[w] = .{ .width = @intCast(w + 1), .ms = best };
        }
        if (e.hasMtp()) {
            var one: f64 = std.math.inf(f64);
            var four: f64 = std.math.inf(f64);
            for (0..3) |_| {
                var t0 = std.Io.Timestamp.now(io, .awake);
                try fl.draft(e, &.{.{ .slot = 0, .prompt = false, .follow = toks[0..1], .depth = 1 }});
                one = @min(one, ms(io, t0));
                try fl.window(e, &.{.{ .slot = 0, .tokens = toks[0..1] }}, &picks);
                try fl.keep(e, 0, 1);
                t0 = std.Io.Timestamp.now(io, .awake);
                try fl.draft(e, &.{.{ .slot = 0, .prompt = false, .follow = toks[0..1], .depth = eng.max_depth }});
                four = @min(four, ms(io, t0));
                try fl.window(e, &.{.{ .slot = 0, .tokens = toks[0..1] }}, &picks);
                try fl.keep(e, 0, 1);
            }
            b.head_ms = @max(0.1, (four - one) / @as(f64, eng.max_depth - 1));
        }
        fl.release(e, 0);
        b.timed = true;
        _ = &toks;
    }

    fn ms(io: std.Io, t0: std.Io.Timestamp) f64 {
        return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e6;
    }
};
