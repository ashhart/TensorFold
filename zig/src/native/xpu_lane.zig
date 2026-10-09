//! The Intel GPU engines as a lane backend: one sequence, one-row rounds, host sampling; the lane core runs it.
const std = @import("std");
const lanes = @import("lanes");

const be = lanes.backend;

/// Drawn tokens a handle can still name (the prompt's first draw, then one a round).
const ring = 1024;
pub const Error = error{ PromptTooLong, PositionMismatch, WindowTooWide, NotPipelined, NoSuchToken, NoDraftHead, ShuttingDown };

/// The top_k largest logits (all if 0, ties to the lower id) are the candidates; the keyed sampler picks one.
pub fn draw(gpa: std.mem.Allocator, logits: []const f32, s: lanes.Sampling, position: u64) !u32 {
    const k: usize = if (s.top_k == 0) logits.len else @min(logits.len, s.top_k);
    const values = try gpa.alloc(f64, k);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, k);
    defer gpa.free(ids);
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        const x: f64 = word;
        if (!std.math.isFinite(x)) return error.NonfiniteLogits;
        if (k == logits.len) {
            values[id] = x;
            ids[id] = id;
            continue;
        }
        if (filled == k and x <= values[k - 1]) continue;
        var at = @min(filled, k - 1);
        while (at > 0 and x > values[at - 1]) : (at -= 1) {
            values[at] = values[at - 1];
            ids[at] = ids[at - 1];
        }
        values[at] = x;
        ids[at] = id;
        filled = @min(filled + 1, k);
    }
    return @intCast(try lanes.sampling.choose(gpa, values, ids, position, s));
}

/// The backend over the family `S`: its Engine and promptRows (rows of one prompt window; above 16 the prefill GEMMs).
pub fn Xpu(comptime S: type) type {
    return struct {
        const Self = @This();
        gpa: std.mem.Allocator,
        e: *S.Engine,
        logits: []f32,
        drawn: [ring]u32 = undefined,
        next: u64 = 0,
        /// Set by the host when the process is asked to stop: rounds and prompt windows then end with ShuttingDown.
        halt: ?*const std.atomic.Value(bool) = null,
        /// Called once on the first prompt, after the server installed its signal handlers.
        started: ?*const fn () void = null,

        pub fn init(gpa: std.mem.Allocator, e: *S.Engine) !Self {
            return .{ .gpa = gpa, .e = e, .logits = try gpa.alloc(f32, e.vocab()) };
        }

        pub fn deinit(self: *Self) void {
            self.gpa.free(self.logits);
        }

        pub fn backend(self: *Self) be.Backend {
            return .{ .ptr = self, .vtable = &.{
                .prefill = prefillFn,
                .first = firstFn,
                .queue = queueFn,
                .read = readFn,
                .verify = verifyFn,
                .keep = keepFn,
                .draft = draftFn,
                .release = releaseFn,
            } };
        }

        /// What the round loop reads at setup: one-row windows, one stream, no head.
        pub fn facts(_: *const Self) lanes.Model {
            return .{ .exact_width = 1, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .max_streams = 1 };
        }

        fn of(ptr: *anyopaque) *Self {
            return @ptrCast(@alignCast(ptr));
        }

        fn stopped(self: *const Self) bool {
            return if (self.halt) |h| h.load(.acquire) else false;
        }

        fn take(self: *Self, token: u32) u64 {
            const h = self.next;
            self.drawn[h % ring] = token;
            self.next += 1;
            return h;
        }

        /// Feeds `token` and draws the next one at `position`: the device argmax when greedy, else the keyed host draw.
        fn step(self: *Self, token: u32, sampling: ?lanes.Sampling, position: u64) !u32 {
            try self.e.feed(token, true);
            const s = sampling orelse return self.e.argmax();
            if (s.temperature <= 0) return self.e.argmax();
            try self.e.fetchLogits(self.logits);
            return draw(self.gpa, self.logits, s, position);
        }

        fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            const self = of(ptr);
            const e = self.e;
            if (self.started) |f| {
                self.started = null;
                f();
            }
            const ids = s.prompt();
            if (ids.len == 0 or ids.len + s.max_new + 1 > e.max_len) return error.PromptTooLong;
            try e.reset();
            const head = ids[0 .. ids.len - 1];
            if (head.len > 16) {
                const window = S.promptRows(e);
                var t: usize = 0;
                while (t < head.len) : (t += window) {
                    if (s.isCancelled()) return error.Cancelled;
                    if (self.stopped()) return error.ShuttingDown;
                    try e.prefillWindows(head[t..@min(t + window, head.len)], window);
                }
            } else for (head) |tok| {
                try e.feed(tok, false);
                try e.sync();
            }
            if (s.isCancelled()) return error.Cancelled;
            _ = self.take(try self.step(ids[ids.len - 1], s.sampling, ids.len));
        }

        fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
            const self = of(ptr);
            if (position != s.prompt_len) return error.PositionMismatch;
            return self.next - 1;
        }

        fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
            return error.NotPipelined;
        }

        fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
            const self = of(ptr);
            if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
            return self.drawn[handle % ring];
        }

        /// One one-row window: the pending token is fed and the next is drawn at its keyed position.
        fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const self = of(ptr);
            if (self.stopped()) return error.ShuttingDown;
            if (windows.len != 1) return error.WindowTooWide;
            const w = windows[0];
            if (w.rows() != 1 or w.parents != null) return error.WindowTooWide;
            if (w.positions[0] != self.e.m.pos + 1) return error.PositionMismatch;
            out[0].sampled[0] = try self.step(w.pending, w.stream.sampling, w.positions[0]);
        }

        /// The row was fed in place: nothing to roll back or commit.
        fn keepFn(_: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            for (windows, paths) |_, path| if (path.len != 1) return error.WindowTooWide;
        }

        fn draftFn(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {
            return error.NoDraftHead;
        }

        fn releaseFn(ptr: *anyopaque, _: *lanes.Stream) void {
            of(ptr).e.sync() catch {};
        }
    };
}

/// A request this engine refuses or ends, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.ShuttingDown => "the server is shutting down",
        else => null,
    };
}

test "a sampled draw is keyed: same seed and position repeat, top_k 1 is the argmax, ties go to the lower id" {
    const gpa = std.testing.allocator;
    var logits: [64]f32 = undefined;
    for (&logits, 0..) |*x, i| x.* = @as(f32, @floatFromInt((i * 37) % 17)) / 4.0;
    logits[9] = 9.0;
    logits[40] = 9.0; // tie for the maximum: the lower id wins the top_k 1 draw
    const s: lanes.Sampling = .{ .seed = 77, .temperature = 0.8, .top_k = 20, .top_p = 0.95 };
    for ([_]u64{ 0, 5, 1000 }) |pos| try std.testing.expectEqual(try draw(gpa, &logits, s, pos), try draw(gpa, &logits, s, pos));
    try std.testing.expectEqual(@as(u32, 9), try draw(gpa, &logits, .{ .seed = 1, .temperature = 1.0, .top_k = 1 }, 3));
}

test "draws over many positions follow the softmax of the temperature-scaled logits" {
    const gpa = std.testing.allocator;
    const logits = [_]f32{ 2.0, 1.0, 0.0, -1.0 };
    const t = 1.0;
    var p: [4]f64 = undefined;
    var z: f64 = 0;
    for (logits, 0..) |x, i| {
        p[i] = @exp(@as(f64, x) / t);
        z += p[i];
    }
    var count: [4]u32 = @splat(0);
    const n: u32 = 40000;
    for (0..n) |pos| count[try draw(gpa, &logits, .{ .seed = 5, .temperature = t, .top_k = 0, .top_p = 1.0 }, pos)] += 1;
    for (count, p) |c, w| try std.testing.expectApproxEqAbs(w / z, @as(f64, @floatFromInt(c)) / n, 0.01);
}
