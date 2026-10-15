//! The Intel GPU engines as a lane backend: one sequence, host sampling; with an MTP head, drafted windows (16 rows).
const std = @import("std");
const lanes = @import("lanes");

const be = lanes.backend;

/// Drawn tokens a handle can still name (the prompt's first draw, then one a round).
const ring = 1024;
pub const Error = error{ PromptTooLong, PositionMismatch, WindowTooWide, NotPipelined, NoSuchToken, NoDraftHead, ShuttingDown, TreesNotBuilt, NoFirstToken, EmptyPath, NoHeldDrafts };

/// Rows of a drafted window at most (the pending token and up to 15 drafts).
const max_rows = 16;

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
        /// The family's engine has drafted rounds (Nemotron: verifyRows, keepRows, draftKept, draftFirst).
        const has_head = @hasDecl(S.Engine, "verifyRows");
        gpa: std.mem.Allocator,
        e: *S.Engine,
        logits: []f32,
        drawn: [ring]u32 = undefined,
        next: u64 = 0,
        /// Set by the host when the process is asked to stop: rounds and prompt windows then end with ShuttingDown.
        halt: ?*const std.atomic.Value(bool) = null,
        /// Called once on the first prompt, after the server installed its signal handlers.
        started: ?*const fn () void = null,
        /// The head's drafts held for the next window, the last window's rows not yet kept and where it started.
        held: [max_rows]u32 = undefined,
        held_n: usize = 0,
        pending_rows: ?u32 = null,
        start: u32 = 0,
        /// Measured at load (measure): window ms by width and one chained head step.
        costs: [max_rows]lanes.config.Cost = undefined,
        mtp_ms: f64 = 0,
        row_logits: []f32 = &.{},

        /// The engine family has drafted rounds (an MTP head) and it loaded one.
        fn drafting(self: *const Self) bool {
            return if (comptime has_head) self.e.head != null else false;
        }

        pub fn init(gpa: std.mem.Allocator, e: *S.Engine) !Self {
            return .{ .gpa = gpa, .e = e, .logits = try gpa.alloc(f32, e.vocab()) };
        }

        pub fn deinit(self: *Self) void {
            self.gpa.free(self.logits);
            if (self.row_logits.len > 0) self.gpa.free(self.row_logits);
        }

        /// Times windows of 1 to 16 rows and a head step for the depth rule (drafting engines only).
        pub fn measure(self: *Self) !void {
            if (comptime !has_head) return;
            if (!self.drafting()) return;
            var ms: [max_rows]f64 = undefined;
            self.mtp_ms = try self.e.timeRounds(&ms);
            for (ms, 1..) |v, w| self.costs[w - 1] = .{ .width = @intCast(w), .ms = v };
            self.row_logits = try self.gpa.alloc(f32, max_rows * self.e.vocab());
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

        /// What the round loop reads at setup: one stream; with a head, drafted windows of up to 16 rows.
        pub fn facts(self: *const Self) lanes.Model {
            if (!self.drafting()) return .{ .exact_width = 1, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .max_streams = 1 };
            return .{
                .exact_width = max_rows,
                .gpu_tokens = false,
                .mtp = true,
                .speculate = true,
                .speculate_early = false,
                .plain_guard = true, // a round drafts only when its depth beats the decode step on the measured costs
                .drafts = max_rows - 1,
                .window_costs = self.costs[0..max_rows],
                .mtp_step_ms = self.mtp_ms,
                .max_streams = 1,
            };
        }

        /// A one-row round on the decode step (a one-row window's bits): greedy on the device, else drawn on the host.
        fn stepOne(self: *Self, token: u32, sampling: ?lanes.Sampling, position: u64) !u32 {
            if (comptime !has_head) unreachable;
            try self.e.stepRow(token);
            const greedy = if (sampling) |s| s.temperature <= 0 else true;
            if (greedy) return self.e.argmax();
            try self.e.fetchLogits(self.logits);
            return draw(self.gpa, self.logits, sampling.?, position);
        }

        /// The window's rows drawn: greedy on the device, else each row's logits drawn at its keyed position.
        fn drawRows(self: *Self, rows: usize, sampling: ?lanes.Sampling, positions: []const u64, out: []u32) !void {
            if (comptime !has_head) unreachable;
            const greedy = if (sampling) |s| s.temperature <= 0 else true;
            if (greedy) {
                var g: [max_rows]i32 = undefined;
                try self.e.rowTokens(g[0..rows]);
                for (out[0..rows], g[0..rows]) |*o, t| o.* = @intCast(t);
                return;
            }
            const v = self.e.vocab();
            try self.e.rowLogits(self.row_logits[0 .. rows * v]);
            for (0..rows) |r| out[r] = try draw(self.gpa, self.row_logits[r * v ..][0..v], sampling.?, positions[r]);
        }

        /// Commit the last window's kept rows if a keep has not yet.
        fn settle(self: *Self, kept: ?usize) !void {
            if (comptime !has_head) return;
            const rows = self.pending_rows orelse return;
            self.pending_rows = null;
            try self.e.keepRows(@intCast(kept orelse rows));
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
            if (comptime has_head) if (self.drafting()) {
                // the whole prompt in prompt windows; the head absorbs every row but the last (draftFirst takes it)
                self.held_n = 0;
                self.pending_rows = null;
                try e.prefillHead(ids, S.promptRows(e));
                if (s.isCancelled()) return error.Cancelled;
                var first: [1]u32 = undefined;
                try self.drawRows(1, s.sampling, &.{ids.len}, &first);
                _ = self.take(first[0]);
                return;
            };
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

        /// One window: the pending token, then the held drafts or the host's; each row drawn at its keyed position.
        fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const self = of(ptr);
            if (self.stopped()) return error.ShuttingDown;
            if (comptime has_head) if (self.drafting()) return self.verifyWindow(windows, out);
            if (windows.len != 1) return error.WindowTooWide;
            const w = windows[0];
            if (w.rows() != 1 or w.parents != null) return error.WindowTooWide;
            if (w.positions[0] != self.e.m.pos + 1) return error.PositionMismatch;
            out[0].sampled[0] = try self.step(w.pending, w.stream.sampling, w.positions[0]);
        }

        fn verifyWindow(self: *Self, windows: []const be.Window, out: []be.Verified) !void {
            if (comptime !has_head) unreachable;
            if (windows.len != 1) return error.WindowTooWide;
            const w = windows[0];
            if (w.parents != null) return error.TreesNotBuilt;
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            try self.settle(null);
            const e = self.e;
            for (w.positions, 0..) |p, r| if (p != e.m.pos + 1 + r) return error.PositionMismatch;
            var ids: [max_rows]u32 = undefined;
            ids[0] = w.pending;
            if (w.held > 0) {
                if (w.held > self.held_n) return error.NoHeldDrafts;
                @memcpy(ids[1..][0..w.held], self.held[0..w.held]);
            } else @memcpy(ids[1..][0..w.tokens.len], w.tokens);
            self.start = e.m.pos;
            self.held_n = 0;
            if (rows == 1) { // no drafts: the decode step, kept in place
                out[0].sampled[0] = try self.stepOne(w.pending, w.stream.sampling, w.positions[0]);
                return;
            }
            try e.verifyRows(ids[0..rows]);
            try self.drawRows(rows, w.stream.sampling, w.positions, out[0].sampled);
            @memcpy(out[0].drafts, ids[1..rows]);
            self.held_n = 0;
            self.pending_rows = @intCast(rows);
        }

        /// One-row rounds were fed in place; drafted windows keep their path's rows (a chain prefix).
        fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            const self = of(ptr);
            for (windows, paths) |_, path| {
                if (!self.drafting()) {
                    if (path.len != 1) return error.WindowTooWide;
                    continue;
                }
                if (path.len == 0) return error.EmptyPath;
                for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
                try self.settle(path.len);
            }
        }

        /// The head absorbs the kept rows (or the prompt's last row with the first token), then holds `depth` drafts.
        fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
            const self = of(ptr);
            if (comptime !has_head) return error.NoDraftHead;
            if (!self.drafting()) return error.NoDraftHead;
            for (requests) |r| {
                try self.settle(if (r.rows) |path| path.len else null);
                if (r.position != self.e.m.pos + 1) return error.PositionMismatch;
                const depth = @min(r.depth, max_rows - 1);
                if (r.rows) |path| {
                    for (path, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                    self.held_n = try self.e.draftKept(r.follow[0..path.len], self.start, depth, &self.held);
                } else {
                    const first = r.first orelse return error.NoFirstToken;
                    const t = switch (first) {
                        .handle => |h| self.drawn[h % ring],
                        .value => |v| v,
                    };
                    self.held_n = try self.e.draftFirst(t, depth, &self.held);
                }
            }
        }

        fn releaseFn(ptr: *anyopaque, _: *lanes.Stream) void {
            const self = of(ptr);
            self.e.sync() catch {};
            self.held_n = 0;
            self.pending_rows = null;
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
