//! Host test fixture modeling Inkling 2 Mini Metal target verification, DSpark drafter, and cache reuse.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const backend = @import("../families/inkling2/metal/backend.zig");
const st = @import("../families/inkling2/metal/state.zig");

pub const gpa = std.testing.allocator;
pub const io = std.testing.io;
pub const vocab_size: usize = 64;

pub const PauseGate = struct {
    arm_after_verify: std.atomic.Value(usize),
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
};

pub const FakeMetalTarget = struct {
    const StreamState = struct {
        committed: std.ArrayList(u32) = .empty,
        base_len: usize = 0,
        last_first_row: lanes.LogprobRow = undefined,
        has_first_row: bool = false,
    };

    const Snapshot = struct {
        tokens: []u32,
    };

    alloc: std.mem.Allocator,
    drafts: bool,
    streams_cap: u32,
    mismatch_at: ?usize = null,
    force_miss: bool = false,
    window_costs: []const lanes.config.Cost = &.{},
    shared_costs: []const lanes.config.Cost = &.{},
    streams: std.AutoHashMap(*lanes.Stream, StreamState),
    handles: std.ArrayList(u32) = .empty,
    dummy_features: [256]f32 = @splat(0),
    verify_calls: std.atomic.Value(usize) = .init(0),
    verify_rows: std.atomic.Value(usize) = .init(0),
    last_rows: std.atomic.Value(usize) = .init(1),
    keep_calls: std.atomic.Value(usize) = .init(0),
    rollbacks: std.atomic.Value(usize) = .init(0),
    released: std.atomic.Value(usize) = .init(0),
    gate: ?*PauseGate = null,

    pub fn init(alloc: std.mem.Allocator, drafts: bool, streams_cap: u32) FakeMetalTarget {
        return .{
            .alloc = alloc,
            .drafts = drafts,
            .streams_cap = streams_cap,
            .streams = std.AutoHashMap(*lanes.Stream, StreamState).init(alloc),
        };
    }

    pub fn deinit(self: *FakeMetalTarget) void {
        var it = self.streams.valueIterator();
        while (it.next()) |s| s.committed.deinit(self.alloc);
        self.streams.deinit();
        self.handles.deinit(self.alloc);
    }

    pub fn facts(self: *const FakeMetalTarget) lanes.Model {
        var m: backend.Metal = .{ .gpa = self.alloc, .model = undefined, .options = .{ .capacity = 256, .streams = self.streams_cap, .drafts = self.drafts, .budget = 0 }, .scratch = undefined, .prompt = undefined, .tokens = &.{}, .values = &.{} };
        var f = m.facts();
        f.window_costs = self.window_costs;
        f.shared_costs = self.shared_costs;
        return f;
    }

    pub fn asBackend(self: *FakeMetalTarget) lanes.backend.Backend {
        return .{
            .keep_full_lone = true,
            .ptr = self,
            .vtable = &.{
                .prefill = prefillFn,
                .first = firstFn,
                .queue = queueFn,
                .read = readFn,
                .verify = verifyFn,
                .keep = keepFn,
                .draft = draftFn,
                .prepare_features = prepareFeaturesFn,
                .features = featuresFn,
                .first_row = firstRowFn,
                .release = releaseFn,
            },
        };
    }

    pub fn snapshots(self: *FakeMetalTarget) api.prompt_cache.Snapshots {
        return .{
            .ptr = self,
            .vtable = &.{
                .bytes = snapBytesFn,
                .save = snapSaveFn,
                .restore = snapRestoreFn,
                .drop = snapDropFn,
            },
        };
    }

    fn computeLogits(self: *const FakeMetalTarget, prefix: []const u32, out_logits: *[vocab_size]f64) void {
        var h: u64 = 0x9e3779b97f4a7c15;
        const tail_start = if (prefix.len > 6) prefix.len - 6 else 0;
        for (prefix[tail_start..]) |tok| {
            h = (h ^ tok) *% 0xbf58476d1ce4e5b9;
            h ^= h >> 27;
        }
        for (0..vocab_size) |v| {
            const mixed = h +% @as(u64, v) *% 0x94d049bb133111eb;
            const bucket: i64 = @intCast(mixed % 40);
            out_logits[v] = @as(f64, @floatFromInt(bucket - 20)) * 0.1;
        }
        if (prefix.len >= 5) {
            const pat = prefix[prefix.len - 4 ..];
            var idx: usize = prefix.len - 4;
            while (idx > 0) {
                idx -= 1;
                if (std.mem.eql(u32, prefix[idx .. idx + 4], pat)) {
                    const next_tok = prefix[idx + 4] % vocab_size;
                    const div = self.force_miss or (if (self.mismatch_at) |m| prefix.len == m else false);
                    const preferred = if (div) (next_tok + 7) % vocab_size else next_tok;
                    out_logits[preferred] += 6.5;
                    break;
                }
            }
        }
    }

    fn drawToken(self: *const FakeMetalTarget, prefix: []const u32, pos: u64, sampling: ?lanes.Sampling, out_row: ?*lanes.LogprobRow) !u32 {
        var logits: [vocab_size]f64 = undefined;
        self.computeLogits(prefix, &logits);
        var ids: [vocab_size]u64 = undefined;
        for (0..vocab_size) |i| ids[i] = i;
        const s = sampling orelse lanes.Sampling{ .seed = 0, .temperature = 0 };
        const chosen: u32 = @intCast(try lanes.sampling.choose(self.alloc, &logits, &ids, pos, s));
        if (out_row) |row| {
            var max_l: f64 = logits[0];
            for (logits[1..]) |l| if (l > max_l) {
                max_l = l;
            };
            var sum_exp: f64 = 0;
            for (logits) |l| sum_exp += @exp(l - max_l);
            const lse = max_l + @log(sum_exp);
            row.* = .{ .token = chosen, .logprob = @floatCast(logits[chosen] - lse), .count = 2 };
            var best_i: usize = 0;
            var second_i: usize = 1;
            if (logits[1] > logits[0]) {
                best_i = 1;
                second_i = 0;
            }
            for (2..vocab_size) |v| {
                if (logits[v] > logits[best_i]) {
                    second_i = best_i;
                    best_i = v;
                } else if (logits[v] > logits[second_i]) {
                    second_i = v;
                }
            }
            row.ids[0] = @intCast(best_i);
            row.ids[1] = @intCast(second_i);
            row.logprobs[0] = @floatCast(logits[best_i] - lse);
            row.logprobs[1] = @floatCast(logits[second_i] - lse);
        }
        return chosen;
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self: *FakeMetalTarget = @ptrCast(@alignCast(ptr));
        var entry = try self.streams.getOrPut(s);
        if (entry.found_existing) entry.value_ptr.committed.clearRetainingCapacity() else entry.value_ptr.* = .{};
        const st_ptr = entry.value_ptr;
        const prompt = s.context.items[0..s.prompt_len];
        var at: usize = 0;
        if (s.reuse.saved) |saved_ptr| {
            const snap: *const Snapshot = @ptrCast(@alignCast(saved_ptr));
            try st_ptr.committed.appendSlice(self.alloc, snap.tokens[0..s.reuse.at]);
            at = s.reuse.at;
            s.cached = s.reuse.at;
        }
        var mark_idx: usize = 0;
        while (mark_idx < s.reuse.marks.len and s.reuse.marks[mark_idx] <= at) : (mark_idx += 1) {}
        while (at < prompt.len) {
            const next_stop = if (mark_idx < s.reuse.marks.len) @min(@as(usize, s.reuse.marks[mark_idx]), prompt.len) else prompt.len;
            try st_ptr.committed.appendSlice(self.alloc, prompt[at..next_stop]);
            at = next_stop;
            if (mark_idx < s.reuse.marks.len and at == s.reuse.marks[mark_idx]) {
                if (s.reuse.hook) |hook| hook.at(hook.ptr, s, @intCast(at));
                mark_idx += 1;
            }
        }
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self: *FakeMetalTarget = @ptrCast(@alignCast(ptr));
        const st_ptr = self.streams.getPtr(s) orelse return error.MissingStream;
        var row: lanes.LogprobRow = undefined;
        const tok = try self.drawToken(st_ptr.committed.items, position, s.sampling, &row);
        st_ptr.last_first_row = row;
        st_ptr.has_first_row = true;
        const h: u64 = self.handles.items.len;
        try self.handles.append(self.alloc, tok);
        return h;
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, position: u64) anyerror!u64 {
        const self: *FakeMetalTarget = @ptrCast(@alignCast(ptr));
        const st_ptr = self.streams.getPtr(s) orelse return error.MissingStream;
        const tok_in: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| self.handles.items[h],
        };
        try st_ptr.committed.append(self.alloc, tok_in);
        const tok = try self.drawToken(st_ptr.committed.items, position, s.sampling, null);
        const h: u64 = self.handles.items.len;
        try self.handles.append(self.alloc, tok);
        return h;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self: *FakeMetalTarget = @ptrCast(@alignCast(ptr));
        return self.handles.items[handle];
    }

    fn verifyFn(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
        const self: *FakeMetalTarget = @ptrCast(@alignCast(ptr));
        const call_num = self.verify_calls.fetchAdd(1, .acq_rel) + 1;
        var round_rows: usize = 0;
        for (windows, out) |w, *res| {
            std.debug.assert(w.parents == null);
            std.debug.assert(w.held == 0);
            const st_ptr = self.streams.getPtr(w.stream) orelse return error.MissingStream;
            st_ptr.base_len = st_ptr.committed.items.len;
            round_rows += w.rows();
            _ = self.verify_rows.fetchAdd(w.rows(), .acq_rel);
            try st_ptr.committed.append(self.alloc, w.pending);
            var row_buf: lanes.LogprobRow = undefined;
            res.sampled[0] = try self.drawToken(st_ptr.committed.items, w.positions[0], w.stream.sampling, &row_buf);
            if (res.rows.len > 0) res.rows[0] = row_buf;
            for (w.tokens, 0..) |draft_tok, i| {
                try st_ptr.committed.append(self.alloc, draft_tok);
                res.sampled[i + 1] = try self.drawToken(st_ptr.committed.items, w.positions[i + 1], w.stream.sampling, &row_buf);
                if (res.rows.len > i + 1) res.rows[i + 1] = row_buf;
            }
        }
        self.last_rows.store(round_rows, .release);
        if (self.gate) |g| {
            if (call_num >= g.arm_after_verify.load(.acquire) and !g.release.load(.acquire)) {
                g.entered.store(true, .release);
                while (!g.release.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const lanes.backend.Window, paths: []const []const u32) anyerror!void {
        const self: *FakeMetalTarget = @ptrCast(@alignCast(ptr));
        _ = self.keep_calls.fetchAdd(1, .acq_rel);
        for (windows, paths) |w, path| {
            const st_ptr = self.streams.getPtr(w.stream) orelse return error.MissingStream;
            for (path, 0..) |idx, k| std.debug.assert(idx == k);
            if (path.len < w.rows()) _ = self.rollbacks.fetchAdd(1, .acq_rel);
            st_ptr.committed.shrinkRetainingCapacity(st_ptr.base_len + path.len);
        }
    }

    fn draftFn(_: *anyopaque, _: []const lanes.backend.DraftRequest) anyerror!void {}
    fn prepareFeaturesFn(_: *anyopaque, _: []const u32) anyerror!void {}

    fn featuresFn(ptr: *anyopaque, _: *lanes.Stream, _: []const u32, _: u64, count: u32) anyerror!lanes.backend.Features {
        const self: *FakeMetalTarget = @ptrCast(@alignCast(ptr));
        return .{ .buffer = @intFromPtr(&self.dummy_features), .rows = count, .row_bytes = @sizeOf(f32), .space = .host, .dtype = .f32 };
    }

    fn firstRowFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!lanes.LogprobRow {
        const self: *FakeMetalTarget = @ptrCast(@alignCast(ptr));
        const st_ptr = self.streams.getPtr(s) orelse return error.MissingStream;
        if (!st_ptr.has_first_row) return error.MissingFirstRow;
        return st_ptr.last_first_row;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self: *FakeMetalTarget = @ptrCast(@alignCast(ptr));
        if (self.streams.fetchRemove(s)) |kv| {
            var st_mut = kv.value;
            st_mut.committed.deinit(self.alloc);
        }
        _ = self.released.fetchAdd(1, .acq_rel);
    }

    fn snapBytesFn(_: *anyopaque, at: u32) u64 {
        return @as(u64, at) * @sizeOf(u32);
    }

    fn snapSaveFn(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
        const self: *FakeMetalTarget = @ptrCast(@alignCast(ptr));
        const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.MissingStream));
        const st_ptr = self.streams.getPtr(s) orelse return error.MissingStream;
        const snap = try self.alloc.create(Snapshot);
        errdefer self.alloc.destroy(snap);
        snap.tokens = try self.alloc.dupe(u32, st_ptr.committed.items[0..at]);
        return snap;
    }

    fn snapRestoreFn(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) anyerror!void {}

    fn snapDropFn(ptr: *anyopaque, saved: api.prompt_cache.Saved) void {
        const self: *FakeMetalTarget = @ptrCast(@alignCast(ptr));
        const snap: *Snapshot = @ptrCast(@alignCast(saved));
        self.alloc.free(snap.tokens);
        self.alloc.destroy(snap);
    }
};

pub const FakeDSpark = struct {
    alloc: std.mem.Allocator,
    depth: u32 = 7,
    open_calls: std.atomic.Value(usize) = .init(0),
    absorb_calls: std.atomic.Value(usize) = .init(0),
    absorb_rows: std.atomic.Value(usize) = .init(0),
    hold_calls: std.atomic.Value(usize) = .init(0),
    release_calls: std.atomic.Value(usize) = .init(0),
    held_map: std.AutoHashMap(*lanes.Stream, u32),

    pub fn init(alloc: std.mem.Allocator) FakeDSpark {
        return .{ .alloc = alloc, .held_map = std.AutoHashMap(*lanes.Stream, u32).init(alloc) };
    }

    pub fn deinit(self: *FakeDSpark) void {
        self.held_map.deinit();
    }

    pub fn drafter(self: *FakeDSpark) lanes.drafter.Drafter {
        return .{
            .kind = "DSpark",
            .ptr = self,
            .vtable = &.{
                .taps = tapsFn,
                .facts = factsFn,
                .open = openFn,
                .absorb = absorbFn,
                .hold = holdFn,
                .held = heldFn,
                .release = releaseFn,
            },
        };
    }

    fn tapsFn(_: *anyopaque) []const u32 {
        return &.{0};
    }

    fn factsFn(ptr: *anyopaque) lanes.drafter.Facts {
        const self: *FakeDSpark = @ptrCast(@alignCast(ptr));
        return .{ .depth = self.depth, .step_ms = 0, .copy_first = true, .copy_match = 12, .plain_guard = true };
    }

    fn openFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self: *FakeDSpark = @ptrCast(@alignCast(ptr));
        _ = self.open_calls.fetchAdd(1, .acq_rel);
        try self.held_map.put(s, 0);
    }

    fn absorbFn(ptr: *anyopaque, items: []const lanes.drafter.Absorb) anyerror!void {
        const self: *FakeDSpark = @ptrCast(@alignCast(ptr));
        _ = self.absorb_calls.fetchAdd(1, .acq_rel);
        for (items) |item| {
            _ = self.absorb_rows.fetchAdd(item.features.rows, .acq_rel);
            const st_ptr = self.held_map.getPtr(item.stream) orelse continue;
            if (item.follow.len > 0) st_ptr.* = item.follow[item.follow.len - 1];
        }
    }

    fn holdFn(ptr: *anyopaque, items: []const lanes.drafter.Hold) anyerror!void {
        const self: *FakeDSpark = @ptrCast(@alignCast(ptr));
        _ = self.hold_calls.fetchAdd(1, .acq_rel);
        for (items) |item| try self.held_map.put(item.stream, item.pending);
    }

    fn heldFn(ptr: *anyopaque, streams: []const *lanes.Stream, out: []const []u32) anyerror!void {
        const self: *FakeDSpark = @ptrCast(@alignCast(ptr));
        for (streams, out) |s, buf| {
            const base = self.held_map.get(s) orelse 0;
            for (buf, 0..) |*slot, i| slot.* = @intCast((base + i + 1) % vocab_size);
        }
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self: *FakeDSpark = @ptrCast(@alignCast(ptr));
        _ = self.held_map.remove(s);
        _ = self.release_calls.fetchAdd(1, .acq_rel);
    }
};

pub const Reply = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    rows: std.ArrayList(lanes.LogprobRow) = .empty,
    cached: ?u32 = null,
    drafted: u64 = 0,
    accepted: u64 = 0,
    done: ?api.Reason = null,

    pub fn sink(self: *Reply) api.Sink {
        return .{ .ctx = self, .event = onEvent };
    }

    fn onEvent(ptr: *anyopaque, _: api.Id, item: *const api.Event) void {
        const self: *Reply = @ptrCast(@alignCast(ptr));
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        switch (item.*) {
            .tokens => |ids| self.tokens.appendSlice(gpa, ids) catch {},
            .logprobs => |rows| self.rows.appendSlice(gpa, rows) catch {},
            .prefilled => |n| self.cached = n,
            .finished => |f| {
                self.done = f.reason;
                self.drafted = f.stats.drafted;
                self.accepted = f.stats.accepted;
            },
            else => {},
        }
    }

    pub fn wait(self: *Reply) !api.Reason {
        for (0..60_000) |_| {
            self.mutex.lockUncancelable(io);
            const done = self.done;
            self.mutex.unlock(io);
            if (done) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }

    pub fn deinit(self: *Reply) void {
        self.tokens.deinit(gpa);
        self.rows.deinit(gpa);
    }
};

pub const StepClock = struct {
    target: *const FakeMetalTarget,
    times: [2]f64,

    pub fn clock(self: *StepClock) lanes.backend.Clock {
        return .{ .ptr = self, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }
    fn start(_: *anyopaque) void {}
    fn elapsed(ptr: *anyopaque, _: lanes.backend.Mark) f64 {
        const self: *StepClock = @ptrCast(@alignCast(ptr));
        return self.times[0] + self.times[1] * @as(f64, @floatFromInt(self.target.last_rows.load(.acquire)));
    }
};

pub const Harness = struct {
    target: FakeMetalTarget,
    drafter: ?FakeDSpark = null,
    drafted_wrap: ?lanes.drafted.Drafted = null,
    clock: lanes.backend.WallClock,
    step_clock: StepClock = undefined,
    cfg: lanes.Config,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null,

    pub fn open(self: *Harness, opts: struct {
        drafts: bool = true,
        lanes_count: u32 = 1,
        min_match: i64 = 4,
        with_dspark: bool = false,
        cache_budget: u64 = 0,
        mismatch_at: ?usize = null,
        force_miss: bool = false,
        window_costs: []const lanes.config.Cost = &.{},
        shared_costs: []const lanes.config.Cost = &.{},
        step_ms: ?[2]f64 = null,
        copy_guard_override: ?bool = null,
        proposers: ?api.LaneHost.Proposers = null,
    }) !void {
        self.target = FakeMetalTarget.init(gpa, opts.drafts, opts.lanes_count);
        self.target.mismatch_at = opts.mismatch_at;
        self.target.force_miss = opts.force_miss;
        self.target.window_costs = opts.window_costs;
        self.target.shared_costs = opts.shared_costs;
        errdefer self.target.deinit();
        self.drafter = null;
        self.drafted_wrap = null;
        if (opts.with_dspark) {
            self.drafter = FakeDSpark.init(gpa);
            self.drafted_wrap = try lanes.drafted.Drafted.init(gpa, self.target.asBackend(), self.drafter.?.drafter());
            self.drafted_wrap.?.first_rows = true;
        }
        errdefer {
            if (self.drafted_wrap) |*w| w.deinit();
            if (self.drafter) |*d| d.deinit();
        }
        var facts = if (self.drafted_wrap) |*w| w.facts(self.target.facts()) else self.target.facts();
        if (opts.copy_guard_override) |g| facts.copy_guard = g;
        self.cfg = try lanes.Config.init(gpa, facts, st.window_rows, st.window_rows - 1);
        errdefer self.cfg.deinit(gpa);
        self.clock = .{ .io = io };
        const clk = if (opts.step_ms) |ts| blk: {
            self.step_clock = .{ .target = &self.target, .times = ts };
            break :blk self.step_clock.clock();
        } else self.clock.clock();
        self.core = lanes.Engine.init(gpa, &self.cfg, if (self.drafted_wrap) |*w| w.backend() else self.target.asBackend(), clk);
        errdefer self.core.deinit();
        self.host = api.LaneHost.init(gpa, io, &self.core, .{
            .lanes = opts.lanes_count,
            .context_window = 256,
            .prefill_step = st.prompt_rows,
            .decode_share = 0.5,
            .structures = true,
            .logprobs = true,
            .draft_sources = .{ .drafter = if (opts.with_dspark) .{ .kind = "DSpark", .name = "dspark" } else null },
        });
        self.host.min_match = opts.min_match;
        self.host.proposers = opts.proposers;
        self.cache = null;
        if (opts.cache_budget > 0) {
            const rules: api.prompt_cache.Rules = .{ .lookahead = 0, .planned = true, .grid = 1, .min_prompt = 0 };
            self.cache = api.prompt_cache.Store.init(gpa, self.target.snapshots(), rules, opts.cache_budget);
            self.host.cache = &self.cache.?;
            self.host.info_.prompt_cache = true;
        }
        try self.host.start();
    }

    pub fn deinit(self: *Harness) void {
        self.host.stop();
        if (self.cache) |*c| c.deinit();
        self.core.deinit();
        self.cfg.deinit(gpa);
        if (self.drafted_wrap) |*w| w.deinit();
        if (self.drafter) |*d| d.deinit();
        self.target.deinit();
    }
};

pub fn runOne(h: *Harness, id: api.Id, req: *const api.Request) !Reply {
    var r: Reply = .{};
    errdefer r.deinit();
    try h.host.engine().submit(id, req, r.sink());
    try std.testing.expectEqual(api.Reason.length, try r.wait());
    try std.testing.expectEqual(@as(usize, req.max_tokens), r.tokens.items.len);
    return r;
}

pub fn expectParity(plain: *const Reply, drafted: *const Reply) !void {
    try std.testing.expectEqualSlices(u32, plain.tokens.items, drafted.tokens.items);
    try std.testing.expectEqual(plain.rows.items.len, drafted.rows.items.len);
    for (plain.rows.items, drafted.rows.items) |want, got| {
        try std.testing.expectEqual(want.token, got.token);
        try std.testing.expectEqual(@as(u32, @bitCast(want.logprob)), @as(u32, @bitCast(got.logprob)));
    }
}
