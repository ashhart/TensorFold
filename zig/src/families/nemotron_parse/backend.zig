//! Parse's target and one-step head behind the existing shared lane engine; no family-owned scheduler.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const glue = serve.backend;
const draw = serve.draw;
const be = lanes.backend;
const c = @import("config.zig");
const decoder = @import("decoder.zig");
const engine = @import("engine.zig");
const Compute = @import("compute.zig").Compute;
const Tensor = hip.ops.Tensor;
const prefill = @import("prefill.zig");

pub const max_window = 32;
pub const Lane = struct {
    state: decoder.State,
    prefix: hip.DeviceBuffer,
    image_rows: usize,
    fill: usize = 0,
    first: u64 = 0,
    first_row: ?[]u16 = null,
    held: [1]u32 = undefined,
    held_n: usize = 0,
    pending: ?struct { start: usize, rows: usize } = null,
};

pub const Backend = struct {
    a: std.mem.Allocator,
    e: *engine.Engine,
    streams: glue.Streams(Lane),
    sampler: draw.Drawer,
    drafts: bool,
    cache: @import("engine_api").prompt_cache.Store,

    pub fn init(a: std.mem.Allocator, e: *engine.Engine, drafts: bool) !*Backend {
        const h = try a.create(Backend);
        errdefer a.destroy(h);
        h.* = .{ .a = a, .e = e, .streams = .init(a), .sampler = try draw.Drawer.init(a, &e.r, .bf16, c.vocab, e.streams * max_window), .drafts = drafts, .cache = undefined };
        h.cache = @import("engine_api").prompt_cache.Store.init(a, @import("cache.zig").snapshots(h), .{ .chunk = 1, .min_prompt = 0, .repeat_min = 1, .repeat_keyed = true, .warm = .on }, 0);
        return h;
    }

    pub fn deinit(h: *Backend) void {
        h.e.stream.synchronize() catch {};
        h.cache.deinit();
        var it = h.streams.map.valueIterator();
        while (it.next()) |l| h.drop(l.*);
        h.streams.deinit();
        h.sampler.deinit();
        const a = h.a;
        a.destroy(h);
    }

    pub fn drop(h: *Backend, l: *Lane) void {
        h.e.stream.synchronize() catch {};
        if (l.first_row) |row| h.a.free(row);
        l.prefix.free();
        l.state.deinit();
        h.a.destroy(l);
    }

    pub fn facts(h: *const Backend) lanes.Model {
        return .{ .exact_width = max_window, .mtp = h.drafts, .speculate = h.drafts, .speculate_early = false, .drafts = 1, .hidden_rows = true, .batch_rows = @intCast(h.e.streams * max_window), .max_streams = @intCast(h.e.streams), .draft_streams = h.drafts };
    }

    pub fn backend(h: *Backend) be.Backend {
        return .{ .ptr = h, .vtable = &vtable, .penalties = true, .chunk_invariant = true };
    }

    const vtable: be.Backend.VTable = .{ .reports_fill = true, .prefill = fillAll, .prefill_step = fillStep, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .tree = tree, .held = held, .release = release };

    pub fn of(ptr: *anyopaque) *Backend {
        const h: *Backend = @ptrCast(@alignCast(ptr));
        glue.bind(&h.e.ctx);
        return h;
    }

    fn fillAll(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        while (!try fillStep(ptr, s, 0)) {}
    }

    fn fillStep(ptr: *anyopaque, s: *lanes.Stream, max_rows: u32) anyerror!bool {
        return prefill.step(of(ptr), s, max_rows);
    }

    fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        if (position != s.prompt_len) return error.PositionMismatch;
        return (try of(ptr).streams.get(s)).first;
    }

    fn firstMasked(ptr: *anyopaque, s: *lanes.Stream, position: u64, bits: []const u32) anyerror!u64 {
        const h = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const l = try h.streams.get(s);
        const row = try h.a.dupe(u16, l.first_row orelse return error.NoFirstRow);
        defer h.a.free(row);
        var r = try h.request(h.a, s, l.image_rows, s.prompt_len, &.{}, position);
        defer h.a.free(r.history);
        r.mask = bits;
        return h.streams.drawn.take(try h.sampler.drawHost(row, r));
    }

    fn queue(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return of(ptr).streams.drawn.read(handle);
    }

    /// Sampling history includes the current window's input prefix, and excludes visual placeholder IDs.
    pub fn request(_: *Backend, a: std.mem.Allocator, s: *lanes.Stream, image_rows: usize, at: usize, extra: []const u32, position: u64) !draw.Request {
        if (at < image_rows or at > s.context.items.len or at < s.prompt_len) return error.PositionMismatch;
        const base = s.context.items[image_rows..at];
        const history = try a.alloc(u32, base.len + extra.len);
        @memcpy(history[0..base.len], base);
        @memcpy(history[base.len..], extra);
        return .{ .sampling = s.sampling, .position = position, .repetition = 1.1, .forced = if (s.max_new > 0 and position == s.prompt_len + s.max_new - 1) 2 else null, .history = history, .reply = history[s.prompt_len - image_rows ..], .penalties = s.penalties };
    }

    fn settle(l: *Lane, rows: ?usize) !void {
        if (l.pending) |p| {
            const n = rows orelse p.rows;
            if (n > p.rows) return error.BadKeep;
            try l.state.keep(p.start + n);
            l.pending = null;
        }
    }

    fn verify(ptr: *anyopaque, ws: []const be.Window, out: []be.Verified) anyerror!void {
        const h = of(ptr);
        if (ws.len == 0 or ws.len > h.e.streams or out.len != ws.len) return error.WindowTooWide;
        var arena: std.heap.ArenaAllocator = .init(h.a);
        defer arena.deinit();
        const a = arena.allocator();
        const spans = try a.alloc(decoder.Span, ws.len);
        const starts = try a.alloc(usize, ws.len);
        const tokens = try a.alloc(u32, ws.len * max_window);
        const reqs = try a.alloc(draw.Request, tokens.len);
        var total: usize = 0;
        for (ws, 0..) |w, i| {
            const l = try h.streams.get(w.stream);
            try settle(l, null);
            const n = try glue.windowTokens(w, l.held[0..l.held_n], l.state.len, tokens[total..][0..max_window]);
            if (w.positions.len != n or out[i].sampled.len != n or out[i].drafts.len != n - 1) return error.BadWindow;
            spans[i] = .{ .state = &l.state, .rows = n };
            starts[i] = l.state.len;
            for (0..n) |j| {
                reqs[total + j] = try h.request(a, w.stream, l.image_rows, l.state.len, tokens[total..][0 .. j + 1], w.positions[j]);
                const words = if (w.masks.len == 0) 0 else w.masks.len / n;
                if (w.masks.len != words * n) return error.BadMask;
                reqs[total + j].mask = w.masks[j * words ..][0..words];
            }
            total += n;
        }
        for (tokens[0..total]) |id| if (id >= c.vocab) return error.BadInputToken;
        _ = try decoder.rowsOf(spans);
        h.e.scratch.reset();
        const compute = h.e.compute();
        var device = try hip.DeviceBuffer.fromHost(&h.e.r, std.mem.sliceAsBytes(tokens[0..total]));
        defer device.free();
        const embedded = try decoder.embed(compute, device.base(), total);
        const hidden = try decoder.forwardMany(compute, spans, embedded);
        const logits = try decoder.logits(compute, hidden, total);
        const choices = try a.alloc(u32, total);
        try h.sampler.draw(compute.ops, h.e.stream, logits, reqs[0..total], choices, false);
        var at: usize = 0;
        for (ws, out, starts) |w, *o, start| {
            const l = try h.streams.get(w.stream);
            const n = w.rows();
            @memcpy(o.sampled, choices[at..][0..n]);
            @memcpy(o.drafts, tokens[at + 1 ..][0 .. n - 1]);
            l.pending = .{ .start = start, .rows = n };
            l.held_n = 0;
            at += n;
        }
    }

    fn keep(ptr: *anyopaque, ws: []const be.Window, paths: []const []const u32) anyerror!void {
        const h = of(ptr);
        for (ws, paths) |w, path| {
            for (path, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
            try settle(try h.streams.get(w.stream), path.len);
        }
    }

    /// The trained auxiliary head predicts one token; all streams' proposals use one packed head forward.
    fn draft(ptr: *anyopaque, rs: []const be.DraftRequest) anyerror!void {
        const h = of(ptr);
        if (rs.len == 0) return;
        if (!h.drafts or rs.len > h.e.streams) return error.NoDraftHead;
        var arena: std.heap.ArenaAllocator = .init(h.a);
        defer arena.deinit();
        const a = arena.allocator();
        const active = try a.alloc(*Lane, rs.len);
        const tokens = try a.alloc(u32, rs.len);
        const reqs = try a.alloc(draw.Request, rs.len);
        h.e.scratch.reset();
        const compute = h.e.compute();
        const hidden = try compute.alloc(rs.len * c.hidden, .bf16);
        var count: usize = 0;
        for (rs) |r| {
            const l = try h.streams.get(r.stream);
            try settle(l, if (r.rows) |rows| rows.len else null);
            l.held_n = 0;
            if (r.depth == 0) continue;
            if (r.position != l.state.len + 1 or r.depth > 1 or r.lanes != null) return error.BadDraftRequest;
            const token = if (r.follow.len > 0) r.follow[r.follow.len - 1] else try glue.firstToken(&h.streams.drawn, r);
            tokens[count] = token;
            active[count] = l;
            reqs[count] = try h.request(a, r.stream, l.image_rows, l.state.len, tokens[count..][0..1], r.position);
            try compute.ops.copyCols(try l.state.hiddenAt(l.state.len - 1), c.hidden, 0, hidden.ptr + count * c.hidden * 2, 1, c.hidden);
            count += 1;
        }
        if (count == 0) return;
        for (tokens[0..count]) |id| if (id >= c.vocab) return error.BadInputToken;
        var device = try hip.DeviceBuffer.fromHost(&h.e.r, std.mem.sliceAsBytes(tokens[0..count]));
        defer device.free();
        const logits = try @import("mtp.zig").logits(compute, hidden, device.base(), count);
        const choices = try a.alloc(u32, count);
        try h.sampler.draw(compute.ops, h.e.stream, logits, reqs[0..count], choices, false);
        for (active[0..count], choices) |l, token| {
            l.held[0] = token;
            l.held_n = 1;
        }
    }

    fn tree(ptr: *anyopaque, s: *lanes.Stream, _: std.mem.Allocator) anyerror!?lanes.stream.Held {
        return .{ .count = @intCast((try of(ptr).streams.get(s)).held_n) };
    }

    fn held(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const l = try of(ptr).streams.get(s);
        if (out.len > l.held_n) return error.NoHeldDrafts;
        @memcpy(out, l.held[0..out.len]);
    }

    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const h = of(ptr);
        h.drop(h.streams.forget(s) orelse return);
    }
};
