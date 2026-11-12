//! The validated dense graph under the shared lane scheduler, drawer, and prompt-cache policy.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const api = @import("engine_api");
const serve = @import("hip_serve");
const engine = @import("hip_engine.zig");
const text = @import("hip_text.zig");
const snapshots = @import("hip_snapshots.zig");
const be = lanes.backend;
const width = engine.plan.max_window;

pub const Lane = struct {
    slot: usize,
    filled: usize = 0,
    /// Position of the saved last-logit row; null after a verification window.
    logits_at: ?usize = null,
    fill: ?serve.backend.Fill = null,
    spans: []text.ImageSpan,
    local: []text.ImageSpan,
    first: ?u64 = null,
    pending: ?struct { start: usize, rows: usize } = null,
};

pub const Backend = struct {
    e: *engine.Engine,
    map: std.AutoHashMapUnmanaged(*const lanes.Stream, *Lane) = .empty,
    sampler: serve.draw.Drawer,
    drawn: serve.backend.Drawn = .{},
    store: api.prompt_cache.Store,

    pub fn init(e: *engine.Engine) !*Backend {
        const h = try e.gpa.create(Backend);
        errdefer e.gpa.destroy(h);
        h.e = e;
        h.map = .empty;
        h.drawn = .{};
        h.sampler = try serve.draw.Drawer.init(e.gpa, &e.driver, .bf16, e.c.vocab, width);
        h.store = api.prompt_cache.Store.init(e.gpa, snapshots.family(h), .{
            .planned = true,
            .grid = engine.plan.prompt_rows,
            .chunk = engine.plan.prompt_rows,
            .warm = .on,
            .warm_usable = true, // requests name no starts: an end off the grid would never resume
            .min_prompt = 4096,
        }, e.p.cache_bytes);
        return h;
    }

    pub fn deinit(h: *Backend) void {
        h.e.ctx.makeCurrent() catch {};
        h.e.stream.synchronize() catch {};
        h.store.deinit();
        var it = h.map.valueIterator();
        while (it.next()) |l| h.drop(l.*);
        h.map.deinit(h.e.gpa);
        h.sampler.deinit();
        h.e.gpa.destroy(h);
    }

    pub fn backend(h: *Backend) be.Backend {
        return .{ .ptr = h, .vtable = &vtable };
    }

    pub fn lane(h: *Backend, s: *const lanes.Stream) !*Lane {
        return h.map.get(s) orelse error.UnknownStream;
    }

    fn drop(h: *Backend, l: *Lane) void {
        h.e.release(l.slot);
        h.e.gpa.free(l.local);
        h.e.gpa.free(l.spans);
        h.e.gpa.destroy(l);
    }

    fn begin(h: *Backend, s: *lanes.Stream) !*Lane {
        const prompt = s.prompt();
        if (prompt.len == 0 or prompt.len > h.e.p.window) return error.DotsContext;
        const slot = try h.e.take();
        errdefer h.e.release(slot);
        const l = try h.e.gpa.create(Lane);
        errdefer h.e.gpa.destroy(l);
        const spans = try h.e.gpa.alloc(text.ImageSpan, s.images.len);
        errdefer h.e.gpa.free(spans);
        const local = try h.e.gpa.alloc(text.ImageSpan, s.images.len);
        errdefer h.e.gpa.free(local);
        l.* = .{ .slot = slot, .spans = spans, .local = local };
        try h.map.put(h.e.gpa, s, l);
        errdefer _ = h.map.remove(s);
        if (s.reuse.saved) |saved| {
            snapshots.restore(h, s, saved) catch |err| {
                try h.e.stream.synchronize();
                s.reuse_failed = true;
                s.reuse_error = err;
                l.filled = 0;
            };
        }
        s.cached = @intCast(l.filled);
        s.fill_at.store(@intCast(l.filled), .release);
        l.fill = serve.backend.Fill.init(l.filled, s.reuse.marks);
        var end: usize = 0;
        var feature_at: usize = 0;
        for (s.images, spans) |im, *span| {
            if (s.isCancelled()) return error.Cancelled;
            if (im.at < end or im.at >= prompt.len or im.tokens == 0 or im.tokens > prompt.len - im.at or
                @as(u64, im.gh) * im.gw != @as(u64, im.tokens) * 4) return error.BadImageSpan;
            end = @as(usize, im.at) + im.tokens;
            for (prompt[im.at..end]) |id| if (id != h.e.c.image_token) return error.BadImageSpan;
            if (end <= l.filled) {
                span.* = .{ .first = im.at, .rows = 0, .features = .{ .ptr = 0, .kind = .bf16 } };
            } else {
                const features = try h.e.encode(slot, im.pixels, im.gh, im.gw, feature_at);
                span.* = .{ .first = im.at, .rows = im.tokens, .features = features };
                feature_at += im.tokens;
            }
        }
        return l;
    }

    fn intersections(h: *Backend, l: *Lane, from: usize, to: usize) []const text.ImageSpan {
        var n: usize = 0;
        for (l.spans) |span| {
            const first = @max(from, span.first);
            const end = @min(to, span.first + span.rows);
            if (end <= first) continue;
            l.local[n] = .{ .first = first - from, .rows = end - first, .features = .{ .ptr = span.features.ptr + (first - span.first) * h.e.c.hidden * 2, .kind = .bf16 } };
            n += 1;
        }
        return l.local[0..n];
    }

    fn step(h: *Backend, s: *lanes.Stream) !bool {
        if (s.isCancelled()) return error.Cancelled;
        const l = h.map.get(s) orelse try h.begin(s);
        if (l.fill == null) return true;
        const f = &l.fill.?;
        const prompt = s.prompt();
        if (f.at < prompt.len) {
            if (s.yields()) { // real work waits: the cache keeps what this pass prefilled, where a later request resumes
                if (h.store.usable(@intCast(f.at), s.chunks)) s.keepInterruptedPrefill(@intCast(f.at));
                return error.Yielded;
            }
            var to = @min(f.at + h.e.p.prompt_rows, f.limit(prompt.len));
            for (s.chunks) |cut| if (cut > f.at) {
                to = @min(to, cut);
                break;
            };
            if (to <= f.at) return error.BadPromptMark;
            const y = try h.e.forward(l.slot, prompt[f.at..to], f.at, h.intersections(l, f.at, to), false, false);
            try hip.raw.copy(h.e.slots[l.slot].logits, 0, y.ptr, @as(usize, h.e.c.vocab) * 2, h.e.stream.handle);
            try h.e.stream.synchronize();
            l.filled = to;
            l.logits_at = to;
            f.at = to;
            s.fill_at.store(@intCast(to), .release);
            if (f.cut(to)) if (s.reuse.hook) |hook| hook.at(hook.ptr, s, @intCast(to));
        }
        if (f.at < prompt.len) return false;
        if (s.isCancelled()) return error.Cancelled;
        l.first = try h.firstDraw(s, l, &.{});
        l.fill = null;
        return true;
    }

    fn firstDraw(h: *Backend, s: *lanes.Stream, l: *Lane, mask: []const u32) !u64 {
        const req = [1]serve.draw.Request{.{ .sampling = s.sampling, .position = s.prompt_len, .mask = mask }};
        var token: [1]u32 = undefined;
        try h.sampler.draw(h.e.graph_(false).ops, h.e.stream, .{ .ptr = h.e.slots[l.slot].logits.base(), .kind = .bf16 }, &req, &token, false);
        return h.drawn.take(token[0]);
    }

    fn keepRows(_: *Backend, l: *Lane, n: usize) !void {
        const pending = l.pending orelse return;
        if (n > pending.rows) return error.BadKeep;
        l.filled = pending.start + n;
        l.pending = null;
    }

    fn of(ptr: *anyopaque) !*Backend {
        const h: *Backend = @ptrCast(@alignCast(ptr));
        try h.e.ctx.makeCurrent();
        return h;
    }

    const vtable: be.Backend.VTable = .{
        .prefill = prefillFn,
        .prefill_step = stepFn,
        .cancel_prefill = cancelFn,
        .first = firstFn,
        .first_masked = maskedFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .release = releaseFn,
        .reports_fill = true,
    };

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        while (!try (try of(ptr)).step(s)) {}
    }
    fn stepFn(ptr: *anyopaque, s: *lanes.Stream, _: u32) anyerror!bool {
        return (try of(ptr)).step(s);
    }
    fn cancelFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const h = try of(ptr);
        try h.e.stream.synchronize();
        const l = h.map.get(s) orelse return;
        if (l.fill != null and l.filled > 0) if (s.reuse.hook) |hook| hook.at(hook.ptr, s, @intCast(l.filled));
    }
    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const h = try of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        return (try h.lane(s)).first orelse error.NotPrefilled;
    }
    fn maskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const h = try of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        return h.firstDraw(s, try h.lane(s), mask);
    }
    fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }
    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return (try of(ptr)).drawn.read(handle);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const h = try of(ptr);
        for (windows, out) |w, *result| {
            const l = try h.lane(w.stream);
            if (l.pending) |p| try h.keepRows(l, p.rows);
            var tokens: [width]u32 = undefined;
            const n = try serve.backend.windowTokens(w, &.{}, l.filled, &tokens);
            l.logits_at = null;
            const y = try h.e.forward(l.slot, tokens[0..n], l.filled, &.{}, true, true);
            var reqs: [width]serve.draw.Request = undefined;
            serve.backend.requests(w, &reqs);
            try h.sampler.draw(h.e.graph_(true).ops, h.e.stream, y, reqs[0..n], result.sampled, false);
            @memcpy(result.drafts, tokens[1..n]);
            l.pending = .{ .start = l.filled, .rows = n };
        }
    }
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const h = try of(ptr);
        for (windows, paths) |w, path| {
            for (path, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
            try h.keepRows(try h.lane(w.stream), path.len);
        }
    }
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const h = try of(ptr);
        for (requests) |r| {
            const l = try h.lane(r.stream);
            if (l.pending) |p| try h.keepRows(l, if (r.rows) |rows| rows.len else p.rows);
            if (r.depth > 0) return error.NoDraftHead;
        }
    }
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const h: *Backend = @ptrCast(@alignCast(ptr));
        h.e.ctx.makeCurrent() catch {};
        if (h.map.fetchRemove(s)) |entry| h.drop(entry.value);
    }
};
