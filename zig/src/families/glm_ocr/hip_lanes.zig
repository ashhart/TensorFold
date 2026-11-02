//! GLM-OCR over the shared lane core, GPU drawer, callback adapter and copy-draft policy.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const c = @import("config.zig");
const Engine = @import("hip_engine.zig").Engine;
const decoder = @import("hip_text.zig");
const Input = @import("hip_input.zig").Input;
const glue = serve.backend;
const be = lanes.backend;

pub const Lane = struct {
    cache: decoder.Cache,
    input: ?Input,
    len: usize = 0,
    delta: i32 = 0,
    id: u32 = 0,
    fill: ?glue.Fill = null,
    pending: ?glue.Pending = null,
    first_row: ?[]u16 = null,
    held: [0]u32 = .{},
    held_n: usize = 0,
};

pub const Hip = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    streams: glue.Streams(Lane),
    tp: serve.protocol.Sender,
    max_shared: usize = 0,
    const Shared = glue.Backend(Hip);
    pub const backend = Shared.backend;

    pub fn init(a: std.mem.Allocator, e: *Engine) !*Hip {
        const h = try a.create(Hip);
        h.* = .{ .gpa = a, .e = e, .streams = .init(a), .tp = .{ .gpa = a } };
        return h;
    }
    pub fn deinit(h: *Hip) void {
        var it = h.streams.map.valueIterator();
        while (it.next()) |lane| h.release(lane.*);
        h.streams.deinit();
        h.tp.deinit();
        h.gpa.destroy(h);
    }
    pub fn facts(h: *const Hip) lanes.Model {
        return .{ .exact_width = 32, .hidden_rows = true, .batch_rows = 32, .max_streams = @intCast(h.e.streams), .mtp = false, .speculate = false, .drafts = 0 };
    }
    pub fn drawer(h: *Hip) *serve.draw.Drawer {
        return &h.e.drawer;
    }
    pub fn release(h: *Hip, lane: *Lane) void {
        if (lane.input) |*input| input.deinit();
        lane.cache.deinit();
        if (lane.first_row) |row| h.gpa.free(row);
        h.gpa.destroy(lane);
    }
    pub fn begin(h: *Hip, s: *lanes.Stream) !*Lane {
        if (s.reuse.saved != null) return error.PromptCacheUnsupported;
        const ids = s.prompt();
        if (ids.len == 0 or ids.len > h.e.window or s.max_new > h.e.window - ids.len) return error.PromptTooLong;
        if (h.streams.map.count() >= h.e.streams) return error.NoFreeLane;
        const images = try h.gpa.alloc(@import("processor.zig").Prepared, s.images.len);
        defer h.gpa.free(images);
        var at: usize = 0;
        for (s.images, images) |image, *p| {
            while (at < ids.len and ids[at] != c.image_token) at += 1;
            if (image.at != at or image.tokens != @as(usize, image.gh) * image.gw / 4) return error.BadImageSpan;
            p.* = .{ .geometry = .{ .h = @as(usize, image.gh) * 14, .w = @as(usize, image.gw) * 14 }, .pixels = @constCast(image.pixels), .hash = image.hash };
            at += image.tokens;
        }
        const lane = try h.gpa.create(Lane);
        errdefer h.gpa.destroy(lane);
        lane.* = .{ .cache = try .init(h.e.stream, h.e.window, try h.e.available()), .input = null, .fill = glue.Fill.init(0, &.{}) };
        errdefer lane.cache.deinit();
        lane.input = try Input.init(h.gpa, &h.e.graph, ids, images, try h.e.available());
        errdefer lane.input.?.deinit();
        lane.delta = lane.input.?.rope.delta;
        lane.id = try h.streams.idOf(s);
        errdefer _ = h.streams.ids.remove(s);
        try h.streams.map.put(h.gpa, s, lane);
        return lane;
    }
    pub fn reach(_: *Hip, _: *Lane, _: usize, limit: usize) !usize {
        return limit;
    }
    pub fn cancelOf(_: *Hip, s: *lanes.Stream) *lanes.Stream {
        return s;
    }
    pub fn advance(_: *Hip, _: *Lane, _: []const u32, _: usize, _: usize, _: *lanes.Stream) !void {
        return error.UnplannedPromptCut;
    }
    pub fn keepCut(_: *Hip, _: *Lane, _: []const u32, _: usize) !void {
        return error.PromptCacheUnsupported;
    }
    const Cancel = struct {
        fn tap(ptr: *anyopaque, _: []const u8, _: u64, _: usize) !void {
            const s: *lanes.Stream = @ptrCast(@alignCast(ptr));
            if (s.isCancelled()) return error.Cancelled;
        }
    };
    pub fn finish(h: *Hip, lane: *Lane, prompt: []const u32, from: usize, request: serve.draw.Request, cancel: *lanes.Stream) !u32 {
        if (from != 0 or prompt.len != lane.input.?.rows) return error.UnplannedPromptCut;
        const input = &lane.input.?;
        const normed = try h.e.graph.forward(&lane.cache, input.embeddings(), try input.rotary(), input.rows, .{ .context = cancel, .capture = Cancel.tap });
        lane.len = lane.cache.len;
        const logits = try h.e.graph.lastLogits(normed, input.rows);
        var token: [1]u32 = undefined;
        h.e.arena.reset();
        try h.e.drawer.draw(h.e.drawOps(), h.e.stream, .{ .ptr = logits, .kind = .bf16 }, &.{request}, &token, false);
        input.deinit();
        lane.input = null;
        return token[0];
    }
    pub fn keepRows(_: *Hip, lane: *Lane, rows: usize) !void {
        const pending = lane.pending orelse return error.NoPendingRows;
        if (rows > pending.rows) return error.Invalid;
        try lane.cache.truncate(lane.len + rows);
        lane.len += rows;
        lane.pending = null;
    }
    pub fn flush(_: *Hip) !void {}
    pub fn draft(h: *Hip, requests: []const be.DraftRequest) !void {
        for (requests) |r| {
            if (r.depth != 0) return error.NoDraftHead;
            try glue.keepFor(h, try h.streams.get(r.stream), r);
        }
    }
    pub fn verify(h: *Hip, windows: []const be.Window, out: []be.Verified) !void {
        if (windows.len == 0 or windows.len > h.e.streams or windows.len != out.len) return error.WindowTooWide;
        var segments: [32]decoder.Segment = undefined;
        var tokens: [32][32]u32 = undefined;
        var positions: [32][3]i32 = undefined;
        var reqs: [32]serve.draw.Request = undefined;
        var drawn: [32]u32 = undefined;
        var total: usize = 0;
        const table = try h.e.weights.get(@import("schema.zig").text ++ "embed_tokens.weight");
        for (windows, out, 0..) |w, result, i| {
            const lane = try h.streams.get(w.stream);
            try glue.settle(h, lane);
            const n = try glue.windowTokens(w, &.{}, lane.len, &tokens[i]);
            if (n > 32 - total or w.positions.len != n or result.sampled.len != n or result.drafts.len != n - 1 or n > lane.cache.capacity - lane.len) return error.WindowTooWide;
            if (w.masks.len != 0 and w.masks.len != n * ((c.vocab + 31) / 32)) return error.Invalid;
            for (tokens[i][0..n], 0..) |token, j| {
                if (token >= c.vocab) return error.Invalid;
                const p = @as(i64, @intCast(lane.len + j)) + lane.delta;
                if (p < 0 or p > c.context) return error.Invalid;
                positions[total + j] = @splat(@intCast(p));
            }
            segments[i] = .{ .cache = &lane.cache, .embeddings = h.e.input.base() + total * c.hidden * 2, .positions = h.e.input.base() + 32 * c.hidden * 2 + total * 12, .rows = n, .phase = .decode };
            glue.requests(w, reqs[total..]);
            total += n;
        }
        var at: usize = 0;
        for (windows, 0..) |w, i| {
            for (tokens[i][0..w.rows()]) |token| {
                try h.e.input.copyFromAsync(at * c.hidden * 2, table + token * c.hidden * 2, c.hidden * 2, h.e.stream);
                at += 1;
            }
        }
        try h.e.input.upload(32 * c.hidden * 2, std.mem.sliceAsBytes(positions[0..total]));
        const normed = try h.e.graph.forwardMany(segments[0..windows.len], null);
        const logits = try h.e.graph.logitsRows(normed, total);
        h.e.arena.reset();
        try h.e.drawer.draw(h.e.drawOps(), h.e.stream, .{ .ptr = logits, .kind = .bf16 }, reqs[0..total], drawn[0..total], false);
        if (windows.len > 1 and windows.len > h.max_shared) {
            h.max_shared = windows.len;
            std.log.info("GLM-OCR shared verify: {d} streams, {d} rows", .{ windows.len, total });
        }
        at = 0;
        for (windows, out) |w, *result| {
            @memcpy(result.sampled, drawn[at..][0..w.rows()]);
            @memcpy(result.drafts, w.tokens);
            (try h.streams.get(w.stream)).pending = .{ .rows = w.rows() };
            at += w.rows();
        }
    }
};
