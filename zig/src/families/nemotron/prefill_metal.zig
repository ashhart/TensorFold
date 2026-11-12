//! Nemotron Metal's prompt pass on the device (prefill_steps.zig drives it): the cache, the prompt buffer, each chunk.
const std = @import("std");
const lanes = @import("lanes");
const backend = @import("backend.zig");
const fwd = @import("forward.zig");
const st = @import("state.zig");
const kv_pages = @import("kv_pages.zig");
const lesson_route = @import("living").lesson_route;
const layers = @import("layers.zig");
const snapshot = @import("snapshot.zig");
const pass = @import("prefill_steps.zig");
const Metal = backend.Metal;

/// One chunk's encode, as it was in backend.zig's prefill: the forward, the head's absorb, the last row kept.
const Chunks = struct {
    n: usize,
    c: *st.Cache,
    at: usize,
    rows: usize,
    pub fn encode(j: @This(), b: *Metal, e: *fwd.Enc) !void {
        const d = b.m.config.hidden * 2;
        const rows = j.rows;
        const fresh = try b.pool.take();
        var out = b.scratch.x; // the chunk's final-normed rows
        if (rows <= backend.fused_rows) {
            const segs = [_]fwd.Seg{.{ .rows = rows, .cache = j.c, .store = .full, .slot = @intCast(fresh) }};
            b.forward().body(e, &segs, b.prompt, j.at * 4);
        } else {
            const w = &b.wide.?;
            w.chunk(e, b.forward(), j.c, b.prompt, j.at * 4, rows, fresh);
            out = w.s.x;
        }
        j.c.advance(&b.pool, rows, fresh);
        // the head's cache takes each row whose next token the prompt holds, as many rows a call as a window
        const absorbed = pass.absorbed(j.at, rows, j.n);
        if (b.head) |*h| {
            var r: usize = 0;
            while (r < absorbed) : (r += backend.max_window) h.absorb(e, j.c, @min(backend.max_window, absorbed - r), out, r * d, b.prompt, (j.at + 1 + r) * 4);
        }
        j.c.start = 0;
        j.c.rows = rows;
        if (rows > backend.fused_rows) {
            // the last row where the first draw and the head read it
            e.pipe(b.m.kernels.get("tf_copy_rows"));
            e.buf(out, (rows - 1) * d, 0);
            e.buf(b.scratch.x, 0, 1);
            e.run(.{ d / 2, 1, 1 }, .{ 256, 1, 1 });
            j.c.rows = 1;
        }
    }
};

/// The device side of a pass for prefill_steps.zig: a stream's cache, the shared prompt buffer, each chunk's encode.
const Ops = struct {
    m: *Metal,

    pub fn steps(o: Ops) *pass.Steps {
        return &o.m.steps;
    }
    pub fn cap(o: Ops) usize {
        return o.m.o.chunk;
    }
    fn cache(o: Ops, s: *lanes.Stream) *st.Cache {
        return o.m.caches.get(s).?;
    }
    pub fn decision(o: Ops, s: *lanes.Stream) ?usize {
        const c = o.cache(s);
        return if (c.decide == lesson_route.none) null else @intCast(c.decide);
    }
    pub fn grow(o: Ops, s: *lanes.Stream, rows: usize) !void {
        try kv_pages.grow(o.cache(s), rows);
    }
    pub fn body(o: Ops, s: *lanes.Stream, at: usize, rows: usize) Chunks {
        return .{ .n = s.prompt_len, .c = o.cache(s), .at = at, .rows = rows };
    }
    /// This prompt's ids into the prompt buffer the chunks read (the queue landed first).
    pub fn load(o: Ops, s: *lanes.Stream) void {
        const ids = s.prompt();
        @memcpy(o.m.prompt.slice(u32, ids.len), ids);
    }

    /// The stream's cache opened, its ids loaded and a kept state restored: the row its pass starts from.
    pub fn open(o: Ops, s: *lanes.Stream) !usize {
        const self = o.m;
        const ids = s.prompt();
        try backend.limits.check(ids.len, s.max_new, self.o.capacity);
        if (s.isCancelled()) return error.Cancelled;
        const rows_needed = @min(self.o.capacity, ids.len + s.max_new + backend.max_lanes + 1); // the reply and a widest window past it
        try self.drain();
        if (self.caches.get(s)) |old| {
            old.deinit(&self.pool);
            self.gpa.destroy(old);
            _ = self.caches.remove(s);
        }
        const c = try @import("residency.zig").openCache(self, s, rows_needed);
        @memcpy(self.prompt.slice(u32, ids.len), ids);
        s.cached = 0;
        var at: usize = 0;
        if (s.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
            const snap: *snapshot.Snap = @ptrCast(@alignCast(saved));
            if (snap.at < ids.len) if (snapshot.restore(self, c, snap)) |_| {
                at = snap.at;
            } else |_| {};
            s.cached = @intCast(at);
            s.reuse_failed = at == 0;
        }
        s.fill_at.store(@intCast(at), .monotonic); // a restored prefix is done the moment it is restored
        return at;
    }

    pub fn finish(o: Ops, s: *lanes.Stream) !void {
        const c = o.cache(s);
        if (o.m.trace and c.decide != lesson_route.none) try traceLessons(o.m, c);
    }
};

fn of(ptr: *anyopaque) *Metal {
    return @ptrCast(@alignCast(ptr));
}

/// The vtable's `prefill`: the whole pass, a command buffer a chunk, each committed before the one ahead is waited on.
pub fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
    return pass.whole(of(ptr), Ops{ .m = of(ptr) }, s);
}

/// The vtable's `prefill_step`: one chunk a call at the same cuts (no chunk_invariant: `max_rows` is never set).
pub fn prefillStep(ptr: *anyopaque, s: *lanes.Stream, _: u32) anyerror!bool {
    return pass.step(of(ptr), Ops{ .m = of(ptr) }, s);
}

/// The vtable's `cancel_prefill`: no prompt work runs; the state at the last submitted chunk is kept once it lands.
pub fn cancelPrefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
    return pass.cancel(of(ptr), Ops{ .m = of(ptr) }, s);
}

/// Dev trace (TF_SLIDE_TRACE): the stream's router scores at its decision row and the lessons acting on its reply.
fn traceLessons(self: *Metal, c: *const st.Cache) !void {
    const ad = layers.firstAdapter(&self.m.weights) orelse return;
    try self.drain();
    const n = lesson_route.max_blocks;
    const at: usize = @intCast(@max(c.rid, 0));
    const score = ad.score.slice(f32, lesson_route.max_slots * n)[at * n ..][0..n];
    const flags = ad.latch.slice(u32, lesson_route.max_slots * n)[at * n ..][0..n];
    const need = ad.need.slice(f32, n);
    var line: [2048]u8 = undefined;
    var w: std.Io.Writer = .fixed(&line);
    for (0..ad.rank / 16) |b| w.print(" {d:.2}/{d:.2}{s}", .{ score[b], need[b], if (flags[b] != 0) "*" else "" }) catch break;
    std.log.info("slide trace: decision row {d}: scores{s} (* acting)", .{ c.decide, w.buffered() });
}
