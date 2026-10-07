//! Nemotron's prompt as staggered segments (cuda/segments.zig): each segment is a whole serial chunk, so the bits match.

const std = @import("std");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");
const state = @import("cuda_state.zig");
const fwd = @import("cuda_forward.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Cancel = @import("cuda_engine.zig").Cancel;
const Head = @import("cuda_mtp.zig").Head;

const segs = cuda.segments;

pub const MAX = segs.MAX;

/// The side streams, events and extra scratch sets a call of up to `n` segments uses (made on the first such prompt).
pub const Segments = struct {
    runner: segs.Runner,
    sibs: [MAX - 1]state.Buffers = undefined,
    n: usize = 1,

    pub fn init(e: *const Engine, n: usize) !Segments {
        var s: Segments = .{ .runner = try segs.Runner.init(e.ctx.d, n) };
        errdefer s.deinit();
        while (s.n < n) : (s.n += 1) s.sibs[s.n - 1] = try e.b.sibling(e.ctx.d, e.c, e.nch);
        return s;
    }

    pub fn deinit(s: *Segments) void {
        for (s.sibs[0 .. s.n - 1]) |*b| b.deinit();
        s.runner.deinit();
        s.* = undefined;
    }
};

/// One segment: a forward on its stream and scratch, and where its chunk stands.
const Seg = struct { f: fwd.Forward, walk: fwd.Walk };

/// Nemotron's hooks, Forward.chunk's parts; every block waits ahead of its mixer (in-place states and cache keys).
const Hooks = struct {
    list: []Seg,

    pub fn begin(h: *Hooks, l: *const segs.Lane) !void {
        try h.list[l.k].f.chunkBegin(&h.list[l.k].walk);
    }
    pub fn wait(_: *Hooks, _: usize) segs.Wait {
        return .mixer;
    }
    pub fn handoff(_: *Hooks, _: *const segs.Lane, _: usize) !void {}
    pub fn pre(h: *Hooks, l: *const segs.Lane, i: usize) !void {
        try h.list[l.k].f.chunkPre(&h.list[l.k].walk, i);
    }
    pub fn mixer(h: *Hooks, l: *const segs.Lane, i: usize) !void {
        try h.list[l.k].f.chunkMixer(&h.list[l.k].walk, i);
    }
    pub fn post(h: *Hooks, l: *const segs.Lane, i: usize) !void {
        try h.list[l.k].f.chunkPost(&h.list[l.k].walk, i);
    }
    pub fn finish(h: *Hooks, l: *const segs.Lane) !void {
        try h.list[l.k].f.chunkFinish(&h.list[l.k].walk);
    }
};

/// The prompt from a reset state, `parts` chunks a call (`cancel` asked between calls); leaves the head and prompt buffers as serial does.
pub fn prefill(e: *Engine, s: *Segments, prompt: []const u32, head: ?*Head, parts: usize, cancel: ?Cancel) !void {
    if (parts < 2 or parts > s.n) return error.Segments;
    const R = state.prefill_rows;
    var bufs: [MAX]*state.Buffers = undefined;
    bufs[0] = &e.b;
    for (1..parts) |k| {
        s.sibs[k - 1].follow(&e.b);
        bufs[k] = &s.sibs[k - 1];
    }
    var at: usize = 0;
    var last: usize = 0;
    var last_rows: usize = 0;
    while (at < prompt.len) {
        if (Cancel.now(cancel)) return error.Cancelled;
        const c = segs.chunked(prompt.len - at, R, parts);
        var seg: [MAX]Seg = undefined;
        try e.copied.synchronize();
        const host = e.promptHost(c.rows);
        @memcpy(host, prompt[at..][0..c.rows]);
        for (0..c.parts) |k| { // every segment's ids go up on the caller's stream, ahead of the call's fork
            const rows = segs.chunkRows(c.rows, R, k);
            try e.ops().upload(bufs[k].p_ids, std.mem.sliceAsBytes(host[k * R ..][0..rows]));
            const ops: kern.Ops = .{ .k = &e.k, .s = s.runner.stream(e.stream, k) };
            const f = fwd.Forward.init(e.c, &e.w, bufs[k], ops, e.max_len, e.nch, e.sampling != null);
            seg[k] = .{ .f = f, .walk = .{ .rows = rows, .pos = e.pos + k * R } };
        }
        try e.copied.record(e.stream);
        var hooks: Hooks = .{ .list = seg[0..c.parts] };
        try s.runner.run(e.stream, c.parts, e.w.blocks.len, &hooks);
        for (0..c.parts) |k| { // the caller's stream has joined every segment
            const start = at + k * R;
            const known = @min(seg[k].walk.rows, prompt.len - 1 - start);
            if (head) |h| if (known > 0) try h.absorb(bufs[k].p_hidden, prompt[start + 1 ..][0..known]);
        }
        last = c.parts - 1;
        last_rows = seg[last].walk.rows;
        e.pos += c.rows;
        at += c.rows;
    }
    if (last == 0) return;
    const o = e.ops();
    try o.copy(e.b.p_hidden, bufs[last].p_hidden, last_rows * e.c.hidden * 2);
    try o.copy(e.b.p_logits, bufs[last].p_logits, e.c.vocab * 2);
    try o.copy(e.b.p_sampled, bufs[last].p_sampled, 4);
}
