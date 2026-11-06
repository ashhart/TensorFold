//! Ling's one-sequence verify windows as CUDA graphs on main's graph cache: ids go up first.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("forward.zig");
const head = @import("mtp.zig");
const latent = @import("graph.zig").latent;

/// A graph's MLA calls run the window's chunk count rounded up to this many (trailing chunks merge as nothing).
pub const bucket = 8;

/// What a captured window depends on besides the weights: the sequence and scratch, rows, KDA copy, chunk count.
pub const Key = struct { arena: u64, buffers: u64, rows: u32, parity: u32, nch: u32 };

pub const Graphs = struct {
    gpa: std.mem.Allocator,
    eng: cuda.graph_cache.CudaEngine,
    cache: cuda.graph_cache.Cache(Key),
    epoch: u64 = 0, // the graphs' fingerprint: a new value drops them all (shared scratch may have moved)
    rows_max: usize = 0, // the widest window since the last drop

    /// The graphs (replays are byte-equal to eager windows).
    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver) !?*Graphs {
        const g = try gpa.create(Graphs);
        g.* = .{ .gpa = gpa, .eng = .{ .d = d }, .cache = undefined };
        g.cache = cuda.graph_cache.Cache(Key).init(gpa, g.eng.engine(), .{ .max = 24 });
        g.cache.name = "ling windows";
        return g;
    }

    pub fn destroy(g: *Graphs) void {
        g.cache.deinit();
        g.gpa.destroy(g);
    }
};

/// After an eager forward (prompt chunk, shared window) shared scratch may have moved: drop graphs that point at it.
pub fn eager(graphs: ?*Graphs, rows: usize, prompt: bool) void {
    const g = graphs orelse return;
    if (!prompt and rows <= g.rows_max) return;
    g.epoch += 1;
    g.rows_max = if (prompt) 0 else rows;
}

/// A shared window's forward (several sequences: eager), after eager()'s check.
pub fn shared(graphs: ?*Graphs, f: fw.Forward, segs: []const fw.Seg, ids: []const i32) !void {
    eager(graphs, try f.rowsOf(segs), false);
    return f.run(segs, ids);
}

/// h.absorb (eager) after eager()'s check: a draft's kept rows may be wider than any window since the last drop.
pub fn absorb(graphs: ?*Graphs, h: *const head.Head, f: fw.Forward, hd: *fw.State, next: []const i32, hidden: u64) !void {
    eager(graphs, next.len, false);
    return h.absorb(f.m, f.s, f.b, hd, next, hidden);
}

const Body = struct {
    f: *const fw.Forward,
    segs: []const fw.Seg,
    rows: usize,

    fn run(ctx: *anyopaque, s: cuda.Stream) anyerror!void {
        const b: *const Body = @ptrCast(@alignCast(ctx));
        _ = s; // the forward's own stream is the one captured
        try b.f.embedRows(b.rows);
        try b.f.stack(b.segs);
        try b.f.head(b.rows);
    }
};

/// f.run(segs, ids); a graph's when `graphs` is set, the window is one sequence's and its cache is not split.
pub fn run(graphs: ?*Graphs, f: fw.Forward, segs: []const fw.Seg, ids: []const i32) !void {
    const g = graphs orelse return f.run(segs, ids);
    if (segs.len != 1 or f.b.prompt or fw.splitOf(f.g.w) != null) return f.run(segs, ids);
    const rows = try f.rowsOf(segs);
    if (ids.len != rows) return error.TooManyRows;
    const st = segs[0].st;
    if (rows > g.rows_max) { // wider than any window since the last drop: eager first, so scratch grows outside a capture
        eager(g, rows, false);
        return f.run(segs, ids);
    }
    var gf = f;
    gf.nch_fix = chunks(f, st, rows);
    try gf.upload(ids);
    const key: Key = .{ .arena = st.arena.buf.ptr, .buffers = f.b.arena.buf.ptr, .rows = @intCast(rows), .parity = if (st.cur.len > 0) st.cur[0] else 0, .nch = @intCast(gf.nch_fix) };
    var body: Body = .{ .f = &gf, .segs = segs, .rows = rows };
    _ = try g.cache.run(key, f.s, g.epoch, .{ .ctx = &body, .run = Body.run });
    st.pending = rows; // what stack() leaves on the host, which a replay does not run
    st.prompt = false;
}

/// The chunk count a graph of `rows` rows at `st`'s position runs (0 without MLA layers): a multiple of `bucket`.
fn chunks(f: fw.Forward, st: *const fw.State, rows: usize) usize {
    if (f.g.mla_layers == 0) return 0;
    return @min(f.b.nch, std.mem.alignForward(usize, latent.chunksFor(st.pos + rows), bucket));
}

/// One chained MTP draft of `tok` beside `hidden`'s row, committed, then norm and head into f.b.logits (graphed if so).
pub fn chain(graphs: ?*Graphs, h: *const head.Head, f: fw.Forward, hd: *fw.State, tok: []const i32, hidden: u64) !void {
    if (tok.len != 1) return error.BadRows;
    try f.upload(tok);
    if (graphs) |g| if (g.rows_max == 0) eager(g, 1, false) else if (fw.splitOf(f.g.w) == null) {
        var gf = f;
        gf.nch_fix = chunks(f, hd, 1);
        const C = struct {
            f: *const fw.Forward,
            h: *const head.Head,
            hd: *fw.State,
            hidden: u64,
            fn run(ctx: *anyopaque, s: cuda.Stream) anyerror!void {
                const c: *const @This() = @ptrCast(@alignCast(ctx));
                _ = s;
                try c.f.embedRows(1);
                try c.h.run(c.f.*, c.hd, 1, c.hidden, null);
                try c.f.final(1);
                try c.f.logitsOf(0, 1);
            }
        };
        var body: C = .{ .f = &gf, .h = h, .hd = hd, .hidden = hidden };
        const key: Key = .{ .arena = hd.arena.buf.ptr, .buffers = f.b.arena.buf.ptr, .rows = 1, .parity = @as(u32, @intCast(hidden - f.b.normed)) | (1 << 31), .nch = @intCast(gf.nch_fix) };
        _ = try g.cache.run(key, f.s, g.epoch, .{ .ctx = &body, .run = C.run });
        hd.pending = 1;
        hd.prompt = false;
        return f.commit(hd, 1);
    };
    try f.embedRows(1);
    try h.step(f, hd, 1, hidden, null);
    try f.final(1);
    try f.logitsOf(0, 1);
}
