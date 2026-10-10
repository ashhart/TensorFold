//! The Nemotron CUDA engine (engine.py): weights, kernels and one sequence's buffers; prefill, windows and commits.

const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const Forward = @import("cuda_forward.zig").Forward;
const Seg = @import("cuda_forward.zig").Seg;
const Dump = @import("cuda_dump.zig").Dump;
const Head = @import("cuda_mtp.zig").Head;
const head_fields = @import("cuda_mtp.zig").seq_fields;
const sampler = @import("cuda_sampler.zig");
const segs = @import("cuda_segments.zig");
const grid = @import("cuda_prompt_grid.zig");
const heat = @import("heat");
const seq_graphs = @import("cuda_seq_graphs.zig");
const sites = @import("cuda_sites.zig");

/// nemotron_h.cuda.CONTEXT: prompt plus reply tokens when --context is not given, as `tensorfold serve` sizes it.
pub const default_context = 16384;

pub const Options = struct {
    context: ?usize = null,
    mtp: bool = true,
    graphs: bool = true,
    sampling: ?sampler.Sampling = null, // the rule the graphs compile in; null or temperature 0 decodes greedily
    segments: usize = 1, // whole prompt chunks a call runs as staggered segments (1: one chunk at a time)
    carveout: ?*cuda.Carveout = null, // KV planes go to the display carveout first while it holds them
    slide: bool = false, // Sliding Weights: every layer's change in the forward before its graphs are captured
};

/// Asked before each prompt chunk (or segmented call): true stops the prompt with error.Cancelled.
pub const Cancel = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque) bool,

    pub fn now(c: ?Cancel) bool {
        const x = c orelse return false;
        return x.check(x.ptr);
    }
};

/// Grid points where a prompt pass copies its state (ascending), called once each chunk ending there is queued.
pub const Keep = struct {
    at: []const u32,
    ctx: *anyopaque,
    call: *const fn (*anyopaque, u32) void,
};

/// Serial rounds a host keeps queued ahead of the one it reads.
pub const lookahead = 4;

/// Streams a shared window holds at most (their rows together at most state.max_rows).
pub const max_streams = 8;

// pinned words: the window's meta row, its ids, its sampled ids, a prompt chunk's ids, then 64 a shared window's stream
const pin_meta = 0;
const pin_ids = 4;
const pin_sampled = 32;
const pin_prompt = 64;
const pin_shared = pin_prompt + segs.MAX * state.prefill_rows;

/// One stream's part of a shared window: its sequence, the ids its first rows take from the host, and its rows.
pub const Shared = struct { seq: *state.Seq, ids: []const u32, rows: usize };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    side: @import("cuda_forward.zig").Side = undefined, // the decode MoE's shared expert runs here
    k: kern.Kernels,
    w: weights.Weights,
    b: state.Buffers,
    c: Config,
    max_len: usize,
    nch: usize = 0, // attention chunk programs a row: a captured _chunk variant's NCH covering max_len (bits ignore it)
    pinned: cuda.HostBuffer,
    history: cuda.HostBuffer, // mapped: a serial round's kernel writes its token here, by position
    history_dev: u64 = 0,
    serial: ?cuda.graph.Exec = null,
    windows: [2][state.max_rows + 1]?cuda.graph.Exec = @splat(@splat(null)), // [greedy, sampled]: argmax or sample.cu
    stats: seq_graphs.Stats = .{},
    done: [lookahead]cuda.Event = undefined,
    copied: cuda.Event = undefined, // the last window's uploads have read the pinned words
    sampled_ready: cuda.Event = undefined,
    pos: usize = 0,
    parity: usize = 0,
    prev_keep: usize = 0,
    rows: usize = 0,
    sampling: ?sampler.Sampling = null,
    own: state.Seq = undefined, // the engine's own buffers, which the graphs were captured on
    bound: *state.Seq = undefined, // the sequence the calls act on
    head: ?*Head = null,
    load_seconds: f64 = 0,
    segments: usize = 1, // Options.segments
    seg: ?segs.Segments = null, // their streams and scratch, made at load (or when setSegments asks for more)
    carve: ?*cuda.Carveout = null, // Options.carveout; it outlives the engine
    heat_gate: heat.Gate = .{}, // off unless TF_HEAT_HIGH and TF_HEAT_LOW are set
    slide: ?sites.Sites = null, // --slide: the change's sites, attached at load, until the trainer takes them
    dir: ?[]u8 = null, // --slide: the model folder lessons are saved into

    /// Loads the checkpoint into the Python engine's layouts and sizes the caches; `triton_dir` null: our own glue.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, model_dir: []const u8, triton_dir: ?[]const u8, opts: Options) !*Engine {
        const t0 = std.Io.Clock.awake.now(io);
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .io = io, .ctx = ctx, .stream = undefined, .k = undefined, .w = undefined, .b = undefined, .c = undefined, .max_len = 0, .pinned = undefined, .history = undefined };
        e.c = try Config.read(gpa, io, model_dir);
        if (opts.segments < 1 or opts.segments > segs.MAX) return error.BadSegments;
        e.segments = opts.segments;
        e.carve = opts.carveout;
        const slots = (opts.context orelse default_context) + state.max_rows;
        e.max_len = (slots + state.chunk_keys - 1) / state.chunk_keys * state.chunk_keys;
        e.stream = try cuda.Stream.init(ctx.d, true);
        errdefer e.stream.deinit();
        e.side = .{ .s = try cuda.Stream.init(ctx.d, true), .fork = try cuda.Event.init(ctx.d, false), .join = try cuda.Event.init(ctx.d, false) };
        errdefer {
            e.side.s.deinit();
            e.side.fork.deinit();
            e.side.join.deinit();
        }
        e.k = try kern.Kernels.load(gpa, io, ctx, triton_dir);
        errdefer e.k.deinit();
        const chunks = e.max_len / state.chunk_keys;
        e.nch = if (e.k.triton) |*t| @intCast(t.smallestConst("_chunk", "NCH", @intCast(chunks)) orelse {
            std.log.err("no captured attention kernel covers {d} chunks of 512 keys; capture one for this context", .{chunks});
            return error.ContextUnsupported;
        }) else chunks;
        e.w = try weights.load(gpa, io, e.ops(), model_dir, e.c, opts.mtp);
        errdefer e.w.deinit();
        e.b = try state.Buffers.init(ctx.d, e.c, e.max_len, e.nch, e.carve);
        errdefer e.b.deinit();
        try e.b.initShared(e.ops(), e.c);
        if (e.segments > 1) e.seg = try segs.Segments.init(e, e.segments);
        errdefer if (e.seg) |*s| s.deinit();
        e.own = state.Seq.view(&e.b);
        e.bound = &e.own;
        e.pinned = try cuda.HostBuffer.alloc(ctx.d, (pin_shared + max_streams * 64) * 4);
        errdefer e.pinned.free();
        e.history = try cuda.HostBuffer.allocMapped(ctx.d, (@as(usize, e.max_len) + state.max_rows) * 4);
        errdefer e.history.free();
        e.history_dev = try e.history.device();
        for (&e.done) |*ev| ev.* = try cuda.Event.init(ctx.d, false);
        e.copied = try cuda.Event.init(ctx.d, false);
        e.sampled_ready = try cuda.Event.init(ctx.d, false);
        try e.copied.record(e.stream);
        // --slide: the change's kernels and sites last, so everything above sits where it would without them
        if (opts.slide) {
            try e.k.loadTrain();
            e.slide = try sites.Sites.init(gpa, ctx.d, e.c, state.prefill_rows);
            errdefer e.slide.?.deinit();
            e.dir = try gpa.dupe(u8, model_dir);
            e.slide.?.attach(&e.w, true);
        }
        errdefer if (e.slide) |*s| s.deinit();
        errdefer if (e.dir) |d| gpa.free(d);
        try e.setSampling(opts.sampling);
        if (opts.graphs) try e.capture(opts.mtp);
        e.heat_gate = heat.Gate.fromEnv() catch |err| {
            std.log.err("TF_HEAT_HIGH and TF_HEAT_LOW: both or neither, in degrees, with low at or under high", .{});
            return err;
        };
        e.load_seconds = seconds(io, t0);
        return e;
    }

    /// The serial round (window plus feed) and, for drafted rounds, a verify window of each row count, each one graph.
    fn capture(e: *Engine, windows: bool) !void {
        try e.reset();
        try e.stream.synchronize();
        e.serial = try e.record(.{ .window = .{ .rows = 1, .feed = true } });
        if (windows) try e.captureWindows();
    }

    /// The verify windows of the bound sequence's draw mode (greedy or sampled), each one graph.
    pub fn captureWindows(e: *Engine) !void {
        const set = &e.windows[@intFromBool(e.sampling != null)];
        for (1..state.max_rows + 1) |r| set[r] = try e.record(.{ .window = .{ .rows = r, .feed = false } });
    }

    /// What a graph runs: a verify window (with the serial feed or the sampled download), or one MTP head level.
    pub const Body = union(enum) {
        window: struct { rows: usize, feed: bool },
        head: struct { h: *Head, rows: usize, j: usize },
    };

    /// One graph of `body` on the bound sequence's buffers, instantiated and uploaded.
    pub fn record(e: *Engine, body: Body) !cuda.graph.Exec {
        const t0 = std.Io.Clock.awake.now(e.io);
        try cuda.graph.beginCapture(e.stream, .thread_local);
        e.recordBody(body) catch |err| {
            if (cuda.graph.endCapture(e.stream)) |g| {
                var x = g;
                x.deinit();
            } else |_| {}
            return err;
        };
        var g = try cuda.graph.endCapture(e.stream);
        defer g.deinit();
        const exec = try g.instantiate();
        try exec.upload(e.stream);
        try e.stream.synchronize();
        e.stats.captures += 1;
        e.stats.capture_ns += @intCast(std.Io.Clock.awake.now(e.io).toNanoseconds() - t0.toNanoseconds());
        return exec;
    }

    fn recordBody(e: *Engine, body: Body) !void {
        switch (body) {
            .window => |w| {
                try e.forward(null).window(w.rows);
                if (w.feed) return e.ops().serialFeed(e.b.sampled, e.b.ids, e.b.meta, e.history_dev);
                try e.ops().download(std.mem.sliceAsBytes(e.pinned.slice(u32)[pin_sampled..][0..w.rows]), e.b.sampled);
            },
            .head => |x| try x.h.level(x.rows, x.j),
        }
    }

    /// Step `id`: the own sequence launches `own`; another launches its own capture of `body`, made the first time.
    pub fn launchGraph(e: *Engine, own: cuda.graph.Exec, id: u32, body: Body) !void {
        if (e.bound == &e.own) return own.launchOn(e.stream);
        const gop = try e.bound.graphs.getOrPut(e.gpa, id);
        if (!gop.found_existing) gop.value_ptr.* = e.record(body) catch |err| {
            _ = e.bound.graphs.remove(id);
            return err;
        };
        e.stats.replays += 1;
        return gop.value_ptr.launchOn(e.stream);
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        if (e.serial) |*g| g.deinit();
        for (&e.windows) |*set| for (set) |*g| if (g.*) |*x| x.deinit();
        if (e.seg) |*s| s.deinit();
        for (&e.done) |*ev| ev.deinit();
        e.copied.deinit();
        e.sampled_ready.deinit();
        e.history.free();
        e.pinned.free();
        e.b.deinit();
        if (e.slide) |*s| s.deinit();
        if (e.dir) |d| e.gpa.free(d);
        e.w.deinit();
        e.k.deinit();
        e.side.s.deinit();
        e.side.fork.deinit();
        e.side.join.deinit();
        e.stream.deinit();
        e.gpa.destroy(e);
    }

    /// The bound sequence's sampling: its rule on the device, which the target and the head draw by (null: greedy).
    pub fn setSampling(e: *Engine, s: ?sampler.Sampling) !void {
        e.sampling = sampler.check(s);
        const x = e.sampling orelse return;
        const r = sampler.rule(x);
        try e.ops().upload(e.b.rule, std.mem.asBytes(&r));
    }

    /// A zeroed sequence for another stream (its own caches, state, head caches and settings).
    pub fn newSeq(e: *Engine) !*state.Seq {
        const s = try e.gpa.create(state.Seq);
        errdefer e.gpa.destroy(s);
        var head: [head_fields.len]usize = undefined;
        if (e.head) |h| head = h.seqSizes();
        s.* = try state.Seq.init(e.ctx.d, e.c, e.max_len, if (e.head != null) &head else &.{}, e.stream, e.carve);
        return s;
    }

    /// The device bytes newSeq allocates.
    pub fn seqBytes(e: *const Engine) usize {
        var head: [head_fields.len]usize = undefined;
        if (e.head) |h| head = h.seqSizes();
        return state.Seq.bytes(e.c, e.max_len, if (e.head != null) &head else &.{});
    }

    pub fn freeSeq(e: *Engine, s: *state.Seq) void {
        if (e.bound == s) e.bind(&e.own);
        seq_graphs.free(e.gpa, &s.graphs);
        s.deinit();
        e.gpa.destroy(s);
    }

    /// Act on sequence `s` from here on: its buffers replace the bound one's, which keeps where it stands.
    pub fn bind(e: *Engine, s: *state.Seq) void {
        const old = e.bound;
        old.* = .{ .arena = old.arena, .carved = old.carved, .ptr = old.ptr, .head = old.head, .graphs = old.graphs, .pos = e.pos, .parity = e.parity, .prev_keep = e.prev_keep, .rows = e.rows, .head_pos = if (e.head) |h| h.pos else 0, .sampling = e.sampling };
        inline for (state.seq_fields, s.ptr) |name, p| @field(e.b, name) = p;
        e.pos = s.pos;
        e.parity = s.parity;
        e.prev_keep = s.prev_keep;
        e.rows = s.rows;
        e.sampling = s.sampling;
        if (e.head) |h| h.bindSeq(s);
        e.bound = s;
    }

    pub fn ops(e: *const Engine) kern.Ops {
        return .{ .k = &e.k, .s = e.stream };
    }

    pub fn forward(e: *const Engine, dump: ?*Dump) Forward {
        var f = Forward.init(e.c, &e.w, &e.b, e.ops(), e.max_len, e.nch, e.sampling != null);
        f.dump = dump;
        if (dump == null and e.c.experts <= 128 and e.c.slots() <= 8) f.side = e.side; // tf_plan_routed's bounds
        return f;
    }

    /// Engine.reset: zeroed caches and state, position 0.
    pub fn reset(e: *Engine) !void {
        try e.b.reset(e.ops());
        e.pos = 0;
        e.parity = 0;
        e.prev_keep = 0;
    }

    /// decode.prefill: a fresh state, the prompt in 2048-row chunks (absorbed by the head, in segments if set); the token.
    pub fn prefill(e: *Engine, prompt: []const u32, dump: ?*Dump, head: ?*Head) !u32 {
        return e.prefillWith(prompt, dump, head, null);
    }

    /// prefill, stopping with error.Cancelled at the next chunk boundary once `cancel` says so.
    pub fn prefillWith(e: *Engine, prompt: []const u32, dump: ?*Dump, head: ?*Head, cancel: ?Cancel) !u32 {
        return e.prefillFrom(prompt, 0, dump, head, cancel, null);
    }

    /// prefill from `from` (0 resets; else a restored state at a grid point); chunks stay on the zero-anchored grid.
    pub fn prefillFrom(e: *Engine, prompt: []const u32, from: usize, dump: ?*Dump, head: ?*Head, cancel: ?Cancel, keep: ?Keep) !u32 {
        if (prompt.len == 0) return error.EmptyPrompt;
        if (from != 0 and !grid.resumable(from, prompt.len, state.prefill_rows)) return error.BadReuse;
        if (prompt.len + state.max_rows > e.max_len) return error.ContextFull;
        if (from == 0) {
            try e.reset();
            if (head) |h| try h.reset();
        } else if (e.pos != from) return error.BadReuse;
        const waited = e.heat_gate.waited_s;
        if (from == 0 and keep == null and e.segments > 1 and dump == null and prompt.len > state.prefill_rows) {
            try segs.prefill(e, try e.segmentSet(), prompt, head, e.segments, cancel);
        } else try e.serialChunks(prompt, from, dump, head, cancel, keep);
        if (e.heat_gate.waited_s > waited) heat.note(e.heat_gate.waited_s - waited);
        const host = e.pinned.slice(u32)[pin_sampled..][0..1];
        try e.ops().download(std.mem.sliceAsBytes(host), e.b.p_sampled);
        try e.stream.synchronize();
        return host[0];
    }

    /// Heat bands, if set, before the chunk; a cancel ends the wait. Unset bands return without reading a zone.
    pub fn beforeChunk(e: *Engine, cancel: ?Cancel) !void {
        try e.heat_gate.beforePromptChunk(e.io, if (cancel) |c| .{ .ptr = c.ptr, .check = c.check } else null);
    }

    /// The prompt's chunks one after another on the engine's stream, starting at `from`.
    fn serialChunks(e: *Engine, prompt: []const u32, from: usize, dump: ?*Dump, head: ?*Head, cancel: ?Cancel, keep: ?Keep) !void {
        const f = e.forward(dump);
        var s: usize = from;
        while (s < prompt.len) {
            if (Cancel.now(cancel)) return error.Cancelled;
            try e.beforeChunk(cancel);
            const end = grid.end(s, prompt.len, state.prefill_rows);
            const chunk = prompt[s..end];
            try e.copied.synchronize();
            const host = e.promptHost(chunk.len);
            @memcpy(host, chunk);
            try e.ops().upload(e.b.p_ids, std.mem.sliceAsBytes(host));
            try e.copied.record(e.stream);
            try f.chunk(@intCast(chunk.len), e.pos);
            e.pos += @intCast(chunk.len);
            const known = @min(chunk.len, prompt.len - 1 - s);
            if (head) |h| if (known > 0) try h.absorb(e.b.p_hidden, prompt[s + 1 ..][0..known]);
            s = end;
            if (keep) |k| if (s <= std.math.maxInt(u32) and std.mem.indexOfScalar(u32, k.at, @intCast(s)) != null) k.call(k.ctx, @intCast(s));
        }
    }

    /// Whole chunks a prompt call runs as staggered segments from here on; their streams and scratch are made now.
    pub fn setSegments(e: *Engine, n: usize) !void {
        if (n < 1 or n > segs.MAX) return error.BadSegments;
        e.segments = n;
        if (n > 1) _ = try e.segmentSet();
    }

    /// Pinned words for `n` prompt ids on their way up (free once `copied` has completed).
    pub fn promptHost(e: *Engine, n: usize) []u32 {
        return e.pinned.slice(u32)[pin_prompt..][0..n];
    }

    /// The segments' streams and scratch for calls of e.segments, remade when a call needs more of them.
    fn segmentSet(e: *Engine) !*segs.Segments {
        if (e.seg) |*s| {
            if (s.n >= e.segments) return s;
            try e.stream.synchronize();
            s.deinit();
            e.seg = null;
        }
        e.seg = try segs.Segments.init(e, e.segments);
        return &e.seg.?;
    }

    /// Engine.forward: a window of `rows` rows at pos whose first `ids.len` inputs come from the host (drafts are on the device).
    pub fn verify(e: *Engine, ids: []const u32, rows: usize, dump: ?*Dump) !void {
        if (rows < 1 or rows > state.max_rows or ids.len > rows) return error.BadWindow;
        if (e.pos + rows > e.max_len) return error.ContextFull;
        try e.copied.synchronize();
        const host = e.pinned.slice(u32);
        host[pin_meta..][0..4].* = e.metaRow();
        @memcpy(host[pin_ids..][0..ids.len], ids);
        try e.ops().upload(e.b.ids, std.mem.sliceAsBytes(host[pin_ids..][0..ids.len]));
        try e.ops().upload(e.b.meta, std.mem.sliceAsBytes(host[pin_meta..][0..4]));
        try e.copied.record(e.stream);
        const m: usize = @intFromBool(e.sampling != null);
        if (dump == null and e.windows[m][rows] != null) {
            try e.launchGraph(e.windows[m][rows].?, @intCast(1 + m * (state.max_rows + 1) + rows), .{ .window = .{ .rows = rows, .feed = false } });
        } else {
            e.stats.eager += 1;
            try e.forward(dump).window(rows);
            try e.ops().download(std.mem.sliceAsBytes(host[pin_sampled..][0..rows]), e.b.sampled);
        }
        try e.sampled_ready.record(e.stream);
        e.parity ^= 1;
        e.rows = rows;
    }

    /// Several streams' windows in one forward (eager): each stream's rows read its own caches, state and rule.
    pub fn verifyShared(e: *Engine, parts: []const Shared) !void {
        if (parts.len < 1 or parts.len > max_streams) return error.BadWindow;
        var total: usize = 0;
        for (parts) |p| {
            if (p.rows < 1 or p.ids.len < 1 or p.ids.len > p.rows) return error.BadWindow;
            total += p.rows;
        }
        if (total > state.max_rows) return error.WindowTooWide;
        try e.copied.synchronize();
        const host = e.pinned.slice(u32)[pin_shared..][0 .. max_streams * 64];
        var views: [max_streams]state.Buffers = undefined;
        var parts_f: [max_streams]Seg = undefined;
        var row0: usize = 0;
        for (parts, 0..) |p, k| {
            e.bind(p.seq);
            if (e.pos + p.rows > e.max_len) return error.ContextFull;
            const slot = host[k * 64 ..][0..64];
            slot[0..4].* = e.metaRow();
            @memcpy(slot[4..][0..p.ids.len], p.ids);
            try e.ops().upload(e.b.ids, std.mem.sliceAsBytes(slot[4..][0..p.ids.len]));
            try e.ops().upload(e.b.meta, std.mem.sliceAsBytes(slot[0..4]));
            views[k] = e.b; // the bound sequence's own fields beside the shared scratch
            parts_f[k] = .{ .b = &views[k], .row0 = row0, .rows = p.rows, .sampled = e.sampling != null };
            row0 += p.rows;
        }
        try e.copied.record(e.stream);
        try e.forward(null).windowSegs(parts_f[0..parts.len]);
        for (parts, 0..) |p, k| try e.ops().download(std.mem.sliceAsBytes(host[k * 64 + 32 ..][0..p.rows]), views[k].sampled);
        try e.sampled_ready.record(e.stream);
        for (parts) |p| {
            e.bind(p.seq);
            e.parity ^= 1;
            e.rows = p.rows;
        }
    }

    /// Part k's drawn ids from the last shared window (row r: the token at its position + r + 1).
    pub fn sharedTokens(e: *Engine, k: usize, rows: usize) ![]const u32 {
        try e.sampled_ready.synchronize();
        return e.pinned.slice(u32)[pin_shared + k * 64 + 32 ..][0..rows];
    }

    /// The window's meta row as the kernels read it: position, buffer parity, previous keep, a spare word.
    fn metaRow(e: *const Engine) [4]u32 {
        return .{ @intCast(e.pos), @intCast(e.parity), @intCast(e.prev_keep), 0 };
    }

    /// Engine.tokens: the last window's sampled ids (row r: the token at position pos + r + 1).
    pub fn tokens(e: *Engine) ![]const u32 {
        try e.sampled_ready.synchronize();
        return e.pinned.slice(u32)[pin_sampled..][0..e.rows];
    }

    /// Engine.commit: keep the first `keep` rows of the last window.
    pub fn commit(e: *Engine, keep: usize) !void {
        if (keep < 1 or keep > e.rows) return error.BadKeep;
        e.pos += keep;
        e.prev_keep = keep;
    }

    /// The first serial window's inputs for the device-fed rounds: meta row and token.
    pub fn upload(e: *Engine, token: u32) !void {
        try e.copied.synchronize();
        const host = e.pinned.slice(u32);
        host[pin_meta..][0..4].* = e.metaRow();
        host[pin_ids] = token;
        try e.ops().upload(e.b.meta, std.mem.sliceAsBytes(host[pin_meta..][0..4]));
        try e.ops().upload(e.b.ids, std.mem.sliceAsBytes(host[pin_ids..][0..1]));
        try e.copied.record(e.stream);
    }

    /// The token a serial round wrote for `position` (visible once that round's event has completed).
    pub fn written(e: *const Engine, position: usize) u32 {
        const p: *volatile u32 = &e.history.slice(u32)[position];
        return p.*;
    }

    /// One serial window of `token` at the current position; returns its sampled token and commits it.
    pub fn step(e: *Engine, token: u32, dump: ?*Dump) !u32 {
        try e.verify(&.{token}, 1, dump);
        const tok = (try e.tokens())[0];
        try e.commit(1);
        return tok;
    }
};

pub fn seconds(io: std.Io, since: std.Io.Timestamp) f64 {
    const now = std.Io.Clock.awake.now(io);
    return @as(f64, @floatFromInt(now.toNanoseconds() - since.toNanoseconds())) / 1e9;
}
