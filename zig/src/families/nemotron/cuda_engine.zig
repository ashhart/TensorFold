//! The Nemotron CUDA engine (engine.py): weights, kernels and one sequence's buffers; windows and commits.

const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const Forward = @import("cuda_forward.zig").Forward;
const Seg = @import("cuda_forward.zig").Seg;
pub const Mask = @import("cuda_forward.zig").Mask;
const Dump = @import("cuda_dump.zig").Dump;
const Head = @import("cuda_mtp.zig").Head;
const head_fields = @import("cuda_mtp.zig").seq_fields;
const sampler = @import("cuda_sampler.zig");
const segs = @import("cuda_segments.zig");
const train_ops = @import("cuda_train_ops.zig");
const lesson_route = @import("lesson_route.zig");
const heat = @import("heat");
const seq_graphs = @import("cuda_seq_graphs.zig");
const prompt_rows = @import("cuda_prompt_rows.zig");
const prompt_pass = @import("cuda_prompt.zig");
const marks = @import("cuda_marks.zig");
pub const PromptRows = prompt_rows.PromptRows;

/// nemotron_h.cuda.CONTEXT: prompt plus reply tokens when --context is not given, as `tensorfold serve` sizes it.
pub const default_context = 16384;

pub const Options = struct {
    context: ?usize = null,
    mtp: bool = true,
    graphs: bool = true,
    sampling: ?sampler.Sampling = null, // the rule the graphs compile in; null or temperature 0 decodes greedily
    segments: usize = 1, // whole prompt chunks a call runs as staggered segments (1: one chunk at a time)
    carveout: ?*cuda.Carveout = null, // KV planes go to the display carveout first while it holds them
    slide: bool = false, // Sliding Weights asked for: refused for a checkpoint the learner cannot read
};

/// Asked before each prompt chunk (or segmented call): true stops the prompt with error.Cancelled.
pub const Cancel = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque) bool,
    yields: bool = false, // a background pass's hook: it steps aside for real work, so its chunks stay short

    pub fn now(c: ?Cancel) bool {
        const x = c orelse return false;
        return x.check(x.ptr);
    }

    /// A background pass: chunks of state.yield_rows, one in flight, and what it prefilled kept when it stops.
    pub fn background(c: ?Cancel) bool {
        return if (c) |x| x.yields else false;
    }
};

/// Rows where a prompt pass keeps its state (ascending): one inside a chunk is called before it, one at its end after.
pub const Keep = struct {
    at: []const u32,
    ctx: *anyopaque,
    call: *const fn (*anyopaque, u32) void,
    cut: bool = false, // every row ends a chunk instead: the reference the marks test compares the armed states to
};

/// Serial rounds a host keeps queued ahead of the one it reads.
pub const lookahead = 4;

/// Streams a shared window holds at most (their rows together at most state.max_rows).
pub const max_streams = 8;

// pinned words: the window's meta row, its ids, its sampled ids, a prompt chunk's ids, then 64 a shared window's stream
const pin_meta = 0;
const pin_ids = 4;
pub const pin_sampled = 32;
pub const pin_prompt = 64;
const pin_shared = pin_prompt + @max(segs.MAX * state.prefill_rows, state.pass_rows);
/// Graph ids of masked verify windows (launchGraph's ids: windows from 1, head levels from 1000).
const masked_ids = 4000;

/// One stream's part of a shared window: its sequence, the ids its first rows take from the host, and its rows.
pub const Shared = struct { seq: *state.Seq, ids: []const u32, rows: usize, mask: Mask = .{} };

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
    epoch: u32 = 0, // cuda_slide's recapture moves it on: other sequences capture their graphs again
    mask_bits: u64 = 0, // the grammar bits a masked window's graphs read (one place for every single-stream window)
    done: [lookahead]cuda.Event = undefined,
    copied: cuda.Event = undefined, // the last window's uploads have read the pinned words
    sampled_ready: cuda.Event = undefined,
    pos: usize = 0,
    parity: usize = 0,
    prev_keep: usize = 0,
    rows: usize = 0,
    sampling: ?sampler.Sampling = null,
    own: state.Seq = undefined, // the engine's own buffers, which the graphs were captured on
    slots: u64 = 1 | (@as(u64, 1) << lesson_route.trainer_slot), // lesson latch slots in use: the own sequence's, the trainer's
    slot: u32 = 0, // the bound sequence's lesson latch slot
    decide: i64 = lesson_route.none, // the bound sequence's decision row
    trace: bool = false, // TF_SLIDE_TRACE: each reply's lesson scores logged (dev)
    bound: *state.Seq = undefined, // the sequence the calls act on
    head: ?*Head = null,
    load_seconds: f64 = 0,
    segments: usize = 1, // Options.segments
    seg: ?segs.Segments = null, // their streams and scratch, made at load (or when setSegments asks for more)
    carve: ?*cuda.Carveout = null, // Options.carveout; it outlives the engine
    heat_gate: heat.Gate = .{}, // off unless TF_HEAT_HIGH and TF_HEAT_LOW are set
    prompt_rows: ?PromptRows = null, // a prompt pass's rows for prompt_logprobs (serial chunks, from 0)
    armed: marks.Armed = .{}, // the marks inside the prompt chunk about to run (cuda_marks.zig)
    spares: marks.Spares = .{}, // free Mamba buffers the prompt cache set aside for the next marks
    last_chunk: usize = 0, // where the last prompt pass's last chunk started (the prompt's last row: len - 1 - this)

    /// Loads the checkpoint into the Python engine's layouts and sizes the caches; `triton_dir` null: our own glue.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, model_dir: []const u8, triton_dir: ?[]const u8, opts: Options) !*Engine {
        const t0 = std.Io.Clock.awake.now(io);
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .io = io, .ctx = ctx, .stream = undefined, .k = undefined, .w = undefined, .b = undefined, .c = undefined, .max_len = 0, .pinned = undefined, .history = undefined };
        e.trace = std.c.getenv("TF_SLIDE_TRACE") != null;
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
        if (e.c.format == .modelopt) {
            if (opts.slide) return error.SlideNeedsMlx; // the learner reads MLX 4-bit projections
            try e.k.loadModelopt(ctx);
        }
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
        window: struct { rows: usize, feed: bool, mask: Mask = .{} },
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
                const f = e.forward(null);
                try f.windowSegs(&.{.{ .b = f.b, .row0 = 0, .rows = w.rows, .sampled = f.sampled, .mask = w.mask, .slot = f.slot, .untaught = f.untaught }});
                if (w.feed) return e.ops().serialFeed(e.b.sampled, e.b.ids, e.b.meta, e.history_dev);
                try e.ops().download(std.mem.sliceAsBytes(e.pinned.slice(u32)[pin_sampled..][0..w.rows]), e.b.sampled);
            },
            .head => |x| try x.h.level(x.rows, x.j),
        }
    }

    /// Step `id`: the own sequence launches `own`; another launches its own capture of `body`, made the first time.
    pub fn launchGraph(e: *Engine, own: cuda.graph.Exec, id: u32, body: Body) !void {
        if (e.bound == &e.own) return own.launchOn(e.stream);
        return e.launchCaptured(id, body);
    }

    /// Step `id` from the bound sequence's own capture of `body`, made the first time (and again after a recapture).
    fn launchCaptured(e: *Engine, id: u32, body: Body) !void {
        if (e.bound.epoch != e.epoch) {
            seq_graphs.free(e.gpa, &e.bound.graphs);
            e.bound.epoch = e.epoch;
        }
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
        e.spares.deinit();
        if (e.serial) |*g| g.deinit();
        seq_graphs.free(e.gpa, &e.own.graphs); // masked windows'
        for (&e.windows) |*set| for (set) |*g| if (g.*) |*x| x.deinit();
        if (e.seg) |*s| s.deinit();
        for (&e.done) |*ev| ev.deinit();
        e.copied.deinit();
        e.sampled_ready.deinit();
        e.history.free();
        e.pinned.free();
        e.b.deinit();
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
        const free = @ctz(~e.slots);
        if (free >= lesson_route.max_slots) return error.TooManySequences;
        e.slots |= @as(u64, 1) << @intCast(free);
        s.slot = @intCast(free);
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
        e.slots &= ~(@as(u64, 1) << @intCast(s.slot));
        s.deinit();
        e.gpa.destroy(s);
    }

    /// Act on sequence `s` from here on: its buffers replace the bound one's, which keeps where it stands.
    pub fn bind(e: *Engine, s: *state.Seq) void {
        const old = e.bound;
        old.* = .{ .arena = old.arena, .carved = old.carved, .ptr = old.ptr, .head = old.head, .graphs = old.graphs, .epoch = old.epoch, .pos = e.pos, .parity = e.parity, .prev_keep = e.prev_keep, .rows = e.rows, .head_pos = if (e.head) |h| h.pos else 0, .sampling = e.sampling, .slot = old.slot, .decide = e.decide, .untaught = old.untaught };
        inline for (state.seq_fields, s.ptr) |name, p| @field(e.b, name) = p;
        e.pos = s.pos;
        e.parity = s.parity;
        e.prev_keep = s.prev_keep;
        e.rows = s.rows;
        e.sampling = s.sampling;
        e.slot = s.slot;
        e.decide = s.decide;
        if (e.head) |h| h.bindSeq(s);
        e.bound = s;
    }

    pub fn ops(e: *const Engine) kern.Ops {
        return .{ .k = &e.k, .s = e.stream, .part = e.b.lane_part };
    }

    pub fn forward(e: *const Engine, dump: ?*Dump) Forward {
        var f = Forward.init(e.c, &e.w, &e.b, e.ops(), e.max_len, e.nch, e.sampling != null);
        f.dump = dump;
        f.slot = e.slot;
        f.untaught = e.bound.untaught;
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

    // the prompt pass, in cuda_prompt.zig
    pub const prefill = prompt_pass.prefill;
    pub const prefillWith = prompt_pass.prefillWith;
    pub const prefillFrom = prompt_pass.prefillFrom;
    pub const redrawFirst = prompt_pass.redrawFirst;
    pub const beforeChunk = prompt_pass.beforeChunk;
    pub const serialChunks = prompt_pass.serialChunks;
    pub const setSegments = prompt_pass.setSegments;
    pub const promptHost = prompt_pass.promptHost;
    pub const segmentSet = prompt_pass.segmentSet;

    /// Any block's change, for the buffers every block shares (null: no lesson is attached).
    pub fn lessons(e: *const Engine) ?weights.Adapter {
        for (e.w.blocks) |blk| switch (blk.kind) {
            .mamba => if (blk.mamba.adapter) |ad| return ad,
            .attention => if (blk.attn.adapter) |ad| return ad,
            .moe => if (blk.moe.adapter) |ad| return ad,
        };
        return null;
    }

    /// Dev trace (TF_SLIDE_TRACE): the bound sequence's router scores at its decision row and which lessons act on it.
    pub fn traceLessons(e: *const Engine) void {
        const ad = e.lessons() orelse return;
        if (e.decide == lesson_route.none) return;
        const n = lesson_route.max_blocks;
        var score: [n]f32 = undefined;
        var need: [n]f32 = undefined;
        var flags: [n]u32 = undefined;
        const o = e.ops();
        o.download(std.mem.sliceAsBytes(&score), ad.score + e.slot * n * 4) catch return;
        o.download(std.mem.sliceAsBytes(&need), ad.need) catch return;
        o.download(std.mem.sliceAsBytes(&flags), ad.latch + e.slot * n * 4) catch return;
        e.stream.synchronize() catch return;
        var line: [2048]u8 = undefined;
        var w: std.Io.Writer = .fixed(&line);
        for (0..ad.rank / 16) |b| w.print(" {d:.2}/{d:.2}{s}", .{ score[b], need[b], if (flags[b] != 0) "*" else "" }) catch break;
        std.log.info("slide trace: decision row {d}: scores{s} (* acting)", .{ e.decide, w.buffered() });
    }

    pub fn trainOps(e: *const Engine) train_ops.Ops {
        return .{ .k = &e.k.train, .s = e.stream };
    }

    /// Engine.forward: a window of `rows` rows at pos, its first `ids.len` inputs from the host (drafts on the device).
    pub fn verify(e: *Engine, ids: []const u32, rows: usize, dump: ?*Dump) !void {
        return e.verifyMasked(ids, rows, dump, .{});
    }

    /// verify with a grammar's bits for each row: its logits masked before its draws, a graph each row count once made.
    pub fn verifyMasked(e: *Engine, ids: []const u32, rows: usize, dump: ?*Dump, mask: Mask) !void {
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
        if (dump == null and mask.bits == 0 and e.windows[m][rows] != null) {
            try e.launchGraph(e.windows[m][rows].?, @intCast(1 + m * (state.max_rows + 1) + rows), .{ .window = .{ .rows = rows, .feed = false } });
        } else if (dump == null and e.windows[m][rows] != null and e.maskGraphs(mask)) {
            try e.launchCaptured(@intCast(masked_ids + m * (state.max_rows + 1) + rows), .{ .window = .{ .rows = rows, .feed = false, .mask = mask } });
        } else {
            e.stats.eager += 1;
            const f = e.forward(dump);
            try f.windowSegs(&.{.{ .b = f.b, .row0 = 0, .rows = rows, .sampled = f.sampled, .mask = mask, .slot = f.slot, .untaught = f.untaught }});
            try e.ops().download(std.mem.sliceAsBytes(host[pin_sampled..][0..rows]), e.b.sampled);
        }
        try e.sampled_ready.record(e.stream);
        e.parity ^= 1;
        e.rows = rows;
    }

    /// Whether masked windows replay graphs: their bits stay where the first one read them (the host rewrites them).
    fn maskGraphs(e: *Engine, mask: Mask) bool {
        if (e.mask_bits == 0) e.mask_bits = mask.bits;
        return mask.bits == e.mask_bits;
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
            parts_f[k] = .{ .b = &views[k], .row0 = row0, .rows = p.rows, .sampled = e.sampling != null, .mask = p.mask, .slot = p.seq.slot, .untaught = p.seq.untaught };
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
