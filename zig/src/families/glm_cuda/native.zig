//! GLM-5.3-Flash on two GB10s: rank 0's lane backend sends each call to rank 1 to replay; tokens from both shards.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const api = @import("api.zig");
const weights = @import("weights.zig");
const forward = @import("forward.zig");
const decode = @import("decode.zig");
const proto = @import("proto.zig");
const Guard = @import("guard.zig").Guard;
const copy = @import("copy.zig");
const dflash = @import("dflash.zig");
const native_follow = @import("native_follow.zig");
const native_prompt = @import("native_prompt.zig");
const native_draft = @import("native_draft.zig");
const native_env = @import("native_env.zig");
const be = lanes.backend;

pub const model_type = "glm5_next";
pub const formats: []const []const u8 = &.{ "glm-affine-experts-v1", "exl3-mcg4-routed-experts" };
pub const default_context: i64 = 1048576;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = 2048;
pub const two_ranks = true;
/// Its tower takes GLM's image preparation and placeholders (the server's qwen_image.glm).
pub const vision_glm = true;
const vision = weights.vision;

pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    tp: u8 = 2,
    rank: u8 = 0,
    master: ?[]const u8 = null,
    master_port: u16 = 29600,
    vision: bool = false, // load the vision tower (rank 0) and take image and video prompts
};

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool; // v1.0.11: `resumed` (the lane host restored the prompt cache entry)

pub const Loaded = struct {
    backend: be.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 0, // every admitted stream takes its own caches, sized to its request
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    follow: ?*const fn (*anyopaque) anyerror!void = null,
};

/// Device bytes this rank's half of the weights takes.
pub fn deviceWeightBytes(io: std.Io, dir: []const u8) u64 {
    return weights.deviceBytes(io, dir) catch 92 << 30;
}

pub const max_rows: usize = @intCast(forward.MAX_ROWS); // a stream's verify window at most (pending + 15 drafts)
pub const max_depth = max_rows - 1;
pub const max_streams = forward.MAX_STREAMS;
/// The head stops at a draft whose chain probability is below this (Python's rule); TF_GLM_DRAFT_CONFIDENCE.
pub const draft_confidence: f64 = 0.7;

const marginalFromEnv = native_env.marginalFromEnv;
const confidenceFromEnv = native_env.confidenceFromEnv;
const windowFromEnv = native_env.windowFromEnv;
const envUsize = native_env.envUsize;
const drafterWanted = native_env.drafterWanted;
const dflashStepMs = native_env.dflashStepMs;
const roceWanted = native_env.roceWanted;
const roceMaxBytes = native_env.roceMaxBytes;
pub const msSince = native_env.msSince;

/// TF_GLM_TRACE=1 (rank 0): every 256 rounds, the rounds' mean verify and draft times and rows, logged.
const Trace = struct {
    on: bool = false,
    rounds: u64 = 0,
    verify_ms: f64 = 0,
    verify_rows: u64 = 0,
    verify_streams: u64 = 0,
    draft_ms: f64 = 0,
    drafts: u64 = 0,
    calls: u64 = 0,
    copies: u64 = 0, // draft rounds that took a copy over the head's drafts
    df_rounds: u64 = 0, // streams' rounds DFlash2 drafted (the drafter mix)
    drafts_calls_total: u64 = 0, // streams' draft rounds
    copy_tokens: u64 = 0,
    desync: u64 = 0, // windows whose pending token was not the copy history's last (a history out of step)
    predicted: f64 = 0, // the verified drafts' chances summed (as chained products) and the drafts they landed
    landed: u64 = 0,
    verified: u64 = 0,

    fn log(t: *Trace, own: *const Owned) void {
        const n: f64 = @floatFromInt(@max(t.rounds, 1));
        const c: f64 = @floatFromInt(@max(t.calls, 1));
        std.log.info("glm trace: {d} verifies, {d:.2} ms each, {d:.2} rows ({d:.2} streams); {d} drafts, {d:.2} ms each, {d:.2} drafted a stream", .{ t.rounds, t.verify_ms / n, @as(f64, @floatFromInt(t.verify_rows)) / n, @as(f64, @floatFromInt(t.verify_streams)) / n, t.calls, t.draft_ms / c, @as(f64, @floatFromInt(t.drafts)) / @as(f64, @floatFromInt(@max(t.verify_streams, 1))) });
        std.log.info("glm drafts: {d} verified, {d} landed ({d:.2}), their chances said {d:.2}", .{ t.verified, t.landed, @as(f64, @floatFromInt(t.landed)) / @as(f64, @floatFromInt(@max(t.verified, 1))), t.predicted / @as(f64, @floatFromInt(@max(t.verified, 1))) });
        std.log.info("glm copies: {d} of {d} draft rounds copied ({d:.2} tokens each), copied drafts land {d:.2}, the drafter's {d:.2}; {d} out of step", .{ t.copies, t.calls, @as(f64, @floatFromInt(t.copy_tokens)) / @as(f64, @floatFromInt(@max(t.copies, 1))), own.copy_rate.chance(), own.draft_rate.chance(), t.desync });
        {
            const ds = &decode.draw_stats;
            const dc: f64 = @floatFromInt(@max(ds.calls, 1));
            std.log.info("glm draws: {d} sampled verifies, {d:.2} ms gpu + {d:.2} ms host each, {d} rows, {d} full rows, {d} check mismatches", .{ ds.calls, ds.gpu_ms / dc, ds.host_ms / dc, ds.rows, ds.full, ds.mismatch });
            ds.* = .{};
        }
        if (own.e.drafter) |dr| {
            const nc: f64 = @floatFromInt(@max(dr.calls, 1));
            std.log.info("glm mix: {d} stream rounds by DFlash2, {d} by the MTP head", .{ t.df_rounds, t.drafts_calls_total - t.df_rounds });
            std.log.info("glm dflash2: {d} rounds, {d:.2} ms each (context {d:.2}, tables {d:.2}, layers {d:.2}, head {d:.2}, selector {d:.2}); {d} streams' contexts restarted", .{ dr.calls, dr.forward_ms / nc, dr.phase_ms[0] / nc, dr.phase_ms[4] / nc, dr.phase_ms[1] / nc, dr.phase_ms[2] / nc, dr.phase_ms[3] / nc, dr.lost });
            dflash.logFine(dr, dr.calls);
            dr.calls = 0;
            dr.forward_ms = 0;
            dr.phase_ms = @splat(0);
        }
        const g: f64 = 1 << 30;
        const avail: u64 = if (own.guard) |gd| gd.available.load(.monotonic) else 0;
        const low: u64 = if (own.guard) |gd| gd.lowest.load(.monotonic) else 0;
        std.log.info("glm memory: {d:.2} GiB available (lowest {d:.2}); pool {d:.2} of {d:.2} GiB; {d} graphs ({d} captured, {d} updated in place, {d} dropped, {d:.0} ms capturing); device {d:.2} GiB", .{ @as(f64, @floatFromInt(avail)) / g, @as(f64, @floatFromInt(low)) / g, @as(f64, @floatFromInt(own.used)) / g, @as(f64, @floatFromInt(own.room)) / g, own.e.graphs.count(), own.e.captures, own.e.updates, own.e.evictions, own.e.capture_ms, @as(f64, @floatFromInt(cuda.usage(false).device)) / g });
        t.* = .{ .on = true };
    }
};
/// A stream's caches for admission: a small request's slots; requests take what they need from the pool (``fits``).
const admission_slots: i64 = 32768;
/// Prompt tokens a stream's copy index covers at most (its last; TF_GLM_COPY on): about 24 MiB of host memory.
pub const copy_index: usize = 1 << 20;

const Op = proto.Op;
const Writer = proto.Writer;
const Reader = proto.Reader;

pub const Lane = struct {
    id: u64,
    s: *forward.Seq,
    held: [max_depth]u32 = undefined, // drafts held for the next window
    probs: [max_depth]f64 = undefined, // the head's chance for each (0 past the confidence cut)
    nheld: u32 = 0,
    rows: u32 = 0, // the last verify's rows, not yet committed (0: nothing pending)
    first: u64 = 0, // the prompt's first token (a handle: readFn)
    media: bool = false, // the prompt had images (its kept state is its own)
    prompt: []u32 = &.{}, // the prompt (rank 0: the kept state's prefix when released)
    copy: ?copy.Copy = null, // rank 0: the history copy drafts come from (copy.zig)
    copied: u32 = 0, // copied drafts held for the next window
    copy_verified: u32 = 0, // of them, the ones the last window verified
    draft_verified: u32 = 0, // the last window's verified drafts when the drafter's alone (its landing rate's sample)
    // the drafter mix (DFlash2 loaded, TF_GLM_MIX): each source's tokens a round and round time, learned on this stream
    use_df: bool = true,
    mix_tok: [2]f64 = .{ 2.7, 3.0 }, // [mtp, dflash2]
    mix_ms: [2]f64 = .{ 50, 55 },
    mix_round: u32 = 0,
    mix_t: ?std.Io.Timestamp = null,
    trace_held: u32 = 0, // drafts the last window verified (TF_GLM_TRACE: how many landed is known at the next draft)
};

/// A released stream's caches kept at its prompt's kept point (``ids``: that prefix).
const Kept = struct { s: *forward.Seq, ids: []u32 };

pub const Owned = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    rank: u8,
    link: cuda.tp_link.Link,
    nccl: cuda.nccl.Library,
    ctx: api.Ctx,
    kernels: api.Kernels,
    store: api.Store,
    e: *forward.Engine,
    lanes_by: std.AutoHashMap(*lanes.Stream, Lane),
    seqs: std.AutoHashMap(u64, *forward.Seq), // every live or kept stream's caches, by uid (both ranks)
    kept: std.ArrayList(Kept) = .empty, // rank 0: released streams' kept states, oldest first
    next_id: u64 = 1,
    next_uid: u64 = 1,
    room: u64 = 0, // the pool's bytes (setRoom), and what the streams hold of it
    used: u64 = 0,
    drawn: [256]u32 = undefined,
    next: u64 = 0,
    costs: [max_rows]lanes.config.Cost = undefined,
    cost_count: usize = 0,
    shared: [8]lanes.config.Cost = undefined,
    shared_count: usize = 0,
    mtp_ms: f64 = 0,
    confidence: f64 = draft_confidence,
    marginal: f64 = 0, // TF_GLM_DRAFT_MARGINAL: the DFlash2 chain's per-draft chance floor (0: the product rule)
    trace: Trace = .{},
    tower: ?vision.Tower = null, // rank 0, with vision: the image encoder
    image_token: u32 = 0, // the placeholder the features replace (config image_token_id)
    // rank 1: the next prefill's images (its image frame's), freed after it
    media_spans: std.ArrayList([2]i64) = .empty,
    media_feats: ?cuda.DeviceBuffer = null,
    media_id: u64 = 0, // rank 1: the request the held image frame came with (its prefill names it; 0: none)
    guard: ?*Guard = null, // the memory and peer watchdog (guard.zig)
    copy_on: bool = true, // TF_GLM_COPY=0: the head's drafts only
    mix_on: bool = true, // TF_GLM_MIX=0: with DFlash2 loaded, it drafts every round (else each stream picks a source a round)
    copy_rate: copy.Rate = .{},
    draft_rate: copy.Rate = .{}, // the drafter's drafts' landing rate (its own chances run high: DFlash2's say ~0.9 where ~0.67 land)
    copy_min: usize = copy.K, // TF_GLM_COPY_MIN / TF_GLM_COPY_MIN_REPLY: the tokens a match takes in the prompt / the reply
    copy_min_reply: usize = copy.K_REPLY,

    pub fn take(self: *Owned, t: u32) u64 {
        self.drawn[self.next % self.drawn.len] = t;
        self.next += 1;
        return self.next - 1;
    }

    pub fn send(self: *Owned, op: Op, w: *Writer) !void {
        try self.link.send(@backingInt(op), w.buf.items);
        w.buf.clearRetainingCapacity();
    }

    pub fn bytesOf(self: *const Owned, capacity: i64) u64 {
        return forward.seqBytes(self.e, capacity);
    }

    /// A new stream's caches of ``capacity`` slots (``uid``), the pool's bytes counted.
    pub fn newSeq(self: *Owned, uid: u64, capacity: i64) !*forward.Seq {
        const s = try forward.newSeq(self.e, capacity);
        s.uid = uid;
        try self.seqs.put(uid, s);
        self.used += self.bytesOf(capacity);
        return s;
    }

    pub fn dropSeq(self: *Owned, uid: u64) void {
        const s = self.seqs.get(uid) orelse return;
        _ = self.seqs.remove(uid);
        self.used -|= self.bytesOf(s.capacity);
        forward.freeSeq(self.e, s);
    }

    /// Rank 0: the oldest kept state dropped on both ranks (false: none is kept).
    pub fn evictOldest(self: *Owned) !bool {
        if (self.kept.items.len == 0) return false;
        const k = self.kept.orderedRemove(0);
        var w: Writer = .{ .gpa = self.gpa };
        defer w.buf.deinit(self.gpa);
        try w.int(k.s.uid);
        try self.send(.evict, &w);
        self.gpa.free(k.ids);
        self.dropSeq(k.s.uid);
        return true;
    }

    pub fn keptBytes(self: *const Owned) u64 {
        var n: u64 = 0;
        for (self.kept.items) |k| n += self.bytesOf(k.s.capacity);
        return n;
    }
};

/// Whether ``prompt`` + ``max_tokens`` fits the pool beside running streams (kept states give way); else it waits.
pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const self = of(p);
    if (self.guard) |g| if (g.isTight()) return false; // the machine's memory runs short: running streams finish first
    const need = self.bytesOf(@as(i64, prompt) + @as(i64, max_tokens) + forward.MAX_ROWS);
    return self.used -| self.keptBytes() + need <= self.room;
}

/// The streams' pool after the model, set once after open: the smaller of both ranks' pools, less 2%.
pub fn setRoom(p: *anyopaque, room: u64) !void {
    const self = of(p);
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    if (self.rank == 1) {
        try w.int(room);
        try self.send(.room, &w);
        self.room = room - room / 50;
        return;
    }
    const m = try self.link.recv(self.gpa);
    defer self.gpa.free(m.bytes);
    if (m.tag != @backingInt(Op.room)) return error.RankOneNoRoom;
    var r: Reader = .{ .bytes = m.bytes };
    const peer = try r.int();
    const both = @min(room, peer);
    self.room = both - both / 50;
    std.log.info("glm: stream pool {d:.2} GiB (rank 0 {d:.2}, rank 1 {d:.2} GiB)", .{ @as(f64, @floatFromInt(self.room)) / (1 << 30), @as(f64, @floatFromInt(room)) / (1 << 30), @as(f64, @floatFromInt(peer)) / (1 << 30) });
    std.log.info("glm: prompt chunks of {d} rows (TF_GLM_PREFILL_ROWS), the EXL3 experts' prompt kernel {s}", .{ self.e.pbuf.rows, @tagName(@import("kern_exl3.zig").promptKernel()) });
}

/// Commits the lanes' last windows' ``kept`` rows before each lane's next call; rank 1 commits the same.
pub fn settle(self: *Owned, ls: []const *Lane, kept: []const u32) !void {
    var n: usize = 0;
    for (ls) |l| n += @intFromBool(l.rows != 0);
    if (n == 0) return;
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(n);
    for (ls, kept) |l, kp| if (l.rows != 0) {
        try w.int(l.s.uid);
        try w.int(l.rows);
        try w.int(kp);
    };
    try self.send(.keep, &w);
    for (ls, kept) |l, kp| if (l.rows != 0) {
        try decode.keep(self.e, l.s, l.rows, kp);
        l.rows = 0;
    };
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, _: ?[]const u8, o: Options) !Loaded {
    if (o.tp != 2) return error.TwoRanksOnly;
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = undefined;
    own.gpa = gpa;
    own.io = io;
    own.rank = o.rank;
    own.kept = .empty;
    own.next_id = 1;
    own.next_uid = 1;
    own.next = 0;
    own.room = 0;
    own.used = 0;
    own.cost_count = 0;
    own.shared_count = 0;
    own.mtp_ms = 0;
    own.confidence = confidenceFromEnv();
    own.marginal = marginalFromEnv();
    own.copy_on = if (std.c.getenv("TF_GLM_COPY")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    own.copy_rate = .{};
    own.mix_on = native_env.mixWanted();
    own.copy_min = envUsize("TF_GLM_COPY_MIN", copy.K);
    own.copy_min_reply = envUsize("TF_GLM_COPY_MIN_REPLY", copy.K_REPLY);
    own.trace = .{ .on = o.rank == 0 and std.c.getenv("TF_GLM_TRACE") != null };
    own.lanes_by = .init(gpa);
    own.seqs = .init(gpa);
    // the link and the communicator first, while memory is fresh (NCCL registers its buffers at its first use)
    own.link = if (o.rank == 0) try cuda.tp_link.Link.lead(o.master_port, 600_000) else try cuda.tp_link.Link.follow(io, try std.Io.net.IpAddress.parse(o.master orelse return error.NoMaster, o.master_port), 600_000);
    errdefer own.link.close();
    try native_env.agree(own.link, .fromEnv()); // before the first collective: a pair that would run apart refuses
    own.nccl = try cuda.nccl.Library.open();
    errdefer own.nccl.close();
    const comm = try cuda.tp_link.communicator(&own.nccl, own.link);
    own.ctx = .{ .d = ctx.d, .ctx = ctx, .stream = try cuda.Stream.init(ctx.d, true), .nccl = &own.nccl, .comm = comm, .rank = o.rank };
    own.guard = try Guard.start(gpa, own.link.fd, &own.nccl, comm); // from here a short machine or a lost peer ends the rank
    errdefer own.guard.?.stop(gpa);
    { // NCCL connects (and registers its buffers) at its first collective: here, before the weights take the memory
        var warm = try cuda.DeviceBuffer.alloc(ctx.d, 4096);
        defer warm.free();
        try own.nccl.check(own.nccl.api.ncclAllGather(warm.ptr, warm.ptr + 2048, 256, .i32, comm, own.ctx.stream.handle), "warm gather");
        try own.ctx.stream.synchronize();
    }
    // small gathers over RoCE (TF_ROCE=0: NCCL for all; TF_ROCE_MAX_KB: a rank's shard at most, 512 KiB)
    own.ctx.roce = try cuda.roce.Gather.init(gpa, io, ctx.d, own.link, roceWanted(), roceMaxBytes());
    errdefer if (own.ctx.roce) |g| g.deinit();
    own.kernels = try api.Kernels.load(ctx.d);
    errdefer own.kernels.deinit();
    own.store = try weights.load(gpa, io, &own.ctx, dir, o.rank);
    errdefer own.store.deinit();
    own.e = try forward.init(gpa, io, &own.ctx, &own.kernels, &own.store, .{ .context = @intCast(o.context) });
    errdefer forward.deinit(own.e);
    own.tower = null;
    own.media_spans = .empty;
    own.media_feats = null;
    own.media_id = 0;
    own.image_token = try imageToken(gpa, io, dir);
    if (o.vision and o.rank == 0) own.tower = try vision.Tower.load(gpa, io, ctx.d, dir);
    errdefer if (own.tower) |*t| t.deinit();
    // TF_GLM_DRAFTER=dflash2: IncoAI's DFlash2 drafter (non-commercial licence) from TF_GLM_DFLASH2 on both ranks
    if (drafterWanted()) {
        const ddir = std.mem.span(std.c.getenv("TF_GLM_DFLASH2") orelse return error.NoDFlash2Dir);
        own.e.drafter = try dflash.load(gpa, io, own.e, ddir);
        own.e.ring_bytes = dflash.ring_bytes;
    }
    errdefer if (own.e.drafter) |dr| dr.deinit();
    { // both ranks draft alike or neither starts
        const mine = [1]u8{@intFromBool(own.e.drafter != null)};
        var theirs: [1]u8 = undefined;
        try own.link.swap(&mine, &theirs);
        if (mine[0] != theirs[0]) return error.DrafterOnOneRank;
    }
    // both halves loaded before rank 0 serves: rank 1 says so once its engine is up
    if (o.rank == 1) try own.link.send(@backingInt(Op.ready), "") else {
        const m = try own.link.recv(gpa);
        defer gpa.free(m.bytes);
        if (m.tag != @backingInt(Op.ready)) return error.RankOneNotReady;
    }
    if (o.rank == 0) { // rank 1 replays it in its follow loop, then hears it ended
        try @import("calibrate.zig").run(own, o.drafts);
        try own.link.send(@backingInt(Op.ready), "");
        if (own.e.drafter != null) if (std.c.getenv("TF_GLM_DFLASH2_STEP_MS") != null) {
            own.mtp_ms = dflashStepMs(); // calibrate timed a round; the variable sets another
        };
    }
    const window = windowFromEnv();
    return .{
        .backend = .{ .ptr = own, .vtable = &vtable },
        .facts = .{ .exact_width = window, .mtp = o.drafts, .speculate = o.drafts, .speculate_early = false, .drafts = if (o.drafts) window - 1 else 1, .hidden_rows = true, .max_streams = max_streams, .batch_rows = @intCast(forward.STEP_ROWS), .window_costs = own.costs[0..own.cost_count], .mtp_step_ms = own.mtp_ms, .shared_costs = own.shared[0..own.shared_count], .draft_probabilities = o.drafts, .draft_streams = o.drafts },
        .rows = if (o.drafts) window else 1,
        .stream_bytes = forward.seqBytes(own.e, admission_slots),
        .ctx = own,
        .deinit = release,
        .follow = if (o.rank == 1) followLoop else null,
    };
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong, error.ContextFull => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoRoom => "the prompt and its reply do not fit the memory left beside the running requests",
        error.NoVision => "image and video inputs need the server started with --vision",
        error.ImageCountMismatch, error.ImageTokensMismatch => "the prompt's image placeholders do not match its images",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.guard) |g| g.stop(own.gpa); // the peer's close that follows is no lost peer
    if (own.rank == 0) own.link.send(@backingInt(Op.stop), "") catch {};
    for (own.kept.items) |k| own.gpa.free(k.ids);
    own.kept.deinit(own.gpa);
    var it = own.lanes_by.valueIterator();
    while (it.next()) |l| {
        own.gpa.free(l.prompt);
        if (l.copy) |*h| h.deinit(own.gpa);
    }
    own.lanes_by.deinit();
    var si = own.seqs.valueIterator();
    while (si.next()) |s| forward.freeSeq(own.e, s.*);
    own.seqs.deinit();
    if (own.tower) |*t| t.deinit();
    if (own.e.drafter) |dr| dr.deinit();
    own.media_spans.deinit(own.gpa);
    if (own.media_feats) |*f| f.free();
    forward.deinit(own.e);
    own.store.deinit();
    own.kernels.deinit();
    if (own.ctx.roce) |g| g.deinit();
    _ = own.nccl.api.ncclCommAbort(own.ctx.comm); // its threads stop before the library unloads
    own.link.close();
    own.nccl.close();
    own.gpa.destroy(own);
}

pub fn of(p: *anyopaque) *Owned {
    return @ptrCast(@alignCast(p));
}

const vtable: be.Backend.VTable = .{
    .prefill = prefillFn,
    .first = firstFn,
    .queue = queueFn,
    .read = readFn,
    .verify = verifyFn,
    .keep = keepFn,
    .draft = draftFn,
    .release = releaseFn,
    .probabilities = probabilitiesFn,
};

const keep_headroom = native_prompt.keep_headroom;
const keepHeadroom = native_prompt.keepHeadroom;
const prefillFn = native_prompt.prefillFn;
const encodeImages = native_prompt.encodeImages;
const imageToken = native_prompt.imageToken;

pub fn laneOf(self: *Owned, s: *lanes.Stream) !*Lane {
    return self.lanes_by.getPtr(s) orelse error.NoLane;
}

fn firstFn(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
    const self = of(p);
    if (position != s.prompt_len) return error.PositionMismatch;
    return (try laneOf(self, s)).first;
}

fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
    return error.NotPipelined;
}

pub fn readFn(p: *anyopaque, handle: u64) anyerror!u32 {
    const self = of(p);
    if (handle >= self.next or self.next - handle > self.drawn.len) return error.NoSuchToken;
    return self.drawn[handle % self.drawn.len];
}

fn verifyFn(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
    const self = of(p);
    if (windows.len > max_streams) return error.TooManyStreams;
    var ls: [max_streams]*Lane = undefined;
    var kept: [max_streams]u32 = undefined;
    for (windows, 0..) |win, i| {
        if (win.parents != null) return error.TreesNotBuilt;
        ls[i] = try laneOf(self, win.stream);
        kept[i] = ls[i].rows;
        if (win.rows() > max_rows) return error.WindowTooWide;
    }
    try settle(self, ls[0..windows.len], kept[0..windows.len]); // the previous windows, accepted whole
    var ids: [max_streams][max_rows]u32 = undefined;
    var wins: [max_streams]decode.Win = undefined;
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(windows.len);
    for (windows, 0..) |win, i| {
        const l = ls[i];
        const rows = win.rows();
        // the core keys row r's draw at position pos + 1 + r, as the forward draws it
        if (win.positions.len > 0 and win.positions[0] != @as(u64, @intCast(l.s.pos + 1))) return error.PositionMismatch;
        ids[i][0] = win.pending;
        if (l.copy) |h| if (self.trace.on and h.hist.items[h.hist.items.len - 1] != win.pending) {
            self.trace.desync += 1;
        };
        l.copy_verified = @min(l.copied, win.held);
        l.draft_verified = if (l.copied == 0) win.held else 0;
        if (self.trace.on and win.held > 0) {
            var chain: f64 = 1;
            for (l.probs[0..win.held]) |pr| {
                chain *= pr;
                self.trace.predicted += chain;
            }
            self.trace.verified += win.held;
            l.trace_held = win.held;
        }
        if (win.held > 0) @memcpy(ids[i][1..][0..win.held], l.held[0..win.held]) else @memcpy(ids[i][1..][0..win.tokens.len], win.tokens);
        wins[i] = .{ .s = l.s, .ids = ids[i][0..rows] };
        try w.int(l.s.uid);
        try w.tokens(ids[i][0..rows]);
    }
    const t0 = std.Io.Timestamp.now(self.io, .awake);
    try self.send(.verify, &w);
    var picks: [forward.STEP_ROWS]decode.Pick = undefined;
    try decode.verify(self.e, wins[0..windows.len], &picks);
    if (self.trace.on) {
        const t = &self.trace;
        t.verify_ms += msSince(self.io, t0);
        t.rounds += 1;
        t.verify_streams += windows.len;
        for (windows) |win| t.verify_rows += win.rows();
        if (t.rounds == 256) t.log(self);
    }
    var at: usize = 0;
    for (windows, 0..) |win, i| {
        const rows = win.rows();
        for (0..rows) |r| out[i].sampled[r] = picks[at + r].tok;
        @memcpy(out[i].drafts[0 .. rows - 1], ids[i][1..rows]);
        ls[i].rows = @intCast(rows);
        ls[i].nheld = 0;
        at += rows;
    }
}

fn keepFn(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
    const self = of(p);
    var ls: [max_streams]*Lane = undefined;
    var kept: [max_streams]u32 = undefined;
    for (windows, paths, 0..) |win, path, i| {
        if (path.len == 0) return error.EmptyPath;
        for (path, 0..) |r, j| if (r != j) return error.TreesNotBuilt;
        ls[i] = try laneOf(self, win.stream);
        kept[i] = @intCast(path.len);
    }
    try settle(self, ls[0..windows.len], kept[0..windows.len]);
}

const draftFn = native_draft.draftFn;
pub const feedsMtp = native_draft.feedsMtp;
const chooseDrafter = native_draft.chooseDrafter;
const copyDrafts = native_draft.copyDrafts;
const expected = native_draft.expected;

fn probabilitiesFn(p: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
    const self = of(p);
    const l = self.lanes_by.getPtr(s) orelse return false;
    if (out.len > l.nheld) return false;
    @memcpy(out, l.probs[0..out.len]);
    return true;
}

/// The stream's lane ends: its caches kept at the prompt's kept point for a later prompt (rank 0 decides), or freed.
fn releaseFn(p: *anyopaque, s: *lanes.Stream) void {
    const self = of(p);
    var l = self.lanes_by.get(s) orelse return;
    _ = self.lanes_by.remove(s);
    if (l.copy) |*h| h.deinit(self.gpa);
    const seq = l.s;
    // the kept entry allocated before rank 1 hears: a keep it was told of is one rank 0 holds too
    var ids: ?[]u32 = null;
    if (seq.snap_at > 0 and !l.media and seq.snap_at <= l.prompt.len) {
        ids = self.gpa.dupe(u32, l.prompt[0..seq.snap_at]) catch null;
        if (ids) |x| self.kept.ensureUnusedCapacity(self.gpa, 1) catch {
            self.gpa.free(x);
            ids = null;
        };
    }
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    w.int(seq.uid) catch {};
    w.int(@intFromBool(ids != null)) catch {};
    self.send(.release, &w) catch {};
    if (ids) |x| self.kept.appendAssumeCapacity(.{ .s = seq, .ids = x }) else self.dropSeq(seq.uid);
    self.gpa.free(l.prompt);
}

const followLoop = native_follow.followLoop;
