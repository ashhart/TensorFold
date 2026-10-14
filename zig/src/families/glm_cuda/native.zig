//! GLM-5.3-Flash (glm5_next) on two GB10s behind the native family interface: rank 0's lane core drives the engine
//! through a lane backend that first sends each call to rank 1, which replays it on its own half of the model
//! (``follow``). Every token is drawn from both ranks' vocabulary shards (an NCCL exchange), so both ranks reach the
//! same tokens and rank 1 needs nothing back. Up to 16 streams decode together, each with its own caches sized to its
//! request out of one memory pool; a released stream's kept prompt state stays for a later prompt that extends it
//! while the pool has room.

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

const max_rows: usize = @intCast(forward.MAX_ROWS); // a stream's verify window at most (pending + 15 drafts)
const max_depth = max_rows - 1;
const max_streams = forward.MAX_STREAMS;
/// The head stops at a draft it gives less than this (the chain's probability product), the Python engine's rule;
/// TF_GLM_DRAFT_CONFIDENCE sets another (rank 0's, sent with each draft frame).
const draft_confidence: f64 = 0.7;

/// TF_GLM_DRAFT_MARGINAL=p (DFlash2, rank 0): a chain stops at the first draft whose own chance (given the drafts
/// before it) is below p, instead of at the chain's product below the confidence cut; 0 (the default) keeps the product rule.
fn marginalFromEnv() f64 {
    const v = std.c.getenv("TF_GLM_DRAFT_MARGINAL") orelse return 0;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch 0;
}

fn confidenceFromEnv() f64 {
    const v = std.c.getenv("TF_GLM_DRAFT_CONFIDENCE") orelse return draft_confidence;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch draft_confidence;
}

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
        std.log.info("glm trace: {d} verifies, {d:.2} ms each, {d:.2} rows ({d:.2} streams); {d} drafts, {d:.2} ms each, {d:.2} drafted a stream",
            .{ t.rounds, t.verify_ms / n, @as(f64, @floatFromInt(t.verify_rows)) / n, @as(f64, @floatFromInt(t.verify_streams)) / n,
               t.calls, t.draft_ms / c, @as(f64, @floatFromInt(t.drafts)) / @as(f64, @floatFromInt(@max(t.verify_streams, 1))) });
        std.log.info("glm drafts: {d} verified, {d} landed ({d:.2}), their chances said {d:.2}", .{ t.verified, t.landed,
            @as(f64, @floatFromInt(t.landed)) / @as(f64, @floatFromInt(@max(t.verified, 1))), t.predicted / @as(f64, @floatFromInt(@max(t.verified, 1))) });
        std.log.info("glm copies: {d} of {d} draft rounds copied ({d:.2} tokens each), copied drafts land {d:.2}, the drafter's {d:.2}; {d} out of step",
            .{ t.copies, t.calls, @as(f64, @floatFromInt(t.copy_tokens)) / @as(f64, @floatFromInt(@max(t.copies, 1))), own.copy_rate.chance(), own.draft_rate.chance(), t.desync });
        {
            const ds = &decode.draw_stats;
            const dc: f64 = @floatFromInt(@max(ds.calls, 1));
            std.log.info("glm draws: {d} sampled verifies, {d:.2} ms gpu + {d:.2} ms host each, {d} rows, {d} full rows, {d} check mismatches", .{ ds.calls, ds.gpu_ms / dc, ds.host_ms / dc, ds.rows, ds.full, ds.mismatch });
            ds.* = .{};
        }
        if (own.e.drafter) |dr| {
            const nc: f64 = @floatFromInt(@max(dr.calls, 1));
            std.log.info("glm mix: {d} stream rounds by DFlash2, {d} by the MTP head", .{ t.df_rounds, t.drafts_calls_total - t.df_rounds });
            std.log.info("glm dflash2: {d} rounds, {d:.2} ms each (context {d:.2}, tables {d:.2}, layers {d:.2}, head {d:.2}, selector {d:.2}); {d} streams' contexts restarted",
                .{ dr.calls, dr.forward_ms / nc, dr.phase_ms[0] / nc, dr.phase_ms[4] / nc, dr.phase_ms[1] / nc, dr.phase_ms[2] / nc, dr.phase_ms[3] / nc, dr.lost });
            dflash.logFine(dr, dr.calls);
            dr.calls = 0;
            dr.forward_ms = 0;
            dr.phase_ms = @splat(0);
        }
        const g: f64 = 1 << 30;
        const avail: u64 = if (own.guard) |gd| gd.available.load(.monotonic) else 0;
        const low: u64 = if (own.guard) |gd| gd.lowest.load(.monotonic) else 0;
        std.log.info("glm memory: {d:.2} GiB available (lowest {d:.2}); pool {d:.2} of {d:.2} GiB; {d} graphs ({d} captured, {d} updated in place, {d} dropped, {d:.0} ms capturing); device {d:.2} GiB",
            .{ @as(f64, @floatFromInt(avail)) / g, @as(f64, @floatFromInt(low)) / g, @as(f64, @floatFromInt(own.used)) / g, @as(f64, @floatFromInt(own.room)) / g,
               own.e.graphs.count(), own.e.captures, own.e.updates, own.e.evictions, own.e.capture_ms, @as(f64, @floatFromInt(cuda.usage(false).device)) / g });
        t.* = .{ .on = true };
    }
};
/// A stream's caches for admission: the slots a small request takes (the host's count of streams); requests take
/// what they need from the pool (``fits``).
const admission_slots: i64 = 32768;
/// Prompt tokens a stream's copy index covers at most (its last; TF_GLM_COPY on): about 24 MiB of host memory.
const copy_index: usize = 1 << 20;

const Op = proto.Op;
const Writer = proto.Writer;
const Reader = proto.Reader;

const Lane = struct {
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
        try self.link.send(@intFromEnum(op), w.buf.items);
        w.buf.clearRetainingCapacity();
    }

    fn bytesOf(self: *const Owned, capacity: i64) u64 {
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
    fn evictOldest(self: *Owned) !bool {
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

    fn keptBytes(self: *const Owned) u64 {
        var n: u64 = 0;
        for (self.kept.items) |k| n += self.bytesOf(k.s.capacity);
        return n;
    }
};

/// Whether a request of ``prompt`` tokens and ``max_tokens`` more fits the pool beside the running streams (kept
/// states give way): the lane host keeps it waiting, in order, until it does.
pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const self = of(p);
    if (self.guard) |g| if (g.isTight()) return false; // the machine's memory runs short: running streams finish first
    const need = self.bytesOf(@as(i64, prompt) + @as(i64, max_tokens) + forward.MAX_ROWS);
    return self.used -| self.keptBytes() + need <= self.room;
}

/// The host's pool for the streams (the bytes left after the model and its buffers), set once after open. Rank 1
/// holds the same caches out of its own memory, so the pair takes the smaller of the two ranks' pools (rank 1 sends
/// its own: the frame after its ready), with a 2% margin.
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
    if (m.tag != @intFromEnum(Op.room)) return error.RankOneNoRoom;
    var r: Reader = .{ .bytes = m.bytes };
    const peer = try r.int();
    const both = @min(room, peer);
    self.room = both - both / 50;
    std.log.info("glm: stream pool {d:.2} GiB (rank 0 {d:.2}, rank 1 {d:.2} GiB)", .{ @as(f64, @floatFromInt(self.room)) / (1 << 30), @as(f64, @floatFromInt(room)) / (1 << 30), @as(f64, @floatFromInt(peer)) / (1 << 30) });
    std.log.info("glm: prompt chunks of {d} rows (TF_GLM_PREFILL_ROWS), the EXL3 experts' prompt kernel {s}", .{ self.e.pbuf.rows, @tagName(@import("kern_exl3.zig").promptKernel()) });
}


/// Commits the lanes' last windows: ``kept`` of their rows (the lane core calls keep only to cut a window short, so a
/// window it accepted whole is committed here, before the lane's next call). Rank 1 commits the same.
fn settle(self: *Owned, ls: []const *Lane, kept: []const u32) !void {
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
    own.mix_on = if (std.c.getenv("TF_GLM_MIX")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    own.copy_min = envUsize("TF_GLM_COPY_MIN", copy.K);
    own.copy_min_reply = envUsize("TF_GLM_COPY_MIN_REPLY", copy.K_REPLY);
    own.trace = .{ .on = o.rank == 0 and std.c.getenv("TF_GLM_TRACE") != null };
    own.lanes_by = .init(gpa);
    own.seqs = .init(gpa);
    // the link and the communicator first, while memory is fresh (NCCL registers its buffers at its first use)
    own.link = if (o.rank == 0) try cuda.tp_link.Link.lead(o.master_port, 600_000)
        else try cuda.tp_link.Link.follow(io, try std.Io.net.IpAddress.parse(o.master orelse return error.NoMaster, o.master_port), 600_000);
    errdefer own.link.close();
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
    if (o.rank == 1) try own.link.send(@intFromEnum(Op.ready), "") else {
        const m = try own.link.recv(gpa);
        defer gpa.free(m.bytes);
        if (m.tag != @intFromEnum(Op.ready)) return error.RankOneNotReady;
    }
    if (o.rank == 0) { // rank 1 replays it in its follow loop, then hears it ended
        try @import("calibrate.zig").run(own, o.drafts);
        try own.link.send(@intFromEnum(Op.ready), "");
        if (own.e.drafter != null) if (std.c.getenv("TF_GLM_DFLASH2_STEP_MS") != null) {
            own.mtp_ms = dflashStepMs(); // calibrate timed a round; the variable sets another
        };
    }
    const window = windowFromEnv();
    return .{
        .backend = .{ .ptr = own, .vtable = &vtable },
        .facts = .{ .exact_width = window, .mtp = o.drafts, .speculate = o.drafts, .speculate_early = false, .drafts = if (o.drafts) window - 1 else 1,
                    .hidden_rows = true, .max_streams = max_streams, .batch_rows = @intCast(forward.STEP_ROWS),
                    .window_costs = own.costs[0..own.cost_count], .mtp_step_ms = own.mtp_ms, .shared_costs = own.shared[0..own.shared_count],
                    .draft_probabilities = o.drafts, .draft_streams = o.drafts },
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
    if (own.rank == 0) own.link.send(@intFromEnum(Op.stop), "") catch {};
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

/// Cache slots a kept state is allocated beyond its own request (TF_GLM_KEEP_HEADROOM, 8192; 0: none). Read once.
var keep_headroom: ?i64 = null;
fn keepHeadroom() i64 {
    if (keep_headroom == null) keep_headroom = if (std.c.getenv("TF_GLM_KEEP_HEADROOM")) |v| @max(0, std.fmt.parseInt(i64, std.mem.span(v), 10) catch 8192) else 8192;
    return keep_headroom.?;
}

/// The stream's prompt into its caches (a kept state it extends, or new ones from the pool); the first token is
/// drawn here.
fn prefillFn(p: *anyopaque, s: *lanes.Stream) anyerror!void {
    const self = of(p);
    const ids = s.prompt();
    const cap: i64 = @intCast(ids.len + s.max_new + max_rows);
    if (ids.len == 0 or cap > self.e.capacity) return error.PromptTooLong;
    const id = self.next_id;
    self.next_id += 1;
    // images: encoded, their placeholder runs found, rank 1 sent both, before the prompt (a refusal stays here)
    var spans: std.ArrayList([2]i64) = .empty;
    defer spans.deinit(self.gpa);
    var feats: ?cuda.DeviceBuffer = null;
    defer if (feats) |*f| f.free();
    if (s.images.len > 0) feats = try encodeImages(self, ids, s.images, id, &spans);
    const media = s.images.len > 0;
    // the longest kept state this prompt extends (not with images: their features are this prompt's alone)
    var best: ?usize = null;
    if (!media) for (self.kept.items, 0..) |k, i| {
        if (k.ids.len >= ids.len or k.s.capacity < cap or !std.mem.eql(u32, k.ids, ids[0..k.ids.len])) continue;
        if (best == null or k.ids.len > self.kept.items[best.?].ids.len) best = i;
    };
    var seq: *forward.Seq = undefined;
    var start: usize = 0;
    var fresh: i64 = 0;
    const keep_at: usize = if (s.drafts and !media) @max(1, ids.len - 1) else 0;
    if (best) |i| {
        const k = self.kept.orderedRemove(i);
        seq = k.s;
        start = k.ids.len;
        self.gpa.free(k.ids);
    } else {
        // a state that will be kept gets headroom when the pool has it: a later turn of the conversation (its reply
        // and the next message more) resumes it instead of a prompt its exact size only
        const head: i64 = if (keep_at > 0) keepHeadroom() else 0;
        var alloc = cap;
        // (kept states give way, as in fits(): a pool full of them is not a full pool)
        if (head > 0 and cap + head <= self.e.capacity and self.used -| self.keptBytes() + self.bytesOf(cap + head) <= self.room) alloc = cap + head;
        while (self.used + self.bytesOf(alloc) > self.room) if (!try self.evictOldest()) return error.NoRoom;
        fresh = alloc;
        seq = try self.newSeq(self.next_uid, alloc);
        self.next_uid += 1;
    }
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(seq.uid);
    try w.int(fresh);
    try w.tokens(ids);
    try proto.writeSampling(&w, s.sampling);
    try w.int(start);
    try w.int(keep_at);
    try self.send(.prefill, &w);
    seq.sampling = s.sampling;
    s.cached = @intCast(start);
    if (feats) |f| forward.attach(seq, .{ .spans = spans.items, .feats = f.ptr });
    const first = try decode.prefill(self.e, seq, ids, start, keep_at);
    var history: ?copy.Copy = if (self.copy_on and s.drafts) try copy.Copy.init(self.gpa, ids, copy_index, self.copy_min, self.copy_min_reply) else null;
    errdefer if (history) |*h| h.deinit(self.gpa);
    try self.lanes_by.put(s, .{ .id = id, .s = seq, .first = self.take(first), .media = media, .prompt = try self.gpa.dupe(u32, ids), .copy = history });
}

/// The prompt's images through the tower (each its own call, features in prompt order) and their placeholder runs
/// (one a image, its length the image's tokens); rank 1 gets both in an image frame.
fn encodeImages(self: *Owned, ids: []const u32, images: []const lanes.Image, id: u64, spans: *std.ArrayList([2]i64)) !cuda.DeviceBuffer {
    const tower = &(self.tower orelse return error.NoVision);
    var i: usize = 0;
    while (i < ids.len) : (i += 1) {
        if (ids[i] != self.image_token) continue;
        var j = i;
        while (j < ids.len and ids[j] == self.image_token) j += 1;
        try spans.append(self.gpa, .{ @intCast(i), @intCast(j) });
        i = j;
    }
    if (spans.items.len != images.len) return error.ImageCountMismatch;
    var rows: usize = 0;
    for (images, spans.items) |im, sp| {
        const n: usize = @intCast(@divExact(im.grid[0] * im.grid[1] * im.grid[2], 4));
        if (n != sp[1] - sp[0]) return error.ImageTokensMismatch;
        rows += n;
    }
    var feats = try cuda.DeviceBuffer.alloc(self.ctx.d, rows * 4096 * 2);
    errdefer feats.free();
    var at: usize = 0;
    for (images, spans.items) |im, sp| {
        try tower.encode(&self.e.k, im.patches, im.grid, feats.ptr + at * 4096 * 2);
        at += @intCast(sp[1] - sp[0]);
    }
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(id);
    try w.int(spans.items.len);
    for (spans.items) |sp| {
        try w.int(sp[0]);
        try w.int(sp[1]);
    }
    try w.int(rows);
    const fat = w.buf.items.len;
    try w.buf.resize(self.gpa, fat + rows * 4096 * 2);
    try self.ctx.d.check(self.ctx.d.api.cuMemcpyDtoH_v2(w.buf.items[fat..].ptr, feats.ptr, rows * 4096 * 2), "image features");
    try self.send(.image, &w);
    return feats;
}

fn imageToken(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !u32 {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const cfg = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer cfg.deinit();
    const v = cfg.value.object.get("image_token_id") orelse return error.NoImageToken;
    return @intCast(v.integer);
}

fn laneOf(self: *Owned, s: *lanes.Stream) !*Lane {
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

fn readFn(p: *anyopaque, handle: u64) anyerror!u32 {
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

fn draftFn(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
    const self = of(p);
    if (requests.len > max_streams) return error.TooManyStreams;
    var ls: [max_streams]*Lane = undefined;
    var kept: [max_streams]u32 = undefined;
    for (requests, 0..) |r, i| {
        if (r.lanes != null) return error.TreesNotBuilt;
        ls[i] = try laneOf(self, r.stream);
        // the kept rows (a lone round has kept already and settles nothing)
        kept[i] = if (r.rows) |rows| @intCast(rows.len) else ls[i].rows;
    }
    try settle(self, ls[0..requests.len], kept[0..requests.len]);
    var follow: [max_streams][max_rows]u32 = undefined;
    var reqs: [max_streams]decode.DraftReq = undefined;
    var depths: [max_streams]u32 = undefined;
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(requests.len);
    for (requests, 0..) |r, i| {
        const l = ls[i];
        var n: usize = 1;
        var first = false;
        if (r.rows) |rows| {
            for (rows, 0..) |row, j| if (row != j) return error.TreesNotBuilt;
            n = rows.len;
            @memcpy(follow[i][0..n], r.follow[0..n]);
        } else {
            follow[i][0] = switch (r.first orelse return error.NoFirstToken) {
                .handle => |h| try readFn(p, h),
                .value => |v| v,
            };
            first = true;
        }
        const depth: u32 = @min(r.depth, max_depth);
        depths[i] = depth;
        if (self.e.drafter != null) chooseDrafter(self, l, if (r.rows != null) n else 0);
        const df = self.e.drafter != null and l.use_df;
        // a stream DFlash2 drafts for still feeds the MTP head its rows (one step, its draft unused): it stays current
        reqs[i] = .{ .s = l.s, .hidden = if (first) l.s.last_hidden else l.s.hidden, .next = follow[i][0..n], .count = if (df) 1 else depth,
            .confidence = self.confidence, .out = l.held[0..depth], .probs = l.probs[0..depth] };
        try w.int(l.s.uid);
        try w.tokens(follow[i][0..n]);
        try w.int(depth);
        try w.int(@intFromBool(first)); // 1: the first draft after the prompt (its hidden row: the prompt's last)
        try w.int(@as(u64, @bitCast(self.confidence))); // the confidence cut (rank 1 must stop where rank 0 does: the head's steps gather)
        try w.int(@intFromBool(df)); // 1: DFlash2 drafts this stream's round
    }
    const t0 = std.Io.Timestamp.now(self.io, .awake);
    try self.send(.draft, &w);
    // the MTP head: every stream's rows, the drafts of the streams it drafts for (none while DFlash2 drafts alone: TF_GLM_MIX=0)
    if (feedsMtp(self)) try decode.draft(self.e, reqs[0..requests.len]);
    if (self.e.drafter) |dr| {
        var dq: [max_streams]dflash.Req = undefined;
        for (requests, 0..) |r, i| {
            const nf = reqs[i].next.len;
            dq[i] = .{ .s = ls[i].s, .n = if (r.rows != null) nf else 0, .anchor = follow[i][nf - 1], .out = ls[i].held[0..max_depth],
                .probs = ls[i].probs[0..max_depth], .drafts = ls[i].use_df };
        }
        try dflash.draft(dr, self.e, dq[0..requests.len]);
        for (reqs[0..requests.len], ls[0..requests.len], depths[0..requests.len]) |*q, l, depth| { // the confidence cut, as the head's chains stop
            if (!l.use_df) continue;
            var chain: f64 = 1;
            q.got = 0;
            while (q.got < @min(depth, dflash.BLOCK - 1)) : (q.got += 1) { // a block's 7 (a copy may run longer)
                chain *= l.probs[q.got];
                if (q.got > 0 and (if (self.marginal > 0) l.probs[q.got] < self.marginal else chain < q.confidence)) break;
            }
        }
    }
    if (self.trace.on) {
        self.trace.draft_ms += msSince(self.io, t0);
        self.trace.calls += 1;
        for (reqs[0..requests.len]) |q| self.trace.drafts += q.got;
        self.trace.drafts_calls_total += requests.len;
    }
    for (requests, 0..) |r, i| {
        const l = ls[i];
        const depth: u32 = @min(r.depth, max_depth);
        const got = reqs[i].got;
        for (got..depth) |j| { // past the confidence cut: held with chance 0 (the allocator leaves them out)
            l.held[j] = if (got > 0) l.held[got - 1] else follow[i][reqs[i].next.len - 1];
            l.probs[j] = 0;
        }
        l.nheld = depth;
        if (self.trace.on and r.rows != null and l.trace_held > 0) self.trace.landed += @min(l.trace_held, reqs[i].next.len - 1);
        l.trace_held = 0;
        if (l.copy) |*h| try copyDrafts(self, l, h, follow[i][0..reqs[i].next.len], r.rows != null, depth, got);
    }
}

/// Whether a draft round runs the MTP head: always without DFlash2, and with it only while the mix may pick the head
/// (both ranks read TF_GLM_MIX, so they agree).
pub fn feedsMtp(self: *const Owned) bool {
    return self.e.drafter == null or (self.mix_on and self.e.w.mtp != null);
}

/// The source for this stream's round (DFlash2 loaded): the one whose measured tokens per round over round time is
/// higher, the other one every 8th round (its numbers stay fresh); ``kept``: the last round's committed tokens (0: none).
fn chooseDrafter(self: *Owned, l: *Lane, kept: usize) void {
    const now = std.Io.Timestamp.now(self.io, .awake);
    if (!self.mix_on or self.e.w.mtp == null) {
        l.use_df = true;
        return;
    }
    if (kept > 0) if (l.mix_t) |t| {
        const d: usize = @intFromBool(l.use_df);
        const ms = @as(f64, @floatFromInt(t.durationTo(now).toNanoseconds())) / 1e6;
        l.mix_tok[d] = 0.85 * l.mix_tok[d] + 0.15 * @as(f64, @floatFromInt(kept));
        l.mix_ms[d] = 0.85 * l.mix_ms[d] + 0.15 * ms;
    };
    l.mix_t = now;
    l.mix_round += 1;
    const df_better = l.mix_tok[1] / l.mix_ms[1] >= l.mix_tok[0] / l.mix_ms[0];
    l.use_df = if (l.mix_round % 8 == 0) !df_better else df_better;
    if (self.trace.on) self.trace.df_rounds += @intFromBool(l.use_df);
}

/// A copy over the head's drafts when its expected landed tokens are more (rank 0: the verify frame carries the
/// held tokens to rank 1). ``follow``: the tokens just committed; ``kept_rows``: the last window's rows are known.
fn copyDrafts(self: *Owned, l: *Lane, h: *copy.Copy, follow: []const u32, kept_rows: bool, depth: u32, got: usize) !void {
    const landed: u32 = @intCast(follow.len - 1);
    if (kept_rows and l.copy_verified > 0) self.copy_rate.record(l.copy_verified, landed);
    if (kept_rows and l.draft_verified > 0) self.draft_rate.record(l.draft_verified, landed);
    l.copy_verified = 0;
    l.draft_verified = 0;
    l.copied = 0;
    try h.append(self.gpa, follow);
    if (depth == 0) return;
    var buf: [max_depth]u32 = undefined;
    const n = h.propose(buf[0..depth]);
    if (n == 0) return;
    const q = self.copy_rate.chance();
    // the copy agrees with every draft and runs on: the drafts stay, the copy's tail follows them
    var same: usize = 0;
    while (same < @min(n, got) and buf[same] == l.held[same]) same += 1;
    const extends = same == got and n > got;
    if (!extends) { // else the source whose drafts are expected to land more, by their measured rates
        const mine = expected(q, n);
        const theirs = expected(self.draft_rate.chance(), got);
        if (mine <= theirs) return;
    }
    const keep = if (extends) got else 0;
    @memcpy(l.held[keep..n], buf[keep..n]);
    for (l.probs[keep..n]) |*pr| pr.* = q;
    for (n..depth) |j| {
        l.held[j] = buf[n - 1];
        l.probs[j] = 0;
    }
    l.copied = @intCast(n);
    if (self.trace.on) {
        self.trace.copies += 1;
        self.trace.copy_tokens += n;
    }
}

/// Tokens a chain of ``n`` drafts lands on average when each lands with chance ``p`` given the ones before.
fn expected(p: f64, n: usize) f64 {
    var sum: f64 = 0;
    var chain: f64 = 1;
    for (0..n) |_| {
        chain *= p;
        sum += chain;
    }
    return sum;
}

/// TF_GLM_WINDOW: a stream's verify window at most, 2 to 16 rows (16: long copy drafts verified in one window).
fn windowFromEnv() u32 {
    const v = std.c.getenv("TF_GLM_WINDOW") orelse return max_rows;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch return max_rows;
    return @max(2, @min(n, max_rows));
}

fn envUsize(name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

fn drafterWanted() bool {
    const v = std.c.getenv("TF_GLM_DRAFTER") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "dflash2");
}

/// TF_GLM_DFLASH2_STEP_MS: a DFlash2 round's cost spread over its 7 drafts (1.6 ms: about 11 ms a round).
fn dflashStepMs() f64 {
    const v = std.c.getenv("TF_GLM_DFLASH2_STEP_MS") orelse return 1.6;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch 1.6;
}

fn roceWanted() bool {
    const v = std.c.getenv("TF_ROCE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

fn roceMaxBytes() usize {
    const v = std.c.getenv("TF_ROCE_MAX_KB") orelse return 512 << 10;
    return (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 512) << 10;
}

fn msSince(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
}

fn probabilitiesFn(p: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
    const self = of(p);
    const l = self.lanes_by.getPtr(s) orelse return false;
    if (out.len > l.nheld) return false;
    @memcpy(out, l.probs[0..out.len]);
    return true;
}

/// The stream's lane ends: its caches kept at the prompt's kept point for a later prompt (rank 0 decides, rank 1 hears
/// it), or freed.
fn releaseFn(p: *anyopaque, s: *lanes.Stream) void {
    const self = of(p);
    var l = self.lanes_by.get(s) orelse return;
    _ = self.lanes_by.remove(s);
    if (l.copy) |*h| h.deinit(self.gpa);
    const seq = l.s;
    const keep_it = seq.snap_at > 0 and !l.media and seq.snap_at <= l.prompt.len;
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    w.int(seq.uid) catch {};
    w.int(@intFromBool(keep_it)) catch {};
    self.send(.release, &w) catch {};
    if (keep_it) {
        const ids = self.gpa.dupe(u32, l.prompt[0..seq.snap_at]) catch null;
        if (ids) |x| self.kept.append(self.gpa, .{ .s = seq, .ids = x }) catch {
            self.gpa.free(x);
            self.dropSeq(seq.uid);
        } else self.dropSeq(seq.uid);
    } else self.dropSeq(seq.uid);
    self.gpa.free(l.prompt);
}

/// Rank 1: rank 0's calls, replayed in order on this rank's half of the model, until rank 0 stops.
fn followLoop(p: *anyopaque) anyerror!void {
    const self = of(p);
    var picks: [forward.STEP_ROWS]decode.Pick = undefined;
    var held: [max_streams][max_depth]u32 = undefined;
    var probs: [max_streams][max_depth]f64 = undefined;
    while (true) {
        const m = self.link.recv(self.gpa) catch |e| switch (e) {
            error.PeerClosed => return,
            else => return e,
        };
        defer self.gpa.free(m.bytes);
        var r: Reader = .{ .bytes = m.bytes };
        switch (@as(Op, @enumFromInt(m.tag))) {
            .prefill => {
                const uid = try r.int();
                const fresh: i64 = @intCast(try r.int());
                const toks = try r.tokens();
                const sampling = try proto.readSampling(&r);
                const start: usize = @intCast(try r.int());
                const keep_at: usize = @intCast(try r.int());
                const seq = if (fresh > 0) try self.newSeq(uid, fresh) else self.seqs.get(uid) orelse return error.NoSuchSequence;
                seq.sampling = sampling;
                defer { // the image frame's features served this prefill alone
                    self.media_spans.clearRetainingCapacity();
                    if (self.media_feats) |*f| f.free();
                    self.media_feats = null;
                }
                if (self.media_feats) |f| forward.attach(seq, .{ .spans = self.media_spans.items, .feats = f.ptr });
                _ = try decode.prefill(self.e, seq, toks, start, keep_at);
            },
            .image => {
                _ = try r.int();
                const n: usize = @intCast(try r.int());
                self.media_spans.clearRetainingCapacity();
                for (0..n) |_| try self.media_spans.append(self.gpa, .{ @intCast(try r.int()), @intCast(try r.int()) });
                const rows: usize = @intCast(try r.int());
                const bytes = try r.raw(rows * 4096 * 2);
                if (self.media_feats) |*f| f.free();
                self.media_feats = try cuda.DeviceBuffer.alloc(self.ctx.d, rows * 4096 * 2);
                try self.media_feats.?.upload(0, bytes);
            },
            .verify => {
                const n: usize = @intCast(try r.int());
                var wins: [max_streams]decode.Win = undefined;
                for (0..n) |i| {
                    const s = self.seqs.get(try r.int()) orelse return error.NoSuchSequence;
                    wins[i] = .{ .s = s, .ids = try r.tokens() };
                }
                try decode.verify(self.e, wins[0..n], &picks);
            },
            .keep => {
                const n: usize = @intCast(try r.int());
                for (0..n) |_| {
                    const s = self.seqs.get(try r.int()) orelse return error.NoSuchSequence;
                    const rows: i64 = @intCast(try r.int());
                    try decode.keep(self.e, s, rows, @intCast(try r.int()));
                }
            },
            .draft => {
                const n: usize = @intCast(try r.int());
                var reqs: [max_streams]decode.DraftReq = undefined;
                var dq: [max_streams]dflash.Req = undefined;
                for (0..n) |i| {
                    const s = self.seqs.get(try r.int()) orelse return error.NoSuchSequence;
                    const follow = try r.tokens();
                    const depth: usize = @intCast(try r.int());
                    const first = try r.int() == 1;
                    const cut: f64 = @bitCast(try r.int());
                    const df = try r.int() == 1;
                    reqs[i] = .{ .s = s, .hidden = if (first) s.last_hidden else s.hidden, .next = follow, .count = if (df) 1 else depth, .confidence = cut,
                        .out = held[i][0..depth], .probs = probs[i][0..depth] };
                    dq[i] = .{ .s = s, .n = if (first) 0 else follow.len, .anchor = follow[follow.len - 1], .drafts = df };
                }
                if (feedsMtp(self)) try decode.draft(self.e, reqs[0..n]); // as rank 0: the MTP head first, then DFlash2
                if (self.e.drafter) |dr| try dflash.draft(dr, self.e, dq[0..n]);
            },
            .release => {
                const uid = try r.int();
                if (try r.int() == 0) self.dropSeq(uid);
            },
            .evict => self.dropSeq(try r.int()),
            .stop => { // rank 0 closes the link next: no lost peer
                if (self.guard) |g| g.stop(self.gpa);
                self.guard = null;
                return;
            },
            .ready, .room => {},
        }
    }
}
