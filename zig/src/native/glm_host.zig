//! GLM-5.3-Flash behind the native server on the lane core: concurrent requests share rounds; rank 1 replays rank 0's.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const glm = tf.glm;
const ge = glm.engine;
const glm_disk = @import("glm_disk.zig");
const cache_fit = @import("cache_fit.zig");
const Allocator = std.mem.Allocator;

/// The warm-up reply's draft depth: every rank of a pair pays a process's first-round costs at the same step.
const DEPTH = 2;

pub const Host = struct {
    gpa: Allocator,
    eng: *ge.Engine,
    slots: glm.slots.Slots,
    back: glm.backend.Backend,
    markers: [3]u32 = undefined, // the image wrapper's tokens (--vision): begin, image, end
    video_markers: [3]u32 = undefined, // the video wrapper's: begin, video, end
    video_open: api.VisionOpen = .{}, // the video limits the server was started with
    cfg: lanes.Config,
    wall: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // rank 0's kept prompt states (rank 1 holds copies by id)
    learned: ?api.prompt_imprint.Imprint = null, // --learn: shared states on disk (rank 1 keeps its halves there)
    warm: mtl.keepalive.Target,
    follower: ?std.Thread = null, // speed-up mode's rank 1: the thread replaying rank 0's slot commands

    pub fn engine(h: *Host) api.Engine {
        return h.host.engine();
    }
};

/// The words for a request the engine refuses.
fn words(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.SampledPeerUnsupported => "speed-up mode: the two-Mac pair decodes greedily only — a correct full-vocabulary draw needs both Macs' logits; send temperature 0",
        error.FollowsPeer => "speed-up mode: this Mac runs rank 0's requests; send requests to rank 0",
        error.ContextFull => "the prompt and max_tokens do not fit this server's --context",
        else => null,
    };
}

/// Streams fitting beside the weights under the load limit, at most `want` (`fixed`: all or none); pooled: KDA only.
fn fit(eng: *const ge.Engine, want: u32, fixed: bool) !u32 {
    const per = glm.state.stateBytes(&eng.c, if (glm.pool.on(eng)) 0 else eng.s.cap, true) + 64 * 1024;
    const used = eng.w.bytes + eng.arena.bytes + glm.vision.heldBytes(eng.vision);
    const limit = eng.limit.bytes;
    const room: u64 = if (limit > used) (limit - used) / per else 0;
    const n: u32 = @intCast(@min(@as(u64, @max(want, 1)), 1 + room));
    if (n < want and fixed) {
        std.log.err("glm: {d} streams of {d}-token caches pass this Mac's {d:.1} GB load limit ({s}); {d} fit (lower --parallel or --context)", .{ want, eng.s.cap, @as(f64, @floatFromInt(limit)) / 1e9, eng.limit.source(), n });
        return error.OverMemoryLimit;
    }
    return n;
}

/// Marks sit on planned chunk starts that every pass cuts anyway, so a prompt under 4,096 tokens keeps them too.
const cache_rules: api.prompt_cache.Rules = .{ .lookahead = 1, .planned = true, .min_prompt = 0 };

/// Kept prompt states' room: --prompt-cache-gib, else 16 GiB, inside what the load limit read at load leaves.
fn cacheBudget(eng: *const ge.Engine, gib: ?f64) u64 {
    const want: u64 = @intFromFloat((gib orelse 16) * (1 << 30));
    const used = eng.w.bytes + eng.arena.bytes + glm.vision.heldBytes(eng.vision);
    const limit = eng.limit.bytes;
    return if (limit > used) @min(want, limit - used) else 0;
}

/// The host's memory counts: `cache_fit.hostMemory` with this engine's configured load limit.
fn readMemory(ctx: ?*anyopaque, reset_peak: bool) ?api.Memory {
    const h: *Host = @ptrCast(@alignCast(ctx.?));
    return cache_fit.hostMemory(h.eng.limit.bytes, reset_peak);
}

/// Learned states under `root` by checkpoint, split, chip, OS build, kernel sources and a probe's prompt-pass bits.
fn learnedStates(gpa: Allocator, io: std.Io, eng: *const ge.Engine, sl: *glm.slots.Slots, root: []const u8, cap: u64) !api.prompt_imprint.Imprint {
    const m = try api.prompt_imprint.Imprint.open(gpa, root, (try glm_disk.identity(io, eng, sl)).id, cap);
    std.log.info("glm: learned prompt states in {s} ({d} known, {d} of {d} MiB)", .{ m.dir, m.metas.items.len, m.total() >> 20, cap >> 20 });
    return m;
}

/// A GLM-5.3-Flash checkpoint served: `window` tokens of cache a stream, warmed first (the pair: `speed_up`).
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, window: u32, speed_up: ?[]const u8, streams: u32, fixed: bool, cache_gib: ?f64, learn: ?Learn, vision: ?api.VisionOpen) !*Host {
    // a prompt's tower rows and images: the images' budget, and every video's frame groups beside it
    const limits: ?glm.vision.Limits = if (vision) |v| .{ .image_tokens = v.image_tokens + v.video_tokens, .max_images = v.max_images + v.max_videos * ((v.max_frames + 1) / 2) } else null;
    const eng = try ge.Engine.loadWith(gpa, io, dir, window + 64, speed_up, learn != null, limits, true);
    errdefer eng.deinit();
    var toks: [96]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(1000 + i);
    var dummy: u8 = 0;
    _ = try eng.generate(&toks, 16, &.{}, DEPTH, .{ .ctx = &dummy, .prefilled = ge.Quiet.prefilled, .tokens = ge.Quiet.tokens, .cancelled = ge.Quiet.cancelled });
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.eng = eng;
    h.follower = null;
    h.learned = null;
    h.slots = try glm.slots.Slots.init(gpa, eng, try fit(eng, streams, fixed));
    errdefer h.slots.deinit(gpa);
    glm.residency.wire(eng, io);
    const n: u32 = @intCast(h.slots.slots.len);
    h.back = .{ .gpa = gpa, .sl = &h.slots };
    errdefer h.back.deinit();
    if (eng.ep != null) h.back.costs = try glm.timing.measure(&h.slots, gpa, io); // both Macs in step, as the warm-up
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), glm.state.max_rows, glm.state.max_rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.wall = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.wall.clock());
    errdefer h.core.deinit();
    const serves = @min(window, glm.pool.capacity(&h.slots) -| (glm.state.max_rows + 1)); // prompt + reply a stream's caches hold beside its last window (a pool keeps whole blocks)
    var offer: ?api.Vision = null;
    if (eng.vision) |v| {
        const o = vision.?;
        h.markers = .{ v.c.image_start, v.c.image_token, v.c.image_end };
        h.video_open = o;
        offer = .{ .ctx = h, .prepare = prepareImage, .image_token = v.c.image_token, .markers = &h.markers, .image_tokens = o.image_tokens, .max_images = o.max_images, .explain = glm.image.decoderProblem };
        if (v.c.video) |vm| {
            h.video_markers = vm;
            offer.?.video = .{ .prepare = prepareVideo, .video_token = vm[1], .group_open = v.c.image_start, .group_close = v.c.image_end, .markers = &h.video_markers, .video_tokens = o.video_tokens, .max_videos = o.max_videos, .max_frames = o.max_frames };
        }
    }
    // sampled draws on one Mac (the pair refuses them per request)
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "glm-zig", .lanes = n, .context_window = serves, .prefill_step = eng.chunk_rows, .greedy_only = eng.ep != null, .admit = if (h.slots.pool != null) .{ .ctx = &h.back, .fits = glm.backend.Backend.fits } else null, .vision = offer });
    h.host.memory = .{ .ctx = h, .read = readMemory };
    h.cache = null;
    const budget = cacheBudget(eng, cache_gib);
    if (!eng.followsPeer() and budget > 0) {
        const B = glm.backend.Backend;
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = &h.back, .vtable = &.{ .bytes = B.snapBytes, .save = B.snapSave, .restore = B.snapRestore, .drop = B.snapDrop, .forget_peers_clear = B.snapPeersClear, .write = B.snapWrite, .read = B.snapRead, .forget = B.snapForget, .forget_checked = B.snapForgetChecked, .reclaim = B.snapReclaim, .peer_other_used = B.peerOtherUsed, .peer_other_next = B.peerOtherNext, .peer_other_bytes = B.peerOtherBytes, .peer_other_remove = B.peerOtherRemove, .peer_need = B.peerNeed, .peer_reclaim = B.peerReclaim, .peer_reserve = B.peerReserve, .peer_finish = B.peerFinish, .rows = &glm_disk.image } }, cache_rules, budget);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
        h.slots.grow = .{ .ctx = h, .f = growSnap }; // a resident state made whole before its slot writes below it
        h.slots.expire = .{ .ctx = h, .f = expireSnap }; // invalidate now, reclaim after restore borrowers release
        h.slots.borrowed = .{ .ctx = h, .f = borrowedState }; // an outstanding lookup spares its state's rows
    }
    errdefer if (h.cache) |*store| store.deinit();
    if (learn) |l| if (l.conversations) {
        if (eng.ep != null) return error.LearnConversationsOnPair;
        if (h.cache) |*store| {
            const id = try glm_disk.identity(io, eng, &h.slots);
            const r = try api.prompt_cache.rows.Rows.open(gpa, store.family, .{ .root = l.root, .identity = id.id, .cap = l.cap, .floor = l.floor, .daily = l.daily, .lookahead = cache_rules.lookahead, .planned = cache_rules.planned });
            r.prefill_rate = id.rate;
            store.disk = r;
        }
    } else {
        h.learned = try learnedStates(gpa, io, eng, &h.slots, l.root, l.cap);
        h.learned.?.admission.floor = l.floor;
        h.slots.disk.admission.floor = l.floor;
        h.slots.disk.cap = l.cap;
        h.slots.learned = h.learned.?.dir;
        if (h.cache) |*store| store.imprint = &h.learned.?;
    };
    errdefer if (h.learned) |*m| m.deinit();
    errdefer if (h.cache) |*store| if (store.disk) |r| r.deinit();
    h.host.explain = .{ .text = words };
    h.warm = eng.keepalive_target;
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start(); // the host thread first: once the follower runs, nothing after it can fail
    errdefer h.host.stop();
    if (eng.followsPeer()) h.follower = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, follow, .{h});
    const role = if (eng.ep == null) "" else if (eng.followsPeer()) ", speed-up rank 1" else ", speed-up rank 0";
    std.log.info("GLM-5.3-Flash loaded in {d:.1} s ({d:.1} GB of weights{s}), context {d} tokens, {d} streams", .{ eng.load_seconds, @as(f64, @floatFromInt(eng.w.bytes)) / 1e9, role, window, n });
    return h;
}

/// Slots.grow: the prompt cache charges a kept state's prefixes when its budget has room (the host thread's store).
fn growSnap(ctx: *anyopaque, saved: *anyopaque, extra: u64) bool {
    const h: *Host = @ptrCast(@alignCast(ctx));
    return if (h.cache) |*store| store.grow(saved, extra) else false;
}

/// Slots.expire only invalidates: the host reclaims storage and peer state outside the backend operation.
fn expireSnap(ctx: *anyopaque, saved: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    if (h.cache) |*store| store.expired(saved);
}

/// Eviction asks the store: an outstanding lookup still restores this state, so its rows are spared.
fn borrowedState(ctx: *anyopaque, saved: *anyopaque) bool {
    const h: *Host = @ptrCast(@alignCast(ctx));
    if (h.cache) |*store| return store.borrowed(saved);
    return false;
}

/// Info.vision's prepare: an image's bytes to the tower's patches (families/glm/image.zig), on the HTTP thread.
fn prepareImage(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const p = try glm.image.prepare(h.wall.io, a, bytes, h.eng.vision.?.c, max_tokens);
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}

/// Info.vision's video offer: a decoded video's frames to the tower's frame groups (glm/video.zig), on the HTTP thread.
fn prepareVideo(ctx: *anyopaque, a: Allocator, src: api.VideoSource, max_tokens: u32, max_frames: u32) anyerror!api.PreparedVideo {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const Adapter = struct {
        src: api.VideoSource,
        fn decode(c: *anyopaque, indices: []const u32, sink: *glm.video.Sink) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(c));
            const into = struct {
                fn take(s: *anyopaque, k: usize, rgb: []const u8, w: u32, hh: u32, stride: usize) anyerror!void {
                    const g: *glm.video.Sink = @ptrCast(@alignCast(s));
                    try g.take(k, rgb, w, hh, stride);
                }
            }.take;
            try self.src.decode(self.src.ctx, indices, .{ .ctx = sink, .take = into });
        }
    };
    var ad: Adapter = .{ .src = src };
    const s: glm.video.Source = .{ .frames = src.frames, .rate_num = src.rate_num, .rate_den = src.rate_den, .width = src.width, .height = src.height, .ctx = &ad, .decode = Adapter.decode };
    const p = try glm.video.prepare(a, s, h.eng.vision.?.c, max_tokens, max_frames);
    const groups = try a.alloc(api.PreparedImage, p.groups.len);
    for (groups, p.groups) |*g, q| g.* = .{ .pixels = q.pixels, .gh = q.gh, .gw = q.gw, .tokens = q.tokens, .hash = q.hash };
    return .{ .groups = groups, .times = p.times };
}

fn follow(h: *Host) void {
    glm.mirror.follow(h.eng, &h.slots) catch |err| std.log.err("speed-up mode: following rank 0 ended: {s}", .{@errorName(err)});
}

/// --learn's settings: the root, cap, free-disk floor and daily writes, and whether conversations go to disk too.
pub const Learn = struct { root: []const u8, cap: u64, floor: u64, daily: u64, conversations: bool };

pub fn close(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    h.host.stop();
    const rows = if (h.cache) |*store| store.disk else null;
    if (h.cache) |*store| store.deinit();
    if (rows) |r| r.deinit(); // after the store: its entries leave the write queue as they go
    if (h.follower) |th| { // rank 1: its wait for rank 0's next command ends, then the thread
        h.eng.stopFollowing();
        th.join();
    }
    h.core.deinit();
    h.cfg.deinit(h.gpa);
    h.back.deinit();
    h.slots.deinit(h.gpa);
    if (h.learned) |*m| m.deinit();
    h.eng.deinit();
    h.gpa.destroy(h);
}

test "a 3.5k-token prompt keeps its shared system cut, a planned chunk start" {
    const Stub = struct {
        fn bytes(_: *anyopaque, _: u32) u64 {
            return 0;
        }
        fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!api.prompt_cache.Saved {
            return error.Unused;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) anyerror!void {}
        fn drop(_: *anyopaque, _: api.prompt_cache.Saved) void {}
    };
    var unused: u8 = 0;
    var store = api.prompt_cache.Store.init(std.testing.allocator, .{ .ptr = &unused, .vtable = &.{ .bytes = Stub.bytes, .save = Stub.save, .restore = Stub.restore, .drop = Stub.drop } }, cache_rules, 1 << 30);
    defer store.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const prompt = try arena.allocator().alloc(u32, 3547);
    for (prompt, 0..) |*t, i| t.* = @intCast(i);
    // a 3,523-token system prompt: the server cuts at 3,522 and names 1,475, 3,011 and 3,523 as shared cuts
    const marks = try store.marks(arena.allocator(), prompt, 0, 3540, &.{ 1475, 3011, 3523 }, &.{3522}, &.{});
    try std.testing.expectEqualSlices(u32, &.{3522}, marks);
}
