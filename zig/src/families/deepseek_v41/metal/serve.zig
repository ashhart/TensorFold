//! DeepSeek-V4.1 on Metal as the lanes serve it (serve_lanes.Model): the target, DSpark's pass, costs and hooks.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const weights = @import("weights.zig");
const dspark = @import("dspark.zig");
const prompt = @import("prompt.zig");
const target_ = @import("target.zig");
const pass_ = @import("pass.zig");
const early_ = @import("early.zig");
const pipelines = @import("pipelines.zig");
const vision_prepare = @import("vision_prepare.zig");
const pool = @import("pool.zig");
const calib = @import("calib.zig");
const round_prof = @import("round_prof.zig");

const serve_lanes = host.serve_lanes;
const iface = host.iface;
const api = host.api;
const Allocator = std.mem.Allocator;

/// The parts a served model borrows; `owner` (boot.zig's model, or a test's rig) is closed after them.
pub const Parts = struct {
    env: target_.Env,
    top: *const weights.Top,
    /// the drafter's weights (null: serial windows, no drafts)
    dspark: ?*const dspark.Weights,
    owner: ?Owner = null,
    /// the weights' residency owner the target's buffers join (null: none join)
    resident: ?*mtl.resident.Resident = null,
    /// what a request's images may hold (null: the server is told of no image input)
    images: ?Images = null,
};

/// Image input as served: the decoder's io and a request's limits.
pub const Images = struct { io: std.Io, tokens: u32 = 4096, count: u32 = 4 };

pub const Owner = struct { ctx: *anyopaque, close: *const fn (ctx: *anyopaque) void };

pub const Options = struct {
    slots: u32 = 4,
    limit: u32,
    /// DSpark's pass when the checkpoint has one (false: serial windows through the same lanes)
    drafts: bool = true,
    cache_bytes: u64 = 0,
    prompt: prompt.Opts = .{},
    /// prompts a chunk at a time between the other streams' rounds (serve_lanes.stepsFromEnv); false: whole prompts
    steps: bool = false,
    /// --decode-share (serve_lanes.Model.decode_share)
    decode_share: ?f64 = null,
    /// greedy windows' next passes queued behind them (the lanes' speculative pass, on unless TF_DSV41_SPEC_DRAFT=0)
    early: bool = true,
    /// drafts on the GPU over the whole vocabulary (TF_DSV41_MARKOV_GPU, on by default at boot)
    gpu_chain: bool = false,
    /// the drafts of a pass after a window that verified fewer (0: every pass makes the config's block)
    narrow: u32 = 0,
    /// TF_DSV41_POOL_TOKENS: the slots share one pool of positions (null: `limit` positions a slot)
    pool: ?pool.Setting = null,
    /// the device and weights a stored cost table is specific to (0: none named)
    identity: u64 = 0,
    /// time each decode round and the drafter's passes (TF_DSV41_ROUND_PROF)
    round_prof: bool = false,
};

/// A position's draft candidates the pass keeps (the shape the lanes size their buffers by).
const candidates = 64;

pub const Metal = struct {
    a: Allocator,
    parts: Parts,
    opt: Options,
    t: *target_.Target,
    dp: ?*dspark.Pass = null,
    /// the idle ticker's target: the weights' residency set (the target's buffers joined it), on a queue of its own
    warm: ?mtl.keepalive.Target = null,
    ad: pass_.Pass = undefined,
    early: ?*early_.Early = null,
    rp: ?*round_prof.RoundProf = null,
    prep: Prep = undefined,

    const Prep = struct { io: std.Io, cfg: @import("vision.zig").Config };

    pub fn open(a: Allocator, parts: Parts, opt: Options) !*Metal {
        const m = try a.create(Metal);
        errdefer a.destroy(m);
        m.* = .{ .a = a, .parts = parts, .opt = opt, .t = undefined };
        m.t = try target_.Target.init(a, parts.env, .{ .slots = opt.slots, .limit = opt.limit, .prompt = opt.prompt, .cache_bytes = opt.cache_bytes, .resident = parts.resident, .pool = opt.pool });
        errdefer m.t.deinit();
        if (opt.round_prof) {
            // a diagnostic: without memory for it the server runs unprofiled
            m.rp = round_prof.RoundProf.init(a) catch null;
            m.t.prof = m.rp;
        }
        errdefer if (m.rp) |p| p.deinit();
        const e = parts.env;
        if (opt.drafts) if (parts.dspark) |w| {
            m.dp = try dspark.Pass.init(a, e.device, e.queue, e.c, e.p, e.s, e.tables, parts.top, w, .{ .slots = opt.slots, .candidates = @min(candidates, e.c.vocab), .gpu_chain = opt.gpu_chain, .narrow = opt.narrow });
            errdefer m.dp.?.deinit();
            m.ad = .{ .x = m.dp.?, .t = m.t };
            if (opt.early) m.early = try early_.Early.init(a, m.dp.?);
            m.ad.early = m.early;
            m.t.behind = m.ad.behind();
            m.dp.?.prof = m.rp;
        };
        try m.holdWarm();
        return m;
    }

    /// The idle ticker's target: macOS unwires an idle model in 1-2 s and a request then re-wires all of it first.
    fn holdWarm(m: *Metal) !void {
        const r = m.parts.resident orelse return;
        var t = r.target();
        // a plain queue of its own: the ticker's thread never commits on the family's queue mid-round
        t.queue = try m.parts.env.device.queue();
        m.warm = t;
    }

    /// The server's ticker target (null without a residency owner): idle ticks, and probes while a long prompt runs.
    pub fn keepalive(m: *Metal) ?api.keepalive.Target {
        return if (m.warm != null) .{ .ctx = &m.warm.?, .tick = mtl.keepalive.Target.tick, .probe = mtl.keepalive.Target.busy } else null;
    }

    /// The lanes' view: one slot a stream, rows of 16 a forward, the release's DSpark shape on one device.
    pub fn model(m: *Metal) serve_lanes.Model {
        const dshape = m.dsparkShape();
        return .{
            .ptr = m,
            .vision = m.vision(),
            .name = "dsv41-metal",
            .slots = m.opt.slots,
            .limit = m.opt.limit,
            .world = 1,
            .prefill_step = m.opt.prompt.chunk,
            .forward_rows = pipelines.max_rows,
            .shape = dshape,
            .target = targetFn,
            .pass = passFn,
            .calibrate = calibrate,
            .spec_default = m.early != null,
            .memory = .{ .ctx = m, .read = memory },
            .health = .{ .ctx = m, .read = healthy },
            .keepalive = m.keepalive(),
            .close = close,
            // a cost table measured on this machine (TF_DSV41_CALIB or TF_DSV41_CALIB_FILE), else the defaults
            .calib_id = m.opt.identity,
            .calib = knobs(),
            .steps = m.opt.steps,
            .decode_share = m.opt.decode_share,
        };
    }

    /// The server's image input: PNG, JPEG or WebP bytes to the tower's patches (null: no tower, or no limits given).
    fn vision(m: *Metal) ?api.Vision {
        const v = m.parts.env.vision orelse return null;
        const lim = m.parts.images orelse return null;
        m.prep = .{ .io = lim.io, .cfg = v.w.cfg };
        return .{ .ctx = &m.prep, .prepare = prepare, .image_token = v.w.cfg.image_token_id, .image_tokens = lim.tokens, .max_images = lim.count, .explain = explain };
    }

    /// The server's words for this family's prepare errors it cannot word.
    fn explain(err: anyerror) ?[]const u8 {
        return switch (err) {
            error.TooFewTokens => "the visual-token budget is too small",
            else => null,
        };
    }

    fn prepare(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
        const p: *const Prep = @ptrCast(@alignCast(ctx));
        const got = try vision_prepare.prepare(p.io, a, bytes, max_tokens, p.cfg);
        return .{ .pixels = got.pixels, .gh = got.gh, .gw = got.gw, .tokens = got.tokens, .hash = got.hash };
    }

    fn dsparkShape(m: *const Metal) host.dspark.Shape {
        const c = m.parts.env.c;
        return .{ .block = c.dspark_block, .noise = c.dspark_noise_token, .window = c.window, .rank = c.dspark_markov_rank, .hidden = c.hidden, .candidates = @min(candidates, c.vocab) };
    }

    /// TF_DSV41_CALIB and its neighbours; unset or blank keeps TF_DSV41_CALIB_FILE when set, else defaults.
    fn knobs() host.calib_env.Knobs {
        var k = host.calib_env.Knobs.fromEnv();
        const no_mode = std.mem.trim(u8, k.mode orelse "", " \t\r\n").len == 0;
        const no_file = std.mem.trim(u8, k.file orelse "", " \t\r\n").len == 0;
        if (no_mode and no_file) k.mode = "default";
        return k;
    }

    fn self(ctx: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ctx));
    }

    fn targetFn(ctx: *anyopaque) iface.Target {
        return self(ctx).t.target();
    }

    fn passFn(ctx: *anyopaque) ?iface.Pass {
        const m = self(ctx);
        return if (m.dp != null) m.ad.pass() else null;
    }

    fn calibrate(ctx: *anyopaque, a: Allocator, io: std.Io, dir: []const u8, k: host.calib_env.Knobs, want: host.calib_env.Shape) anyerror!host.calib_lanes.Boot {
        return self(ctx).calibrateReport(a, io, dir, k, want, null);
    }

    /// `calibrate`, with what the measurement did written to `report` (the tests bound it).
    pub fn calibrateReport(m: *Metal, a: Allocator, io: std.Io, dir: []const u8, k: host.calib_env.Knobs, want: host.calib_env.Shape, report: ?*calib.Report) !host.calib_lanes.Boot {
        return calib.run(a, io, dir, k, want, .{ .t = m.t, .ad = if (m.dp != null) &m.ad else null, .slots = m.opt.slots, .limit = m.opt.limit, .vocab = m.parts.env.c.vocab, .shape = m.dsparkShape(), .report = report });
    }

    /// The device's allocated bytes now (the weights, slots, prompt buffers and kept-state host copies aside).
    fn memory(ctx: ?*anyopaque, _: bool) ?api.Memory {
        const m = self(ctx.?);
        const used = m.parts.env.device.allocated();
        return .{ .active = used, .cache = m.t.cache.held, .peak = used };
    }

    fn healthy(ctx: *anyopaque) bool {
        return !self(ctx).t.failed;
    }

    fn close(ctx: *anyopaque) void {
        const m = self(ctx);
        if (m.early) |e| {
            var buf: [160]u8 = undefined;
            std.log.info("dsv41 metal: {s}", .{e.describe(&buf)});
            e.deinit();
        }
        m.t.behind = null;
        if (m.warm) |w| w.queue.deinit();
        if (m.dp) |p| p.deinit();
        if (m.rp) |p| p.log();
        m.t.deinit();
        if (m.rp) |p| p.deinit();
        if (m.parts.owner) |o| o.close(o.ctx);
        m.a.destroy(m);
    }
};
