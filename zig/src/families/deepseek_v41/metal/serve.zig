//! DeepSeek-V4.1 on Metal as the lanes serve it (serve_lanes.Model): the target, DSpark's pass, costs and hooks.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const weights = @import("weights.zig");
const dspark = @import("dspark.zig");
const prompt = @import("prompt.zig");
const target_ = @import("target.zig");
const pass_ = @import("pass.zig");
const pipelines = @import("pipelines.zig");
const vision_prepare = @import("vision_prepare.zig");

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
};

/// A position's draft candidates the pass keeps (the shape the lanes size their buffers by).
const candidates = 64;

pub const Metal = struct {
    a: Allocator,
    parts: Parts,
    opt: Options,
    t: *target_.Target,
    dp: ?*dspark.Pass = null,
    ad: pass_.Pass = undefined,
    prep: Prep = undefined,

    const Prep = struct { io: std.Io, cfg: @import("vision.zig").Config };

    pub fn open(a: Allocator, parts: Parts, opt: Options) !*Metal {
        const m = try a.create(Metal);
        errdefer a.destroy(m);
        m.* = .{ .a = a, .parts = parts, .opt = opt, .t = undefined };
        m.t = try target_.Target.init(a, parts.env, .{ .slots = opt.slots, .limit = opt.limit, .prompt = opt.prompt, .cache_bytes = opt.cache_bytes });
        errdefer m.t.deinit();
        const e = parts.env;
        if (opt.drafts) if (parts.dspark) |w| {
            m.dp = try dspark.Pass.init(a, e.device, e.queue, e.c, e.p, e.s, e.tables, parts.top, w, .{ .slots = opt.slots, .candidates = @min(candidates, e.c.vocab) });
            m.ad = .{ .x = m.dp.?, .t = m.t };
        };
        return m;
    }

    /// The lanes' view: one slot a stream, rows of 16 a forward, the release's DSpark shape on one device.
    pub fn model(m: *Metal) serve_lanes.Model {
        const c = m.parts.env.c;
        const shape: host.dspark.Shape = .{ .block = c.dspark_block, .noise = c.dspark_noise_token, .window = c.window, .rank = c.dspark_markov_rank, .hidden = c.hidden, .candidates = @min(candidates, c.vocab) };
        return .{
            .ptr = m,
            .vision = m.vision(),
            .name = "dsv41-metal",
            .slots = m.opt.slots,
            .limit = m.opt.limit,
            .world = 1,
            .prefill_step = m.opt.prompt.chunk,
            .forward_rows = pipelines.max_rows,
            .shape = shape,
            .target = targetFn,
            .pass = passFn,
            .calibrate = calibrate,
            .memory = .{ .ctx = m, .read = memory },
            .health = .{ .ctx = m, .read = healthy },
            .close = close,
            // the default cost table: TF_DSV41_CALIB has nothing to time on Metal
            .calib = .{ .mode = "default" },
        };
    }

    /// The server's image input: PNG, JPEG or WebP bytes to the tower's patches (null: no tower, or no limits given).
    fn vision(m: *Metal) ?api.Vision {
        const v = m.parts.env.vision orelse return null;
        const lim = m.parts.images orelse return null;
        m.prep = .{ .io = lim.io, .cfg = v.w.cfg };
        return .{ .ctx = &m.prep, .prepare = prepare, .image_token = v.w.cfg.image_token_id, .image_tokens = lim.tokens, .max_images = lim.count };
    }

    fn prepare(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
        const p: *const Prep = @ptrCast(@alignCast(ctx));
        const got = try vision_prepare.prepare(p.io, a, bytes, max_tokens, p.cfg);
        return .{ .pixels = got.pixels, .gh = got.gh, .gw = got.gw, .tokens = got.tokens, .hash = got.hash };
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

    fn calibrate(_: *anyopaque, _: Allocator, _: std.Io, _: []const u8, _: host.calib_env.Knobs, _: host.calib_env.Shape) anyerror!host.calib_lanes.Boot {
        return error.Unmeasured; // the mode is "default": never asked to measure
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
        if (m.dp) |p| p.deinit();
        m.t.deinit();
        if (m.parts.owner) |o| o.close(o.ctx);
        m.a.destroy(m);
    }
};
