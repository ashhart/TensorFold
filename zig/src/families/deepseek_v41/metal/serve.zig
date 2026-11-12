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
const calib = @import("calib.zig");

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
};

pub const Owner = struct { ctx: *anyopaque, close: *const fn (ctx: *anyopaque) void };

pub const Options = struct {
    slots: u32 = 4,
    limit: u32,
    /// DSpark's pass when the checkpoint has one (false: serial windows through the same lanes)
    drafts: bool = true,
    cache_bytes: u64 = 0,
    prompt: prompt.Opts = .{},
    /// the draft cost table is timed at load unless TF_DSV41_CALIB names another source (false: the defaults)
    measure_costs: bool = false,
    /// the device and weights a stored cost table is specific to (0: none named)
    identity: u64 = 0,
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
        const dshape = m.dsparkShape();
        return .{
            .ptr = m,
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
            .memory = .{ .ctx = m, .read = memory },
            .health = .{ .ctx = m, .read = healthy },
            .close = close,
            .calib_id = m.opt.identity,
            .calib = m.knobs(),
        };
    }

    fn dsparkShape(m: *const Metal) host.dspark.Shape {
        const c = m.parts.env.c;
        return .{ .block = c.dspark_block, .noise = c.dspark_noise_token, .window = c.window, .rank = c.dspark_markov_rank, .hidden = c.hidden, .candidates = @min(candidates, c.vocab) };
    }

    /// TF_DSV41_CALIB and its neighbours as the environment gives them; unset: timed when asked for, else defaults.
    fn knobs(m: *const Metal) host.calib_env.Knobs {
        var k = host.calib_env.Knobs.fromEnv();
        if (k.mode == null) k.mode = if (m.opt.measure_costs) "measure" else "default";
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
        if (m.dp) |p| p.deinit();
        m.t.deinit();
        if (m.parts.owner) |o| o.close(o.ctx);
        m.a.destroy(m);
    }
};
