//! DeepSeek-V4.1's CUDA entry point for the native server: the model behind serve_lanes.Model; other ranks follow.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const api = @import("engine_api");
const model = @import("cuda_model.zig");
const calib_env = @import("draft/calib_env.zig");
const calib_gpu = @import("calib_gpu.zig");
const calib_lanes = @import("calib_lanes.zig");
const gate_lanes = @import("gate_lanes.zig");
const ph = @import("phases.zig");
const serve_lanes = @import("serve_lanes.zig");
const Allocator = std.mem.Allocator;
const sched_host = @import("sched_host.zig");
const iface = @import("draft/iface.zig");

pub const backends: []const []const u8 = &.{"cuda"};
pub const families: []const api.Family = &.{.{ .model_type = "deepseek_v41", .formats = &.{"exl3"}, .raw_text = false }};

pub fn chip(_: Allocator) ?[]const u8 {
    return null;
}

/// The engine for `o.dir`, or null with `problem` set (the native_engines contract).
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (@import("sys").experimental.refusal(o.model_type, @import("sys").getenv("TF_EXPERIMENTAL"))) |text| {
        problem.* = text;
        return null;
    }
    if (!std.mem.eql(u8, o.model_type, "deepseek_v41")) {
        problem.* = try std.fmt.allocPrint(a, "this build serves deepseek_v41 checkpoints only, not {s}", .{o.model_type});
        return null;
    }
    const assets = std.mem.span(std.c.getenv("TF_DSV41_ASSETS") orelse {
        problem.* = model.asset_names.unset;
        return null;
    });
    const s = Served.open(gpa, io, o.dir, assets, o.drafts, o.context) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the DeepSeek-V4.1 CUDA engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
    return .{ .engine = s.engine(), .close = Served.close, .ctx = s, .halt = Served.halt };
}

/// The CUDA backend behind serve_lanes: the loaded model and the sched_host the rounds hook adds.
const CudaBackend = struct {
    gpa: Allocator,
    m: *model.Model,
    /// TF_DSV41_PREFILL_PIECES=1 (sched_host.zig): the batcher plans the host's rounds
    sched: ?*sched_host.SchedHost = null,

    fn target(ctx: *anyopaque) iface.Target {
        const b: *CudaBackend = @ptrCast(@alignCast(ctx));
        return b.m.gt.target();
    }

    fn pass(ctx: *anyopaque) ?iface.Pass {
        const b: *CudaBackend = @ptrCast(@alignCast(ctx));
        return b.m.draftPass();
    }

    fn calibrate(ctx: *anyopaque, gpa: Allocator, io: std.Io, dir: []const u8, knobs: calib_env.Knobs, want: calib_env.Shape) anyerror!calib_lanes.Boot {
        const b: *CudaBackend = @ptrCast(@alignCast(ctx));
        return calib_gpu.boot(gpa, io, b.m, dir, knobs, want);
    }

    /// The phases report's "engram" and "rowgraphs" counters (cumulative): row reader records and the gate's rounds.
    fn writeEngram(ctx: *anyopaque, w: *std.Io.Writer) anyerror!void {
        const b: *CudaBackend = @ptrCast(@alignCast(ctx));
        const m = b.m;
        try w.writeAll(", \"engram\": {");
        if (m.f.engram_rows) |r| {
            const st = r.stats;
            try w.print("\"hits\": {d}, \"waited\": {d}, \"misses\": {d}, \"issued\": {d}, \"reads\": {d}, \"bytes\": {d}, \"wait_ms\": {d:.2}", .{ st.hits, if (@hasField(@TypeOf(st), "waited")) st.waited else 0, st.misses, st.issued, st.reads, st.bytes, @as(f64, @floatFromInt(if (@hasField(@TypeOf(st), "wait_ns")) st.wait_ns else 0)) / 1e6 });
        } else try w.writeAll("\"rows\": null");
        if (@import("engram_gate_fwd.zig").of(&m.f)) |g| try w.print(", \"gate_rounds\": {d}, \"gate_syncs\": {d}, \"gate_warm\": {d}, \"gate_failed\": {d}", .{ g.stats.rounds, g.stats.syncs, g.stats.warm, g.stats.failed });
        try w.writeAll("}");
        // the row windows' first uses (batch.zig): programs emitted and graphs captured, their host ms
        if (m.gt.rows) |rows_| {
            const st = rows_.stats;
            try w.print(", \"rowgraphs\": {{\"windows\": {d}, \"rows\": {d}, \"padded\": {d}, \"builds\": {d}, \"build_ms\": {d:.2}, \"captures\": {d}, \"capture_ms\": {d:.2}, \"eager\": {d}, \"prep_ms\": {d:.2}, \"stage_ms\": {d:.2}, \"launch_ms\": {d:.2}, \"keep_ms\": {d:.2}}}", .{ st.windows, st.rows, st.padded, st.builds, @as(f64, @floatFromInt(st.build_ns)) / 1e6, st.captures, @as(f64, @floatFromInt(st.capture_ns)) / 1e6, st.eager, @as(f64, @floatFromInt(st.prep_ns)) / 1e6, @as(f64, @floatFromInt(st.stage_ns)) / 1e6, @as(f64, @floatFromInt(st.launch_ns)) / 1e6, @as(f64, @floatFromInt(st.keep_ns)) / 1e6 });
        }
    }

    fn syncStream(ctx: *anyopaque) anyerror!void {
        const b: *CudaBackend = @ptrCast(@alignCast(ctx));
        try b.m.stream.synchronize();
    }

    fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
        const u = @import("cuda").usage(reset_peak);
        return .{ .active = u.device, .cache = 0, .peak = u.peak };
    }

    fn contextHealthy(ctx: *anyopaque) bool {
        const b: *CudaBackend = @ptrCast(@alignCast(ctx));
        return b.m.ctx.healthy();
    }

    fn bindContext(ctx: *anyopaque) anyerror!void {
        const b: *CudaBackend = @ptrCast(@alignCast(ctx));
        try b.m.ctx.makeCurrent();
    }

    /// TF_DSV41_PREFILL_PIECES=1: the batcher plans the rounds; served shape only, same setting on every rank.
    fn rounds(ctx: *anyopaque, gpa: Allocator, io: std.Io, host: *api.LaneHost) anyerror!void {
        const b: *CudaBackend = @ptrCast(@alignCast(ctx));
        const m = b.m;
        if (!model.Model.piecesEnv()) {
            std.log.info("dsv41 rounds: whole prompts at admission (TF_DSV41_PREFILL_PIECES=1: the reference server's pieces between decode rounds)", .{});
            return;
        }
        if (!(m.gt.own_prefill and m.gt.prompt_tail == .verify)) {
            if (!quiet) std.log.warn("TF_DSV41_PREFILL_PIECES=1 needs TF_DSV41_OWN_PREFILL=1 and TF_DSV41_PROMPT_TAIL=verify (the served shape); prompts run whole", .{});
            return;
        }
        const sh = try gpa.create(sched_host.SchedHost);
        sh.* = sched_host.SchedHost.init(gpa, io, &m.gt, try sched_host.SchedHost.settings(&m.gt));
        b.sched = sh;
        host.rounds = sh.rounds();
        m.gt.defer_release = true;
        var buf: [256]u8 = undefined;
        std.log.info("dsv41 rounds: {s}", .{sh.describe(&buf)});
    }

    /// The sched_host's releases settle before it dies (its planned admissions' puts are the target's).
    fn settle(ctx: *anyopaque) void {
        const b: *CudaBackend = @ptrCast(@alignCast(ctx));
        if (b.sched) |sh| {
            b.m.gt.settleReleases();
            sh.deinit();
            b.gpa.destroy(sh);
            b.sched = null;
        }
    }

    /// The followers' loops end.
    fn stop(ctx: *anyopaque) void {
        const b: *CudaBackend = @ptrCast(@alignCast(ctx));
        b.m.f.stop() catch {};
    }

    /// TF_DSV41_PHASES: the forward's timers point at the served Phases (null clears them).
    fn setPhases(ctx: *anyopaque, p: ?*ph.Phases) void {
        const b: *CudaBackend = @ptrCast(@alignCast(ctx));
        b.m.f.phases = p;
    }

    fn close(ctx: *anyopaque) void {
        const b: *CudaBackend = @ptrCast(@alignCast(ctx));
        b.m.close();
        b.gpa.destroy(b);
    }
};

/// Rank 0's served engine: the neutral lanes over the CUDA model (`m`, for the slot and session check CLIs).
pub const Served = struct {
    inner: *serve_lanes.Served,
    m: *model.Model,

    pub fn engine(s: *Served) api.Engine {
        return s.inner.engine();
    }

    /// Rank 0's engine over a loaded model (`drafts`: DSpark's pass, else serial windows through the same lanes).
    pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, assets: []const u8, drafts: bool, context: ?i64) !*Served {
        const m = try model.Model.open(gpa, io, dir, assets, .{ .drafts = drafts, .own_prefill = model.ownPrefillEnv(), .layers = try model.layersEnv(gpa) });
        errdefer m.close();
        if (m.rank() != 0) return error.NotRankZero; // the other ranks run `tf-dsv41 follow`
        const b = try gpa.create(CudaBackend);
        errdefer gpa.destroy(b);
        b.* = .{ .gpa = gpa, .m = m };
        const backend: serve_lanes.Model = .{
            .ptr = b,
            .name = "dsv41-cuda",
            .slots = m.slots,
            .limit = @intCast(m.f.opts.limit),
            .world = m.f.comm.world(),
            .prefill_step = m.gt.prefill_rows,
            .forward_rows = if (m.rows) |rows_| rows_.cap else 16,
            .shape = gate_lanes.shapeOf(&m.f),
            .target = CudaBackend.target,
            .pass = CudaBackend.pass,
            .calibrate = CudaBackend.calibrate,
            .sync = .{ .ctx = b, .run = CudaBackend.syncStream },
            .extra = .{ .ctx = b, .write = CudaBackend.writeEngram },
            .on_thread = .{ .ctx = b, .run = CudaBackend.bindContext },
            .memory = .{ .read = CudaBackend.readMemory },
            .health = .{ .ctx = b, .read = CudaBackend.contextHealthy },
            .rounds = CudaBackend.rounds,
            .settle = CudaBackend.settle,
            .stop = CudaBackend.stop,
            .set_phases = CudaBackend.setPhases,
            .close = CudaBackend.close,
        };
        const s = try gpa.create(Served);
        errdefer gpa.destroy(s);
        s.* = .{ .inner = try serve_lanes.Served.open(gpa, io, backend, dir, context), .m = m };
        return s;
    }

    /// A drain's deadline: rounds end at a boundary first, so `close` can stop the followers cleanly (exit 0, not 70).
    pub fn halt(ctx: *anyopaque, reason: []const u8) void {
        const s: *Served = @ptrCast(@alignCast(ctx));
        serve_lanes.Served.halt(s.inner, reason);
    }

    pub fn close(ctx: *anyopaque) void {
        const s: *Served = @ptrCast(@alignCast(ctx));
        const gpa = s.inner.gpa;
        serve_lanes.Served.close(s.inner);
        gpa.destroy(s);
    }
};

comptime {
    if (@import("builtin").is_test) _ = serve_lanes; // its tests (`draft sources:`) run under this root's host step
}
