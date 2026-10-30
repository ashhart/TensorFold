//! TF_DSV41_CALIB=measure (or `zig` without a stored table): the engine's draft-cost table over the CUDA model.
const std = @import("std");
const model = @import("cuda_model.zig");
const cm = @import("draft/calib_measure.zig");
const rowtab = @import("rowtab.zig");
const calib_env = @import("draft/calib_env.zig");
const calib_lanes = @import("calib_lanes.zig");
const gate_lanes = @import("gate_lanes.zig");
const sg = @import("sessions_gpu.zig");
const Allocator = std.mem.Allocator;

pub const promptOf = calib_lanes.promptOf;

/// The CUDA pieces the neutral measurement drives: the model, and the session store detached across it.
const CudaCalib = struct {
    m: *model.Model,
    store: ?*sg.Sessions = null,

    fn mark(ctx: *anyopaque) anyerror!u64 {
        const c: *CudaCalib = @ptrCast(@alignCast(ctx));
        return c.m.f.calibMark();
    }

    fn gather(ctx: *anyopaque, ours: []const i64, all: []i64) anyerror!void {
        const c: *CudaCalib = @ptrCast(@alignCast(ctx));
        try c.m.f.calibGather(ours, all);
    }

    fn hasPending(ctx: *anyopaque, slot: u32) bool {
        const c: *CudaCalib = @ptrCast(@alignCast(ctx));
        return c.m.gt.hasPending(slot);
    }

    fn dropWindow(ctx: *anyopaque, slot: u32) anyerror!void {
        const c: *CudaCalib = @ptrCast(@alignCast(ctx));
        try c.m.gt.dropWindow(slot);
    }

    /// Rank 0's session store out for the calibration texts, or back after them (neither resumed nor saved).
    fn detach(ctx: *anyopaque) void {
        const c: *CudaCalib = @ptrCast(@alignCast(ctx));
        std.mem.swap(?*sg.Sessions, &c.m.gt.sessions, &c.store);
    }
};

/// The CUDA model as calib_lanes sees it; `c` holds the detached session store across the run.
fn deviceOf(m: *model.Model, c: *CudaCalib) calib_lanes.Device {
    const rows: u32 = @min(cm.max_rows, if (m.gt.rows != null) rowtab.seg_max else m.gt.max_rows);
    return .{ .ctx = c, .target = m.gt.target(), .pass = m.draftPass(), .slots = m.slots, .limit = @intCast(m.f.opts.limit), .rows = rows, .world = m.f.comm.world(), .vocab = @intCast(m.f.cfg.vocab), .shape = gate_lanes.shapeOf(&m.f), .mark = CudaCalib.mark, .gather = CudaCalib.gather, .has_pending = CudaCalib.hasPending, .drop_window = CudaCalib.dropWindow, .detach = CudaCalib.detach };
}

/// Rank 0: the measurement on the loaded engine; the followers run it through the plan link.
pub fn measure(gpa: Allocator, m: *model.Model, ids: []const u32, how: cm.Method, rows_cap: u64) !calib_lanes.Measured {
    var c: CudaCalib = .{ .m = m };
    return calib_lanes.measure(gpa, deviceOf(m, &c), ids, how, rows_cap);
}

/// TF_DSV41_CALIB=measure / zig at open: prompt, measurement, stored entry, boot lines; table + stored file (or null).
pub fn boot(gpa: Allocator, io: std.Io, m: *model.Model, dir: []const u8, k: calib_env.Knobs, want: calib_env.Shape) !calib_lanes.Boot {
    var c: CudaCalib = .{ .m = m };
    return calib_lanes.boot(gpa, io, deviceOf(m, &c), dir, k, want);
}
