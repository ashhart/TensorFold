//! --drafter over the Qwen CUDA engine: a DFlash2 draft model, or the MoE target's own MTP layer.
const std = @import("std");
const lanes = @import("lanes");
const native = @import("cuda_native.zig");
const Draft = @import("cuda_draft.zig").Draft;
const Mtp = @import("cuda_mtp.zig").Mtp;
const draft_config = @import("cuda_draft_config.zig");
const cost_adapter = native.cost_adapter;
const Lanes = native.Lanes;

/// A drafter opened over the engine; DFlash2's `prepare` times the target's windows once the pool is reserved.
pub const Opened = struct {
    drafter: lanes.drafter.Drafter,
    ctx: *anyopaque,
    close: *const fn (*anyopaque) void,
    reserve: *const fn (*anyopaque, u32) anyerror!void,
    prepare: ?struct { ctx: *anyopaque, run: *const fn (*anyopaque) anyerror!lanes.drafter.Facts, bytes: usize } = null,
};

/// --drafter's directory over engine `p`: a DFlash2 draft (its config's dflash_config), else the MoE's MTP layer.
pub fn openDrafter(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8) !Opened {
    const l: *Lanes = @ptrCast(@alignCast(p));
    if (!l.e.g.moe() or draft_config.declared(gpa, io, dir)) return openDFlash(gpa, io, p, dir, bits);
    return openMtp(gpa, io, p, dir, bits);
}

/// The DFlash2 drafter in `dir` over engine `p`, 4-bit only, and the target's windows timed for the round loop.
fn openDFlash(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8) !Opened {
    if (bits != 4) return error.DrafterBitsUnsupported;
    const l: *Lanes = @ptrCast(@alignCast(p));
    if (l.e.g.exl3 and l.e.g.moe()) return error.Exl3DrafterUnsupported; // a dense checkpoint drafts as an MLX one
    if (l.caches.count() != 0 or l.pool.items.len != 0) return error.TapsAfterStreams;
    const d = try Draft.init(gpa, io, l.e, dir);
    errdefer d.deinit();
    try l.e.setTaps(d.taps());
    d.target_taps = l.e.tapBytes();
    cost_adapter.identify(l, dir, .dflash);
    l.calibration = d;
    l.draft = d;
    return .{ .drafter = d.drafter(), .ctx = d, .close = closeDrafter, .reserve = reserveDrafter, .prepare = .{ .ctx = l, .run = calibrateDrafter, .bytes = if (l.capacity < l.e.window_rows) try native.cacheBytes(l.e, l.e.window_rows, l.page_test) else 0 } };
}

/// The MTP layer beside the checkpoint in `dir` as the drafter over engine `p` (qwen3_5_moe), its ring the window.
fn openMtp(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8) !Opened {
    if (bits != 4) return error.DrafterBitsUnsupported;
    const l: *Lanes = @ptrCast(@alignCast(p));
    if (l.e.g.exl3 and l.e.g.moe()) return error.Exl3DrafterUnsupported; // a dense checkpoint drafts as an MLX one
    if (l.caches.count() != 0 or l.pool.items.len != 0) return error.TapsAfterStreams;
    const d = try Mtp.init(gpa, io, l.e, dir, l.capacity);
    errdefer d.deinit();
    try l.e.setTaps(d.taps());
    cost_adapter.identify(l, dir, .mtp);
    try l.timeWindows(d, null);
    l.mtp = d;
    return .{ .drafter = d.drafter(), .ctx = d, .close = closeMtp, .reserve = reserveMtp };
}

fn reserveMtp(p: *anyopaque, streams: u32) anyerror!void {
    const d: *Mtp = @ptrCast(@alignCast(p));
    try d.reserve(streams);
}

fn closeMtp(p: *anyopaque) void {
    const d: *Mtp = @ptrCast(@alignCast(p));
    d.deinit();
}

fn calibrateDrafter(p: *anyopaque) !lanes.drafter.Facts {
    const l: *Lanes = @ptrCast(@alignCast(p));
    const d = l.calibration orelse return error.MissingCalibrationDrafter;
    if (d.n_costs == 0) {
        if (l.pool.items.len == 0 and !l.engine_paged) return error.MissingReservedQwenCache;
        // engine_paged: no reserved window, so the timing makes a window_rows cache of its own (null)
        try l.timeWindows(d, if (l.pool.items.len > 0) l.pool.items[0] else null);
    }
    if (d.n_costs != l.e.window_rows) return error.IncompleteWindowMeasurements;
    l.calibration = null;
    var phases = l.startup;
    phases.add(d.startup);
    phases.print();
    return d.drafter().facts();
}

fn reserveDrafter(p: *anyopaque, streams: u32) anyerror!void {
    const d: *Draft = @ptrCast(@alignCast(p));
    try d.reserve(streams);
}

fn closeDrafter(p: *anyopaque) void {
    const d: *Draft = @ptrCast(@alignCast(p));
    d.deinit();
}
