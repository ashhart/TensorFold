//! Flash Next residency includes the runner, transport, snapshot pool, shared sessions and optional image tower.
const std = @import("std");
const mtl = @import("metal");
const Resident = mtl.resident.Resident;
const PairWarm = @import("flashnext_pair_warm.zig").PairWarm;

fn initial(h: anytype, list: *std.ArrayList(mtl.Buffer)) !void {
    try list.appendSlice(h.gpa, h.eng.r.held.items);
    if (h.eng.r.tp) |tp| try mtl.resident.fields.append(h.gpa, list, tp);
    for (h.eng.snap_pool.bufs) |b| if (b) |buffer| try list.append(h.gpa, buffer);
    if (h.vis) |v| try list.appendSlice(h.gpa, &v.buffers());
}

/// The idle keepalive's command buffer without its flush, for the engine's staggered segments to wait for.
fn probeWarm(ctx: *const anyopaque) mtl.CommandBuffer {
    const t: *const mtl.keepalive.Target = @ptrCast(@alignCast(ctx));
    return t.probe();
}

fn attach(h: anytype, io: std.Io, buffers: []const mtl.Buffer, label: []const u8) !void {
    // Keep the pair's explicit residency request independent of command-queue attachment.
    h.resident = try Resident.holdUnderOptions(h.gpa, io, h.eng.r.device, h.eng.r.queue, buffers, label, if (h.load_limit > 0) h.load_limit else null, .{ .queue_attach = h.eng.r.tp == null });
    if (h.resident) |*r| {
        h.warm = r.target();
        h.eng.r.follow = r.follow();
        h.eng.r.warm_probe = .{ .ctx = &h.warm, .run = probeWarm };
        h.eng.snap_pool.follow = r.follow();
    }
}

pub fn hold(h: anytype) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    try initial(h, &list);
    if (h.eng.r.tp) |tp| h.pair_warm = PairWarm.init(h.eng.r.device, h.io, &tp.request_gate.active) catch |err| blk: {
        std.log.warn("flash next: no residency probe during paired requests ({s})", .{@errorName(err)});
        break :blk null;
    };
    errdefer dropPairWarm(h);
    if (h.pair_warm) |*w| try list.append(h.gpa, w.target.scratch.?);
    try attach(h, h.io, list.items, "flash next");
    if (h.resident == null) dropPairWarm(h);
    if (h.pair_warm) |*w| {
        w.track(h.warm);
        w.start() catch |err| std.log.warn("flash next: no residency probe during paired requests ({s})", .{@errorName(err)});
    }
}

fn dropPairWarm(h: anytype) void {
    if (h.pair_warm) |*w| w.deinit();
    h.pair_warm = null;
}

pub fn drop(h: anytype) void {
    if (h.pair_warm) |*w| w.stop();
    release(h);
    dropPairWarm(h);
}

fn release(h: anytype) void {
    h.eng.r.follow = null;
    h.eng.r.warm_probe = null;
    h.eng.snap_pool.follow = null;
    if (h.resident) |*r| r.deinit();
    h.resident = null;
    h.warm = .{ .queue = h.eng.r.queue };
}

/// Join a failed open's follower before releasing the callbacks and memory it borrows.
pub fn unwind(h: anytype) void {
    @import("flashnext_follow.zig").stop(h);
    drop(h);
}

pub fn holdShared(h: anytype) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    try initial(h, &list);
    for (h.back.sessions) |s| try list.appendSlice(h.gpa, s.owned.items);
    try list.append(h.gpa, h.back.hidden.b);
    try attach(h, h.host.io, list.items, "flash next shared");
    if (h.resident) |*r| for (h.back.sessions) |*session| session.setResident(r.follow());
}

pub fn dropShared(h: anytype) void {
    for (h.back.sessions) |*session| session.setResident(null);
    release(h);
}
