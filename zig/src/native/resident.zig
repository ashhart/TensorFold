//! A served model's long-lived buffers in one residency set, kept wired by the server's idle keepalive.
const std = @import("std");
const mtl = @import("metal");

/// macOS unwires GPU memory after ~1-2 s idle and a request re-wires it all: keepalive ticks use the set.
pub const Resident = struct {
    set: mtl.ResidencySet,
    work: mtl.Queue,

    /// `buffers` join one set that `work` holds; the set is requested resident at once.
    pub fn init(device: mtl.Device, work: mtl.Queue, buffers: []const mtl.Buffer) !Resident {
        const set = try device.residencySet(buffers.len);
        errdefer set.deinit();
        for (buffers) |b| set.add(b);
        set.commit();
        set.requestResidency();
        work.addResidencySet(set);
        return .{ .set = set, .work = work };
    }

    /// More buffers into the set (a slot's memory taken after load), requested resident with the rest.
    pub fn add(r: *Resident, buffers: []const mtl.Buffer) void {
        for (buffers) |b| r.set.add(b);
        r.set.commit();
        r.set.requestResidency();
    }

    /// The idle keepalive's target: a tiny command buffer on `work` using the set; `r` must not move.
    pub fn target(r: *const Resident) mtl.keepalive.Target {
        return .{ .queue = r.work, .sets = @as(*const [1]mtl.ResidencySet, &r.set) };
    }

    pub fn deinit(r: *Resident) void {
        r.work.removeResidencySet(r.set);
        r.set.deinit();
    }
};
