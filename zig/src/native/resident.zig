//! A served model's long-lived buffers in one residency set, kept wired by the server's idle keepalive.
const std = @import("std");
const mtl = @import("metal");
const cache_fit = @import("cache_fit.zig");

/// macOS unwires GPU memory after ~1-2 s idle and a request re-wires it all: keepalive ticks use the set.
pub const Resident = struct {
    set: mtl.ResidencySet,
    work: mtl.Queue,
    scratch: mtl.Buffer, // inside the set: the keepalive's tick copies 16 bytes within it
    bytes: u64, // what the set holds: kept inside the 70% load limit with the margin
    ram: u64,

    /// One set of `buffers` that `work` holds, requested resident at once; null (logged for `name`) past the 70% limit.
    pub fn hold(device: mtl.Device, work: mtl.Queue, buffers: []const mtl.Buffer, name: []const u8) !?Resident {
        const ram = cache_fit.ram() orelse 0;
        const bytes = sum(buffers) + 64;
        if (!cache_fit.fitsLoadLimit(bytes, ram)) {
            std.log.warn("{s}: no residency set: its {d:.1} GiB and a {d} GiB margin pass 70% of {d:.0} GiB of RAM, so the first request after an idle second waits for the model's memory", .{ name, cache_fit.gibs(bytes), cache_fit.MARGIN >> 30, cache_fit.gibs(ram) });
            return null;
        }
        const set = try device.residencySet(buffers.len + 1);
        errdefer set.deinit();
        for (buffers) |b| set.add(b);
        const scratch = try device.buffer(64, mtl.ResourceOptions.shared);
        errdefer scratch.deinit();
        set.add(scratch);
        set.commit();
        set.requestResidency();
        work.addResidencySet(set);
        std.log.info("{s}: {d:.1} GiB kept wired between requests, inside 70% of {d:.0} GiB of RAM", .{ name, cache_fit.gibs(bytes), cache_fit.gibs(ram) });
        return .{ .set = set, .work = work, .scratch = scratch, .bytes = bytes, .ram = ram };
    }

    /// More buffers into the set (a slot's memory taken after load) while it stays inside the 70% limit; else left out.
    pub fn add(r: *Resident, buffers: []const mtl.Buffer) void {
        const bytes = r.bytes + sum(buffers);
        if (!cache_fit.fitsLoadLimit(bytes, r.ram)) return std.log.info("residency set: {d:.1} GiB more would pass 70% of RAM: these buffers stay out of it", .{cache_fit.gibs(bytes - r.bytes)});
        for (buffers) |b| r.set.add(b);
        r.set.commit();
        r.set.requestResidency();
        r.bytes = bytes;
    }

    /// The idle keepalive's target: a tiny command buffer on `work` using the set; `r` must not move.
    pub fn target(r: *const Resident) mtl.keepalive.Target {
        return .{ .queue = r.work, .sets = @as(*const [1]mtl.ResidencySet, &r.set), .scratch = r.scratch };
    }

    pub fn deinit(r: *Resident) void {
        r.work.removeResidencySet(r.set);
        r.set.deinit();
        r.scratch.deinit();
    }
};

fn sum(buffers: []const mtl.Buffer) u64 {
    var n: u64 = 0;
    for (buffers) |b| n += b.length();
    return n;
}
