//! The saved profile while it still describes this machine and binary, else a new scan saved in its place.
const std = @import("std");
const machine = @import("machine.zig");
const support = @import("support.zig");
const profile = @import("profile.zig");
const Allocator = std.mem.Allocator;
const Machine = machine.Machine;

/// This binary's side: its version, what it serves, and how to ask the GPU's driver (slow: `tensorfold device` only).
pub const Binary = struct {
    version: []const u8,
    offer: support.Offer,
    /// The driver's GPUs (CUDA), or null when it does not answer; null itself in a build that cannot ask.
    driver: ?*const fn (Allocator) ?[]const machine.Gpu = null,
};

pub const Current = struct {
    doc: profile.Doc,
    /// What this binary derives for the machine (the saved record is the scanning binary's).
    support: support.Support,
    /// Why this call scanned; null when the saved profile was current.
    scanned: ?profile.Due = null,
    /// The new scan reached the file (a read-only home or no home leaves it unsaved).
    saved: bool = false,
};

/// `quick` is this call's own listing (read.quick); `full` asks the driver when the profile is only a quick one.
pub fn current(a: Allocator, io: std.Io, file: ?[]const u8, quick: Machine, bin: Binary, now: []const u8, asked: bool, full: bool) !Current {
    const fp = machine.fingerprint(quick);
    const old = if (file) |f| profile.load(a, io, f) else null;
    const ask = full and bin.driver != null and quick.os != .macos;
    const why = profile.due(old, &fp, bin.version, asked, ask) orelse {
        const doc = old.?;
        return .{ .doc = doc, .support = try support.derive(a, doc.machine, bin.offer) };
    };
    var m = quick;
    // a Mac's sysctl readings are its whole story; elsewhere a scan is full once the driver has listed the GPUs
    var kind: profile.Scan = if (quick.os == .macos) .full else .quick;
    if (ask) if (bin.driver.?(a)) |gpus| if (gpus.len > 0) {
        m.gpus = try merge(a, quick.gpus, gpus);
        kind = .full;
    };
    const s = try support.derive(a, m, bin.offer);
    const doc: profile.Doc = .{ .scanned_at = now, .binary_version = bin.version, .fingerprint = try a.dupe(u8, &fp), .scan = kind, .machine = m, .support = s };
    var saved = false;
    if (file) |f| {
        profile.save(a, io, f, doc) catch {};
        saved = profile.load(a, io, f) != null;
    }
    return .{ .doc = doc, .support = s, .scanned = why, .saved = saved };
}

/// The saved profile when it is current for this machine and binary; null otherwise. Never scans or writes.
pub fn fresh(a: Allocator, io: std.Io, file: ?[]const u8, quick: Machine, version: []const u8) ?profile.Doc {
    const fp = machine.fingerprint(quick);
    const old = profile.load(a, io, file orelse return null) orelse return null;
    return if (profile.due(old, &fp, version, false, false) == null) old else null;
}

/// The driver's GPUs in its own order (CUDA's ordinals), in place of the OS's listing of that vendor; the rest after.
pub fn merge(a: Allocator, listed: []const machine.Gpu, driver: []const machine.Gpu) ![]const machine.Gpu {
    var out: std.ArrayList(machine.Gpu) = .empty;
    try out.appendSlice(a, driver);
    for (listed) |g| {
        const replaced = for (driver) |d| {
            if (d.vendor == g.vendor) break true;
        } else false;
        if (!replaced) try out.append(a, g);
    }
    return out.items;
}
