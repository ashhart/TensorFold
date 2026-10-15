//! A fact deleted on request: its neurons' inputs zeroed, their layers rounded as a save writes them, kept to put back.
const std = @import("std");
const pocket = @import("pocket.zig");

/// What the last delete changed, bit for bit as it was: whole layers' neurons and counts, every neuron's record.
pub const Kept = struct {
    held: bool = false, // a delete not yet written by a save nor put back
    stale: bool = false, // put back in memory, but a layer's upload failed: kept until the device takes them all
    slots: std.ArrayList(usize) = .empty, // the pocket slots it changed
    up: std.ArrayList(f32) = .empty, // [slots, neurons, D]: their input weights before
    down: std.ArrayList(f32) = .empty, // [slots, neurons, D]: their outputs before
    used: std.ArrayList(usize) = .empty, // [slots]: their counts before
    learned: std.ArrayList(pocket.Learned) = .empty, // the session's neuron records before
    learned_before: usize = 0,
    marks: usize = 0, // retired lessons before

    pub fn deinit(k: *Kept, gpa: std.mem.Allocator) void {
        inline for (.{ "slots", "up", "down", "used", "learned" }) |f| @field(k, f).deinit(gpa);
    }

    fn clear(k: *Kept) void {
        inline for (.{ "slots", "up", "down", "used", "learned" }) |f| @field(k, f).clearRetainingCapacity();
        k.held = false;
        k.stale = false;
    }
};

/// Whether a delete waits to be written: a save then writes the pocket even when nothing is left in it.
pub fn pending(p: anytype) bool {
    return p.forgot.held;
}

/// Whether a put-back is in memory but not yet on the device: replies are refused until `unforget` lands it.
pub fn stale(p: anytype) bool {
    return p.forgot.stale;
}

/// The last delete written into the shards: nothing is left to put back.
pub fn settle(p: anytype) void {
    p.forgot.clear();
}

/// Neurons `gone` names that hold anything silenced, their layers rounded, freed at the top; how many held anything.
pub fn forget(p: anytype, t: anytype, host: anytype, gone: []const [2]u16) !usize {
    try p.load(t, host);
    const d = t.config().hidden;
    const n = pocket.neurons * d;
    const k = &p.forgot;
    // a put-back the device has not taken yet is never dropped for a new delete
    if (k.stale) return error.PutBackNotOnDevice;
    k.clear();
    var held: usize = 0;
    for (gone) |g| {
        const slot = std.mem.indexOfScalar(usize, p.moe.items, g[0]) orelse continue;
        if (g[1] >= p.used.items[slot] or std.mem.allEqual(f32, p.neuron(p.up.items, slot, g[1], d), 0)) continue;
        held += 1;
        if (std.mem.indexOfScalar(usize, k.slots.items, slot) == null) try k.slots.append(p.gpa, slot);
    }
    // a replaced fact's neurons are zero already: nothing changes, nothing is kept
    if (held == 0) return 0;
    for (k.slots.items) |slot| {
        try k.up.appendSlice(p.gpa, p.up.items[slot * n ..][0..n]);
        try k.down.appendSlice(p.gpa, p.down.items[slot * n ..][0..n]);
        try k.used.append(p.gpa, p.used.items[slot]);
    }
    try k.learned.appendSlice(p.gpa, p.learned.items);
    // everything that can fail before the pocket moves is done here; anything that fails after puts it all back
    try p.retired_lessons.ensureUnusedCapacity(p.gpa, p.learned.items.len); // a lesson at most for each record
    k.learned_before = p.learned_before;
    k.marks = p.retired_lessons.items.len;
    k.held = true;
    cut(p, t, host, gone) catch |e| {
        // put back whole; if the device cannot take it, that is the failure to report, the snapshot kept
        _ = unforget(p, t, host) catch |back| {
            std.log.warn("slide pocket: a delete failed ({t}), and its put-back did not reach the device ({t})", .{ e, back });
            return back;
        };
        return e;
    };
    std.log.info("slide pocket: {d} neurons of a deleted fact zeroed in {d} layers", .{ held, k.slots.items.len });
    return held;
}

/// The cut itself, after the snapshot: input weights zeroed, records out, layers rounded, freed and uploaded.
fn cut(p: anytype, t: anytype, host: anytype, gone: []const [2]u16) !void {
    const d = t.config().hidden;
    const n = pocket.neurons * d;
    const k = &p.forgot;
    for (gone) |g| {
        const slot = std.mem.indexOfScalar(usize, p.moe.items, g[0]) orelse continue;
        if (g[1] >= p.used.items[slot]) continue;
        // its input weights zeroed, so it never fires; its output stays, so the 4-bit groups it shares keep every bit
        @memset(p.neuron(p.up.items, slot, g[1], d), 0);
        // its record goes, so no later fact on the neuron is refitted as this one; its lesson's answers retire
        var i: usize = 0;
        while (i < p.learned.items.len) {
            const e = p.learned.items[i];
            if (e.slot != slot or e.neuron != g[1]) {
                i += 1;
                continue;
            }
            if (std.mem.indexOfScalar(u32, p.retired_lessons.items, e.lesson) == null) p.retired_lessons.appendAssumeCapacity(e.lesson);
            _ = p.learned.orderedRemove(i);
            if (i < p.learned_before) p.learned_before -= 1;
        }
    }
    for (k.slots.items) |slot| {
        // the layer as its next save writes it, so the checks after this read what the shards will hold
        try host.pocketRound(p.up.items[slot * n ..][0..n], p.down.items[slot * n ..][0..n]);
        // neurons at the top whose input weights are zero are free; one below a kept neuron stays taken
        while (p.used.items[slot] > 0 and std.mem.allEqual(f32, p.neuron(p.up.items, slot, p.used.items[slot] - 1, d), 0)) p.used.items[slot] -= 1;
        try p.upload(t, host, slot);
    }
}

/// The last delete undone bit for bit; a layer the device does not take keeps the snapshot, and a call again retries.
pub fn unforget(p: anytype, t: anytype, host: anytype) !usize {
    const k = &p.forgot;
    if (!k.held) return 0;
    try p.load(t, host);
    try p.learned.ensureTotalCapacity(p.gpa, k.learned.items.len);
    // the pocket whole as it was before any layer goes to the forward, so a failed upload leaves no layer half put back
    const n = pocket.neurons * t.config().hidden;
    for (k.slots.items, 0..) |slot, j| {
        @memcpy(p.up.items[slot * n ..][0..n], k.up.items[j * n ..][0..n]);
        @memcpy(p.down.items[slot * n ..][0..n], k.down.items[j * n ..][0..n]);
        p.used.items[slot] = k.used.items[j];
    }
    p.learned.clearRetainingCapacity();
    p.learned.appendSliceAssumeCapacity(k.learned.items);
    p.learned_before = k.learned_before;
    p.retired_lessons.shrinkRetainingCapacity(k.marks);
    const back = k.slots.items.len;
    var failed: ?anyerror = null;
    for (k.slots.items) |slot| p.upload(t, host, slot) catch |e| {
        failed = failed orelse e;
    };
    if (failed) |e| {
        k.stale = true;
        std.log.warn("slide pocket: a deleted fact's {d} layers are put back in memory, but not all on the device ({t}): replies are refused until they are", .{ back, e });
        return error.PutBackNotOnDevice;
    }
    k.clear();
    std.log.info("slide pocket: a deleted fact's {d} layers put back as they were", .{back});
    return back;
}
