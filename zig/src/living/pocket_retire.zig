//! A replaced fact's neurons zeroed where they sit, and put back if the newer fact is not learned.
const std = @import("std");
const pk = @import("pocket.zig");
const pocket_fit = @import("pocket_fit.zig");
const decouple = @import("pocket_decouple.zig");
const subspace = @import("subspace.zig");
const clock = @import("pocket_clock.zig");
const pocket_dump = @import("pocket_dump.zig");
const joint = @import("pocket_joint.zig");
const Kind = pocket_fit.Kind;
const Fit = pocket_fit.Fit;
const fit = pocket_fit.fit;
const fitLayer = pocket_fit.fitLayer;
const dotBf16 = pocket_fit.dotBf16;
const fit_steps = pocket_fit.fit_steps;

/// A replaced fact's neurons, (layer, neuron) each, zeroed where they sit; their places stay taken.
pub fn retire(p: anytype, t: anytype, host: anytype, gone: []const [2]u16) !void {
    try p.load(t, host);
    const d = p.hidden;
    p.retired.clearRetainingCapacity();
    p.retired_w.clearRetainingCapacity();
    p.retired_marks = p.retired_lessons.items.len;
    for (gone) |g| {
        const slot = std.mem.indexOfScalar(usize, p.moe.items, g[0]) orelse continue;
        if (g[1] >= p.used.items[slot]) continue;
        for (p.learned.items) |e| if (e.slot == slot and e.neuron == g[1] and std.mem.indexOfScalar(u32, p.retired_lessons.items, e.lesson) == null) try p.retired_lessons.append(p.gpa, e.lesson);
        try p.retired.append(p.gpa, .{ slot, g[1] });
        try p.retired_w.appendSlice(p.gpa, p.neuron(p.up.items, slot, g[1], d));
        try p.retired_w.appendSlice(p.gpa, p.neuron(p.down.items, slot, g[1], d));
        if (@TypeOf(p.*).swiglu) try p.retired_w.appendSlice(p.gpa, p.neuron(p.gate.items, slot, g[1], d));
        @memset(p.neuron(p.up.items, slot, g[1], d), 0);
        @memset(p.neuron(p.down.items, slot, g[1], d), 0);
        if (@TypeOf(p.*).swiglu) @memset(p.neuron(p.gate.items, slot, g[1], d), 0);
        try p.upload(t, host, slot);
    }
    std.log.info("slide pocket: {d} @TypeOf(p.*).neurons of replaced facts zeroed", .{gone.len});
}

/// The last retire undone: its neurons as they were (the fact that replaced theirs was not learned).
pub fn unretire(p: anytype, t: anytype, host: anytype) !void {
    try p.load(t, host);
    const d = p.hidden;
    for (p.retired.items, 0..) |r, i| {
        const w = p.retired_w.items[@TypeOf(p.*).retired_rows * i * d ..][0 .. @TypeOf(p.*).retired_rows * d];
        @memcpy(p.neuron(p.up.items, r[0], r[1], d), w[0..d]);
        @memcpy(p.neuron(p.down.items, r[0], r[1], d), w[d..][0..d]);
        if (@TypeOf(p.*).swiglu) @memcpy(p.neuron(p.gate.items, r[0], r[1], d), w[2 * d ..][0..d]);
        try p.upload(t, host, r[0]);
    }
    std.log.info("slide pocket: {d} @TypeOf(p.*).neurons of replaced facts put back", .{p.retired.items.len});
    p.retired_lessons.shrinkRetainingCapacity(p.retired_marks);
    p.retired.clearRetainingCapacity();
    p.retired_w.clearRetainingCapacity();
}
