//! Re-drawn router rows for the open fact's existing neurons; its outputs and Adam state keep going.
const std = @import("std");
const examples = @import("lesson_examples.zig");
const fitting = @import("pocket_fit.zig");
const pockets = @import("pocket.zig");

/// Own the new wordings before replacing the old ones; answers and the lesson's undo mark stay put.
pub fn begin(l: anytype) !void {
    if (!l.lesson.more) return error.NothingToContinue;
    const p = &l.pockets;
    // An old wording that owns a guarded answer cannot be freed by a router-only refresh.
    for (p.keeps.items) |k| for ([_][]const examples.Example{ l.owned.asked, l.owned.unasked }) |xs| {
        for (xs) |ex| if (ex.ids.ptr == k.ex.ids.ptr) {
            fallback(p, "RouterWordingOwnsAnswer");
            return;
        };
    };
    var fresh = try examples.Wordings.init(l.gpa, l.lesson);
    defer fresh.deinit(l.gpa);
    fresh.install(l.gpa, &l.owned);
    l.stay[3] = l.owned.unasked;
    p.prompts = l.owned.asked;
    p.rows.clearRetainingCapacity();
    p.labels.clearRetainingCapacity();
    p.subjects.clearRetainingCapacity();
    p.leads.clearRetainingCapacity();
    p.read.keys.clearRetainingCapacity();
    p.read.talks.clearRetainingCapacity();
    p.read.whole = false;
    p.refreshing = true;
    p.next(.read);
}

/// Fit input routes in their existing slots, then resume ordinary output steps without reinitializing Adam.
pub fn fit(p: anytype, t: anytype, host: anytype) !bool {
    const P = @TypeOf(p.*);
    const d = p.hidden;
    const n = p.moe.items.len;
    const m = p.labels.items.len;
    const up = try p.gpa.dupe(f32, p.up.items);
    defer p.gpa.free(up);
    const kinds = try p.gpa.alloc(fitting.Kind, m);
    defer p.gpa.free(kinds);
    var pos: std.ArrayList(u16) = .empty;
    defer pos.deinit(p.gpa);
    try pos.appendSlice(p.gpa, p.pos.items[0..p.pos_before]);
    const learned = try p.gpa.dupe(pockets.Learned, p.learned.items);
    defer p.gpa.free(learned);
    for (p.taken.items) |x| {
        for (p.labels.items, kinds) |label, *kind| kind.* = pockets.kindFor(label, x.key);
        const u = p.neuron(up, x.slot, x.neuron, d);
        var result: fitting.Fit = undefined;
        fitting.fitLayer(p.gpa, p.rows.items, kinds, n, x.slot, d, u, &result, P.of.neuron.margin);
        if (result.fires == std.math.maxInt(usize)) {
            fallback(p, "RouterFitFailed");
            return false;
        }
        for (learned[p.learned_before..]) |*e| {
            if (e.slot != x.slot or e.neuron != x.neuron) continue;
            e.start = pos.items.len / d;
            for (kinds, 0..) |kind, i| if (kind == .fact) {
                try pos.appendSlice(p.gpa, p.rows.items[(i * n + x.slot) * d ..][0..d]);
            };
            e.count = pos.items.len / d - e.start;
        }
    }
    // Every allocation precedes changes to the live router; a partial upload closes serving until undo succeeds.
    try p.pos.ensureTotalCapacity(p.gpa, pos.items.len);
    @memcpy(p.up.items, up);
    for (p.taken.items) |x| p.gateFrom(x.slot, x.neuron);
    p.pos.clearRetainingCapacity();
    p.pos.appendSliceAssumeCapacity(pos.items);
    @memcpy(p.learned.items, learned);
    p.restore_pending = true;
    for (0..n) |slot| try p.upload(t, host, slot);
    p.restore_pending = false;
    p.more();
    return false;
}

fn fallback(p: anytype, reason: []const u8) void {
    std.log.info("slide pocket: {s}; continuing the open fit without router refresh", .{reason});
    p.more();
}
