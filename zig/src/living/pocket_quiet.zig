//! Earlier facts' neurons that fire on the open fact's questions, fitted again to stay silent there.
const std = @import("std");
const pocket_fit = @import("pocket_fit.zig");

const Kind = pocket_fit.Kind;
const fit = pocket_fit.fit;
const dotBf16 = pocket_fit.dotBf16;
const fit_steps = pocket_fit.fit_steps;

pub const kept_least: f32 = 0.5; // an earlier neuron refitted must still reach this on every one of its own fact's rows
pub const refit_above: f32 = 0.5; // an earlier neuron is refitted only where it fires above this on a new fact's rows

/// Earlier neurons firing on this fact's rows refitted silent there, undo copies kept; same-subject ones stay.
pub fn quiet(p: anytype, t: anytype, host: anytype) !void {
    const P = @TypeOf(p.*);
    const d = p.hidden;
    const n = p.moe.items.len;
    for (p.learned.items[0..p.learned_before]) |e| {
        if (e.key == .subject and std.mem.indexOfScalar(u32, p.subjects.items, e.subject) != null) continue;
        const u = p.neuron(p.up.items, e.slot, e.neuron, d);
        var most: f32 = -std.math.inf(f32);
        for (p.labels.items, 0..) |l, i| if (l.kind == .fact or l.kind == .held) {
            most = @max(most, dotBf16(u, p.rows.items[(i * n + e.slot) * d ..][0..d]));
        };
        if (most <= refit_above) continue; // weak there: left exactly as it is, this fact's outputs fitted with it
        try p.refits.append(p.gpa, .{ .slot = e.slot, .neuron = e.neuron });
        try p.refit_up.appendSlice(p.gpa, u);
        if (P.swiglu) try p.refit_gate.appendSlice(p.gpa, p.neuron(p.gate.items, e.slot, e.neuron, d));
        var xs: std.ArrayList([]const u16) = .empty;
        defer xs.deinit(p.gpa);
        var kinds: std.ArrayList(Kind) = .empty;
        defer kinds.deinit(p.gpa);
        for (0..e.count) |r| {
            try xs.append(p.gpa, p.pos.items[(e.start + r) * d ..][0..d]);
            try kinds.append(p.gpa, .fact);
        }
        for (p.labels.items, 0..) |l, i| {
            try xs.append(p.gpa, p.rows.items[(i * n + e.slot) * d ..][0..d]);
            try kinds.append(p.gpa, switch (l.kind) {
                .fact, .held => .steady, // this fact's questions are another fact's to it
                .none => .other,
                else => l.kind,
            });
        }
        const f = try fit(p.gpa, xs.items, kinds.items, d, u, true, fit_steps / 2, e.count, P.of.neuron.margin);
        if (f.fact_least < kept_least or @max(f.steady_most, f.other_most) > 0) {
            // a refit that forgets its fact or spreads is undone; this fact's outputs are fitted with it firing
            @memcpy(u, p.refit_up.items[p.refit_up.items.len - d ..]);
            p.refits.shrinkRetainingCapacity(p.refits.items.len - 1);
            p.refit_up.shrinkRetainingCapacity(p.refit_up.items.len - d);
            if (P.swiglu) p.refit_gate.shrinkRetainingCapacity(p.refit_gate.items.len - d);
            std.log.info("slide pocket: layer {d}, neuron {d}, an earlier fact's, fires on this fact's questions (up to {d:.2}); fitted again its own would read at least {d:.2} and other rows at most {d:.2}, so it stays as it was", .{ p.moe.items[e.slot], e.neuron, most, f.fact_least, @max(f.steady_most, f.other_most) });
            continue;
        }
        p.gateFrom(e.slot, e.neuron);
        try p.upload(t, host, e.slot);
        std.log.info("slide pocket: layer {d}, neuron {d}, an earlier fact's, fired on this fact's questions (up to {d:.2}); fitted again: there and elsewhere at most {d:.2}, its own questions' least {d:.2}", .{ p.moe.items[e.slot], e.neuron, most, @max(f.steady_most, f.other_most), f.fact_least });
    }
}
