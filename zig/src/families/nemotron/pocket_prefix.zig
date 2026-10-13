//! A pocket fact's answers' prompts run once: their rows before any of its neurons fires never change while it learns.
const std = @import("std");
const pack = @import("pocket_pack.zig");
const train_mode = @import("train_mode.zig");
const dotBf16 = @import("pocket_fit.zig").dotBf16;

const Seq = train_mode.Seq;
const Span = train_mode.Span;

const least = 16; // rows a prefix must hold to be worth a slot

/// An answer's prefix: its rows before any open neuron fires (once a whole pass shows them), and its slot once drawn.
const Kept = struct { seen: bool = false, skip: usize = 0, slot: ?u32 = null };

/// The open fact's answers' prefixes (its own answers first, then those that are to stay).
pub const Prefixes = struct {
    list: std.ArrayList(Kept) = .empty,
    slots: u32 = 0,
    skipped: usize = 0, // rows its steps did not run again
    inputs: u64 = 0, // the pocket's input weights the prefixes were found under

    pub fn deinit(s: *Prefixes, gpa: std.mem.Allocator) void {
        s.list.deinit(gpa);
    }

    /// A fact's `n` answers begun, no prefix known.
    pub fn begin(s: *Prefixes, gpa: std.mem.Allocator, n: usize) !void {
        s.list.clearRetainingCapacity();
        try s.list.appendNTimes(gpa, .{}, n);
        s.slots = 0;
        s.skipped = 0;
        s.inputs = 0;
    }
};

/// A round of outputs begun: the prefixes found kept while the pocket's input weights are as they were found under.
pub fn round(p: anytype, s: *Prefixes) void {
    var h = std.hash.Wyhash.init(0);
    h.update(std.mem.sliceAsBytes(p.up.items));
    h.update(std.mem.sliceAsBytes(p.used.items));
    const now = h.final();
    if (now == s.inputs) return;
    for (s.list.items) |*k| k.* = .{};
    s.slots = 0;
    s.inputs = now;
}

/// The slots freed with their states (a further round runs its answers' prefixes again from their next draw).
pub fn release(t: anytype, s: *Prefixes) !void {
    try t.prefixSlots(0);
    for (s.list.items) |*k| k.slot = null;
    s.slots = 0;
}

/// The drawn answers (`which`: their indexes) set to skip prefixes in slots; those shown but not yet kept primed first.
pub fn use(t: anytype, s: *Prefixes, seqs: []Seq, which: []const usize) !void {
    if (!t.packs()) return;
    var fresh: [32]Seq = undefined;
    if (seqs.len > fresh.len) return error.BadPass;
    var n: usize = 0;
    for (seqs, which) |*q, w| {
        const k = &s.list.items[w];
        if (k.slot == null and k.seen and k.skip >= least and s.slots < @TypeOf(t.prefixes).most) {
            k.slot = s.slots;
            s.slots += 1;
            fresh[n] = .{ .ids = q.ids[0 .. k.skip + 1], .start = k.skip, .decide = k.skip - 1, .slot = k.slot.? };
            n += 1;
        }
        if (k.slot) |slot| {
            q.skip = k.skip;
            q.slot = slot;
            s.skipped += k.skip;
        }
    }
    if (n == 0) return;
    try t.prefixSlots(s.slots);
    var got: [32]train_mode.Result = undefined;
    var at: usize = 0;
    while (at < n) {
        const m = pack.fits(t, fresh[at..n]);
        try t.pass(fresh[at..][0..m], .prime, pack.turn, got[0..m]);
        at += m;
    }
}

/// Answer `which` (`sq` at `sp`, run whole): its rows before an open neuron fires, its decision, answer and nudges.
pub fn learn(p: anytype, t: anytype, s: *Prefixes, sq: Seq, sp: Span, which: usize) void {
    const k = &s.list.items[which];
    if (k.seen or sq.skip > 0) return;
    k.seen = true;
    const d = t.config().hidden;
    var limit = @min(sq.decide, sq.start - 1);
    for (sq.nudge.layers, sq.nudge.rows) |mask, row| if (mask != 0) {
        limit = @min(limit, row);
    };
    k.skip = rows: for (0..limit) |r| {
        for (p.taken.items) |x| if (dotBf16(p.neuron(p.up.items, x.slot, x.neuron, d), t.normed(p.moe.items[x.slot], sp.row(r))) > 0) break :rows r;
        for (p.joint.items) |x| if (dotBf16(p.neuron(p.up.items, x.slot, x.neuron, d), t.normed(p.moe.items[x.slot], sp.row(r))) > 0) break :rows r;
    } else limit;
}
