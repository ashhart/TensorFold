//! A pocket lesson's read phase: its examples side by side through the trainer, each row's slot inputs labelled.
const std = @import("std");
const examples = @import("lesson_examples.zig");
const pocket = @import("pocket.zig");
const pack = @import("pocket_pack.zig");
const Kind = @import("pocket_fit.zig").Kind;

const Example = examples.Example;
const Label = pocket.Label;
const subjectRow = pocket.subjectRow;
const valueRow = pocket.valueRow;

const stride = 3; // every third row of another prompt is one the fact's neurons stay silent on

/// What the last read phase read: each example's fingerprint, the pocket's weights' then, whether it read to its end.
pub const Seen = struct {
    keys: std.ArrayList(u64) = .empty,
    state: u64 = 0,
    whole: bool = false,

    pub fn deinit(s: *Seen, gpa: std.mem.Allocator) void {
        s.keys.deinit(gpa);
    }
};

/// An example and what it is to the fact, as one number.
fn key(ex: Example, kind: Kind) u64 {
    var h = std.hash.Wyhash.init(@backingInt(kind));
    h.update(std.mem.sliceAsBytes(ex.ids));
    h.update(std.mem.asBytes(&[4]u32{ ex.start, ex.decide, ex.subject, ex.value }));
    return h.final();
}

/// The pocket's weights as one number.
fn state(p: anytype) u64 {
    var h = std.hash.Wyhash.init(0);
    h.update(std.mem.sliceAsBytes(p.up.items));
    h.update(std.mem.sliceAsBytes(p.down.items));
    h.update(std.mem.sliceAsBytes(p.used.items));
    return h.final();
}

/// Examples whose rows a lesson begun again keeps: all the last read's if the pocket and their list are as it found.
pub fn again(p: anytype) !usize {
    const now = state(p);
    const s = &p.read;
    if (s.whole and s.state == now) {
        const same = for (s.keys.items, 0..) |k, i| {
            const got = p.pickRead(i) orelse break false;
            if (key(got.ex, got.kind) != k) break false;
        } else true;
        if (same) {
            std.log.info("slide pocket: the lesson begins again over the same pocket: {d} examples' rows kept as read", .{s.keys.items.len});
            s.whole = false;
            return s.keys.items.len;
        }
    }
    s.keys.clearRetainingCapacity();
    s.state = now;
    s.whole = false;
    return 0;
}

/// As many of the next examples as one pass holds, read side by side; the fit next once all are read.
pub fn once(p: anytype, t: anytype) !bool {
    const per_pass = @typeInfo(@TypeOf(t.spans)).array.len;
    var seqs: [per_pass]pack.Seq = undefined;
    var kinds: [per_pass]Kind = undefined;
    var n: usize = 0;
    while (n < per_pass) : (n += 1) {
        const got = p.pickRead(p.at + n) orelse break;
        seqs[n] = pack.seq(got.ex);
        kinds[n] = got.kind;
    }
    if (n == 0) {
        p.read.whole = true;
        p.next(.fit);
        return false;
    }
    const m = pack.fits(t, seqs[0..n]);
    var out: [per_pass]pack.Result = undefined;
    try t.pass(seqs[0..m], .keys, pack.turn, out[0..m]);
    for (0..m) |i| {
        const ex = p.pickRead(p.at + i).?.ex;
        try label(p, t, ex, kinds[i], @intCast(p.at + i), t.spans[i].at);
        try p.read.keys.append(p.gpa, key(ex, kinds[i]));
    }
    p.at += m;
    return false;
}

/// Example `ex` (read `which`, of kind `kind`) labelled row by row, as the last pass read it from row `base`.
fn label(p: anytype, t: anytype, ex: Example, kind: Kind, which: u32, base: usize) !void {
    const own = kind != .steady;
    // the fact's own subjects and the tokens before them: another prompt naming the same is where its neurons may fire
    const subject = if (own and p.split[1] > 0) subjectRow(ex) else null;
    if (subject) |r| {
        if (std.mem.indexOfScalar(u32, p.subjects.items, ex.ids[r]) == null) try p.subjects.append(p.gpa, ex.ids[r]);
        if (r > 0 and std.mem.indexOfScalar(u32, p.leads.items, ex.ids[r - 1]) == null) try p.leads.append(p.gpa, ex.ids[r - 1]);
    }
    // its value row: the fact's own where the value comes, another answer's (an earlier fact's) one to stay silent on
    const value = if (p.split[2] > 0) valueRow(ex) else null;
    try keep(p, t, base + ex.decide, .{ .kind = kind, .key = .reply, .ex = which, .row = @intCast(ex.decide) });
    if (subject) |r| try keep(p, t, base + r, .{ .kind = kind, .key = .subject, .ex = which, .row = @intCast(r) });
    if (value) |r| try keep(p, t, base + r, if (own) .{ .kind = kind, .key = .value, .ex = which, .row = @intCast(r) } else .{ .kind = .other, .key = null, .ex = which, .row = @intCast(r) });
    // rows to stay silent on: every row of the fact's own examples, every third of others', each naming the subject
    var r: usize = 0;
    while (r < ex.ids.len - 1) : (r += 1) {
        if (r == ex.decide or (subject != null and r == subject.?) or (value != null and r == value.?)) continue;
        const named = p.split[1] > 0 and std.mem.indexOfScalar(u32, p.subjects.items, ex.ids[r]) != null;
        if (named and (own or !p.strict or (!p.silencing and r > 0 and std.mem.indexOfScalar(u32, p.leads.items, ex.ids[r - 1]) != null))) {
            // the fact's own subject (in its examples, or after the token its questions put first): left free
            try keep(p, t, base + r, .{ .kind = .none, .key = .subject, .ex = which, .row = @intCast(r) });
        } else if (own or named or r % stride == 0) try keep(p, t, base + r, .{ .kind = .other, .key = null, .ex = which, .row = @intCast(r) });
    }
}

/// The pass's row `r` at every slot's layer, labelled `l`.
fn keep(p: anytype, t: anytype, r: usize, l: Label) !void {
    for (p.moe.items) |layer| try p.rows.appendSlice(p.gpa, t.normed(layer, r));
    try p.labels.append(p.gpa, l);
}
