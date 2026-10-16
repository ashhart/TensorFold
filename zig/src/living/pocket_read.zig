//! A pocket lesson's read phase: its examples side by side through the trainer, each row's slot inputs labelled.
const std = @import("std");
const examples = @import("lesson_examples.zig");
const pocket = @import("pocket.zig");
const pack = @import("pocket_pack.zig");
const Kind = @import("pocket_fit.zig").Kind;
const clock = @import("pocket_clock.zig");

const Example = examples.Example;
const Label = pocket.Label;
const subjectRow = pocket.subjectRow;
const valueRow = pocket.valueRow;

const stride = 3; // every third row of another prompt is one the fact's neurons stay silent on
const sample = 8; // every eighth row of a conversation before a question, read with the first question asked after it

/// What the last read phase read: each example's fingerprint, the pocket's weights' then, whether it read to its end.
pub const Seen = struct {
    keys: std.ArrayList(u64) = .empty,
    state: u64 = 0,
    whole: bool = false,
    talks: std.ArrayList(u64) = .empty, // the conversations whose own rows were read (each with its first question)

    pub fn deinit(s: *Seen, gpa: std.mem.Allocator) void {
        s.keys.deinit(gpa);
        s.talks.deinit(gpa);
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
    if (@TypeOf(p.*).swiglu) h.update(std.mem.sliceAsBytes(p.gate.items));
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
    s.talks.clearRetainingCapacity();
    s.state = now;
    s.whole = false;
    return 0;
}

/// As many of the next examples as one pass holds, read side by side.
pub fn once(p: anytype, t: anytype) !bool {
    const per_pass = @typeInfo(@TypeOf(t.spans)).array.len;
    const tapping = comptime @hasDecl(@TypeOf(t.*), "tapRows");
    const first = p.pickRead(p.at) orelse {
        p.read.whole = true;
        p.next(.fit);
        return false;
    };
    if (tapping and first.ex.context > 0) {
        const t0 = clock.mono();
        defer clock.add(.read_tap, clock.mono() - t0);
        if (comptime @hasDecl(@TypeOf(t.*), "tapMany")) {
            if (!std.mem.eql(u8, std.mem.span(std.c.getenv("TF_SLIDE_TAPS") orelse "many"), "one")) {
                p.at += try tapMany(p, t, @intCast(p.at));
                return false;
            }
        }
        try tapOne(p, t, first.ex, first.kind, @intCast(p.at));
        try p.read.keys.append(p.gpa, key(first.ex, first.kind));
        p.at += 1;
        return false;
    }
    var seqs: [per_pass]pack.Seq = undefined;
    var kinds: [per_pass]Kind = undefined;
    var n: usize = 0;
    while (n < per_pass) : (n += 1) {
        const got = p.pickRead(p.at + n) orelse break;
        if (tapping and got.ex.context > 0) break;
        seqs[n] = pack.seq(got.ex);
        kinds[n] = got.kind;
    }
    const m = pack.fits(t, seqs[0..n]);
    var out: [per_pass]pack.Result = undefined;
    // the forward stops at the highest layer a neuron may sit at: no row above it is ever read
    if (comptime @hasField(@TypeOf(t.*), "keys_top")) t.keys_top = p.readTop();
    defer {
        if (comptime @hasField(@TypeOf(t.*), "keys_top")) t.keys_top = null;
    }
    const t0 = clock.mono();
    try t.pass(seqs[0..m], .keys, pack.turn, out[0..m]);
    const t1 = clock.mono();
    clock.add(.read_pass, t1 - t0);
    defer clock.add(.read_copy, clock.mono() - t1);
    var rows: std.ArrayList(Row) = .empty;
    defer rows.deinit(p.gpa);
    for (0..m) |i| {
        const ex = p.pickRead(p.at + i).?.ex;
        rows.clearRetainingCapacity();
        try plan(p, ex, kinds[i], @intCast(p.at + i), 0, 0, &rows);
        for (rows.items) |r| try keep(p, t, t.spans[i].at + r.row, r.l);
        try p.read.keys.append(p.gpa, key(ex, kinds[i]));
    }
    p.at += m;
    return false;
}

/// A read row: its place in its example and what it is to the fact.
const Row = struct { row: u32, l: Label };

/// A prompt asked after a conversation (`ex.context` its rows), read through the trainer's taps.
fn tapOne(p: anytype, t: anytype, ex: Example, kind: Kind, which: u32) !void {
    const talk = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ex.ids[0..ex.context]));
    const fresh = std.mem.indexOfScalar(u64, p.read.talks.items, talk) == null;
    if (fresh) try p.read.talks.append(p.gpa, talk);
    var rows: std.ArrayList(Row) = .empty;
    defer rows.deinit(p.gpa);
    try plan(p, ex, kind, which, ex.context, if (fresh) sample else 0, &rows);
    std.mem.sort(Row, rows.items, {}, struct {
        fn less(_: void, a: Row, b: Row) bool {
            return a.row < b.row;
        }
    }.less);
    const want = try p.gpa.alloc(u32, rows.items.len);
    defer p.gpa.free(want);
    for (rows.items, want) |r, *w| w.* = r.row;
    const d = p.hidden;
    const slots = p.moe.items.len;
    const got = try p.gpa.alloc(u16, rows.items.len * slots * d);
    defer p.gpa.free(got);
    // the rows a pass reads: all but the last token, which only a pass's target reads
    try t.tapRows(ex.ids[0 .. ex.ids.len - 1], ex.context, want, p.moe.items, got);
    if (std.c.getenv("TF_SLIDE_TAP_CHECK") != null) tapCheck(p, t, ex, want, got);
    try p.rows.appendSlice(p.gpa, got);
    for (rows.items) |r| try p.labels.append(p.gpa, r.l);
}

/// The next examples asked after one conversation (as many as the trainer's tapMany takes), read as tapOne reads each.
fn tapMany(p: anytype, t: anytype, first: u32) !usize {
    const Item = @typeInfo(@TypeOf(@TypeOf(t.*).tapMany)).@"fn".param_types[1].?;
    const It = @typeInfo(Item).pointer.child;
    const most = 8;
    var exs: [most]Example = undefined;
    var kinds: [most]Kind = undefined;
    var n: usize = 0;
    const talk0 = blk: {
        const ex = p.pickRead(first).?.ex;
        break :blk std.mem.sliceAsBytes(ex.ids[0..ex.context]);
    };
    while (n < most) : (n += 1) {
        const got = p.pickRead(first + n) orelse break;
        if (got.ex.context == 0 or !std.mem.eql(u8, std.mem.sliceAsBytes(got.ex.ids[0..got.ex.context]), talk0)) break;
        exs[n] = got.ex;
        kinds[n] = got.kind;
    }
    const d = p.hidden;
    const slots = p.moe.items.len;
    var plans: [most]std.ArrayList(Row) = @splat(.empty);
    defer for (&plans) |*x| x.deinit(p.gpa);
    var wants: [most][]u32 = @splat(&.{});
    defer for (wants) |w| p.gpa.free(w);
    var outs: [most][]u16 = @splat(&.{});
    defer for (outs) |o| p.gpa.free(o);
    var items: [most]It = undefined;
    for (0..n) |i| {
        const ex = exs[i];
        const talk = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ex.ids[0..ex.context]));
        const fresh = std.mem.indexOfScalar(u64, p.read.talks.items, talk) == null;
        if (fresh) try p.read.talks.append(p.gpa, talk);
        try plan(p, ex, kinds[i], first + @as(u32, @intCast(i)), ex.context, if (fresh) sample else 0, &plans[i]);
        std.mem.sort(Row, plans[i].items, {}, struct {
            fn less(_: void, a: Row, b: Row) bool {
                return a.row < b.row;
            }
        }.less);
        wants[i] = try p.gpa.alloc(u32, plans[i].items.len);
        for (plans[i].items, wants[i]) |r, *w| w.* = r.row;
        outs[i] = try p.gpa.alloc(u16, plans[i].items.len * slots * d);
        items[i] = .{ .ids = ex.ids[0 .. ex.ids.len - 1], .context = ex.context, .rows = wants[i], .out = outs[i] };
    }
    try t.tapMany(items[0..n], p.moe.items);
    for (0..n) |i| {
        try p.rows.appendSlice(p.gpa, outs[i]);
        for (plans[i].items) |r| try p.labels.append(p.gpa, r.l);
        try p.read.keys.append(p.gpa, key(exs[i], kinds[i]));
    }
    return n;
}

/// A diagnostic (TF_SLIDE_TAP_CHECK): the tapped rows of a prompt short enough for a pass, against the pass's own.
fn tapCheck(p: anytype, t: anytype, ex: Example, rows: []const u32, got: []const u16) void {
    const d = p.hidden;
    const slots = p.moe.items.len;
    var out: [1]pack.Result = undefined;
    t.pass(&.{pack.seq(ex)}, .keys, pack.turn, &out) catch |e| return std.log.info("slide pocket: tap check: no pass of {d} rows ({t})", .{ ex.ids.len, e });
    var most: f32 = 0;
    var differ: usize = 0;
    for (rows, 0..) |r, i| for (p.moe.items, 0..) |layer, j| {
        const a = t.normed(layer, t.spans[0].at + r);
        const b = got[(i * slots + j) * d ..][0..d];
        for (a, b) |x, y| {
            const fx: f32 = @bitCast(@as(u32, x) << 16);
            const fy: f32 = @bitCast(@as(u32, y) << 16);
            most = @max(most, @abs(fx - fy));
            differ += @intFromBool(x != y);
        }
    };
    std.log.info("slide pocket: tap check: {d} rows of a {d}-token prompt: {d} values differ from the pass's, most by {d:.4}", .{ rows.len, ex.ids.len, differ, most });
}

/// Example `ex` (read `which`, of kind `kind`) planned row by row from row `from` on.
fn plan(p: anytype, ex: Example, kind: Kind, which: u32, from: usize, every: usize, out: *std.ArrayList(Row)) !void {
    const a = p.gpa;
    const own = kind != .steady;
    // the fact's own subjects and the tokens before them: another prompt naming the same is where its neurons may fire
    const subject = if (own and p.split[1] > 0) subjectRow(ex) else null;
    if (subject) |r| {
        if (std.mem.indexOfScalar(u32, p.subjects.items, ex.ids[r]) == null) try p.subjects.append(p.gpa, ex.ids[r]);
        if (r > 0 and std.mem.indexOfScalar(u32, p.leads.items, ex.ids[r - 1]) == null) try p.leads.append(p.gpa, ex.ids[r - 1]);
    }
    // its value row: the fact's own where the value comes, another answer's (an earlier fact's) one to stay silent on
    const value = if (p.split[2] > 0) valueRow(ex) else null;
    try out.append(a, .{ .row = ex.decide, .l = .{ .kind = kind, .key = .reply, .ex = which, .row = ex.decide } });
    if (subject) |r| try out.append(a, .{ .row = @intCast(r), .l = .{ .kind = kind, .key = .subject, .ex = which, .row = @intCast(r) } });
    if (value) |r| try out.append(a, .{ .row = @intCast(r), .l = if (own) .{ .kind = kind, .key = .value, .ex = which, .row = @intCast(r) } else .{ .kind = .other, .key = null, .ex = which, .row = @intCast(r) } });
    // rows to stay silent on: every row of the fact's own examples, every third of others', each naming the subject.
    var r: usize = if (every > 0) 0 else from;
    while (r < ex.ids.len - 1) : (r += 1) {
        if (r == ex.decide or (subject != null and r == subject.?) or (value != null and r == value.?)) continue;
        const named = p.split[1] > 0 and std.mem.indexOfScalar(u32, p.subjects.items, ex.ids[r]) != null;
        const before = r < from; // a conversation's row, read only every `every`-th
        if (named and (own or !p.strict or (!p.silencing and r > 0 and std.mem.indexOfScalar(u32, p.leads.items, ex.ids[r - 1]) != null))) {
            // the fact's own subject (in its examples, or after the token its questions put first): left free
            try out.append(a, .{ .row = @intCast(r), .l = .{ .kind = .none, .key = .subject, .ex = which, .row = @intCast(r) } });
        } else if (if (before) named or r % every == 0 else own or named or r % stride == 0) try out.append(a, .{ .row = @intCast(r), .l = .{ .kind = .other, .key = null, .ex = which, .row = @intCast(r) } });
    }
}

/// The pass's row `r` at every slot's layer, labelled `l`.
fn keep(p: anytype, t: anytype, r: usize, l: Label) !void {
    for (p.moe.items) |layer| try p.rows.appendSlice(p.gpa, t.normed(layer, r));
    try p.labels.append(p.gpa, l);
}
