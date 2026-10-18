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
    h.update(std.mem.asBytes(&[5]u32{ ex.start, ex.decide, ex.subject, ex.value, ex.context }));
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

/// As many of the next examples as one pass holds, side by side (one in a conversation alone, by taps); then the fit.
pub fn once(p: anytype, t: anytype) !bool {
    const per_pass = @typeInfo(@TypeOf(t.spans)).array.len;
    const tapping = comptime @hasDecl(@TypeOf(t.*), "tapRows");
    const first = p.pickRead(p.at) orelse {
        p.read.whole = true;
        p.next(.fit);
        return false;
    };
    if (tapping and first.ex.context > 0) {
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
    try t.pass(seqs[0..m], .keys, pack.turn, out[0..m]);
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

/// A prompt after a conversation, by the trainer's taps: its turn's rows, every `sample`-th of a new conversation's.
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
    // a row two labels share (a reply row that is also its value's) tapped once, read by both
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(p.gpa);
    for (rows.items) |r| if (want.items.len == 0 or want.items[want.items.len - 1] != r.row) try want.append(p.gpa, r.row);
    const per = p.moe.items.len * p.hidden;
    const got = try p.gpa.alloc(u16, want.items.len * per);
    defer p.gpa.free(got);
    // the rows a pass reads: all but the last token, which only a pass's target reads
    try t.tapRows(ex.ids[0 .. ex.ids.len - 1], ex.context, want.items, p.moe.items, got);
    if (comptime @hasDecl(@TypeOf(t.*), "pass")) if (std.c.getenv("TF_SLIDE_TAP_CHECK") != null) tapCheck(p, t, ex, want.items, got);
    var at: usize = 0;
    for (rows.items) |r| {
        while (want.items[at] != r.row) at += 1;
        try p.rows.appendSlice(p.gpa, got[at * per ..][0..per]);
        try p.labels.append(p.gpa, r.l);
    }
}

/// A diagnostic (TF_SLIDE_TAP_CHECK): tapped rows against a fresh run's (nothing kept) and a pass's, if one holds it.
fn tapCheck(p: anytype, t: anytype, ex: Example, rows: []const u32, got: []const u16) void {
    const d = p.hidden;
    const slots = p.moe.items.len;
    const fresh = p.gpa.alloc(u16, got.len) catch return;
    defer p.gpa.free(fresh);
    t.tapRows(ex.ids[0 .. ex.ids.len - 1], 0, rows, p.moe.items, fresh) catch |e| return std.log.info("slide pocket: tap check: no fresh run ({t})", .{e});
    const runs = differ(got, fresh);
    var out: [1]pack.Result = undefined;
    t.pass(&.{pack.seq(ex)}, .keys, pack.turn, &out) catch |e| return std.log.info("slide pocket: tap check: {d} rows of a {d}-token prompt after {d}: {d} values differ from a fresh run's (most {d:.4}); no pass holds it ({t})", .{ rows.len, ex.ids.len, ex.context, runs[0], runs[1], e });
    for (rows, 0..) |r, i| for (p.moe.items, 0..) |layer, j| @memcpy(fresh[(i * slots + j) * d ..][0..d], t.normed(layer, t.spans[0].at + r));
    const pass = differ(got, fresh);
    std.log.info("slide pocket: tap check: {d} rows of a {d}-token prompt after {d}: {d} values differ from a fresh run's (most {d:.4}), {d} from the pass's (most {d:.4})", .{ rows.len, ex.ids.len, ex.context, runs[0], runs[1], pass[0], pass[1] });
}

/// How many bf16 values of `a` and `b` differ, and by how much at most.
fn differ(a: []const u16, b: []const u16) struct { usize, f32 } {
    var n: usize = 0;
    var most: f32 = 0;
    for (a, b) |x, y| {
        const fx: f32 = @bitCast(@as(u32, x) << 16);
        const fy: f32 = @bitCast(@as(u32, y) << 16);
        most = @max(most, @abs(fx - fy));
        n += @intFromBool(x != y);
    }
    return .{ n, most };
}

/// Example `ex` (read `which`, of kind `kind`) planned row by row from `from` on, and every `every`-th row before it.
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
    // rows to stay silent on: all of its own examples', every third of others' (every `every`-th of a conversation)
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

test "a prompt whose reply row is also its value's row is tapped once, and both labels read that row" {
    const gpa = std.testing.allocator;
    const d = 64;
    // taps as the CUDA trainer takes them: rows strictly increasing, else error.BadTap; each row's values its index
    const Taps = struct {
        pub fn tapRows(_: *@This(), _: []const u32, _: usize, want: []const u32, layers: []const usize, out: []u16) !void {
            for (want, 0..) |r, i| {
                if (i > 0 and r <= want[i - 1]) return error.BadTap;
                @memset(out[i * layers.len * d ..][0 .. layers.len * d], @intCast(r));
            }
        }
    };
    var p: pocket.Pocket(Taps, @import("families.zig").nemotron_h) = .{ .gpa = gpa, .mode = .auto, .hidden = d, .layers = 4 };
    defer p.deinit();
    try p.moe.append(gpa, 1);
    p.split = .{ 6, 0, 4 }; // value neurons
    // an answer that begins with its value: the row before it is the reply's first row too
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var t: Taps = .{};
    try tapOne(&p, &t, .{ .ids = &ids, .start = 6, .decide = 5, .value = 5, .context = 3 }, .fact, 0);
    var at_five: usize = 0;
    for (p.labels.items, 0..) |l, i| if (l.row == 5) {
        at_five += 1;
        try std.testing.expectEqual(@as(u16, 5), p.rows.items[i * d]);
    };
    try std.testing.expectEqual(@as(usize, 2), at_five);
    try std.testing.expectEqual(p.labels.items.len * d, p.rows.items.len);
}
