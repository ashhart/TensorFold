//! The Metal target's prompt in pieces (begin / piece / finish) against its whole prompt: state, choice and tokens.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fs = @import("fixture_serve.zig");
const ff = @import("fixture_fwd.zig");

const testing = std.testing;
const gpa = testing.allocator;
const io = testing.io;
const seed = 73;
const n_prompt = 900;
const n_decode = 8;

fn say(ok: bool, comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s} dsv41 metal steps: ", .{if (ok) "PASS" else "FAIL"});
    std.debug.print(fmt ++ "\n", args);
}

/// Every byte a slot's prompt state holds: each layer's window ring, pool, index keys and carry, the taps, the tail.
fn stateBytes(slot: *const pkg.target.Slot) ![]u8 {
    return bytesOf(slot, true);
}

/// The slot's bytes, its prompt tail's too when `tail` (a resumed prompt's tail holds only the rows it ran).
fn bytesOf(slot: *const pkg.target.Slot, tail: bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for (slot.kv.regions) |r| for ([_]?mtl.Buffer{ r.window, r.pool, r.keys, r.carry }) |ob| if (ob) |b| {
        try out.appendSlice(gpa, b.contents()[0..b.length()]);
    };
    try out.appendSlice(gpa, slot.kv.taps.?.contents()[0..slot.kv.taps.?.length()]);
    if (tail) for ([_]mtl.Buffer{ slot.tail.x, slot.tail.pre }) |b| try out.appendSlice(gpa, b.contents()[0..b.length()]);
    return out.toOwnedSlice(gpa);
}

fn diff(x: []const u8, y: []const u8) usize {
    if (x.len != y.len) return @max(x.len, y.len);
    var n: usize = 0;
    for (x, y) |s, t| n += @intFromBool(s != t);
    return n;
}

/// One plain row on a live slot after `tok` at `at`: the next greedy choice.
fn step(tg: host.iface.Target, slot: u32, at: u64, tok: u32) !u32 {
    var ch: [1]u32 = undefined;
    var cs = [_][]u32{&ch};
    try tg.window(&.{.{ .slot = slot, .start = at, .tokens = &.{tok}, .parents = null, .draws = &.{at + 1}, .sampling = null }}, &cs);
    return ch[0];
}

/// The prompt's first choice and `n_decode` greedy tokens after it on a live slot.
fn decode(tg: host.iface.Target, slot: u32, len: usize, first: u32, out: []u32) !void {
    var tok = first;
    for (out, 0..) |*o, i| {
        o.* = tok;
        tok = try step(tg, slot, len + i, tok);
    }
}

/// The whole-prompt reference: state bytes right after the prompt, and its tokens.
const Ref = struct {
    state: []u8,
    kv: []u8,
    toks: [n_decode]u32,

    fn free(r: Ref) void {
        gpa.free(r.state);
        gpa.free(r.kv);
    }
};

fn reference(prompt: []const u32) !Ref {
    const m = try fs.open(gpa, io, seed, .{ .slots = 1, .drafts = false });
    defer fs.close(m);
    const tg = m.t.target();
    const first = try tg.prefill(0, prompt, null, prompt.len);
    var r: Ref = .{ .state = try stateBytes(&m.t.slots[0]), .kv = undefined, .toks = undefined };
    errdefer gpa.free(r.state);
    r.kv = try bytesOf(&m.t.slots[0], false);
    errdefer gpa.free(r.kv);
    try decode(tg, 0, prompt.len, first, &r.toks);
    tg.release(0);
    return r;
}

test "dsv41 metal steps: pieces of 1, 7, 64, 100 and 300 rows == the whole prompt; a decode stream between them" {
    const p = try ff.ids(gpa, &fs.config(), n_prompt, 74);
    defer gpa.free(p);
    const q = try ff.ids(gpa, &fs.config(), 200, 75);
    defer gpa.free(q);
    const want = try reference(p);
    defer want.free();
    const solo = try reference(q);
    defer solo.free();
    const m = try fs.open(gpa, io, seed, .{ .slots = 2, .drafts = false });
    defer fs.close(m);
    const tg = m.t.target();
    var bad: usize = 0;
    for ([_]u64{ 1, 7, 64, 100, 300 }) |rows| {
        // slot 0 decodes one row between every two pieces of slot 1's prompt
        var tok = try tg.prefill(0, q, null, q.len);
        var made: std.ArrayList(u32) = .empty;
        defer made.deinit(gpa);
        const b = try tg.begin(1, p);
        try testing.expectEqual(@as(u64, 0), b.at);
        var at: u64 = 0;
        const before = m.t.stats.pieces;
        while (p.len - at > rows) : (at += rows) {
            try tg.piece(1, p, at, at + rows, false);
            if (made.items.len < n_decode) {
                try made.append(gpa, tok);
                tok = try step(tg, 0, q.len + made.items.len - 1, tok);
            }
        }
        const first = try tg.finish(1, p, null, p.len);
        const st = try stateBytes(&m.t.slots[1]);
        defer gpa.free(st);
        var toks: [n_decode]u32 = undefined;
        try decode(tg, 1, p.len, first, &toks);
        while (made.items.len < n_decode) {
            try made.append(gpa, tok);
            tok = try step(tg, 0, q.len + made.items.len - 1, tok);
        }
        const sd = diff(st, want.state);
        const td = @intFromBool(!std.mem.eql(u32, &toks, &want.toks));
        const dd = @intFromBool(!std.mem.eql(u32, made.items, &solo.toks));
        say(sd + td + dd == 0, "(a) {d}-row pieces ({d} of them) vs whole: {d} state bytes, tokens {s}; the decode between them {s} alone", .{ rows, m.t.stats.pieces - before, sd, if (td == 0) "equal" else "differ", if (dd == 0) "equals its run" else "differs from its run" });
        bad += sd + td + dd;
        tg.release(0);
        tg.release(1);
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal steps: a refused or released mid-prompt prefill frees the slot; the next prompt there == whole" {
    const p = try ff.ids(gpa, &fs.config(), n_prompt, 76);
    defer gpa.free(p);
    const want = try reference(p);
    defer want.free();
    const m = try fs.open(gpa, io, seed, .{ .slots = 1, .drafts = false, .cache_bytes = 1 << 30 });
    defer fs.close(m);
    const tg = m.t.target();
    // a piece that skips rows fails and abandons the prompt; another begun prompt is released between pieces
    _ = try tg.begin(0, p);
    try tg.piece(0, p, 0, 300, false);
    const refused = tg.piece(0, p, 301, 400, false);
    try testing.expectError(error.NotAChain, refused);
    try testing.expectError(error.NotBegun, tg.piece(0, p, 300, 400, false));
    _ = try tg.begin(0, p);
    try tg.piece(0, p, 0, 500, false);
    tg.release(0);
    const kept = m.t.cache.entries.items.len;
    const abandoned = m.t.stats.abandoned;
    // the slot takes a whole prompt again: no kept state of the unfinished ones, the result the reference's
    const hits = m.t.stats.hits;
    const first = try tg.prefill(0, p, null, p.len);
    const st = try stateBytes(&m.t.slots[0]);
    defer gpa.free(st);
    var toks: [n_decode]u32 = undefined;
    try decode(tg, 0, p.len, first, &toks);
    tg.release(0);
    const sd = diff(st, want.state);
    const ok = sd == 0 and std.mem.eql(u32, &toks, &want.toks) and kept == 0 and abandoned == 2 and m.t.stats.hits == hits;
    say(ok, "(b) {d} abandoned, {d} entries kept of them, the next prompt: {d} state bytes differ, tokens {s}", .{ abandoned, kept, sd, if (std.mem.eql(u32, &toks, &want.toks)) "equal" else "differ" });
    try testing.expect(ok);
}

test "dsv41 metal steps: a kept prompt resumed in pieces (a later turn, the whole prompt again) == fresh" {
    const p = try ff.ids(gpa, &fs.config(), 500, 77);
    defer gpa.free(p);
    const more = try ff.ids(gpa, &fs.config(), 300, 78);
    defer gpa.free(more);
    const turn = try std.mem.concat(gpa, u32, &.{ p, more });
    defer gpa.free(turn);
    const want_turn = try reference(turn);
    defer want_turn.free();
    const want_p = try reference(p);
    defer want_p.free();
    const m = try fs.open(gpa, io, seed, .{ .slots = 1, .drafts = false, .cache_bytes = 1 << 30 });
    defer fs.close(m);
    const tg = m.t.target();
    var bad: usize = 0;
    // the first prompt in pieces, kept at its finish; then the turn (a partial hit) and the prompt (a whole hit)
    for ([_][]const u32{ p, turn, p }, [_]*const Ref{ &want_p, &want_turn, &want_p }, [_]u64{ 0, 500, 500 }) |ids, want, hit| {
        const b = try tg.begin(0, ids);
        var at = b.at;
        while (ids.len - at > 64) : (at += 64) try tg.piece(0, ids, at, at + 64, false);
        const first = try tg.finish(0, ids, null, ids.len);
        const st = try bytesOf(&m.t.slots[0], false);
        defer gpa.free(st);
        var toks: [n_decode]u32 = undefined;
        try decode(tg, 0, ids.len, first, &toks);
        tg.release(0);
        // a resumed tail holds only its own rows; a whole hit restores no launch taps: compare KV and taps, or none
        const sd = if (b.at == ids.len) 0 else diff(st, want.kv);
        const ok = b.at == hit and sd == 0 and std.mem.eql(u32, &toks, &want.toks);
        say(ok, "(c) {d} tokens, {d} resumed: {d} state bytes differ, tokens {s}", .{ ids.len, b.at, sd, if (std.mem.eql(u32, &toks, &want.toks)) "equal" else "differ" });
        bad += @intFromBool(!ok);
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal steps: two prompts' pieces taking turns at the same positions == each prompt whole" {
    const p0 = try ff.ids(gpa, &fs.config(), 600, 79);
    defer gpa.free(p0);
    const p1 = try ff.ids(gpa, &fs.config(), 600, 80);
    defer gpa.free(p1);
    const w0 = try reference(p0);
    defer w0.free();
    const w1 = try reference(p1);
    defer w1.free();
    const m = try fs.open(gpa, io, seed, .{ .slots = 2, .drafts = false });
    defer fs.close(m);
    const tg = m.t.target();
    var bad: usize = 0;
    // 64-row pieces (the chunk): each piece's next chunk is prepared ahead, then the other slot's piece runs
    for ([_]u64{ 64, 100 }) |rows| {
        const ps = [2][]const u32{ p0, p1 };
        for (ps, 0..) |p, i| _ = try tg.begin(@intCast(i), p);
        var at: u64 = 0;
        while (p0.len - at > rows) : (at += rows) for (ps, 0..) |p, i| try tg.piece(@intCast(i), p, at, at + rows, false);
        var firsts: [2]u32 = undefined;
        for (ps, &firsts, 0..) |p, *f, i| f.* = try tg.finish(@intCast(i), p, null, p.len);
        for (ps, firsts, [2]*const Ref{ &w0, &w1 }, 0..) |p, f, w, i| {
            const st = try stateBytes(&m.t.slots[i]);
            defer gpa.free(st);
            var toks: [n_decode]u32 = undefined;
            try decode(tg, @intCast(i), p.len, f, &toks);
            const sd = diff(st, w.state);
            const same = std.mem.eql(u32, &toks, &w.toks);
            say(sd == 0 and same, "(d) {d}-row pieces taking turns, slot {d}: {d} state bytes differ, tokens {s}", .{ rows, i, sd, if (same) "equal" else "differ" });
            bad += sd + @intFromBool(!same);
            tg.release(@intCast(i));
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}
