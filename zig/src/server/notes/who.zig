//! Who first: the notes of the subjects a question names and of the entities those notes name; nobody known, none.
const std = @import("std");
const index = @import("index.zig");
const lk = @import("lookup.zig");
const Index = index.Index;
const Scratch = lk.Scratch;
const Allocator = std.mem.Allocator;

/// One entity's weight from a name that several entities hold (always below 1).
const Weight = struct { e: u32, w: f64 };

/// The subjects a question names, the weights of those a shared name found, and the user's circle once made.
const Found = struct {
    a: Allocator,
    subjects: std.ArrayList(u32) = .empty,
    weights: std.ArrayList(Weight) = .empty,
    circle: ?[]const u32 = null,

    fn has(f: *const Found, e: u32) bool {
        return std.mem.indexOfScalar(u32, f.subjects.items, e) != null;
    }

    fn add(f: *Found, e: u32) Allocator.Error!void {
        if (!f.has(e)) try f.subjects.append(f.a, e);
    }

    fn weightOf(f: *const Found, e: u32) ?f64 {
        for (f.weights.items) |x| if (x.e == e) return x.w;
        return null;
    }

    /// weights[e] = min(weights.get(e, 1.0), w)
    fn lower(f: *Found, e: u32, w: f64) Allocator.Error!void {
        for (f.weights.items) |*x| if (x.e == e) {
            x.w = @min(x.w, w);
            return;
        };
        try f.weights.append(f.a, .{ .e = e, .w = @min(1.0, w) });
    }

    /// The user's circle: "me", "my ..." and "the <word>", and the entities their current notes name.
    fn circleOf(f: *Found, idx: *const Index) Allocator.Error![]const u32 {
        if (f.circle) |c| return c;
        var out: std.ArrayList(u32) = .empty;
        for (0..idx.entityCount()) |e| if (circleName(idx.entityName(@intCast(e)))) try out.append(f.a, @intCast(e));
        const named = out.items.len;
        for (0..named) |k| {
            var cur = idx.entCur(out.items[k]);
            while (cur.next()) |p| {
                const v = idx.valueEntityAt(p);
                if (v >= 0 and std.mem.indexOfScalar(u32, out.items, @intCast(v)) == null) try out.append(f.a, @intCast(v));
            }
        }
        f.circle = out.items;
        return out.items;
    }
};

/// lookup.lookup() who first: the subjects' notes and one hop, ranked by its scores times the name weights.
pub fn lookup(idx: *const Index, sc: *Scratch, text: anytype, a: Allocator, question: []const u8, qids: []const u32, past: bool, cap: u32, exclude: []const u32, long: index.Long) Allocator.Error!lk.Packed {
    _ = try lk.scored(idx, sc, qids, past, cap, exclude, long);
    var f: Found = .{ .a = a };
    try phrases(idx, sc, &f);
    if (idx.self_entity) |me| if (speaks(question) or mine(idx, question)) try f.add(me);
    try names(idx, text, a, question, &f);
    const long_n: u32 = @intCast(sc.long.items.len);
    if (f.subjects.items.len == 0) return .{ .chosen = &.{}, .tokens = 0, .candidates = 0, .long = long_n };
    for (sc.pool.items) |i| sc.slot[i].pstamp = 0; // the pool is the subjects' notes from here
    sc.pool.clearRetainingCapacity();
    for (f.subjects.items) |e| take(idx, sc, e, past);
    const own = sc.pool.items.len;
    for (0..own) |k| { // one hop: the entities the subjects' own notes name
        const i = sc.pool.items[k];
        const v = idx.valueEntityAt(i);
        if (v < 0 or f.has(@intCast(v))) continue;
        if (f.weightOf(idx.entityAt(i))) |w| try f.lower(@intCast(v), w);
        take(idx, sc, @intCast(v), past);
    }
    sc.keys.clearRetainingCapacity();
    for (sc.pool.items) |i| sc.keys.appendAssumeCapacity(lk.orderKey(sc.get(i) * (f.weightOf(idx.entityAt(i)) orelse 1.0), idx.rankAt(i)));
    return lk.pack(idx, sc, cap, past, @intCast(sc.pool.items.len), long_n);
}

/// An entity's current notes (and for a question about the past every note of its pairs), excluded ones left out.
fn take(idx: *const Index, sc: *Scratch, e: u32, past: bool) void {
    var cur = idx.entCur(e);
    while (cur.next()) |i| if (sc.xstamp[i] != sc.gen) sc.toPool(i);
    if (!past) return;
    var pairs = idx.entKeys(e);
    while (pairs.next()) |pr| {
        var all = idx.pairNotes(pr);
        while (all.next()) |i| if (sc.xstamp[i] != sc.gen) sc.toPool(i);
    }
}

/// The entity phrases the question holds, longest first, none inside another found one (NoteStore.subjects).
fn phrases(idx: *const Index, sc: *Scratch, f: *Found) Allocator.Error!void {
    const q = sc.qkeys.items;
    var spans: std.ArrayList([2]usize) = .empty;
    var n: usize = @min(q.len, index.max_phrase, idx.phraseMax());
    while (n >= 2) : (n -= 1) {
        var j: usize = 0;
        while (j + n <= q.len) : (j += 1) {
            const e = idx.phraseOf(index.packPhrase(q[j..][0..n])) orelse continue;
            if (inside(spans.items, j, j + n)) continue;
            try f.add(e);
            try spans.append(f.a, .{ j, j + n });
        }
    }
}

fn inside(spans: []const [2]usize, a: usize, b: usize) bool {
    for (spans) |sp| if (sp[0] <= a and b <= sp[1]) return true;
    return false;
}

/// Each run of ASCII letters names its entities: one holder, or the user's circle's, or all of them weighted down.
fn names(idx: *const Index, text: anytype, a: Allocator, q: []const u8, f: *Found) Allocator.Error!void {
    var i: usize = 0;
    while (i < q.len) {
        if (!std.ascii.isAlphabetic(q[i])) {
            i += 1;
            continue;
        }
        var j = i;
        while (j < q.len and std.ascii.isAlphabetic(q[j])) j += 1;
        try name(idx, text, a, q[i..j], f);
        i = j;
    }
}

/// The entities a current name fact calls `word` (alias.zig's table), added as subjects.
fn name(idx: *const Index, text: anytype, a: Allocator, word: []const u8, f: *Found) Allocator.Error!void {
    const spaced = try std.mem.concat(a, u8, &.{ " ", word });
    const ids = text.encode(a, spaced, false) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return, // a word the tokenizer cannot encode names nobody
    };
    if (ids.len == 0 or ids.len > index.max_phrase) return;
    var keys: [index.max_phrase]u32 = undefined;
    for (ids, keys[0..ids.len]) |t, *k| k.* = idx.keyOf(t);
    const cs = idx.aliasCandidates(index.packPhrase(keys[0..ids.len])) orelse return;
    if (cs.len == 0) return;
    if (cs.len == 1) return f.add(cs[0]);
    const circle = try f.circleOf(idx);
    var mine_found = false;
    for (cs) |e| if (std.mem.indexOfScalar(u32, circle, e) != null) {
        mine_found = true;
        try f.add(e);
    };
    if (mine_found) return;
    const entities: f64 = @floatFromInt(@max(idx.entityCount(), cs.len));
    const count: f64 = @floatFromInt(cs.len);
    const w = idx.log(1.0 + (entities - count + 0.5) / (count + 0.5)) / idx.log(1.0 + (entities - 0.5) / 1.5);
    for (cs) |e| {
        try f.add(e);
        try f.lower(e, w);
    }
}

/// "me", "my ..." or "the" and one lower-case word: the user's own wording of an entity.
fn circleName(n: []const u8) bool {
    if (std.mem.eql(u8, n, "me")) return true;
    if (n.len > 3 and std.mem.startsWith(u8, n, "my ")) return std.mem.indexOfScalar(u8, n[3..], '\n') == null;
    if (n.len <= 4 or !std.mem.startsWith(u8, n, "the ")) return false;
    for (n[4..]) |c| if (c < 'a' or c > 'z') return false;
    return true;
}

fn isWordChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

/// I, me, mine or myself as a word as written; a "me" right after tell, give or show is not the user speaking.
pub fn speaks(q: []const u8) bool {
    var prev: []const u8 = "";
    var gap = false; // only whitespace between the previous word and this one
    var i: usize = 0;
    while (i < q.len) {
        if (!isWordChar(q[i])) {
            i += 1;
            continue;
        }
        const start = i;
        while (i < q.len and isWordChar(q[i])) i += 1;
        const w = q[start..i];
        if (std.mem.eql(u8, w, "I") or std.mem.eql(u8, w, "mine") or std.mem.eql(u8, w, "myself")) return true;
        const asked = gap and (std.ascii.eqlIgnoreCase(prev, "tell") or std.ascii.eqlIgnoreCase(prev, "give") or std.ascii.eqlIgnoreCase(prev, "show"));
        if (std.mem.eql(u8, w, "me") and !asked) return true;
        prev = if (std.ascii.eqlIgnoreCase(w, "me") and asked) "" else w;
        var j = i;
        while (j < q.len and std.ascii.isWhitespace(q[j])) j += 1;
        gap = j > i and j < q.len and isWordChar(q[j]);
        i = j;
    }
    return false;
}

/// "my X" where X (to its first apostrophe) is a word the store tells the user's own facts with.
pub fn mine(idx: *const Index, q: []const u8) bool {
    var i: usize = 0;
    while (i + 2 <= q.len) {
        if (!std.ascii.eqlIgnoreCase(q[i..][0..2], "my") or (i > 0 and isWordChar(q[i - 1]))) {
            i += 1;
            continue;
        }
        var j = i + 2;
        while (j < q.len and std.ascii.isWhitespace(q[j])) j += 1;
        const g = j;
        while (j < q.len and (std.ascii.isAlphabetic(q[j]) or q[j] == '\'')) j += 1;
        if (g == i + 2 or j == g) { // no whitespace after my, or no letter after it
            i += 1;
            continue;
        }
        const x = q[g..j][0 .. std.mem.indexOfScalar(u8, q[g..j], '\'') orelse j - g];
        if (x.len > 0 and selfWord(idx, x)) return true;
        i = j;
    }
    return false;
}

/// No word of an entity's name; then a part of a stored attribute's name, or a word of one of the user's current notes.
fn selfWord(idx: *const Index, x: []const u8) bool {
    for (0..idx.entityCount()) |e| if (hasWord(idx.entityName(@intCast(e)), x)) return false;
    for (0..idx.attributeCount()) |k| {
        var parts = std.mem.tokenizeAny(u8, idx.attributeName(@intCast(k)), "_- \t\r\n");
        while (parts.next()) |p| if (std.ascii.eqlIgnoreCase(p, x)) return true;
    }
    const me = idx.self_entity orelse return false;
    var cur = idx.entCur(me);
    while (cur.next()) |p| if (hasWord(idx.lineAt(p), x)) return true;
    return false;
}

/// Whether a run of ASCII letters in `s` is `x`, ignoring case.
fn hasWord(s: []const u8, x: []const u8) bool {
    var i: usize = 0;
    while (i < s.len) {
        if (!std.ascii.isAlphabetic(s[i])) {
            i += 1;
            continue;
        }
        const start = i;
        while (i < s.len and std.ascii.isAlphabetic(s[i])) i += 1;
        if (std.ascii.eqlIgnoreCase(s[start..i], x)) return true;
    }
    return false;
}

test "the speaker: I, me, mine and myself as written; tell me is not the user speaking" {
    try std.testing.expect(speaks("Do I like plums?"));
    try std.testing.expect(speaks("Is the van mine?"));
    try std.testing.expect(speaks("What colour do my eyes look to me?"));
    try std.testing.expect(!speaks("Tell me the colour of my eyes."));
    try std.testing.expect(!speaks("Show  me my van."));
    try std.testing.expect(speaks("Tell me, is this mine?"));
    try std.testing.expect(speaks("Tell, me?"));
    try std.testing.expect(!speaks("Is it Ivo's or Ida's?"));
    try std.testing.expect(!speaks("Ime and meat, i and ME."));
    try std.testing.expect(speaks("tell me me"));
}

test "the user's circle: me, my and the with one lower-case word" {
    try std.testing.expect(circleName("me") and circleName("my uncle") and circleName("the van"));
    try std.testing.expect(!circleName("my") and !circleName("the Van") and !circleName("the old van") and !circleName("Ivo's van"));
    try std.testing.expect(hasWord("[3] \"I have grey eyes.\"", "EYES") and !hasWord("eyesight", "eyes"));
}
