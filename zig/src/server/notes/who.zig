//! Who first: the notes of the subjects a question names and of the entities those notes name; nobody known, none.
const std = @import("std");
const index = @import("index.zig");
const lk = @import("lookup.zig");
const Index = index.Index;
const Scratch = lk.Scratch;
const Allocator = std.mem.Allocator;

/// The most runs of name letters one name is made of, one space apart.
const chain = 4;

/// The attribute prefix of a note the server wrote from free text without sorting it (write.zig: "note:SOURCE").
const unsorted = "note:";

const Set = std.AutoHashMapUnmanaged(u32, void);

/// The subjects a question names, the weights of those a shared name found, and the user's circle once made.
const Found = struct {
    gpa: Allocator,
    subjects: std.ArrayList(u32) = .empty,
    member: Set = .empty,
    weights: std.AutoHashMapUnmanaged(u32, f64) = .empty,
    circle: ?Set = null,

    fn deinit(f: *Found) void {
        f.subjects.deinit(f.gpa);
        f.member.deinit(f.gpa);
        f.weights.deinit(f.gpa);
        if (f.circle) |*c| c.deinit(f.gpa);
    }

    fn add(f: *Found, e: u32) Allocator.Error!void {
        try f.subjects.ensureUnusedCapacity(f.gpa, 1);
        if (!(try f.member.getOrPut(f.gpa, e)).found_existing) f.subjects.appendAssumeCapacity(e);
    }

    /// weights[e] = min(weights.get(e, 1.0), w)
    fn lower(f: *Found, e: u32, w: f64) Allocator.Error!void {
        const r = try f.weights.getOrPut(f.gpa, e);
        r.value_ptr.* = @min(if (r.found_existing) r.value_ptr.* else 1.0, w);
    }

    /// The user's circle: "me", "my ..." and "the <word>", and the entities their current notes name.
    fn circleOf(f: *Found, idx: *const Index) Allocator.Error!*const Set {
        if (f.circle) |*c| return c;
        f.circle = .empty;
        const c = &f.circle.?;
        var own: std.ArrayList(u32) = .empty; // the circle's names, whose notes widen it
        defer own.deinit(f.gpa);
        for (0..idx.entityCount()) |e| if (circleName(idx.entityName(@intCast(e)))) {
            try own.append(f.gpa, @intCast(e));
            try c.put(f.gpa, @intCast(e), {});
        };
        for (own.items) |e| {
            var cur = idx.entCur(e);
            while (cur.next()) |p| {
                const v = idx.valueEntityAt(p);
                if (v >= 0) try c.put(f.gpa, @intCast(v), {});
            }
        }
        return c;
    }
};

/// lookup.lookup() who first: the subjects' notes, one hop, and the unsorted notes it ranks, by its scores and weights.
pub fn lookup(idx: *const Index, sc: *Scratch, text: anytype, question: []const u8, qids: []const u32, past: bool, cap: u32, exclude: []const u32, long: index.Long) Allocator.Error!lk.Packed {
    _ = try lk.scored(idx, sc, qids, past, cap, exclude, long);
    var f: Found = .{ .gpa = sc.gpa };
    defer f.deinit();
    try phrases(idx, sc, &f);
    if (idx.self_entity) |me| if (speaks(question) or try mine(idx, sc.gpa, question)) try f.add(me);
    try names(idx, text, sc.gpa, question, &f);
    // the user's unsorted notes do not say whom they are about: those the plain lookup ranks are always candidates
    if (idx.self_entity) |me| for (sc.pool.items) |i| if (idx.entityAt(i) == me and isUnsorted(idx, i)) try sc.loose.append(sc.gpa, i);
    const long_n: u32 = @intCast(sc.long.items.len);
    if (f.subjects.items.len == 0 and sc.loose.items.len == 0) return .{ .chosen = &.{}, .tokens = 0, .candidates = 0, .long = long_n };
    for (sc.pool.items) |i| sc.slot[i].pstamp = 0; // the pool is who first's candidates from here
    sc.pool.clearRetainingCapacity();
    for (f.subjects.items) |e| take(idx, sc, e, past);
    const own = sc.pool.items.len;
    for (0..own) |k| { // one hop: the entities the subjects' own notes name
        const i = sc.pool.items[k];
        const v = idx.valueEntityAt(i);
        if (v < 0 or f.member.contains(@intCast(v))) continue;
        if (f.weights.get(idx.entityAt(i))) |w| try f.lower(@intCast(v), w);
        take(idx, sc, @intCast(v), past);
    }
    for (sc.loose.items) |i| sc.toPool(i);
    sc.keys.clearRetainingCapacity();
    for (sc.pool.items) |i| sc.keys.appendAssumeCapacity(lk.orderKey(sc.get(i) * (f.weights.get(idx.entityAt(i)) orelse 1.0), idx.rankAt(i)));
    return lk.pack(idx, sc, cap, past, @intCast(sc.pool.items.len), long_n);
}

fn isUnsorted(idx: *const Index, i: u32) bool {
    return std.mem.startsWith(u8, idx.attributeName(idx.pairAttribute(idx.pairAt(i))), unsorted);
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
    defer spans.deinit(f.gpa);
    var n: usize = @min(q.len, index.max_phrase, idx.phraseMax());
    while (n >= 2) : (n -= 1) {
        var j: usize = 0;
        while (j + n <= q.len) : (j += 1) {
            const e = idx.phraseOf(index.packPhrase(q[j..][0..n])) orelse continue;
            if (inside(spans.items, j, j + n)) continue;
            try f.add(e);
            try spans.append(f.gpa, .{ j, j + n });
        }
    }
}

fn inside(spans: []const [2]usize, a: usize, b: usize) bool {
    for (spans) |sp| if (sp[0] <= a and b <= sp[1]) return true;
    return false;
}

/// One run of name letters in the question, and the keys of " " and it (ok: encoded in at most max_phrase keys).
const Run = struct { start: usize = 0, end: usize = 0, keys: [index.max_phrase]u32 = undefined, len: usize = 0, ok: bool = false };

/// Each run of name letters names entities alone and as the last of up to `chain` runs one space apart, keys joined.
fn names(idx: *const Index, text: anytype, gpa: Allocator, q: []const u8, f: *Found) Allocator.Error!void {
    if (!anyName(idx)) return;
    var arena = std.heap.ArenaAllocator.init(gpa); // one run's encode at a time
    defer arena.deinit();
    var before: [chain - 1]Run = @splat(.{}); // the runs before this one, the nearest first
    var i: usize = 0;
    while (i < q.len) {
        if (letterAt(q, i) == 0) {
            i += 1;
            continue;
        }
        var run: Run = .{ .start = i, .end = runEnd(q, i) };
        _ = arena.reset(.retain_capacity);
        try encodeRun(idx, text, arena.allocator(), q[run.start..run.end], &run);
        if (run.ok) {
            var keys: [index.max_phrase]u32 = undefined;
            var n = run.len;
            @memcpy(keys[0..n], run.keys[0..n]);
            try named(idx, f, keys[0..n]);
            var from = run.start;
            for (before) |p| {
                if (!p.ok or p.end + 1 != from or q[p.end] != ' ' or n + p.len > index.max_phrase) break;
                std.mem.copyBackwards(u32, keys[p.len..][0..n], keys[0..n]);
                @memcpy(keys[0..p.len], p.keys[0..p.len]);
                n += p.len;
                from = p.start;
                try named(idx, f, keys[0..n]);
            }
        }
        std.mem.copyBackwards(Run, before[1..], before[0 .. before.len - 1]);
        before[0] = run;
        i = run.end;
    }
}

/// Whether some current name fact names an entity (else no run can).
fn anyName(idx: *const Index) bool {
    if (idx.alias_candidates_off) return false;
    if (idx.alias_entities.count() > 0) return true;
    return if (idx.layer.core) |c| c.alias_entities.count() > 0 else false;
}

fn encodeRun(idx: *const Index, text: anytype, a: Allocator, word: []const u8, run: *Run) Allocator.Error!void {
    const spaced = try std.mem.concat(a, u8, &.{ " ", word });
    const ids = text.encode(a, spaced, false) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return, // a run the tokenizer cannot encode names nobody
    };
    if (ids.len == 0 or ids.len > index.max_phrase) return;
    for (ids, run.keys[0..ids.len]) |t, *k| k.* = idx.keyOf(t);
    run.len = ids.len;
    run.ok = true;
}

/// The entities a current name fact calls `keys` (alias.zig's table): one holder, the user's circle's, or all weighted.
fn named(idx: *const Index, f: *Found, keys: []const u32) Allocator.Error!void {
    const cs = idx.aliasCandidates(index.packPhrase(keys)) orelse return;
    if (cs.len == 0) return;
    if (cs.len == 1) return f.add(cs[0]);
    const circle = try f.circleOf(idx);
    var mine_found = false;
    for (cs) |e| if (circle.contains(e)) {
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

/// The byte length of the name letter at q[i] (a Latin, Greek or Cyrillic letter, not U+00D7 or U+00F7), or 0.
fn letterAt(q: []const u8, i: usize) usize {
    const c = q[i];
    if (c < 0x80) return @intFromBool(std.ascii.isAlphabetic(c));
    const n: usize = std.unicode.utf8ByteSequenceLength(c) catch return 0;
    if (i + n > q.len) return 0;
    const cp = std.unicode.utf8Decode(q[i..][0..n]) catch return 0;
    return if ((cp >= 0xC0 and cp <= 0x4FF and cp != 0xD7 and cp != 0xF7) or (cp >= 0x1E00 and cp <= 0x1EFF)) n else 0;
}

/// Where the run of name letters from q[i] ends: letters, with a hyphen between two letters.
fn runEnd(q: []const u8, i: usize) usize {
    var j = i;
    while (true) {
        while (j < q.len) {
            const n = letterAt(q, j);
            if (n == 0) break;
            j += n;
        }
        if (j + 1 >= q.len or q[j] != '-' or letterAt(q, j + 1) == 0) return j;
        j += 1;
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

/// Words compared as ASCII without case.
const Fold = struct {
    pub fn hash(_: Fold, s: []const u8) u64 {
        var h: u64 = 0xcbf29ce484222325;
        for (s) |c| h = (h ^ std.ascii.toLower(c)) *% 0x100000001b3;
        return h;
    }

    pub fn eql(_: Fold, x: []const u8, y: []const u8) bool {
        return std.ascii.eqlIgnoreCase(x, y);
    }
};

/// The runs of ASCII letters of a text.
const Letters = struct {
    s: []const u8,
    i: usize = 0,

    fn next(w: *Letters) ?[]const u8 {
        while (w.i < w.s.len and !std.ascii.isAlphabetic(w.s[w.i])) w.i += 1;
        if (w.i == w.s.len) return null;
        const start = w.i;
        while (w.i < w.s.len and std.ascii.isAlphabetic(w.s[w.i])) w.i += 1;
        return w.s[start..w.i];
    }
};

/// "my X" (X cut at an apostrophe): no entity name's word, an attribute part or a user's note word; one store read.
pub fn mine(idx: *const Index, gpa: Allocator, q: []const u8) Allocator.Error!bool {
    var xs: std.HashMapUnmanaged([]const u8, bool, Fold, 80) = .empty; // each X: whether an entity's name has it
    defer xs.deinit(gpa);
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
        if (x.len > 0) try xs.put(gpa, x, false);
        i = j;
    }
    var left = xs.count();
    if (left == 0) return false;
    for (0..idx.entityCount()) |e| {
        var words: Letters = .{ .s = idx.entityName(@intCast(e)) };
        while (words.next()) |w| if (xs.getPtr(w)) |has| if (!has.*) {
            has.* = true;
            left -= 1;
            if (left == 0) return false;
        };
    }
    for (0..idx.attributeCount()) |k| {
        var parts = std.mem.tokenizeAny(u8, idx.attributeName(@intCast(k)), "_- \t\r\n");
        while (parts.next()) |p| if (xs.get(p)) |has| if (!has) return true;
    }
    const me = idx.self_entity orelse return false;
    var cur = idx.entCur(me);
    while (cur.next()) |p| {
        var words: Letters = .{ .s = idx.lineAt(p) };
        while (words.next()) |w| if (xs.get(w)) |has| if (!has) return true;
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
}

test "name runs: Latin, Greek and Cyrillic letters, a hyphen between two letters, cut at an apostrophe" {
    for ([_][2][]const u8{ .{ "Zo\u{eb}'s cat", "Zo\u{eb}" }, .{ "Anna-Liisa's", "Anna-Liisa" }, .{ "V\u{e4}in\u{f6} x", "V\u{e4}in\u{f6}" }, .{ "Ivo- x", "Ivo" }, .{ "a--b", "a" }, .{ "\u{416}\u{435}\u{43d}\u{44f}!", "\u{416}\u{435}\u{43d}\u{44f}" }, .{ "Ivo\u{2019}s", "Ivo" }, .{ "x\u{d7}y", "x" } }) |c| {
        try std.testing.expectEqualStrings(c[1], c[0][0..runEnd(c[0], 0)]);
    }
    try std.testing.expect(letterAt("\u{a0}", 0) == 0 and letterAt("\u{f7}", 0) == 0 and letterAt("\u{e9}", 0) == 2 and letterAt("\u{1e9e}", 0) == 3);
}

test "my X: an attribute part or a word of the user's notes names the user, a word of an entity's name never does" {
    const idx = try Index.load(std.testing.allocator, @embedFile("testdata/who_bytes_snapshot.json"));
    defer idx.deinit();
    const a = std.testing.allocator;
    for ([_][]const u8{ "my eyes", "MY EYES?", "Where are my bees?", "my town's name", "my van, my son and my eyes" }) |q| try std.testing.expect(try mine(idx, a, q));
    for ([_][]const u8{ "my van", "my son's eyes", "my cousin", "tummy eyes", "myeyes", "my", "my  ", "my 'eyes", "my\u{a0}eyes" }) |q| try std.testing.expect(!try mine(idx, a, q));
    var many: std.ArrayList(u8) = .empty; // one pass over the store for any number of X
    defer many.deinit(a);
    for (0..5000) |k| try many.print(a, "my z{c}q{c} ", .{ @as(u8, 'a') + @as(u8, @intCast(k % 26)), @as(u8, 'a') + @as(u8, @intCast(k / 26 % 26)) });
    try std.testing.expect(!try mine(idx, a, many.items));
    try many.appendSlice(a, "and my eyes");
    try std.testing.expect(try mine(idx, a, many.items));
}
