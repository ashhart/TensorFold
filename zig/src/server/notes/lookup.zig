//! Vendored from LLWZ notes-zig a81d1a9 (zig/notes/src/notes.zig; MIT, (c) 2026 ThinkOff), Scratch.init from f54344f.
const std = @import("std");
const builtin = @import("builtin");
const index = @import("index.zig");
const sa = @import("sa.zig");
const Index = index.Index;
const Gram = index.Gram;
const Allocator = std.mem.Allocator;

/// One note's per-lookup state, kept together so a score update touches one cache line.
const Slot = struct {
    score: f64,
    stamp: u32, // == gen: the note has a score (Python: i in score)
    pstamp: u32, // == gen: the note is in the pool
};

/// One thread's lookup state: a slot per note, reset by a generation number, and the result's buffers.
pub const Scratch = struct {
    gpa: Allocator,
    slot: []Slot,
    xstamp: []u32, // == gen: the note is excluded from this lookup (apart from slot: read only when some are)
    gen: u32,
    seen: []u32 = &.{}, // sa: == sgen: the note is counted for this n-gram already (a range may hold a note twice)
    sgen: u32 = 0,
    touched: std.ArrayList(u32) = .empty,
    pool: std.ArrayList(u32) = .empty,
    keys: std.ArrayList(u128) = .empty,
    qkeys: std.ArrayList(u32) = .empty,
    qgrams: std.ArrayList(Gram) = .empty,
    found: std.ArrayList(u32) = .empty,
    found_weights: std.ArrayList(f64) = .empty,
    spans: std.ArrayList([2]usize) = .empty,
    chosen: std.ArrayList(u32) = .empty,
    order: std.ArrayList([2]u32) = .empty,
    ids: std.ArrayList(u32) = .empty,
    scores: std.ArrayList(f64) = .empty,
    block: std.ArrayList(u8) = .empty,
    hits: std.ArrayList(u32) = .empty, // sa: one n-gram's notes in the views searched
    codes: std.ArrayList(u32) = .empty, // sa: a pattern's codes
    long: std.ArrayList([2]usize) = .empty, // sa, longer matches: the n-grams added (start and length in qkeys)

    const lists = .{ "touched", "pool", "keys", "qkeys", "qgrams", "found", "found_weights", "spans", "chosen", "order", "ids", "scores", "block", "hits", "codes", "long" };

    pub fn init(gpa: Allocator, idx: *const Index) Allocator.Error!Scratch {
        return initRoom(gpa, idx, 0);
    }

    /// init() with room for `room` notes more (written after it): a store that grows makes one now and then.
    pub fn initRoom(gpa: Allocator, idx: *const Index, room: usize) Allocator.Error!Scratch {
        const n = idx.notes() + room;
        const slot = try gpa.alloc(Slot, n);
        errdefer gpa.free(slot);
        const xstamp = try gpa.alloc(u32, n);
        errdefer gpa.free(xstamp);
        const seen = try gpa.alloc(u32, if (idx.kind == .sa) n else 0);
        errdefer gpa.free(seen);
        @memset(seen, 0);
        var sc: Scratch = .{ .gpa = gpa, .slot = slot, .xstamp = xstamp, .gen = 0, .seen = seen };
        errdefer inline for (lists) |f| @field(sc, f).deinit(gpa);
        @memset(sc.slot, .{ .score = 0, .stamp = 0, .pstamp = 0 });
        @memset(sc.xstamp, 0);
        try sc.touched.ensureTotalCapacity(gpa, n);
        try sc.pool.ensureTotalCapacity(gpa, n);
        try sc.keys.ensureTotalCapacity(gpa, n);
        return sc;
    }

    pub fn deinit(sc: *Scratch) void {
        sc.gpa.free(sc.slot);
        sc.gpa.free(sc.xstamp);
        sc.gpa.free(sc.seen);
        inline for (lists) |f| @field(sc, f).deinit(sc.gpa);
    }

    fn next(sc: *Scratch) void {
        sc.gen +%= 1;
        if (sc.gen == 0) {
            @memset(sc.slot, .{ .score = 0, .stamp = 0, .pstamp = 0 });
            @memset(sc.xstamp, 0);
            sc.gen = 1;
        }
        inline for (lists) |f| @field(sc, f).clearRetainingCapacity();
    }

    /// score[i] = score.get(i, 0.0) + w
    inline fn add(sc: *Scratch, i: u32, w: f64) void {
        const x = &sc.slot[i];
        if (x.stamp != sc.gen) {
            x.stamp = sc.gen;
            x.score = 0.0;
            sc.touched.appendAssumeCapacity(i);
        }
        x.score = x.score + w;
    }

    pub inline fn get(sc: *const Scratch, i: u32) f64 {
        return if (sc.slot[i].stamp == sc.gen) sc.slot[i].score else 0.0;
    }

    pub inline fn toPool(sc: *Scratch, i: u32) void {
        if (sc.slot[i].pstamp != sc.gen) {
            sc.slot[i].pstamp = sc.gen;
            sc.pool.appendAssumeCapacity(i);
        }
    }
};

/// pack()'s choice: note places best first (Python's `chosen`, which fit() trims from the end), and its token sum.
pub const Packed = struct {
    chosen: []const u32,
    tokens: u32, // pack()'s sum: the head, the titles, the lines and the line breaks
    candidates: u32,
    long: u32 = 0, // index.Long: the n-grams longer than n_max that were weighted
};

/// render() of some packed notes.
pub const Rendered = struct {
    ids: []const u32, // Python's note ids: the current ones, then the earlier ones, each by told (render order)
    scores: []const f64, // each one's score, for diagnostics
    block: []const u8, // "" when nothing is retrieved
};

/// Python's order key (-score, -told, id) as one u128 that sorts ascending: score bits, then the tie rank (rankAt).
pub fn orderKey(score: f64, rank: u64) u128 {
    const b: u64 = @bitCast(score);
    const up: u64 = if (b >> 63 != 0) ~b else b | (@as(u64, 1) << 63); // ascending with the score
    return (@as(u128, ~up) << 64) | rank;
}

/// The place an order key was made for.
pub inline fn placeOf(idx: *const Index, k: u128) u32 {
    return idx.placeOfRank(@truncate(k));
}

/// The k smallest keys first, in no order (nth_element); keys are distinct, so this is heapq.nsmallest(k, ...).
fn selectFirst(ks: []u128, k: usize) void {
    if (k >= ks.len) return;
    var lo: usize = 0;
    var hi: usize = ks.len - 1;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2; // median of three, moved to hi
        if (ks[mid] < ks[lo]) std.mem.swap(u128, &ks[mid], &ks[lo]);
        if (ks[hi] < ks[lo]) std.mem.swap(u128, &ks[hi], &ks[lo]);
        if (ks[mid] < ks[hi]) std.mem.swap(u128, &ks[mid], &ks[hi]);
        const p = ks[hi];
        var store = lo;
        var i = lo;
        while (i < hi) : (i += 1) { // Lomuto, without a branch on the comparison
            const x = ks[i];
            ks[i] = ks[store];
            ks[store] = x;
            store += @intFromBool(x < p);
        }
        std.mem.swap(u128, &ks[store], &ks[hi]);
        if (store == k) return;
        if (store < k) lo = store + 1 else hi = store - 1;
    }
}

/// NoteStore.subjects: entities whose phrase the question holds, longest first, none inside another; repeats count.
fn subjects(idx: *const Index, sc: *Scratch) Allocator.Error!void {
    const q = sc.qkeys.items;
    if (idx.phraseCount() == 0 and !idx.settings.resolve_subjects) return;
    const maximum = if (idx.settings.resolve_subjects) @max(idx.phraseMax(), idx.aliasMax()) else idx.phraseMax();
    var n: usize = @min(q.len, index.max_phrase, maximum);
    while (n >= 1) : (n -= 1) {
        var j: usize = 0;
        while (j + n <= q.len) : (j += 1) {
            const phrase = index.packPhrase(q[j..][0..n]);
            const ordinary = if (n >= 2) idx.phraseOf(phrase) else null;
            const alias = if (idx.settings.resolve_subjects) idx.aliasOf(phrase) else null;
            const candidates = if (idx.settings.resolve_subjects and ordinary == null) idx.aliasCandidates(phrase) else null;
            if (ordinary == null and alias == null and candidates == null) continue;
            var inside = false;
            for (sc.spans.items) |sp| {
                if (sp[0] <= j and j + n <= sp[1]) {
                    inside = true;
                    break;
                }
            }
            if (inside) continue;
            const before = sc.found.items.len;
            if (candidates) |cs| {
                // Normalized entity IDF: a unique alias has full strength;

                // common names retain all candidates with weaker evidence.
                const entities: f64 = @floatFromInt(@max(idx.entityCount(), cs.len));
                const count: f64 = @floatFromInt(cs.len);
                const weight = idx.log(1.0 + (entities - count + 0.5) / (count + 0.5)) /
                    idx.log(1.0 + (entities - 0.5) / 1.5);
                for (cs) |e| {
                    if (!idx.settings.resolve_self and idx.self_entity == e) continue;
                    try sc.found.append(sc.gpa, e);
                    try sc.found_weights.append(sc.gpa, weight);
                }
            } else {
                const e = ordinary orelse alias.?;
                if (e == index.no_note) continue;
                if (!idx.settings.resolve_self and idx.self_entity == e) continue;
                try sc.found.append(sc.gpa, e);
                try sc.found_weights.append(sc.gpa, 1.0);
            }
            if (sc.found.items.len > before) try sc.spans.append(sc.gpa, .{ j, j + n });
        }
    }
}

/// The base's notes holding packed n-gram `g`, each once: the current array less notes retired since, or both arrays.
fn baseHits(idx: *const Index, b: *const sa.Base, sc: *Scratch, g: Gram, past: bool) Allocator.Error!void {
    sc.hits.clearRetainingCapacity();
    var keys: [index.max_n]u32 = undefined;
    for (&keys, 0..) |*k, j| k.* = @truncate(g >> @intCast(32 * j));
    var len: usize = index.max_n; // a packed n-gram's unused places hold none_key
    while (len > 0 and keys[len - 1] == index.none_key) len -= 1;
    if (len == 0) return;
    try keyHits(idx, b, sc, keys[0..len], past);
}

/// The base's notes that hold the key sequence `keys`, each once, in the views searched, into sc.hits.
fn keyHits(idx: *const Index, b: *const sa.Base, sc: *Scratch, keys: []const u32, past: bool) Allocator.Error!void {
    sc.hits.clearRetainingCapacity();
    for (keys) |k| if (k == index.none_key) return; // no note holds none_key (the build checks the vocabulary)
    try sc.codes.resize(sc.gpa, keys.len);
    const pat = b.codes(keys, sc.codes.items) orelse return;
    if (keys.len == 1 and !idx.settings.punct and !b.wordyCode(pat[0])) return; // punctuation alone is no note's key
    sc.sgen +%= 1;
    if (sc.sgen == 0) {
        @memset(sc.seen, 0);
        sc.sgen = 1;
    }
    const views = [2]sa.Packed{ b.sa_cur, b.sa_hist };
    for (views[0..if (past) 2 else 1]) |arr| {
        const r = b.range(arr, pat);
        for (r[0]..r[1]) |k| {
            const place = b.placeOf(arr.get(k));
            if (sc.seen[place] == sc.sgen) continue;
            sc.seen[place] = sc.sgen;
            if (!past and idx.stateAt(place) != .current) continue; // retired since the base was built
            try sc.hits.append(sc.gpa, place);
        }
    }
}

/// One n-gram's weight added to its notes: the base's from ranges, the side buffer's from postings; df counts notes.
fn scoreSa(idx: *const Index, b: *const sa.Base, sc: *Scratch, g: Gram, past: bool, N: u32) Allocator.Error!void {
    try baseHits(idx, b, sc, g, past);
    const side = idx.grams.get(g);
    var df: u32 = @intCast(sc.hits.items.len);
    if (side) |p| df += p.cur_len + (if (past) p.hist_len else 0);
    if (df == 0) return;
    const w = idx.log(1.0 + (@as(f64, @floatFromInt(N - df)) + 0.5) / (@as(f64, @floatFromInt(df)) + 0.5));
    for (sc.hits.items) |i| sc.add(i, w);
    if (side) |p| {
        for (idx.post[p.cur_off..][0..p.cur_len]) |i| sc.add(i, w);
        if (past) for (idx.post[p.hist_off..][0..p.hist_len]) |i| sc.add(i, w);
    }
}

/// index.Long: n-grams over n_max keys weighed by their notes as the shorter ones are; one met again counts once.
fn scoreLong(idx: *const Index, b: *const sa.Base, sc: *Scratch, past: bool, N: u32, long: index.Long) Allocator.Error!void {
    const q = sc.qkeys.items;
    const n_max = idx.settings.n_max;
    var reach: usize = 0; // where the longest match so far ends: a match inside it is a part of it
    for (0..q.len) |j| {
        var best: usize = 0;
        var m: usize = n_max + 1;
        const limit = if (long.mode == .all) @min(q.len - j, long.max) else q.len - j;
        while (m <= limit) : (m += 1) {
            if (!try longHits(idx, b, sc, q[j..][0..m], past)) break;
            best = m;
            if (long.mode == .all) try addLong(idx, sc, j, m, N);
        }
        if (long.mode == .longest and best > 0 and j + best > reach) {
            _ = try longHits(idx, b, sc, q[j..][0..best], past);
            try addLong(idx, sc, j, best, N);
            reach = j + best;
        }
    }
}

/// sc.hits: the notes holding `keys` in the views searched (base ranges, side notes read through); whether any do.
fn longHits(idx: *const Index, b: *const sa.Base, sc: *Scratch, keys: []const u32, past: bool) Allocator.Error!bool {
    try keyHits(idx, b, sc, keys, past);
    for (0..idx.side_keys.len()) |k| {
        const place: u32 = @intCast(b.notes + k);
        if (!past and idx.stateAt(place) != .current) continue;
        if (std.mem.indexOf(u32, idx.side_keys.of(k), keys) != null) try sc.hits.append(sc.gpa, place);
    }
    return sc.hits.items.len > 0;
}

/// The weight of the n-gram at q[j..j+m] (its notes in sc.hits) added, once per distinct n-gram.
fn addLong(idx: *const Index, sc: *Scratch, j: usize, m: usize, N: u32) Allocator.Error!void {
    const q = sc.qkeys.items;
    for (sc.long.items) |x| if (x[1] == m and std.mem.eql(u32, q[x[0]..][0..m], q[j..][0..m])) return;
    try sc.long.append(sc.gpa, .{ j, m });
    const df: u32 = @intCast(sc.hits.items.len);
    const w = idx.log(1.0 + (@as(f64, @floatFromInt(N - df)) + 0.5) / (@as(f64, @floatFromInt(df)) + 0.5));
    for (sc.hits.items) |i| sc.add(i, w);
}

/// NoteStore._lookup up to pack() from a question's ids; `exclude` ids are never given; sums run in first-met order.
pub fn lookup(idx: *const Index, sc: *Scratch, qids: []const u32, past: bool, cap: u32, exclude: []const u32, long: index.Long) Allocator.Error!Packed {
    const cands = try scored(idx, sc, qids, past, cap, exclude, long);
    const long_n: u32 = @intCast(sc.long.items.len);
    if (cands == 0) return .{ .chosen = &.{}, .tokens = 0, .candidates = 0, .long = long_n };
    sc.keys.clearRetainingCapacity();
    for (sc.pool.items) |i| sc.keys.appendAssumeCapacity(orderKey(sc.slot[i].score, idx.rankAt(i)));
    return pack(idx, sc, cap, past, cands, long_n);
}

/// lookup() up to its pool: every note's score after the hop (sc.get), the pool, and how many notes scored.
pub fn scored(idx: *const Index, sc: *Scratch, qids: []const u32, past: bool, cap: u32, exclude: []const u32, long: index.Long) Allocator.Error!u32 {
    const st = idx.settings;
    const gpa = sc.gpa;
    sc.next();
    var excluding = false;
    for (exclude) |id| {
        if (idx.placeOfId(id)) |p| {
            sc.xstamp[p] = sc.gen;
            excluding = true;
        }
    }
    for (qids) |t| try sc.qkeys.append(gpa, idx.keyOf(t));
    try idx.gramsOf(gpa, qids, sc.qkeys.items, &sc.qgrams);

    // IDF over the views searched: the current notes, and for a question about the past the earlier ones too
    const N: u32 = idx.n_cur + (if (past) idx.n_hist else 0);
    if (idx.base) |b| {
        for (sc.qgrams.items) |g| try scoreSa(idx, b, sc, g, past, N);
        if (long.mode != .off) try scoreLong(idx, b, sc, past, N, long);
    } else for (sc.qgrams.items) |g| {
        const p = idx.grams.get(g) orelse continue;
        const df: u32 = p.cur_len + (if (past) p.hist_len else 0);
        if (df == 0) continue;
        const w = idx.log(1.0 + (@as(f64, @floatFromInt(N - df)) + 0.5) / (@as(f64, @floatFromInt(df)) + 0.5));
        for (idx.post[p.cur_off..][0..p.cur_len]) |i| sc.add(i, w);
        if (past) for (idx.post[p.hist_off..][0..p.hist_len]) |i| sc.add(i, w);
    }
    // the subject boost: the top IDF for each note of an entity the question names
    if (st.subject > 0 and N > 0) {
        const top = idx.log(1.0 + (@as(f64, @floatFromInt(N)) - 0.5) / 1.5) * st.subject;
        try subjects(idx, sc);
        for (sc.found.items, sc.found_weights.items) |e, weight| {
            var cur = idx.entCur(e);
            while (cur.next()) |i| sc.add(i, top * weight);
            if (past) {
                var ks = idx.entKeys(e);
                while (ks.next()) |k| {
                    var ns = idx.pairNotes(k);
                    while (ns.next()) |i| sc.add(i, top * weight);
                }
            }
        }
    }
    if (excluding) { // for i in exclude: score.pop(i, None)
        var w: usize = 0;
        for (sc.touched.items) |i| {
            if (sc.xstamp[i] == sc.gen) {
                sc.slot[i].stamp = 0; // gen is never 0: the note has no score now
            } else {
                sc.touched.items[w] = i;
                w += 1;
            }
        }
        sc.touched.shrinkRetainingCapacity(w);
    }
    const cands: u32 = @intCast(sc.touched.items.len);
    if (cands == 0) return 0;

    // the candidate cut: the 4*room best when there are more than 8*room candidates (4 tokens a line or more)
    const room: usize = @intCast(@max(1, @divFloor(@as(i64, cap) - @as(i64, idx.fx_head), 4)));
    for (sc.touched.items) |i| sc.keys.appendAssumeCapacity(orderKey(sc.slot[i].score, idx.rankAt(i)));
    var direct = sc.keys.items;
    if (direct.len > 8 * room) {
        selectFirst(direct, 4 * room);
        direct = direct[0 .. 4 * room];
    }
    // Python reads direct's order only for the best one (the floor) and the first hop_from: only those are sorted
    const head_n = @min(direct.len, @max(st.hop_from, 1));
    selectFirst(direct, head_n);
    std.sort.pdq(u128, direct[0..head_n], {}, std.sort.asc(u128));
    if (st.min_rel > 0) {
        // the kept ones keep their order, so the sorted head stays first
        const floor = sc.slot[placeOf(idx, direct[0])].score * st.min_rel;
        var w: usize = 0;
        for (direct) |k| {
            if (sc.slot[placeOf(idx, k)].score >= floor) {
                direct[w] = k;
                w += 1;
            }
        }
        direct = direct[0..w];
    }
    for (direct) |k| sc.toPool(placeOf(idx, k));
    // one hop from the best direct matches; a hop note scores hop_weight of the note that brought it in
    for (direct[0..@min(direct.len, st.hop_from)]) |k| {
        const i = placeOf(idx, k);
        const s = sc.slot[i].score * st.hop_weight;
        var near: [3]index.Iter = undefined;
        var nn: usize = 0;
        if (st.hop != .none) {
            near[nn] = idx.entCur(idx.entityAt(i));
            nn += 1;
            if (st.hop == .subject_value and idx.valueEntityAt(i) >= 0) {
                near[nn] = idx.entCur(@intCast(idx.valueEntityAt(i)));
                nn += 1;
            }
        }
        if (past and st.hop_history) {
            near[nn] = idx.pairNotes(idx.pairAt(i));
            nn += 1;
        }
        for (near[0..nn]) |*js| while (js.next()) |j| {
            if (excluding and sc.xstamp[j] == sc.gen) continue;
            if (s > sc.get(j)) {
                sc.slot[j].stamp = sc.gen;
                sc.slot[j].score = s;
            }
            sc.toPool(j);
        };
    }
    return cands;
}

/// pack() over sc.keys: sorted a part at a time; the walk stops once no line fits, where Python's walk only skips.
pub fn pack(idx: *const Index, sc: *Scratch, cap: u32, past: bool, cands: u32, long_n: u32) Allocator.Error!Packed {
    const gpa = sc.gpa;
    var used: u32 = idx.fx_head;
    var has_cur = false;
    var has_hist = false;
    var rest = sc.keys.items;
    var first = true;
    walk: while (rest.len > 0) {
        const m = if (first) @min(rest.len, 64) else rest.len; // the first 64 by selection, then the rest whole
        first = false;
        selectFirst(rest, m);
        std.sort.pdq(u128, rest[0..m], {}, std.sort.asc(u128));
        for (rest[0..m]) |k| {
            if (@as(i64, cap) - @as(i64, used) < idx.min_line) break :walk;
            const i = placeOf(idx, k);
            const cur = idx.stateAt(i) == .current;
            if (!cur and !past) continue;
            const titled = if (cur) has_cur else has_hist;
            const c = idx.costAt(i) + idx.fx_nl + (if (titled) 0 else (if (cur) idx.fx_cur else idx.fx_hist) + idx.fx_nl);
            if (@as(u64, used) + c <= cap) {
                used += c;
                try sc.chosen.append(gpa, i);
                if (cur) has_cur = true else has_hist = true;
            }
        }
        rest = rest[m..];
    }
    if (sc.chosen.items.len == 0) return .{ .chosen = &.{}, .tokens = 0, .candidates = cands, .long = long_n };
    return .{ .chosen = sc.chosen.items, .tokens = used, .candidates = cands, .long = long_n };
}

/// render(): the head, then the current notes and the earlier ones, each by told (ties keep pack order).
pub fn render(idx: *const Index, sc: *Scratch, chosen: []const u32) Allocator.Error!Rendered {
    const gpa = sc.gpa;
    sc.order.clearRetainingCapacity();
    sc.ids.clearRetainingCapacity();
    sc.scores.clearRetainingCapacity();
    sc.block.clearRetainingCapacity();
    if (chosen.len == 0) return .{ .ids = &.{}, .scores = &.{}, .block = "" };
    for (chosen, 0..) |i, k| try sc.order.append(gpa, .{ i, @intCast(k) });
    const ByTold = struct {
        idx: *const Index,
        fn less(c: @This(), a: [2]u32, b: [2]u32) bool {
            const ca = c.idx.stateAt(a[0]) == .current;
            const cb = c.idx.stateAt(b[0]) == .current;
            if (ca != cb) return ca;
            if (c.idx.toldAt(a[0]) != c.idx.toldAt(b[0])) return c.idx.toldAt(a[0]) < c.idx.toldAt(b[0]);
            return a[1] < b[1];
        }
    };
    std.sort.pdq([2]u32, sc.order.items, ByTold{ .idx = idx }, ByTold.less);
    try sc.block.appendSlice(gpa, idx.head);
    var in_cur = false;
    var in_hist = false;
    for (sc.order.items) |x| {
        const i = x[0];
        if (idx.stateAt(i) == .current) {
            if (!in_cur) {
                try sc.block.append(gpa, '\n');
                try sc.block.appendSlice(gpa, idx.cur_title);
                in_cur = true;
            }
        } else if (!in_hist) {
            try sc.block.append(gpa, '\n');
            try sc.block.appendSlice(gpa, idx.hist_title);
            in_hist = true;
        }
        try sc.block.append(gpa, '\n');
        try sc.block.appendSlice(gpa, idx.lineAt(i));
        try sc.ids.append(gpa, idx.idAt(i));
        try sc.scores.append(gpa, sc.slot[i].score);
    }
    return .{ .ids = sc.ids.items, .scores = sc.scores.items, .block = sc.block.items };
}

test "the order key sorts as Python's (-score, -told, id), and selectFirst as sorted()[:k]" {
    var rng = std.Random.DefaultPrng.init(7);
    const r = rng.random();
    const n = 500;
    var score: [n]f64 = undefined;
    var told: [n]i64 = undefined;
    var keys: [n]u128 = undefined;
    var places: [n]u32 = undefined;
    for (0..n) |i| {
        score[i] = @as(f64, @floatFromInt(r.uintLessThan(u32, 20))) * 0.37; // many ties on score
        told[i] = @intCast(r.uintLessThan(u32, 5)); // and on told
        places[i] = @intCast(i);
    }
    const T = struct {
        told: []const i64,
        fn less(c: @This(), x: u32, y: u32) bool {
            if (c.told[x] != c.told[y]) return c.told[x] > c.told[y];
            return x < y;
        }
    };
    var by = places;
    std.sort.pdq(u32, &by, T{ .told = &told }, T.less);
    var trank: [n]u32 = undefined;
    for (by, 0..) |x, k| trank[x] = @intCast(k);
    for (0..n) |i| keys[i] = orderKey(score[i], (@as(u64, trank[i]) << 32) | i); // the full build's layout
    const P = struct {
        score: []const f64,
        told: []const i64,
        fn less(c: @This(), a: u32, b: u32) bool {
            if (c.score[a] != c.score[b]) return c.score[a] > c.score[b];
            if (c.told[a] != c.told[b]) return c.told[a] > c.told[b];
            return a < b;
        }
    };
    std.sort.pdq(u32, &places, P{ .score = &score, .told = &told }, P.less);
    var all = keys;
    std.sort.pdq(u128, &all, {}, std.sort.asc(u128));
    for (all, 0..) |k, i| try std.testing.expectEqual(places[i], @as(u32, @truncate(k)));
    for ([_]usize{ 1, 2, 37, 250, 499 }) |k| {
        var ks = keys;
        selectFirst(&ks, k);
        std.sort.pdq(u128, ks[0..k], {}, std.sort.asc(u128));
        try std.testing.expectEqualSlices(u128, all[0..k], ks[0..k]);
    }
    try std.testing.expect(orderKey(0.5, 0) < orderKey(0.25, 0)); // a higher score sorts first
    try std.testing.expect(orderKey(0.5, 9) < orderKey(0.5, 1 << 32)); // then the tie order
}

test "the weights' log is the C library's bit for bit, where Zig's own log differs" {
    // inputs of the IDF weight where Zig's own log and macOS's differ in the last bit; libc is CPython's math.log
    const Case = struct { n: u32, df: u32, x: u64, libc: u64 };
    const cases = [_]Case{
        .{ .n = 5, .df = 5, .x = 0x3ff1745d1745d174, .libc = 0x3fb64660aa8ce621 }, // log(12/11)
        .{ .n = 32, .df = 12, .x = 0x40051eb851eb851e, .libc = 0x3fef109ef29ae864 },
        .{ .n = 57, .df = 48, .x = 0x3ff3224f2cddb0d3, .libc = 0x3fc6e58398083264 },
        .{ .n = 62, .df = 30, .x = 0x4000864b8a7de6d2, .libc = 0x3fe7368aed7f9a61 },
        .{ .n = 62, .df = 54, .x = 0x3ff27ed3604b27ed, .libc = 0x3fc28d32b8214a56 },
        .{ .n = 73, .df = 29, .x = 0x4004115b1e5f7527, .libc = 0x3fed6df9e8a6470e },
    };
    if (!builtin.os.tag.isDarwin()) return error.SkipZigTest; // the bits are macOS's C library's
    const f = index.libmLog();
    try std.testing.expect(!index.isZigLog(f));
    var own_differs: usize = 0;
    inline for (cases) |c| {
        const x: f64 = comptime 1.0 + (@as(f64, @floatFromInt(c.n - c.df)) + 0.5) / (@as(f64, @floatFromInt(c.df)) + 0.5);
        try std.testing.expectEqual(c.x, @as(u64, @bitCast(x)));
        try std.testing.expectEqual(c.libc, @as(u64, @bitCast(f(x))));
        // negative control: Zig's own log of the same input gives other bits, so this test can fail
        const own: u64 = comptime @bitCast(@log(x));
        if (own != c.libc) own_differs += 1;
    }
    try std.testing.expectEqual(cases.len, own_differs);
}
