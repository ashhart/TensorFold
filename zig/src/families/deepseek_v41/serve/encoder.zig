//! DeepSeek-V4.1's encoder: the pre-tokenizer splits and byte-level BPE, one thread's state at a time.
const std = @import("std");
const tokenizer_core = @import("tokenizer");
const unicode = tokenizer_core.unicode;
const Allocator = std.mem.Allocator;
const tokenizer = @import("tokenizer.zig");
const Tokenizer = tokenizer.Tokenizer;
const Id = tokenizer.Id;
const missing = tokenizer.missing;
const L = tokenizer.L;
const M = tokenizer.M;
const N = tokenizer.N;
const P = tokenizer.P;
const S = tokenizer.S;
const WS = tokenizer.WS;
const CJK = tokenizer.CJK;
const CRLF = tokenizer.CRLF;

/// One thread's encoding state: BPE scratch and a direct-mapped cache of short pieces' ids.
pub const Encoder = struct {
    gpa: Allocator,
    tok: *const Tokenizer,
    cache: []Slot,
    sym: std.ArrayList(Id) = .empty,
    prev: std.ArrayList(u32) = .empty,
    next: std.ArrayList(u32) = .empty,
    heap: std.ArrayList(Pending) = .empty,
    lossy: std.ArrayList(u8) = .empty,

    const cache_slots = 1 << 15;
    const key_max = 22;
    const ids_max = 8;
    const none: u32 = std.math.maxInt(u32);

    const Slot = struct {
        key: [key_max]u8 = undefined,
        key_len: u8 = 0xFF, // 0xFF: empty
        n: u8 = 0,
        ids: [ids_max]Id = undefined,
    };

    /// A merge waiting in the heap: ordered by rank, then position (``tokenizers``' ``Merge``).
    const Pending = struct { key: u64, id: Id };

    pub fn create(gpa: Allocator, tok: *const Tokenizer) Allocator.Error!*Encoder {
        const e = try gpa.create(Encoder);
        errdefer gpa.destroy(e);
        const cache = try gpa.alloc(Slot, cache_slots);
        @memset(cache, .{});
        e.* = .{ .gpa = gpa, .tok = tok, .cache = cache };
        return e;
    }

    pub fn destroy(e: *Encoder) void {
        const gpa = e.gpa;
        gpa.free(e.cache);
        e.sym.deinit(gpa);
        e.prev.deinit(gpa);
        e.next.deinit(gpa);
        e.heap.deinit(gpa);
        e.lossy.deinit(gpa);
        gpa.destroy(e);
    }

    /// Appends the ids of ``text`` to ``out`` (``a`` grows ``out``). Invalid UTF-8 reads as U+FFFD.
    pub fn encode(e: *Encoder, a: Allocator, text: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        var valid = text;
        if (!std.unicode.utf8ValidateSlice(text)) {
            e.lossy.clearRetainingCapacity();
            try unicode.appendLossy(&e.lossy, e.gpa, text);
            valid = e.lossy.items;
        }
        var done: usize = 0;
        var at: usize = 0;
        while (e.tok.raw.next(valid, at)) |hit| {
            try e.normalizedPass(a, valid[done..hit.start], out);
            try out.append(a, hit.id);
            done = hit.end;
            at = hit.end;
        }
        try e.normalizedPass(a, valid[done..], out);
    }

    fn normalizedPass(e: *Encoder, a: Allocator, s: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        var done: usize = 0;
        while (e.tok.normalized.next(s, done)) |hit| {
            try e.digits(a, s[done..hit.start], out);
            try out.append(a, hit.id);
            done = hit.end;
        }
        try e.digits(a, s[done..], out);
    }

    inline fn cpAt(s: []const u8, i: usize) unicode.Decoded {
        return unicode.decodeAt(s, i);
    }

    /// Split 1: ``\p{N}{1,3}`` isolated.
    fn digits(e: *Encoder, a: Allocator, s: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        const t = e.tok;
        var gap: usize = 0;
        var i: usize = 0;
        while (i < s.len) {
            const d = cpAt(s, i);
            if (t.class(d.cp) & N == 0) {
                i += d.len;
                continue;
            }
            try e.cjk(a, s[gap..i], out);
            var j = i;
            var count: usize = 0;
            while (j < s.len and count < 3) {
                const x = cpAt(s, j);
                if (t.class(x.cp) & N == 0) break;
                j += x.len;
                count += 1;
            }
            try e.piece(a, s[i..j], out); // a run of numbers has no further split (the word regex matches none of it)
            i = j;
            gap = j;
        }
        try e.cjk(a, s[gap..], out);
    }

    /// Split 2: the CJK and kana runs isolated.
    fn cjk(e: *Encoder, a: Allocator, s: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        if (s.len == 0) return;
        const t = e.tok;
        var gap: usize = 0;
        var i: usize = 0;
        while (i < s.len) {
            if (s[i] < 0xE3 or s[i] > 0xE9) { // every CJK and kana code point here is 3 bytes from E3 to E9
                i += unicode.decodeAt(s, i).len;
                continue;
            }
            const d = cpAt(s, i);
            if (t.class(d.cp) & CJK == 0) {
                i += d.len;
                continue;
            }
            try e.words(a, s[gap..i], out);
            var j = i + d.len;
            while (j < s.len) {
                const x = cpAt(s, j);
                if (t.class(x.cp) & CJK == 0) break;
                j += x.len;
            }
            try e.words(a, s[i..j], out);
            i = j;
            gap = j;
        }
        try e.words(a, s[gap..], out);
    }

    /// Split 3: the word regex, leftmost-first, as matches and the gaps between them.
    fn words(e: *Encoder, a: Allocator, s: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        if (s.len == 0) return;
        const t = e.tok;
        var gap: usize = 0;
        var i: usize = 0;
        while (i < s.len) {
            const end = matchAt(t, s, i) orelse {
                i += unicode.decodeAt(s, i).len;
                continue;
            };
            if (gap < i) try e.piece(a, s[gap..i], out);
            try e.piece(a, s[i..end], out);
            i = end;
            gap = end;
        }
        if (gap < s.len) try e.piece(a, s[gap..], out);
    }

    /// The end of the word regex's match starting at ``i``, or null.
    fn matchAt(t: *const Tokenizer, s: []const u8, i: usize) ?usize {
        const c0 = cpAt(s, i);
        const f0 = t.class(c0.cp);
        const j = i + c0.len;
        const c1: ?unicode.Decoded = if (j < s.len) cpAt(s, j) else null;
        const f1: u8 = if (c1) |c| t.class(c.cp) else 0;
        // [!-/:-@[-`{-~][A-Za-z]+
        if (c0.cp < 0x80 and std.ascii.isPrint(@intCast(c0.cp)) and !std.ascii.isAlphanumeric(@intCast(c0.cp)) and c0.cp != ' ') {
            if (c1 != null and c1.?.cp < 0x80 and std.ascii.isAlphabetic(@intCast(c1.?.cp))) {
                var k = j + 1;
                while (k < s.len and std.ascii.isAlphabetic(s[k])) k += 1;
                return k;
            }
        }
        // [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
        if (f0 & (CRLF | L | P | S) == 0 and f1 & (L | M) != 0) return runOf(t, s, j + c1.?.len, L | M);
        if (f0 & (L | M) != 0) return runOf(t, s, j, L | M);
        // ' '?[\p{P}\p{S}]+[\r\n]*
        if (c0.cp == ' ' and f1 & (P | S) != 0) return runOf(t, s, runOf(t, s, j + c1.?.len, P | S), CRLF);
        if (f0 & (P | S) != 0) return runOf(t, s, runOf(t, s, j, P | S), CRLF);
        if (f0 & WS == 0) return null;
        // \s*[\r\n]+ | \s+(?!\S) | \s+
        var last_crlf: ?usize = if (f0 & CRLF != 0) i else null;
        var last_start = i;
        var k = j;
        while (k < s.len) {
            const x = cpAt(s, k);
            const fx = t.class(x.cp);
            if (fx & WS == 0) break;
            if (fx & CRLF != 0) last_crlf = k;
            last_start = k;
            k += x.len;
        }
        if (last_crlf) |p| return p + 1;
        if (k == s.len) return k;
        if (last_start > i) return last_start; // give the last space back to the next word
        return k;
    }

    inline fn runOf(t: *const Tokenizer, s: []const u8, from: usize, mask: u8) usize {
        var k = from;
        while (k < s.len) {
            const x = cpAt(s, k);
            if (t.class(x.cp) & mask == 0) break;
            k += x.len;
        }
        return k;
    }

    fn hashOf(s: []const u8) u64 {
        return std.hash.Wyhash.hash(0x6473763431, s);
    }

    /// A piece's BPE ids, through the cache.
    fn piece(e: *Encoder, a: Allocator, s: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        if (s.len == 1) return if (e.tok.byte_id[s[0]] != missing) out.append(a, e.tok.byte_id[s[0]]);
        if (s.len <= key_max) {
            const slot = &e.cache[@intCast(hashOf(s) & (cache_slots - 1))];
            if (slot.key_len == s.len and std.mem.eql(u8, slot.key[0..s.len], s)) return out.appendSlice(a, slot.ids[0..slot.n]);
            const before = out.items.len;
            try e.bpe(a, s, out);
            const got = out.items[before..];
            if (got.len <= ids_max) {
                @memcpy(slot.key[0..s.len], s);
                slot.key_len = @intCast(s.len);
                slot.n = @intCast(got.len);
                @memcpy(slot.ids[0..got.len], got);
            }
            return;
        }
        return e.bpe(a, s, out);
    }

    /// ``tokenizers``' ``Word::merge_all``: a min-heap of (rank, position) merges, stale ones skipped.
    fn bpe(e: *Encoder, a: Allocator, s: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        const gpa = e.gpa;
        const n = s.len;
        const m = &e.tok.merges;
        try e.sym.resize(gpa, n);
        try e.prev.resize(gpa, n);
        try e.next.resize(gpa, n);
        const sym = e.sym.items;
        const prev = e.prev.items;
        const next = e.next.items;
        for (s, 0..) |b, i| {
            sym[i] = e.tok.byte_id[b];
            prev[i] = if (i == 0) none else @intCast(i - 1);
            next[i] = if (i + 1 == n) none else @intCast(i + 1);
        }
        e.heap.clearRetainingCapacity();
        for (0..n - 1) |i| if (m.get(sym[i], sym[i + 1])) |v| try e.push(.{ .key = (v >> 32) << 32 | i, .id = @truncate(v) });
        while (e.pop()) |top| {
            const pos: u32 = @truncate(top.key);
            if (sym[pos] == none) continue; // merged into its left neighbour
            const right = next[pos];
            if (right == none) continue;
            const v = m.get(sym[pos], sym[right]) orelse continue;
            if (@as(Id, @truncate(v)) != top.id) continue; // the pair changed since this merge was queued
            sym[pos] = top.id;
            sym[right] = none;
            next[pos] = next[right];
            if (next[right] != none) prev[next[right]] = pos;
            if (prev[pos] != none) if (m.get(sym[prev[pos]], sym[pos])) |pv| try e.push(.{ .key = (pv >> 32) << 32 | prev[pos], .id = @truncate(pv) });
            if (next[pos] != none) if (m.get(sym[pos], sym[next[pos]])) |nv| try e.push(.{ .key = (nv >> 32) << 32 | pos, .id = @truncate(nv) });
        }
        var i: u32 = 0;
        while (i != none) : (i = next[i]) if (sym[i] != missing) try out.append(a, sym[i]);
    }

    fn push(e: *Encoder, p: Pending) Allocator.Error!void {
        try e.heap.append(e.gpa, p);
        const h = e.heap.items;
        var i = h.len - 1;
        while (i > 0) {
            const parent = (i - 1) / 2;
            if (h[parent].key <= p.key) break;
            h[i] = h[parent];
            i = parent;
        }
        h[i] = p;
    }

    fn pop(e: *Encoder) ?Pending {
        const h = e.heap.items;
        if (h.len == 0) return null;
        const top = h[0];
        const last = h[h.len - 1];
        e.heap.items.len -= 1;
        const n = h.len - 1;
        if (n == 0) return top;
        var i: usize = 0;
        while (true) {
            const l = 2 * i + 1;
            if (l >= n) break;
            const c = if (l + 1 < n and h[l + 1].key < h[l].key) l + 1 else l;
            if (h[c].key >= last.key) break;
            h[i] = h[c];
            i = c;
        }
        h[i] = last;
        return top;
    }
};
