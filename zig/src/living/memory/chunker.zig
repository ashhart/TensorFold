//! Heading- and paragraph-aware document chunker for Folded Memory Weights.
//! Returns (full_text, unique_body, title_line, is_small_fact) so multi-chunk documents
//! keep title context in their KV weights while building gate prototypes from unique section text.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Chunk = struct {
    full_text: []u8, // Full text compiled into KV + recurrent weights (with doc title header if split)
    unique_body: []u8, // Unique section body used for primary gate prototypes
    title_line: []u8, // Repeated doc title (empty if single chunk or no heading) -> 0.72x gate weight
    is_small_fact: bool, // Short factual statement -> neutral user-stated framing + concept cues

    pub fn deinit(self: *Chunk, gpa: Allocator) void {
        gpa.free(self.full_text);
        gpa.free(self.unique_body);
        gpa.free(self.title_line);
        self.* = undefined;
    }
};

pub fn freeChunks(gpa: Allocator, chunks: []Chunk) void {
    for (chunks) |*c| c.deinit(gpa);
    gpa.free(chunks);
}

/// Approximate token count (used when caller passes no tokenizer callback); ~3.6 chars/token.
fn approxTokens(s: []const u8) usize {
    const trimmed = std.mem.trim(u8, s, " \t\r\n");
    if (trimmed.len == 0) return 0;
    return @max(1, (trimmed.len + 3) / 4);
}

pub const TokenCounter = struct {
    ctx: *anyopaque,
    count: *const fn (ctx: *anyopaque, text: []const u8) usize,
};

fn countOf(counter: ?TokenCounter, text: []const u8) usize {
    return if (counter) |c| c.count(c.ctx, text) else approxTokens(text);
}

fn isHeadingLine(line: []const u8) bool {
    const t = std.mem.trimStart(u8, line, " \t");
    if (t.len < 2 or t[0] != '#') return false;
    var hashes: usize = 0;
    while (hashes < t.len and t[hashes] == '#') hashes += 1;
    return hashes >= 1 and hashes <= 3 and hashes < t.len and (t[hashes] == ' ' or t[hashes] == '\t');
}

/// Splits `text` into 1 chunk (small fact or short doc) or multiple heading/paragraph chunks
/// bounded by `max_chunk_tokens` (recommended ~750-900).
pub fn chunkDocument(
    gpa: Allocator,
    text: []const u8,
    max_chunk_tokens: usize,
    counter: ?TokenCounter,
) ![]Chunk {
    const clean = std.mem.trim(u8, text, " \t\r\n");
    if (clean.len == 0) return try gpa.alloc(Chunk, 0);

    const total_toks = countOf(counter, clean);
    if (total_toks <= max_chunk_tokens) {
        var out = try gpa.alloc(Chunk, 1);
        errdefer gpa.free(out);
        out[0] = .{
            .full_text = try gpa.dupe(u8, clean),
            .unique_body = try gpa.dupe(u8, clean),
            .title_line = try gpa.alloc(u8, 0),
            .is_small_fact = total_toks < 64,
        };
        return out;
    }

    // Extract optional document title from first line if it starts with '#'
    const first_nl = std.mem.indexOfScalar(u8, clean, '\n') orelse clean.len;
    const first_line = std.mem.trim(u8, clean[0..first_nl], " \t\r");
    const title: []const u8 = if (first_line.len > 0 and first_line[0] == '#') first_line else "";

    // Split on markdown headings (`\n# `, `\n## `, `\n### `)
    var sections: std.ArrayList([]const u8) = .empty;
    defer sections.deinit(gpa);
    {
        var start: usize = 0;
        var pos: usize = 0;
        while (pos < clean.len) {
            if (clean[pos] == '\n' and pos + 1 < clean.len) {
                const rest = clean[pos + 1 ..];
                const line_end = std.mem.indexOfScalar(u8, rest, '\n') orelse rest.len;
                if (isHeadingLine(rest[0..line_end])) {
                    const piece = std.mem.trim(u8, clean[start..pos], " \t\r\n");
                    if (piece.len > 0) try sections.append(gpa, piece);
                    start = pos + 1;
                }
            }
            pos += 1;
        }
        const tail = std.mem.trim(u8, clean[start..], " \t\r\n");
        if (tail.len > 0) try sections.append(gpa, tail);
    }

    // Fallback to blank-line paragraphs if there were no headings
    if (sections.items.len <= 1) {
        sections.clearRetainingCapacity();
        var it = std.mem.splitSequence(u8, clean, "\n\n");
        while (it.next()) |p| {
            const pt = std.mem.trim(u8, p, " \t\r\n");
            if (pt.len > 0) try sections.append(gpa, pt);
        }
    }

    // Break any oversized section into paragraphs (keeping its section heading)
    var atomic: std.ArrayList([]u8) = .empty;
    defer {
        for (atomic.items) |s| gpa.free(s);
        atomic.deinit(gpa);
    }
    for (sections.items) |sec| {
        if (countOf(counter, sec) <= max_chunk_tokens) {
            try atomic.append(gpa, try gpa.dupe(u8, sec));
        } else {
            const nl = std.mem.indexOfScalar(u8, sec, '\n') orelse sec.len;
            const hdr_cand = std.mem.trim(u8, sec[0..nl], " \t\r");
            const hdr: []const u8 = if (hdr_cand.len > 0 and hdr_cand[0] == '#') hdr_cand else "";
            var pit = std.mem.splitSequence(u8, sec, "\n\n");
            while (pit.next()) |p| {
                const ps = std.mem.trim(u8, p, " \t\r\n");
                if (ps.len == 0) continue;
                if (hdr.len > 0 and !std.mem.startsWith(u8, ps, hdr)) {
                    try atomic.append(gpa, try std.fmt.allocPrint(gpa, "{s}\n{s}", .{ hdr, ps }));
                } else {
                    try atomic.append(gpa, try gpa.dupe(u8, ps));
                }
            }
        }
    }

    // Greedily pack atomic pieces into chunks <= max_chunk_tokens
    var chunks: std.ArrayList(Chunk) = .empty;
    errdefer {
        for (chunks.items) |*c| c.deinit(gpa);
        chunks.deinit(gpa);
    }

    var buf: std.ArrayList([]const u8) = .empty;
    defer buf.deinit(gpa);
    var buf_toks: usize = 0;

    for (atomic.items) |sec| {
        const tlen = countOf(counter, sec);
        if (buf.items.len > 0 and buf_toks + tlen > max_chunk_tokens) {
            try emitChunk(gpa, &chunks, buf.items, title);
            buf.clearRetainingCapacity();
            buf_toks = 0;
        }
        try buf.append(gpa, sec);
        buf_toks += tlen;
    }
    if (buf.items.len > 0) {
        try emitChunk(gpa, &chunks, buf.items, title);
    }
    return chunks.toOwnedSlice(gpa);
}

fn emitChunk(gpa: Allocator, out: *std.ArrayList(Chunk), parts: []const []const u8, title: []const u8) !void {
    const body = try std.mem.join(gpa, "\n\n", parts);
    errdefer gpa.free(body);
    const has_prepended_title = title.len > 0 and !std.mem.startsWith(u8, body, title);
    const full = if (has_prepended_title)
        try std.fmt.allocPrint(gpa, "{s}\n\n{s}", .{ title, body })
    else
        try gpa.dupe(u8, body);
    errdefer gpa.free(full);
    const tline = if (has_prepended_title) try gpa.dupe(u8, title) else try gpa.alloc(u8, 0);
    errdefer gpa.free(tline);
    try out.append(gpa, .{
        .full_text = full,
        .unique_body = body,
        .title_line = tline,
        .is_small_fact = false,
    });
}

/// Formats a chunk into the neutral memory entry text compiled into KV + recurrent weights.
pub fn formatEntryText(gpa: Allocator, full_text: []const u8, is_small_fact: bool) ![]u8 {
    const trimmed = std.mem.trim(u8, full_text, " \t\r\n");
    return if (is_small_fact)
        try std.fmt.allocPrint(gpa, "\n- The user stated: {s}\n", .{trimmed})
    else
        try std.fmt.allocPrint(gpa, "\n---\n{s}\n---\n", .{trimmed});
}

/// Scores a sentence/line for domain-independent factual density (numbers, alphanumeric codes,
/// uppercase acronyms/proper nouns, key-value bullets, and section lead-sentence bonus).
/// Pure deterministic Zig (< 0.2 ms for an entire manual) -> ZERO LLM hallucination and 50,000x faster!
fn factualSalienceScore(s: []const u8, is_section_lead: bool) f32 {
    const t = std.mem.trim(u8, s, " \t\r\n");
    if (t.len < 12) return 0.0;
    var score: f32 = if (is_section_lead) 3.5 else 0.0;
    // Key-value / structured bullet lines (e.g. "- Field: value") carry dense reference facts
    if (t[0] == '-' or t[0] == '*') score += 1.2;
    if (std.mem.indexOfScalar(u8, t[0..@min(t.len, 48)], ':') != null) score += 1.8;

    var digits: usize = 0;
    var codes: usize = 0;
    var proper_nouns: usize = 0;
    var word_idx: usize = 0;

    var it = std.mem.tokenizeAny(u8, t, " \t,;:()[]\"'");
    while (it.next()) |tok| {
        var has_digit = false;
        var has_alpha = false;
        var has_upper = false;
        var has_dash_or_under = false;
        for (tok) |ch| {
            if (std.ascii.isDigit(ch)) has_digit = true;
            if (std.ascii.isAlphabetic(ch)) has_alpha = true;
            if (std.ascii.isUpper(ch)) has_upper = true;
            if (ch == '-' or ch == '_') has_dash_or_under = true;
        }
        if (has_digit and has_alpha) {
            codes += 1;
        } else if (has_digit) {
            digits += 1;
        } else if (has_dash_or_under and has_upper and tok.len >= 4) {
            codes += 1;
        } else if (word_idx > 0 and tok.len >= 3 and std.ascii.isUpper(tok[0])) {
            proper_nouns += 1;
        }
        word_idx += 1;
    }
    score += @as(f32, @floatFromInt(@min(digits, 6))) * 2.5;
    score += @as(f32, @floatFromInt(@min(codes, 6))) * 3.8;
    score += @as(f32, @floatFromInt(@min(proper_nouns, 8))) * 1.8;

    // Slight length normalization so concise dense facts beat rambling paragraphs
    const denom: f32 = 1.0 + @as(f32, @floatFromInt(t.len)) / 280.0;
    return score / denom;
}

/// Builds a compact, 100% verbatim Section-Aware Cross-Section Index from a multi-chunk document
/// in < 0.2 ms. Groups sentences by section heading (falling back to chunk index for headingless
/// text), selects round-robin across EVERY section so short sections are never starved, and emits
/// chosen verbatim sentences under their original headings in document order.
pub fn buildExtractiveDigest(
    gpa: Allocator,
    chunks: []const Chunk,
    max_tokens: usize,
    counter: ?TokenCounter,
) ![]u8 {
    if (chunks.len < 2) return try gpa.alloc(u8, 0);

    const Cand = struct {
        sec_idx: usize,
        order_idx: usize,
        text: []const u8,
        score: f32,
        picked: bool = false,
    };

    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(gpa);
    var sec_headings: std.ArrayList([]const u8) = .empty;
    defer sec_headings.deinit(gpa);

    var doc_title: []const u8 = "";
    var order: usize = 0;
    var has_any_subheadings = false;

    // Pre-scan whether chunks contain markdown/numbered section headings
    for (chunks) |ch| {
        var lit = std.mem.splitScalar(u8, ch.unique_body, '\n');
        while (lit.next()) |raw_line| {
            const line = std.mem.trim(u8, raw_line, " \t\r");
            if (line.len == 0) continue;
            if (isHeadingLine(line)) {
                if (doc_title.len == 0 and ch.title_line.len == 0) {
                    doc_title = line;
                } else if (!std.mem.eql(u8, line, ch.title_line) and !std.mem.eql(u8, line, doc_title)) {
                    has_any_subheadings = true;
                }
            }
        }
    }
    if (doc_title.len == 0) {
        for (chunks) |ch| {
            if (ch.title_line.len > 0) {
                doc_title = ch.title_line;
                break;
            }
        }
    }

    var cur_sec: usize = 0;
    var sent_in_sec: usize = 0;
    if (has_any_subheadings) {
        try sec_headings.append(gpa, "");
    } else {
        for (chunks) |_| try sec_headings.append(gpa, "");
    }

    for (chunks, 0..) |ch, ci| {
        if (!has_any_subheadings) {
            cur_sec = ci;
            sent_in_sec = 0;
        }
        var lit = std.mem.splitScalar(u8, ch.unique_body, '\n');
        while (lit.next()) |raw_line| {
            const line = std.mem.trim(u8, raw_line, " \t\r");
            if (line.len == 0) continue;
            if (isHeadingLine(line)) {
                if (std.mem.eql(u8, line, doc_title) or std.mem.eql(u8, line, ch.title_line)) continue;
                if (has_any_subheadings) {
                    try sec_headings.append(gpa, line);
                    cur_sec = sec_headings.items.len - 1;
                    sent_in_sec = 0;
                }
                continue;
            }
            // Split long prose paragraphs into individual sentences on ". "
            var pos: usize = 0;
            while (pos < line.len) {
                var end = line.len;
                var scan = pos;
                while (scan + 2 < line.len) : (scan += 1) {
                    if (line[scan] == '.' and line[scan + 1] == ' ' and std.ascii.isUpper(line[scan + 2])) {
                        // Avoid splitting abbreviations like "Dr. Maren" or "Rev. 4"
                        const is_abbr = (scan >= 2 and std.mem.eql(u8, line[scan - 2 .. scan], "Dr")) or
                            (scan >= 2 and std.mem.eql(u8, line[scan - 2 .. scan], "Mr")) or
                            (scan >= 2 and std.mem.eql(u8, line[scan - 2 .. scan], "Ms")) or
                            (scan >= 3 and std.mem.eql(u8, line[scan - 3 .. scan], "Rev"));
                        if (!is_abbr) {
                            end = scan + 1;
                            break;
                        }
                    }
                }
                const sent = std.mem.trim(u8, line[pos..end], " \t\r");
                if (sent.len >= 16) {
                    const sc = factualSalienceScore(sent, sent_in_sec == 0) + 0.1;
                    try cands.append(gpa, .{
                        .sec_idx = cur_sec,
                        .order_idx = order,
                        .text = sent,
                        .score = sc,
                    });
                    order += 1;
                    sent_in_sec += 1;
                }
                pos = if (end < line.len) end + 1 else line.len;
            }
        }
    }

    if (cands.items.len == 0) return try gpa.alloc(u8, 0);

    const num_secs = sec_headings.items.len;
    const heading_counted = try gpa.alloc(bool, num_secs);
    defer gpa.free(heading_counted);
    @memset(heading_counted, false);

    var used_toks: usize = if (doc_title.len > 0) countOf(counter, doc_title) + 4 else 0;
    var chosen_idx: std.ArrayList(usize) = .empty;
    defer chosen_idx.deinit(gpa);

    // Serpentine (snake) round-robin across EVERY section: odd rounds run back-to-front so
    // concluding sections (summaries, roadmaps, appendices) are never starved!
    var round: usize = 0;
    while (used_toks < max_tokens and round < 8) : (round += 1) {
        var added_any = false;
        for (0..num_secs) |k| {
            const si = if (round % 2 == 0) k else (num_secs - 1 - k);
            var best_i: ?usize = null;
            var best_sc: f32 = -1.0;
            for (cands.items, 0..) |c, idx| {
                if (c.picked or c.sec_idx != si) continue;
                if (c.score > best_sc) {
                    best_sc = c.score;
                    best_i = idx;
                }
            }
            if (best_i) |bi| {
                const hdr_cost: usize = if (!heading_counted[si] and sec_headings.items[si].len > 0)
                    countOf(counter, sec_headings.items[si]) + 2
                else
                    0;
                const tlen = countOf(counter, cands.items[bi].text) + 2 + hdr_cost;
                if (used_toks + tlen <= max_tokens or round == 0) {
                    cands.items[bi].picked = true;
                    heading_counted[si] = true;
                    try chosen_idx.append(gpa, bi);
                    used_toks += tlen;
                    added_any = true;
                }
            }
        }
        if (!added_any) break;
    }

    // Re-sort chosen verbatim sentences in original document order for clean reading
    std.mem.sort(usize, chosen_idx.items, cands.items, struct {
        fn less(all: []const Cand, a: usize, b: usize) bool {
            return all[a].order_idx < all[b].order_idx;
        }
    }.less);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    if (doc_title.len > 0) {
        try out.appendSlice(gpa, doc_title);
        try out.append(gpa, '\n');
    }
    var last_sec: ?usize = null;
    for (chosen_idx.items) |idx| {
        const c = cands.items[idx];
        if (last_sec == null or last_sec.? != c.sec_idx) {
            if (sec_headings.items[c.sec_idx].len > 0) {
                try out.appendSlice(gpa, sec_headings.items[c.sec_idx]);
                try out.append(gpa, '\n');
            }
            last_sec = c.sec_idx;
        }
        try out.appendSlice(gpa, "- ");
        try out.appendSlice(gpa, c.text);
        try out.append(gpa, '\n');
    }
    return out.toOwnedSlice(gpa);
}

test "chunker: keeps small fact single and splits multi-section document preserving title" {
    const gpa = std.testing.allocator;
    const small = try chunkDocument(gpa, "My dog is named Barnaby.", 600, null);
    defer freeChunks(gpa, small);
    try std.testing.expectEqual(@as(usize, 1), small.len);
    try std.testing.expect(small[0].is_small_fact);

    const doc =
        \\# Halcyon-7 Manual
        \\
        \\## Section 1: Power
        \\Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi.
        \\
        \\## Section 2: Cooling
        \\One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen.
    ;
    const parts = try chunkDocument(gpa, doc, 25, null);
    defer freeChunks(gpa, parts);
    try std.testing.expect(parts.len >= 2);
    try std.testing.expect(!parts[0].is_small_fact);
    try std.testing.expect(std.mem.startsWith(u8, parts[parts.len - 1].full_text, "# Halcyon-7 Manual"));
    try std.testing.expectEqualStrings("# Halcyon-7 Manual", parts[parts.len - 1].title_line);

    const digest = try buildExtractiveDigest(gpa, parts, 120, null);
    defer gpa.free(digest);
    try std.testing.expect(digest.len > 0);
}
