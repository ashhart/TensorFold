//! Parity, parallel, cache-growth, overrideSplit and OOM leak tests for fast_bpe and fast_bpe_parallel.
const std = @import("std");
const tokenizer = @import("tokenizer.zig");
const fast_bpe = @import("fast_bpe.zig");
const fast_bpe_parallel = @import("fast_bpe_parallel.zig");
const fast_split = @import("fast_split.zig");
const steps = @import("steps.zig");
const parse = tokenizer.parse;

fn buildO200kTokenizerJson(a: std.mem.Allocator) ![]u8 {
    var vocab: std.ArrayList(u8) = .empty;
    defer vocab.deinit(a);
    var tmp: [128]u8 = undefined;
    const bytes = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";
    var id: u32 = 0;
    for (bytes, 0..) |b, idx| {
        if (idx > 0) try vocab.appendSlice(a, ",");
        const entry = if (b == ' ')
            try std.fmt.bufPrint(&tmp, "\"\\u0120\":{d}", .{id})
        else if (b == '"' or b == '\\')
            try std.fmt.bufPrint(&tmp, "\"\\{c}\":{d}", .{ b, id })
        else
            try std.fmt.bufPrint(&tmp, "\"{c}\":{d}", .{ b, id });
        try vocab.appendSlice(a, entry);
        id += 1;
    }
    for (0..256) |b| {
        if (b >= 0x20 and b <= 0x7E) continue;
        var ubuf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(steps.byte_chars[b], &ubuf) catch unreachable;
        try vocab.appendSlice(a, try std.fmt.bufPrint(&tmp, ",\"{s}\":{d}", .{ ubuf[0..n], id }));
        id += 1;
    }
    const words = [_][]const u8{
        "th",  "the",
        "Ġthe",
        "in",  "ing",
        "er",  "re",
        "he",  "ll",
        "llo", "hello",
        "Ġhello",
        "Ġa",
        "ab",  "abc",
        "Ġabc",
        "12",  "123",
        "fn",
        "Ġfn",
        "pu",  "pub",
        "Ġpub",
        "ur",  "urn",
        "ret", "return",
        "Ġreturn",
    };
    for (words) |w| {
        try vocab.appendSlice(a, try std.fmt.bufPrint(&tmp, ",\"{s}\":{d}", .{ w, id }));
        id += 1;
    }
    const merges =
        \\["t","h"],["th","e"],["Ġ","the"],["i","n"],["in","g"],["e","r"],["r","e"],["h","e"],
        \\["l","l"],["ll","o"],["he","llo"],["Ġ","hello"],["Ġ","a"],["a","b"],["ab","c"],["Ġ","abc"],
        \\["1","2"],["12","3"],["f","n"],["Ġ","fn"],["p","u"],["pu","b"],["Ġ","pub"],
        \\["u","r"],["ur","n"],["re","t"],["ret","urn"],["Ġ","return"]
    ;
    const pat_escaped = std.json.Stringify.valueAlloc(a, fast_split.o200k_pattern, .{}) catch unreachable;
    defer a.free(pat_escaped);
    return std.fmt.allocPrint(a,
        \\{{"added_tokens":[
        \\  {{"id":{d},"content":"<|sys|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}},
        \\  {{"id":{d},"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}}
        \\],
        \\"pre_tokenizer":{{"type":"Sequence","pretokenizers":[
        \\  {{"type":"Split","pattern":{{"Regex":{s}}},"behavior":"Isolated","invert":false}},
        \\  {{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}}
        \\]}},
        \\"decoder":{{"type":"ByteLevel"}},
        \\"model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":true,
        \\"vocab":{{{s}}},
        \\"merges":[{s}]}}}}
    , .{ id, id + 1, pat_escaped, vocab.items, merges });
}

const ConcurrencyDetectingAllocator = struct {
    in_use: std.atomic.Value(bool) = .init(false),
    collisions: std.atomic.Value(usize) = .init(0),
    calls: std.atomic.Value(usize) = .init(0),
    inner: std.mem.Allocator,

    fn enter(self: *ConcurrencyDetectingAllocator) void {
        _ = self.calls.fetchAdd(1, .monotonic);
        if (self.in_use.swap(true, .acq_rel)) _ = self.collisions.fetchAdd(1, .acq_rel);
        std.Thread.yield() catch {};
    }

    fn leave(self: *ConcurrencyDetectingAllocator) void {
        self.in_use.store(false, .release);
    }

    fn allocator(self: *ConcurrencyDetectingAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, align_val: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *ConcurrencyDetectingAllocator = @ptrCast(@alignCast(ctx));
        self.enter();
        defer self.leave();
        return self.inner.rawAlloc(len, align_val, ret_addr);
    }

    fn resize(ctx: *anyopaque, buf: []u8, align_val: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *ConcurrencyDetectingAllocator = @ptrCast(@alignCast(ctx));
        self.enter();
        defer self.leave();
        return self.inner.rawResize(buf, align_val, new_len, ret_addr);
    }

    fn remap(ctx: *anyopaque, buf: []u8, align_val: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *ConcurrencyDetectingAllocator = @ptrCast(@alignCast(ctx));
        self.enter();
        defer self.leave();
        return self.inner.rawRemap(buf, align_val, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, buf: []u8, align_val: std.mem.Alignment, ret_addr: usize) void {
        const self: *ConcurrencyDetectingAllocator = @ptrCast(@alignCast(ctx));
        self.enter();
        defer self.leave();
        self.inner.rawFree(buf, align_val, ret_addr);
    }
};

fn splitSegmentsForTest(a: std.mem.Allocator, text: []const u8, sys_id: u32, end_id: u32) ![]fast_bpe_parallel.Segment {
    var segs: std.ArrayList(fast_bpe_parallel.Segment) = .empty;
    var i: usize = 0;
    var start: usize = 0;
    while (i < text.len) {
        if (std.mem.startsWith(u8, text[i..], "<|sys|>")) {
            if (i > start) try segs.append(a, .{ .start = start, .end = i, .id = null });
            try segs.append(a, .{ .start = i, .end = i + 7, .id = sys_id });
            i += 7;
            start = i;
        } else if (std.mem.startsWith(u8, text[i..], "<|end|>")) {
            if (i > start) try segs.append(a, .{ .start = start, .end = i, .id = null });
            try segs.append(a, .{ .start = i, .end = i + 7, .id = end_id });
            i += 7;
            start = i;
        } else {
            i += 1;
        }
    }
    if (start < text.len) try segs.append(a, .{ .start = start, .end = text.len, .id = null });
    return segs.toOwnedSlice(a);
}

test "fast_bpe matches slow generic path across encode, encodePlain, encodeMasked and cache-growth" {
    const a = std.testing.allocator;
    const json = try buildO200kTokenizerJson(a);
    defer a.free(json);
    var fast_tok = try parse(a, json);
    defer fast_tok.deinit();
    try std.testing.expect(fast_tok.fast != null);

    var slow_tok = try parse(a, json);
    defer slow_tok.deinit();
    slow_tok.fast = null;

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    var line_buf: [128]u8 = undefined;
    for (0..120) |i| {
        const line = try std.fmt.bufPrint(&line_buf, "<|sys|>hello world pub fn return const 123456 café 🚀 #{d}<|end|>\n", .{i});
        try buf.appendSlice(a, line);
    }
    try buf.appendSlice(a, "<|sys|>partial\xffutf8<|end|>");

    const got = try fast_tok.encode(a, buf.items);
    defer a.free(got);
    const want = try slow_tok.encode(a, buf.items);
    defer a.free(want);
    try std.testing.expectEqualSlices(u32, want, got);

    const got_plain = try fast_tok.encodePlain(a, buf.items);
    defer a.free(got_plain);
    const want_plain = try slow_tok.encodePlain(a, buf.items);
    defer a.free(want_plain);
    try std.testing.expectEqualSlices(u32, want_plain, got_plain);

    const spans = [_]tokenizer.Span{ .{ 0, 7 }, .{ 15, 25 } };
    const got_masked = try fast_tok.encodeMasked(a, buf.items[0..80], &spans);
    defer a.free(got_masked);
    const want_masked = try slow_tok.encodeMasked(a, buf.items[0..80], &spans);
    defer a.free(want_masked);
    try std.testing.expectEqualSlices(u32, want_masked, got_masked);
}

test "fast_bpe_parallel chunk, span and multi-segment paths match serial bit-for-bit" {
    const a = std.testing.allocator;
    const json = try buildO200kTokenizerJson(a);
    defer a.free(json);
    var tok = try parse(a, json);
    defer tok.deinit();
    var slow_tok = try parse(a, json);
    defer slow_tok.deinit();
    slow_tok.fast = null;

    const cfg = fast_bpe_parallel.Config{ .parallel_min_bytes = 64, .parallel_max_threads = 4 };
    const wide_cfg = fast_bpe_parallel.Config{ .parallel_min_bytes = 32, .parallel_max_threads = 64 };

    var lines: std.ArrayList(u8) = .empty;
    defer lines.deinit(a);
    for (0..80) |_| try lines.appendSlice(a, "hello world pub fn return const 123456\n");
    var got_chunks: std.ArrayList(u32) = .empty;
    defer got_chunks.deinit(a);
    try fast_bpe_parallel.encodeWithConfig(&tok.fast.?, &tok.model, &tok.vocab, a, a, lines.items, &got_chunks, cfg);
    const want_chunks = try slow_tok.encode(a, lines.items);
    defer a.free(want_chunks);
    try std.testing.expectEqualSlices(u32, want_chunks, got_chunks.items);

    var got_wide: std.ArrayList(u32) = .empty;
    defer got_wide.deinit(a);
    try fast_bpe_parallel.encodeWithConfig(&tok.fast.?, &tok.model, &tok.vocab, a, a, lines.items, &got_wide, wide_cfg);
    try std.testing.expectEqualSlices(u32, want_chunks, got_wide.items);

    var single: std.ArrayList(u8) = .empty;
    defer single.deinit(a);
    for (0..400) |_| try single.appendSlice(a, "hello world abc 123 ");
    var got_spans: std.ArrayList(u32) = .empty;
    defer got_spans.deinit(a);
    try fast_bpe_parallel.encodeWithConfig(&tok.fast.?, &tok.model, &tok.vocab, a, a, single.items, &got_spans, cfg);
    const want_spans = try slow_tok.encode(a, single.items);
    defer a.free(want_spans);
    try std.testing.expectEqualSlices(u32, want_spans, got_spans.items);

    var chat: std.ArrayList(u8) = .empty;
    defer chat.deinit(a);
    for (0..40) |_| try chat.appendSlice(a, "<|sys|>hello world const return<|end|>");
    const sys_id = tok.specialTokenId("<|sys|>").?;
    const end_id = tok.specialTokenId("<|end|>").?;
    const segs = try splitSegmentsForTest(a, chat.items, sys_id, end_id);
    defer a.free(segs);
    var got_chat: std.ArrayList(u32) = .empty;
    defer got_chat.deinit(a);
    try fast_bpe_parallel.encodeSegmentsWithConfig(&tok.fast.?, &tok.model, &tok.vocab, a, a, chat.items, segs, &got_chat, cfg);
    const want_chat = try slow_tok.encode(a, chat.items);
    defer a.free(want_chat);
    try std.testing.expectEqualSlices(u32, want_chat, got_chat.items);
}

test "overrideSplit updates fast path availability cleanly" {
    const a = std.testing.allocator;
    const json = try buildO200kTokenizerJson(a);
    defer a.free(json);
    var tok = try parse(a, json);
    defer tok.deinit();
    try std.testing.expect(tok.fast != null);

    try tok.overrideSplit(" ?\\S+|\\s+");
    try std.testing.expect(tok.fast == null);
    const custom_ids = try tok.encode(a, "hello world");
    defer a.free(custom_ids);
    try std.testing.expect(custom_ids.len > 0);

    try tok.overrideSplit(fast_split.o200k_pattern);
    try std.testing.expect(tok.fast != null);
}

fn oomTrial(a: std.mem.Allocator, tok: *const tokenizer.Tokenizer, input: []const u8, min_bytes: usize) !void {
    const cfg = fast_bpe_parallel.Config{ .parallel_min_bytes = min_bytes, .parallel_max_threads = 4 };
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    try fast_bpe_parallel.encodeWithConfig(&tok.fast.?, &tok.model, &tok.vocab, a, a, input, &out, cfg);

    const mid = input.len / 2;
    const segs = [_]fast_bpe_parallel.Segment{
        .{ .start = 0, .end = mid, .id = null },
        .{ .start = mid, .end = mid, .id = 0 },
        .{ .start = mid, .end = input.len, .id = null },
    };
    var out_segs: std.ArrayList(u32) = .empty;
    defer out_segs.deinit(a);
    try fast_bpe_parallel.encodeSegmentsWithConfig(&tok.fast.?, &tok.model, &tok.vocab, a, a, input, &segs, &out_segs, cfg);
}

test "Tokenizer.encode serializes worker access on a non-thread-safe arena >= 128 KiB" {
    const a = std.testing.allocator;
    const json = try buildO200kTokenizerJson(a);
    defer a.free(json);
    var slow_tok = try parse(a, json);
    defer slow_tok.deinit();
    slow_tok.fast = null;

    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var detector: ConcurrencyDetectingAllocator = .{ .inner = arena.allocator() };
    var tok = try parse(detector.allocator(), json);
    defer tok.deinit();
    detector.calls.store(0, .monotonic);

    var lines: std.ArrayList(u8) = .empty;
    defer lines.deinit(a);
    while (lines.items.len < 160 * 1024) try lines.appendSlice(a, "hello world pub fn return const 123456\n");
    const got_lines = try tok.encode(a, lines.items);
    defer a.free(got_lines);
    const want_lines = try slow_tok.encode(a, lines.items);
    defer a.free(want_lines);
    try std.testing.expectEqualSlices(u32, want_lines, got_lines);

    var single: std.ArrayList(u8) = .empty;
    defer single.deinit(a);
    while (single.items.len < 160 * 1024) try single.appendSlice(a, "hello world abc 123 ");
    const got_single = try tok.encodePlain(a, single.items);
    defer a.free(got_single);
    const want_single = try slow_tok.encodePlain(a, single.items);
    defer a.free(want_single);
    try std.testing.expectEqualSlices(u32, want_single, got_single);

    var chat: std.ArrayList(u8) = .empty;
    defer chat.deinit(a);
    while (chat.items.len < 160 * 1024) try chat.appendSlice(a, "<|sys|>hello world const return<|end|>");
    const got_chat = try tok.encode(a, chat.items);
    defer a.free(got_chat);
    const want_chat = try slow_tok.encode(a, chat.items);
    defer a.free(want_chat);
    try std.testing.expectEqualSlices(u32, want_chat, got_chat);

    try std.testing.expect(detector.calls.load(.acquire) > 0);
    try std.testing.expectEqual(@as(usize, 0), detector.collisions.load(.acquire));
}

test "fast_bpe_parallel leaks zero memory on allocation failures in serial and parallel modes" {
    const a = std.testing.allocator;
    const json = try buildO200kTokenizerJson(a);
    defer a.free(json);
    var tok = try parse(a, json);
    defer tok.deinit();
    const sample = "hello world pub fn return const 123\nhello world pub fn return const 456\n";
    try std.testing.checkAllAllocationFailures(a, oomTrial, .{ &tok, sample, @as(usize, 64 * 1024) });
    try std.testing.checkAllAllocationFailures(a, oomTrial, .{ &tok, sample, @as(usize, 16) });
}
