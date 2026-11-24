//! Arbitrarily nested percent encoding matched against original spans, for containment and streaming redaction.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Match = struct { secret_len: usize = 0, prefix_len: usize = 0, consumed: usize = 0, hold: bool = false };
pub const Span = struct { start: usize, end: usize };
/// A complete key of any length, or this many decoded prefix bytes, fails containment.
pub const min_prefix = 4;
/// Fixed non-derived outward text for a withheld candidate; not the redaction marker, which may itself be configured.
pub const withheld = "[withheld]";

/// A reusable matcher; memo entries retain every endpoint so nested escapes and literal bytes stay alternatives.
pub const Matcher = struct {
    a: Allocator,
    text: []const u8,
    memo: std.AutoHashMap(Key, Result),
    states: std.ArrayList(usize) = .empty,
    next: std.ArrayList(usize) = .empty,

    const Key = struct { at: usize, byte: u8 };
    const Result = struct {
        first: ?usize = null,
        rest: []const usize = &.{},
        partial: bool = false,

        fn len(r: Result) usize {
            return @as(usize, @intFromBool(r.first != null)) + r.rest.len;
        }

        fn at(r: Result, index: usize) usize {
            return if (index == 0) r.first.? else r.rest[index - 1];
        }
    };
    const Ends = struct {
        first: ?usize = null,
        rest: std.ArrayList(usize) = .empty,

        fn len(e: Ends) usize {
            return @as(usize, @intFromBool(e.first != null)) + e.rest.items.len;
        }

        fn at(e: Ends, index: usize) usize {
            return if (index == 0) e.first.? else e.rest.items[index - 1];
        }
    };
    const Frame = struct {
        key: Key,
        ends: Ends = .{},
        partial: bool,
        state: State,

        const Percent = struct { scan: usize = 0, two_index: usize = 0 };
        const Encoded = struct { pair: usize = 0, percent_index: usize = 0, high_index: usize = 0 };
        const State = union(enum) { percent: Percent, encoded: Encoded, finished: void };
    };

    pub fn init(a: Allocator, text: []const u8) Matcher {
        return .{ .a = a, .text = text, .memo = std.AutoHashMap(Key, Result).init(a) };
    }

    pub fn deinit(m: *Matcher) void {
        var values = m.memo.valueIterator();
        while (values.next()) |value| if (value.rest.len > 0) m.a.free(value.rest);
        m.memo.deinit();
        m.states.deinit(m.a);
        m.next.deinit(m.a);
    }

    /// The longest full key or decoded-prefix match; an unfinished later encoding holds the suffix.
    pub fn secretAt(m: *Matcher, start: usize, secrets: anytype) Allocator.Error!Match {
        var found: Match = .{};
        var hold = false;
        for (secrets) |secret| {
            if (secret.len == 0) continue;
            const matched = try m.matchSecret(start, secret);
            hold = hold or matched.hold;
            // Longest original span, then decoded prefix and key, regardless of which key completes first.
            if (matched.consumed > found.consumed or (matched.consumed == found.consumed and
                (matched.prefix_len > found.prefix_len or (matched.prefix_len == found.prefix_len and matched.secret_len > found.secret_len)))) found = matched;
        }
        found.hold = found.hold or hold;
        return found;
    }

    /// The leftmost-longest full occurrence of one secret at or after `start`.
    pub fn spanFrom(m: *Matcher, start: usize, secret: []const u8) Allocator.Error!?Span {
        if (secret.len == 0) return null;
        var at = start;
        while (at < m.text.len) : (at += 1) {
            const matched = try m.matchSecret(at, secret);
            if (matched.consumed > 0) return .{ .start = at, .end = at + matched.consumed };
        }
        return null;
    }

    fn matchSecret(m: *Matcher, start: usize, secret: []const u8) Allocator.Error!Match {
        if (start > m.text.len) return .{};
        m.states.clearRetainingCapacity();
        m.next.clearRetainingCapacity();
        try m.states.append(m.a, start);
        var states = &m.states;
        var next = &m.next;
        var hold = false;
        for (secret, 0..) |wanted, matched_len| {
            next.clearRetainingCapacity();
            var partial_byte = false;
            for (states.items) |at| {
                if (at == m.text.len) {
                    hold = true;
                    continue;
                }
                if (m.text[at] != '%') {
                    if (m.text[at] == wanted) try appendUnique(m.a, next, at + 1);
                    continue;
                }
                const decoded = try m.decode(at, wanted);
                partial_byte = partial_byte or decoded.partial;
                hold = hold or decoded.partial;
                for (0..decoded.len()) |index| try appendUnique(m.a, next, decoded.at(index));
            }
            if (next.items.len == 0) {
                // Unfinished decoding and a hex high nibble are partial; a non-hex continuation is not.
                return .{ .secret_len = secret.len, .prefix_len = matched_len + @intFromBool(partial_byte), .hold = hold };
            }
            const swap = states;
            states = next;
            next = swap;
        }
        var consumed: usize = 0;
        for (states.items) |end| consumed = @max(consumed, end - start);
        return .{ .secret_len = secret.len, .prefix_len = secret.len, .consumed = consumed, .hold = hold };
    }

    /// One byte's literal or `%XX` representations; nested dependencies use heap frames, not call recursion.
    fn decode(m: *Matcher, at: usize, byte: u8) Allocator.Error!Result {
        const wanted: Key = .{ .at = at, .byte = byte };
        if (m.memo.get(wanted)) |cached| return cached;
        var work: std.ArrayList(Frame) = .empty;
        defer {
            for (work.items) |*frame| frame.ends.rest.deinit(m.a);
            work.deinit(m.a);
        }
        try work.append(m.a, newFrame(m.text, wanted));
        frames: while (work.items.len > 0) {
            const top = &work.items[work.items.len - 1];
            switch (top.state) {
                .percent => |*state| {
                    // S('%') = literal '%' | S('%') S('2') S('5'); each dependency strictly advances.
                    while (state.scan < top.ends.len()) {
                        const two_key: Key = .{ .at = top.ends.at(state.scan), .byte = '2' };
                        const two = m.memo.get(two_key) orelse {
                            try work.append(m.a, newFrame(m.text, two_key));
                            continue :frames;
                        };
                        top.partial = top.partial or two.partial;
                        if (state.two_index == two.len()) {
                            state.scan += 1;
                            state.two_index = 0;
                            continue;
                        }
                        const five_key: Key = .{ .at = two.at(state.two_index), .byte = '5' };
                        const five = m.memo.get(five_key) orelse {
                            try work.append(m.a, newFrame(m.text, five_key));
                            continue :frames;
                        };
                        top.partial = top.partial or five.partial;
                        for (0..five.len()) |index| try add(m.a, &top.ends, five.at(index));
                        state.two_index += 1;
                    }
                    top.state = .finished;
                },
                .encoded => |*state| {
                    const percent_key: Key = .{ .at = top.key.at, .byte = '%' };
                    const percent = m.memo.get(percent_key) orelse {
                        try work.append(m.a, newFrame(m.text, percent_key));
                        continue :frames;
                    };
                    top.partial = top.partial or percent.partial;
                    const digits = "0123456789ABCDEF";
                    const high = digits[top.key.byte >> 4];
                    const low = digits[top.key.byte & 15];
                    while (state.pair < 4) {
                        if (state.percent_index == percent.len()) {
                            state.pair += 1;
                            state.percent_index = 0;
                            state.high_index = 0;
                            continue;
                        }
                        const hi = if (state.pair & 1 == 0) high else std.ascii.toLower(high);
                        const lo = if (state.pair & 2 == 0) low else std.ascii.toLower(low);
                        const high_key: Key = .{ .at = percent.at(state.percent_index), .byte = hi };
                        const high_result = m.memo.get(high_key) orelse {
                            try work.append(m.a, newFrame(m.text, high_key));
                            continue :frames;
                        };
                        top.partial = top.partial or high_result.partial;
                        if (state.high_index == high_result.len()) {
                            state.percent_index += 1;
                            state.high_index = 0;
                            continue;
                        }
                        const low_key: Key = .{ .at = high_result.at(state.high_index), .byte = lo };
                        const low_result = m.memo.get(low_key) orelse {
                            try work.append(m.a, newFrame(m.text, low_key));
                            continue :frames;
                        };
                        top.partial = top.partial or low_result.partial;
                        for (0..low_result.len()) |index| try add(m.a, &top.ends, low_result.at(index));
                        state.high_index += 1;
                    }
                    top.state = .finished;
                },
                .finished => {
                    const rest = if (top.ends.rest.items.len == 0) &.{} else try top.ends.rest.toOwnedSlice(m.a);
                    const result: Result = .{ .first = top.ends.first, .rest = rest, .partial = top.partial };
                    errdefer if (rest.len > 0) m.a.free(rest);
                    try m.memo.put(top.key, result);
                    _ = work.pop();
                },
            }
        }
        return m.memo.get(wanted).?;
    }

    fn newFrame(text: []const u8, key: Key) Frame {
        var frame: Frame = .{ .key = key, .partial = key.at == text.len, .state = .finished };
        if (key.at < text.len and text[key.at] == key.byte) frame.ends.first = key.at + 1;
        if (key.byte == '%') {
            frame.state = .{ .percent = .{} };
        } else if (key.at < text.len and text[key.at] == '%') {
            frame.state = .{ .encoded = .{} };
        }
        return frame;
    }
};

fn add(a: Allocator, ends: *Matcher.Ends, end: usize) Allocator.Error!void {
    if (ends.first == null) {
        ends.first = end;
        return;
    }
    if (ends.first.? == end) return;
    for (ends.rest.items) |present| if (present == end) return;
    try ends.rest.append(a, end);
}

fn appendUnique(a: Allocator, ends: *std.ArrayList(usize), end: usize) Allocator.Error!void {
    for (ends.items) |present| if (present == end) return;
    try ends.append(a, end);
}

/// A possible first matcher byte; percent can begin any literal or nested encoding.
fn prefixStart(text: []const u8, secrets: []const []const u8) bool {
    if (text.len == 0) return false;
    if (text[0] == '%') return secrets.len > 0;
    for (secrets) |secret| if (secret.len > 0 and secret[0] == text[0]) return true;
    return false;
}

/// Original bytes forming a decoded credential prefix of at least ``min_prefix`` bytes before a disambiguating suffix.
pub fn sensitivePrefix(a: Allocator, text: []const u8, secrets: []const []const u8) Allocator.Error!usize {
    if (!prefixStart(text, secrets)) return 0;
    var sensitive: usize = 0;
    // Hold and prefix length belong to one secret; combining them across secrets can consume a safe suffix.
    for (secrets) |secret| {
        if (secret.len == 0) continue;
        var own: usize = 0;
        for (1..text.len + 1) |end| {
            var matcher = Matcher.init(a, text[0..end]);
            defer matcher.deinit();
            const found = try matcher.matchSecret(0, secret);
            if (found.hold) {
                if (found.prefix_len >= min_prefix) own = @max(own, end);
                continue;
            }
            if (found.prefix_len >= min_prefix) {
                own = end - 1;
                break;
            }
            // A disproved escape below the real decoded prefix minimum is safe.
            own = 0;
            break;
        }
        sensitive = @max(sensitive, own);
    }
    return sensitive;
}

/// Whether any complete credential or decoded prefix of at least ``min_prefix`` bytes occurs at any encoding depth.
pub fn contains(a: Allocator, text: []const u8, secrets: []const []const u8) Allocator.Error!bool {
    var matcher = Matcher.init(a, text);
    defer matcher.deinit();
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if ((try matcher.secretAt(i, secrets)).consumed > 0) return true;
        if (try sensitivePrefix(a, text[i..], secrets) > 0) return true;
    }
    return false;
}

test "credential matcher: 1024 nested encodings use bounded call depth" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var encoded: std.ArrayList(u8) = .empty;
    try encoded.appendSlice(a, "a/b");
    for (0..1024) |_| {
        var next: std.ArrayList(u8) = .empty;
        for (encoded.items) |byte| {
            if (byte == '%') try next.appendSlice(a, "%25") else try next.append(a, byte);
        }
        encoded = next;
    }
    var matcher = Matcher.init(std.testing.allocator, encoded.items);
    defer matcher.deinit();
    const span = (try matcher.spanFrom(0, "a/b")).?;
    try std.testing.expectEqual(@as(usize, 0), span.start);
    try std.testing.expectEqual(encoded.items.len, span.end);
}
