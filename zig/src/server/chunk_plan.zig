//! Where a prompt's prefill chunks start (Python's PrefillPlan and message_markers): prompts that agree up to a start cut alike up to it.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const prompt = @import("prompt.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// Chunk starts: 0, then the first resume point `min_chunk` or more tokens on, else `step` tokens on.
pub const Plan = struct {
    step: u32 = 0, // 0: the engine cuts prompts itself
    openers: []const u32 = &.{}, // the special tokens that open any message
    assistant: []const u32 = &.{}, // the tokens that open an assistant message
    min_chunk: u32 = 256,

    /// Resume points: every assistant message's start, and the second message's.
    fn points(p: Plan, a: Allocator, ids: []const u32) ![]u32 {
        var found: std.ArrayList(u32) = .empty;
        var seen: usize = 0;
        if (p.openers.len > 0) for (ids, 0..) |t, i| {
            if (std.mem.indexOfScalar(u32, p.openers, t) == null) continue;
            seen += 1;
            if (seen == 2) {
                try found.append(a, @intCast(i));
                break;
            }
        };
        const k = p.assistant.len;
        if (k > 0 and ids.len >= k) for (0..ids.len - k + 1) |i| {
            if (std.mem.eql(u32, ids[i..][0..k], p.assistant)) try found.append(a, @intCast(i));
        };
        std.mem.sort(u32, found.items, {}, std.sort.asc(u32));
        var out: std.ArrayList(u32) = .empty;
        for (found.items) |q| if (q > 0 and (out.items.len == 0 or out.items[out.items.len - 1] != q)) try out.append(a, q);
        return out.items;
    }

    /// The starts after 0 of `ids`' chunks; empty with no step.
    pub fn starts(p: Plan, a: Allocator, ids: []const u32) ![]const u32 {
        if (p.step == 0) return &.{};
        const found = if ((p.openers.len > 0 or p.assistant.len > 0) and ids.len > 1) try p.points(a, ids) else &[_]u32{};
        var out: std.ArrayList(u32) = .empty;
        var last: usize = 0;
        var i: usize = 0;
        while (true) {
            while (i < found.len and found[i] < last + p.min_chunk) i += 1; // too close to the last start: merged into its chunk
            var next = last + p.step;
            if (i < found.len and found[i] < next) next = found[i];
            if (next >= ids.len) return out.items;
            try out.append(a, @intCast(next));
            last = next;
        }
    }
};

/// `starts` with `cut` too: a state kept just before a conversation's own text reads only shared tokens.
pub fn withCut(a: Allocator, starts: []const u32, cut: u32, len: usize, min_chunk: u32) ![]const u32 {
    if (cut == 0 or cut >= len) return starts;
    var out: std.ArrayList(u32) = .empty;
    for (starts) |s| if (@max(s, cut) - @min(s, cut) >= min_chunk) try out.append(a, s); // starts too near the cut go
    try out.append(a, cut);
    std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    return out.items;
}

const Role = enum { user, assistant };

/// ``message_markers``: the openers and assistant header the chat template renders, from probe conversations.
pub fn markers(srv: *Server, a: Allocator, step: u32) !Plan {
    var pieces: [2]std.ArrayList([]const u32) = .{ .empty, .empty };
    var parted: std.ArrayList(struct { before: []const u32, after: []const u32, role: Role }) = .empty;
    const words = [2][4][]const u8{ .{ "Alpha", "Beta", "Gamma", "Delta" }, .{ "one two", "three four", "five six", "seven eight" } };
    for (words) |set| for ([_]bool{ false, true }) |thinking| {
        var talk: [4]Value = undefined;
        for (&talk, set, 0..) |*m, text, i| {
            const o = try json.newObject(a);
            try o.put(a, "role", .{ .string = if (i % 2 == 0) "user" else "assistant" });
            try o.put(a, "content", .{ .string = text });
            m.* = .{ .object = o };
        }
        var r: [4][]const u32 = undefined;
        for (&r, 1..) |*out, k| out.* = render(srv, a, talk[0..k], thinking, false) orelse return .{ .step = step };
        const g1 = render(srv, a, talk[0..1], thinking, true) orelse return .{ .step = step };
        const g3 = render(srv, a, talk[0..3], thinking, true) orelse return .{ .step = step };
        const pairs = [_]struct { []const u32, []const u32, Role }{ .{ r[0], r[1], .assistant }, .{ r[1], r[2], .user }, .{ r[2], r[3], .assistant }, .{ r[0], g1, .assistant }, .{ r[2], g3, .assistant } };
        for (pairs) |pr| {
            if (pr[1].len > pr[0].len and std.mem.eql(u32, pr[1][0..pr[0].len], pr[0])) try pieces[@intFromEnum(pr[2])].append(a, pr[1][pr[0].len..]) else try parted.append(a, .{ .before = pr[0], .after = pr[1], .role = pr[2] });
        }
    };
    // templates can render the last reply differently; new messages start at the next opener past the difference
    var openers = try openersOf(srv, a, &pieces);
    for (parted.items) |pt| {
        var split = @min(pt.before.len, pt.after.len);
        for (pt.before[0..split], pt.after[0..split], 0..) |x, y, i| if (x != y) {
            split = i;
            break;
        };
        var at: ?usize = null;
        var count: usize = 0;
        for (pt.after[split..], split..) |t, i| if (std.mem.indexOfScalar(u32, openers, t) != null) {
            count += 1;
            at = i;
        };
        if (count == 1) try pieces[@intFromEnum(pt.role)].append(a, pt.after[at.?..]);
    }
    if (pieces[0].items.len == 0 or pieces[1].items.len == 0) return .{ .step = step };
    openers = try openersOf(srv, a, &pieces);
    const header = common(pieces[1].items);
    const user = common(pieces[0].items);
    const cut = for (header, 0..) |t, i| {
        if (i >= user.len or user[i] != t) break i;
    } else null;
    const special = header.len > 0 and srv.text.isSpecial(header[0]);
    return .{ .step = step, .openers = openers, .assistant = if (cut != null and special) header[0 .. cut.? + 1] else &.{} };
}

fn render(srv: *Server, a: Allocator, talk: []Value, thinking: bool, generation: bool) ?[]const u32 {
    var cx: errors.Cx = .{ .a = a };
    return prompt.renderIds(srv, &cx, .{ .array = talk }, &.{}, thinking, null, generation) catch null;
}

/// The special tokens every piece of a role starts with.
fn openersOf(srv: *Server, a: Allocator, pieces: *const [2]std.ArrayList([]const u32)) ![]const u32 {
    var out: std.ArrayList(u32) = .empty;
    for (pieces) |group| {
        if (group.items.len == 0) continue;
        const first = group.items[0][0];
        if (!srv.text.isSpecial(first)) continue;
        for (group.items) |p| {
            if (p[0] != first) break;
        } else if (std.mem.indexOfScalar(u32, out.items, first) == null) try out.append(a, first);
    }
    std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    return out.items;
}

/// The longest prefix every piece shares.
fn common(pieces: []const []const u32) []const u32 {
    const first = pieces[0];
    var n = first.len;
    for (pieces) |p| n = @min(n, p.len);
    for (0..n) |i| for (pieces) |p| if (p[i] != first[i]) return first[0..i];
    return first[0..n];
}

test "a harness cut joins the starts, nearer ones go" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualSlices(u32, &.{ 4096, 4648 }, try withCut(a, &.{ 4096, 4650 }, 4648, 4680, 256));
    try std.testing.expectEqualSlices(u32, &.{ 4096, 4650 }, try withCut(a, &.{ 4096, 4650 }, 0, 4680, 256));
    try std.testing.expectEqualSlices(u32, &.{ 4096, 4650 }, try withCut(a, &.{ 4096, 4650 }, 4680, 4680, 256));
    try std.testing.expectEqualSlices(u32, &.{ 600, 4096 }, try withCut(a, &.{4096}, 600, 4680, 256));
}

test "chunk starts match PrefillPlan.chunks" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Python's PrefillPlan(8, (10,), min_chunk, (10, 7)).chunks(ids).starts[1:] on the same ids
    const ids = [_]u32{ 10, 1, 2, 10, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 10, 7, 9 };
    try std.testing.expectEqualSlices(u32, &.{ 3, 11, 18 }, try (Plan{ .step = 8, .openers = &.{10}, .assistant = &.{ 10, 7 }, .min_chunk = 3 }).starts(a, &ids));
    try std.testing.expectEqualSlices(u32, &.{ 8, 16 }, try (Plan{ .step = 8, .openers = &.{10}, .assistant = &.{ 10, 7 }, .min_chunk = 4 }).starts(a, &ids));
    const talk = [_]u32{ 10, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 10, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 7, 1 };
    try std.testing.expectEqualSlices(u32, &.{ 8, 14, 22, 28 }, try (Plan{ .step = 8, .openers = &.{10}, .assistant = &.{ 10, 7 }, .min_chunk = 4 }).starts(a, &talk));
    try std.testing.expectEqualSlices(u32, &.{ 8, 16 }, try (Plan{ .step = 8 }).starts(a, &ids));
    try std.testing.expectEqualSlices(u32, &.{}, try (Plan{}).starts(a, &ids));
}
