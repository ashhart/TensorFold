//! Copy drafts (prompt lookup): a repeated 8-token run (16 in the reply) drafts what followed it, verified as MTP's.

const std = @import("std");

pub const K: usize = 8; // tokens a match in the prompt takes by default (TF_GLM_COPY_MIN)
pub const K_REPLY: usize = 16; // tokens a match inside the reply takes by default (TF_GLM_COPY_MIN_REPLY)

pub const Copy = struct {
    hist: std.ArrayList(u32) = .empty,
    index: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    prompt_len: usize = 0,
    from: usize = 0, // the first position indexed (a long prompt's head is left out)
    last: ?u32 = null, // where the history's last k tokens ended before (an earlier occurrence)
    k: usize = K, // the indexed runs' length: the shorter of the two matches
    k_prompt: usize = K,
    k_reply: usize = K_REPLY,

    /// ``prompt`` kept and indexed, at most ``most`` last tokens; matches take ``k_prompt`` or ``k_reply`` tokens.
    pub fn init(gpa: std.mem.Allocator, prompt: []const u32, most: usize, k_prompt: usize, k_reply: usize) !Copy {
        const kp = @max(2, k_prompt);
        const kr = @max(2, k_reply);
        var c: Copy = .{ .prompt_len = prompt.len, .from = prompt.len -| most, .k = @min(kp, kr), .k_prompt = kp, .k_reply = kr };
        errdefer c.deinit(gpa);
        try c.hist.appendSlice(gpa, prompt);
        try c.index.ensureTotalCapacity(gpa, @intCast(@min(prompt.len, most) + 1024));
        var i = @max(c.from, c.k - 1);
        while (i < c.hist.items.len) : (i += 1) c.last = c.note(i);
        return c;
    }

    pub fn deinit(c: *Copy, gpa: std.mem.Allocator) void {
        c.hist.deinit(gpa);
        c.index.deinit(gpa);
    }

    /// Committed tokens appended.
    pub fn append(c: *Copy, gpa: std.mem.Allocator, toks: []const u32) !void {
        try c.index.ensureUnusedCapacity(gpa, @intCast(toks.len));
        for (toks) |t| {
            try c.hist.append(gpa, t);
            const i = c.hist.items.len - 1;
            if (i >= c.k - 1 and i >= c.from) c.last = c.note(i);
        }
    }

    /// The run ending at ``i`` indexed; returns where it ended before.
    fn note(c: *Copy, i: usize) ?u32 {
        const gop = c.index.getOrPutAssumeCapacity(hash(c.hist.items[i + 1 - c.k .. i + 1]));
        const prev: ?u32 = if (gop.found_existing) gop.value_ptr.* else null;
        gop.value_ptr.* = @intCast(i);
        return prev;
    }

    /// The tokens that followed the last match, into ``out``; how many (0: no match).
    pub fn propose(c: *const Copy, out: []u32) usize {
        const p: usize = c.last orelse return 0;
        const h = c.hist.items;
        const end = h.len - 1;
        if (p >= end) return 0;
        const need: usize = if (p >= c.prompt_len) c.k_reply else c.k_prompt;
        if (p + 1 < need) return 0;
        if (!std.mem.eql(u32, h[p + 1 - need .. p + 1], h[end + 1 - need .. end + 1])) return 0; // a hash collision, or short of 16
        const n = @min(out.len, end - p);
        @memcpy(out[0..n], h[p + 1 .. p + 1 + n]);
        return n;
    }
};

fn hash(run: []const u32) u64 {
    return std.hash.Wyhash.hash(0x636f7079, std.mem.sliceAsBytes(run));
}

/// Copied drafts' landing rate, learned from the verifies (shared by the streams): starts at 0.8.
pub const Rate = struct {
    tried: f64 = 0,
    landed: f64 = 0,

    pub fn chance(r: *const Rate) f64 {
        return (r.landed + 4) / (r.tried + 5);
    }

    /// A window that verified ``drafted`` copied tokens kept ``kept`` of them; older windows fade.
    pub fn record(r: *Rate, drafted: u32, kept: u32) void {
        r.tried = r.tried * 0.995 + @as(f64, @floatFromInt(drafted));
        r.landed = r.landed * 0.995 + @as(f64, @floatFromInt(@min(kept, drafted)));
    }
};

test "a prompt's run is copied, a reply's needs 16 tokens" {
    const gpa = std.testing.allocator;
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 51 };
    var c = try Copy.init(gpa, &prompt, 1 << 20, K, K_REPLY);
    defer c.deinit(gpa);
    var out: [7]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 0), c.propose(&out));
    try c.append(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    const n = c.propose(&out);
    try std.testing.expectEqual(@as(usize, 7), n);
    try std.testing.expectEqualSlices(u32, &.{ 9, 10, 11, 12, 50, 51, 1 }, out[0..n]);
    // a run of 8 that last ran inside the reply: not enough
    var d = try Copy.init(gpa, &.{ 99, 98 }, 1 << 20, K, K_REPLY);
    defer d.deinit(gpa);
    try d.append(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 77, 1, 2, 3, 4, 5, 6, 7, 8 });
    try std.testing.expectEqual(@as(usize, 0), d.propose(&out));
}

test "the rate learns" {
    var r: Rate = .{};
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), r.chance(), 1e-9);
    for (0..50) |_| r.record(7, 1);
    try std.testing.expect(r.chance() < 0.3);
}
