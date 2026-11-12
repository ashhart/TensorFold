//! Request history and bounded row descriptors for the GLM target logit processor.
const std = @import("std");

pub const max_rows = 32;
pub const Row = extern struct { slot: u32, position: u32, factor: f32, reserved: u32 = 0 };

pub fn validFactor(factor: f32) bool {
    return std.math.isFinite(factor) and factor > 0;
}

fn mark(bits: []u32, id: u32) void {
    bits[id / 32] |= @as(u32, 1) << @intCast(id % 32);
}

fn validIds(vocab: usize, ids: []const u32) !void {
    for (ids) |id| if (id >= vocab) return error.InvalidToken;
}

pub const History = struct {
    vocab: usize,
    bits: []u32,
    prompt_bits: []u32,
    prefix: std.ArrayList(u32) = .empty,

    pub fn init(gpa: std.mem.Allocator, vocab: usize, prompt: []const u32) !History {
        if (vocab == 0 or vocab > std.math.maxInt(u32)) return error.BadProcessorTable;
        try validIds(vocab, prompt);
        const bits = try gpa.alloc(u32, (vocab + 31) / 32);
        errdefer gpa.free(bits);
        @memset(bits, 0);
        for (prompt) |id| mark(bits, id);
        const prompt_bits = try gpa.dupe(u32, bits);
        errdefer gpa.free(prompt_bits);
        var h: History = .{ .vocab = vocab, .bits = bits, .prompt_bits = prompt_bits };
        errdefer h.prefix.deinit(gpa);
        try h.prefix.appendSlice(gpa, prompt);
        return h;
    }

    /// Only authoritative committed IDs update this set; proposed rows use Stage's separate masks.
    pub fn sync(h: *History, gpa: std.mem.Allocator, prefix: []const u32) !void {
        try validIds(h.vocab, prefix);
        const extends = prefix.len >= h.prefix.items.len and std.mem.eql(u32, h.prefix.items, prefix[0..h.prefix.items.len]);
        try h.prefix.ensureTotalCapacity(gpa, prefix.len);
        const start = if (extends) h.prefix.items.len else 0;
        if (!extends) @memset(h.bits, 0);
        for (prefix[start..]) |id| mark(h.bits, id);
        h.prefix.clearRetainingCapacity();
        h.prefix.appendSliceAssumeCapacity(prefix);
    }

    pub fn deinit(h: *History, gpa: std.mem.Allocator) void {
        h.prefix.deinit(gpa);
        gpa.free(h.bits);
        gpa.free(h.prompt_bits);
    }
};

pub const Stage = struct {
    vocab: usize,
    words: usize,
    rows: [max_rows]Row = undefined,
    len: usize = 0,
    active: bool = false,
    masks: std.ArrayList(u32) = .empty,

    pub fn init(vocab: usize) !Stage {
        if (vocab == 0 or vocab > std.math.maxInt(u32)) return error.BadProcessorTable;
        return .{ .vocab = vocab, .words = (vocab + 31) / 32 };
    }

    pub fn deinit(s: *Stage, gpa: std.mem.Allocator) void {
        s.masks.deinit(gpa);
    }

    pub fn reset(s: *Stage) void {
        s.len = 0;
        s.active = false;
        s.masks.clearRetainingCapacity();
    }

    pub fn descriptors(s: *const Stage) []const Row {
        return s.rows[0..if (s.active) s.len else 0];
    }

    /// `ids` is pending followed by resolved drafts; an empty suffix stages one prompt prediction.
    pub fn add(s: *Stage, gpa: std.mem.Allocator, slot: u32, first: u64, factor: f32, h: ?*const History, ids: []const u32) !void {
        const count = @max(ids.len, 1);
        if (!validFactor(factor) or s.len + count > max_rows or first == 0 or
            first > std.math.maxInt(u32) - (count - 1)) return error.BadProcessorTable;
        try validIds(s.vocab, ids);
        const enabled = factor != 1;
        if (enabled and (h == null or h.?.vocab != s.vocab)) return error.NoProcessorHistory;
        if (enabled or s.active) {
            try s.masks.resize(gpa, (s.len + count) * s.words);
            if (!s.active) @memset(s.masks.items[0 .. s.len * s.words], 0);
            for (0..count) |r| {
                const dst = s.masks.items[(s.len + r) * s.words ..][0..s.words];
                if (enabled) {
                    const src = if (r == 0) h.?.bits else s.masks.items[(s.len + r - 1) * s.words ..][0..s.words];
                    @memcpy(dst, src);
                    if (ids.len != 0) mark(dst, ids[r]);
                } else @memset(dst, 0);
            }
        }
        for (0..count) |r| s.rows[s.len + r] = .{ .slot = slot, .position = @intCast(first + r), .factor = factor };
        s.len += count;
        s.active = s.active or enabled;
    }
};

pub const Table = struct {
    rows: [max_rows]Row = undefined,
    len: usize = 0,

    pub fn set(t: *Table, rows: []const Row, vocab: usize, words: usize, slots: usize, capacity: usize) !void {
        if (vocab == 0 or vocab > std.math.maxInt(u32) or words != (vocab + 31) / 32 or rows.len > max_rows)
            return error.BadProcessorTable;
        for (rows, 0..) |r, i| {
            if (r.slot >= slots or r.position == 0 or r.position > capacity or r.position > std.math.maxInt(i32) or
                r.reserved != 0 or !validFactor(r.factor))
                return error.BadProcessorTable;
            if (i != 0 and rows[i - 1].slot == r.slot) {
                const previous = rows[i - 1];
                if (r.position != previous.position + 1 or r.factor != previous.factor)
                    return error.BadProcessorTable;
            } else {
                for (rows[0..i]) |p| if (p.slot == r.slot) return error.BadProcessorTable;
            }
        }
        @memcpy(t.rows[0..rows.len], rows);
        t.len = rows.len;
    }

    pub fn find(t: *const Table, slot: u32, first: u64, count: usize) !?usize {
        if (t.len == 0) return null;
        if (count == 0 or count > t.len) return error.ProcessorPositionMismatch;
        for (t.rows[0..t.len], 0..) |r, i| {
            if (r.slot != slot or r.position != first) continue;
            if (i + count > t.len) return error.ProcessorPositionMismatch;
            for (t.rows[i..][0..count], 0..) |p, j| {
                if (p.slot != slot or p.position != first + j or p.factor != r.factor)
                    return error.ProcessorPositionMismatch;
            }
            return i;
        }
        return error.ProcessorPositionMismatch;
    }
};

pub fn decode(data: []const u32, count: usize, vocab: usize, words: usize, slots: usize, capacity: usize) !Table {
    if (count > max_rows or data.len != count * 4) return error.BadProcessorTable;
    var rows: [max_rows]Row = undefined;
    for (rows[0..count], 0..) |*r, i| r.* = .{
        .slot = data[i * 4],
        .position = data[i * 4 + 1],
        .factor = @bitCast(data[i * 4 + 2]),
        .reserved = data[i * 4 + 3],
    };
    var t: Table = .{};
    try t.set(rows[0..count], vocab, words, slots, capacity);
    return t;
}

test "GLM repetition: full prompt membership includes old tokens and repeated IDs once" {
    var h = try History.init(std.testing.allocator, 65, &.{ 0, 0, 32, 64, 32 });
    defer h.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1, 1 }, h.bits);
}
