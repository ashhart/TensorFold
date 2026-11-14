//! Host candidate selection preserves the full stable sort's value, id and input-position order.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Candidate = struct { value: f64, id: u64 };

pub fn byValueThenId(_: void, a: Candidate, b: Candidate) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

/// Null leaves every nonfinite row to the existing full sort, including its NaN behavior.
pub fn greedy(values: []const f64, ids: []const u64) ?u64 {
    std.debug.assert(values.len > 0 and values.len == ids.len);
    var best: usize = 0;
    for (values, ids, 0..) |value, id, i| {
        if (!std.math.isFinite(value)) return null;
        if (byValueThenId({}, .{ .value = value, .id = id }, .{ .value = values[best], .id = ids[best] })) best = i;
    }
    return ids[best];
}

const Order = struct {
    values: []const f64,
    ids: []const u64,

    fn valid(o: Order, i: usize) bool {
        return std.math.isFinite(o.values[i]);
    }

    fn before(o: Order, a: usize, b: usize) bool {
        if (o.values[a] != o.values[b]) return o.values[a] > o.values[b];
        if (o.ids[a] != o.ids[b]) return o.ids[a] < o.ids[b];
        return a < b;
    }
};

/// Keep the worst retained index at the root, including stable ties with duplicate ids and signed zero.
fn sink(indices: []usize, o: anytype, start: usize) void {
    var at = start;
    while (at < indices.len / 2) {
        var child = 2 * at + 1;
        const right = child + 1;
        if (right < indices.len and o.before(indices[child], indices[right])) child = right;
        if (!o.before(indices[at], indices[child])) break;
        std.mem.swap(usize, &indices[at], &indices[child]);
        at = child;
    }
}

/// Allocate only k candidates and k indices; null asks the caller to use the unchanged nonfinite path.
pub fn topK(gpa: Allocator, values: []const f64, ids: []const u64, k: usize) !?[]Candidate {
    std.debug.assert(values.len == ids.len and k > 0 and k <= values.len);
    const indices = try gpa.alloc(usize, k);
    defer gpa.free(indices);
    const o: Order = .{ .values = values, .ids = ids };
    if (!selectIndices(indices, o, values.len)) return null;
    const selected = try gpa.alloc(Candidate, k);
    for (selected, indices) |*candidate, index| candidate.* = .{ .value = values[index], .id = ids[index] };
    return selected;
}

/// The common bounded heap, sorted afterward in the caller's exact total order.
fn selectIndices(indices: []usize, o: anytype, len: usize) bool {
    for (indices, 0..) |*index, i| {
        if (!o.valid(i)) return false;
        index.* = i;
    }
    var parent = indices.len / 2;
    while (parent > 0) {
        parent -= 1;
        sink(indices, o, parent);
    }
    for (indices.len..len) |i| {
        if (!o.valid(i)) return false;
        if (o.before(i, indices[0])) {
            indices[0] = i;
            sink(indices, o, 0);
        }
    }
    std.mem.sort(usize, indices, o, @TypeOf(o).before);
    return true;
}

pub const Candidate32 = struct { v: f32, i: u32 };

fn Order32(comptime better: anytype) type {
    return struct {
        values: []const f32,
        inv_t: f32,

        fn valid(_: @This(), _: usize) bool {
            return true;
        }

        fn candidate(o: @This(), i: usize) Candidate32 {
            return .{ .v = o.values[i] * o.inv_t, .i = @intCast(i) };
        }

        fn before(o: @This(), a: usize, b: usize) bool {
            return better({}, o.candidate(a), o.candidate(b));
        }
    };
}

/// At least k sorted float32 candidates; a full sort when k exceeds half the row, including nonfinite keys.
pub fn topK32(gpa: Allocator, values: []const f32, inv_t: f32, k: usize, comptime better: anytype) ![]Candidate32 {
    std.debug.assert(k > 0 and k <= values.len);
    const o: Order32(better) = .{ .values = values, .inv_t = inv_t };
    if (k > values.len / 2) {
        const all = try gpa.alloc(Candidate32, values.len);
        for (all, 0..) |*c, i| c.* = o.candidate(i);
        std.mem.sort(Candidate32, all, {}, better);
        return all;
    }
    const indices = try gpa.alloc(usize, k);
    defer gpa.free(indices);
    const valid = selectIndices(indices, o, values.len);
    std.debug.assert(valid);
    const selected = try gpa.alloc(Candidate32, k);
    for (selected, indices) |*c, i| c.* = o.candidate(i);
    return selected;
}
