//! LFM2.5-24B-A2B's kept states: liquid's over this model (the conv states, then the ten attention layers' pages).
const std = @import("std");
const liquid = @import("liquid_cuda");
const mdl = @import("cuda_model.zig");
const kern = @import("cuda_kernels.zig");

const S = liquid.snap.States(mdl);
pub const Saved = S.Saved;
pub const bytes = S.bytes;
pub const save = S.save;
pub const restore = S.restore;
pub const drop = S.drop;
pub const page = liquid.snap.page;

const page_bytes = mdl.page_elems * 2;
const row_bytes = mdl.kv_width * 2;
const state_bytes = mdl.slot_elems * 2;

const testing = std.testing;

const Count = struct {
    runs: *std.ArrayList([3]usize),
    fn add(c: Count, dev: u64, off: usize, len: usize) anyerror!void {
        try c.runs.append(testing.allocator, .{ dev, off, len });
    }
};

fn fake(ids: []u32, caps: []usize) mdl.Model {
    var m: mdl.Model = undefined;
    m.context = 4 * kern.page;
    m.slots = caps.len;
    m.arena = 1 << 40;
    m.states = 1 << 36;
    m.ids = ids;
    m.caps = caps;
    return m;
}

test "a state: the conv states, consecutive whole pages as one copy, the last page's rows a layer's K and V" {
    // slot 1 leases pages 7, 8, 3, 9: positions 0..1023 one run, 1024..1535 a page, 1536.. a partial page
    var ids = [_]u32{ 0, 0, 0, 0, 7, 8, 3, 9 };
    var caps = [_]usize{ 0, 4 * kern.page };
    const m = fake(&ids, &caps);
    const at = 3 * kern.page + 100;
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try S.each(&m, 1, at, null, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 1 + 2 + 20), runs.items.len);
    try testing.expectEqual([3]usize{ (1 << 36) + state_bytes, 0, state_bytes }, runs.items[0]);
    try testing.expectEqual([3]usize{ (1 << 40) + 7 * page_bytes, state_bytes, 2 * page_bytes }, runs.items[1]);
    try testing.expectEqual([3]usize{ (1 << 40) + 3 * page_bytes, state_bytes + 2 * page_bytes, page_bytes }, runs.items[2]);
    const v9 = (1 << 40) + 9 * page_bytes + kern.page * row_bytes; // page 9, layer 0's values
    try testing.expectEqual([3]usize{ v9, state_bytes + 3 * page_bytes + 100 * row_bytes, 100 * row_bytes }, runs.items[4]);
    var total: usize = 0;
    for (runs.items) |r| total += r[2];
    try testing.expectEqual(state_bytes + at * 20 * 1024, total);
    try testing.expectEqual((total + page - 1) / page * page, S.bytes(at));
}

test "a state on whole pages has no partial copies; one past a slot's pages is refused" {
    var ids = [_]u32{ 4, 5, 6, 7 };
    var caps = [_]usize{2 * kern.page};
    const m = fake(&ids, &caps);
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try S.each(&m, 0, 2 * kern.page, null, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 2), runs.items.len);
    try testing.expectEqual(@as(usize, 2 * page_bytes), runs.items[1][2]);
    try testing.expectError(error.PastPages, S.each(&m, 0, 2 * kern.page + 1, null, Count{ .runs = &runs }, Count.add));
}
