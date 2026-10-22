//! Prefix-161 ID and row-gather oracles, retained only for the default compatibility gate.
const std = @import("std");
pub fn rawIds(gpa: std.mem.Allocator, vocab_text: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, vocab_text, " \t\r\n,");
    while (it.next()) |part| {
        const id = try std.fmt.parseInt(u32, part, 10);
        if (id >= 248320) continue;
        try out.append(gpa, id);
    }
    std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    var n: usize = 0;
    for (out.items, 0..) |id, i| {
        if (i > 0 and id == out.items[n - 1]) continue;
        out.items[n] = id;
        n += 1;
    }
    out.shrinkRetainingCapacity(n);
    return out.toOwnedSlice(gpa);
}

pub fn paddedIds(gpa: std.mem.Allocator, text: []const u8, vocab: usize) ![]u32 {
    var listed: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer listed.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
    while (it.next()) |token| {
        const id = std.fmt.parseInt(u32, token, 10) catch return error.BadDraftVocab;
        if (id >= vocab) return error.DraftIdPastVocab;
        try listed.put(gpa, id, {});
    }
    if (listed.count() == 0) return error.BadDraftVocab;
    var extra: std.ArrayList(u32) = .empty;
    defer extra.deinit(gpa);
    var candidate: u32 = 0;
    while ((listed.count() + extra.items.len) % 64 != 0) {
        if (candidate >= vocab) return error.DraftVocabExhausted; // a pad past the vocabulary would gather past lm_head
        if (!listed.contains(candidate)) try extra.append(gpa, candidate);
        candidate += 1;
    }
    const all = try gpa.alloc(u32, listed.count() + extra.items.len);
    errdefer gpa.free(all);
    var n: usize = 0;
    var it2 = listed.keyIterator();
    while (it2.next()) |k| {
        all[n] = k.*;
        n += 1;
    }
    @memcpy(all[n..], extra.items);
    std.mem.sort(u32, all, {}, std.sort.asc(u32));
    return all;
}

pub fn putDraftHead(comptime pack: type, out: *pack.Out, gpa: std.mem.Allocator, a: std.mem.Allocator, name: []const u8, lm_head: pack.Q, ids: []const u32) !void {
    const rows = ids.len;
    const kw = lm_head.kw;
    const kg = lm_head.s.dim(1);
    const wq = try gpa.alloc(u8, rows * kw * 4); // putLane copies the gathered rows, so they free here
    defer gpa.free(wq);
    const s = try gpa.alloc(u8, rows * kg * 2);
    defer gpa.free(s);
    const b = try gpa.alloc(u8, rows * kg * 2);
    defer gpa.free(b);
    const w32 = std.mem.bytesAsSlice(u32, wq);
    const src32 = std.mem.bytesAsSlice(u32, lm_head.part(.weight));
    const s16 = std.mem.bytesAsSlice(u16, s);
    const src_s16 = std.mem.bytesAsSlice(u16, lm_head.part(.scales));
    const b16 = std.mem.bytesAsSlice(u16, b);
    const src_b16 = std.mem.bytesAsSlice(u16, lm_head.part(.biases));
    for (ids, 0..) |id, r| {
        if (id * kw + kw > src32.len or id * kg + kg > src_s16.len) return error.DraftIdPastVocab;
        @memcpy(w32[r * kw ..][0..kw], src32[id * kw ..][0..kw]);
        @memcpy(s16[r * kg ..][0..kg], src_s16[id * kg ..][0..kg]);
        @memcpy(b16[r * kg ..][0..kg], src_b16[id * kg ..][0..kg]);
    }
    const gathered = [_]pack.Q{.{ .w = .{ .dtype = .u32, .rank = 2, .shape = .{ rows, kw, 0, 0, 0 }, .bytes = wq }, .s = .{ .dtype = .bf16, .rank = 2, .shape = .{ rows, kg, 0, 0, 0 }, .bytes = s }, .b = .{ .dtype = .bf16, .rank = 2, .shape = .{ rows, kg, 0, 0, 0 }, .bytes = b }, .rows = rows, .k = lm_head.k, .kw = kw, .spec = lm_head.spec }};
    try pack.lane.putLane(out, gpa, name, gathered[0..]);
    _ = a;
}
