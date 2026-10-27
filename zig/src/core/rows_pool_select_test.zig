//! A pooled restore admits only reclaimable storage and keeps the borrowed RAM fallback alive.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const fresh = @import("rows_fake.zig").fresh;
const t = std.testing;
const a = t.allocator;
const Image = struct {
    const api: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter };
    fn parts(_: *anyopaque, _: u32, out: *std.ArrayList(rows.image.Part), allocator: std.mem.Allocator) !void {
        try out.append(allocator, .{ .bytes = 16 });
    }
    fn gather(_: *anyopaque, saved: pc.Saved, _: []const rows.image.Range, out: []u8) !void {
        const state: *Fake.State = @ptrCast(@alignCast(saved));
        std.mem.writeInt(u64, out[0..8], state.at, .little);
        std.mem.writeInt(u64, out[8..16], state.sum, .little);
    }
    fn blank(ptr: *anyopaque, at: u32) !pc.Saved {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        const previous = f.at;
        f.at = at;
        defer f.at = previous;
        return f.pooled().vtable.save(ptr, null, at);
    }
    fn scatter(_: *anyopaque, saved: pc.Saved, _: []const rows.image.Range, bytes: []const u8) !void {
        const state: *Fake.State = @ptrCast(@alignCast(saved));
        state.at = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
        state.sum = std.mem.readInt(u64, bytes[8..16], .little);
    }
};
fn options(root: []const u8) rows.Options {
    return .{ .sync_writes = true, .root = root, .identity = 1691, .cap = 1 << 20, .floor = 0, .daily = 1 << 20, .lookahead = 1 };
}
test "restore selection: pooled buffers trim only when supported and never steal a borrowed state" {
    for ([_]bool{ false, true }) |can_trim| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
        defer a.free(dir);
        var prompt: [64]u32 = undefined;
        for (&prompt, 0..) |*token, i| token.* = @intCast(100 + i);
        const rules: pc.Rules = .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 };
        {
            var f: Fake = .{ .gpa = a, .at = 20, .sum = fresh(prompt[0..20]) };
            var table = f.pooled().vtable.*;
            table.rows = &Image.api;
            const family: pc.Snapshots = .{ .ptr = &f, .vtable = &table };
            const r = try rows.Rows.open(a, family, options(dir));
            defer r.deinit();
            var s = pc.Store.init(a, family, rules, 1000);
            defer s.deinit();
            s.disk = r;
            try t.expect(s.keep(&prompt, 20, null, &.{}, &.{}));
            while (rows.write.step(r)) {}
            try t.expectEqual(@as(usize, 1), r.states.items.len);
        }
        var f: Fake = .{ .gpa = a, .at = 5, .sum = fresh(prompt[0..5]) };
        var table = f.pooled().vtable.*;
        table.rows = &Image.api;
        if (!can_trim) table.trim = null;
        const family: pc.Snapshots = .{ .ptr = &f, .vtable = &table };
        const r = try rows.Rows.open(a, family, options(dir));
        defer r.deinit();
        var s = pc.Store.init(a, family, rules, 240);
        defer s.deinit();
        try t.expect(s.keep(&prompt, 5, null, &.{}, &.{}));
        const borrowed = s.entries.items[0];
        borrowed.borrowers = 1;
        prompt[0] = 999;
        f.at = 7;
        f.sum = fresh(prompt[0..7]);
        try t.expect(s.keep(&prompt, 7, null, &.{}, &.{}));
        prompt[0] = 100;
        s.disk = r;
        const status = rows.read.resumable(&s, &prompt, &.{}, &.{});
        try t.expectEqual(can_trim, status.from == .disk);
        const restored = rows.read.recall(&s, &prompt, 64, &.{}, &.{}, borrowed).?;
        try t.expectEqual(@as(u32, if (can_trim) 20 else 5), restored.at);
        try t.expectEqual(@as(u64, @intFromBool(can_trim)), r.counts.reads);
        try t.expectEqual(@as(u64, @intFromBool(can_trim)), s.counts.evicted);
        try t.expectEqual(@as(u32, 1), borrowed.borrowers);
        try t.expect(s.held + s.spare() <= s.budget);
        try family.vtable.restore(family.ptr, null, restored.saved);
        try t.expectEqual(fresh(&prompt), f.pass(&s, &prompt, .{ .from = restored.at }));
        s.release(borrowed);
    }
}
