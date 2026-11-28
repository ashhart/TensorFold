//! A non-GLM family's actual CUDA cut declarations use the shared affordable-checkpoint planner.
const std = @import("std");
const pc = @import("engine_api").prompt_cache;
const family = @import("family").native;
const adapter = @import("copies");
const t = std.testing;

const Cpu = struct {
    live: [2]u64 = .{ 0, 0 },
    fn bytes(_: *anyopaque, at: u32) u64 {
        return at;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) !pc.Saved {
        const f: *Cpu = @ptrCast(@alignCast(ptr));
        try t.expectEqual(@as(u64, at), f.live[0]);
        const saved = try t.allocator.create([2]u64);
        saved.* = f.live;
        return saved;
    }
    fn restore(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) !void {
        const f: *Cpu = @ptrCast(@alignCast(ptr));
        f.live = @as(*[2]u64, @ptrCast(@alignCast(saved))).*;
    }
    fn drop(_: *anyopaque, saved: pc.Saved) void {
        t.allocator.destroy(@as(*[2]u64, @ptrCast(@alignCast(saved))));
    }
    fn pass(f: *Cpu, s: *pc.Store, ids: []const u32, p: pc.Plan) ![2]u64 {
        if (p.from == 0) f.live = .{ 0, 0 };
        for (ids[p.from..]) |id| {
            f.live[1] = f.live[1] *% 31 +% id;
            f.live[0] += 1;
            const at: u32 = @intCast(f.live[0]);
            if (std.mem.indexOfScalar(u32, p.marks, at) != null) try t.expect(s.keep(ids, at, null, &.{}, &.{}));
        }
        return f.live;
    }
};

test "CUDA cache budget: real LFM2-MoE declarations retain and restore an affordable cold cut" {
    try t.expectEqualStrings("lfm2_moe", family.model_type);
    var cpu: Cpu = .{};
    const copies: pc.Snapshots = .{ .ptr = &cpu, .vtable = &.{ .bytes = Cpu.bytes, .save = Cpu.save, .restore = Cpu.restore, .drop = Cpu.drop } };
    var store = pc.Store.init(t.allocator, copies, adapter.passRules(family, family.prompt_rows), 8192);
    defer store.deinit();
    var ids: [12719]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i % 101);
    const p = try store.begin(t.allocator, &ids, 12705, &.{}, &.{}, null, &.{});
    defer t.allocator.free(p.marks);
    try t.expectEqualSlices(u32, &.{8192}, p.marks);
    try t.expectEqual(@as(u32, 0), 8192 % family.prompt_rows);
    const cold = try cpu.pass(&store, &ids, p);
    const q = try store.begin(t.allocator, &ids, 12705, &.{}, &.{}, null, &.{});
    defer t.allocator.free(q.marks);
    try t.expectEqual(@as(u32, 8192), q.from);
    const resumed = try cpu.pass(&store, &ids, q);
    try t.expectEqualSlices(u8, std.mem.asBytes(&cold), std.mem.asBytes(&resumed));
}
