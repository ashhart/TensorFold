//! Two nodes on the fake network: the one without a shard pulls its ranges from the one with it, over the cluster's links.
const std = @import("std");
const node = @import("node.zig");
const fake_net = @import("fake_net.zig");
const membership = @import("membership.zig");
const checkpoint = @import("checkpoint.zig");
const loader = @import("loader.zig");
const pull = @import("pull.zig");

const io = std.testing.io;
const gpa = std.testing.allocator;

const Sim = struct {
    net: *fake_net.Net,
    members: [2]membership.Membership,
    agents: [2]pull.Agent = undefined,
    now: u64 = 0,

    fn pump(ctx: *anyopaque) anyerror!void {
        const s: *Sim = @ptrCast(@alignCast(ctx));
        s.now += std.time.ns_per_ms;
        for (&s.members, &s.agents, 0..) |*m, *ag, i| try m.step(s.now, s.net.transport(@intCast(i)), ag.sink());
    }
};

test "a node pulls a missing shard's ranges from its peer over the links and loads it" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var have = std.testing.tmpDir(.{});
    defer have.cleanup();
    var lack = std.testing.tmpDir(.{});
    defer lack.cleanup();
    const bytes = try checkpoint.image(a, &.{
        .{ .name = "model.layers.0.mlp.experts.0.down_proj.weight", .dtype = "U8", .shape = &.{ 100, 900 }, .fill = 7 },
        .{ .name = "model.layers.0.mlp.experts.1.down_proj.weight", .dtype = "U8", .shape = &.{ 100, 900 }, .fill = 8 },
    });
    try have.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes });
    const files = [_]checkpoint.File{.{ .name = "model.safetensors", .size = bytes.len }};
    const sim = try a.create(Sim);
    sim.* = .{ .net = try fake_net.Net.init(gpa, 2, 2), .members = .{ .init(gpa, .{}, .{ .id = 1 }, 1), .init(gpa, .{}, .{ .id = 2 }, 1) } };
    defer {
        for (&sim.members) |*m| m.deinit();
        for (&sim.agents) |*ag| ag.deinit();
        sim.net.deinit();
    }
    sim.net.limit_bytes = 64 << 10;
    sim.net.cable(0, 0, 1, 1);
    for (&sim.agents, [_]std.Io.Dir{ have.dir, lack.dir }, 0..) |*ag, d, i| {
        ag.* = .{ .io = io, .gpa = gpa, .dir = d, .files = &files, .members = &sim.members[i], .t = sim.net.transport(@intCast(i)), .pump = Sim.pump, .pump_ctx = sim, .sources = &.{ 1, 2 } };
    }
    for (0..400) |_| try Sim.pump(sim);
    try std.testing.expect(sim.members[1].settled());
    const header = bytes.len - 180_000;
    const ranges = [_]loader.Range{.{ .file = 0, .offset = header + 90_000, .len = 90_000 }};
    const Check = struct {
        want: []const u8,
        fn run(ptr: *anyopaque, r: loader.Range, view_offset: u64, view: []align(loader.page) const u8) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            try std.testing.expectEqualSlices(u8, c.want[@intCast(r.offset)..][0..@intCast(r.len)], view[@intCast(r.offset - view_offset)..][0..@intCast(r.len)]);
        }
    };
    var check: Check = .{ .want = bytes };
    var rep = try loader.load(io, a, lack.dir, .{ .files = &files, .chunk = 32 << 10 }, &ranges, sim.agents[1].source(), .{ .ptr = &check, .run = Check.run }, null);
    defer rep.release(a);
    try std.testing.expectEqual(@as(u64, 90_000), rep.pulled_bytes);
    try std.testing.expectEqual(@as(u64, 3), sim.agents[0].served);
    for (bytes[header + 90_000 ..][0..10]) |b| try std.testing.expectEqual(@as(u8, 8), b);
}
