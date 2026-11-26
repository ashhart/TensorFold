//! `ling-size-check DIR RANK WORLD [--fake]`: a rank's weights loaded, each store buffer against weights_size's census.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const ling = @import("ling_cuda");

const W = ling.weights;
const fake = ling.fake_driver;
const Map = std.StringArrayHashMapUnmanaged(u64);

/// Buffer bytes by name, names copied into the arena.
const Sink = struct {
    a: std.mem.Allocator,
    m: *Map,
    pub fn put(s: Sink, name: []const u8, bytes: u64) !void {
        const g = try s.m.getOrPut(s.a, name);
        if (!g.found_existing) {
            g.key_ptr.* = try s.a.dupe(u8, name);
            g.value_ptr.* = 0;
        }
        g.value_ptr.* += bytes;
    }
};

fn total(m: *const Map) u64 {
    var t: u64 = 0;
    for (m.values()) |v| t += v;
    return t;
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 3) return error.MissingArgument;
    const dir = rest[0];
    const rank = try std.fmt.parseInt(u32, rest[1], 10);
    const world = try std.fmt.parseInt(u32, rest[2], 10);
    var fd: cuda.Driver = undefined;
    const d: *const cuda.Driver = if (rest.len > 3 and std.mem.eql(u8, rest[3], "--fake")) blk: {
        fd = fake.driver();
        fake.reset(null, cuda.abi.success);
        break :blk &fd;
    } else gpu.d;
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var why: ling.config.Why = .{};
    var opened = try W.open(gpu.gpa, gpu.io, dir, &why);
    defer opened.ck.close();
    var want: Map = .empty;
    var walk: W.size.Walk(Sink) = .{ .ck = &opened.ck, .c = &opened.c, .rank = rank, .world = world, .sink = .{ .a = a, .m = &want } };
    try walk.all();
    var stream = try cuda.Stream.init(d, false);
    defer stream.deinit();
    var pack = try cuda.modelopt.Packer.load(d);
    defer pack.unload();
    const layers = try a.alloc(u32, opened.c.layers);
    for (layers, 0..) |*l, i| l.* = @intCast(i);
    var w = try W.load(gpu.gpa, d, stream, &pack, &opened.ck, opened.c, layers, .{ .rank = rank, .world = world });
    defer w.deinit();
    var got: Map = .empty;
    const sink: Sink = .{ .a = a, .m = &got };
    for (w.store.named.items) |n| try sink.put(n.name, @max(n.len, 16));
    var differ: usize = 0;
    for (got.keys(), got.values()) |name, bytes| {
        const c = want.get(name);
        if (c != null and c.? == bytes) continue;
        differ += 1;
        if (differ <= 10) std.debug.print("  {s}: stored {d} bytes, census {?d}\n", .{ name, bytes, c });
    }
    for (want.keys(), want.values()) |name, bytes| if (!got.contains(name)) {
        differ += 1;
        if (differ <= 10) std.debug.print("  {s}: census {d} bytes, nothing stored\n", .{ name, bytes });
    };
    std.debug.print("RESULT ling-size-check rank {d} of {d}: {d} buffers, {d} bytes stored; census {d} buffers, {d} bytes; {d} differ\n", .{ rank, world, got.count(), total(&got), want.count(), total(&want), differ });
    try check.expect(differ == 0 and total(&got) == total(&want), "every buffer the loader keeps has the census's name and length", .{});
    check.pass("ling-size-check: rank {d} of {d}, all {d} buffers and {d} bytes as weights_size counts them", .{ rank, world, got.count(), total(&got) });
}
