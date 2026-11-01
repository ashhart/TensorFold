//! The release loader over mapped shards: every result bit for bit what the copying loader gives.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const fixture = @import("fixture.zig");
const ff = @import("fixture_fwd.zig");
const ft = @import("forward_test.zig");

const testing = std.testing;
const limit = 1280;
const Mode = fixture.Rig.Mode;

fn run(a: std.mem.Allocator, mode: Mode) !struct { f: *ff.Fwd, out: ft.Out } {
    const f = try ff.Fwd.initMode(a, testing.io, limit, 61, mode);
    errdefer f.deinit();
    fixture.fillPast(&f.r.c, &f.r.slot, 1101, 7);
    const hist = try ff.ids(a, &f.r.c, ft.look + 6, 91);
    defer a.free(hist);
    return .{ .f = f, .out = try ft.runChunks(a, f, &f.r.slot, hist, 1101, 3) };
}

test "dsv41 metal zero-copy: 3 mapped shards (aligned, then skewed so every tensor is copied) == the copy, logits and taps bit for bit" {
    const a = testing.allocator;
    const base = try run(a, .copy);
    defer base.f.deinit();
    defer a.free(base.out.logits);
    defer a.free(base.out.taps);
    var placed: [2]u64 = undefined;
    var refused: [2]u64 = undefined;
    for ([_]Mode{ .mapped, .skewed }, 0..) |mode, i| {
        const got = try run(a, mode);
        defer got.f.deinit();
        defer a.free(got.out.logits);
        defer a.free(got.out.taps);
        const st = got.f.r.map.?.stats;
        placed[i] = st.tensors;
        refused[i] = st.refused;
        std.debug.print("zero-copy {t}: {d} shards, {d} tensors in place ({d} B), {d} refused ({d} B)\n", .{ mode, st.shards, st.tensors, st.bytes, st.refused, st.refused_bytes });
        if (mode == .mapped) try testing.expectEqual(@as(u32, 3), st.shards) else try testing.expectEqual(@as(u64, 0), st.tensors);
        try testing.expectEqualSlices(u32, @ptrCast(base.out.logits), @ptrCast(got.out.logits));
        try testing.expectEqualSlices(u16, base.out.taps, got.out.taps);
    }
    try testing.expect(placed[0] > 0);
    try testing.expect(refused[1] > refused[0]);
}

test "dsv41 metal zero-copy: the float64 stage checks hold on mapped weights" {
    const a = testing.allocator;
    const f = try ff.Fwd.initMode(a, testing.io, limit, 51, .mapped);
    defer f.deinit();
    fixture.fillPast(&f.r.c, &f.r.slot, 1101, 7);
    try testing.expectEqual(@as(usize, 0), try ft.stagewise(a, f, 1101, 3, 13, false));
}

const host = @import("dsv41");

test "dsv41 metal zero-copy: a shard that is mostly Engram table is never mapped; others are, byte for byte" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const x = try host.pack.image(a, &.{.{ .name = "norm.weight", .dtype = "BF16", .shape = &.{64}, .fill = 7 }});
    defer a.free(x);
    const t = try host.pack.image(a, &.{
        .{ .name = "layers.1.engram.embed.weight", .dtype = "F8_E4M3", .shape = &.{ 64, 256 }, .fill = 1 },
        .{ .name = "layers.1.engram.embed.scale", .dtype = "F8_E8M0", .shape = &.{ 64, 8 }, .fill = 2 },
        .{ .name = "other.weight", .dtype = "BF16", .shape = &.{64}, .fill = 9 },
    });
    defer a.free(t);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "a.safetensors", .data = x });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "b.safetensors", .data = t });
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    defer a.free(dir);
    var pack = try host.pack.Pack.open(a, testing.io, dir);
    defer pack.deinit();
    const device = try mtl.Device.init();
    defer device.deinit();
    var m = try pkg.mapped.Map.open(a, device, &pack);
    defer m.deinit();
    const norm = pack.get("norm.weight").?;
    const p = (try m.place(norm, 1)).?;
    try testing.expectEqual(@as(usize, norm.start), p.off);
    try testing.expectEqual(@as(u8, 7), (p.buf.contents() + p.off)[norm.nbytes - 1]);
    try testing.expectEqual(@as(u32, 1), m.stats.shards);
    try testing.expect((try m.place(pack.get("other.weight").?, 1)) == null);
    try testing.expect((try m.place(pack.get("layers.1.engram.embed.weight").?, 1)) == null);
    try testing.expectEqual(@as(u32, 1), m.stats.shards);
    // the buffer is whole pages of the file, and a width that does not divide the offset is refused
    try testing.expectEqual(@as(u64, 0), m.stats.wired % std.heap.pageSize());
    try testing.expect((try m.place(norm, 1 << 20)) == null);
    try testing.expectEqual(@as(u64, 1), m.stats.refused);
}

test "dsv41 metal zero-copy: the whole tiny model over 3 shards in one residency set, and its report" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var c = fixture.forwardConfig();
    try c.validate();
    const img = try fixture.image(a, &c, 5);
    defer a.free(img);
    try @import("shards.zig").write(a, testing.io, tmp.dir, img, 3, 0);
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    defer a.free(dir);
    var pack = try host.pack.Pack.open(a, testing.io, dir);
    defer pack.deinit();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const m = try pkg.model_weights.Model.load(a, testing.io, device, queue, &pack, &c, null);
    defer m.deinit();
    const r = m.report();
    std.debug.print("model: {d} shards {d} B wired, {d} tensors in place, {d} B in arenas, resident {d} B, set {}\n", .{ r.shards, r.shard_bytes, r.zero_copy_tensors, r.arena_bytes, r.resident(), m.resident != null });
    try testing.expectEqual(c.layers, m.view().layers.len);
    try testing.expectEqual(@as(u32, 3), r.shards);
    try testing.expect(m.resident != null);
    try testing.expect(r.zero_copy_bytes <= r.shard_bytes);
}
