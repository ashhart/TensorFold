//! Synthetic GPU gate; its compile-only step never executes device code.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

pub fn runGate() !void {
    const a = std.testing.allocator;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    for ([_]usize{ 4, 6 }) |bits| {
        const fixture = try tf.flashnext_fsr_fixture.Fixture.init(a, device, queue, 2, bits, 0x372, false);
        defer fixture.deinit(a);
        const resident = try fixture.run(false);
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "experts", .data = fixture.data });
        const path = try tmp.dir.realPathFileAlloc(std.testing.io, "experts", a);
        defer a.free(path);
        const name = try std.fmt.allocPrintSentinel(a, "{s}", .{path}, 0);
        defer a.free(name);
        const native = tf.flashnext_fsr_fixture.NativeFile;
        const file = try native.open(name);
        defer file.close();
        defer fixture.cache.join() catch {};
        fixture.cache.source.file = file;
        for (0..5) |_| {
            const streamed = try fixture.run(true);
            try std.testing.expectEqualSlices(u8, resident, streamed);
            try std.testing.expect(fixture.intact());
        }
        fixture.wrong_map = true;
        const wrong = try fixture.run(true);
        try std.testing.expect(!std.mem.eql(u8, resident, wrong));
        std.debug.print("FSR_METAL bits={d} rows=2 layers=2 experts=32 slots=4 all nodes and 32 logits exact, five repeats, uncached reads, wrong-map control caught\n", .{bits});
    }
}

test "FSR Metal synthetic MoE uses four expert slots and preserves every logit" {
    try runGate();
}
