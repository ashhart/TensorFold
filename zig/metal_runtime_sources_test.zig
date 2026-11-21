//! Compile assembled runtime libraries with no buffers, pipelines, command queues or dispatch.
const std = @import("std");
const mtl = @import("metal");
const inventory = @import("inventory").metal_runtime_sources;

test "Metal runtime source compile gate: every family" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const set = try inventory.generate(arena.allocator(), device.tensorUnits(), device.generation());
    const packed_ok = blk: {
        const probe = mtl.Library.fromSource(device, inventory.packed_probe, mtl.CompileOptions.mlx()) catch break :blk false;
        probe.deinit();
        break :blk true;
    };
    var failed: usize = 0;
    var checked: usize = 0;
    var prebuilt: usize = 0;
    for (set.entries.items, 0..) |e, i| {
        const family = @import("source_options").family;
        if (!std.mem.startsWith(u8, e.family, family)) continue;
        if (e.packed_inline and !packed_ok) {
            std.debug.print("Metal runtime source uses loader's packed metallib fallback: {s}/{s}\n", .{ e.family, e.name });
            prebuilt += 1;
            continue;
        }
        const library_pool = mtl.objc.Pool.push();
        defer library_pool.pop();
        checked += 1;
        std.debug.print("Metal runtime source {d}/{d}: {s}/{s}\n", .{ i + 1, set.entries.items.len, e.family, e.name });
        const library = mtl.Library.fromSource(device, e.text, e.options) catch |err| {
            std.debug.print("FAIL Metal runtime source {s}/{s}: {t}\n", .{ e.family, e.name, err });
            failed += 1;
            continue;
        };
        library.deinit();
    }
    std.debug.print("SUMMARY Metal runtime sources: {d} attempted, {d} failed, {d} prebuilt-fallback libraries\n", .{ checked, failed, prebuilt });
    if (checked == 0) return error.NoRuntimeSourceCases;
    if (failed != 0) return error.MetalRuntimeSourcesFailed;
}
