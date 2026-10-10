//! Lists every Level Zero device with its compute geometry and which one the tests pick.

const std = @import("std");
const xpu = @import("xpu");
const rt = @import("rt.zig");

pub fn run() !void {
    var d = try xpu.Driver.open();
    defer d.close();
    var all: [xpu.context.max_devices]xpu.context.Entry = undefined;
    const list = try xpu.context.enumerate(&d, &all);
    var names: [xpu.context.max_devices][]const u8 = undefined;
    for (list, 0..) |*e, i| names[i] = xpu.context.entryName(e);
    const pick = try xpu.context.select(names[0..list.len], rt.ordinal);
    std.debug.print("{d} device(s)\n", .{list.len});
    for (list, 0..) |*e, i| {
        const p = e.props;
        const eus = p.num_slices * p.num_subslices_per_slice * p.num_eus_per_subslice;
        std.debug.print("device {d}{s}: {s} type={d} id=0x{x} EUs={d} threads/EU={d} simd={d} clock={d}MHz maxalloc={d}MiB\n", .{ i, if (i == pick) " (picked)" else "", names[i], @intFromEnum(p.type), p.device_id, eus, p.num_threads_per_eu, p.physical_eu_simd_width, p.core_clock_rate, p.max_mem_alloc_size >> 20 });
    }
}
