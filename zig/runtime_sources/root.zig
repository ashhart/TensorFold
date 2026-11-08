//! Assemble production Metal libraries without a device, command queue or dispatch.
const std = @import("std");
const mtl = @import("metal");
pub const Set = @import("set.zig").Set;
pub fn generate(a: std.mem.Allocator, units: bool, generation: u32) !Set {
    var s = Set{ .a = a, .units = units, .generation = generation };
    const device = mtl.Device{ .units = units, .gen = generation };
    try @import("catalogs.zig").add(&s, device);
    try @import("core.zig").add(&s, device);
    try @import("../src/families/glm/kernels.zig").visitSources(a, device, &s);
    try @import("models.zig").add(&s, device);
    try @import("qwen27.zig").add(&s);
    try @import("dsv41.zig").add(&s);
    try @import("flashnext.zig").add(&s);
    for (try @import("../src/families/qwen4/metal_source_inventory.zig").generate(a)) |e| {
        if (std.mem.endsWith(u8, e.name, "-production") or std.mem.endsWith(u8, e.name, "-0") or std.mem.startsWith(u8, e.name, "draw-"))
            try s.raw("qwen4", e.name, e.text);
    }
    return s;
}

pub const packed_probe = @import("../src/families/nemotron/kernels.zig").probe_source;
