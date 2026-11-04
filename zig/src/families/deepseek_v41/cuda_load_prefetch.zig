//! CUDA pack readers leave the prepared-image and device-upload order with their existing owners.
const std = @import("std");
const core = @import("core");
const Config = @import("config.zig").Config;
const Pack = @import("pack.zig").Pack;
pub fn begin(gpa: std.mem.Allocator, io: std.Io, pack: *const Pack, cfg: *const Config, index: u32, count: usize) !core.checkpoint_read_ahead.Scope {
    if (count == 0) return .{};
    var buf: [160]u8 = undefined;
    const prefix = try @import("names.zig").blockPrefix(cfg, index, &buf);
    const end = prefix.len;
    buf[end] = '.';
    return core.checkpoint_read_ahead.pack(gpa, io, pack, buf[0 .. end + 1], count);
}

/// The low-level all-model loader retains its original builder path verbatim when disabled.
pub fn all(b: *@import("named.zig").Builder, p: *const @import("plan.zig").Plan, cfg: *const Config, count: usize) !void {
    if (count == 0) return b.all(p, cfg.o_groups);
    for (p.layers) |*layer| {
        var reading = try begin(b.gpa, b.io, b.pack, cfg, layer.index, count);
        defer reading.deinit();
        try b.layer(layer, cfg.o_groups, p.rank, p.world);
        _ = try reading.check();
    }
    try b.top(p);
    try b.dspark(p);
}
