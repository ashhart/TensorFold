//! Isolated non-finite prompt-mask panels registered alongside the families' general operator contracts.
const std = @import("std");
const metal = @import("metal");
const ct = @import("contract");
const generic = @import("masked_value_probe.zig");
const Basic = generic.Types(@import("masked_attention_basic.zig")).Suite;
const More = generic.Types(@import("masked_attention_more.zig")).Suite;
pub const Suite = struct {
    a: std.mem.Allocator,
    device: metal.Device,
    queue: metal.Queue,
    kernels: *std.ArrayList(ct.gates.Kernel),
    basic: ?Basic = null,
    more: ?More = null,
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        s.basic = .{ .a = s.a, .device = s.device, .queue = s.queue, .kernels = s.kernels };
        try s.basic.?.collect(only);
        try s.basic.?.controls(.qwen35, only);
        s.more = .{ .a = s.a, .device = s.device, .queue = s.queue, .kernels = s.kernels };
        try s.more.?.collect(only);
    }
    pub fn deinit(s: *Suite) void {
        if (s.basic) |*x| x.deinit();
        if (s.more) |*x| x.deinit();
    }
};
