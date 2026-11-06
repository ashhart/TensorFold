//! LFM2 contract registration and lifetime, including the prompt selector and the existing shared-round controls.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;
const lfm2_argmax = @import("metal_lfm2_argmax.zig");
const lfm2_streams = @import("metal_lfm2_streams.zig");
const lfm2_tap = @import("metal_lfm2_tap.zig");
const lfm2_controls = @import("metal_lfm2_controls.zig");
const Prompt = @import("metal_lfm2_prompt.zig").Suite;
fn wanted(only: ?[]const u8, name: []const u8) bool {
    return if (only) |o| std.mem.indexOf(u8, name, o) != null else true;
}
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    prompt: ?Prompt = null,
    argmax: std.ArrayList(*lfm2_argmax.Probe) = .empty,
    streams: std.ArrayList(*lfm2_streams.Probe) = .empty,
    taps: std.ArrayList(*lfm2_tap.Probe) = .empty,
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        const a = s.a;
        const device = s.device;
        const queue = s.queue;
        s.prompt = .{ .a = a, .device = device, .queue = queue, .kernels = s.kernels };
        try s.prompt.?.collect(only);
        for (std.enums.values(lfm2_controls.Mutation)) |m| {
            const argmax_side = m == .none or m == .argmax_last_tie or m == .argmax_no_flag;
            if (argmax_side) {
                if (!wanted(only, if (m == .none) lfm2_argmax.name else m.name())) continue;
                const p = try a.create(lfm2_argmax.Probe);
                errdefer a.destroy(p);
                p.* = try lfm2_argmax.Probe.init(a, device, queue, m);
                try s.argmax.append(a, p);
                try s.kernels.append(a, p.kernel());
                if (m != .none) continue;
            }
            for (std.enums.values(lfm2_streams.Which)) |w| {
                const control_for = (m == .conv_first_state and w == .conv) or (m == .attention_first_cache and w == .attention);
                if (m != .none and !control_for) continue;
                if (!wanted(only, if (m == .none) lfm2_streams.spec(w).name else m.name())) continue;
                const p = try a.create(lfm2_streams.Probe);
                errdefer a.destroy(p);
                p.* = try lfm2_streams.Probe.init(a, device, queue, w, m);
                try s.streams.append(a, p);
                try s.kernels.append(a, p.kernel());
            }
        }
        for ([_]lfm2_controls.Mutation{ .none, .tap_packed_rows, .tap_clobbers_column }) |m| {
            if (!wanted(only, if (m == .none) lfm2_tap.name else m.name())) continue;
            const p = try a.create(lfm2_tap.Probe);
            errdefer a.destroy(p);
            p.* = try lfm2_tap.Probe.init(a, device, queue, m);
            try s.taps.append(a, p);
            try s.kernels.append(a, p.kernel());
        }
    }
    pub fn deinit(s: *Suite) void {
        if (s.prompt) |*p| p.deinit();
        inline for (.{ "argmax", "streams", "taps" }) |field| {
            const list = &@field(s, field);
            for (list.items) |p| {
                p.deinit();
                s.a.destroy(p);
            }
            list.deinit(s.a);
        }
    }
};
