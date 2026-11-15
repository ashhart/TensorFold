//! Flash Next's real pair shaders and relay: epoch refusal, loopback ordering and reused receive storage.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;
const tp = @import("tensorfold").flashnext_replay.Tp2;
const epochs = @import("metal_flashnext_tp_epochs.zig");
const loopback = @import("metal_flashnext_tp_loopback.zig");

pub const epoch: u64 = 0x1_0000_0003;
pub const stale: u64 = @as(u32, @truncate(epoch));
const capacity = 2 * std.enums.values(epochs.Kind).len;

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: [capacity]epochs.Probe = undefined,
    len: usize = 0,
    relay: ?loopback.Probe = null,

    pub fn collect(s: *Suite, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        var libraries: [2]?mtl.Library = @splat(null);
        defer for (libraries) |lib| if (lib) |l| l.deinit();
        inline for (std.enums.values(epochs.Kind)) |kind| {
            const name = comptime epochs.name(kind, false);
            for ([_]bool{ false, true }, &libraries) |control, *lib| {
                if (only) |needle| if (std.mem.indexOf(u8, name, needle) == null) continue;
                if (lib.* == null) lib.* = try s.library(control);
                const pipe = try mtl.Pipeline.init(s.device, lib.*.?, @tagName(kind), false);
                errdefer pipe.deinit();
                try kernels.ensureUnusedCapacity(s.a, 1);
                s.probes[s.len] = .{
                    .rig = .{ .a = s.a, .device = s.device, .queue = s.queue },
                    .pipe = pipe,
                    .kind = kind,
                };
                kernels.appendAssumeCapacity(s.probes[s.len].kernel(epochs.name(kind, control), control));
                s.len += 1;
            }
        }
        const name = loopback.name;
        if (only) |needle| if (std.mem.indexOf(u8, name, needle) == null) return;
        s.relay = .{
            .rig = .{ .a = s.a, .device = s.device, .queue = s.queue },
            .pipe = try @import("rig.zig").pipeline(s.device, tp.shader, "tp_copy"),
        };
        try kernels.append(s.a, s.relay.?.kernel());
    }

    fn library(s: *const Suite, control: bool) !mtl.Library {
        if (!control) return mtl.Library.fromSource(s.device, tp.shader, mtl.CompileOptions.mlx());
        const old = "previous[0] != expected";
        const old_ready = "ready[0] != expected";
        if (std.mem.count(u8, tp.shader, old) != 2 or std.mem.count(u8, tp.shader, old_ready) != 5)
            return error.EpochControlMissing;
        const previous = try std.mem.replaceOwned(u8, s.a, tp.shader, old, "uint(previous[0]) != uint(expected)");
        defer s.a.free(previous);
        const ready = try std.mem.replaceOwned(u8, s.a, previous, old_ready, "uint(ready[0]) != uint(expected)");
        defer s.a.free(ready);
        return mtl.Library.fromSource(s.device, ready, mtl.CompileOptions.mlx());
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.len]) |*p| {
            p.rig.deinit();
            p.pipe.deinit();
        }
        if (s.relay) |*p| {
            p.rig.deinit();
            p.pipe.deinit();
        }
    }
};
