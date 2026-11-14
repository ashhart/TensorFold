//! Shared libraries and lifetime for the GLM exchange contract probes.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;
const rig = @import("rig.zig");
const copy = @import("metal_glm_ep_copy.zig");
const reduce = @import("metal_glm_ep_reduce.zig");
const source = @import("kernel_sources").glm_ep;

pub const epoch: u64 = 0x1_0000_0003;
pub const stale: u64 = @as(u32, @truncate(epoch));

pub fn name(comptime kind: anytype) []const u8 {
    return "metal glm ep " ++ @tagName(kind);
}

fn mutation(a: std.mem.Allocator, input: []const u8, old: []const u8, new: []const u8, count: usize) ![]u8 {
    if (std.mem.count(u8, input, old) != count) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, input, old, new);
}

const Probe = union(enum) { copy: copy.Probe, reduce: reduce.Probe };
const capacity = 2 * (std.enums.values(copy.Kind).len + std.enums.values(reduce.Kind).len);

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: [capacity]Probe = undefined,
    len: usize = 0,

    pub fn collect(s: *Suite, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        var libraries: [2]?mtl.Library = @splat(null);
        defer for (libraries) |lib| if (lib) |l| l.deinit();
        inline for (.{ copy, reduce }, .{ "copy", "reduce" }) |suite, field| {
            inline for (std.enums.values(suite.Kind)) |kind| {
                for ([_]bool{ false, true }, &libraries) |control, *lib| {
                    const label = suite.caseName(kind, control);
                    const filter = if (suite == copy) name(kind) else label;
                    if (only) |needle| if (std.mem.indexOf(u8, filter, needle) == null) continue;
                    if (lib.* == null) lib.* = try s.library(control);
                    try kernels.ensureUnusedCapacity(s.a, 1);
                    const pipe = try mtl.Pipeline.init(s.device, lib.*.?, @tagName(kind), false);
                    errdefer pipe.deinit();
                    if (suite == copy and (pipe.maxThreads() < kind.threads() or rig.tgMemory(pipe) != 0))
                        return error.UnexpectedLaunchLimits;
                    s.probes[s.len] = @unionInit(Probe, field, .{
                        .rig = .{ .a = s.a, .device = s.device, .queue = s.queue },
                        .pipe = pipe,
                        .kind = kind,
                    });
                    kernels.appendAssumeCapacity(@field(s.probes[s.len], field).kernel(label, control));
                    s.len += 1;
                }
            }
        }
    }

    fn library(s: *const Suite, control: bool) !mtl.Library {
        const opts = mtl.CompileOptions.mlx();
        if (!control) return mtl.Library.fromSource(s.device, source, opts);
        const sends = try mutation(s.a, source, "previous[0] != expected", "uint(previous[0]) != uint(expected)", 4);
        defer s.a.free(sends);
        const receives = try mutation(s.a, sends, "ready[0] == x", "uint(ready[0]) == uint(x)", 3);
        defer s.a.free(receives);
        const merges = try mutation(s.a, receives, "ready[0] != x", "uint(ready[0]) != uint(x)", 1);
        defer s.a.free(merges);
        return mtl.Library.fromSource(s.device, merges, opts);
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.len]) |*probe| switch (probe.*) {
            inline else => |*p| {
                p.rig.deinit();
                p.pipe.deinit();
            },
        };
    }
};
