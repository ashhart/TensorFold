//! EXL3 chain registrations: actual fused bodies, independent FP64 formulas and exact unfused GPU comparators.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const source = @import("exl3_chain_source.zig");
const c = contract.spec;
const g = contract.gates;

fn depth(shape: c.Shape) usize {
    return shape.k / 256 + 30;
}

pub fn For(comptime ex: type) type {
    return struct {
        const Self = @This();
        const Old = @import("exl3_chain_unfused.zig").For(ex);
        const Probe = @import("exl3_chain_probe.zig").For(ex, Old);
        a: std.mem.Allocator,
        old: Old,
        pipelines: [1 + std.enums.values(source.Fault).len][8]?mtl.Pipeline = @splat(@splat(null)),
        probes: std.ArrayList(*Probe) = .empty,

        pub fn matches(only: ?[]const u8) bool {
            const text = only orelse return true;
            var name: [128]u8 = undefined;
            for (std.enums.values(source.Kind)) |kind| {
                const label = std.fmt.bufPrint(&name, "exl3-chain {t}", .{kind}) catch unreachable;
                if (std.mem.indexOf(u8, label, text) != null) return true;
                for (std.enums.values(source.Fault)) |fault| {
                    if (fault == .half_norm and kind != .rms and kind != .qkv) continue;
                    const control = std.fmt.bufPrint(&name, "control: exl3-chain {t} {t}", .{ kind, fault }) catch unreachable;
                    if (std.mem.indexOf(u8, control, text) != null) return true;
                }
            }
            return false;
        }

        pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Self {
            const s = try a.create(Self);
            errdefer a.destroy(s);
            s.* = .{ .a = a, .old = try Old.load(device) };
            errdefer s.clean();
            for (0..s.pipelines.len) |variant| {
                const fault: ?source.Fault = if (variant == 0) null else @fromBackingInt(@intCast(variant - 1));
                const text = try source.source(a, fault);
                defer a.free(text);
                const lib = try source.library(device, text);
                defer lib.deinit();
                for (std.enums.values(source.Kind)) |kind| {
                    if (fault == .half_norm and kind != .rms and kind != .qkv) continue;
                    const name = if (fault) |f| try std.fmt.allocPrint(a, "control: exl3-chain {t} {t}", .{ kind, f }) else try std.fmt.allocPrint(a, "exl3-chain {t}", .{kind});
                    errdefer a.free(name);
                    if (only) |needle| if (std.mem.indexOf(u8, name, needle) == null) {
                        a.free(name);
                        continue;
                    };
                    const pipeline = try mtl.Pipeline.init(device, lib, source.entry(kind), false);
                    s.pipelines[variant][@backingInt(kind)] = pipeline;
                    const need: usize = switch (kind) {
                        .qkv => 320,
                        .rms, .down => 256,
                        else => 128,
                    };
                    if (pipeline.simdWidth() != 32 or pipeline.maxThreads() < need) return error.UnservedExl3ChainPipeline;
                    if (variant == 0) std.debug.print("RESOURCE exl3-chain {t}: maxThreads={d}, staticTG={d}\n", .{ kind, pipeline.maxThreads(), @import("rig.zig").tgMemory(pipeline) });
                    var shapes: std.ArrayList(c.Shape) = .empty;
                    errdefer shapes.deinit(a);
                    const widths: []const usize = switch (kind) {
                        .swiglu, .gate_up => &.{ 128, 256, 1024, 2304, 4096 },
                        .rms => &.{ 128, 512, 1280, 4096 },
                        .qkv => &.{1280},
                        .down => &.{ 128, 256, 5120 },
                        else => &.{ 128, 512, 1024, 1280, 4096, 5120, 8192, 16384, 32768 },
                    };
                    const rows: []const usize = if (kind == .gate_up or kind == .down or kind == .qkv) &.{ 1, 2, 3, 4, 7 } else &.{ 1, 2, 3, 4, 7, 8, 16, 32, 64 };
                    for (widths) |k| for (rows) |n| {
                        if (kind == .rms and 32 * @max(8, k / 128) > pipeline.maxThreads()) continue;
                        try shapes.append(a, .{ .k = k, .n = k, .rows = n });
                    };
                    const served = try shapes.toOwnedSlice(a);
                    errdefer a.free(served);
                    const p = try a.create(Probe);
                    errdefer a.destroy(p);
                    try s.probes.ensureUnusedCapacity(a, 1);
                    try kernels.ensureUnusedCapacity(a, 1);
                    p.* = .{ .rig = .{ .a = a, .device = device, .queue = queue }, .pipeline = pipeline, .old = &s.old, .kind = kind, .fault = fault, .spec = .{
                        .name = name,
                        .accum = .f32,
                        .depth = depth,
                        .ops = 12,
                        .out = .f32,
                        .store = .bf16,
                        .in_place = kind == .rope_in or kind == .gate_up or kind == .rms,
                        .replaces = "unfused EXL3 rotation and native activation, norm, RoPE, KV and combine launches",
                        .nan = .{ .stated = "Metal min/clamp select the finite bound for NaN; other arithmetic retains its IEEE class" },
                        .served = served,
                        .tiles = .{ .{ .size = 4 }, .{}, .{} },
                    } };
                    s.probes.appendAssumeCapacity(p);
                    kernels.appendAssumeCapacity(p.kernel());
                }
            }
            return s;
        }

        fn clean(s: *Self) void {
            for (s.probes.items) |p| {
                p.deinit();
                s.a.free(p.spec.name);
                s.a.free(p.spec.served);
                s.a.destroy(p);
            }
            s.probes.deinit(s.a);
            for (s.pipelines) |variant| for (variant) |pipeline| {
                if (pipeline) |p| p.deinit();
            };
            s.old.deinit();
        }

        pub fn destroy(s: *Self) void {
            s.clean();
            s.a.destroy(s);
        }
    };
}
