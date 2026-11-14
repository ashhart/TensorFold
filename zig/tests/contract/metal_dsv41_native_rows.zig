//! Native batched RoPE and packed-cache gates, not the separate release-format attention implementations.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const source = @import("dsv41_native_rows_source.zig");
const c = contract.spec;
const g = contract.gates;

fn depth(_: c.Shape) usize {
    return 10;
}

pub fn For(comptime native: type) type {
    return struct {
        const Self = @This();
        const Probe = @import("dsv41_native_rows_probe.zig").For(native);
        a: std.mem.Allocator,
        old: [3]?mtl.Pipeline = @splat(null),
        pipelines: [5][3]?mtl.Pipeline = @splat(@splat(null)),
        probes: std.ArrayList(*Probe) = .empty,

        pub fn matches(only: ?[]const u8) bool {
            const text = only orelse return true;
            var name: [160]u8 = undefined;
            for (std.enums.values(source.Kind)) |kind| for ([_]bool{ false, true }) |inverse| {
                if (kind != .rope and inverse) continue;
                for (0..5) |variant| {
                    const fault: ?source.Fault = if (variant == 0) null else @fromBackingInt(@intCast(variant - 1));
                    if (fault == .half_sum and kind != .glue) continue;
                    const label = labelInto(&name, kind, inverse, fault) catch unreachable;
                    if (std.mem.indexOf(u8, label, text) != null) return true;
                }
            };
            return false;
        }

        fn labelInto(buf: []u8, kind: source.Kind, inverse: bool, fault: ?source.Fault) ![]const u8 {
            const direction = if (kind != .rope) "" else if (inverse) " inverse" else " forward";
            return if (fault) |f| std.fmt.bufPrint(buf, "control: dsv41-native-rows {t}{s} {t}", .{ kind, direction, f }) else std.fmt.bufPrint(buf, "dsv41-native-rows {t}{s}", .{ kind, direction });
        }

        pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Self {
            const s = try a.create(Self);
            errdefer a.destroy(s);
            s.* = .{ .a = a };
            errdefer s.clean();
            var options = mtl.CompileOptions.mlx();
            options.math_functions = .precise;
            for (std.enums.values(source.Kind)) |kind| {
                const original = try mtl.Library.fromSource(device, source.raw(kind), options);
                defer original.deinit();
                s.old[@backingInt(kind)] = try mtl.Pipeline.init(device, original, source.entry(kind, true), false);
                for (0..5) |variant| {
                    const fault: ?source.Fault = if (variant == 0) null else @fromBackingInt(@intCast(variant - 1));
                    if (fault == .half_sum and kind != .glue) continue;
                    const text = try source.source(a, kind, fault);
                    defer a.free(text);
                    const lib = try mtl.Library.fromSource(device, text, options);
                    defer lib.deinit();
                    const pipeline = try mtl.Pipeline.init(device, lib, source.entry(kind, false), false);
                    s.pipelines[variant][@backingInt(kind)] = pipeline;
                    if (pipeline.simdWidth() != 32 or pipeline.maxThreads() < (if (kind == .glue) @as(usize, 256) else 128)) return error.UnservedNativeRowsPipeline;
                    if (variant == 0) std.debug.print("RESOURCE dsv41-native-rows {t}: maxThreads={d}, staticTG={d}\n", .{ kind, pipeline.maxThreads(), @import("rig.zig").tgMemory(pipeline) });
                    for ([_]bool{ false, true }) |inverse| {
                        if (kind != .rope and inverse) continue;
                        var label: [160]u8 = undefined;
                        const name = try a.dupe(u8, try labelInto(&label, kind, inverse, fault));
                        errdefer a.free(name);
                        if (only) |needle| if (std.mem.indexOf(u8, name, needle) == null) {
                            a.free(name);
                            continue;
                        };
                        var shapes: std.ArrayList(c.Shape) = .empty;
                        errdefer shapes.deinit(a);
                        const widths: []const usize = switch (kind) {
                            .rope => &.{ 8192, 16384, 32768 },
                            .glue => &.{512},
                            .publish => &.{528},
                        };
                        for (widths) |n| for (1..9) |rows| try shapes.append(a, .{ .rows = rows, .n = n, .k = if (kind == .publish) 528 else 512 });
                        const served = try shapes.toOwnedSlice(a);
                        errdefer a.free(served);
                        const p = try a.create(Probe);
                        errdefer a.destroy(p);
                        try s.probes.ensureUnusedCapacity(a, 1);
                        try kernels.ensureUnusedCapacity(a, 1);
                        p.* = .{ .rig = .{ .a = a, .device = device, .queue = queue }, .pipeline = pipeline, .old = s.old[@backingInt(kind)].?, .kind = kind, .inverse = inverse, .fault = fault, .spec = .{
                            .name = name,
                            .accum = if (kind == .glue) .f32 else .none,
                            .depth = depth,
                            .ops = if (kind == .glue) 12 else 3,
                            .out = if (kind == .rope) .bf16 else .raw,
                            .in_place = kind != .glue,
                            .replaces = source.entry(kind, true),
                            .nan = .{ .stated = switch (kind) {
                                .rope => "every rotated or preserved element keeps its independent float64 formula's class",
                                .glue => "packed non-finite inputs retain byte-exact per-slot glue encoding; storage bytes are not IEEE floats",
                                .publish => "every byte, including encoded NaN markers and untouched ring state, is preserved exactly",
                            } },
                            .served = served,
                            .tiles = .{ .{ .size = 4 }, .{}, .{} },
                            .sums = .{ .bf16 = 0, .why = "positive equal squares sum exactly in balanced BF16; packed FP8 hides small norm errors" },
                        } };
                        s.probes.appendAssumeCapacity(p);
                        kernels.appendAssumeCapacity(p.kernel());
                    }
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
            for (s.old) |pipeline| if (pipeline) |p| p.deinit();
            for (s.pipelines) |variant| for (variant) |pipeline| {
                if (pipeline) |p| p.deinit();
            };
        }

        pub fn destroy(s: *Self) void {
            s.clean();
            s.a.destroy(s);
        }
    };
}
