//! The native decode-row dense table and tensor-unit scan, including both existing split-merge launch forms.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const source = @import("dsv41_attention_rows_source.zig");
const ref = @import("dsv41_attention_rows_ref.zig");
const gpu = @import("dsv41_attention_rows_probe.zig");
const c = contract.spec;
const g = contract.gates;
fn depth(_: c.Shape) usize {
    return 1320;
}

fn nameInto(buf: []u8, kind: source.Kind, config: ref.Config, fault: ?source.Fault) ![]const u8 {
    return if (fault) |f| std.fmt.bufPrint(buf, "control: dsv41-native-attention {t} r{d} s{d} o{d} {t}", .{ kind, config.ratio, config.splits, config.owner, f }) else std.fmt.bufPrint(buf, "dsv41-native-attention {t} r{d} s{d} o{d}", .{ kind, config.ratio, config.splits, config.owner });
}

pub fn For(comptime native: type, comptime frags: type) type {
    return struct {
        const Self = @This();
        const Probe = gpu.For(native);
        a: std.mem.Allocator,
        old: ?gpu.Old = null,
        owned: std.ArrayList(mtl.Pipeline) = .empty,
        probes: std.ArrayList(*Probe) = .empty,

        pub fn matches(only: ?[]const u8) bool {
            const text = only orelse return true;
            var name: [192]u8 = undefined;
            for (std.enums.values(source.Kind)) |kind| for (ref.configs) |config| for (0..(std.enums.values(source.Fault).len + 1)) |variant| {
                const fault: ?source.Fault = if (variant == 0) null else @fromBackingInt(@intCast(variant - 1));
                if (kind == .dense and (fault == .half_score or fault == .half_value or fault == .split_tail)) continue;
                if (fault == .split_tail and config.ratio != 2) continue;
                const label = nameInto(&name, kind, config, fault) catch unreachable;
                if (std.mem.indexOf(u8, label, text) != null) return true;
            };
            return false;
        }
        fn own(s: *Self, device: mtl.Device, lib: mtl.Library, entry: []const u8) !mtl.Pipeline {
            const pipeline = try mtl.Pipeline.init(device, lib, entry, false);
            errdefer pipeline.deinit();
            try s.owned.append(s.a, pipeline);
            return pipeline;
        }
        pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Self {
            if (!device.tensorUnits()) return error.NativeAttentionRowsRequireTensorUnits;
            const s = try a.create(Self);
            errdefer a.destroy(s);
            s.* = .{ .a = a };
            errdefer s.clean();
            var options = mtl.CompileOptions.mlx();
            options.math_functions = .precise;
            const clean_text = try source.text(frags, device, a, .dense, null);
            defer a.free(clean_text);
            const clean_lib = try mtl.Library.fromSource(device, clean_text, options);
            defer clean_lib.deinit();
            s.old = try gpu.Old.load(device, clean_lib);
            const dense = try s.own(device, clean_lib, source.entry(.dense));
            for (std.enums.values(source.Kind)) |kind| for (0..(std.enums.values(source.Fault).len + 1)) |variant| {
                const fault: ?source.Fault = if (variant == 0) null else @fromBackingInt(@intCast(variant - 1));
                if (kind == .dense and (fault == .half_score or fault == .half_value or fault == .split_tail)) continue;
                const mutant_text = if (fault != null) try source.text(frags, device, a, kind, fault) else null;
                defer if (mutant_text) |text| a.free(text);
                const lib = if (mutant_text) |text| try mtl.Library.fromSource(device, text, options) else clean_lib;
                defer if (mutant_text != null) lib.deinit();
                const pipeline = if (kind == .dense and fault == null) dense else try s.own(device, lib, source.entry(kind));
                if (pipeline.simdWidth() != 32 or pipeline.maxThreads() < 128) return error.UnservedNativeAttentionRowsPipeline;
                if (variant == 0) std.debug.print("RESOURCE dsv41-native-attention {t}: maxThreads={d}, staticTG={d}\n", .{ kind, pipeline.maxThreads(), @import("rig.zig").tgMemory(pipeline) });
                const merge = if (fault == .half_value) try s.own(device, lib, "dsv41_sparse_merge") else s.old.?.merge;
                const merge_split = if (fault == .half_value) try s.own(device, lib, "dsv41_sparse_merge_split") else s.old.?.merge_split;
                for (ref.configs) |config| {
                    if (fault == .split_tail and config.ratio != 2) continue;
                    var label: [192]u8 = undefined;
                    const name = try a.dupe(u8, try nameInto(&label, kind, config, fault));
                    errdefer a.free(name);
                    if (only) |needle| if (std.mem.indexOf(u8, name, needle) == null) {
                        a.free(name);
                        continue;
                    };
                    var shapes: std.ArrayList(c.Shape) = .empty;
                    errdefer shapes.deinit(a);
                    const widths: []const usize = if (kind == .dense) &.{512} else &.{ 8192, 16384, 32768 };
                    for (widths) |n| for (1..9) |rows| try shapes.append(a, .{ .rows = rows, .n = n, .k = 512 });
                    const served = try shapes.toOwnedSlice(a);
                    errdefer a.free(served);
                    const p = try a.create(Probe);
                    errdefer a.destroy(p);
                    try s.probes.ensureUnusedCapacity(a, 1);
                    try kernels.ensureUnusedCapacity(a, 1);
                    p.* = .{ .rig = .{ .a = a, .device = device, .queue = queue }, .pipeline = pipeline, .dense_pipeline = dense, .merge = if (config.merge_split) merge_split else merge, .old = &s.old.?, .kind = kind, .config = config, .fault = fault, .spec = .{
                        .name = name,
                        .accum = if (kind == .dense) .none else .f32,
                        .depth = depth,
                        .ops = 12,
                        .out = .bf16,
                        .in_place = kind == .scan,
                        .replaces = if (kind == .dense) "per-visible-row dsv41_dense_kv" else "per-slot dsv41_sparse_nax and native split merge",
                        .nan = .{ .stated = if (kind == .dense) "independent packed-code formulas preserve each finite, NaN and infinity class; unused dense rows retain poison" else "NaN and -Inf query scores are masked; +Inf scores produce NaN probabilities; finite positive keys retain positive attention" },
                        .served = served,
                        .tiles = .{ .{ .size = 4 }, .{}, .{} },
                    } };
                    s.probes.appendAssumeCapacity(p);
                    kernels.appendAssumeCapacity(p.kernel());
                }
            };
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
            for (s.owned.items) |pipeline| pipeline.deinit();
            s.owned.deinit(s.a);
            if (s.old) |old| old.deinit();
        }
        pub fn destroy(s: *Self) void {
            s.clean();
            s.a.destroy(s);
        }
    };
}
