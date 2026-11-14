//! The production native learner pipelines run on guarded device buffers through the common primitive contracts.
const std = @import("std");
const metal = @import("metal");
const contract = @import("contract");
const F = @import("qwen4").family;
const tf = @import("tensorfold");
const rig_ = @import("rig.zig");
fn Group(comptime spec: type) type {
    const K = F.train_rows_metal.Adapter(metal, spec);
    const Backend = struct {
        pub const Rig = rig_.Rig;
        pub const probability_cut = false;
        kernels: *K,
        views: [11]K.View = undefined,
        pub fn bind(b: *@This(), r: *Rig, f: anytype) !void {
            const needs = try spec.needs(f.raw.op, f.raw.p);
            for (&b.views, needs.bytes, 0..) |*view, size, i| {
                const ref = if (i < 8) try r.input(f.raw.data[i][0..size]) else try r.output(size);
                view.* = .{ .buffer = ref.buf, .offset = ref.off, .len = size };
            }
        }
        pub fn launch(b: *@This(), r: *Rig, f: anytype) !void {
            for (b.views[8..]) |view| @memset(view.buffer.contents()[view.offset..][0..view.len], 0);
            const pass = r.begin();
            b.kernels.encode(pass.e, f.raw.op, f.raw.p, b.views) catch |err| {
                try r.finish(pass);
                return err;
            };
            try r.finish(pass);
            for (b.views[8..], 8..) |view, i| @memcpy(f.raw.data[i][0..view.len], view.buffer.contents()[view.offset..][0..view.len]);
        }
    };
    const Case = F.learning_contract.Case(Backend, spec);
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        kernels: [5]?K = @splat(null),
        cases: [spec.names.len * 10]Case = undefined,
        made: usize = 0,
        pub fn create(a: std.mem.Allocator, device: metal.Device, queue: metal.Queue, out: *std.ArrayList(contract.gates.Kernel), only: ?[]const u8) !*Self {
            const group = try a.create(Self);
            errdefer a.destroy(group);
            group.* = .{ .a = a };
            errdefer group.cleanup();
            for (std.enums.values(tf.living_rows.Control)) |control| for (std.enums.values(spec.Op), spec.names) |op, name| for (0..2) |variant| {
                if (variant == 1 and !(spec == tf.living_gdn and op == .readout)) continue;
                if (control == .sums and !F.learning_contract_data.reduces(spec, op)) continue;
                const label = try std.fmt.allocPrint(a, "metal living-native {s}{s}{s}", .{ name, tf.living_rows.suffixes[@backingInt(control)], if (variant == 1) " sigmoid" else "" });
                if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) {
                    a.free(label);
                    continue;
                };
                errdefer a.free(label);
                const index = @backingInt(control);
                if (group.kernels[index] == null) group.kernels[index] = try K.loadTest(a, device, control);
                const c = &group.cases[group.made];
                c.* = .{ .a = a, .backend = .{ .kernels = &group.kernels[index].? }, .rig = .{ .base = .{ .a = a, .device = device, .queue = queue } }, .op = op, .control = control, .sigmoid = variant == 1, .label = label };
                try out.append(a, c.kernel());
                group.made += 1;
            };
            return group;
        }
        fn cleanup(g: *Self) void {
            for (g.cases[0..g.made]) |*c| {
                g.a.free(c.label);
                c.deinit();
            }
            for (&g.kernels) |*k| if (k.*) |*value| value.deinitDrained();
        }
        pub fn destroy(g: *Self) void {
            g.cleanup();
            g.a.destroy(g);
        }
    };
}
pub const Rows = Group(tf.living_rows);
pub const Gdn = Group(tf.living_gdn);
pub const Qsa = Group(tf.living_qsa);
