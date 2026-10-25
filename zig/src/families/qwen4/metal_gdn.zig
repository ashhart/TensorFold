//! Flash Next's maintained Metal recurrence with model-specific heads, F32 weight views and per-row kept states.
const std = @import("std");
const gdn = @import("cuda_gdn.zig");
const wo = @import("core").weight_only;
pub const Control = @import("gdn_control_source.zig").Control;
pub fn source(a: std.mem.Allocator, keys: usize, values: usize, gate: gdn.Gate) ![]u8 {
    return @import("../flashnext/hybrid_sources.zig").Gdn.metal(a, @embedFile("../../../kernels/metal/decode/fn_gdn.metal"), @embedFile("../../../kernels/metal/flashnext/q4_gdn_step_af87f500308a9afb.metal"), keys, values, gate == .sigmoid);
}
pub fn Kernels(comptime metal: type) type {
    return struct {
        const Self = @This();
        pub const View = struct { buffer: metal.Buffer, offset: usize = 0, len: usize };
        device: metal.Device,
        pipe: metal.Pipeline,
        keys: usize,
        values: usize,
        gate: gdn.Gate,
        control: Control = .production,
        pub fn load(device: metal.Device, text: []const u8, keys: usize, values: usize, gate: gdn.Gate) !Self {
            return loadControl(std.heap.page_allocator, device, text, keys, values, gate, .production);
        }
        pub fn loadControl(a: std.mem.Allocator, device: metal.Device, text: []const u8, keys: usize, values: usize, gate: gdn.Gate, control: Control) !Self {
            _ = try gdn.sizes(.{ .rows = 1, .keys = keys, .values = values, .eps = 1e-6 });
            if (values > std.math.maxInt(i32) / (16 * 514)) return error.Qwen4GdnShape;
            const controlled = try @import("gdn_control_source.zig").metal(a, text, control);
            defer a.free(controlled);
            const lib = try metal.Library.fromSource(device, controlled, metal.CompileOptions.mlx());
            defer lib.deinit();
            const pipe = try metal.Pipeline.init(device, lib, "qwen4_gdn_chain", false);
            errdefer pipe.deinit();
            if (pipe.maxThreads() < 1024 or pipe.simdWidth() != 32) return error.Qwen4GdnThreads;
            return .{ .device = device, .pipe = pipe, .keys = keys, .values = values, .gate = gate, .control = control };
        }
        /// The owner drains its queue before releasing a recurrence used by queued work.
        pub fn deinit(self: *Self) void {
            self.pipe.deinit();
            self.* = undefined;
        }
        pub fn encode(self: *Self, encoder: metal.ComputeEncoder, p: gdn.Params, views: [10]View) !void {
            return self.testEncode(encoder, p, views, 0);
        }
        pub fn testEncode(self: *Self, encoder: metal.ComputeEncoder, p: gdn.Params, views: [10]View, repeat: u32) !void {
            if (p.rows > 16 or p.weight_flags & 3 != 3 or p.keys != self.keys or p.values != self.values or p.gate != self.gate) return error.Qwen4GdnShape;
            const sizes = try gdn.sizes(p);
            if (encoder.deviceId() != self.device.id) return error.Qwen4GdnContext;
            const needs = [_]usize{ sizes.projection, sizes.conv_state, sizes.state, sizes.conv_weight, sizes.scalar, sizes.scalar, sizes.norm, sizes.output, try gdn.extent(p, sizes.conv_state), sizes.snapshots };
            var ranges: [10]wo.Range = undefined;
            for (views, needs, 0..) |view, need, i| {
                if (view.buffer.deviceId() != self.device.id) return error.Qwen4GdnContext;
                if (view.offset > view.buffer.length() or view.len > view.buffer.length() - view.offset or view.len < need) return error.Qwen4GdnBuffer;
                const address = try std.math.add(usize, @intCast(view.buffer.gpuAddress()), view.offset);
                const alignment: usize = if (i == 2 or i == 9) 16 else if (i == 0 or i == 1 or i == 7 or i == 8) 2 else 4;
                if (address == 0 or address % alignment != 0) return error.Qwen4GdnAlignment;
                _ = try std.math.add(usize, address, view.len);
                ranges[i] = .{ .address = address, .len = view.len };
            }
            for (7..10) |i| for (0..i) |j| {
                const x = ranges[i];
                const y = ranges[j];
                if (i == 9 and j == 8 and p.row_stride != 0) {
                    // Conv and recurrent outputs share complete frames, but their actual per-row writes are disjoint.
                    for (0..p.rows) |sr| for (0..p.rows) |cr| {
                        const sa = x.address + sr * p.row_stride;
                        const ca = y.address + cr * p.row_stride;
                        if (if (sa <= ca) ca - sa < sizes.state else sa - ca < sizes.conv_state) return error.Qwen4GdnAlias;
                    };
                    continue;
                }
                const overlap = if (x.address <= y.address) y.address - x.address < x.len else x.address - y.address < y.len;
                if (overlap) return error.Qwen4GdnAlias;
            };
            encoder.setPipeline(self.pipe);
            if (self.control == .repeat) @import("metal_control.zig").bind(encoder, repeat);
            for (views[0..7], 0..) |view, i| encoder.setBuffer(view.buffer, view.offset, i);
            encoder.setBytes(std.mem.asBytes(&p.eps), 7);
            const rows: c_int = @intCast(p.rows);
            encoder.setBytes(std.mem.asBytes(&rows), 8);
            for (views[7..], 9..) |view, i| encoder.setBuffer(view.buffer, view.offset, i);
            const head_offset: c_int = 0;
            encoder.setBytes(std.mem.asBytes(&head_offset), 12);
            const stride: u64 = @intCast(p.row_stride);
            encoder.setBytes(std.mem.asBytes(&stride), 14);
            const norm_round: u32 = if (p.weight_flags & 4 == 0) 1 else 0;
            encoder.setBytes(std.mem.asBytes(&norm_round), 15);
            const conv_round: u32 = @intFromBool(p.weight_flags & 8 != 0);
            encoder.setBytes(std.mem.asBytes(&conv_round), 16);
            encoder.dispatchGroups(metal.Size.of(p.values, 1, 1), metal.Size.of(1024, 1, 1));
            encoder.barrier();
        }
    };
}
