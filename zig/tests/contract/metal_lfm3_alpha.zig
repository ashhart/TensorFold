//! Finite rescaling-underflow coverage at a served shape under the unchanged production attention bound.
const std = @import("std");
const mtl = @import("metal");
const ct = @import("contract");
const base = @import("metal_lfm3.zig");
const rig = @import("rig.zig");
const sources = @import("kernel_sources");
const c = ct.spec;
const g = ct.gates;

pub const name = "metal LFM3 attention alpha underflow";
const Mode = enum { alpha, product, overflow, nonfinite_weight };
pub const Probe = Witness(.alpha, .none);
pub const ProductProbe = Witness(.product, .none);
pub const product_name = "metal LFM3 attention product underflow";
pub const OverflowProbe = Witness(.overflow, .none);
pub const NonfiniteWeightProbe = Witness(.nonfinite_weight, .none);
pub const overflow_name = "metal LFM3 attention overflow history";
pub const nonfinite_weight_name = "metal LFM3 attention nonfinite weight";
const Fault = enum { none, skip, clear };
pub const SkipProbe = Witness(.alpha, .skip);
pub const ClearProbe = Witness(.product, .clear);
pub const skip_name = "metal LFM3 attention alpha skip-repair control";
pub const clear_name = "metal LFM3 attention product clear-risk control";
fn Witness(comptime mode: Mode, comptime fault: Fault) type {
    return struct {
        const Self = @This();
        inner: base.Probe,
        const vt = blk: {
            var hooks = g.hooks(base.Probe);
            hooks.prepare = prepare;
            break :blk hooks;
        };
        fn parent(ctx: *anyopaque) *Self {
            const inner: *base.Probe = @ptrCast(@alignCast(ctx));
            return @fieldParentPtr("inner", inner);
        }
        pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Self {
            var inner = try base.Probe.init(a, device, queue, .attention_full);
            errdefer inner.deinit();
            if (fault != .none) {
                const define = if (fault == .skip) "#define L3_RANGE_SKIP_REPAIR\n" else "#define L3_RANGE_CLEAR_RISK\n";
                const source = try std.mem.concat(a, u8, &.{ define, sources.lfm3, "\n", sources.lfm3_attention });
                defer a.free(source);
                const partial = try rig.pipeline(device, source, "lfm3_attn_partial");
                errdefer partial.deinit();
                const merge = try rig.pipeline(device, source, "lfm3_attn_merge");
                inner.pipeline.deinit();
                inner.merge.?.deinit();
                inner.pipeline = partial;
                inner.merge = merge;
            }
            return .{ .inner = inner };
        }
        pub fn deinit(p: *Self) void {
            p.inner.deinit();
        }
        pub fn kernel(p: *Self) g.Kernel {
            var out = p.inner.kernel();
            out.vt = &vt;
            out.spec.name = switch (mode) {
                .alpha => name,
                .product => product_name,
                .overflow => overflow_name,
                .nonfinite_weight => nonfinite_weight_name,
            };
            if (fault != .none) {
                out.spec.name = if (fault == .skip) skip_name else clear_name;
                out.control = .shapes;
            }
            out.spec.served = &.{.{ .rows = 17, .n = 2048, .k = 1025 }};
            return out;
        }
        fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            const b = &parent(ctx).inner;
            if (!try b.prepare(shape, fill, seed)) return false;
            if (fill == .nonfinite) return true;
            const cap = shape.k + shape.rows;
            if (cap < 64) return false;
            @memset(b.data.x, 0);
            @memset(b.data.w, 0);
            @memset(b.data.z, 0);
            if (mode == .overflow or mode == .nonfinite_weight) {
                for (0..shape.rows) |row| for (0..16) |head| {
                    b.data.x[(row * 16 + head) * 128] = c.toBf16(1);
                };
                for (0..4) |head| {
                    if (mode == .overflow) {
                        for (16..cap) |key| b.data.w[(head * cap + key) * 128] = c.toBf16(64);
                        for (0..16) |key| @memset(b.data.z[(head * cap + key) * 128 ..][0..128], c.toBf16(0x1p125));
                    } else {
                        b.data.w[head * cap * 128] = c.toBf16(-1024);
                        @memset(b.data.z[head * cap * 128 ..][0..128], c.toBf16(std.math.inf(f32)));
                    }
                }
            } else if (mode == .product) {
                for (0..shape.rows) |row| for (0..16) |head| {
                    b.data.x[(row * 16 + head) * 128] = c.toBf16(1);
                };
                for (0..4) |head| for (1..cap) |key| {
                    b.data.w[(head * cap + key) * 128] = c.toBf16(-160);
                    @memset(b.data.z[(head * cap + key) * 128 ..][0..128], c.toBf16(0x1p-110));
                };
            } else {
                for (0..shape.rows) |row| for (0..16) |head| {
                    b.data.x[(row * 16 + head) * 128] = c.toBf16(@floatFromInt((row + 1) * (head + 1)));
                };
                for (0..4) |head| {
                    for (32..64) |key| b.data.w[(head * cap + key) * 128] = c.toBf16(8);
                    for (0..32) |key| @memset(b.data.z[(head * cap + key) * 128 ..][0..128], c.toBf16(1));
                }
            }
            for ([_][]const u16{ b.data.x, b.data.w, b.data.z }, 0..) |values, i|
                @memcpy(b.rig.bytes(b.args[i], values.len * 2), std.mem.sliceAsBytes(values));
            return true;
        }
    };
}
