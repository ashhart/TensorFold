//! The EXL3 projection and chain, DSV41 native rows and batched attention rows: this branch's suites, created together.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const g = @import("contract").gates;
const Projection = @import("metal_exl3_projection.zig").Suite;
const Chain = @import("metal_exl3_chain.zig").For(tf.exl3_projection);
const Rows = @import("metal_dsv41_native_rows.zig").For(tf.deepseek_v41);
const Attention = @import("metal_dsv41_attention_rows.zig").For(tf.deepseek_v41, tf.frags);

/// Each suite is created only when `only` names it, in the contract's order; `deinit` frees them in reverse.
pub const Suites = struct {
    projection: ?*Projection = null,
    chain: ?*Chain = null,
    rows: ?*Rows = null,
    attention: ?*Attention = null,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8, tensor: bool) !Suites {
        var s: Suites = .{};
        errdefer s.deinit();
        if (Projection.matches(only)) s.projection = try Projection.create(a, device, queue, kernels, only);
        if (Chain.matches(only)) s.chain = try Chain.create(a, device, queue, kernels, only);
        if (Rows.matches(only)) s.rows = try Rows.create(a, device, queue, kernels, only);
        if (!Attention.matches(only)) return s;
        if (tensor) s.attention = try Attention.create(a, device, queue, kernels, only) else std.debug.print("SKIP dsv41-native-attention: native batched scan requires tensor units\n", .{});
        return s;
    }

    pub fn deinit(s: *Suites) void {
        if (s.attention) |x| x.destroy();
        if (s.rows) |x| x.destroy();
        if (s.chain) |x| x.destroy();
        if (s.projection) |x| x.destroy();
        s.* = .{};
    }
};
