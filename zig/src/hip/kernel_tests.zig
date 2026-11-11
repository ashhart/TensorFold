//! The model-free kernels on a real GPU: every kernel resolves, and each launch matches its host reference.
const std = @import("std");

/// Flash Next's model-free side: its config, caches, window plan and mixers, on this module as `hip`.
pub fn flashnext() type {
    return flashnext_mod;
}
const flashnext_mod = @import("flashnext_kernel");
pub const cache = flashnext_mod.cache;
pub const gdn_replay = flashnext_mod.gdn_replay;
pub const forward = flashnext_mod.forward;
pub const mixers = flashnext_mod.mixers;
pub const model = flashnext_mod.model;

const hipmod = @import("root.zig");
pub const Launcher = hipmod.Launcher;
pub const Runtime = hipmod.Runtime;
pub const Context = hipmod.Context;
pub const kernels = hipmod.kernels;
pub const raw = hipmod.raw;
pub const usage = hipmod.usage;
pub const ops = hipmod.ops;
pub const plan_ops = hipmod.plan_ops;
pub const counting = hipmod.counting;
pub const moe = hipmod.moe;
pub const collective = hipmod.collective;
pub const abi = hipmod.abi;
pub const runtime = hipmod.runtime;
pub const Stream = hipmod.Stream;
pub const DeviceBuffer = hipmod.DeviceBuffer;
pub const HostBuffer = hipmod.HostBuffer;

test "the code objects load on this device and every kernel the launchers use resolves" {
    var r = try Runtime.open();
    defer r.close();
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    var arch_buffer: [256]u8 = undefined;
    const arch = try @import("device_arch.zig").query(&r, 0, &arch_buffer);
    try std.testing.expectEqualStrings(kernels.arch, arch);
    var l = try Launcher.load(&r, .{}, kernels.images);
    defer l.unload();
}

test {
    _ = @import("kernel_tests/gdn.zig");
    _ = @import("kernel_tests/gdn_bits.zig");
    _ = @import("kernel_tests/gdn_plan.zig");
    _ = @import("kernel_tests/gdn_window.zig");
    _ = @import("kernel_tests/mamba.zig");
    _ = @import("kernel_tests/decode.zig");
    _ = @import("kernel_tests/tail_pin.zig");
    _ = @import("kernel_tests/chain.zig");
    _ = @import("kernel_tests/attention.zig");
    _ = @import("kernel_tests/attention_pin.zig");
    _ = @import("kernel_tests/attention_edge.zig");
    _ = @import("kernel_tests/stats.zig");
    _ = @import("kernel_tests/tap.zig");
    _ = @import("kernel_tests/draft.zig");
    _ = @import("kernel_tests/ops.zig");
    _ = @import("kernel_tests/conv_split.zig");
    _ = @import("kernel_tests/dense.zig");
    _ = @import("kernel_tests/silu_fuse.zig");
    _ = @import("kernel_tests/argmax.zig");
    _ = @import("kernel_tests/embed.zig");
    _ = @import("kernel_tests/tp.zig");
    _ = @import("kernel_tests/affine/affine.zig");
    _ = @import("kernel_tests/affine/rows.zig");
    _ = @import("kernel_tests/affine/order.zig");
    _ = @import("kernel_tests/affine/round.zig");
    _ = @import("kernel_tests/affine/layout.zig");
    _ = @import("kernel_tests/affine/golden.zig");
    _ = @import("kernel_tests/graph_split.zig");
    _ = @import("kernel_tests/flashnext.zig");
    _ = @import("kernel_tests/flashnext_attn.zig");
    _ = @import("kernel_tests/moe_gate.zig");
    _ = @import("kernel_tests/flashnext_sparse.zig");
    _ = @import("kernel_tests/nemotron/moe.zig");
    _ = @import("kernel_tests/nemotron/pick.zig");
    _ = @import("kernel_tests/nemotron/attention.zig");
    _ = @import("kernel_tests/nemotron/fixture.zig");
    _ = @import("kernel_tests/nemotron/fused.zig");
    _ = @import("kernel_tests/nemotron/window.zig");
    _ = @import("kernel_tests/nemotron/epilogue.zig");
}
