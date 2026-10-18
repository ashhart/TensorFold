/// Our Metal sources, compiled into one metallib that executables embed.
pub const kernels = [_][]const u8{
    "zig/kernels/metal/runtime_tests.metal",
    "zig/kernels/metal/nemotron_expert_up.metal",
    "zig/kernels/metal/nemotron_expert_down.metal",
};

/// Generated Nemotron kernels, also compiled under fast math as a control.
pub const generated = kernels[1..];

/// Kimi K3's kernels: one metallib, sharing kimi_common.h.
pub const kimi_dir = "zig/kernels/metal/kimi";
pub const kimi_kernels = [_][]const u8{ "dense.metal", "dense_mma.metal", "norms.metal", "experts.metal", "kda.metal", "mla.metal", "synth.metal" };

/// Packed kernels for compilers refusing uint4b_format; keep in step with tools/zig/gen_nemotron_kernels.py.
pub const nemotron_packed = [_][]const u8{
    "zig/kernels/metal/nemotron/coop_down_1_0.metal",
    "zig/kernels/metal/nemotron/coop_down_2_0.metal",
    "zig/kernels/metal/nemotron/coop_down_2_1.metal",
    "zig/kernels/metal/nemotron/coop_draft_1_0.metal",
    "zig/kernels/metal/nemotron/coop_draft_2_0.metal",
    "zig/kernels/metal/nemotron/coop_draft_2_1.metal",
    "zig/kernels/metal/nemotron/coop_eh_1_0.metal",
    "zig/kernels/metal/nemotron/coop_eh_1_0_sk.metal",
    "zig/kernels/metal/nemotron/coop_eh_2_0.metal",
    "zig/kernels/metal/nemotron/coop_eh_2_0_sk.metal",
    "zig/kernels/metal/nemotron/coop_eh_2_1.metal",
    "zig/kernels/metal/nemotron/coop_eh_2_1_sk.metal",
    "zig/kernels/metal/nemotron/coop_head_1_0.metal",
    "zig/kernels/metal/nemotron/coop_head_2_0.metal",
    "zig/kernels/metal/nemotron/coop_head_2_1.metal",
    "zig/kernels/metal/nemotron/coop_in_1_0.metal",
    "zig/kernels/metal/nemotron/coop_in_1_0_sk.metal",
    "zig/kernels/metal/nemotron/coop_in_2_0.metal",
    "zig/kernels/metal/nemotron/coop_in_2_0_sk.metal",
    "zig/kernels/metal/nemotron/coop_in_2_1.metal",
    "zig/kernels/metal/nemotron/coop_in_2_1_sk.metal",
    "zig/kernels/metal/nemotron/coop_out_1_0.metal",
    "zig/kernels/metal/nemotron/coop_out_1_0_sk.metal",
    "zig/kernels/metal/nemotron/coop_out_2_0.metal",
    "zig/kernels/metal/nemotron/coop_out_2_0_sk.metal",
    "zig/kernels/metal/nemotron/coop_out_2_1.metal",
    "zig/kernels/metal/nemotron/coop_out_2_1_sk.metal",
    "zig/kernels/metal/nemotron/coop_qkv_1_0.metal",
    "zig/kernels/metal/nemotron/coop_qkv_2_0.metal",
    "zig/kernels/metal/nemotron/coop_qkv_2_1.metal",
    "zig/kernels/metal/nemotron/coop_up_1_0.metal",
    "zig/kernels/metal/nemotron/coop_up_2_0.metal",
    "zig/kernels/metal/nemotron/coop_up_2_1.metal",
    "zig/kernels/metal/nemotron/up_relu2_1_0.metal",
    "zig/kernels/metal/nemotron/up_relu2_2_0.metal",
    "zig/kernels/metal/nemotron/up_relu2_2_1.metal",
};
