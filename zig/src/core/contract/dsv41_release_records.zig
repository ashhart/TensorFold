//! DeepSeek V4.1's release-format CUDA launches (release_proj.cu, release_glue.cu): sm_121a images only.
const registry = @import("registry.zig");
const U = @import("limits.zig").Usage;
const off: U = .{ .regs = 0, .static = 0, .bound = 0 };

fn on121(regs: u32, static: u32, bound: u32) [5]U {
    return .{ off, off, off, off, .{ .regs = regs, .static = static, .bound = bound } };
}

fn launch(comptime image: []const u8, comptime symbol: []const u8, threads: u32, dynamic: u32) @import("limits.zig").CudaLaunch {
    return .{ .name = image ++ " " ++ symbol, .symbol = symbol, .image = image, .threads = threads, .dynamic = dynamic, .opt_in = dynamic > 48 * 1024, .min_sm = 121 };
}

/// nvcc 13.0's usage, the compiler of the release fatbins.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = launch("dsv41_rel", "tf_dsv41_rel_fp8", 64, 41472), .usage = on121(56, 0, 64) },
    .{ .launch = launch("dsv41_rel", "tf_dsv41_rel_fp8_mid", 128, 33792), .usage = on121(80, 0, 128) },
    .{ .launch = launch("dsv41_rel", "tf_dsv41_rel_fp8_prompt", 512, 99840), .usage = on121(127, 0, 512) },
    .{ .launch = launch("dsv41_rel", "tf_dsv41_rel_bf16", 64, 40960), .usage = on121(56, 0, 64) },
    .{ .launch = launch("dsv41_rel", "tf_dsv41_rel_bf16_mid", 128, 32768), .usage = on121(64, 0, 128) },
    .{ .launch = launch("dsv41_rel", "tf_dsv41_rel_bf16_prompt", 512, 98304), .usage = on121(128, 0, 512) },
    .{ .launch = launch("dsv41_rel", "tf_dsv41_rel_fp8_warp", 128, 0), .usage = on121(78, 0, 128) },
    .{ .launch = launch("dsv41_rel", "tf_dsv41_rel_bf16_warp", 128, 0), .usage = on121(58, 0, 128) },
    .{ .launch = launch("dsv41_rel", "tf_dsv41_rel_sum", 256, 0), .usage = on121(40, 0, 256) },
    .{ .launch = launch("dsv41_rel_ex", "tf_dsv41_rel_group", 1024, 4096), .usage = on121(32, 0, 1024) },
    .{ .launch = launch("dsv41_rel_ex", "tf_dsv41_rel_gateup", 32, 26368), .usage = on121(128, 0, 32) },
    .{ .launch = launch("dsv41_rel_ex", "tf_dsv41_rel_gateup_prompt", 128, 69632), .usage = on121(149, 0, 128) },
    .{ .launch = launch("dsv41_rel_ex", "tf_dsv41_rel_gateup_warp", 128, 0), .usage = on121(128, 0, 128) },
    .{ .launch = launch("dsv41_rel_ex", "tf_dsv41_rel_down", 32, 17664), .usage = on121(75, 0, 32) },
    .{ .launch = launch("dsv41_rel_ex", "tf_dsv41_rel_down_prompt", 128, 52224), .usage = on121(96, 0, 128) },
    .{ .launch = launch("dsv41_rel_ex", "tf_dsv41_rel_down_warp", 128, 0), .usage = on121(96, 0, 128) },
    .{ .launch = launch("dsv41_rel_ex", "tf_dsv41_rel_combine", 256, 0), .usage = on121(38, 0, 256) },
    .{ .launch = launch("dsv41_rel_ex", "tf_dsv41_rel_swiglu_actq", 256, 0), .usage = on121(26, 0, 256) },
    .{ .launch = launch("dsv41_rel_glue", "tf_dsv41_rel_actq", 256, 0), .usage = on121(28, 0, 256) },
    .{ .launch = launch("dsv41_rel_glue", "tf_dsv41_rel_rms_actq", 256, 0), .usage = on121(40, 32, 256) },
};

/// nvcc 13.5's usage where it differs from 13.0's (the build's compiler on the GB10 hosts).
pub const cuda_135 = [_]registry.Override{
    .{ .name = "dsv41_rel tf_dsv41_rel_fp8", .usage = on121(71, 0, 64) },
    .{ .name = "dsv41_rel tf_dsv41_rel_fp8_mid", .usage = on121(118, 0, 128) },
    .{ .name = "dsv41_rel tf_dsv41_rel_bf16", .usage = on121(64, 0, 64) },
    .{ .name = "dsv41_rel tf_dsv41_rel_bf16_mid", .usage = on121(72, 0, 128) },
    .{ .name = "dsv41_rel tf_dsv41_rel_bf16_prompt", .usage = on121(127, 0, 512) },
    .{ .name = "dsv41_rel tf_dsv41_rel_fp8_warp", .usage = on121(76, 0, 128) },
    .{ .name = "dsv41_rel tf_dsv41_rel_bf16_warp", .usage = on121(64, 0, 128) },
    .{ .name = "dsv41_rel_ex tf_dsv41_rel_gateup", .usage = on121(104, 0, 32) },
    .{ .name = "dsv41_rel_ex tf_dsv41_rel_gateup_prompt", .usage = on121(152, 0, 128) },
    .{ .name = "dsv41_rel_ex tf_dsv41_rel_down", .usage = on121(77, 0, 32) },
    .{ .name = "dsv41_rel_glue tf_dsv41_rel_actq", .usage = on121(26, 0, 256) },
    .{ .name = "dsv41_rel_glue tf_dsv41_rel_rms_actq", .usage = on121(46, 32, 256) },
};

test "release launches: sm_121 only, every record applicable there and nowhere else" {
    const t = @import("std").testing;
    const limits = @import("limits.zig");
    for (cuda) |r| for (limits.sms, r.usage) |sm, u| {
        try t.expectEqual(sm.sm == 121, registry.targetDecision(r, sm, u) == .applicable);
        try t.expectEqual(sm.sm == 121, u.bound > 0);
    };
    try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &cuda, true));
    for (cuda_135) |o| {
        const r = for (cuda) |x| {
            if (@import("std").mem.eql(u8, x.launch.name, o.name)) break x;
        } else return error.UnregisteredOverride;
        try t.expectEqual(r.usage[4].bound, o.usage[4].bound);
    }
}
