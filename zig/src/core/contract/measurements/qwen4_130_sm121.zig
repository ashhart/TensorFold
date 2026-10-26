//! nvcc 13.0 live resources on sm_121 only; absent targets and independent later compilers remain pending.
const std = @import("std");
const limits = @import("../limits.zig");
const Row = struct { name: []const u8, usage: limits.Usage };

// Live contract receipt SHA-256: 82ba3746a61fabe186af4edd4d0e84e7e5adb58c7b9c6c054254f4290e4e8fc3
pub const records = [_]Row{
    .{ .name = "qwen4_projection tf_qwen4_projection", .usage = .{ .regs = 40, .static = 0, .bound = 128 } },
    .{ .name = "qwen4_projection tf_qwen4_projection_bad_nan", .usage = .{ .regs = 40, .static = 0, .bound = 128 } },
    .{ .name = "qwen4_projection tf_qwen4_projection_bad_repeat", .usage = .{ .regs = 40, .static = 0, .bound = 128 } },
    .{ .name = "qwen4_projection tf_qwen4_projection_bad_tail", .usage = .{ .regs = 38, .static = 0, .bound = 128 } },
    .{ .name = "qwen4_projection tf_qwen4_projection_bad_sums", .usage = .{ .regs = 40, .static = 0, .bound = 128 } },
    .{ .name = "qwen4_gdn qwen4_gdn_chain", .usage = .{ .regs = 61, .static = 2060, .bound = 1024 } },
    .{ .name = "qwen4_hc qwen4_hc_activation", .usage = .{ .regs = 17, .static = 0, .bound = 256 } },
    .{ .name = "qwen4_hc qwen4_hc_activation_bad_nan", .usage = .{ .regs = 17, .static = 0, .bound = 256 } },
    .{ .name = "qwen4_hc qwen4_hc_activation_bad_repeat", .usage = .{ .regs = 17, .static = 0, .bound = 256 } },
    .{ .name = "qwen4_hc qwen4_hc_activation_bad_tail", .usage = .{ .regs = 17, .static = 0, .bound = 256 } },
    .{ .name = "qwen4_hc qwen4_hc_pointwise", .usage = .{ .regs = 26, .static = 0, .bound = 256 } },
    .{ .name = "qwen4_hc qwen4_hc_pointwise_bad_nan", .usage = .{ .regs = 26, .static = 0, .bound = 256 } },
    .{ .name = "qwen4_hc qwen4_hc_pointwise_bad_repeat", .usage = .{ .regs = 26, .static = 0, .bound = 256 } },
    .{ .name = "qwen4_hc qwen4_hc_pointwise_bad_tail", .usage = .{ .regs = 26, .static = 0, .bound = 256 } },
    .{ .name = "qwen4_hc qwen4_dense_act", .usage = .{ .regs = 18, .static = 0, .bound = 256 } },
    .{ .name = "qwen4_hc qwen4_dense_act_bad_nan", .usage = .{ .regs = 18, .static = 0, .bound = 256 } },
    .{ .name = "qwen4_hc qwen4_dense_act_bad_repeat", .usage = .{ .regs = 18, .static = 0, .bound = 256 } },
    .{ .name = "qwen4_hc qwen4_dense_act_bad_tail", .usage = .{ .regs = 18, .static = 0, .bound = 256 } },
    .{ .name = "qwen4_ple qwen4_ple_gate", .usage = .{ .regs = 37, .static = 128, .bound = 512 } },
    .{ .name = "qwen4_ple qwen4_ple_gate_bad_nan", .usage = .{ .regs = 37, .static = 128, .bound = 512 } },
    .{ .name = "qwen4_ple qwen4_ple_gate_bad_repeat", .usage = .{ .regs = 38, .static = 128, .bound = 512 } },
    .{ .name = "qwen4_ple qwen4_ple_gate_bad_tail", .usage = .{ .regs = 38, .static = 128, .bound = 512 } },
    .{ .name = "qwen4_ple qwen4_ple_gate_bad_sums", .usage = .{ .regs = 37, .static = 128, .bound = 512 } },
    .{ .name = "qwen4_ple qwen4_ple_conv", .usage = .{ .regs = 35, .static = 0, .bound = 512 } },
    .{ .name = "qwen4_ple qwen4_ple_conv_bad_nan", .usage = .{ .regs = 35, .static = 0, .bound = 512 } },
    .{ .name = "qwen4_ple qwen4_ple_conv_bad_repeat", .usage = .{ .regs = 35, .static = 0, .bound = 512 } },
    .{ .name = "qwen4_ple qwen4_ple_conv_bad_tail", .usage = .{ .regs = 8, .static = 0, .bound = 512 } },
    .{ .name = "qwen4_ple qwen4_ple_conv_bad_sums", .usage = .{ .regs = 35, .static = 0, .bound = 512 } },
    .{ .name = "qwen4_gdn qwen4_gdn_chain_bad_nan", .usage = .{ .regs = 61, .static = 2060, .bound = 1024 } },
    .{ .name = "qwen4_gdn qwen4_gdn_chain_bad_repeat", .usage = .{ .regs = 61, .static = 2060, .bound = 1024 } },
    .{ .name = "qwen4_gdn qwen4_gdn_chain_bad_tail", .usage = .{ .regs = 62, .static = 2060, .bound = 1024 } },
    .{ .name = "qwen4_gdn qwen4_gdn_chain_bad_sums", .usage = .{ .regs = 61, .static = 2060, .bound = 1024 } },
};

pub fn usage(comptime name: []const u8) [limits.sms.len]limits.Usage {
    @setEvalBranchQuota(100000);
    const measured = comptime for (records) |row| {
        if (std.mem.eql(u8, row.name, name)) break row.usage;
    } else @compileError("No sm_121 nvcc 13.0 Qwen4 receipt for " ++ name);
    var out: [limits.sms.len]limits.Usage = @splat(.{ .regs = 0, .static = 0 });
    for (limits.sms, 0..) |sm, i| if (sm.sm == 121) {
        out[i] = measured;
    };
    return out;
}
