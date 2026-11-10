//! Prompt-contract owners stay alive through every registered gate and release in their original reverse order.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const nvfp4_prompt_k = @import("contract_nvfp4_prompt.zig");
const fp8_prompt_k = @import("contract_fp8_prompt.zig");

pub const Cases = struct {
    compact_attention: @import("contract_attention_tuned.zig").Cases = .{},
    fp8_cases: fp8_prompt_k.Cases = .{},
    nvfp4_tma_cases: @import("contract_nvfp4_tma.zig").Cases = .{},
    nvfp4_prompt_cases: [nvfp4_prompt_k.entries.len]nvfp4_prompt_k.Case = undefined,
    nvfp4_prompt_count: usize = 0,

    pub fn append(self: *Cases, kernels: *std.ArrayList(g.Kernel), gpu: check.Gpu, s: cuda.Stream, cap: u32) !void {
        const a = gpu.gpa;
        try self.compact_attention.append(kernels, gpu, s, @intCast(cap), cuda.kernels.prefill_attention);
        try self.fp8_cases.append(kernels, gpu, s, cap);
        try self.nvfp4_tma_cases.append(kernels, gpu, s, cap);
        if (cuda.nvfp4.native_prompt.supported(cap)) {
            for (&self.nvfp4_prompt_cases, nvfp4_prompt_k.entries) |*item, config| {
                item.* = .{ .rig = .{ .gpu = gpu, .s = s }, .config = config };
                self.nvfp4_prompt_count += 1;
                try kernels.append(a, item.kernel());
            }
        }
    }

    pub fn deinit(self: *Cases) void {
        for (self.nvfp4_prompt_cases[0..self.nvfp4_prompt_count]) |*item| item.deinit();
        self.nvfp4_tma_cases.deinit();
        self.fp8_cases.deinit();
        self.compact_attention.deinit();
    }
};
