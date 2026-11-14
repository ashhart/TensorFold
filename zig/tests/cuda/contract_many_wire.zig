//! The three many-projection prompt registrations wired in one place, so `contract_tests` stays under its line limit.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const gpu_check = @import("check.zig");
const prefill_many_k = @import("contract_qmm_prefill_many.zig");
const prefill_tile22_k = @import("contract_qmm_prefill_tile22.zig");
const prompt_many_k = @import("contract_prompt_many.zig");

const g = core.contract.gates;

/// The registrations, held by the caller until every appended launch has run.
pub const Keep = struct {
    prefill_many: prefill_many_k.Registrations,
    prefill_tile22: prefill_tile22_k.Registrations,
    prompt_many: prompt_many_k.Registrations,

    pub fn deinit(self: *Keep) void {
        self.prompt_many.deinit();
        self.prefill_tile22.deinit();
        self.prefill_many.deinit();
    }
};

/// Appends the three registrations into `keep`: a pointer, since each Product is the ctx of a kernel appended here.
pub fn wire(gpu: gpu_check.Gpu, s: cuda.Stream, cap: u32, a: std.mem.Allocator, affine: *cuda.qlinear.Affine4Kernels, prompt: *cuda.nvfp4.Prompt, kernels: *std.ArrayList(g.Kernel), keep: *Keep) !void {
    keep.* = .{
        .prefill_many = prefill_many_k.Registrations.init(gpu, s, affine),
        .prefill_tile22 = prefill_tile22_k.Registrations.init(gpu, s, affine),
        .prompt_many = prompt_many_k.Registrations.init(gpu, s, prompt),
    };
    try keep.prefill_many.append(cap, a, kernels);
    try keep.prefill_tile22.append(cap, a, kernels);
    try keep.prompt_many.append(cap, a, kernels);
}
