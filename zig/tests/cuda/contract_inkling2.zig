//! Inkling 2's CUDA kernels under the contract: attention, glue, experts, tower and loop, each with controls.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    attn: @import("contract_inkling2_attn.zig").Set = .{},
    ops: @import("contract_inkling2_ops.zig").Set = .{},
    dspark: @import("contract_inkling2_dspark.zig").Set = .{},
    experts: @import("contract_inkling2_experts.zig").Set = .{},
    loop_math: @import("contract_inkling2_loop_math_extra.zig").Set = .{},
    loop: @import("contract_inkling2_loop.zig").Set = .{},

    /// The set, its kernels appended to `list`.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.* = .{ .gpa = gpu.gpa };
        try set.attn.init(gpu, s, a, list);
        errdefer set.attn.deinit();
        try set.ops.init(gpu, s, a, list);
        errdefer set.ops.deinit();
        try set.dspark.init(gpu, s, a, list);
        errdefer set.dspark.deinit();
        try set.experts.init(gpu, s, a, list);
        errdefer set.experts.deinit();
        try set.loop_math.init(gpu, s, a, list);
        errdefer set.loop_math.deinit();
        try set.loop.init(gpu, s, a, list);
        errdefer set.loop.deinit();
        return set;
    }

    pub fn deinit(set: *Set) void {
        set.loop.deinit();
        set.loop_math.deinit();
        set.experts.deinit();
        set.dspark.deinit();
        set.ops.deinit();
        set.attn.deinit();
        set.gpa.destroy(set);
    }
};
