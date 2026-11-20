//! Inkling 2's CUDA kernels under the contract: attention, glue, drafter, experts and tower, each with controls.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    attn: @import("contract_inkling2_attn.zig").Set = .{},

    /// The set, its kernels appended to `list`.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.* = .{ .gpa = gpu.gpa };
        try set.attn.init(gpu, s, a, list);
        return set;
    }

    pub fn deinit(set: *Set) void {
        set.attn.deinit();
        set.gpa.destroy(set);
    }
};
