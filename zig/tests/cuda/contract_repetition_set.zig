//! Ownership for repetition sampling registrations in the complete CUDA contract.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const gates = @import("core").contract.gates;
const Group = @import("contract_sample_repetition.zig").Group;

pub const Set = struct {
    gpa: std.mem.Allocator,
    image: cuda.Module,
    group: Group,

    pub fn create(gpu: check.Gpu, stream: cuda.Stream, a: std.mem.Allocator, kernels: *std.ArrayList(gates.Kernel)) !*Set {
        var image = try cuda.Module.load(gpu.d, cuda.kernels.sample);
        errdefer image.unload();
        const self = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(self);
        self.* = .{ .gpa = gpu.gpa, .image = image, .group = try Group.init(gpu, stream, image) };
        errdefer self.group.deinit();
        try self.group.append(a, kernels);
        return self;
    }

    pub fn deinit(self: *Set) void {
        self.group.deinit();
        self.image.unload();
        self.gpa.destroy(self);
    }
};
