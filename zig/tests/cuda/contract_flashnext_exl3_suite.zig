//! Flash Next EXL3 contract kernels the runner builds together and releases together.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const tp_ = @import("contract_flashnext_exl3_tp.zig");
const prefill_ = @import("contract_flashnext_exl3_prefill.zig");

pub const Suite = struct {
    a: std.mem.Allocator,
    masks: tp_.Masks,
    prefill: *prefill_.Group,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Suite {
        const self = try a.create(Suite);
        errdefer a.destroy(self);
        self.a = a;
        self.masks = try tp_.Masks.init(gpu, s);
        errdefer self.masks.deinit();
        try self.masks.add(a, kernels);
        self.prefill = try prefill_.Group.create(a, gpu, s, kernels);
        return self;
    }

    pub fn destroy(self: *Suite) void {
        self.prefill.destroy();
        self.masks.deinit();
        self.a.destroy(self);
    }
};
