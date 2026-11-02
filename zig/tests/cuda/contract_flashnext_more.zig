//! Flash Next's newer CUDA kernels under the contract, registered as one group: pending rows and their rollback.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const tc = @import("check.zig");
const late_k = @import("contract_flashnext_late.zig");
const roll_k = @import("contract_flashnext_roll.zig");

const g = core.contract.gates;

/// The kernels and their modules, registered in the contract's list; destroyed after the gates have run.
pub const Set = struct {
    a: std.mem.Allocator,
    batch: cuda.Module,
    late_fns: late_k.Fns,
    late: [2]late_k.Late,
    roll_fns: roll_k.Fns,
    roll: roll_k.Roll,

    pub fn init(gpu: tc.Gpu, s: cuda.Stream, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !*Set {
        const self = try a.create(Set);
        errdefer a.destroy(self);
        self.a = a;
        self.batch = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_batch);
        errdefer self.batch.unload();
        const b = &self.batch;
        self.late_fns = .{ .front = try b.function("fn_gdn_front"), .front_log = try b.function("fn_gdn_front_log"), .rows_to = try b.function("fn_gdn_rows_to"), .rows_late = try b.function("fn_gdn_rows_late") };
        for (std.enums.values(late_k.Kind), &self.late) |kind, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .f = &self.late_fns };
        for (&self.late) |*x| try kernels.append(a, x.kernel());
        self.roll_fns = .{ .shift = try b.function("fn_shift_window"), .roll = try b.function("fn_late_roll") };
        self.roll = .{ .rig = .{ .gpu = gpu, .s = s }, .f = &self.roll_fns };
        try kernels.append(a, self.roll.kernel());
        return self;
    }

    pub fn deinit(self: *Set) void {
        for (&self.late) |*x| x.deinit();
        self.roll.deinit();
        self.batch.unload();
        self.a.destroy(self);
    }
};
