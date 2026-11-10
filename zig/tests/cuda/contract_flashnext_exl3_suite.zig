//! Flash Next EXL3 contract kernels the runner builds together and releases together.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const canon_ = @import("contract_flashnext_exl3_canon.zig");
const writeback = @import("contract_flashnext_writeback.zig");

pub const Suite = struct {
    a: std.mem.Allocator,
    writebacks: [3]writeback.Writeback,
    canons: canon_.Canons,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel), hc: cuda.Module, controls: cuda.Module) !*Suite {
        const self = try a.create(Suite);
        errdefer a.destroy(self);
        self.a = a;
        for (std.enums.values(writeback.Mode), &self.writebacks) |mode, *w| {
            const symbol: [:0]const u8 = switch (mode) {
                .sums => "fn_hc_writeback0",
                .moe => "fn_hc_writeback_moe",
                .slices => "fn_hc_writeback_slices",
            };
            w.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try hc.function(symbol), .mode = mode };
        }
        errdefer for (&self.writebacks) |*w| w.deinit();
        for (&self.writebacks) |*w| try kernels.append(a, w.kernel());
        self.canons = try canon_.Canons.init(gpu, s, controls);
        errdefer self.canons.deinit();
        try self.canons.add(a, kernels);
        return self;
    }

    pub fn destroy(self: *Suite) void {
        self.canons.deinit();
        for (&self.writebacks) |*w| w.deinit();
        self.a.destroy(self);
    }
};
