//! Flash Next EXL3 contract kernels the runner builds together and releases together.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const tp_ = @import("contract_flashnext_exl3_tp.zig");
const canon_ = @import("contract_flashnext_exl3_canon.zig");
const writeback = @import("contract_flashnext_writeback.zig");
const prefill_ = @import("contract_flashnext_exl3_prefill.zig");
const promptd_ = @import("contract_flashnext_exl3_promptd.zig");
const tppf_ = @import("contract_flashnext_exl3_tppf.zig");

pub const Suite = struct {
    a: std.mem.Allocator,
    writebacks: [3]writeback.Writeback,
    masks: tp_.Masks,
    canons: canon_.Canons,
    prefill: *prefill_.Group,
    promptd: *promptd_.Set,
    tppf: *tppf_.Set,

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
        self.masks = try tp_.Masks.init(gpu, s);
        errdefer self.masks.deinit();
        try self.masks.add(a, kernels);
        self.canons = try canon_.Canons.init(gpu, s, controls);
        errdefer self.canons.deinit();
        try self.canons.add(a, kernels);
        self.prefill = try prefill_.Group.create(a, gpu, s, kernels);
        errdefer self.prefill.destroy();
        self.promptd = try a.create(promptd_.Set);
        errdefer a.destroy(self.promptd);
        self.promptd.* = .{};
        errdefer self.promptd.deinit();
        try self.promptd.add(gpu, s, kernels);
        self.tppf = try tppf_.Set.create(a, gpu, s, controls);
        errdefer self.tppf.destroy();
        try self.tppf.add(a, kernels);
        return self;
    }

    pub fn destroy(self: *Suite) void {
        self.tppf.destroy();
        self.canons.deinit();
        self.promptd.deinit();
        self.a.destroy(self.promptd);
        self.prefill.destroy();
        self.masks.deinit();
        for (&self.writebacks) |*w| w.deinit();
        self.a.destroy(self);
    }
};
