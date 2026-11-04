//! Unlimited-OCR-2's CUDA rows under the contract: the tower's adds, norms and GELUs, the decoder's norms, rotary and SwiGLU.
const std = @import("std");
const core = @import("core");
const uocr = @import("contract_uocr.zig");

const g = core.contract.gates;

/// The group's registrations, on the heap: the gates hold pointers to them.
pub const Group = struct {
    gpa: std.mem.Allocator,

    pub fn init(ctx: uocr.Ctx, list: *std.ArrayList(g.Kernel)) !*Group {
        _ = list;
        const grp = try ctx.gpu.gpa.create(Group);
        grp.* = .{ .gpa = ctx.gpu.gpa };
        return grp;
    }

    pub fn deinit(grp: *Group) void {
        grp.gpa.destroy(grp);
    }
};
