//! MTLFence: orders untracked work across encoders and command buffers of one queue (each waits on the last update).
const mtl = @import("metal");
/// The real runtime's encoders may be Metal 4's; recording fakes in tests have no such field.
const metal4 = @hasField(mtl.ComputeEncoder, "m4");

pub const Fence = struct {
    id: mtl.objc.Id,

    pub fn init(device: mtl.Device) !Fence {
        return .{ .id = mtl.objc.msg(?mtl.objc.Id, device.id, "newFence", .{}) orelse return error.NoFence };
    }

    /// The encoder's dispatches start after the work that last updated the fence.
    pub fn wait(self: Fence, enc: mtl.ComputeEncoder) void {
        if (comptime metal4) if (enc.m4 != null) return mtl.metal4.waitForFence(enc.id, self.id);
        mtl.objc.msg(void, enc.id, "waitForFence:", .{self.id});
    }

    /// Later waits see this encoder's work complete.
    pub fn update(self: Fence, enc: mtl.ComputeEncoder) void {
        if (comptime metal4) if (enc.m4 != null) return mtl.metal4.updateFence(enc.id, self.id);
        mtl.objc.msg(void, enc.id, "updateFence:", .{self.id});
    }

    pub fn deinit(self: Fence) void {
        mtl.objc.release(self.id);
    }
};
