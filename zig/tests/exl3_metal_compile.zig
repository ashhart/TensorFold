const ex = @import("exl3_projection");
const mtl = @import("metal");
export fn loadMetal(d: *const mtl.Device, p: *const ex.checkpoint.Parts, g: *const ex.checkpoint.Group) void {
    var owned = ex.load.fromParts(d.*, p.*, g.*) catch return;
    owned.deinit();
}
