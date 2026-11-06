//! Device bytes must equal every source tensor; run on one selected GPU within its available memory budget.
const std = @import("std");
const hip = @import("hip");
const family = @import("glm_ocr");
const checkpoint = family.checkpoint;
const platform = @import("sys");

test "GLM-OCR HIP device weights: unchanged BF16 bytes and complete release" {
    const dir = platform.getenv("TF_GLM_OCR_DIR") orelse return error.MissingCheckpointPath;
    const pci = platform.getenv("TF_GLM_OCR_PCI") orelse return error.MissingSelectedDevice;
    const a = std.testing.allocator;
    var r = try hip.Runtime.open();
    defer r.close();
    if (try r.deviceCount() != 1) return error.SelectOneHipDevice;
    const Pci = *const fn ([*]u8, c_int, c_int) callconv(.c) c_int;
    const get_pci = r.lib.lookup(Pci, "hipDeviceGetPCIBusId") orelse return error.MissingSymbol;
    var bus: [32]u8 = @splat(0);
    try hip.runtime.check(get_pci(&bus, bus.len, 0));
    try std.testing.expectEqualStrings(pci, std.mem.sliceTo(&bus, 0));
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    const available = try ctx.memInfo();
    if (available.free < 1 << 30) return error.NoDeviceBudget;
    const before = hip.usage(false).device;
    {
        var weights = try family.weights.Weights.load(a, &r, std.testing.io, dir, available.free - (512 << 20));
        defer weights.deinit();
        var ck = try checkpoint.Checkpoint.open(a, std.testing.io, dir);
        defer ck.close();
        for (weights.layout.tensors.keys(), weights.layout.tensors.values()) |name, place| {
            const source = try ck.get(name);
            const actual = try a.alloc(u8, place.bytes);
            defer a.free(actual);
            try weights.buffer.download(place.offset, actual);
            try std.testing.expectEqualSlices(u8, source.bytes, actual);
        }
        try std.testing.expectEqual(before + weights.layout.bytes, hip.usage(false).device);
        try ctx.synchronize();
        std.debug.print("GLM-OCR HIP weights: 526 tensors, {d} unchanged bytes\n", .{weights.layout.bytes});
    }
    try std.testing.expectEqual(before, hip.usage(false).device);
}
