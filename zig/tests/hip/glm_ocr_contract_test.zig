//! Focused common-kernel gates and planted controls; the full HIP runner registers these same adapters.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const Ops = @import("glm_ocr_ops").Ops;
const shaders = @import("glm_ocr_kernels");
const controls = @import("hip_controls");
const pointwise = @import("contract_glm_ocr_ops.zig");
const norms = @import("contract_glm_ocr_norm.zig");
const row_controls = @import("contract_rows.zig");
const g = core.contract.gates;

test "GLM-OCR HIP contract: pointwise, rotary, layout and norms through all common gates" {
    const a = std.testing.allocator;
    var r = try hip.Runtime.open();
    defer r.close();
    if (try r.deviceCount() != 1) return error.SelectOneHipDevice;
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    if ((try ctx.memInfo()).free < 128 << 20) return error.NoDeviceBudget;
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var ops = try Ops.load(&r);
    defer ops.unload();
    var ctl = try hip.Module.load(&r, &controls.bytes);
    defer ctl.unload();
    var points: [std.enums.values(pointwise.Which).len]pointwise.Pointwise = undefined;
    for (&points, std.enums.values(pointwise.Which)) |*p, w| p.* = .init(a, &r, stream, &ops, w);
    defer for (&points) |*p| p.deinit();
    var normalized: [std.enums.values(norms.Which).len]norms.Norm = undefined;
    for (&normalized, std.enums.values(norms.Which)) |*p, w| p.* = .init(a, &r, stream, &ops, w);
    defer for (&normalized) |*p| p.deinit();
    // These five cases resolve only the planted-control module, never the unused production launcher.
    const unused: hip.Launcher = undefined;
    var planted: [5]row_controls.Row = undefined;
    var count: usize = 0;
    defer for (planted[0..count]) |*p| p.deinit();
    for ([_]row_controls.Which{ .fmax_max, .atomic_sum, .scribble_sum, .tail_sum, .half_dot }) |w| {
        planted[count] = try row_controls.Row.init(a, &r, stream, &unused, ctl, w);
        count += 1;
    }
    var kernels: [points.len + normalized.len + planted.len]g.Kernel = undefined;
    for (&points, kernels[0..points.len]) |*p, *k| k.* = p.kernel();
    for (&normalized, kernels[points.len..][0..normalized.len]) |*p, *k| k.* = p.kernel();
    for (&planted, kernels[points.len + normalized.len ..]) |*p, *k| k.* = p.kernel();
    var tally: g.Tally = .{};
    try g.runAll(a, &kernels, 0x6b63, &tally);
    try std.testing.expectEqual(@as(usize, 0), tally.problems());
    try std.testing.expectEqual(@as(usize, 5), tally.caught);
    const metadata = try core.contract.hsaco.kernels(a, shaders.bytes[0..]);
    defer a.free(metadata);
    const records = core.contract.hip_records;
    const chip = for (core.contract.limits.rdna, 0..) |item, i| {
        if (std.mem.eql(u8, item.name, shaders.arch)) break i;
    } else return error.UnknownArchitecture;
    var checked: usize = 0;
    for (records.records) |record| {
        if (!std.mem.eql(u8, record.launch.image, "glm_ocr")) continue;
        try std.testing.expectEqual(@as(usize, 0), try records.check(a, &.{record}, false));
        const compiled = core.contract.hsaco.find(metadata, record.launch.symbol).?.usage;
        try std.testing.expectEqualDeep(record.usage[chip], compiled);
        const symbol = try a.dupeSentinel(u8, record.launch.symbol, 0);
        defer a.free(symbol);
        const function = try ops.module.function(symbol);
        try std.testing.expectEqual(compiled.vgprs, @as(u32, @intCast(try function.attribute(.num_regs))));
        try std.testing.expectEqual(compiled.lds, @as(u32, @intCast(try function.attribute(.shared_size_bytes))));
        try std.testing.expect(record.launch.threads <= try function.attribute(.max_threads_per_block));
        checked += 1;
    }
    try std.testing.expectEqual(@as(usize, 12), checked);
    std.debug.print("GLM-OCR HIP contract: {d} kernels, {d} controls caught, {d} current launch records\n", .{ tally.kernels, tally.caught, checked });
}
