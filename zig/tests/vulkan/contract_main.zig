//! Vulkan contract entry point, mirroring HIP's runAll tally and naming the live-device gate-5 evidence.
const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("vulkan_shaders");
const core = @import("core");
const fx = @import("contract_qmm.zig");
const g = core.gates;
const records = core.vulkan_records;

pub fn main(init: std.process.Init) !u8 {
    var rt = try vk.Runtime.open(); defer rt.close();
    var dev = try vk.Device.open(&rt, .{ .features = .{ .storage16 = true } }); defer dev.close();
    const props = dev.properties;
    const lim = props.limits;
    const name = std.mem.sliceTo(&props.deviceName, 0);
    std.debug.print("LIMITS deviceName=\"{s}\"\n", .{name});
    std.debug.print("LIMITS driverName=\"{s}\" driverInfo=\"{s}\" subgroupSize={d}\n", .{ std.mem.sliceTo(&dev.v12.driverName, 0), std.mem.sliceTo(&dev.v12.driverInfo, 0), dev.v11.subgroupSize });
    std.debug.print("LIMITS maxComputeWorkGroupInvocations={d} maxComputeWorkGroupSize={d},{d},{d} maxComputeSharedMemorySize={d}\n", .{ lim.maxComputeWorkGroupInvocations, lim.maxComputeWorkGroupSize[0], lim.maxComputeWorkGroupSize[1], lim.maxComputeWorkGroupSize[2], lim.maxComputeSharedMemorySize });
    std.debug.print("LIMITS maxStorageBufferRange={d} minStorageBufferOffsetAlignment={d}\n", .{ lim.maxStorageBufferRange, lim.minStorageBufferOffsetAlignment });
    var problems: usize = 0;
    var gate5_controls: usize = 0;
    var tally: g.Tally = .{};
    for (records.records) |record| {
        const local = record.launch.local_size;
        const invocations: u64 = @as(u64, local[0]) * local[1] * local[2];
        const fits = invocations <= lim.maxComputeWorkGroupInvocations and local[0] <= lim.maxComputeWorkGroupSize[0] and local[1] <= lim.maxComputeWorkGroupSize[1] and local[2] <= lim.maxComputeWorkGroupSize[2] and record.launch.shared_bytes <= lim.maxComputeSharedMemorySize;
        const expected_fail = record.control_gate == 5;
        const caught = fits != expected_fail;
        std.debug.print("{s} limits {s}: local={d}x{d}x{d} invocations={d} shared={d}\n", .{ if (caught) "PASS" else "FAIL", record.launch.name, local[0], local[1], local[2], invocations, record.launch.shared_bytes });
        if (!caught) problems += 1;
        if (expected_fail) {
            if (caught) gate5_controls += 1 else {
                tally.missed += 1;
                std.debug.print("CONTROL MISSED {s}: gate 5 did not reject the oversized declaration\n", .{record.launch.name});
            }
        }
    }
    var cmds = try vk.Commands.init(&dev); defer cmds.deinit();
    var staging = try vk.Buffer.init(&dev, .staging, 64 << 10); defer staging.deinit();
    const layout: vk.dispatch.Layout = .{ .bindings = 5, .push_bytes = @sizeOf(fx.Params), .writes = 1 << 4 };
    var pipeline = try vk.dispatch.Pipeline.init(&dev, &shaders.qmm_bf16, layout); defer pipeline.deinit();
    if (!std.mem.eql(u32, &pipeline.local, &.{ fx.local_size_x, 1, 1 })) return error.LocalSizeMismatch;
    var fixture = fx.Fixture.init(init.gpa, &dev, &cmds, staging, &pipeline); defer fixture.deinit();
    const control_layout: vk.dispatch.Layout = .{ .bindings = 3, .push_bytes = 8, .writes = 1 << 2 };
    var p_fmax = try vk.dispatch.Pipeline.init(&dev, &shaders.qmm_fmax, control_layout); defer p_fmax.deinit();
    var p_atomic = try vk.dispatch.Pipeline.init(&dev, &shaders.atomic_sum, control_layout); defer p_atomic.deinit();
    var p_scribble = try vk.dispatch.Pipeline.init(&dev, &shaders.scribble_sum, control_layout); defer p_scribble.deinit();
    var p_tail = try vk.dispatch.Pipeline.init(&dev, &shaders.drop_tail, control_layout); defer p_tail.deinit();
    var p_fp16 = try vk.dispatch.Pipeline.init(&dev, &shaders.fp16_accum, control_layout); defer p_fp16.deinit();
    var controls = [_]fx.Control{
        fx.Control.init(init.gpa, &dev, &cmds, staging, &p_fmax, .qmm_fmax),
        fx.Control.init(init.gpa, &dev, &cmds, staging, &p_atomic, .atomic_sum),
        fx.Control.init(init.gpa, &dev, &cmds, staging, &p_scribble, .scribble_sum),
        fx.Control.init(init.gpa, &dev, &cmds, staging, &p_tail, .drop_tail),
        fx.Control.init(init.gpa, &dev, &cmds, staging, &p_fp16, .fp16_accum),
    };
    defer for (&controls) |*control| control.deinit();
    var kernels: [6]g.Kernel = undefined;
    kernels[0] = fixture.kernel();
    for (&controls, 1..) |*control, i| kernels[i] = control.kernel();
    const start = std.Io.Clock.awake.now(init.io);
    tally.controls += gate5_controls;
    tally.caught += gate5_controls;
    try g.runAll(init.gpa, &kernels, 0x766b, &tally);
    const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).toMilliseconds();
    std.debug.print("SUMMARY {s} {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} caught, {d} missed; {d} limit failures; gates {d} ms\n", .{ name, tally.kernels, tally.passed, tally.failed, tally.known, tally.controls, tally.caught, tally.missed, problems, elapsed });
    if (tally.problems() != 0 or tally.missed != 0 or problems != 0) return 1;
    return 0;
}
