// Stage 4 GDN contract uses production Gdn and planted controls; gate 5 checks live device limits and records.
const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("vulkan_shaders");
const shaders4 = @import("stage4_shaders");
const core = @import("core");
const ctl = @import("contract_qmm.zig");
const fx = @import("contract_gdn.zig");
const g = core.gates;
const records = core.vulkan_records;

// Shared bytes as each shader source declares (stage 4 doc block + prefill [2][DK] staging); values on the next line
const shared_bytes = [7]u64{ 0, 1024, 128, 3072, 3 * 128 * 4 + 32 * 129 * 4, 1024, 0 };
const shared_names = [7][]const u8{ "copy_u16", "conv_dw1d_bf16", "gdn_conv_decode_bf16", "gdn_split_bf16", "prefill_recur_bf16", "gated_delta_decode_bf16", "fast_norm_gated_bf16" };

pub fn main(init: std.process.Init) !u8 {
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{ .features = .{ .storage16 = true, .int16 = true } });
    defer dev.close();
    const props = dev.properties;
    const lim = props.limits;
    const name = std.mem.sliceTo(&props.deviceName, 0);
    std.debug.print("LIMITS deviceName=\"{s}\"\n", .{name});
    std.debug.print("LIMITS driverName=\"{s}\" driverInfo=\"{s}\" subgroupSize={d} shuffle-relative={} storage16={} int16={}\n", .{ std.mem.sliceTo(&dev.v12.driverName, 0), std.mem.sliceTo(&dev.v12.driverInfo, 0), dev.v11.subgroupSize, dev.v11.subgroupSupportedOperations & vk.c.VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT != 0, dev.features.storage16, dev.features.int16 });
    std.debug.print("LIMITS maxComputeWorkGroupInvocations={d} maxComputeWorkGroupSize={d},{d},{d} maxComputeSharedMemorySize={d}\n", .{ lim.maxComputeWorkGroupInvocations, lim.maxComputeWorkGroupSize[0], lim.maxComputeWorkGroupSize[1], lim.maxComputeWorkGroupSize[2], lim.maxComputeSharedMemorySize });
    std.debug.print("LIMITS maxStorageBufferRange={d} minStorageBufferOffsetAlignment={d}\n", .{ lim.maxStorageBufferRange, lim.minStorageBufferOffsetAlignment });
    if (props.deviceType == vk.c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
        std.debug.print("FAIL gdn-contract: a CPU (software) Vulkan device is never a GPU result\n", .{});
        return 1;
    }
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
    var cmds = try vk.Commands.init(&dev);
    defer cmds.deinit();
    var staging = try vk.Buffer.init(&dev, .staging, 32 << 20); // the T=130 prefill's snapshot slot
    defer staging.deinit();
    var m = try vk.Gdn.init(&dev, .{
        .conv_dw1d_bf16 = &shaders4.conv_dw1d_bf16,
        .gdn_conv_decode_bf16 = &shaders4.gdn_conv_decode_bf16,
        .gdn_split_bf16 = &shaders4.gdn_split_bf16,
        .prefill_recur_bf16 = &shaders4.prefill_recur_bf16,
        .gated_delta_decode_bf16 = &shaders4.gated_delta_decode_bf16,
        .fast_norm_gated_bf16 = &shaders4.fast_norm_gated_bf16,
        .copy_u16 = &shaders4.copy_u16,
    });
    defer m.deinit();
    // Gate 5 checks every reflected local size and each shader's declared shared bytes against this device's limits.
    const locals = [7][3]u32{ m.copy_u16.local, m.conv_bf16.local, m.conv_decode_bf16.local, m.split_bf16.local, m.prefill_bf16.local, m.decode_bf16.local, m.norm_bf16.local };
    for (locals, shared_bytes, shared_names) |local, shared, sname| {
        const invocations: u64 = @as(u64, local[0]) * local[1] * local[2];
        const fits = invocations <= lim.maxComputeWorkGroupInvocations and local[0] <= lim.maxComputeWorkGroupSize[0] and local[1] <= lim.maxComputeWorkGroupSize[1] and local[2] <= lim.maxComputeWorkGroupSize[2] and shared <= lim.maxComputeSharedMemorySize;
        std.debug.print("{s} limits gdn {s}: local={d}x{d}x{d} invocations={d} shared={d}\n", .{ if (fits) "PASS" else "FAIL", sname, local[0], local[1], local[2], invocations, shared });
        if (!fits) problems += 1;
    }

    var split_soa = fx.Gdn.init(init.gpa, &dev, &cmds, staging, &m, .split_soa);
    defer split_soa.deinit();
    var split_g = fx.Gdn.init(init.gpa, &dev, &cmds, staging, &m, .split_g);
    defer split_g.deinit();
    var conv_pre = fx.Gdn.init(init.gpa, &dev, &cmds, staging, &m, .conv_prefill);
    defer conv_pre.deinit();
    var conv_dec = fx.Gdn.init(init.gpa, &dev, &cmds, staging, &m, .conv_decode);
    defer conv_dec.deinit();
    var pre_y = fx.Gdn.init(init.gpa, &dev, &cmds, staging, &m, .prefill_y);
    defer pre_y.deinit();
    var pre_state = fx.Gdn.init(init.gpa, &dev, &cmds, staging, &m, .prefill_state);
    defer pre_state.deinit();
    var dec_y = fx.Gdn.init(init.gpa, &dev, &cmds, staging, &m, .decode_y);
    defer dec_y.deinit();
    var dec_state = fx.Gdn.init(init.gpa, &dev, &cmds, staging, &m, .decode_state);
    defer dec_state.deinit();

    const control_layout: vk.dispatch.Layout = .{ .bindings = 3, .push_bytes = 8, .writes = 1 << 2 };
    var p_fmax = try vk.dispatch.Pipeline.init(&dev, &shaders.qmm_fmax, control_layout);
    defer p_fmax.deinit();
    var p_atomic = try vk.dispatch.Pipeline.init(&dev, &shaders.atomic_sum, control_layout);
    defer p_atomic.deinit();
    var p_scribble = try vk.dispatch.Pipeline.init(&dev, &shaders.scribble_sum, control_layout);
    defer p_scribble.deinit();
    var p_tail = try vk.dispatch.Pipeline.init(&dev, &shaders.drop_tail, control_layout);
    defer p_tail.deinit();
    var p_fp16 = try vk.dispatch.Pipeline.init(&dev, &shaders.fp16_accum, control_layout);
    defer p_fp16.deinit();
    var controls = [_]ctl.Control{
        ctl.Control.init(init.gpa, &dev, &cmds, staging, &p_fmax, .qmm_fmax),
        ctl.Control.init(init.gpa, &dev, &cmds, staging, &p_atomic, .atomic_sum),
        ctl.Control.init(init.gpa, &dev, &cmds, staging, &p_scribble, .scribble_sum),
        ctl.Control.init(init.gpa, &dev, &cmds, staging, &p_tail, .drop_tail),
        ctl.Control.init(init.gpa, &dev, &cmds, staging, &p_fp16, .fp16_accum),
    };
    defer for (&controls) |*control| control.deinit();

    var kernels: [13]g.Kernel = undefined;
    kernels[0] = split_soa.kernel();
    kernels[1] = split_g.kernel();
    kernels[2] = conv_pre.kernel();
    kernels[3] = conv_dec.kernel();
    kernels[4] = pre_y.kernel();
    kernels[5] = pre_state.kernel();
    kernels[6] = dec_y.kernel();
    kernels[7] = dec_state.kernel();
    for (&controls, 8..) |*control, i| kernels[i] = control.kernel();
    const start = std.Io.Clock.awake.now(init.io);
    tally.controls += gate5_controls;
    tally.caught += gate5_controls;
    try g.runAll(init.gpa, &kernels, 0x6764, &tally);
    const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).toMilliseconds();
    std.debug.print("SUMMARY {s} gdn group: {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} caught, {d} missed; {d} limit failures; gates {d} ms\n", .{ name, tally.kernels, tally.passed, tally.failed, tally.known, tally.controls, tally.caught, tally.missed, problems, elapsed });
    if (tally.problems() != 0 or tally.missed != 0 or problems != 0) return 1;
    return 0;
}
