// tf-vulkan-greedy-contract: the greedy draw through the kernel contract's gates and its planted control, on a real GPU. One registration of the composed draw — finite_logits, argreduce_chunk and argreduce_combine behind one submission, three launches inside the run hook (no pass alone carries the servable contract) — through gates 1-4, plus the control, zig/kernels/vulkan/contract_controls_greedy.comp (the mirror of HIP's tf_ctl_fmax_max: a row max that drops a NaN), registered with control = .nonfinite and required to go red. Gate 5 rides the device's own limits: dispatch.Pipeline.init checks workgroup, push and binding limits per device before creating anything, so an over-limit pipeline never runs. Exit 0 pass, 1 fail, 2 usage, 124 a fence timeout left the GPU owning its resources. The acceptance is exact: the token equals the host greedyRef (max wins, ties to the lowest index, +0/-0 equal), a planted non-finite row refuses naming its first offender, repeats are byte-identical, inputs and guards untouched — and the control fails the nonfinite gate as planted.
//  The greedy_controls module (the control's SPIR-V, one decl: fmax_max) and the tf-vulkan-greedy-contract target are wired by the parent in zig/build/vulkan.zig alongside the stage5_shaders module.
const std = @import("std");
const vk = @import("vulkan");
const contract = @import("contract");
const g = contract.gates;
const shaders = @import("stage5_shaders");
const controls = @import("greedy_controls");
const greedy_k = @import("contract_greedy.zig");

const usage = "usage: tf-vulkan-greedy-contract [--vendor ID] [--timeout-ms MS] | --help\n" ++
    "  --help          show this text and exit without opening a loader or device\n" ++
    "  --vendor ID     refuse a device whose Vulkan vendorID differs (decimal or 0x hex; no default)\n" ++
    "  --timeout-ms MS bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n";

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    for (args[1..]) |arg| if (std.mem.eql(u8, arg, "--help")) {
        std.debug.print(usage, .{});
        return 0;
    };
    var vendor: ?u32 = null;
    var timeout_ms: u64 = 10_000;
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        const value = if (i + 1 < args.len) args[i + 1] else "";
        if (std.mem.eql(u8, args[i], "--vendor")) {
            vendor = std.fmt.parseInt(u32, value, 0) catch null;
            if (vendor == null) break;
        } else if (std.mem.eql(u8, args[i], "--timeout-ms")) {
            timeout_ms = std.fmt.parseInt(u64, value, 10) catch 0;
            if (timeout_ms == 0 or timeout_ms > 600_000) break;
        } else break;
    } else return run(a, vendor, timeout_ms * std.time.ns_per_ms);
    std.debug.print(usage, .{});
    return 2;
}

fn shaderSet() vk.greedy.Shaders {
    return .{
        .finite_logits = &shaders.finite_logits,
        .argreduce_chunk = &shaders.argreduce_chunk,
        .argreduce_combine = &shaders.argreduce_combine,
    };
}

fn run(a: std.mem.Allocator, vendor: ?u32, timeout: u64) !u8 {
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{ .features = .{ .storage16 = true, .int16 = true } });
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    const v = pr.apiVersion;
    std.debug.print("MILESTONE greedy-contract device-found name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4} " ++
        "api={d}.{d}.{d} storage16={} int16={} fence-timeout={d}ms\n", .{
        name,                         pr.vendorID,
        pr.deviceID,                  v >> 22 & 0x7f,
        v >> 12 & 0x3ff,              v & 0xfff,
        dev.features.storage16,       dev.features.int16,
        timeout / std.time.ns_per_ms,
    });
    if (pr.deviceType == vk.c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
        std.debug.print("FAIL greedy-contract: a CPU (software) Vulkan device is never a GPU result\n", .{});
        return 1;
    }
    if (vendor) |want| if (pr.vendorID != want) {
        std.debug.print("FAIL greedy-contract: vendor 0x{x:0>4} is not the requested 0x{x:0>4}\n", .{ pr.vendorID, want });
        return 1;
    };
    var cmds = try vk.Commands.init(&dev);
    defer cmds.deinit();
    // The largest contract shell (max_chunks of 4096 bf16) rides staging: 2 MiB.
    var staging = try vk.Buffer.init(&dev, .staging, 4 << 20);
    defer staging.deinit();
    var draw = greedy_k.Greedy.init(a, &dev, &cmds, staging, timeout, shaderSet(), null);
    defer draw.deinit();
    var ctl = greedy_k.Greedy.init(a, &dev, &cmds, staging, timeout, shaderSet(), &controls.fmax_max);
    defer ctl.deinit();
    var kernels = [_]g.Kernel{ draw.kernel(), ctl.kernel() };
    var t: g.Tally = .{};
    try g.runAll(a, &kernels, 0x6b63, &t);
    std.debug.print("SUMMARY vulkan greedy contract on \"{s}\": {d} kernel, {d} pass gates 1-4, {d} fail; " ++
        "{d} control: {d} red as planted, {d} missed\n", .{ name, t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    return if (t.problems() > 0) 1 else 0;
}
