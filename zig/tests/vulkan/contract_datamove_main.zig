//! The datamove group's contract runner: the stage 1 exact kernels through gates 1-4 on a real GPU.
//! Exit 0 pass, 1 fail, 2 usage; every gate prints its own line.
const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("stage1_shaders");
const contract = @import("contract");
const dm = @import("contract_datamove.zig");

const g = contract.gates;

const seed: u64 = 0xda7a;

const usage = "usage: tf-stage1-datamove [--vendor ID] [--timeout-ms MS] | --help\n" ++
    "  --help          show this text and exit without opening a loader or device\n" ++
    "  --vendor ID     refuse a device whose Vulkan vendorID differs (decimal or 0x hex; no default)\n" ++
    "  --timeout-ms MS bound on every submission and transfer, 1 to 600000 (default 10000)\n";

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var vendor: ?u32 = null;
    var timeout_ms: u64 = 10_000;
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        const value = if (i + 1 < args.len) args[i + 1] else "";
        if (std.mem.eql(u8, args[i], "--help")) {
            std.debug.print(usage, .{});
            return 0;
        } else if (std.mem.eql(u8, args[i], "--vendor")) {
            vendor = std.fmt.parseInt(u32, value, 0) catch null;
            if (vendor == null) break;
        } else if (std.mem.eql(u8, args[i], "--timeout-ms")) {
            timeout_ms = std.fmt.parseInt(u64, value, 10) catch 0;
            if (timeout_ms == 0 or timeout_ms > 600_000) break;
        } else break;
    } else return run(init.gpa, vendor, timeout_ms * std.time.ns_per_ms);
    std.debug.print(usage, .{});
    return 2;
}

fn run(a: std.mem.Allocator, vendor: ?u32, timeout_ns: u64) !u8 {
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{ .features = .{ .storage16 = true, .int16 = true } });
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    if (pr.deviceType == vk.c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
        std.debug.print("FAIL datamove: a CPU (software) Vulkan device is never a GPU result\n", .{});
        return 1;
    }
    if (vendor) |want| if (pr.vendorID != want) {
        std.debug.print("FAIL datamove: vendor 0x{x:0>4} is not the requested 0x{x:0>4}\n", .{ pr.vendorID, want });
        return 1;
    };
    std.debug.print("MILESTONE device-found name=\"{s}\" vendor=0x{x:0>4} storage16={} int16={} timeout={d}ms\n", .{ name, pr.vendorID, dev.features.storage16, dev.features.int16, timeout_ns / std.time.ns_per_ms });
    var cmds = try vk.Commands.init(&dev);
    defer cmds.deinit();
    var staging = try vk.Buffer.init(&dev, .staging, 1 << 20);
    defer staging.deinit();
    const sh = vk.ops.Shaders{
        .qmm_bf16 = &shaders.qmm_bf16,
        .dequant_bf16 = &shaders.dequant_bf16,
        .gather_u32 = &shaders.gather_u32,
        .gather_u16 = &shaders.gather_u16,
        .cast_bf16_f32 = &shaders.cast_bf16_f32,
        .cast_f32_bf16 = &shaders.cast_f32_bf16,
        .copy_u32 = &shaders.copy_u32,
        .copy_u16 = &shaders.copy_u16,
    };
    var o = try vk.ops.Ops.init(&dev, sh);
    defer o.deinit();

    var cast_to = dm.Cast(true).init(a, &dev, &cmds, staging, &o, timeout_ns);
    defer cast_to.deinit();
    var cast_from = dm.Cast(false).init(a, &dev, &cmds, staging, &o, timeout_ns);
    defer cast_from.deinit();
    var copy_u32 = dm.Copy(.u32).init(a, &dev, &cmds, staging, &o, timeout_ns);
    defer copy_u32.deinit();
    var copy_u16 = dm.Copy(.u16).init(a, &dev, &cmds, staging, &o, timeout_ns);
    defer copy_u16.deinit();
    var gather_u32 = dm.Gather(.u32).init(a, &dev, &cmds, staging, &o, timeout_ns);
    defer gather_u32.deinit();
    var gather_u16 = dm.Gather(.u16).init(a, &dev, &cmds, staging, &o, timeout_ns);
    defer gather_u16.deinit();
    var dequant = dm.Dequant.init(a, &dev, &cmds, staging, &o, timeout_ns);
    defer dequant.deinit();

    var kernels = [_]g.Kernel{
        cast_to.kernel(),
        cast_from.kernel(),
        copy_u32.kernel(),
        copy_u16.kernel(),
        gather_u32.kernel(),
        gather_u16.kernel(),
        dequant.kernel(),
    };
    var t: g.Tally = .{};
    try g.runAll(a, &kernels, seed, &t);
    std.debug.print("RESULT datamove on \"{s}\": kernels={d} passed={d} failed={d} controls={d} known={d}\n", .{ name, t.kernels, t.passed, t.failed, t.controls, t.known });
    return if (t.problems() == 0) 0 else 1;
}
