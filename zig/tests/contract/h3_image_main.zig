//! tf-h3-image-contract: MiniMax-H3-Image's Metal kernels through gates 1-4, their controls, then gate 5 on this GPU.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const common = @import("metal_h3_image_common.zig");
const g = contract.gates;
const records = contract.h3_image_records.records;
const usage = "usage: tf-h3-image-contract [--only TEXT] [--seed N] [--no-limits]";

/// Each launch compiled as its launcher compiles it: the pipeline's thread limit and threadgroup bytes.
fn limitsHere(a: std.mem.Allocator, device: mtl.Device, only: ?[]const u8) !usize {
    var n: usize = 0;
    for (records) |d| {
        if (!common.keep(only, d.name)) continue;
        const part: common.Part = if (std.mem.indexOf(u8, d.name, " encoder ") != null) .encoder else if (std.mem.indexOf(u8, d.name, " prompt ") != null) .prompt else .decoder;
        const lib = try common.library(device, part, common.source(part));
        defer lib.deinit();
        const p = try mtl.Pipeline.init(device, lib, d.name[std.mem.lastIndexOfScalar(u8, d.name, ' ').? + 1 ..], false);
        defer p.deinit();
        const most = p.maxThreads();
        const tg = mtl.objc.msg(usize, p.id, "staticThreadgroupMemoryLength", .{});
        const bad = most < d.threads or tg > d.tg_memory or (d.reserved != 0 and most > d.reserved);
        if (bad) n += 1;
        std.debug.print("{s} limits here {s}: {d} threads launched, the pipeline takes {d} (reserved {d}); {d} threadgroup bytes (declared {d})\n", .{ if (bad) "FAIL" else "PASS", d.name, d.threads, most, d.reserved, tg, d.tg_memory });
    }
    return n + try contract.registry.checkMetal(a, &records, false);
}

pub fn main(init: std.process.Init) !u8 {
    const a = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var only: ?[]const u8 = null;
    var seed: u64 = 0x6b63;
    var limits = true;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--no-limits")) limits = false else if (i + 1 < args.len and std.mem.eql(u8, args[i], "--only")) {
            i += 1;
            only = args[i];
        } else if (i + 1 < args.len and std.mem.eql(u8, args[i], "--seed")) {
            i += 1;
            seed = try std.fmt.parseInt(u64, args[i], 0);
        } else {
            std.debug.print("{s}\n", .{usage});
            return 2;
        }
    }
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    std.debug.print("H3_IMAGE_DEVICE {s} tensor_units={any}\n", .{ device.name(), device.tensorUnits() });
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    const encoder = try @import("metal_h3_image_encoder.zig").Suite.create(a, device, queue, &kernels, only);
    defer encoder.destroy();
    const prompt = try @import("metal_h3_image_prompt.zig").Suite.create(a, device, queue, &kernels, only);
    defer prompt.destroy();
    const decoder = try @import("metal_h3_image_decoder.zig").Suite.create(a, device, queue, &kernels, only);
    defer decoder.destroy();
    var t: g.Tally = .{};
    try g.runAll(a, kernels.items, seed, &t);
    const lim = if (limits) try limitsHere(a, device, only) else 0;
    std.debug.print("H3_IMAGE_SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed; {d} limit problems\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim });
    return if (t.problems() + lim == 0) 0 else 1;
}
