//! Bounded registered repetition gates: only sample, norms and the shared planted-control image are loaded.
const std = @import("std");
const cuda = @import("cuda");
const contract = @import("core").contract;
const check = @import("check.zig");
const g = contract.gates;
const registry = contract.registry;
const limits = contract.limits;
const records = @import("core").contract.registry.repetition_sampling.records;
const options = @import("contract_options");
const control_image = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

fn resourceGates(gpu: check.Gpu, sample: cuda.Module, controls: cuda.Module) !usize {
    const a = gpu.gpa;
    const images = try contract.cubin.images(a, cuda.kernels.sample);
    defer contract.cubin.freeImages(a, images);
    var problems: usize = 0;
    const sm: u32 = @intCast(try gpu.ctx.capability());
    for (records) |r| {
        std.debug.print("RECORD {s}: ", .{r.launch.name});
        for (limits.sms, 0..) |target, at| {
            const image = for (images) |im| {
                if (im.sm == target.sm) break im;
            } else return error.MissingRepetitionTargetImage;
            const kernels = try contract.cubin.kernels(a, image.elf, target.sm);
            defer a.free(kernels);
            const measured = for (kernels) |k| {
                if (std.mem.eql(u8, k.name, r.launch.symbol)) break k.usage;
            } else return error.MissingRepetitionSymbol;
            std.debug.print(".{{ .regs = {d}, .static = {d}, .bound = {d} }}, ", .{ measured.regs, measured.static, measured.bound });
            var notes: std.ArrayList(u8) = .empty;
            defer notes.deinit(a);
            problems += try limits.cuda(r.launch, measured, target, &notes, a);
            if (notes.items.len > 0) std.debug.print("\n{s}", .{notes.items});
            const release = try contract.cubin.release(a, image.elf);
            if (!std.meta.eql(measured, registry.usageFor(r, release)[at])) problems += 1;
            if (target.sm == sm) {
                const symbol = try a.dupeSentinel(u8, r.launch.symbol, 0);
                defer a.free(symbol);
                const f = try sample.function(symbol);
                const regs: u32 = @intCast(try f.attribute(.num_regs));
                const shared: u32 = @intCast(try f.attribute(.shared_size_bytes));
                const most: u32 = @intCast(try f.attribute(.max_threads_per_block));
                if (regs != measured.regs or shared != measured.static or most < r.launch.threads) problems += 1;
            }
        }
        std.debug.print("\n", .{});
    }
    var red: usize = 0;
    for (registry.cuda_controls) |r| {
        const symbol = try a.dupeSentinel(u8, r.launch.symbol, 0);
        defer a.free(symbol);
        const f = try controls.function(symbol);
        const most: u32 = @intCast(try f.attribute(.max_threads_per_block));
        const regs: u32 = @intCast(try f.attribute(.num_regs));
        const caught = r.launch.threads > most or limits.blockRegs(regs, r.launch.threads) > 65536;
        if (caught) red += 1 else problems += 1;
        std.debug.print("{s} limits {s}: {d} threads, driver bound {d}, {d} registers/thread\n", .{ if (caught) "CONTROL RED" else "CONTROL MISSED", r.launch.name, r.launch.threads, most, regs });
    }
    std.debug.print("{s} limits repetition: 2 launches, five compiled SMs, live sm_{d}, {d}/2 limit controls red, {d} problems\n", .{ if (problems == 0) "PASS" else "FAIL", sm, red, problems });
    return problems;
}

pub fn run(gpu: check.Gpu) !void {
    if (!cuda.kernels.available or !options.controls) return error.RepetitionContractNeedsImagesAndControls;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var sample = try cuda.Module.load(gpu.d, cuda.kernels.sample);
    defer sample.unload();
    var ctl = try cuda.Module.load(gpu.d, &control_image.bytes);
    defer ctl.unload();
    var norms = try cuda.norms.Norms.load(gpu.d);
    defer norms.unload();
    var group = try @import("contract_sample_repetition.zig").Group.init(gpu, s, sample);
    defer group.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try group.append(gpu.gpa, &kernels);
    const rows = @import("contract_rows.zig");
    const plants = [_]rows.Which{ .fmax_max, .atomic_sum, .scribble_sum, .tail_sum, .half_dot };
    var shared: [plants.len]rows.Row = undefined;
    var initialized: usize = 0;
    defer for (shared[0..initialized]) |*item| item.deinit();
    for (plants, &shared) |which, *item| {
        item.* = try rows.Row.init(gpu, s, &norms, ctl, which);
        initialized += 1;
        try kernels.append(gpu.gpa, item.kernel());
    }
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x7265706574697469, &tally);
    try group.finiteRules();
    const problems = try resourceGates(gpu, sample, ctl);
    std.debug.print("SUMMARY repetition: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known; {d} controls, {d} red, {d} missed; {d} limit problems\n", .{ tally.kernels, tally.passed, tally.failed, tally.known, tally.controls, tally.caught, tally.missed, problems });
    try check.expect(tally.problems() + problems == 0, "repetition kernel contract failed", .{});
}
