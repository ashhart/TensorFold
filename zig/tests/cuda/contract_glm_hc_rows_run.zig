//! HC-only entry points avoid constructing unrelated fixtures and inspect their own embedded CUDA image.
const std = @import("std");
const cuda = @import("cuda");
const contract = @import("core").contract;
const check = @import("check.zig");
const fixture = @import("contract_glm_hc_rows.zig");
const plan = @import("contract_glm_hc_rows_plan.zig");

pub fn dispatch(gpu: check.Gpu, cmd: []const u8, args: []const [:0]const u8, image: []const u8) !bool {
    const mode = plan.mode(cmd) orelse return false;
    if (args.len != 0) return error.HcRowsUnexpectedArguments;
    switch (mode) {
        .numerical => try numerical(gpu, image),
        .limits => try resources(gpu, image),
    }
    return true;
}

pub fn numerical(gpu: check.Gpu, image: []const u8) !void {
    var tally: contract.gates.Tally = .{};
    defer std.debug.print(plan.summary, .{ plan.names[0], plan.names[1], tally.kernels, tally.passed, tally.failed, tally.controls, tally.caught, tally.missed });
    errdefer std.debug.print("FAIL glm HC rows: incomplete or failed numerical contract\n", .{});
    if (image.len == 0) return error.BuiltWithoutKernels;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var mod = try cuda.Module.load(gpu.d, image);
    defer mod.unload();
    const kernels = try @import("glm_cuda").glue.Kernels.load(mod);
    var cases = [_]fixture.Fixture{
        .{ .rig = .{ .gpu = gpu, .s = stream }, .glue = kernels },
        .{ .rig = .{ .gpu = gpu, .s = stream }, .glue = kernels, .fault = true },
    };
    defer for (&cases) |*item| item.deinit();
    cases[1].glue.hc_partial8_rows = try mod.function("tf_glm_hc_partial8_rows_control");
    const selected = [_]contract.gates.Kernel{ cases[0].kernel(), cases[1].kernel() };
    try contract.gates.runAll(gpu.gpa, &selected, 0x6b63, &tally);
    try check.expect(tally.kernels == 1 and tally.passed == 1 and tally.failed == 0 and tally.known == 0 and tally.controls == 1 and tally.caught == 1 and tally.missed == 0, "glm HC rows: require one passing replacement and one caught wrong-row control", .{});
    std.debug.print("INFO glm HC rows: gates 1-4 only; run glm-hc-rows-limits for observed resources; registry remains pending\n", .{});
}

pub fn resources(gpu: check.Gpu, image: []const u8) !void {
    var checked: usize = 0;
    var problems: usize = 0;
    var complete = false;
    defer std.debug.print("SUMMARY glm HC rows limits: embedded glm_glue; {d}/{d} symbols, {d} problems; {s}; pending records unchanged\n", .{ checked, plan.launches.len, problems, if (complete) "complete" else "INCOMPLETE" });
    if (image.len == 0) return error.BuiltWithoutKernels;
    const sm: u32 = @intCast(try gpu.ctx.capability());
    const target = contract.limits.target(sm) orelse return error.UnrecordedSm;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(image, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    std.debug.print("INFO glm HC rows limits: embedded glm_glue sha256={s}, sm_{d}\n", .{ hex, sm });
    const images = try contract.cubin.images(gpu.gpa, image);
    defer contract.cubin.freeImages(gpu.gpa, images);
    const elf = for (images) |candidate| {
        if (candidate.sm == sm) break candidate.elf;
    } else return error.HcRowsMissingSmImage;
    const compiler = try contract.cubin.release(gpu.gpa, elf) orelse return error.HcRowsUnknownCompiler;
    std.debug.print("INFO glm HC rows limits: nvcc {d}.{d}; observed current image, not a pinned qualification\n", .{ compiler.major, compiler.minor });
    const all = try contract.cubin.kernels(gpu.gpa, elf, sm);
    defer gpu.gpa.free(all);
    var mod = try cuda.Module.load(gpu.d, image);
    defer mod.unload();
    for (plan.launches) |launch| {
        const compiled = for (all) |kernel| {
            if (std.mem.eql(u8, kernel.name, launch.symbol)) break kernel.usage;
        } else return error.HcRowsMissingSymbol;
        const name = try gpu.gpa.dupeSentinel(u8, launch.symbol, 0);
        defer gpu.gpa.free(name);
        const function = try mod.function(name);
        const driver: contract.limits.Usage = .{
            .regs = @intCast(try function.attribute(.num_regs)),
            .static = @intCast(try function.attribute(.shared_size_bytes)),
            .bound = @intCast(try function.attribute(.max_threads_per_block)),
            .local = @intCast(try function.attribute(.local_size_bytes)),
        };
        const resident = try function.occupancy(launch.threads, launch.dynamic);
        var notes: std.ArrayList(u8) = .empty;
        defer notes.deinit(gpu.gpa);
        const red = try plan.audit(gpu.gpa, launch, compiled, driver, resident, target, &notes);
        problems += red;
        checked += 1;
        std.debug.print("{s} glm HC rows limits {s}: threads={d} regs={d} shared={d} bound={d} local={d} resident_blocks={d}; cubin_regs={d} cubin_shared={d} cubin_bound={d}\n", .{ if (red == 0) "PASS" else "FAIL", launch.symbol, launch.threads, driver.regs, driver.static, driver.bound, driver.local, resident, compiled.regs, compiled.static, compiled.bound });
        if (notes.items.len > 0) std.debug.print("{s}", .{notes.items});
    }
    complete = checked == plan.launches.len;
    try check.expect(complete and problems == 0, "glm HC rows limits: incomplete inventory or resource mismatch", .{});
}
