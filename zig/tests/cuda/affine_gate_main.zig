//! Focused affine FP8 qualification, independent of the server and its checkpoint loader.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const command = if (args.len > 1) args[1] else "all";
    if (args.len > 2 or !(equal(command, "all") or equal(command, "shared") or
        equal(command, "prepared") or equal(command, "wide")))
    {
        std.debug.print("usage: tf-affine-gate [all|shared|prepared|wide]\n", .{});
        return 2;
    }
    if (try cuda.fp8_prefill.tile_geometry.requested() != .baseline) return error.UnexpectedFp8TileOverride;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var context = try cuda.Context.init(&driver, 0);
    defer context.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &context, .gpa = init.gpa, .io = init.io };
    run(gpu, command) catch |err| {
        std.debug.print("FAIL affine gate {s}: {t}\n", .{ command, err });
        return 1;
    };
    std.debug.print("SUMMARY affine gate {s}: PASS\n", .{command});
    return 0;
}

fn equal(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn run(gpu: check.Gpu, command: []const u8) !void {
    const sm = try gpu.ctx.capability();
    if (sm != 120 and sm != 121) return error.UnsupportedAffineGateDevice;
    try limits(gpu);
    if (equal(command, "all") or equal(command, "shared")) try @import("contract_affine4_shared.zig").run(gpu);
    if (equal(command, "all") or equal(command, "prepared")) try @import("contract_affine_prepared.zig").run(gpu);
    if (equal(command, "all") or equal(command, "wide")) try @import("contract_affine_wide.zig").run(gpu);
}

fn limits(gpu: check.Gpu) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.affine_fp8_prepared);
    defer module.unload();
    const exports = [_]struct { symbol: [:0]const u8, registers: c_int, shared: c_int }{
        .{ .symbol = "tf_affine_fp8_prepare", .registers = 21, .shared = 0 },
        .{ .symbol = "tf_affine_fp8_prepared", .registers = 96, .shared = 26624 },
        .{ .symbol = "tf_affine_fp8_prepared_f32", .registers = 96, .shared = 26624 },
    };
    for (exports) |item| {
        const f = try module.function(item.symbol);
        const registers = try f.attribute(.num_regs);
        const shared = try f.attribute(.shared_size_bytes);
        const threads = try f.attribute(.max_threads_per_block);
        if (registers != item.registers or shared != item.shared or threads < 256)
            return error.AffineGateResourceMismatch;
        std.debug.print("PASS affine limits {s}: regs={d} shared={d} threads={d}\n", .{
            item.symbol, registers, shared, threads,
        });
    }
}
