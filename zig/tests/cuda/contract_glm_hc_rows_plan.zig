//! Explicit HC-only commands and resource checks; no pending entry is promoted to qualified.
const std = @import("std");
const limits = @import("core").contract.limits;
pub const Mode = enum { numerical, limits };
pub const numerical_command = "glm-hc-rows-contract";
pub const limits_command = "glm-hc-rows-limits";
pub fn mode(cmd: []const u8) ?Mode {
    if (std.mem.eql(u8, cmd, numerical_command)) return .numerical;
    if (std.mem.eql(u8, cmd, limits_command)) return .limits;
    return null;
}
pub const names = [_][]const u8{ "cuda glm HC shared-row partial8 and finish", "control: glm HC shared weights read the next input row" };

pub const launches = [_]limits.CudaLaunch{
    .{ .name = "HC rows replacement", .symbol = "tf_glm_hc_partial8_rows", .image = "glm_glue", .threads = 1024, .min_sm = 80 },
    .{ .name = "HC rows wrong-row control", .symbol = "tf_glm_hc_partial8_rows_control", .image = "glm_glue", .threads = 1024, .min_sm = 80 },
    .{ .name = "HC rows original", .symbol = "tf_glm_hc_partial8", .image = "glm_glue", .threads = 1024, .min_sm = 80 },
    .{ .name = "HC rows finish", .symbol = "tf_glm_hc_finish", .image = "glm_glue", .threads = 256, .min_sm = 80 },
};
pub fn audit(a: std.mem.Allocator, launch: limits.CudaLaunch, compiled: limits.Usage, driver: limits.Usage, resident: u32, target: limits.Sm, notes: *std.ArrayList(u8)) !usize {
    var problems = try limits.cuda(launch, compiled, target, notes, a);
    problems += @intFromBool(compiled.regs == 0 or driver.regs == 0);
    problems += @intFromBool(compiled.regs != driver.regs or compiled.static != driver.static);
    problems += @intFromBool(driver.bound < launch.threads or resident == 0);
    problems += @intFromBool(driver.local != 0);
    return problems;
}
pub const summary = "SUMMARY glm HC rows: replacement={s}; control={s}; {d} kernels, {d} pass, {d} fail; {d} controls, {d} red, {d} missed\n";
