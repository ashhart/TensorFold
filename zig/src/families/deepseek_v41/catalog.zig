//! Resolve the complete binding matrix against a recording driver, without loading device code.
const std = @import("std");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const abi = cuda.abi;
pub const Entry = struct { image: []const u8, name: [512]u8 = undefined, len: usize, shared: u32 = 0, threads: u32 = 0, cluster: u32 = 1, optional_shared: bool = false, available: bool = true, opt_in: bool = false };
pub var entries: [2048]Entry = undefined;
pub var count: usize = 0;
const images = [_][]const u8{ "dsv41_exl3_experts", "dsv41_x3ld", "dsv41_x3pf", "dsv41_x3gm", "dsv41_linear", "dsv41_x3seg", "dsv41_dense3", "dsv41_attn", "dsv41_topk", "dsv41_mhc", "dsv41_mhc_pf", "dsv41_router_gemv", "dsv41_pfdense_k8", "dsv41_pfdense_k10", "dsv41_pfdense_k12", "dsv41_pfdense_k16", "dsv41_engram_gate", "dsv41_l2pace", "dsv41_l2pf", "dsv41_x3gm_plan", "dsv41_topk_keys", "dsv41_pointwise", "dsv41_kvsplit", "dsv41_glue", "dsv41_x3gm3", "dsv41_pfglue", "dsv41_router_glue", "dsv41_kv_glue", "dsv41_x3ld_epi", "dsv41_topk_b", "dsv41_lin2", "dsv41_sampling", "exl3_linear", "exl3_experts", "exl3_candidate" };
fn function(out: *abi.Function, module: abi.Module, name: [*:0]const u8) callconv(.c) abi.Result {
    const image = images[@intFromPtr(module.?) - 1];
    const text = std.mem.span(name);
    for (entries[0..count], 0..) |e, i| if (std.mem.eql(u8, e.image, image) and std.mem.eql(u8, e.name[0..e.len], text)) {
        out.* = @ptrFromInt(i + 1);
        return 0;
    };
    if (count == entries.len or text.len > 512) return 1;
    entries[count] = .{ .image = image, .len = text.len };
    @memcpy(entries[count].name[0..text.len], text);
    count += 1;
    out.* = @ptrFromInt(count);
    return 0;
}
fn attribute(out: *c_int, attr: abi.FunctionAttribute, _: abi.Function) callconv(.c) abi.Result {
    out.* = if (attr == .shared_size_bytes) 0 else 1024;
    return 0;
}
fn setAttribute(handle: abi.Function, attr: abi.FunctionAttribute, value: c_int) callconv(.c) abi.Result {
    if (attr == .max_dynamic_shared_size_bytes) {
        const e = &entries[@intFromPtr(handle.?) - 1];
        e.shared = @max(e.shared, @as(u32, @intCast(value)));
        e.opt_in = true;
    }
    return 0;
}
fn occupancy(out: *c_int, handle: abi.Function, threads: c_int, shared: usize) callconv(.c) abi.Result {
    const e = &entries[@intFromPtr(handle.?) - 1];
    e.threads = @intCast(threads);
    e.shared = @max(e.shared, @as(u32, @intCast(shared)));
    out.* = 1;
    return 0;
}
fn load(out: *abi.Module, _: ?*const anyopaque) callconv(.c) abi.Result {
    out.* = @ptrFromInt(32);
    return 0;
}
fn unload(_: abi.Module) callconv(.c) abi.Result {
    return 0;
}
pub fn collect() !void {
    return collectWithLimit(262144);
}
pub fn collectWithLimit(optin: u32) !void {
    count = 0;
    var driver: cuda.Driver = undefined;
    driver.api.cuModuleGetFunction = function;
    driver.api.cuFuncGetAttribute = attribute;
    driver.api.cuFuncSetAttribute = setAttribute;
    driver.api.cuOccupancyMaxActiveBlocksPerMultiprocessor = occupancy;
    driver.api.cuModuleLoadData = load;
    driver.api.cuModuleUnload = unload;
    var m: [images.len]cuda.Module = undefined;
    for (&m, 0..) |*mod, i| mod.* = .{ .d = &driver, .handle = @ptrFromInt(i + 1) };
    @import("catalog_geometry.zig").experts(try dk.exl3.Functions.resolve(m[0], m[1], m[2], m[3], m[19], m[24], m[26], m[28], optin));
    @import("catalog_geometry.zig").dense(try dk.dense.Functions.resolve(m[4], m[5], m[6]));
    @import("catalog_geometry.zig").lin2(try dk.dense.Lin2.resolve(m[30], .{ .lanes = true, .linear = true, .cluster = true }));
    @import("catalog_geometry.zig").lin2(try dk.dense.Lin2.resolve(m[30], .{ .lanes = true, .linear = true, .cluster = true, .pf = false }));
    @import("catalog_geometry.zig").operators(try dk.ops.Functions.resolve(m[7], m[8], m[9], m[10], m[11], .{ m[12], m[13], m[14], m[15] }, m[16], m[17], m[18], m[20], m[21], m[22], m[23], m[25], m[29], 262144));
    const kv = try dk.kv_glue.Functions.resolve(m[27]);
    entries[@intFromPtr(kv.norm_store.handle.?) - 1].threads = 128;
    var sampling = try dk.sampling.Functions.load(&driver);
    @import("catalog_geometry.zig").sampling(sampling);
    sampling.unload();
    // Existing common T20 kernels are registered in this composition as well.
    _ = try m[32].function(cuda.exl3.rot_in_symbol);
    inline for (cuda.exl3.variants) |v| _ = try m[32].function(cuda.exl3.symbol(v.halves, v.codebook, v.warps));
    inline for (.{ "tf_exl3_group", "tf_exl3_experts_rot_in", "tf_exl3_grouped_0", "tf_exl3_grouped_1", "tf_exl3_grouped_2", "tf_exl3_gateup_epilogue", "tf_exl3_down_combine", "tf_exl3_plan_rotate", "tf_exl3_plan_finish" }) |name| _ = try m[33].function(name);
    inline for (.{ "tf_exl3_rotate_f", "tf_exl3_rotate_h", "tf_exl3_codebook", "tf_exl3_unpack", "tf_exl3_tc_0", "tf_exl3_tc_1", "tf_exl3_tc_2", "tf_exl3_prompt_0", "tf_exl3_prompt_1", "tf_exl3_prompt_2" }) |name| _ = try m[34].function(name);
    for (entries[0..count]) |entry| @import("catalog_geometry.zig").base(entry.image, entry.name[0..entry.len]);
}
pub fn main(init: std.process.Init) !void {
    try collect();
    var out: std.Io.Writer.Allocating = .init(init.gpa);
    defer out.deinit();
    for (entries[0..count]) |e| try out.writer.print("{s}|{s}|{d}|{d}|{d}\n", .{ e.image, e.name[0..e.len], e.threads, e.shared, e.cluster });
    try std.Io.File.stdout().writeStreamingAll(init.io, out.written());
}
