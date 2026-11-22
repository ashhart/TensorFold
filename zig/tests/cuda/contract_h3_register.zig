//! MiniMax-H3's CUDA kernels: the video transformer's, the image decoder's and the prompt's, on the contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const h3_k = @import("contract_h3.zig");
const h3_prep = @import("contract_h3_prep.zig");
const h3_topk_k = @import("contract_h3_topk.zig");
const h3d_k = @import("contract_h3_image_decoder.zig");
const h3p_k = @import("contract_h3_image_prompt.zig");

const g = core.contract.gates;

const images = struct {
    pub const h3 align(16) = @embedFile("fatbin_h3").*;
    pub const decoder align(16) = @embedFile("fatbin_h3_image_decoder").*;
    pub const prompt align(16) = @embedFile("fatbin_h3_image_prompt").*;
};

/// H3's loaded images and kernel rigs: the gates point into it, so it lives in the caller's frame until `deinit`.
pub const Loaded = struct {
    h3_mod: cuda.Module,
    h3d_mod: cuda.Module,
    h3p_mod: cuda.Module,
    h3_list: [std.enums.values(h3_k.Which).len]h3_k.H3,
    h3_topk: h3_topk_k.TopK,
    /// Each decoder and prompt kernel twice: served, then its planted control.
    h3d_list: [2 * std.enums.values(h3d_k.Which).len]h3d_k.Dec,
    h3p_list: [2 * std.enums.values(h3p_k.Which).len]h3p_k.Prm,

    /// Frees every rig's arena and unloads the three images; call once the gates are done.
    pub fn deinit(l: *Loaded) void {
        for (&l.h3_list) |*m| m.deinit();
        l.h3_topk.deinit();
        for (&l.h3d_list) |*m| m.deinit();
        for (&l.h3p_list) |*m| m.deinit();
        l.h3p_mod.unload();
        l.h3d_mod.unload();
        l.h3_mod.unload();
    }
};

/// Loads H3's three images and appends every kernel they register; `l` must outlive the gates.
pub fn append(l: *Loaded, a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !void {
    l.h3_mod = try cuda.Module.load(gpu.d, &images.h3);
    const h3_which = comptime std.enums.values(h3_k.Which);
    for (h3_which, &l.h3_list) |w, *m| {
        const f = try l.h3_mod.function(h3_k.symbol(w));
        if (w == .attention) try f.allowDynamicShared(h3_prep.attention_shared);
        m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .f = f, .arena = std.heap.ArenaAllocator.init(a) };
    }
    for (&l.h3_list) |*m| try kernels.append(a, m.kernel());
    l.h3_topk = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try l.h3_mod.function("h3_topk") };
    try kernels.append(a, l.h3_topk.kernel());
    l.h3d_mod = try cuda.Module.load(gpu.d, &images.decoder);
    const h3d_which = comptime std.enums.values(h3d_k.Which);
    for (&l.h3d_list, 0..) |*m, i| {
        const w = h3d_which[i % h3d_which.len];
        m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .f = try l.h3d_mod.function(h3d_k.symbol(w)), .arena = std.heap.ArenaAllocator.init(a), .planted = i >= h3d_which.len };
    }
    for (&l.h3d_list) |*m| try kernels.append(a, m.kernel());
    l.h3p_mod = try cuda.Module.load(gpu.d, &images.prompt);
    const h3p_which = comptime std.enums.values(h3p_k.Which);
    for (&l.h3p_list, 0..) |*m, i| {
        const w = h3p_which[i % h3p_which.len];
        m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .f = try l.h3p_mod.function(h3p_k.symbol(w)), .arena = std.heap.ArenaAllocator.init(a), .planted = i >= h3p_which.len };
    }
    for (&l.h3p_list) |*m| try kernels.append(a, m.kernel());
}
