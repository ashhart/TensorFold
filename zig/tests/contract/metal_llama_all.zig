//! The llama family's registrations as one set: every pipeline the loader builds, and gate 5 on this GPU.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const limits_k = @import("metal_limits.zig");
const llama = @import("metal_llama.zig");
const dense = @import("metal_llama_dense.zig");
const attention = @import("metal_llama_attention.zig");
const controls = @import("metal_llama_controls.zig");
const int4 = @import("metal_llama_int4.zig");
const g = contract.gates;
const family = llama.family;

const matvec_rows = [_]u32{ 1, 2, 4, 8, 16, 32 };
const matvec_names = [_][]const u8{ "ll_matvec_1", "ll_matvec_2", "ll_matvec_4", "ll_matvec_8", "ll_matvec_16", "ll_matvec_32" };
const q4_names = [_][]const u8{ "ll_q4_matvec_1", "ll_q4_matvec_2", "ll_q4_matvec_4", "ll_q4_matvec_8", "ll_q4_matvec_16", "ll_q4_matvec_32" };

fn wanted(only: ?[]const u8, name: []const u8) bool {
    const text = only orelse return true;
    return std.mem.indexOf(u8, name, text) != null;
}

/// The probes made for a run, freed together.
pub const Set = struct {
    glue: std.ArrayList(*llama.Glue) = .empty,
    matvec: std.ArrayList(*dense.Matvec) = .empty,
    q4: std.ArrayList(*int4.Matvec) = .empty,
    dequant: std.ArrayList(*int4.Dequant) = .empty,
    gemm: std.ArrayList(*dense.Gemm) = .empty,
    argmax: std.ArrayList(*llama.Argmax) = .empty,
    decode: std.ArrayList(*attention.DecodeVec) = .empty,
    merge: std.ArrayList(*attention.MergeVec) = .empty,
    prompt: std.ArrayList(*attention.Prompt) = .empty,

    pub fn deinit(s: *Set, a: std.mem.Allocator) void {
        for (s.glue.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        for (s.matvec.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        for (s.gemm.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        for (s.q4.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        for (s.dequant.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        s.dequant.deinit(a);
        for (s.prompt.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        for (s.argmax.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        for (s.decode.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        for (s.merge.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        s.argmax.deinit(a);
        s.decode.deinit(a);
        s.merge.deinit(a);
        s.glue.deinit(a);
        s.matvec.deinit(a);
        s.gemm.deinit(a);
        s.q4.deinit(a);
        s.prompt.deinit(a);
    }

    /// Every registration whose name holds `only`; the roll call checks the registrations cover every pipeline.
    pub fn collect(s: *Set, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, tensor: bool, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        var functions: [family.pipelines + 1][]const u8 = undefined;
        var n: usize = 0;
        for (std.enums.values(llama.Which)) |w| {
            functions[n] = llama.function(w);
            n += 1;
            if (!wanted(only, llama.spec(w).name)) continue;
            const p = try a.create(llama.Glue);
            errdefer a.destroy(p);
            p.* = try llama.Glue.init(a, device, queue, w, .none);
            try s.glue.append(a, p);
            try kernels.append(a, p.kernel());
        }
        functions[n] = "ll_argmax";
        n += 1;
        if (wanted(only, "llama glue ll_argmax")) {
            const p = try a.create(llama.Argmax);
            errdefer a.destroy(p);
            p.* = try llama.Argmax.init(a, device, queue, .none);
            try s.argmax.append(a, p);
            try kernels.append(a, p.kernel());
        }
        for ([_][]const u8{ "ll_decode_vec", "ll_merge_vec" }) |f| {
            functions[n] = f;
            n += 1;
        }
        if (wanted(only, "llama glue ll_decode_vec")) {
            const p = try a.create(attention.DecodeVec);
            errdefer a.destroy(p);
            p.* = try attention.DecodeVec.init(a, device, queue, .none);
            try s.decode.append(a, p);
            try kernels.append(a, p.kernel());
        }
        if (wanted(only, "llama glue ll_merge_vec")) {
            const p = try a.create(attention.MergeVec);
            errdefer a.destroy(p);
            p.* = try attention.MergeVec.init(a, device, queue, .none);
            try s.merge.append(a, p);
            try kernels.append(a, p.kernel());
        }
        for (matvec_rows, matvec_names) |rows, fun| {
            functions[n] = fun;
            n += 1;
            var name_buf: [40]u8 = undefined;
            if (!wanted(only, try std.fmt.bufPrint(&name_buf, "llama glue {s}", .{fun}))) continue;
            const p = try a.create(dense.Matvec);
            errdefer a.destroy(p);
            p.* = try dense.Matvec.init(a, device, queue, rows, .none);
            try s.matvec.append(a, p);
            try kernels.append(a, p.kernel());
        }
        for (matvec_rows, q4_names) |rows, fun| {
            functions[n] = fun;
            n += 1;
            var name_buf: [40]u8 = undefined;
            if (!wanted(only, try std.fmt.bufPrint(&name_buf, "llama glue {s}", .{fun}))) continue;
            const p = try a.create(int4.Matvec);
            errdefer a.destroy(p);
            p.* = try int4.Matvec.init(a, device, queue, rows, .none);
            try s.q4.append(a, p);
            try kernels.append(a, p.kernel());
        }
        functions[n] = "ll_q4_dequant";
        n += 1;
        if (wanted(only, "llama glue ll_q4_dequant")) {
            const p = try a.create(int4.Dequant);
            errdefer a.destroy(p);
            p.* = try int4.Dequant.init(a, device, queue, .none);
            try s.dequant.append(a, p);
            try kernels.append(a, p.kernel());
        }
        for ([_][2]bool{ .{ false, false }, .{ false, true }, .{ true, false }, .{ true, true } }) |form| {
            const fun = family.gemm_names[@as(usize, @intFromBool(form[0])) * 2 + @intFromBool(form[1])];
            functions[n] = fun;
            n += 1;
            var name_buf: [40]u8 = undefined;
            if (!tensor or !wanted(only, try std.fmt.bufPrint(&name_buf, "llama gemm {s}", .{fun}))) continue;
            const p = try a.create(dense.Gemm);
            errdefer a.destroy(p);
            p.* = try dense.Gemm.init(a, device, queue, form[0], form[1], .none);
            try s.gemm.append(a, p);
            try kernels.append(a, p.kernel());
        }
        functions[n] = family.attention_name;
        n += 1;
        for (family.geometries) |geo| {
            var name_buf: [64]u8 = undefined;
            if (!tensor or !wanted(only, try std.fmt.bufPrint(&name_buf, "llama {s} attention ll_prompt_attention", .{geo.name}))) continue;
            const p = try a.create(attention.Prompt);
            errdefer a.destroy(p);
            p.* = try attention.Prompt.init(a, device, queue, geo, .none);
            try s.prompt.append(a, p);
            try kernels.append(a, p.kernel());
        }
        try family.requireComplete(functions[0..n]);
        try s.plant(a, device, queue, tensor, kernels, only);
    }

    /// Each control: its kernel built with one planted fault, registered beside the kernels.
    fn plant(s: *Set, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, tensor: bool, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (std.enums.values(controls.Mutation)) |m| {
            if (m == .none or !wanted(only, m.name())) continue;
            const glue: ?llama.Which = switch (m) {
                .embed_own_row => .embed,
                .norm_mean_over_threads => .norm,
                .add_wrong_sign => .add,
                .add_norm_no_residual => .add_norm,
                .silu_swapped => .silu,
                .prep_first_position => .prep,
                else => null,
            };
            if (glue) |w| {
                const p = try a.create(llama.Glue);
                errdefer a.destroy(p);
                p.* = try llama.Glue.init(a, device, queue, w, m);
                try s.glue.append(a, p);
                try kernels.append(a, p.kernel());
                continue;
            }
            switch (m) {
                .argmax_nan_loses => {
                    const p = try a.create(llama.Argmax);
                    errdefer a.destroy(p);
                    p.* = try llama.Argmax.init(a, device, queue, m);
                    try s.argmax.append(a, p);
                    try kernels.append(a, p.kernel());
                },
                .decode_no_rescale => {
                    const p = try a.create(attention.DecodeVec);
                    errdefer a.destroy(p);
                    p.* = try attention.DecodeVec.init(a, device, queue, m);
                    try s.decode.append(a, p);
                    try kernels.append(a, p.kernel());
                },
                .merge_unscaled_mass => {
                    const p = try a.create(attention.MergeVec);
                    errdefer a.destroy(p);
                    p.* = try attention.MergeVec.init(a, device, queue, m);
                    try s.merge.append(a, p);
                    try kernels.append(a, p.kernel());
                },
                .matvec_drops_tail => {
                    const p = try a.create(dense.Matvec);
                    errdefer a.destroy(p);
                    p.* = try dense.Matvec.init(a, device, queue, 4, m);
                    try s.matvec.append(a, p);
                    try kernels.append(a, p.kernel());
                },
                .gemm_untransposed => if (tensor) {
                    const p = try a.create(dense.Gemm);
                    errdefer a.destroy(p);
                    p.* = try dense.Gemm.init(a, device, queue, false, false, m);
                    try s.gemm.append(a, p);
                    try kernels.append(a, p.kernel());
                },
                .prompt_sees_next_key => if (tensor) {
                    const p = try a.create(attention.Prompt);
                    errdefer a.destroy(p);
                    p.* = try attention.Prompt.init(a, device, queue, family.nanbeige41, m);
                    try s.prompt.append(a, p);
                    try kernels.append(a, p.kernel());
                },
                .q4_matvec_first_zero => {
                    const p = try a.create(int4.Matvec);
                    errdefer a.destroy(p);
                    p.* = try int4.Matvec.init(a, device, queue, 4, m);
                    try s.q4.append(a, p);
                    try kernels.append(a, p.kernel());
                },
                .q4_dequant_reversed_nibbles => {
                    const p = try a.create(int4.Dequant);
                    errdefer a.destroy(p);
                    p.* = try int4.Dequant.init(a, device, queue, m);
                    try s.dequant.append(a, p);
                    try kernels.append(a, p.kernel());
                },
                else => unreachable,
            }
        }
    }
};

/// Gate 5 for the family's pipelines as built here: the threads each launch takes and its static threadgroup bytes.
pub fn limits(a: std.mem.Allocator, device: mtl.Device, tensor: bool, only: ?[]const u8) !usize {
    var problems: usize = 0;
    const glue = try family.glueSource(a);
    defer a.free(glue);
    const Decl = struct { name: []const u8, threads: u32, tg: u32 };
    const decls = [_]Decl{
        .{ .name = "llama glue ll_embed", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_norm", .threads = 256, .tg = 128 },
        .{ .name = "llama glue ll_add", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_add_norm", .threads = 256, .tg = 128 },
        .{ .name = "llama glue ll_prep", .threads = 32, .tg = 0 },
        .{ .name = "llama glue ll_silu_gu", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_decode_vec", .threads = 32 * (family.nanbeige43.heads / family.nanbeige43.kv_heads), .tg = 0 },
        .{ .name = "llama glue ll_merge_vec", .threads = 1024, .tg = 32 * 32 * 4 },
        .{ .name = "llama glue ll_argmax", .threads = 1024, .tg = 32 * 8 },
        .{ .name = "llama glue ll_matvec_1", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_matvec_2", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_matvec_4", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_matvec_8", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_matvec_16", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_matvec_32", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_q4_matvec_1", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_q4_matvec_2", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_q4_matvec_4", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_q4_matvec_8", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_q4_matvec_16", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_q4_matvec_32", .threads = 256, .tg = 0 },
        .{ .name = "llama glue ll_q4_dequant", .threads = 256, .tg = 0 },
    };
    for (decls) |d| {
        if (!wanted(only, d.name)) continue;
        const fun = d.name[std.mem.lastIndexOfScalar(u8, d.name, ' ').? + 1 ..];
        problems += try limits_k.onThisGpu(a, device, .{ .decl = .{ .name = d.name, .threads = d.threads, .tg_memory = d.tg, .offline = false }, .function = fun, .here = glue, .pre_m5 = glue }, false);
    }
    if (!tensor) return problems;
    const gemm = try family.gemmSource(device, a);
    defer a.free(gemm);
    for (family.gemm_names) |fun| {
        const name = try std.fmt.allocPrint(a, "llama gemm {s}", .{fun});
        defer a.free(name);
        if (!wanted(only, name)) continue;
        problems += try limits_k.onThisGpu(a, device, .{ .decl = .{ .name = name, .threads = 256, .tg_memory = 0, .offline = false }, .function = fun, .here = gemm, .pre_m5 = gemm }, false);
    }
    for (family.geometries) |geo| {
        const name = try std.fmt.allocPrint(a, "llama {s} attention ll_prompt_attention", .{geo.name});
        defer a.free(name);
        if (!wanted(only, name)) continue;
        const src = try family.attentionSource(device, a, geo);
        defer a.free(src);
        // pt[32 x 136] floats and rowmax[2][32]
        problems += try limits_k.onThisGpu(a, device, .{ .decl = .{ .name = name, .threads = 64, .tg_memory = 32 * 136 * 4 + 2 * 32 * 4, .offline = false }, .function = family.attention_name, .here = src, .pre_m5 = src }, false);
    }
    return problems;
}
