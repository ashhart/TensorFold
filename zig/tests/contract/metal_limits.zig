//! Gate 5 on Metal: each registered pipeline on this GPU (Metal's reflection), and its M1/M2 limits compiled offline.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const topk_text = tf.bf16_topk_gpu.source;
const rig_ = @import("rig.zig");

const limits = contract.limits;
const registry = contract.registry;

/// A registered pipeline as built here: its declaration, function, this GPU's source and the pre-M5 source.
pub const Launch = struct { decl: limits.MetalLaunch, function: []const u8, here: []const u8, pre_m5: []const u8 };

/// The pre-M5 text of a source that includes nax.h (frags.source's branch for GPUs without tensor units).
fn preM5(a: std.mem.Allocator, head: []const u8, text: []const u8) ![]u8 {
    const body = try std.mem.replaceOwned(u8, a, text, "#include \"../nax.h\"", sources.nax);
    defer a.free(body);
    return std.mem.concat(a, u8, &.{ head, "#define TF_SIMD_FRAGS 1\n", body });
}

/// Each registration's function is its label's last word; the label's words say which source it compiles.
fn launchOf(a: std.mem.Allocator, device: mtl.Device, d: limits.MetalLaunch) !Launch {
    const fun = d.name[std.mem.lastIndexOfScalar(u8, d.name, ' ').? + 1 ..];
    var text: []const u8 = undefined;
    var head: []const u8 = "";
    var frags = false;
    const has = struct {
        fn f(s: []const u8, p: []const u8) bool {
            return std.mem.indexOf(u8, s, p) != null;
        }
    }.f;
    if (std.mem.startsWith(u8, d.name, "qwen4-native ")) {
        const body = try @import("qwen4").family.limits_sources.source(a, d.name);
        return .{ .decl = d, .function = fun, .here = body, .pre_m5 = body };
    }
    if (std.mem.startsWith(u8, d.name, "hc-pointwise ")) {
        const body = try @import("metal_hc_pointwise.zig").text(a);
        return .{ .decl = d, .function = fun, .here = body, .pre_m5 = body };
    }
    if (std.mem.startsWith(u8, d.name, "hc-activation ")) {
        const body = try @import("metal_hc_activation.zig").text(a);
        return .{ .decl = d, .function = fun, .here = body, .pre_m5 = body };
    }
    if (std.mem.startsWith(u8, d.name, "weight-only ")) {
        const body = try @import("metal_weight_only.zig").text(a);
        return .{ .decl = d, .function = fun, .here = body, .pre_m5 = body };
    }
    if (std.mem.startsWith(u8, d.name, "flashnext FSR ")) {
        const fsr = try tf.flashnext_fsr_contract.source(a, fun);
        const pre = if (fsr.fragments) try preM5(a, "", fsr.text) else fsr.text;
        const here = if (fsr.fragments and device.tensorUnits()) try std.mem.replaceOwned(u8, a, fsr.text, "#include \"../nax.h\"", sources.nax) else pre;
        return .{ .decl = d, .function = fun, .here = here, .pre_m5 = pre };
    }
    if (std.mem.startsWith(u8, d.name, "flashnext vision ") or std.mem.startsWith(u8, d.name, "flashnext rotary ")) {
        const w: tf.flashnext_vision_contract.Which = if (std.mem.eql(u8, fun, "fz_mask_rows")) .mask else if (std.mem.eql(u8, fun, "fn_attn_prep_mrope")) .mrope else if (std.mem.indexOf(u8, d.name, " target ") != null) .target else if (std.mem.indexOf(u8, d.name, " mtp ") != null) .mtp else if (std.mem.indexOf(u8, d.name, " absolute ") != null) .absolute else if (std.mem.indexOf(u8, d.name, " positioned ") != null) .positioned else .pooled;
        text = tf.flashnext_vision_contract.source(w);
        return .{ .decl = d, .function = fun, .here = text, .pre_m5 = text };
    }
    if (std.mem.startsWith(u8, d.name, "metal lfm2 ")) return .{ .decl = d, .function = fun, .here = sources.lfm2, .pre_m5 = sources.lfm2 };
    if (std.mem.startsWith(u8, d.name, "nemotron paged ")) return pagedLaunch(a, device, d, fun);
    if (std.mem.startsWith(u8, d.name, "nemotron attention ")) return attentionLaunch(a, device, d, fun);
    if (std.mem.startsWith(u8, d.name, "drama3 ")) {
        const text_d3 = try @import("metal_drama3.zig").source(a, d.name);
        return .{ .decl = d, .function = fun, .here = text_d3, .pre_m5 = text_d3 };
    }
    if (std.mem.startsWith(u8, fun, "tf_ctl_")) {
        text = sources.core_contract_controls;
        if (has(fun, "big_tg")) head = "#define TF_CTL_BIG_TG 1\n";
    } else if (std.mem.startsWith(u8, d.name, "h3 gemm ")) {
        text = for (sources.prefill) |f| {
            if (std.mem.eql(u8, f.name, "gemm_nax")) break f.text;
        } else return error.UnknownLaunch;
        frags = true;
    } else if (std.mem.startsWith(u8, d.name, "h3 ")) {
        text = @import("h3_kernels").source;
    } else if (std.mem.startsWith(u8, d.name, "glm gather ") or std.mem.startsWith(u8, d.name, "glm rows ")) {
        text = tf.glm.wide.gather_source;
    } else if (std.mem.startsWith(u8, d.name, "nemotron experts ")) {
        text = sources.nemotron_experts;
    } else if (std.mem.startsWith(u8, d.name, "nemotron geometry ")) {
        text = if (has(d.name, " up ")) tf.nemotron.kernels.geo_up else tf.nemotron.kernels.geo_down;
    } else if (std.mem.startsWith(u8, d.name, "glm ")) {
        const key = d.name[4..std.mem.indexOfScalarPos(u8, d.name, 4, ' ').?];
        text = for (sources.glm.all) |k| {
            if (std.mem.eql(u8, k.key, key)) break k.source;
        } else return error.UnknownLaunch;
    } else if (std.mem.startsWith(u8, d.name, "qwen vision ")) {
        text = sources.core_qwen_vision;
        frags = true;
    } else if (std.mem.startsWith(u8, d.name, "qwen image ")) {
        text = sources.qwen_image;
        frags = true;
    } else if (std.mem.startsWith(u8, d.name, "moss image gemm ")) {
        text = sources.core_qwen_vision;
        frags = true;
    } else if (std.mem.startsWith(u8, d.name, "moss image ")) {
        text = sources.moss_image;
        frags = true;
    } else if (std.mem.startsWith(u8, d.name, "core deltanet ")) {
        text = sources.core_deltanet;
    } else if (std.mem.startsWith(u8, d.name, "qwen27 ")) {
        text = if (has(d.name, " copy ")) tf.qwen27.runtime_sources.copy else tf.qwen27.runtime_sources.glue;
    } else if (has(d.name, "tile_qmm")) text = sources.core_tile_qmm else if (has(d.name, "softmax")) text = sources.ops_softmax else if (has(d.name, "rms")) text = sources.ops_embed_norm else if (has(d.name, "top-k")) text = topk_text else if (has(d.name, "row projection")) text = sources.core_row_projection else if (has(d.name, "simd_qmm_bits")) text = sources.core_simd_qmm_bits else if (has(d.name, "simd_qmm")) text = sources.core_simd_qmm else if (has(d.name, "affine_mm")) {
        const bits: u8 = if (has(d.name, "q8")) 8 else 4;
        const group: u16 = if (has(d.name, "g32")) 32 else 64;
        head = try std.fmt.allocPrint(a, "#define TF_BITS {d}\n#define TF_GROUP {d}\n#define TF_OUT_T bfloat\n", .{ bits, group });
        text = sources.core_affine_mm;
        frags = true;
    } else if (has(d.name, "lane projection")) {
        const sk: usize = if (has(d.name, "sk4")) 4 else 1;
        const full = try tf.lane_projection.source(a, .{ .n = 2048, .k = 2048, .format = .{ .bits = 6, .group = 32 }, .sk = sk });
        head = full[0..std.mem.indexOf(u8, full, sources.core_lane_projection[0..64]).?];
        text = sources.core_lane_projection;
        frags = true;
    } else return error.UnknownLaunch;
    const pre = if (frags) try preM5(a, head, text) else try std.mem.concat(a, u8, &.{ head, text });
    const here = if (frags and device.tensorUnits()) blk: {
        const body = try std.mem.replaceOwned(u8, a, text, "#include \"../nax.h\"", sources.nax);
        break :blk try std.mem.concat(a, u8, &.{ head, body });
    } else pre;
    return .{ .decl = d, .function = fun, .here = here, .pre_m5 = pre };
}

/// Main's Nemotron attention library as kernels.zig and prefill_kernels.zig build it here, and before the M5.
fn attentionLaunch(a: std.mem.Allocator, device: mtl.Device, d: limits.MetalLaunch, fun: []const u8) !Launch {
    const tensor = device.tensorUnits();
    if (std.mem.indexOf(u8, d.name, " partial ") != null or std.mem.indexOf(u8, d.name, " merge ") != null) {
        const text = for (sources.nemotron.all) |k| {
            if (std.mem.eql(u8, k.function, fun)) break k.source;
        } else return error.UnknownLaunch;
        const pre = if (std.mem.indexOf(u8, d.name, " partial ") != null) try tf.nemotron.simd_attention.rewrite(a, text) else text;
        return .{ .decl = d, .function = fun, .here = if (tensor) text else pre, .pre_m5 = pre };
    }
    if (std.mem.indexOf(u8, d.name, " tree ") != null) {
        const pre = try std.mem.concat(a, u8, &.{ "#define TF_SIMD_LAYOUT 1\n", sources.nemotron_tree });
        return .{ .decl = d, .function = fun, .here = if (tensor) sources.nemotron_tree else pre, .pre_m5 = pre };
    }
    if (std.mem.indexOf(u8, d.name, " prompt ") != null) {
        const file = if (std.mem.indexOf(u8, fun, "kv_put") != null) "glue" else "attention_nax";
        const text = for (sources.prefill) |f| {
            if (std.mem.eql(u8, f.name, file)) break f.text;
        } else return error.UnknownLaunch;
        const pre = try preM5(a, "", text);
        return .{ .decl = d, .function = fun, .here = if (tensor) try std.mem.replaceOwned(u8, a, text, "#include \"../nax.h\"", sources.nax) else pre, .pre_m5 = pre };
    }
    return .{ .decl = d, .function = fun, .here = sources.nemotron_glue, .pre_m5 = sources.nemotron_glue };
}

/// A Nemotron paged twin's library as its loader would build it here, and before the M5.
fn pagedLaunch(a: std.mem.Allocator, device: mtl.Device, d: limits.MetalLaunch, fun: []const u8) !Launch {
    const paged = sources.nemotron_paged;
    const tensor = device.tensorUnits();
    if (std.mem.indexOf(u8, d.name, " partial ") != null) {
        const pre = try tf.nemotron.simd_attention.rewrite(a, paged.partial);
        return .{ .decl = d, .function = fun, .here = if (tensor) paged.partial else pre, .pre_m5 = pre };
    }
    if (std.mem.indexOf(u8, d.name, " tail ") != null) {
        const pre = try std.mem.concat(a, u8, &.{ "#define TF_SIMD_LAYOUT 1\n", paged.tree });
        return .{ .decl = d, .function = fun, .here = if (tensor) paged.tree else pre, .pre_m5 = pre };
    }
    if (std.mem.indexOf(u8, d.name, " prompt ") != null) {
        const pre = try preM5(a, "", paged.prefill);
        return .{ .decl = d, .function = fun, .here = if (tensor) try std.mem.replaceOwned(u8, a, paged.prefill, "#include \"../nax.h\"", sources.nax) else pre, .pre_m5 = pre };
    }
    return .{ .decl = d, .function = fun, .here = paged.glue, .pre_m5 = paged.glue };
}

/// One launch on this GPU: the pipeline's own thread limit and threadgroup bytes against its declaration.
pub fn onThisGpu(a: std.mem.Allocator, device: mtl.Device, l: Launch, control: bool) !usize {
    const p = rig_.pipeline(device, l.here, l.function) catch |err| {
        std.debug.print("{s} {s}: no pipeline on this GPU ({t})\n", .{ if (control) "RED " else "FAIL", l.decl.name, err });
        return 1;
    };
    defer p.deinit();
    _ = a;
    const most = p.maxThreads();
    const tg = rig_.tgMemory(p);
    const cap = mtl.objc.msg(usize, device.id, "maxThreadgroupMemoryLength", .{});
    var n: usize = 0;
    if (most < l.decl.threads and !l.decl.adapts) n += 1;
    if (tg > l.decl.tg_memory or tg + l.decl.dynamic > cap) n += 1;
    const word = if (n == 0) "PASS" else if (control) "RED " else "FAIL";
    std.debug.print("{s} limits here {s}: {d} threads launched, the pipeline takes {d}; {d} threadgroup bytes and {d} set (declared at most {d}, the GPU's most {d})\n", .{ word, l.decl.name, l.decl.threads, most, tg, l.decl.dynamic, l.decl.tg_memory, cap });
    return n;
}

/// Gate 5: declarations against every family, each pipeline on this GPU, and offline M1/M2 compiles when asked.
pub fn gate(a: std.mem.Allocator, io: std.Io, device: mtl.Device, offline: ?[]const u8) !usize {
    var n = try registry.checkMetal(a, &registry.all_metal, false);
    std.debug.print("{s} limits: {d} registered Metal launches against M1-M5, {d} problems\n", .{ if (n == 0) "PASS" else "FAIL", registry.all_metal.len, n });
    for (registry.all_metal) |d| n += try onThisGpu(a, device, try launchOf(a, device, d), false);
    for (registry.metal_controls) |d| {
        const red = try registry.checkMetal(a, &.{d}, true) + try onThisGpu(a, device, try launchOf(a, device, d), true);
        if (red == 0) n += 1;
        std.debug.print("CONTROL {s} {s}: {d} limit problems\n", .{ if (red > 0) "red as planted," else "MISSED", d.name, red });
    }
    if (offline) |dir| n += try @import("metal_offline.zig").run(a, io, dir, &(registry.all_metal ++ registry.metal_controls), launchOf, device);
    return n;
}
