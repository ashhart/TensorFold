//! tf-kernel-contract: every registered Metal kernel through gates 1-4, the planted controls, then gate 5's limits.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rows_k = @import("metal_rows.zig");
const affine_k = @import("metal_affine.zig");
const topk_k = @import("metal_topk.zig");
const tile_k = @import("metal_tile.zig");
const limits_k = @import("metal_limits.zig");
const h3_k = @import("metal_h3.zig");
const h3_topk = @import("metal_h3_topk.zig");
const drama3_k = @import("metal_drama3.zig");
const drama_nvfp4 = @import("metal_drama3_nvfp4.zig");
const drama_router = @import("metal_drama_router.zig");
const drama_combine = @import("metal_drama_combine.zig");
const drama_cfg = @import("metal_drama_cfg.zig");
const glm_eh_k = @import("metal_glm_eh.zig");
const glm_rows_k = @import("metal_glm_rows.zig");
const pair_k = @import("metal_pair.zig");
const pair_controls = @import("metal_pair_controls.zig");

const flash_pool = @import("metal_flashnext_pool.zig");
const flash = @import("metal_flashnext.zig");
const fsr = @import("metal_fsr.zig");
const gemma = @import("metal_gemma4.zig");
const gemma_embed = @import("metal_gemma4_embed.zig");
const gemma_mixed = @import("metal_gemma4_mixed.zig");
const lfm2_argmax = @import("metal_lfm2_argmax.zig");
const lfm2_streams = @import("metal_lfm2_streams.zig");
const lfm2_tap = @import("metal_lfm2_tap.zig");
const lfm2_controls = @import("metal_lfm2_controls.zig");
const g = contract.gates;
const usage = "usage: tf-kernel-contract [--only TEXT] [--no-gates] [--no-limits] [--offline DIR] [--seed N] [--repro] [--rotary-long] [--rotary-accuracy FILE]";

const Opts = struct { only: ?[]const u8 = null, gates: bool = true, limits: bool = true, offline: ?[]const u8 = null, seed: u64 = 0x6b63, repro: bool = false, rotary_long: bool = false, rotary_accuracy: ?[]const u8 = null };

fn parse(args: []const [:0]const u8) !Opts {
    var o = Opts{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--rotary-long")) o.rotary_long = true else if (std.mem.eql(u8, a, "--repro")) o.repro = true else if (std.mem.eql(u8, a, "--no-gates")) o.gates = false else if (std.mem.eql(u8, a, "--no-limits")) o.limits = false else if (i + 1 < args.len and std.mem.eql(u8, a, "--only")) {
            i += 1;
            o.only = args[i];
        } else if (i + 1 < args.len and std.mem.eql(u8, a, "--offline")) {
            i += 1;
            o.offline = args[i];
        } else if (i + 1 < args.len and std.mem.eql(u8, a, "--seed")) {
            i += 1;
            o.seed = try std.fmt.parseInt(u64, args[i], 0);
        } else if (i + 1 < args.len and std.mem.eql(u8, a, "--rotary-accuracy")) {
            i += 1;
            o.rotary_accuracy = args[i];
        } else {
            std.debug.print("{s}\n", .{usage});
            return error.Usage;
        }
    }
    return o;
}

const Reg = struct { v: affine_k.Variant, name: []const u8 };

const common = [_]Reg{
    .{ .v = .{ .kind = .row, .bits = 4, .group = 64 }, .name = "metal row projection q4 g64" },
    .{ .v = .{ .kind = .row, .bits = 6, .group = 32 }, .name = "metal row projection q6 g32" },
    .{ .v = .{ .kind = .row, .bits = 8, .group = 64 }, .name = "metal row projection q8 g64" },
    .{ .v = .{ .kind = .row, .bits = 3, .group = 32 }, .name = "metal row projection q3 g32" },
    .{ .v = .{ .kind = .affine_mm, .bits = 4, .group = 64 }, .name = "metal affine_mm q4 g64" },
    .{ .v = .{ .kind = .affine_mm, .bits = 4, .group = 64, .f32_out = true }, .name = "metal affine_mm q4 g64 fp32 out" },
    .{ .v = .{ .kind = .affine_mm, .bits = 8, .group = 32 }, .name = "metal affine_mm q8 g32" },
    .{ .v = .{ .kind = .simd, .bits = 4, .group = 64 }, .name = "metal simd_qmm q4 g64" },
    .{ .v = .{ .kind = .simd_bits, .bits = 3, .group = 64 }, .name = "metal simd_qmm_bits q3 g64" },
    .{ .v = .{ .kind = .simd_bits, .bits = 6, .group = 64 }, .name = "metal simd_qmm_bits q6 g64" },
    .{ .v = .{ .kind = .simd_bits, .bits = 8, .group = 64 }, .name = "metal simd_qmm_bits q8 g64" },
    .{ .v = .{ .kind = .lane, .bits = 6, .group = 32 }, .name = "metal lane projection q6 g32 sk1" },
    .{ .v = .{ .kind = .lane, .bits = 6, .group = 32, .sk = 4, .f32_out = true }, .name = "metal lane projection q6 g32 sk4 fp32 out" },
};

/// The lane kernel takes every format on the M5's tensor units, and 6-bit groups of 32 alone before them.
const m5_only = [_]Reg{
    .{ .v = .{ .kind = .lane, .bits = 4, .group = 64, .sk = 2 }, .name = "metal lane projection q4 g64 sk2" },
    .{ .v = .{ .kind = .lane, .bits = 4, .group = 64, .coop = true }, .name = "metal lane projection q4 g64 cooperative" },
    .{ .v = .{ .kind = .lane_reg, .bits = 4, .group = 64 }, .name = "metal lane projection q4 g64 reg" },
    .{ .v = .{ .kind = .lane, .bits = 8, .group = 128, .sk = 4 }, .name = "metal lane projection q8 g128 sk4" },
};

fn wanted(o: Opts, name: []const u8) bool {
    const only = o.only orelse return true;
    return std.mem.indexOf(u8, name, only) != null;
}

pub fn main(init: std.process.Init) !u8 {
    const a = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const o = parse(args) catch return 2;
    if (o.rotary_long) std.debug.print("MODE rotary-long: FP64 references through position 1048639; pooling ends at the allocation limit\n", .{});
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var receipt: ?std.json.Parsed(tf.flashnext_vision_contract.rotary_accuracy.Receipt) = null;
    defer if (receipt) |*value| value.deinit();
    var needs_receipt = false;
    for (std.enums.values(tf.flashnext_vision_contract.Which)) |w| {
        if (w != .mask and (wanted(o, tf.flashnext_vision_contract.name(w, false)) or wanted(o, tf.flashnext_vision_contract.name(w, true)))) needs_receipt = true;
    }
    if (o.gates and needs_receipt) {
        const deployment = @import("rotary_deployment.zig");
        receipt = deployment.load(a, init.io, o.rotary_accuracy, try deployment.current(a, init.io, device)) catch |err| {
            std.debug.print("rotary probes refuse this deployment: run tf-rotary-sweep and pass --rotary-accuracy FILE ({s})\n", .{@errorName(err)});
            return err;
        };
    }
    const queue = try device.queue();
    defer queue.deinit();
    const tensor = device.tensorUnits();
    std.debug.print("tf-kernel-contract on {s} (applegpu_g{d}, tensor units {}), seed 0x{x}\n", .{ device.name(), device.generation(), tensor, o.seed });
    if (o.repro) {
        try @import("metal_repro.zig").affineInf(a, device, queue);
        return 0;
    }
    if (o.gates) try @import("metal_delta_slots.zig").run(a, device, queue, o.only, o.seed);
    try gemma.requireComplete(gemma.entries);
    var gemma_list: std.ArrayList(*gemma.Probe) = .empty;
    defer {
        for (gemma_list.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        gemma_list.deinit(a);
    }
    var kernels: std.ArrayList(g.Kernel) = .empty;
    for (gemma.all_entries) |v| {
        if (!wanted(o, v.name)) continue;
        if (!tensor) {
            std.debug.print("UNSUPPORTED {s}: tensor units required; Gemma coverage unchecked\n", .{v.name});
            return error.TensorUnitsRequired;
        }
        try gemma.append(a, device, queue, v, false, &gemma_list, &kernels);
        if (v.geo.hidden == 5376 and (v.kind != 1 or std.mem.endsWith(u8, v.name, "sliding"))) {
            try gemma.append(a, device, queue, v, true, &gemma_list, &kernels);
        }
    }
    var embeddings: [24]gemma_embed.Probe = undefined;
    var embed_made: usize = 0;
    defer for (embeddings[0..embed_made]) |*p| p.deinit();
    for (gemma_embed.entries) |v| {
        if (!wanted(o, v.name)) continue;
        embeddings[embed_made] = try gemma_embed.Probe.init(a, device, queue, v);
        embed_made += 1;
    }
    for (embeddings[0..embed_made]) |*p| try kernels.append(a, p.kernel());
    var mixed = gemma_mixed.Probe.init(a, device, queue);
    defer mixed.deinit();
    if (wanted(o, mixed.kernel().spec.name)) {
        if (!tensor) return error.TensorUnitsRequired;
        try kernels.append(a, mixed.kernel());
    }
    var lfm2_list: std.ArrayList(*lfm2_argmax.Probe) = .empty;
    var lfm2_streams_list: std.ArrayList(*lfm2_streams.Probe) = .empty;
    defer {
        for (lfm2_list.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        for (lfm2_streams_list.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        lfm2_list.deinit(a);
        lfm2_streams_list.deinit(a);
    }
    for (std.enums.values(lfm2_controls.Mutation)) |m| {
        const argmax_side = m == .none or m == .argmax_last_tie or m == .argmax_no_flag;
        if (argmax_side) {
            if (!wanted(o, if (m == .none) lfm2_argmax.name else m.name())) continue;
            const p = try a.create(lfm2_argmax.Probe);
            errdefer a.destroy(p);
            p.* = try lfm2_argmax.Probe.init(a, device, queue, m);
            try lfm2_list.append(a, p);
            try kernels.append(a, p.kernel());
            if (m != .none) continue;
        }
        for (std.enums.values(lfm2_streams.Which)) |w| {
            const control_for = (m == .conv_first_state and w == .conv) or (m == .attention_first_cache and w == .attention);
            if (m != .none and !control_for) continue;
            if (!wanted(o, if (m == .none) lfm2_streams.spec(w).name else m.name())) continue;
            const p = try a.create(lfm2_streams.Probe);
            errdefer a.destroy(p);
            p.* = try lfm2_streams.Probe.init(a, device, queue, w, m);
            try lfm2_streams_list.append(a, p);
            try kernels.append(a, p.kernel());
        }
    }
    var tap: ?lfm2_tap.Probe = null;
    defer if (tap) |*p| p.deinit();
    if (wanted(o, lfm2_tap.name)) {
        tap = try lfm2_tap.Probe.init(a, device, queue);
        try kernels.append(a, tap.?.kernel());
    }
    var rows_list: std.ArrayList(*rows_k.Row) = .empty;
    var proj_list: std.ArrayList(*affine_k.Proj) = .empty;
    defer for (rows_list.items) |r| r.deinit();
    defer for (proj_list.items) |p| p.deinit();
    for (std.enums.values(rows_k.Which)) |w| {
        if (!wanted(o, rows_k.spec(w).name)) continue;
        const r = try a.create(rows_k.Row);
        r.* = try rows_k.Row.init(a, device, queue, w);
        try rows_list.append(a, r);
        try kernels.append(a, r.kernel());
    }
    var pair_list: std.ArrayList(*pair_k.Pair) = .empty;
    defer {
        for (pair_list.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        pair_list.deinit(a);
    }
    for (std.enums.values(pair_controls.Mutation)) |m| {
        for (pair_k.all) |w| {
            const control_for = switch (m) {
                .none => true,
                .stage_peer_half => w == .stage,
                .gather_own_halves => w == .gather,
                .argmax_high_tie, .argmax_no_flag => w == .argmax,
                .pick_no_flag, .pick_high_tie => w == .pick,
            };
            if (!control_for) continue;
            if (!wanted(o, if (m == .none) pair_k.spec(w).name else m.name())) continue;
            const p = try a.create(pair_k.Pair);
            errdefer a.destroy(p);
            p.* = try pair_k.Pair.init(a, device, queue, w, m);
            try pair_list.append(a, p);
            try kernels.append(a, p.kernel());
        }
    }
    var flash_list: [8]flash.Probe = undefined;
    var flash_made: usize = 0;
    defer for (flash_list[0..flash_made]) |*probe| probe.deinit();
    var pool_list: [6]flash_pool.Probe = undefined;
    var pool_made: usize = 0;
    defer for (pool_list[0..pool_made]) |*probe| probe.deinit();
    for (std.enums.values(tf.flashnext_vision_contract.Which)) |w| for ([_]f64{ 0, 4 }) |factor| {
        if ((w == .mask and factor != 0) or !wanted(o, tf.flashnext_vision_contract.name(w, factor > 1))) continue;
        if (tf.flashnext_vision_contract.isPool(w)) {
            pool_list[pool_made] = try flash_pool.Probe.init(a, device, queue, w);
            pool_list[pool_made].factor = factor;
            pool_list[pool_made].long_positions = o.rotary_long;
            pool_list[pool_made].accuracy = if (receipt) |*value| &value.value else null;
            pool_made += 1;
            try kernels.append(a, pool_list[pool_made - 1].kernel());
        } else {
            flash_list[flash_made] = try flash.Probe.init(a, device, queue, w);
            flash_list[flash_made].factor = factor;
            flash_list[flash_made].long_positions = o.rotary_long;
            flash_list[flash_made].accuracy = if (receipt) |*value| &value.value else null;
            flash_made += 1;
            try kernels.append(a, flash_list[flash_made - 1].kernel());
        }
    };
    var fsr_probes: [4]fsr.Probe = undefined;
    var fsr_made: usize = 0;
    defer for (fsr_probes[0..fsr_made]) |*probe| probe.deinit();
    for ([_]usize{ 4, 6 }) |bits| for ([_]bool{ false, true }) |control| {
        if (!wanted(o, fsr.Probe.name(bits, control))) continue;
        fsr_probes[fsr_made] = .{ .a = a, .device = device, .queue = queue, .bits = bits, .control = control };
        fsr_made += 1;
        try kernels.append(a, fsr_probes[fsr_made - 1].kernel());
    };
    var d3norm: ?drama3_k.Norm = null;
    defer if (d3norm) |*n| n.deinit();
    if (wanted(o, drama3_k.spec().name)) {
        d3norm = try drama3_k.Norm.init(a, device, queue);
        try kernels.append(a, d3norm.?.kernel());
    }
    var d3projections: [std.enums.values(drama_nvfp4.Variant).len]drama_nvfp4.Proj = undefined;
    var d3projection_count: usize = 0;
    defer for (d3projections[0..d3projection_count]) |*p| p.deinit();
    for (std.enums.values(drama_nvfp4.Variant)) |v| {
        if (!wanted(o, drama_nvfp4.spec(v).name)) continue;
        d3projections[d3projection_count] = try drama_nvfp4.Proj.init(a, device, queue, v);
        d3projection_count += 1;
        try kernels.append(a, d3projections[d3projection_count - 1].kernel());
    }
    var d3routers: [std.enums.values(drama_router.Variant).len]drama_router.Router = undefined;
    var d3router_count: usize = 0;
    defer for (d3routers[0..d3router_count]) |*p| p.deinit();
    for (std.enums.values(drama_router.Variant)) |v| {
        if (!wanted(o, drama_router.spec(v).name)) continue;
        d3routers[d3router_count] = try drama_router.Router.init(a, device, queue, v);
        d3router_count += 1;
        try kernels.append(a, d3routers[d3router_count - 1].kernel());
    }
    var d3combines: [std.enums.values(drama_combine.Variant).len]drama_combine.Combine = undefined;
    var d3combine_count: usize = 0;
    defer for (d3combines[0..d3combine_count]) |*p| p.deinit();
    for (std.enums.values(drama_combine.Variant)) |v| {
        if (!wanted(o, drama_combine.spec(v).name)) continue;
        d3combines[d3combine_count] = try drama_combine.Combine.init(a, device, queue, v);
        d3combine_count += 1;
        try kernels.append(a, d3combines[d3combine_count - 1].kernel());
    }
    var d3mixes: [drama_cfg.gammas.len]drama_cfg.Mix = undefined;
    var d3mix_count: usize = 0;
    defer for (d3mixes[0..d3mix_count]) |*p| p.deinit();
    for (drama_cfg.gammas) |gamma| {
        if (!wanted(o, drama_cfg.spec(gamma).name)) continue;
        d3mixes[d3mix_count] = try drama_cfg.Mix.init(a, device, queue, gamma);
        d3mix_count += 1;
        try kernels.append(a, d3mixes[d3mix_count - 1].kernel());
    }
    var glm_eh: ?glm_eh_k.Nax = null;
    defer if (glm_eh) |*n| n.deinit();
    if (wanted(o, glm_eh_k.spec().name)) {
        glm_eh = try glm_eh_k.Nax.init(a, device, queue);
        try kernels.append(a, glm_eh.?.kernel());
    }
    var topk: ?topk_k.TopK = null;
    defer if (topk) |*t| t.deinit();
    if (wanted(o, topk_k.spec().name)) {
        topk = try topk_k.TopK.init(a, device, queue);
        try kernels.append(a, topk.?.kernel());
    }
    for ([_][]const Reg{ &common, if (tensor) &m5_only else &.{} }) |list| for (list) |v| {
        if (!wanted(o, v.name)) continue;
        const p = try a.create(affine_k.Proj);
        p.* = try affine_k.Proj.init(a, device, queue, v.v, v.name);
        try proj_list.append(a, p);
        try kernels.append(a, p.kernel());
    };
    var h3_list: std.ArrayList(*h3_k.H3) = .empty;
    defer for (h3_list.items) |x| x.deinit();
    if (tensor) for (std.enums.values(h3_k.Which)) |w| {
        if (!wanted(o, h3_k.spec(w).name)) continue;
        const x = try a.create(h3_k.H3);
        x.* = try h3_k.H3.init(a, device, queue, w);
        try h3_list.append(a, x);
        try kernels.append(a, x.kernel());
    };
    var h3_top: ?h3_topk.TopK = null;
    defer if (h3_top) |*x| x.deinit();
    if (wanted(o, h3_topk.spec().name)) {
        h3_top = try h3_topk.TopK.init(a, device, queue);
        try kernels.append(a, h3_top.?.kernel());
    }
    var tile_p = try tf.tile_qmm.Pipelines.load(device);
    defer tile_p.deinit();
    var tiles: [2 * tf.tile_qmm.bit_widths.len]tile_k.Tile = undefined;
    var ntiles: usize = 0;
    defer for (tiles[0..ntiles]) |*x| x.deinit();
    for ([_]tile_k.Kind{ .tiles, .twin }) |kind| for (tf.tile_qmm.bit_widths) |bits| {
        const name = try std.fmt.allocPrint(a, "metal tile_qmm {s} q{d} g64", .{ if (kind == .tiles) "tiles" else "twin", bits });
        if (!wanted(o, name)) continue;
        tiles[ntiles] = tile_k.Tile.init(a, device, queue, &tile_p, kind, bits, name);
        ntiles += 1;
    };
    for (tiles[0..ntiles]) |*x| try kernels.append(a, x.kernel());
    var glm_windows: glm_rows_k.Suite = .{ .a = a, .device = device, .queue = queue, .kernels = &kernels };
    defer glm_windows.deinit();
    try glm_windows.collect(o.only);
    var nemotron_experts: @import("metal_nemotron_experts.zig").Suite = .{ .a = a, .device = device, .queue = queue, .kernels = &kernels };
    defer nemotron_experts.deinit();
    try nemotron_experts.collect(o.only);
    var nemotron_pages: @import("metal_nemotron_pages.zig").Suite = .{ .a = a, .device = device, .queue = queue, .kernels = &kernels };
    defer nemotron_pages.deinit();
    try nemotron_pages.collect(o.only);
    var nemotron_attention: @import("metal_nemotron_attention.zig").Suite = .{ .a = a, .device = device, .queue = queue, .kernels = &kernels };
    defer nemotron_attention.deinit();
    try nemotron_attention.collect(o.only);
    const weight_only = @import("metal_weight_only.zig");
    const weight_source = try weight_only.text(a);
    defer a.free(weight_source);
    var weight_kernels = try weight_only.Kernels.load(device, weight_source, true);
    defer weight_kernels.deinit();
    var weight_cases: [weight_only.count]weight_only.Case = undefined;
    try weight_only.collect(a, device, queue, &weight_kernels, &weight_cases, &kernels, o.only);
    defer for (&weight_cases) |*case| case.deinit();
    const hc = @import("metal_hc_activation.zig");
    const hc_source = try hc.text(a);
    defer a.free(hc_source);
    var hc_kernels = try hc.Kernels.load(device, hc_source, true);
    defer hc_kernels.deinit();
    var hc_cases: [hc.count]hc.Case = undefined;
    try hc.collect(a, device, queue, &hc_kernels, &hc_cases, &kernels, o.only);
    defer for (&hc_cases) |*case| case.deinit();
    const pointwise = @import("metal_hc_pointwise.zig");
    const pointwise_source = try pointwise.text(a);
    defer a.free(pointwise_source);
    var pointwise_kernels = try pointwise.Kernels.load(device, pointwise_source, true);
    defer pointwise_kernels.deinit();
    var pointwise_cases: [pointwise.count]pointwise.Case = undefined;
    try pointwise.collect(a, device, queue, &pointwise_kernels, &pointwise_cases, &kernels, o.only);
    defer for (&pointwise_cases) |*case| case.deinit();
    const native_hc = try @import("metal_qwen4_hc.zig").Suite.create(a, device, queue, &kernels, o.only);
    defer native_hc.destroy();
    const native_qsa_select = try @import("metal_qwen4_qsa_select.zig").Suite.create(a, device, queue, &kernels, o.only);
    defer native_qsa_select.destroy();
    const native_qsa_scores = try @import("metal_qwen4_qsa_scores.zig").Suite.create(a, device, queue, &kernels, o.only);
    defer native_qsa_scores.destroy();
    const native_qsa_attn = try @import("metal_qwen4_qsa_attn.zig").Suite.create(a, device, queue, &kernels, o.only);
    defer native_qsa_attn.destroy();
    const native_qsa_norm = try @import("metal_qwen4_qsa_norm.zig").Suite.create(a, device, queue, &kernels, o.only);
    defer native_qsa_norm.destroy();
    const native_ple = try @import("metal_qwen4_ple.zig").Suite.create(a, device, queue, &kernels, o.only);
    defer native_ple.destroy();
    const native_gdn = try @import("metal_qwen4_gdn.zig").Suite.create(a, device, queue, &kernels, o.only);
    defer native_gdn.destroy();
    const moss_image = try @import("metal_moss_image.zig").Suite.create(a, device, queue, &kernels, o.only);
    defer moss_image.destroy();
    var t: g.Tally = .{};
    const t0 = std.Io.Clock.awake.now(init.io);
    if (o.gates) {
        try g.runAll(a, kernels.items, o.seed, &t);
        for (flash_list[0..flash_made]) |*probe| try probe.rowCountInvariant();
    }
    const t1 = std.Io.Clock.awake.now(init.io);
    var lim: usize = 0;
    if (o.limits) {
        lim = try limits_k.gate(a, init.io, device, o.offline);
        lim += try gemma.limits(a, device, o.only);
        lim += try gemma_embed.limits(a, device, o.only);
        lim += try gemma_mixed.limits(a, device, o.only);
        lim += try lfm2_tap.limits(a, device, o.only);
    }
    const t2 = std.Io.Clock.awake.now(init.io);
    const ms = struct {
        fn of(x: std.Io.Timestamp, y: std.Io.Timestamp) i64 {
            return @intCast(@divTrunc(x.durationTo(y).toNanoseconds(), std.time.ns_per_ms));
        }
    };
    std.debug.print("SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim, ms.of(t0, t1), ms.of(t1, t2) });
    return if (t.problems() + lim == 0) 0 else 1;
}
