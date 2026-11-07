//! tf-hip-kernel-contract: HIP kernels through gates 1-4, controls and gate 5; `image`: one family; `limits DIR`.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const controls = @import("hip_controls");
const gate5 = @import("contract_limits.zig");
const families = @import("contract_families.zig");
const select = @import("contract_select.zig");
const rows_k = @import("contract_rows.zig");
const wide_k = @import("contract_rms_wide.zig");
const stream_k = @import("contract_stream.zig");
const mamba_k = @import("contract_mamba.zig");
const scan_k = @import("contract_scan.zig");
const moe_k = @import("contract_nemo_moe.zig");
const walk_k = @import("contract_walk.zig");
const prefill_k = @import("contract_prefill.zig");
const add_rms_k = @import("contract_add_rms.zig");
const tail_k = @import("contract_tail.zig");
const window_k = @import("contract_window.zig");
const relu2_k = @import("contract_relu2.zig");
const kv_k = @import("contract_kv_write.zig");
const argmax_k = @import("contract_argmax.zig");
const pair_k = @import("contract_relu2_pair.zig");
const heads_rows_k = @import("contract_heads_rows.zig");
const dense_k = @import("contract_dense.zig");
const dense12_k = @import("contract_dense_wmma12.zig");
const gate_k = @import("contract_moe_gate.zig");
const router_k = @import("contract_router.zig");
const kit = @import("contract_kit.zig");
const item_k = @import("contract_wmma_item.zig");
const embed_k = @import("contract_embed.zig");
const columns_k = @import("contract_columns.zig");
const vision_k = @import("contract_vision.zig");
const full_k = @import("contract_vision_attention.zig");
const text_k = @import("contract_dense_text.zig");
const sparse_k = @import("contract_sparse_lanes.zig");

const g = core.contract.gates;

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 1) {
        var bad: []const u8 = "";
        select.validateAll(a, &families.entries, &core.contract.hip_records.records, &bad) catch |err| {
            if (err == error.OutOfMemory) return err;
            std.debug.print("FAIL kernel-contract: {s} ({s})\n", .{ select.explain(err), bad });
            return 2;
        };
        return if (try run(a, init.io) > 0) 1 else 0;
    }
    const image_args = args.len == 3 or args.len == 4;
    if (image_args and std.mem.eql(u8, args[1], "image")) return image(a, init.io, args[2], if (args.len == 4) args[3] else "full");
    if (args.len != 3 or !std.mem.eql(u8, args[1], "limits")) {
        std.debug.print("usage: tf-hip-kernel-contract [image GROUP [smoke|full] | limits DIR]\n", .{});
        return 2;
    }
    const n = try gate5.objects(a, init.io, args[2]);
    if (n > 0) {
        std.debug.print("FAIL kernel-contract-limits: {d} problems\n", .{n});
        return 1;
    }
    std.debug.print("PASS kernel-contract-limits: every registered launch fits gfx1030, gfx1100, gfx1151 and " ++
        "gfx1201, every record current, every control red\n", .{});
    return 0;
}

/// `image GROUP [smoke|full]`: the entry's own gates 1-4, then gate 5 on its records; returns the exit status.
fn image(a: std.mem.Allocator, io: std.Io, group_name: []const u8, profile_name: []const u8) !u8 {
    const profile = std.meta.stringToEnum(families.Profile, profile_name) orelse {
        std.debug.print("FAIL image {s}: the profile is smoke or full, not {s}\n", .{ group_name, profile_name });
        return 2;
    };
    var bad: []const u8 = "";
    const sel = select.resolve(a, &families.entries, &core.contract.hip_records.records, group_name, &bad) catch |err| {
        if (err == error.OutOfMemory) return err;
        std.debug.print("FAIL image {s}: {s} ({s})\n", .{ group_name, select.explain(err), bad });
        if (err == error.UnknownGroup) {
            std.debug.print("groups:", .{});
            for (@typeInfo(hip.kernels.Group).@"enum".field_names) |name| std.debug.print(" {s}", .{name});
            std.debug.print("\n", .{});
        }
        return 2;
    };
    defer a.free(sel.records);
    var built: usize = hip.kernels.images[@backingInt(sel.entry.group)].len;
    _ = &built; // runtime-known, so a stub build still compiles all that follows
    if (built == 0) {
        std.debug.print("FAIL image {s}: built without hipcc, so no gate is proven\n", .{group_name});
        return 1;
    }
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var arch_buffer: [256]u8 = undefined;
    const arch = try hip.Device.archName(&r, 0, &arch_buffer);
    if (!std.mem.eql(u8, arch, hip.kernels.arch)) {
        const say = "FAIL image {s}: this GPU is {s}, the kernels are built for {s}\n";
        std.debug.print(say, .{ group_name, arch, hip.kernels.arch });
        return 1;
    }
    const t0 = ms(io);
    const t = try sel.entry.run(a, io, profile);
    const t1 = ms(io);
    const lim = try gate5.tableOf(a, sel.records) + try gate5.liveOf(a, &r, ctx, arch, sel.records);
    const t2 = ms(io);
    const tally = .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim, t1 - t0, t2 - t1 };
    std.debug.print("SUMMARY {s} HIP {t}: {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: " ++
        "{d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms\n", .{ group_name, profile } ++ tally);
    return select.verdict(t, lim);
}

fn ms(io: std.Io) i96 {
    return @divTrunc(std.Io.Clock.awake.now(io).toNanoseconds(), std.time.ns_per_ms);
}

/// Gates 1-4 on every registered HIP kernel, each control and every family entry, then gate 5; returns the problems.
fn run(a: std.mem.Allocator, io: std.Io) !usize {
    var built: [2]usize = .{ hip.kernels.images[0].len, controls.bytes.len };
    _ = &built; // runtime-known, so a stub build still compiles all that follows
    if (built[0] == 0 or built[1] == 0) {
        std.debug.print("FAIL kernel-contract: built without hipcc, so no gate is proven\n", .{});
        return 1;
    }
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var arch_buffer: [256]u8 = undefined;
    const arch = try hip.Device.archName(&r, 0, &arch_buffer);
    if (!std.mem.eql(u8, arch, hip.kernels.arch)) {
        const say = "FAIL kernel-contract: this GPU is {s}, the kernels are built for {s}\n";
        std.debug.print(say, .{ arch, hip.kernels.arch });
        return 1;
    }
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var l = try hip.Launcher.load(&r, .{}, hip.kernels.images);
    defer l.unload();
    var ctl = try hip.Module.load(&r, &controls.bytes);
    defer ctl.unload();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    const which = comptime std.enums.values(rows_k.Which);
    var row_list: [which.len]rows_k.Row = undefined;
    var nrows: usize = 0;
    defer for (row_list[0..nrows]) |*x| x.deinit();
    for (which) |w| {
        row_list[nrows] = try rows_k.Row.init(a, &r, s, &l, ctl, w);
        nrows += 1;
    }
    for (row_list[0..nrows]) |*x| try kernels.append(a, x.kernel());
    var mambas: [3]mamba_k.Mamba = undefined;
    for (&mambas, std.enums.values(mamba_k.Which)) |*m, w| m.* = .init(a, &r, s, &l, w);
    defer for (&mambas) |*m| m.deinit();
    for (&mambas) |*m| try kernels.append(a, m.kernel());
    var scans: [2]scan_k.Scan = undefined;
    for (&scans, std.enums.values(scan_k.Part)) |*m, p| m.* = .init(a, &r, s, &l, p);
    defer for (&scans) |*m| m.deinit();
    for (&scans) |*m| try kernels.append(a, m.kernel());
    var picks: [3]moe_k.Moe = undefined;
    for (&picks, [_]moe_k.Which{ .pass, .rank, .localize }) |*m, w| m.* = .init(a, &r, s, &l, w);
    defer for (&picks) |*m| m.deinit();
    for (&picks) |*m| try kernels.append(a, m.kernel());
    var moes: [3]moe_k.Moe = undefined;
    for (&moes, [_]moe_k.Which{ .relu2, .combine, .combine_f32 }) |*m, w| m.* = .init(a, &r, s, &l, w);
    defer for (&moes) |*m| m.deinit();
    for (&moes) |*m| try kernels.append(a, m.kernel());
    var wide = wide_k.Wide.init(a, &r, s, &l);
    defer wide.deinit();
    try kernels.append(a, wide.kernel());
    var walk = walk_k.Walk.init(a, &r, s, &l);
    defer walk.deinit();
    try kernels.append(a, walk.kernel());
    var walk_act = walk_k.Walk.init(a, &r, s, &l);
    walk_act.act = true;
    defer walk_act.deinit();
    try kernels.append(a, walk_act.kernel());
    var attends: [2]walk_k.Walk = undefined;
    for (&attends, [_]bool{ false, true }) |*w, act| {
        w.* = .init(a, &r, s, &l);
        w.act = act;
        w.attend = true;
    }
    defer for (&attends) |*w| w.deinit();
    for (&attends) |*w| try kernels.append(a, w.kernel());
    var prefills: [2]prefill_k.Prefill = undefined;
    for (&prefills, [_]bool{ true, false }) |*p, paged| p.* = .init(a, &r, s, &l, paged);
    defer for (&prefills) |*p| p.deinit();
    for (&prefills) |*p| try kernels.append(a, p.kernel());
    // non-finite values only past rows' edges, flat and paged, both cache types; then each without its edge pass
    var edges: [8]prefill_k.Prefill = undefined;
    var edge_count: usize = 0;
    defer for (edges[0..edge_count]) |*p| p.deinit();
    for (0..8) |i| {
        const paged = i % 2 == 1;
        const kind: kit.Kind = if (i / 2 % 2 == 1) .bf16 else .f16;
        if (kind == .bf16 and kit.act() != .bf16) continue; // RDNA2 serves fp16 caches only
        const p = &edges[edge_count];
        p.* = .init(a, &r, s, &l, paged);
        edge_count += 1;
        p.k = kind;
        p.mode = .edge;
        if (i >= 4) p.planted = try ctl.function(([_][:0]const u8{
            "tf_ctl_fa_wide_unmasked_f16", "tf_ctl_fa_paged_unmasked_f16", "tf_ctl_fa_wide_unmasked_bf16", "tf_ctl_fa_paged_unmasked_bf16",
        })[i - 4]);
        try kernels.append(a, p.kernel());
    }
    var add_rms: [3]add_rms_k.AddRms = undefined;
    for (&add_rms, std.enums.values(add_rms_k.Mode)) |*m, w| m.* = .init(a, &r, s, &l, w);
    defer for (&add_rms) |*m| m.deinit();
    for (&add_rms) |*m| try kernels.append(a, m.kernel());
    var tails: [2]tail_k.Tail = undefined;
    for (&tails, std.enums.values(tail_k.Mode)) |*m, w| m.* = .init(a, &r, s, &l, w);
    defer for (&tails) |*m| m.deinit();
    for (&tails) |*m| try kernels.append(a, m.kernel());
    var convs: [2]window_k.ConvWindow = undefined;
    for (&convs, std.enums.values(window_k.Part)) |*m, p| m.init(a, &r, s, &l, p);
    defer for (&convs) |*m| m.deinit();
    for (&convs) |*m| try kernels.append(a, m.kernel());
    var snaps: [2]window_k.ScanWindow = undefined;
    for (&snaps, std.enums.values(scan_k.Part)) |*m, p| m.init(a, &r, s, &l, p);
    defer for (&snaps) |*m| m.deinit();
    for (&snaps) |*m| try kernels.append(a, m.kernel());
    var streams: [stream_k.cases.len]stream_k.Stream = undefined;
    for (&streams, 0..) |*x, i| x.* = .init(a, &r, s, &l, i);
    defer for (&streams) |*x| x.deinit();
    for (&streams) |*x| try kernels.append(a, x.kernel());
    var relu2 = relu2_k.Relu2.init(a, &r, s, &l);
    defer relu2.deinit();
    try kernels.append(a, relu2.kernel());
    var kv_write = kv_k.KvWrite.init(a, &r, s, &l);
    defer kv_write.deinit();
    try kernels.append(a, kv_write.kernel());
    var argmax = argmax_k.Argmax.init(a, &r, s, &l);
    defer argmax.deinit();
    try kernels.append(a, argmax.kernel());
    var pair = pair_k.Pair.init(a, &r, s, &l);
    defer pair.deinit();
    try kernels.append(a, pair.kernel());
    var heads_rows = heads_rows_k.HeadsRows.init(a, &r, s, &l);
    defer heads_rows.deinit();
    try kernels.append(a, heads_rows.kernel());
    var dense: [2]dense_k.DenseGemm = undefined;
    for (&dense, [_]bool{ true, false }) |*d, fp32_out| d.* = .init(a, &r, s, &l, fp32_out);
    defer for (&dense) |*d| d.deinit();
    // the BF16 dense GEMM runs on BF16 builds only; an FP16 build keeps its trap body and serves no shape of it
    if (l.dense.bf16) {
        for (&dense) |*d| try kernels.append(a, d.kernel());
    } else std.debug.print("N/A  hip dense_gemm_block: not served on this FP16 build\n", .{});
    var dense12: [5]dense12_k.DenseWmma12 = undefined;
    for (&dense12, [_]bool{ true, false, true, false, true }, 0..) |*d, fp32_out, i| d.* = .init(a, &r, s, &l, fp32_out, if (i < 2 or i == 4) .w12_small else .w12_big);
    defer for (&dense12) |*d| d.deinit();
    // the WMMA dense tile serves gfx12 only; elsewhere its symbol, and its control's, keep a trap body
    if (l.dense.wmma12) {
        dense12[4].planted = try ctl.function("tf_ctl_wmma12_bf16_sum");
        for (&dense12) |*d| try kernels.append(a, d.kernel());
    } else std.debug.print("N/A  hip dense_wmma12_gemm: not served off gfx12\n", .{});
    var gates: [3]gate_k.Gate = undefined;
    for (&gates, std.enums.values(gate_k.Which)) |*m, w| m.* = .init(a, &r, s, &l, w);
    defer for (&gates) |*m| m.deinit();
    for (&gates) |*m| try kernels.append(a, m.kernel());
    // both activation kinds: the rows are bf16 whichever one the chip's activations are
    var routers: [2]router_k.Router = undefined;
    for (&routers, [_]kit.Kind{ .f16, .bf16 }) |*x, k| x.* = router_k.Router.init(a, &r, s, &l, k);
    defer for (&routers) |*x| x.deinit();
    for (&routers) |*x| try kernels.append(a, x.kernel());
    // the few-row matrix-core tiles run on gfx11's matrix cores only
    const matrix = if (hip.Caps.of(arch)) |v| v.matrix == .wmma11 else false;
    var items: [item_k.cases.len]item_k.Item = undefined;
    for (&items, 0..) |*x, i| x.* = .init(a, &r, s, &l, i);
    defer for (&items) |*x| x.deinit();
    if (matrix) for (&items) |*x| try kernels.append(a, x.kernel());
    var embeds: [4]embed_k.Embed = undefined;
    var nembeds: usize = 0;
    defer for (embeds[0..nembeds]) |*x| x.deinit();
    for (std.enums.values(embed_k.Which)) |w| {
        embeds[nembeds] = try .init(a, &r, s, &l, ctl, w);
        nembeds += 1;
    }
    for (embeds[0..nembeds]) |*x| try kernels.append(a, x.kernel());
    var columns: [std.enums.values(columns_k.Which).len]columns_k.Columns = undefined;
    var columns_count: usize = 0;
    defer for (columns[0..columns_count]) |*x| x.deinit();
    for (std.enums.values(columns_k.Which)) |which_columns| {
        columns[columns_count] = try columns_k.Columns.init(a, &r, s, &l, ctl, which_columns);
        columns_count += 1;
    }
    for (&columns) |*x| try kernels.append(a, x.kernel());
    var vision: [std.enums.values(vision_k.Which).len]vision_k.Vision = undefined;
    for (&vision, std.enums.values(vision_k.Which)) |*v, w| v.* = .init(a, &r, s, &l, w);
    defer for (&vision) |*v| v.deinit();
    for (&vision) |*v| try kernels.append(a, v.kernel());
    var full = full_k.Attention.init(a, &r, s, &l);
    defer full.deinit();
    try kernels.append(a, full.kernel());
    var text: [std.enums.values(text_k.Which).len]text_k.Text = undefined;
    for (&text, std.enums.values(text_k.Which)) |*v, w| v.* = .init(a, &r, s, &l, w);
    defer for (&text) |*v| v.deinit();
    for (&text) |*v| try kernels.append(a, v.kernel());
    var sparse = sparse_k.SparseLanes.init(a, &r, s, &l);
    defer sparse.deinit();
    try kernels.append(a, sparse.kernel());
    var t: g.Tally = .{};
    const t0 = ms(io);
    try g.runAll(a, kernels.items, 0x6b63, &t);
    const own = t.problems();
    const fam = try select.runEntries(a, io, &families.entries, .full); // each family's own gates 1-4
    t = select.merge(t, fam.tally);
    const t1 = ms(io);
    const lim = try gate5.table(a) + try gate5.live(a, &r, ctx, arch);
    const t2 = ms(io);
    const tally = .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim, t1 - t0, t2 - t1 };
    std.debug.print("SUMMARY {s} {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: " ++
        "{d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms\n", .{arch} ++ tally);
    return own + fam.problems + lim;
}
