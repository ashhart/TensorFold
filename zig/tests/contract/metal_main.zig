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

const flash = @import("metal_flashnext.zig");
const gemma = @import("metal_gemma4.zig");
const g = contract.gates;
const usage = "usage: tf-kernel-contract [--only TEXT] [--no-gates] [--no-limits] [--offline DIR] [--seed N] [--repro]";

const Opts = struct { only: ?[]const u8 = null, gates: bool = true, limits: bool = true, offline: ?[]const u8 = null, seed: u64 = 0x6b63, repro: bool = false };

fn parse(args: []const [:0]const u8) !Opts {
    var o = Opts{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--repro")) o.repro = true else if (std.mem.eql(u8, a, "--no-gates")) o.gates = false else if (std.mem.eql(u8, a, "--no-limits")) o.limits = false else if (i + 1 < args.len and std.mem.eql(u8, a, "--only")) {
            i += 1;
            o.only = args[i];
        } else if (i + 1 < args.len and std.mem.eql(u8, a, "--offline")) {
            i += 1;
            o.offline = args[i];
        } else if (i + 1 < args.len and std.mem.eql(u8, a, "--seed")) {
            i += 1;
            o.seed = try std.fmt.parseInt(u64, args[i], 0);
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
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const tensor = device.tensorUnits();
    std.debug.print("tf-kernel-contract on {s} (applegpu_g{d}, tensor units {}), seed 0x{x}\n", .{ device.name(), device.generation(), tensor, o.seed });
    if (o.repro) {
        try @import("metal_repro.zig").affineInf(a, device, queue);
        return 0;
    }
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
    for (gemma.entries) |v| {
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
    var flash_list: [2]flash.Probe = undefined;
    var flash_made: usize = 0;
    defer for (flash_list[0..flash_made]) |*probe| probe.deinit();
    for (std.enums.values(tf.flashnext_vision_contract.Which)) |w| {
        if (!wanted(o, tf.flashnext_vision_contract.spec(contract.spec, w).name)) continue;
        flash_list[flash_made] = try flash.Probe.init(a, device, queue, w);
        flash_made += 1;
        try kernels.append(a, flash_list[flash_made - 1].kernel());
    }
    var d3norm: ?drama3_k.Norm = null;
    defer if (d3norm) |*n| n.deinit();
    if (wanted(o, drama3_k.spec().name)) {
        d3norm = try drama3_k.Norm.init(a, device, queue);
        try kernels.append(a, d3norm.?.kernel());
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
