//! On-device bit-exactness: every qmm_q4_tile output byte equals the pre-k-split kernel's, per shape and seed.
const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("vulkan_shaders");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const fx = @import("contract_qmm.zig");

const timeout_ns: u64 = 10 * std.time.ns_per_s;
const param_group: usize = 64;
const cancel_scale_pos = 0x4a00; // bf16 2^21
const cancel_scale_neg = 0xca00; // bf16 -2^21

// Decode m=1 over Qwen3.5-2B's real projections: q/k/v/o 2048-wide, gate/up 6144-wide, down k=6144, lm_head 248320.
const production = [_]Case{
    .{ .name = "decode-q", .rows = 1, .n = 2048, .k = 2048 },
    .{ .name = "decode-k", .rows = 1, .n = 512, .k = 2048 },
    .{ .name = "decode-v", .rows = 1, .n = 512, .k = 2048 },
    .{ .name = "decode-o", .rows = 1, .n = 2048, .k = 2048 },
    .{ .name = "decode-gate", .rows = 1, .n = 6144, .k = 2048 },
    .{ .name = "decode-up", .rows = 1, .n = 6144, .k = 2048 },
    .{ .name = "decode-down", .rows = 1, .n = 2048, .k = 6144 },
    .{ .name = "decode-lm-head", .rows = 1, .n = 248320, .k = 2048 },
    .{ .name = "prefill-q-5rows", .rows = 5, .n = 6144, .k = 2048 },
    .{ .name = "prefill-33rows", .rows = 33, .n = 6144, .k = 2048 },
    .{ .name = "prefill-down-32rows", .rows = 32, .n = 2048, .k = 6144 },
    .{ .name = "chunk-2rows-257cols", .rows = 2, .n = 257, .k = 128 },
    .{ .name = "chunk-2rows-16cols", .rows = 2, .n = 16, .k = 6144 },
    .{ .name = "tail-n33", .rows = 1, .n = 33, .k = 2048 },
    .{ .name = "tail-n63", .rows = 1, .n = 63, .k = 2048 },
    .{ .name = "tail-n2049", .rows = 1, .n = 2049, .k = 2048 },
    .{ .name = "k64", .rows = 1, .n = 2048, .k = 64 },
    .{ .name = "k128", .rows = 1, .n = 2048, .k = 128 },
    .{ .name = "k192", .rows = 1, .n = 2048, .k = 192 },
    .{ .name = "stress-decode", .rows = 1, .n = 2048, .k = 2048, .fill = .stress },
    .{ .name = "stress-gate", .rows = 1, .n = 6144, .k = 2048, .fill = .stress },
    .{ .name = "stress-down", .rows = 1, .n = 2048, .k = 6144, .fill = .stress },
    .{ .name = "stress-prefill-33rows", .rows = 33, .n = 6144, .k = 2048, .fill = .stress },
};

const Case = struct {
    name: []const u8,
    rows: usize,
    n: usize,
    k: usize,
    fill: ?g.Fill = null,
    seed: u64 = 0,
    act_f32: bool = false,
};

// The explicit cancellation case leads; every other case runs on three seeds, bf16 then f32.
fn buildCases() [production.len * 6 + 2]Case {
    var out: [production.len * 6 + 2]Case = undefined;
    var at: usize = 0;
    out[at] = .{ .name = "finite-bf16-cancel", .rows = 1, .n = 1, .k = 2048 };
    at += 1;
    out[at] = .{ .name = "finite-bf16-cancel", .rows = 1, .n = 1, .k = 2048, .act_f32 = true };
    at += 1;
    for (production, 0..) |base, i| {
        out[at] = .{ .name = base.name, .rows = base.rows, .n = base.n, .k = base.k, .fill = base.fill, .seed = 1000 + i };
        at += 1;
    }
    for (production, 0..) |base, i| {
        out[at] = .{ .name = base.name, .rows = base.rows, .n = base.n, .k = base.k, .fill = base.fill, .seed = 2000 + i, .act_f32 = true };
        at += 1;
    }
    for (production, 0..) |base, i| {
        out[at] = .{ .name = base.name, .rows = base.rows, .n = base.n, .k = base.k, .fill = base.fill, .seed = 3000 + i };
        at += 1;
    }
    for (production, 0..) |base, i| {
        out[at] = .{ .name = base.name, .rows = base.rows, .n = base.n, .k = base.k, .fill = base.fill, .seed = 4000 + i, .act_f32 = true };
        at += 1;
    }
    for (production, 0..) |base, i| {
        out[at] = .{ .name = base.name, .rows = base.rows, .n = base.n, .k = base.k, .fill = base.fill, .seed = 5000 + i };
        at += 1;
    }
    for (production, 0..) |base, i| {
        out[at] = .{ .name = base.name, .rows = base.rows, .n = base.n, .k = base.k, .fill = base.fill, .seed = 6000 + i, .act_f32 = true };
        at += 1;
    }
    return out;
}

const case_list = buildCases();

// The maintainer's finite-BF16 cancellation shape: the serial sum is exactly 0 while any slice reordering is not.
fn plantCancel(f: *fx.Fixture, act_f32: bool) !void {
    const s = f.shape;
    const stored = vk.tile.padded(s.n);
    if (act_f32) {
        for (0..s.rows * s.k) |i| {
            f.x[2 * i] = 0;
            f.x[2 * i + 1] = 0x3f80;
        }
    } else {
        @memset(f.x[0 .. s.rows * s.k], 0x3f80);
    }
    const w_row = try f.a.alloc(u32, s.n * s.k / 8);
    defer f.a.free(w_row);
    @memset(w_row, 0);
    for (0..8) |j| w_row[j] = 0x88888888;
    for (8..16) |j| w_row[j] = 0x11111111;
    for (64..72) |j| w_row[j] = 0x88888888;
    const scales_row = try f.a.alloc(u16, s.n * (s.k / param_group));
    defer f.a.free(scales_row);
    @memset(scales_row, 0x3f80);
    scales_row[0] = cancel_scale_pos;
    scales_row[1] = 0x3f80;
    scales_row[8] = cancel_scale_neg;
    const biases_row = try f.a.alloc(u16, scales_row.len);
    defer f.a.free(biases_row);
    @memset(biases_row, 0);
    vk.tile.tileWords(std.mem.sliceAsBytes(f.w[0 .. stored * s.k / 8]), std.mem.sliceAsBytes(w_row), s.n, s.k);
    vk.tile.tileParams(std.mem.sliceAsBytes(f.scales[0 .. stored * (s.k / param_group)]), std.mem.sliceAsBytes(scales_row), s.n, s.k);
    vk.tile.tileParams(std.mem.sliceAsBytes(f.biases[0 .. stored * (s.k / param_group)]), std.mem.sliceAsBytes(biases_row), s.n, s.k);
    const uploads = [_][]const u8{
        std.mem.sliceAsBytes(f.x),
        std.mem.sliceAsBytes(f.w[0 .. stored * s.k / 8]),
        std.mem.sliceAsBytes(f.scales[0 .. stored * (s.k / param_group)]),
        std.mem.sliceAsBytes(f.biases[0 .. stored * (s.k / param_group)]),
    };
    for (uploads, 0..) |bytes, i| {
        try vk.orExit(vk.upload(f.cmds, f.buffers[i], 0, f.staging, bytes, timeout_ns));
    }
}

fn tag(act_f32: bool) []const u8 {
    return if (act_f32) "-f32" else "-bf16";
}

pub fn main(init: std.process.Init) !u8 {
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{ .features = .{ .storage16 = true } });
    defer dev.close();
    var cmds = try vk.Commands.init(&dev);
    defer cmds.deinit();
    var staging = try vk.Buffer.init(&dev, .staging, 64 << 10);
    defer staging.deinit();
    const layout: vk.dispatch.Layout = .{ .bindings = 5, .push_bytes = @sizeOf(fx.Params), .writes = 1 << 4 };
    var tile = try vk.dispatch.Pipeline.init(&dev, &shaders.qmm_q4_tile, layout);
    defer tile.deinit();
    var tile_f32 = try vk.dispatch.Pipeline.init(&dev, &shaders.qmm_q4_tile_f32, layout);
    defer tile_f32.deinit();
    var ref = try vk.dispatch.Pipeline.init(&dev, &shaders.qmm_q4_tile_order_ref, layout);
    defer ref.deinit();
    var ref_f32 = try vk.dispatch.Pipeline.init(&dev, &shaders.qmm_q4_tile_order_ref_f32, layout);
    defer ref_f32.deinit();
    std.debug.print("EXACT device \"{s}\" maxShared={d}\n", .{ std.mem.sliceTo(&dev.properties.deviceName, 0), dev.properties.limits.maxComputeSharedMemorySize });

    var fixture = fx.Fixture.init(init.gpa, &dev, &cmds, staging, &tile);
    defer fixture.deinit();
    fixture.tiled = true;

    var pass: usize = 0;
    var fail: usize = 0;
    for (case_list) |case| {
        const shape: c.Shape = .{ .rows = case.rows, .n = case.n, .k = case.k };
        const name = std.fmt.allocPrint(init.arena.allocator(), "{s}{s}", .{ case.name, tag(case.act_f32) }) catch case.name;
        fixture.act_f32 = case.act_f32;
        if (!try fixture.prepare(shape, case.fill orelse .random, case.seed)) {
            fail += 1;
            std.debug.print("EXACT FAIL {s}: prepare refused\n", .{name});
            continue;
        }
        if (std.mem.indexOf(u8, case.name, "cancel") != null) try plantCancel(&fixture, case.act_f32);
        const units: usize = if (case.act_f32) 2 else 1;
        const data_u16 = shape.rows * shape.n * units;
        const ref_out = try init.gpa.alloc(u16, data_u16);
        defer init.gpa.free(ref_out);
        var p = fx.Params{ .count = @intCast(shape.rows * shape.n), .output_size = @intCast(shape.rows * shape.n), .matrix_m = @intCast(shape.rows), .matrix_n = @intCast(shape.n), .matrix_k = @intCast(shape.k) };
        const old = if (case.act_f32) &ref_f32 else &ref;
        const new = if (case.act_f32) &tile_f32 else &tile;
        try old.bind(&fixture.bindings);
        const old_groups = (p.count + fx.local_size_x - 1) / fx.local_size_x;
        try vk.orExit(old.run(&cmds, std.mem.asBytes(&p), .{ old_groups, 1, 1 }, timeout_ns));
        try vk.orExit(vk.download(&cmds, fixture.buffers[4], 0, staging, std.mem.sliceAsBytes(ref_out), timeout_ns));
        try new.bind(&fixture.bindings);
        const tiles: u32 = @intCast((shape.n + 31) / 32);
        try vk.orExit(new.run(&cmds, std.mem.asBytes(&p), .{ @as(u32, @intCast(shape.rows)) * tiles, 1, 1 }, timeout_ns));
        try vk.orExit(vk.download(&cmds, fixture.buffers[4], 0, staging, std.mem.sliceAsBytes(fixture.out[0..data_u16]), timeout_ns));
        var mismatches: usize = 0;
        var first: usize = 0;
        for (ref_out, fixture.out[0..data_u16], 0..) |a, b, i| {
            if (a != b) {
                if (mismatches == 0) first = i;
                mismatches += 1;
            }
        }
        if (mismatches == 0) {
            pass += 1;
            std.debug.print("EXACT pass {s}: {d} elements byte-identical (rows={d} n={d} k={d})\n", .{ name, data_u16, case.rows, case.n, case.k });
        } else {
            fail += 1;
            const at = first - first % units;
            std.debug.print("EXACT FAIL {s}: {d}/{d} elements differ, first at output[{d}]: ref bytes {x:0>4}{x:0>4} got {x:0>4}{x:0>4}", .{
                name,       mismatches,
                data_u16,   first / units,
                ref_out[at + 1], ref_out[at],
                fixture.out[at + 1], fixture.out[at],
            });
            if (case.act_f32) {
                const ref_bits = (@as(u32, ref_out[at + 1]) << 16) | ref_out[at];
                const got_bits = (@as(u32, fixture.out[at + 1]) << 16) | fixture.out[at];
                std.debug.print(" (ref={d} got={d})", .{ @as(f32, @bitCast(ref_bits)), @as(f32, @bitCast(got_bits)) });
            }
            std.debug.print("\n", .{});
        }
    }
    std.debug.print("SUMMARY exact {d} cases: {d} byte-identical, {d} mismatched\n", .{ pass + fail, pass, fail });
    return if (fail == 0) 0 else 1;
}
