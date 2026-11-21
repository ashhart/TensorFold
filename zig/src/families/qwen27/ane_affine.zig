//! Hybrid ANE+GPU MLP builder for SenseNova-Loop MLX-affine 4-bit and 8-bit g64 checkpoints.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const ane = mtl.ane;
const cp = core.checkpoint_metal;
const lane = core.lane_projection;
const Config = @import("config.zig").Config;
const wts = @import("gpu_weights.zig");
const projection = @import("projection.zig");
const ane_awq = @import("ane_awq.zig");
const width: u32 = 128;

/// True when `w` holds physical SenseNova-Loop MLX-affine g64 MLP matrices compatible with a `columns` ANE split.
pub fn supports(w: *const wts.Weights, c: Config, columns: u32) bool {
    if (!c.looped or columns == 0 or columns > c.intermediate or columns % 128 != 0) return false;
    if (c.hidden % 128 != 0 or c.intermediate % 128 != 0 or w.physical_layers.len != c.physicalLayers()) return false;
    for (w.physical_layers) |l| {
        for ([_]projection.Linear{ l.gu, l.down }) |p| {
            if (p.sn != null or p.runs.len > 0 or p.group != 64) return false;
            if (p.bits != 4 and p.bits != 8) return false;
        }
    }
    return true;
}

/// GPU bytes required for physical MLX-affine remainder weights `[columns .. intermediate)`.
pub fn tailBytes(w: *const wts.Weights, c: Config, columns: u32) u64 {
    if (!c.looped or columns >= c.intermediate or w.physical_layers.len != c.physicalLayers()) return 0;
    const rem: u64 = c.intermediate - columns;
    const h: u64 = c.hidden;
    var total: u64 = 0;
    for (w.physical_layers) |l| {
        const gu_b = (2 * rem) * (h * l.gu.bits / 8 + (h / 64) * 4);
        const dn_b = h * (rem * l.down.bits / 8 + (rem / 64) * 4);
        total += gu_b + dn_b;
    }
    return total;
}

pub fn build(
    a: std.mem.Allocator,
    device: mtl.Device,
    ck: *const cp.Checkpoint,
    w: *const wts.Weights,
    c: Config,
    columns: u32,
    slices: u32,
    want_int8: bool,
    inputs: []const ane.Surface,
    outputs: []const ane.Surface,
    programs: []ane.Program,
    requests: []ane.Request,
) !*ane_awq.SnTail {
    const phys_n = c.physicalLayers();
    if (!supports(w, c, columns) or inputs.len != slices or outputs.len != slices or programs.len != phys_n or requests.len != phys_n * slices) {
        return error.BadAneShare;
    }
    const add_lib = try mtl.Library.fromSource(device, ane_awq.add_ane_bf16_source, mtl.CompileOptions.mlx());
    errdefer add_lib.deinit();
    const add_pipe = try mtl.Pipeline.init(device, add_lib, "q27_add_ane_bf16", false);
    errdefer add_pipe.deinit();
    const t = try a.create(ane_awq.SnTail);
    errdefer a.destroy(t);
    const rem: u32 = @intCast(c.intermediate - columns);
    t.* = .{ .a = a, .gu = try a.alloc(projection.Linear, phys_n), .down = try a.alloc(projection.Linear, phys_n), .add_lib = add_lib, .add_pipe = add_pipe, .rem = rem, .int8_active = want_int8 };
    errdefer {
        for (t.owned.items) |b| b.deinit();
        t.owned.deinit(a);
        a.free(t.gu);
        a.free(t.down);
    }

    const prefix = ane_awq.textNamespace(ck);
    var stage = cp.Stage{ .device = device };
    defer stage.deinit();
    var made: usize = 0;
    errdefer for (0..made) |layer| {
        for (requests[layer * slices ..][0..slices]) |req| req.deinit();
        programs[layer].deinit();
    };

    const f16_blob = try ane.Blob.init(a, &.{ 2 * columns * c.hidden, c.hidden * columns });
    defer f16_blob.deinit(a);
    const f16_mil = try ane.mlpSlice(a, c.hidden, columns, width, f16_blob.offsets[0], f16_blob.offsets[1]);
    defer a.free(f16_mil);
    var i8_blob: ?ane.Blob = if (want_int8) try ane.Blob.initInt8(a, 2 * columns * c.hidden, 2 * columns, c.hidden * columns, c.hidden) else null;
    defer if (i8_blob) |b| b.deinit(a);
    const i8_mil: ?[]u8 = if (i8_blob) |b| try ane.mlpSliceInt8(a, c.hidden, columns, width, b.offsets[0], b.offsets[1], b.offsets[2], b.offsets[3]) else null;
    defer if (i8_mil) |m| a.free(m);

    const dn_stage: ?mtl.Buffer = if (rem > 0) try device.buffer(
        c.hidden * rem + 2 * c.hidden * (rem / 64) * 2,
        mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked,
    ) else null;
    defer if (dn_stage) |b| b.deinit();

    var reg_jobs: std.ArrayList(lane.Order) = .empty;
    defer reg_jobs.deinit(a);

    for (0..phys_n) |phys| {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const ta = arena.allocator();
        const gu_bits = w.physical_layers[phys].gu.bits;
        const dn_bits = w.physical_layers[phys].down.bits;
        var names: [9][]const u8 = undefined;
        var n_idx: usize = 0;
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
            for ([_][]const u8{ ".weight", ".scales", ".biases" }) |suf| {
                names[n_idx] = try std.fmt.allocPrint(ta, "{s}layers.{d}.mlp.{s}{s}", .{ prefix, phys, proj, suf });
                n_idx += 1;
            }
        }
        try stage.reserve(ck, &names);
        defer stage.reset();

        const g_t = try readAffine(&stage, ck, names[0..3], c.intermediate, c.hidden, gu_bits);
        const u_t = try readAffine(&stage, ck, names[3..6], c.intermediate, c.hidden, gu_bits);
        const d_t = try readAffine(&stage, ck, names[6..9], c.hidden, c.intermediate, dn_bits);

        const f1 = f16_blob.values(0, 2 * columns * c.hidden);
        try dequantAffinePrefix(g_t, columns, c.hidden, f1[0 .. columns * c.hidden]);
        try dequantAffinePrefix(u_t, columns, c.hidden, f1[columns * c.hidden ..]);
        try dequantAffinePrefix(d_t, c.hidden, columns, f16_blob.values(1, c.hidden * columns));

        var prog: ?ane.Program = null;
        if (t.int8_active) {
            ane_awq.quantRows(f1, 2 * columns, c.hidden, i8_blob.?.int8Values(0, 2 * columns * c.hidden), i8_blob.?.values(1, 2 * columns));
            ane_awq.quantRows(f16_blob.values(1, c.hidden * columns), c.hidden, columns, i8_blob.?.int8Values(2, c.hidden * columns), i8_blob.?.values(3, c.hidden));
            prog = ane.Program.init(i8_mil.?, i8_blob.?.bytes) catch blk: {
                std.log.warn("qwen27: ANE INT8 MIL compile failed on layer {d}; falling back to FP16 ANE slices", .{phys});
                t.int8_active = false;
                break :blk null;
            };
        }
        programs[phys] = prog orelse try ane.Program.init(f16_mil, f16_blob.bytes);
        const own = requests[phys * slices ..][0..slices];
        for (own, inputs, outputs, 0..) |*r, inp, outp, req_made| r.* = programs[phys].request(inp, outp) catch |err| {
            for (own[0..req_made]) |x| x.deinit();
            programs[phys].deinit();
            return err;
        };
        if (rem > 0) {
            const g_rem = sliceRows(g_t, columns, rem);
            const u_rem = sliceRows(u_t, columns, rem);
            const d_rem = compactCols(d_t, dn_stage.?, c.hidden, c.intermediate, columns, rem);
            t.gu[phys] = try projection.prepare(device, a, &t.owned, &.{ g_rem, u_rem }, 32);
            t.down[phys] = try projection.prepare(device, a, &t.owned, &.{d_rem}, 32);
            if (device.tensorUnits()) {
                for ([_]projection.Linear{ t.gu[phys], t.down[phys] }) |p| {
                    if (p.bits == 4 and p.group == 64 and p.n % 32 == 0) {
                        try reg_jobs.append(a, .{ .w = .{ .buf = p.words.buffer, .off = p.words.offset }, .n = p.n, .k = p.k });
                    }
                }
            }
        }
        made += 1;
    }
    if (reg_jobs.items.len > 0) try lane.regOrder(device, reg_jobs.items);
    std.log.info("qwen27: ANE affine share built layers={d} columns={d} rem={d} mode={s}", .{ phys_n, columns, rem, if (t.int8_active) "int8" else "fp16" });
    return t;
}

fn readAffine(stage: *cp.Stage, ck: *const cp.Checkpoint, names: []const []const u8, n: usize, k: usize, bits: u8) !projection.Triple {
    const w = try stage.read(ck, names[0]);
    const s = try stage.read(ck, names[1]);
    const b = try stage.read(ck, names[2]);
    if (w.dtype != .u32 or s.dtype != .bf16 or b.dtype != .bf16 or w.rank != 2 or s.rank != 2 or b.rank != 2) return error.BadAneShare;
    if (w.shape[0] != n or w.shape[1] != k * bits / 32 or s.shape[0] != n or b.shape[0] != n or s.shape[1] != k / 64 or b.shape[1] != k / 64) return error.BadAneShare;
    return .{ .weight = w, .scales = s, .biases = b, .bits = bits, .group = 64, .gn = n };
}

fn dequantAffinePrefix(p: projection.Triple, rows: usize, cols: usize, out: []f16) !void {
    const k = p.weight.shape[1] * 32 / p.bits;
    if (p.group != 64 or rows > p.weight.shape[0] or cols > k or cols % 64 != 0 or out.len != rows * cols) return error.BadAneShare;
    return dequantAffineSlices(p.weight.host(u32), p.scales.host(u16), p.biases.host(u16), p.bits, k, rows, cols, out);
}

fn dequantAffineSlices(words: []const u32, scales: []const u16, biases: []const u16, bits: u8, full_k: usize, rows: usize, cols: usize, out: []f16) !void {
    if (bits != 4 and bits != 8) return error.BadAneShare;
    const rw = full_k * bits / 32;
    const rg = full_k / 64;
    const per_word: usize = 32 / bits;
    const mask: u32 = if (bits == 4) 0xf else 0xff;
    for (0..rows) |r| {
        for (0..cols / 64) |g| {
            const sc: f32 = @bitCast(@as(u32, scales[r * rg + g]) << 16);
            const bi: f32 = @bitCast(@as(u32, biases[r * rg + g]) << 16);
            for (0..64) |j| {
                const ki = g * 64 + j;
                const sh: u5 = @intCast((ki % per_word) * bits);
                const q = (words[r * rw + ki / per_word] >> sh) & mask;
                out[r * cols + ki] = @floatCast(@as(f32, @floatFromInt(q)) * sc + bi);
            }
        }
    }
}

fn sliceRowTensor(t: cp.Tensor, row0: usize, rows: usize) cp.Tensor {
    const row_bytes = t.shape[1] * t.dtype.size();
    var out = t;
    out.offset += row0 * row_bytes;
    out.bytes = rows * row_bytes;
    out.shape[0] = rows;
    return out;
}

fn sliceRows(p: projection.Triple, row0: usize, rows: usize) projection.Triple {
    return .{
        .weight = sliceRowTensor(p.weight, row0, rows),
        .scales = sliceRowTensor(p.scales, row0, rows),
        .biases = sliceRowTensor(p.biases, row0, rows),
        .bits = p.bits,
        .group = p.group,
        .gn = rows,
    };
}

fn compactCols(p: projection.Triple, buf: mtl.Buffer, rows: usize, full_k: usize, col0: usize, rem_k: usize) projection.Triple {
    const rw = rem_k * p.bits / 32;
    const fw = full_k * p.bits / 32;
    const c_w = col0 * p.bits / 32;
    const rg = rem_k / 64;
    const fg = full_k / 64;
    const c_g = col0 / 64;
    const w_bytes = rows * rw * 4;
    const s_bytes = rows * rg * 2;
    const w_off: usize = 0;
    const s_off: usize = w_bytes;
    const b_off: usize = w_bytes + s_bytes;
    const base = buf.contents();
    const dst_w: [*]u32 = @ptrCast(@alignCast(base + w_off));
    const dst_s: [*]u16 = @ptrCast(@alignCast(base + s_off));
    const dst_b: [*]u16 = @ptrCast(@alignCast(base + b_off));
    const src_w = p.weight.host(u32);
    const src_s = p.scales.host(u16);
    const src_b = p.biases.host(u16);
    for (0..rows) |r| {
        @memcpy(dst_w[r * rw ..][0..rw], src_w[r * fw + c_w ..][0..rw]);
        @memcpy(dst_s[r * rg ..][0..rg], src_s[r * fg + c_g ..][0..rg]);
        @memcpy(dst_b[r * rg ..][0..rg], src_b[r * fg + c_g ..][0..rg]);
    }
    return .{
        .weight = .{ .buffer = buf, .offset = w_off, .bytes = w_bytes, .dtype = .u32, .shape = .{ rows, rw, 1, 1, 1 }, .rank = 2 },
        .scales = .{ .buffer = buf, .offset = s_off, .bytes = s_bytes, .dtype = .bf16, .shape = .{ rows, rg, 1, 1, 1 }, .rank = 2 },
        .biases = .{ .buffer = buf, .offset = b_off, .bytes = s_bytes, .dtype = .bf16, .shape = .{ rows, rg, 1, 1, 1 }, .rank = 2 },
        .bits = p.bits,
        .group = p.group,
        .gn = rows,
    };
}

test "ane_affine dequantizes 4-bit and 8-bit g64 affine prefixes accurately" {
    const a = std.testing.allocator;
    const bf_one: u16 = 0x3f80;
    const bf_neg: u16 = 0xbf80;
    const scales = [_]u16{ bf_one, bf_one };
    const biases = [_]u16{ bf_neg, 0x3f00 };
    var w4: [16]u32 = @splat(0);
    w4[0] = 0x0000_0031;
    w4[8] = 0x0000_0025;
    const out4 = try a.alloc(f16, 128);
    defer a.free(out4);
    try dequantAffineSlices(&w4, &scales, &biases, 4, 64, 2, 64, out4);
    try std.testing.expectEqual(@as(f32, 0.0), @as(f32, @floatCast(out4[0])));
    try std.testing.expectEqual(@as(f32, 2.0), @as(f32, @floatCast(out4[1])));
    try std.testing.expectEqual(@as(f32, 5.5), @as(f32, @floatCast(out4[64])));
    try std.testing.expectEqual(@as(f32, 2.5), @as(f32, @floatCast(out4[65])));

    var w8: [32]u32 = @splat(0);
    w8[0] = 0x0000_0402;
    w8[16] = 0x0000_0107;
    try dequantAffineSlices(&w8, &scales, &biases, 8, 64, 2, 64, out4);
    try std.testing.expectEqual(@as(f32, 1.0), @as(f32, @floatCast(out4[0])));
    try std.testing.expectEqual(@as(f32, 3.0), @as(f32, @floatCast(out4[1])));
    try std.testing.expectEqual(@as(f32, 7.5), @as(f32, @floatCast(out4[64])));
    try std.testing.expectEqual(@as(f32, 1.5), @as(f32, @floatCast(out4[65])));
}

test "ane_affine supports checks looped g64 physical MLPs and computes tailBytes" {
    const a = std.testing.allocator;
    const c = Config{
        .hidden = 4096,
        .intermediate = 12288,
        .layers = 4,
        .vocab = 248320,
        .heads = 16,
        .kv_heads = 4,
        .head_dim = 256,
        .k_heads = 16,
        .v_heads = 32,
        .dk = 128,
        .dv = 128,
        .conv_kernel = 4,
        .eps = 1e-6,
        .rope_dims = 64,
        .rope_theta = 1e7,
        .looped = true,
        .physical_layers = 2,
    };
    const phys = try a.alloc(wts.Layer, 2);
    defer a.free(phys);
    const dummy_lin = projection.Linear{ .words = .{ .buffer = undefined }, .pairs = .{ .buffer = undefined }, .n = 4096, .k = 4096, .tile = 32, .slices = 1, .bits = 8, .group = 64 };
    for (phys) |*l| l.* = .{ .norm = undefined, .post_norm = undefined, .gu = dummy_lin, .down = dummy_lin, .mixer = .{ .attention = .{ .q = dummy_lin, .kv = dummy_lin, .out = dummy_lin, .q_norm = undefined, .k_norm = undefined } } };
    phys[1].gu.bits = 4;
    var w = wts.Weights{ .allocator = a, .layers = &.{}, .physical_layers = phys, .embed = undefined, .norm = undefined, .head = dummy_lin };
    try std.testing.expect(supports(&w, c, 4096));
    try std.testing.expect(!supports(&w, c, 4096 - 64));
    try std.testing.expect(tailBytes(&w, c, 4096) > 0);
    phys[0].gu.group = 128;
    try std.testing.expect(!supports(&w, c, 4096));
}
