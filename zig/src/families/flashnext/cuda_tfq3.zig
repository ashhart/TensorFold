//! Flash Next TFQ3 (3/4/5-bit 2D-FWHT) CUDA loader: layer_00..47.tffpw + mtp_00.tffpw + MLX 8b/32 & 3b/32 PLE.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const wt = @import("cuda_prompt_wt.zig");
const embed = @import("cuda_embed.zig");
const projection = @import("cuda_projection.zig");
const mixed = @import("cuda_mixed_experts.zig");

const bf16Of = core.bf16_round.bits;

fn bf16f(b: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, b[2 * i ..][0..2], .little)) << 16);
}

/// Returns true if `dir` is a Flash Next TFQ3 checkpoint (`pw_v3_tables.bin` + `layer_00.tffpw`).
pub fn isTfq3(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    return cuda.tfq3_format.hasTables(gpa, io, dir) and cuda.tfq3_format.hasRecords(gpa, io, dir, "layer_00.tffpw", cuda.tfq3_format.magic_tfpw);
}

/// Resolves the `.tffpw` file for decoder or MTP layer `root`, or null if absent.
pub fn layerFile(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, root: []const u8) ?[]u8 {
    if (!cuda.tfq3_format.hasTables(gpa, io, dir)) return null;
    var name_buf: [32]u8 = undefined;
    const fname: []const u8 = if (std.mem.startsWith(u8, root, "language_model.model.layers.")) blk: {
        const idx = std.fmt.parseInt(u32, root["language_model.model.layers.".len..], 10) catch return null;
        break :blk std.fmt.bufPrint(&name_buf, "layer_{d:0>2}.tffpw", .{idx}) catch return null;
    } else if (std.mem.startsWith(u8, root, "language_model.mtp.layers.")) blk: {
        const idx = std.fmt.parseInt(u32, root["language_model.mtp.layers.".len..], 10) catch return null;
        break :blk std.fmt.bufPrint(&name_buf, "mtp_{d:0>2}.tffpw", .{idx}) catch return null;
    } else return null;
    if (!cuda.tfq3_format.hasFile(gpa, io, dir, fname)) return null;
    return std.fs.path.join(gpa, &.{ dir, fname }) catch null;
}

/// Dequantizes one MLX 3-bit group-32 row (`3` u32 words per 32 elements) to BF16.
pub fn dequantRow3(weight: []const u8, scales: []const u8, biases: []const u8, dims: usize, row: usize, out: []u16) !void {
    if (dims == 0 or dims % 32 != 0 or out.len != dims) return error.UnexpectedTensor;
    const groups = dims / 32;
    const wbytes = groups * 12;
    const gbytes = groups * 2;
    const woff = row * wbytes;
    const soff = row * gbytes;
    if (woff + wbytes > weight.len or soff + gbytes > scales.len or soff + gbytes > biases.len) return error.UnexpectedTensor;
    for (0..groups) |g| {
        const sc = bf16f(scales[soff..], g);
        const bi = bf16f(biases[soff..], g);
        const gw = weight[woff + g * 12 ..][0..12];
        for (0..32) |j| {
            const bit = j * 3;
            const wi = bit >> 5;
            const sh: u5 = @intCast(bit & 31);
            var v = std.mem.readInt(u32, gw[wi * 4 ..][0..4], .little) >> sh;
            if (sh > 29) v |= std.mem.readInt(u32, gw[(wi + 1) * 4 ..][0..4], .little) << @intCast(32 - @as(u6, sh));
            const q: f32 = @floatFromInt(v & 7);
            out[g * 32 + j] = bf16Of(q * sc + bi);
        }
    }
}

/// Dequantizes one MLX 8-bit group-32 row (`8` u32 words per 32 elements) to BF16.
pub fn dequantRow8(weight: []const u8, scales: []const u8, biases: []const u8, dims: usize, row: usize, out: []u16) !void {
    if (dims == 0 or dims % 32 != 0 or out.len != dims) return error.UnexpectedTensor;
    const groups = dims / 32;
    const wbytes = dims;
    const gbytes = groups * 2;
    const woff = row * wbytes;
    const soff = row * gbytes;
    if (woff + wbytes > weight.len or soff + gbytes > scales.len or soff + gbytes > biases.len) return error.UnexpectedTensor;
    for (0..groups) |g| {
        const sc = bf16f(scales[soff..], g);
        const bi = bf16f(biases[soff..], g);
        const gw = weight[woff + g * 32 ..][0..32];
        for (0..32) |j| {
            const q: f32 = @floatFromInt(gw[j]);
            out[g * 32 + j] = bf16Of(q * sc + bi);
        }
    }
}

/// Dispatches MLX 3/4/8-bit group-32 PLE row dequantization.
pub fn dequantPleRow(bits: u8, weight: []const u8, scales: []const u8, biases: []const u8, dh: usize, row: usize, out: []u16) !void {
    return switch (bits) {
        3 => dequantRow3(weight, scales, biases, dh, row, out),
        4 => embed.dequant(weight, scales, biases, dh, row, out),
        8 => dequantRow8(weight, scales, biases, dh, row, out),
        else => error.UnsupportedQuantization,
    };
}

/// Dequantizes a 2D MLX 8-bit group-32 projection `[n, k]` into contiguous BF16 bytes.
pub fn dequantMatrix8(gpa: std.mem.Allocator, store: *wt.Store, prefix: []const u8, n: usize, k: usize) ![]u8 {
    var buf: [512]u8 = undefined;
    const planes = try wt.join3(gpa, store, &buf, prefix, "");
    defer gpa.free(planes.w);
    defer gpa.free(planes.s);
    defer gpa.free(planes.b);
    if (planes.w.len != n * k or planes.s.len != n * (k / 32) * 2 or planes.b.len != n * (k / 32) * 2) return error.UnexpectedTensor;
    const out = try gpa.alloc(u16, n * k);
    errdefer gpa.free(out);
    for (0..n) |r| try dequantRow8(planes.w, planes.s, planes.b, k, r, out[r * k ..][0..k]);
    return std.mem.sliceAsBytes(out);
}

/// Loads one MLX 8-bit group-32 projection `[n, k]` as a dense BF16 qmmf weight.
pub fn loadMlx8Dense(gpa: std.mem.Allocator, d: *const cuda.Driver, store: *wt.Store, prefix: []const u8, n: usize, k: usize) !wt.Q4 {
    const bytes = try dequantMatrix8(gpa, store, prefix, n, k);
    defer gpa.free(bytes);
    return projection.dense(gpa, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = bytes });
}

/// Loads stitched MLX 8-bit group-32 projections `[n, k]` as one dense BF16 qmmf weight.
pub fn loadMlx8StitchedDense(gpa: std.mem.Allocator, d: *const cuda.Driver, store: *wt.Store, prefix: []const u8, names: []const []const u8, n: usize, k: usize) !wt.Q4 {
    const out = try gpa.alloc(u8, try std.math.mul(usize, try std.math.mul(usize, n, k), 2));
    defer gpa.free(out);
    var buf: [512]u8 = undefined;
    var at: usize = 0;
    for (names) |name| {
        var mp_buf: [512]u8 = undefined;
        const mod_path = try std.fmt.bufPrint(&mp_buf, "{s}{s}", .{ prefix, name });
        const tw = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{mod_path}));
        if (tw.dtype != .u32 or tw.rank != 2 or tw.dim(1) != k / 4) return error.UnexpectedTensor;
        const rn = tw.dim(0);
        const part = try dequantMatrix8(gpa, store, mod_path, rn, k);
        defer gpa.free(part);
        if (part.len > out.len - at) return error.UnexpectedTensor;
        @memcpy(out[at..][0..part.len], part);
        at += part.len;
    }
    if (at != out.len) return error.UnexpectedTensor;
    return projection.dense(gpa, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = out });
}

/// Loads a hyper-connection block whose weights are MLX 8-bit/32 into dense BF16 qmmf weights.
pub fn loadMlx8Hc(gpa: std.mem.Allocator, d: *const cuda.Driver, store: *wt.Store, prefix: []const u8, inject: bool) !wt.HcW {
    var buf: [240]u8 = undefined;
    const n = wt.low + if (inject) wt.streams else @as(usize, 0);
    const data = try gpa.alloc(u8, n * wt.wide * 2);
    defer gpa.free(data);
    const down_bytes = try dequantMatrix8(gpa, store, try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_down", .{prefix}), wt.low, wt.wide);
    defer gpa.free(down_bytes);
    @memcpy(data[0..down_bytes.len], down_bytes);
    if (inject) {
        const inj_bytes = try dequantMatrix8(gpa, store, try std.fmt.bufPrint(&buf, "{s}.block_inject_weight", .{prefix}), wt.streams, wt.wide);
        defer gpa.free(inj_bytes);
        @memcpy(data[wt.low * wt.wide * 2 ..], inj_bytes);
    }
    var dq = try projection.dense(gpa, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, wt.wide, 1, 1, 1 }, .bytes = data });
    errdefer dq.free();
    var uq = try loadMlx8Dense(gpa, d, store, try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_up", .{prefix}), wt.wide, wt.low);
    errdefer uq.free();
    const gamma = try store.normScale(gpa, try store.tensor(try std.fmt.bufPrint(&buf, "{s}.hc_norm.weight", .{prefix})));
    defer gpa.free(gamma);
    if (gamma.len != wt.wide) return error.UnexpectedTensor;
    const scale = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(gamma));
    return .{ .scale = scale, .down = dq, .up = uq, .n_down = n, .has_inj = @intFromBool(inject) };
}

/// Loads an MLX 8-bit/32 `embed_tokens` table as a dense BF16 device table.
pub fn uploadEmbedMlx8(gpa: std.mem.Allocator, d: *const cuda.Driver, store: *wt.Store) !embed.Table {
    const bytes = try dequantMatrix8(gpa, store, "language_model.model.embed_tokens", wt.vocab, wt.dims);
    defer gpa.free(bytes);
    return embed.Table.uploadDense(d, .{ .dtype = .bf16, .rank = 2, .shape = .{ wt.vocab, wt.dims, 1, 1, 1 }, .bytes = bytes });
}

/// Dequantizes selected proposal vocabulary rows (`ids`) from an MLX 8-bit/32 `lm_head` to BF16 bytes.
pub fn draftRows8(gpa: std.mem.Allocator, store: *wt.Store, ids: []const u32) ![]u8 {
    var buf: [120]u8 = undefined;
    const head = try wt.join3(gpa, store, &buf, "language_model.lm_head", "");
    defer gpa.free(head.w);
    defer gpa.free(head.s);
    defer gpa.free(head.b);
    if (head.w.len != wt.vocab * wt.dims or head.s.len != wt.vocab * (wt.dims / 32) * 2) return error.UnexpectedTensor;
    const out = try gpa.alloc(u16, ids.len * wt.dims);
    errdefer gpa.free(out);
    for (ids, 0..) |id, i| {
        if (id >= wt.vocab) return error.UnexpectedTensor;
        try dequantRow8(head.w, head.s, head.b, wt.dims, id, out[i * wt.dims ..][0..wt.dims]);
    }
    return std.mem.sliceAsBytes(out);
}

/// Loads one projection (`BF16`, `MXFP8`, `MLX-8b/32`, or `MLX-4b/32`) into a native qmmf weight for `shared_expert`.
fn loadNativeShared(gpa: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, key: []const u8, n: usize, k: usize) !wt.Q4 {
    var buf: [512]u8 = undefined;
    const t = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{key}));
    if (t.dtype == .u32 and t.rank == 2 and t.dim(1) == k / 8) {
        const planes = try wt.join3(gpa, store, &buf, key, "");
        defer gpa.free(planes.w);
        defer gpa.free(planes.s);
        defer gpa.free(planes.b);
        const out = try gpa.alloc(u16, n * k);
        defer gpa.free(out);
        for (0..n) |r| try embed.dequant(planes.w, planes.s, planes.b, k, r, out[r * k ..][0..k]);
        return projection.dense(gpa, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(out) });
    }
    const q = try @import("cuda_projection_load.zig").load(gpa, d, store, key, n, k);
    if (q.native == null) return error.UnsupportedQuantization;
    return q;
}

/// Loads a TFQ3 `.tffpw` routed-expert layer and its native shared expert into `mixed.Owned`.
pub fn loadMixed(gpa: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, mlp_root: []const u8, tffpw_path: []const u8) !mixed.Owned {
    var tables = try cuda.tfq3.uploadTables(gpa, store.io, d, store.dir);
    errdefer tables.free();
    var moe_layer = try cuda.tfq3.uploadTfpwLayer(store.io, d, tffpw_path, tables.ptr, wt.experts_n, wt.dims, wt.moe_w);
    errdefer moe_layer.deinit();
    var shared: [3]wt.Q4 = @splat(wt.Q4.empty(d));
    errdefer for (&shared) |*p| p.free();
    var buf: [256]u8 = undefined;
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |name, i| {
        const n = if (i == 2) wt.dims else wt.moe_w;
        const k = if (i == 2) wt.moe_w else wt.dims;
        const key = try std.fmt.bufPrint(&buf, "{s}.shared_expert.{s}", .{ mlp_root, name });
        shared[i] = try loadNativeShared(gpa, d, store, key, n, k);
    }
    const zero: cuda.DeviceBuffer = .{ .d = d, .ptr = 0, .len = 0 };
    return .{
        .experts = .{ .up = zero, .down = zero, .up_scale = zero, .down_scale = zero, .width = wt.moe_w, .dims = wt.dims, .count = wt.experts_n },
        .tfq3 = moe_layer,
        .tfq3_tables = tables,
        .shared = shared,
        .rank = 0,
        .world = 1,
    };
}

/// Resident GPU byte estimate for a Flash Next TFQ3 checkpoint (layers + non-MoE BF16 + optional MTP).
pub fn deviceWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, drafts: bool) !u64 {
    var total: u64 = 12 << 30; // dequantized non-MoE BF16 weights + tables + scratch
    var name_buf: [32]u8 = undefined;
    for (0..wt.layers_n) |i| {
        const fname = try std.fmt.bufPrint(&name_buf, "layer_{d:0>2}.tffpw", .{i});
        const p = try std.fs.path.join(gpa, &.{ dir, fname });
        defer gpa.free(p);
        const f = std.Io.Dir.cwd().openFile(io, p, .{}) catch continue;
        defer f.close(io);
        total += try f.length(io);
    }
    if (drafts) {
        const p = try std.fs.path.join(gpa, &.{ dir, "mtp_00.tffpw" });
        defer gpa.free(p);
        if (std.Io.Dir.cwd().openFile(io, p, .{})) |f| {
            defer f.close(io);
            total += try f.length(io);
        } else |_| {}
        total += 1 << 30;
    }
    return total;
}

test "MLX 3-bit and 8-bit group-32 dequantization matches exact affine values" {
    // 3-bit: pack codes 0,1,2,3,4,5,6,7 repeated 4 times into 3 u32 words (12 bytes)
    var w3: [12]u8 = @splat(0);
    for (0..32) |j| {
        const code: u32 = @intCast(j & 7);
        const bit = j * 3;
        const wi = bit >> 5;
        const sh: u5 = @intCast(bit & 31);
        var lo = std.mem.readInt(u32, w3[wi * 4 ..][0..4], .little);
        lo |= code << sh;
        std.mem.writeInt(u32, w3[wi * 4 ..][0..4], lo, .little);
        if (sh > 29) {
            var hi = std.mem.readInt(u32, w3[(wi + 1) * 4 ..][0..4], .little);
            hi |= code >> @intCast(32 - @as(u6, sh));
            std.mem.writeInt(u32, w3[(wi + 1) * 4 ..][0..4], hi, .little);
        }
    }
    var sc: [2]u8 = undefined;
    var bi: [2]u8 = undefined;
    std.mem.writeInt(u16, &sc, bf16Of(2.0), .little);
    std.mem.writeInt(u16, &bi, bf16Of(-3.0), .little);
    var out3: [32]u16 = undefined;
    try dequantRow3(&w3, &sc, &bi, 32, 0, &out3);
    for (0..32) |j| try std.testing.expectEqual(bf16Of(@as(f32, @floatFromInt(j & 7)) * 2.0 - 3.0), out3[j]);

    // 8-bit: codes 0..31 directly in 32 bytes
    var w8: [32]u8 = undefined;
    for (&w8, 0..) |*b, j| b.* = @intCast(j);
    var out8: [32]u16 = undefined;
    try dequantRow8(&w8, &sc, &bi, 32, 0, &out8);
    for (0..32) |j| try std.testing.expectEqual(bf16Of(@as(f32, @floatFromInt(j)) * 2.0 - 3.0), out8[j]);
}
