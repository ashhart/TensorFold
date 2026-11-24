//! Inkling 2 Mini DFlash 2 --drafter-bits 4: quantizes rank-2 linears at load and projects via 4-bit copies.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const ck = @import("../../../core/checkpoint_metal.zig");
const simd = core.row_projection.simd;

pub const Ref = @import("inkling2_metal_ref").Ref;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// The draft's linears: its rank-2 `.weight` tensors; norms, conv base kernels and codebooks remain BF16.
pub fn linear(name: []const u8, rank: usize) bool {
    return rank == 2 and std.mem.endsWith(u8, name, ".weight");
}

/// Stable lookup key for a named linear tensor inside one DFlash2 checkpoint.
pub fn keyOf(name: []const u8) usize {
    return @truncate(std.hash.Wyhash.hash(0, name));
}

/// K-slice split matching qwen27/projection.zig for tensor-unit 4-bit lane projections.
pub fn split(n: usize, k: usize) !u32 {
    if (n == 0 or k == 0 or n % 4 != 0 or k % 64 != 0) return error.BadProjectionShape;
    var slices: u32 = 1;
    while (slices < 8 and (n + 31) / 32 * slices < 1024 and k / 64 / (2 * slices) >= 8) slices *= 2;
    return slices;
}

/// Pre-M5 / non-tensor-unit chips: a prepared q4/g64 matrix in MLX row layout for simdgroup-matrix tiles.
pub const Tiled = struct {
    w: mtl.Buffer,
    scales: mtl.Buffer,
    biases: mtl.Buffer,
    n: usize,
    k: usize,

    pub fn init(device: mtl.Device, p: core.affine4.Prepared) !Tiled {
        const w = try device.buffer(p.words.len * 4, opts);
        errdefer w.deinit();
        const scales = try device.buffer(p.scales.len * 2, opts);
        errdefer scales.deinit();
        const biases = try device.buffer(p.biases.len * 2, opts);
        @memcpy(w.slice(u32, p.words.len), p.words);
        @memcpy(scales.slice(u16, p.scales.len), p.scales);
        @memcpy(biases.slice(u16, p.biases.len), p.biases);
        return .{ .w = w, .scales = scales, .biases = biases, .n = p.n, .k = p.k };
    }

    pub fn deinit(t: Tiled) void {
        t.w.deinit();
        t.scales.deinit();
        t.biases.deinit();
    }

    pub fn matrix(t: Tiled) simd.Matrix {
        return .{ .w = t.w, .scales = t.scales, .biases = t.biases, .n = t.n, .k = t.k };
    }
};

/// Every linear's 4-bit copy, keyed by tensor name hash (keyOf): lane adapters with tensor units, else SIMD tiles.
pub const Copies = struct {
    gpa: std.mem.Allocator,
    adapters: std.AutoHashMapUnmanaged(usize, core.affine4_lane.Adapter) = .empty,
    tiled: std.AutoHashMapUnmanaged(usize, Tiled) = .empty,
    simd: ?simd.Pipelines = null,

    pub fn deinit(c: *Copies) void {
        var adapters = c.adapters.valueIterator();
        while (adapters.next()) |a| a.deinit();
        c.adapters.deinit(c.gpa);
        var tiled = c.tiled.valueIterator();
        while (tiled.next()) |t| t.deinit();
        c.tiled.deinit(c.gpa);
        if (c.simd) |*p| p.deinit();
        c.* = undefined;
    }

    /// Resident device bytes held by prepared 4-bit weights, metadata and adapter scratch buffers.
    pub fn bytes(c: *const Copies) u64 {
        var n: u64 = 0;
        var adapters = c.adapters.valueIterator();
        while (adapters.next()) |a| {
            n += a.words.length() + a.metadata.length();
            if (a.projection.sums_buffer) |b| n += b.length();
        }
        var tiled = c.tiled.valueIterator();
        while (tiled.next()) |t| n += t.w.length() + t.scales.length() + t.biases.length();
        return n;
    }

    /// Quantizes `checkpoint`'s `linear_count` rank-2 `.weight` tensors on `device`.
    pub fn prepare(gpa: std.mem.Allocator, device: mtl.Device, checkpoint: *ck.Checkpoint, linear_count: usize) !Copies {
        return prepareFor(gpa, device, checkpoint, linear_count, device.tensorUnits());
    }

    /// Quantizes with explicit `use_tensor_units` selection so tests can exercise both M5 and pre-M5 paths.
    pub fn prepareFor(gpa: std.mem.Allocator, device: mtl.Device, checkpoint: *ck.Checkpoint, linear_count: usize, use_tensor_units: bool) !Copies {
        if (use_tensor_units and !device.tensorUnits()) return error.TensorUnitsRequired;
        var c: Copies = .{ .gpa = gpa };
        errdefer c.deinit();
        const order = if (use_tensor_units) try core.lane_projection.RegOrder.init(device) else null;
        defer if (order) |o| o.deinit();
        if (!use_tensor_units) c.simd = try simd.Pipelines.load(device);
        var rows: std.ArrayList(u16) = .empty;
        defer rows.deinit(gpa);
        var kept = checkpoint.tensors.iterator();
        while (kept.next()) |entry| if (linear(entry.key_ptr.*, entry.value_ptr.rank)) {
            const t = entry.value_ptr.*;
            const count = std.math.mul(usize, t.shape[0], t.shape[1]) catch return error.DraftPreparationUnqualified;
            const want_bytes = std.math.mul(usize, count, 2) catch return error.DraftPreparationUnqualified;
            if (t.dtype != .bf16 or t.bytes != want_bytes) return error.DraftPreparationUnqualified;
            try c.one(device, order, keyOf(entry.key_ptr.*), t.dtype, t.shape, t.host(u16));
        };
        var later = checkpoint.deferred.iterator();
        while (later.next()) |entry| if (linear(entry.key_ptr.*, entry.value_ptr.rank)) {
            const d = entry.value_ptr.*;
            const count = std.math.mul(usize, d.shape[0], d.shape[1]) catch return error.DraftPreparationUnqualified;
            const want_bytes = std.math.mul(usize, count, 2) catch return error.DraftPreparationUnqualified;
            if (d.dtype != .bf16 or d.bytes != want_bytes) return error.DraftPreparationUnqualified;
            try rows.resize(gpa, count);
            try checkpoint.read(entry.key_ptr.*, 0, std.mem.sliceAsBytes(rows.items));
            try c.one(device, order, keyOf(entry.key_ptr.*), d.dtype, d.shape, rows.items);
        };
        if ((if (use_tensor_units) c.adapters.count() else c.tiled.count()) != linear_count) return error.DraftTensorCoverage;
        return c;
    }

    pub fn addBf16(c: *Copies, device: mtl.Device, order: ?core.lane_projection.RegOrder, key: usize, n: usize, k: usize, rows: []const u16) !void {
        var shape: [core.safetensors.max_rank]usize = @splat(1);
        shape[0] = n;
        shape[1] = k;
        return c.one(device, order, key, .bf16, shape, rows);
    }

    fn one(c: *Copies, device: mtl.Device, order: ?core.lane_projection.RegOrder, key: usize, dtype: ck.DType, shape: [core.safetensors.max_rank]usize, rows: []const u16) !void {
        if (dtype != .bf16 or shape[0] % 32 != 0 or shape[1] % 64 != 0) return error.DraftPreparationUnqualified;
        var words = try core.affine4.prepare(c.gpa, rows, shape[0], shape[1]);
        defer words.deinit();
        if (order) |o| {
            const adapter = try core.affine4_lane.Adapter.init(c.gpa, device, words, try split(shape[0], shape[1]), o);
            errdefer adapter.deinit();
            if (try c.adapters.fetchPut(c.gpa, key, adapter)) |old| old.value.deinit();
        } else {
            const t = try Tiled.init(device, words);
            errdefer t.deinit();
            if (try c.tiled.fetchPut(c.gpa, key, t)) |old| old.value.deinit();
        }
    }

    /// Projects contiguous BF16 `rows` from `input` through `key`'s 4-bit copy into `output`.
    pub fn project(c: *const Copies, e: mtl.ComputeEncoder, key: usize, n: usize, k: usize, input: Ref, output: Ref, rows: usize) !void {
        if (rows == 0 or rows > std.math.maxInt(u32)) return error.InvalidRows;
        if (c.tiled.get(key)) |t| {
            if (t.n != n or t.k != k) return error.DraftPreparationUnqualified;
            simd.encode(e, try simd.call(&c.simd.?, t.n, t.k, rows, false, false, 0), t.matrix(), input.buf, input.off, output.buf, output.off);
        } else {
            const adapter = c.adapters.get(key) orelse return error.DraftPreparationUnqualified;
            if (adapter.projection.n != n or adapter.projection.k != k) return error.DraftPreparationUnqualified;
            const most = adapter.projection.max_rows;
            const total: u32 = @intCast(rows);
            var first: u32 = 0;
            while (first < total) : (first += most) {
                try adapter.encode(
                    e,
                    .{ .buf = input.buf, .off = input.off + @as(usize, first) * k * 2 },
                    .{ .buf = output.buf, .off = output.off + @as(usize, first) * n * 2 },
                    @min(total - first, most),
                );
            }
        }
        e.barrier();
    }
};

fn writeFixture(a: std.mem.Allocator, tmp: *std.testing.TmpDir, entries: []const struct { name: []const u8, shape: []const usize, data: []const u16 }) ![:0]u8 {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    try header.append(a, '{');
    var offset: usize = 0;
    for (entries, 0..) |entry, i| {
        if (i > 0) try header.append(a, ',');
        const byte_len = entry.data.len * 2;
        try header.print(a, "\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[", .{entry.name});
        for (entry.shape, 0..) |dim, j| try header.print(a, "{s}{d}", .{ if (j > 0) "," else "", dim });
        try header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + byte_len });
        offset += byte_len;
    }
    try header.append(a, '}');
    var file = try tmp.dir.createFile(std.testing.io, "model.safetensors", .{});
    defer file.close(std.testing.io);
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.items.len, .little);
    var fw = file.writer(std.testing.io, &.{});
    try fw.interface.writeAll(&len_buf);
    try fw.interface.writeAll(header.items);
    for (entries) |entry| try fw.interface.writeAll(std.mem.sliceAsBytes(entry.data));
    try fw.interface.flush();
    return std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/model.safetensors", .{tmp.sub_path}, 0);
}

fn fillRandomBf16(rng: std.Random, buf: []u16, scale: f32) void {
    for (buf) |*v| v.* = core.affine4.asBf16((rng.float(f32) * 2.0 - 1.0) * scale);
}

fn checkProjection(device: mtl.Device, queue: mtl.Queue, copies: *const Copies, key: usize, w_bf16: []const u16, n: usize, k: usize, rows: usize, x_off_rows: usize) !void {
    const gpa = std.testing.allocator;
    const total_x_rows = rows + x_off_rows;
    const x_buf = try device.buffer(total_x_rows * k * 2, opts);
    defer x_buf.deinit();
    const y_buf = try device.buffer(rows * n * 2, opts);
    defer y_buf.deinit();
    var prng = std.Random.DefaultPrng.init(0xdf2_0001 +% n +% rows);
    const x_slice = x_buf.slice(u16, total_x_rows * k);
    fillRandomBf16(prng.random(), x_slice, 0.5);

    var q4 = try core.affine4.prepare(gpa, w_bf16, n, k);
    defer q4.deinit();

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try copies.project(e, key, n, k, .{ .buf = x_buf, .off = x_off_rows * k * 2 }, .{ .buf = y_buf, .off = 0 }, rows);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;

    const got = y_buf.slice(u16, rows * n);
    var max_q4_err: f64 = 0;
    var sq_err: f64 = 0;
    var sq_ref: f64 = 0;
    const groups = k / 64;
    for (0..rows) |r| {
        const xr = x_slice[(x_off_rows + r) * k ..][0..k];
        for (0..n) |col| {
            var ref_bf16: f32 = 0;
            var ref_q4: f32 = 0;
            for (0..groups) |g| {
                const sc = core.affine4.asFloat(q4.scales[col * groups + g]);
                const bi = core.affine4.asFloat(q4.biases[col * groups + g]);
                for (0..64) |i| {
                    const idx = g * 64 + i;
                    const xv = core.affine4.asFloat(xr[idx]);
                    const wv = core.affine4.asFloat(w_bf16[col * k + idx]);
                    ref_bf16 += xv * wv;
                    const code = (q4.words[(col * groups + g) * 8 + i / 8] >> @intCast(4 * (i % 8))) & 0xf;
                    ref_q4 += xv * (sc * @as(f32, @floatFromInt(code)) + bi);
                }
            }
            const got_v = core.affine4.asFloat(got[r * n + col]);
            max_q4_err = @max(max_q4_err, @abs(@as(f64, got_v) - @as(f64, ref_q4)));
            const diff = @as(f64, got_v) - @as(f64, ref_bf16);
            sq_err += diff * diff;
            sq_ref += @as(f64, ref_bf16) * @as(f64, ref_bf16);
        }
    }
    const rel_l2 = @sqrt(sq_err / @max(sq_ref, 1e-12));
    try std.testing.expect(max_q4_err < 0.08);
    try std.testing.expect(rel_l2 < 0.12);
}

test "DFlash2 linear predicate matches all 47 rank-2 weights and excludes norms, base kernels and codebooks" {
    const a = std.testing.allocator;
    var matched: usize = 0;
    var excluded: usize = 0;
    const globals = [_]struct { name: []const u8, rank: usize, want: bool }{
        .{ .name = "fc.weight", .rank = 2, .want = true },
        .{ .name = "hidden_norm.weight", .rank = 1, .want = false },
        .{ .name = "norm.weight", .rank = 1, .want = false },
        .{ .name = "candidate_selector.hidden_projection.weight", .rank = 2, .want = true },
        .{ .name = "candidate_selector.predecessor_codebook", .rank = 2, .want = false },
        .{ .name = "candidate_selector.successor_codebook", .rank = 2, .want = false },
    };
    for (globals) |g| {
        try std.testing.expectEqual(g.want, linear(g.name, g.rank));
        if (g.want) matched += 1 else excluded += 1;
    }
    const per_layer = [_]struct { suffix: []const u8, rank: usize, want: bool }{
        .{ .suffix = "input_layernorm.weight", .rank = 1, .want = false },
        .{ .suffix = "post_attention_layernorm.weight", .rank = 1, .want = false },
        .{ .suffix = "self_attn.q_norm.weight", .rank = 1, .want = false },
        .{ .suffix = "self_attn.k_norm.weight", .rank = 1, .want = false },
        .{ .suffix = "self_attn.q_proj.weight", .rank = 2, .want = true },
        .{ .suffix = "self_attn.k_proj.weight", .rank = 2, .want = true },
        .{ .suffix = "self_attn.v_proj.weight", .rank = 2, .want = true },
        .{ .suffix = "self_attn.o_proj.weight", .rank = 2, .want = true },
        .{ .suffix = "mlp.gate_proj.weight", .rank = 2, .want = true },
        .{ .suffix = "mlp.up_proj.weight", .rank = 2, .want = true },
        .{ .suffix = "mlp.down_proj.weight", .rank = 2, .want = true },
        .{ .suffix = "attention_conv.base_kernel", .rank = 3, .want = false },
        .{ .suffix = "attention_conv.kernel_projection.weight", .rank = 2, .want = true },
        .{ .suffix = "mlp_conv.base_kernel", .rank = 3, .want = false },
        .{ .suffix = "mlp_conv.kernel_projection.weight", .rank = 2, .want = true },
    };
    var buf: [128]u8 = undefined;
    for (0..5) |i| for (per_layer) |item| {
        const name = try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, item.suffix });
        try std.testing.expectEqual(item.want, linear(name, item.rank));
        if (item.want) matched += 1 else excluded += 1;
    };
    try std.testing.expectEqual(@as(usize, 47), matched);
    try std.testing.expectEqual(@as(usize, 34), excluded);
    for ([_][2]usize{ .{ 2304, 9216 }, .{ 256, 2304 }, .{ 576, 2304 }, .{ 1024, 2304 }, .{ 4096, 2304 }, .{ 2304, 4096 }, .{ 6912, 2304 }, .{ 2304, 6912 } }) |nk| {
        try std.testing.expect((try split(nk[0], nk[1])) >= 4);
    }
    _ = a;
}

test "DFlash2 Q4 prepared copies project within tolerance on tensor-unit and SIMD-tile paths" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const w_fc = try gpa.alloc(u16, 2304 * 9216);
    defer gpa.free(w_fc);
    const w_conv = try gpa.alloc(u16, 576 * 2304);
    defer gpa.free(w_conv);
    const w_hproj = try gpa.alloc(u16, 256 * 2304);
    defer gpa.free(w_hproj);
    const norm_w = try gpa.alloc(u16, 2304);
    defer gpa.free(norm_w);
    const base_w = try gpa.alloc(u16, 2 * 2 * 2304);
    defer gpa.free(base_w);
    const cb_w = try gpa.alloc(u16, 64 * 256);
    defer gpa.free(cb_w);
    var prng = std.Random.DefaultPrng.init(0xdf2_4096);
    for ([_][]u16{ w_fc, w_conv, w_hproj, norm_w, base_w, cb_w }) |buf| fillRandomBf16(prng.random(), buf, 0.25);

    const path = try writeFixture(gpa, &tmp, &.{
        .{ .name = "fc.weight", .shape = &.{ 2304, 9216 }, .data = w_fc },
        .{ .name = "layers.0.attention_conv.kernel_projection.weight", .shape = &.{ 576, 2304 }, .data = w_conv },
        .{ .name = "candidate_selector.hidden_projection.weight", .shape = &.{ 256, 2304 }, .data = w_hproj },
        .{ .name = "norm.weight", .shape = &.{2304}, .data = norm_w },
        .{ .name = "layers.0.attention_conv.base_kernel", .shape = &.{ 2, 2, 2304 }, .data = base_w },
        .{ .name = "candidate_selector.predecessor_codebook", .shape = &.{ 64, 256 }, .data = cb_w },
    });
    defer gpa.free(path);

    const modes: [2]bool = if (device.tensorUnits()) .{ true, false } else .{ false, false };
    const n_modes: usize = if (device.tensorUnits()) 2 else 1;
    for (modes[0..n_modes]) |use_tu| {
        var split_ck = ck.Checkpoint.init(gpa);
        defer split_ck.deinit();
        try split_ck.addFileSplit(device, path, "", null, linear);
        const kept_only = (norm_w.len + base_w.len + cb_w.len) * 2;
        try std.testing.expect(split_ck.residentBytes() < kept_only + 1024);
        var bad = Copies.prepareFor(gpa, device, &split_ck, 2, use_tu);
        try std.testing.expectError(error.DraftTensorCoverage, bad);
        _ = &bad;

        var copies = try Copies.prepareFor(gpa, device, &split_ck, 3, use_tu);
        defer copies.deinit();
        split_ck.closeFiles();

        const linear_bf16 = (w_fc.len + w_conv.len + w_hproj.len) * 2;
        try std.testing.expect(copies.bytes() * 3 < linear_bf16);

        const k_fc = keyOf("fc.weight");
        const k_conv = keyOf("layers.0.attention_conv.kernel_projection.weight");
        const k_hproj = keyOf("candidate_selector.hidden_projection.weight");
        try checkProjection(device, queue, &copies, k_fc, w_fc, 2304, 9216, 256, 0);
        try checkProjection(device, queue, &copies, k_conv, w_conv, 576, 2304, 8, 0);
        try checkProjection(device, queue, &copies, k_hproj, w_hproj, 256, 2304, 7, 1);

        const empty_x = try device.buffer(2304 * 2, opts);
        defer empty_x.deinit();
        const empty_y = try device.buffer(256 * 2, opts);
        defer empty_y.deinit();
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        try std.testing.expectError(error.InvalidRows, copies.project(enc, k_hproj, 256, 2304, .{ .buf = empty_x }, .{ .buf = empty_y }, 0));
        try std.testing.expectError(error.DraftPreparationUnqualified, copies.project(enc, 0xdeadbeef, 256, 2304, .{ .buf = empty_x }, .{ .buf = empty_y }, 1));
        enc.end();
    }
}
