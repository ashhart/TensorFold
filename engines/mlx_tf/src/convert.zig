//! Bare-metal parallel on-disk 4-bit affine group-64 quantizer (`mlx_tf convert`).
//!
//! Replaces `mlx_lm.convert` with ZERO Apple MLX dependency:
//!   - Reads raw HuggingFace BF16 / FP16 / F32 safetensors checkpoints
//!   - Quantizes 2D/3D linear & embedding matrices in parallel across all CPU cores via GCD
//!   - Sanitizes raw HF Qwen3.5/3.8 checkpoints (strips vision/mtp, normalizes conv1d [C,1,4]->[C,4,1], shifts zero-centered RMSNorms +1.0)
//!   - Writes standard MLX 4-bit group-64 `.safetensors` (sharded if > 4 GiB) + updated `config.json` + tokenizer files
//!   - Output checkpoints load at native speed in TensorFold `mlx_tf` (and remain 100% compatible with MLX-format readers)
const std = @import("std");
const st = @import("safetensors.zig");
const model_mod = @import("model.zig");

const nowMs = model_mod.nowMs;

extern "c" fn dispatch_apply_f(
    iterations: usize,
    queue: ?*anyopaque,
    context: ?*anyopaque,
    work: *const fn (?*anyopaque, usize) callconv(.c) void,
) void;
extern "c" fn dispatch_get_global_queue(identifier: isize, flags: usize) ?*anyopaque;

extern "c" fn mkdir(path: [*:0]const u8, mode: u16) c_int;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, count: usize) isize;
extern "c" fn close(fd: c_int) c_int;

const O_WRONLY: c_int = 0x0001;
const O_CREAT: c_int = 0x0200;
const O_TRUNC: c_int = 0x0400;

pub const ConvertOptions = struct {
    bits: usize = 4,
    group_size: usize = 64,
};

pub const ConvertStats = struct {
    tensors_in: usize = 0,
    tensors_quantized: usize = 0,
    tensors_out: usize = 0,
    bytes_in: usize = 0,
    bytes_out: usize = 0,
    elapsed_ms: f64 = 0,
};

const OutTensor = struct {
    name: []const u8,
    dtype_str: []const u8,
    shape: [6]usize = @splat(0),
    rank: usize = 0,
    data: []u8,
    owned: bool = true,
};

inline fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

inline fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    const lsb = (u >> 16) & 1;
    return @intCast((u +% 0x7FFF +% lsb) >> 16);
}

inline fn f16ToF32(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

fn readElemF32(t: st.Tensor, idx: usize) f32 {
    return switch (t.dtype) {
        .bf16 => bf16ToF32(t.asSlice(u16)[idx]),
        .f16 => f16ToF32(t.asSlice(u16)[idx]),
        .f32 => t.asSlice(f32)[idx],
        else => 0.0,
    };
}

const RowQuantJob = struct {
    t: st.Tensor,
    cols: usize,
    groups: usize,
    gs: usize,
    bits: usize,
    words_per_row: usize,
    dst_words: [*]u32,
    dst_scales: [*]u16,
    dst_biases: [*]u16,
};

fn quantRowWorker(raw_ctx: ?*anyopaque, row_idx: usize) callconv(.c) void {
    const job: *const RowQuantJob = @ptrCast(@alignCast(raw_ctx.?));
    const gs = job.gs;
    const groups = job.groups;
    const cols = job.cols;
    const bits = job.bits;
    const max_q: f32 = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
    const chunks_per_g = gs / 32;
    const base_in = row_idx * cols;
    const dst_w = job.dst_words + row_idx * job.words_per_row;
    @memset(dst_w[0..job.words_per_row], 0);
    const dst_s = job.dst_scales + row_idx * groups;
    const dst_b = job.dst_biases + row_idx * groups;

    for (0..groups) |g| {
        const g_in = base_in + g * gs;
        var vmin: f32 = std.math.inf(f32);
        var vmax: f32 = -std.math.inf(f32);
        for (0..gs) |i| {
            const v = readElemF32(job.t, g_in + i);
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        const span = vmax - vmin;
        const scale: f32 = if (span > 1e-12) span / max_q else 0.0;
        const s_bf = f32ToBf16(scale);
        const b_bf = f32ToBf16(vmin);
        const s_f = bf16ToF32(s_bf);
        const b_f = bf16ToF32(b_bf);
        const inv_s: f32 = if (s_f > 0.0) 1.0 / s_f else 0.0;

        for (0..chunks_per_g) |ci| {
            const c_idx = g * chunks_per_g + ci;
            const w_chunk = dst_w + c_idx * bits;
            for (0..32) |lane| {
                const v = readElemF32(job.t, g_in + ci * 32 + lane);
                const q: u32 = @intFromFloat(@round(std.math.clamp((v - b_f) * inv_s, 0.0, max_q)));
                const bit_off = lane * bits;
                const wi = bit_off >> 5;
                const sh: u5 = @intCast(bit_off & 31);
                w_chunk[wi] |= q << sh;
                if (@as(usize, sh) + bits > 32) {
                    const rsh: u5 = @intCast(32 - @as(usize, sh));
                    w_chunk[wi + 1] |= q >> rsh;
                }
            }
        }
        dst_s[g] = s_bf;
        dst_b[g] = b_bf;
    }
}

fn shouldQuantize(name: []const u8, t: st.Tensor, gs: usize) bool {
    if (t.dtype != .bf16 and t.dtype != .f16 and t.dtype != .f32) return false;
    if (t.rank < 2) return false;
    const last_dim = t.shape[t.rank - 1];
    if (last_dim == 0 or last_dim % gs != 0) return false;
    if (!std.mem.endsWith(u8, name, ".weight")) return false;
    if (std.mem.indexOf(u8, name, "conv1d") != null) return false;
    if (std.mem.indexOf(u8, name, "norm") != null) return false;
    if (std.mem.endsWith(u8, name, ".gate.weight")) return false; // Keep MoE routers in BF16
    return true;
}

fn isZeroCenteredQwenNorm(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".input_layernorm.weight") or
        std.mem.endsWith(u8, name, ".post_attention_layernorm.weight") or
        std.mem.endsWith(u8, name, "model.norm.weight") or
        std.mem.endsWith(u8, name, ".q_norm.weight") or
        std.mem.endsWith(u8, name, ".k_norm.weight");
}

fn dtypeName(dt: st.DType) []const u8 {
    return switch (dt) {
        .u32 => "U32",
        .bf16 => "BF16",
        .f16 => "F16",
        .f32 => "F32",
        .other => "U8",
    };
}

fn writeAllFd(fd: c_int, bytes: []const u8) !void {
    var off: usize = 0;
    while (off < bytes.len) {
        const chunk = @min(bytes.len - off, 64 * 1024 * 1024);
        const n = write(fd, bytes.ptr + off, chunk);
        if (n <= 0) return error.WriteFailed;
        off += @intCast(n);
    }
}

fn writeFileBytes(a: std.mem.Allocator, path: []const u8, bytes: []const u8) !void {
    const zpath = try a.dupeSentinel(u8, path, 0);
    defer a.free(zpath);
    const fd = open(zpath.ptr, O_WRONLY | O_CREAT | O_TRUNC, @as(c_int, 0o644));
    if (fd < 0) return error.FileCreateFailed;
    defer _ = close(fd);
    try writeAllFd(fd, bytes);
}

fn copyFileIfExists(a: std.mem.Allocator, src_dir: []const u8, dst_dir: []const u8, fname: []const u8) !void {
    const src_p = try std.fs.path.join(a, &.{ src_dir, fname });
    defer a.free(src_p);
    const bytes = st.mapFile(a, src_p) catch return;
    defer st.unmapFile(bytes);
    const dst_p = try std.fs.path.join(a, &.{ dst_dir, fname });
    defer a.free(dst_p);
    try writeFileBytes(a, dst_p, bytes);
}

fn makeDirP(a: std.mem.Allocator, path: []const u8) !void {
    var cur: std.ArrayList(u8) = .empty;
    defer cur.deinit(a);
    for (path, 0..) |ch, i| {
        try cur.append(a, ch);
        if (ch == '/' and i > 0) {
            try cur.append(a, 0);
            _ = mkdir(@ptrCast(cur.items.ptr), 0o755);
            cur.items.len -= 1;
        }
    }
    try cur.append(a, 0);
    _ = mkdir(@ptrCast(cur.items.ptr), 0o755);
}

pub fn convertModel(a: std.mem.Allocator, src_dir: []const u8, dst_dir: []const u8, opts: ConvertOptions) !ConvertStats {
    const valid_bits = (opts.bits == 2 or opts.bits == 3 or opts.bits == 4 or opts.bits == 5 or opts.bits == 6 or opts.bits == 8);
    const valid_gs = (opts.group_size == 32 or opts.group_size == 64 or opts.group_size == 128);
    if (!valid_bits or !valid_gs) return error.UnsupportedQuantConfig;
    const t0 = nowMs();
    try makeDirP(a, dst_dir);

    var ckpt = try st.Checkpoint.openDir(a, src_dir);
    defer ckpt.close();

    // Detect raw HF Qwen3.5/3.8 format (unsanitized conv1d [C, 1, 4] & zero-centered RMSNorms)
    var shift_qwen_norms = false;
    {
        var it = ckpt.tensors.iterator();
        while (it.next()) |e| {
            const k = e.key_ptr.*;
            const t = e.value_ptr.*;
            if (std.mem.indexOf(u8, k, "conv1d.weight") != null and t.rank == 3 and t.shape[2] != 1) {
                shift_qwen_norms = true;
                break;
            }
        }
    }

    // Gather sorted tensor keys for deterministic output
    var keys: std.ArrayList([]const u8) = .empty;
    defer keys.deinit(a);
    {
        var it = ckpt.tensors.keyIterator();
        while (it.next()) |kp| try keys.append(a, kp.*);
    }
    std.mem.sort([]const u8, keys.items, {}, struct {
        fn lt(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.order(u8, lhs, rhs) == .lt;
        }
    }.lt);

    var out_tensors: std.ArrayList(OutTensor) = .empty;
    defer {
        for (out_tensors.items) |ot| {
            a.free(ot.name);
            if (ot.owned) a.free(ot.data);
        }
        out_tensors.deinit(a);
    }

    var stats = ConvertStats{};
    const q = dispatch_get_global_queue(0, 0);

    for (keys.items) |raw_key| {
        if (std.mem.startsWith(u8, raw_key, "model.visual.") or
            std.mem.startsWith(u8, raw_key, "vision_tower.") or
            std.mem.indexOf(u8, raw_key, "mtp.") != null)
        {
            continue;
        }
        const t = ckpt.tensors.get(raw_key).?;
        stats.tensors_in += 1;
        stats.bytes_in += t.bytes.len;

        // Normalize VLM text key prefix: "model.language_model." -> "language_model.model."
        const norm_key = if (std.mem.startsWith(u8, raw_key, "model.language_model."))
            try std.fmt.allocPrint(a, "language_model.model.{s}", .{raw_key["model.language_model.".len..]})
        else
            try a.dupe(u8, raw_key);

        if (shouldQuantize(norm_key, t, opts.group_size)) {
            defer a.free(norm_key);
            const base = norm_key[0 .. norm_key.len - ".weight".len];
            const cols = t.shape[t.rank - 1];
            const total_elems = t.elemCount();
            const rows = total_elems / cols;
            const groups = cols / opts.group_size;
            const words_per_row = (cols / 32) * opts.bits;

            const codes_buf = try a.alloc(u8, rows * words_per_row * 4);
            errdefer a.free(codes_buf);
            const scales_buf = try a.alloc(u8, rows * groups * 2);
            errdefer a.free(scales_buf);
            const biases_buf = try a.alloc(u8, rows * groups * 2);
            errdefer a.free(biases_buf);

            var job = RowQuantJob{
                .t = t,
                .cols = cols,
                .groups = groups,
                .gs = opts.group_size,
                .bits = opts.bits,
                .words_per_row = words_per_row,
                .dst_words = @ptrCast(@alignCast(codes_buf.ptr)),
                .dst_scales = @ptrCast(@alignCast(scales_buf.ptr)),
                .dst_biases = @ptrCast(@alignCast(biases_buf.ptr)),
            };
            dispatch_apply_f(rows, q, &job, quantRowWorker);

            var w_shape = t.shape;
            w_shape[t.rank - 1] = words_per_row; // U32 packed (cols/32)*bits words
            var sb_shape = t.shape;
            sb_shape[t.rank - 1] = groups;

            try out_tensors.append(a, .{
                .name = try std.fmt.allocPrint(a, "{s}.weight", .{base}),
                .dtype_str = "U32",
                .shape = w_shape,
                .rank = t.rank,
                .data = codes_buf,
                .owned = true,
            });
            try out_tensors.append(a, .{
                .name = try std.fmt.allocPrint(a, "{s}.scales", .{base}),
                .dtype_str = "BF16",
                .shape = sb_shape,
                .rank = t.rank,
                .data = scales_buf,
                .owned = true,
            });
            try out_tensors.append(a, .{
                .name = try std.fmt.allocPrint(a, "{s}.biases", .{base}),
                .dtype_str = "BF16",
                .shape = sb_shape,
                .rank = t.rank,
                .data = biases_buf,
                .owned = true,
            });
            stats.tensors_quantized += 1;
            stats.bytes_out += codes_buf.len + scales_buf.len + biases_buf.len;
        } else {
            // Preserve unquantized tensor (applying Qwen3.5/3.8 sanitization if raw HF)
            var shape = t.shape;
            if (std.mem.indexOf(u8, norm_key, "conv1d.weight") != null and t.rank == 3 and t.shape[2] != 1) {
                shape[1] = t.shape[2];
                shape[2] = t.shape[1]; // [C, 1, 4] -> [C, 4, 1] (identical contiguous memory!)
            }
            if (shift_qwen_norms and isZeroCenteredQwenNorm(norm_key) and t.rank == 1) {
                const n = t.elemCount();
                const buf = try a.alloc(u8, n * 2);
                const dst = @as([*]u16, @ptrCast(@alignCast(buf.ptr)))[0..n];
                for (0..n) |i| dst[i] = f32ToBf16(readElemF32(t, i) + 1.0);
                try out_tensors.append(a, .{
                    .name = norm_key,
                    .dtype_str = "BF16",
                    .shape = shape,
                    .rank = t.rank,
                    .data = buf,
                    .owned = true,
                });
                stats.bytes_out += buf.len;
            } else {
                const buf = try a.dupe(u8, t.bytes);
                try out_tensors.append(a, .{
                    .name = norm_key,
                    .dtype_str = dtypeName(t.dtype),
                    .shape = shape,
                    .rank = t.rank,
                    .data = buf,
                    .owned = true,
                });
                stats.bytes_out += buf.len;
            }
        }
    }
    stats.tensors_out = out_tensors.items.len;

    // Write safetensors (single model.safetensors when <= 4 GiB, else sharded + index.json)
    const max_shard_bytes: usize = 4 * 1024 * 1024 * 1024;
    if (stats.bytes_out <= max_shard_bytes) {
        const out_path = try std.fs.path.join(a, &.{ dst_dir, "model.safetensors" });
        defer a.free(out_path);
        try writeSafetensorsShard(a, out_path, out_tensors.items);
    } else {
        // Split into <= 4 GiB shards
        var shard_ranges: std.ArrayList(struct { s: usize, e: usize }) = .empty;
        defer shard_ranges.deinit(a);
        var s_idx: usize = 0;
        while (s_idx < out_tensors.items.len) {
            var cur_bytes: usize = 0;
            var e_idx = s_idx;
            while (e_idx < out_tensors.items.len) {
                const sz = out_tensors.items[e_idx].data.len;
                if (e_idx > s_idx and cur_bytes + sz > max_shard_bytes) break;
                cur_bytes += sz;
                e_idx += 1;
            }
            try shard_ranges.append(a, .{ .s = s_idx, .e = e_idx });
            s_idx = e_idx;
        }
        const n_shards = shard_ranges.items.len;
        var idx_json: std.ArrayList(u8) = .empty;
        defer idx_json.deinit(a);
        try idx_json.appendSlice(a, "{\"metadata\":{\"format\":\"mlx_tf\"},\"weight_map\":{");
        var first_entry = true;
        for (shard_ranges.items, 0..) |rng, si| {
            const fname = try std.fmt.allocPrint(a, "model-{d:0>5}-of-{d:0>5}.safetensors", .{ si + 1, n_shards });
            defer a.free(fname);
            const fpath = try std.fs.path.join(a, &.{ dst_dir, fname });
            defer a.free(fpath);
            try writeSafetensorsShard(a, fpath, out_tensors.items[rng.s..rng.e]);
            for (out_tensors.items[rng.s..rng.e]) |ot| {
                if (!first_entry) try idx_json.append(a, ',');
                first_entry = false;
                const entry = try std.fmt.allocPrint(a, "\"{s}\":\"{s}\"", .{ ot.name, fname });
                defer a.free(entry);
                try idx_json.appendSlice(a, entry);
            }
        }
        try idx_json.appendSlice(a, "}}\n");
        const idx_path = try std.fs.path.join(a, &.{ dst_dir, "model.safetensors.index.json" });
        defer a.free(idx_path);
        try writeFileBytes(a, idx_path, idx_json.items);
    }

    // Write updated config.json injecting "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
    {
        const src_cfg_path = try std.fs.path.join(a, &.{ src_dir, "config.json" });
        defer a.free(src_cfg_path);
        const cfg_raw = try st.mapFile(a, src_cfg_path);
        defer st.unmapFile(cfg_raw);

        const dst_cfg_path = try std.fs.path.join(a, &.{ dst_dir, "config.json" });
        defer a.free(dst_cfg_path);

        if (std.mem.indexOf(u8, cfg_raw, "\"quantization\"") != null) {
            try writeFileBytes(a, dst_cfg_path, cfg_raw);
        } else if (std.mem.lastIndexOfScalar(u8, cfg_raw, '}')) |close_pos| {
            var new_cfg: std.ArrayList(u8) = .empty;
            defer new_cfg.deinit(a);
            try new_cfg.appendSlice(a, std.mem.trimEnd(u8, cfg_raw[0..close_pos], " \t\r\n"));
            const q_block = try std.fmt.allocPrint(a, ",\n  \"quantization\": {{\n    \"group_size\": {d},\n    \"bits\": {d},\n    \"mode\": \"affine\"\n  }}\n}}\n", .{
                opts.group_size, opts.bits,
            });
            defer a.free(q_block);
            try new_cfg.appendSlice(a, q_block);
            try writeFileBytes(a, dst_cfg_path, new_cfg.items);
        } else {
            try writeFileBytes(a, dst_cfg_path, cfg_raw);
        }
    }

    // Copy tokenizer & generation config files
    const aux_files = [_][]const u8{
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    };
    for (aux_files) |f| try copyFileIfExists(a, src_dir, dst_dir, f);

    stats.elapsed_ms = nowMs() - t0;
    return stats;
}

fn writeSafetensorsShard(a: std.mem.Allocator, path: []const u8, tensors: []const OutTensor) !void {
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(a);
    try hdr.appendSlice(a, "{\"__metadata__\":{\"format\":\"mlx_tf\"}");

    var offset: usize = 0;
    for (tensors) |ot| {
        const end_off = offset + ot.data.len;
        const pre = try std.fmt.allocPrint(a, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ ot.name, ot.dtype_str });
        defer a.free(pre);
        try hdr.appendSlice(a, pre);
        for (0..ot.rank) |di| {
            if (di > 0) try hdr.append(a, ',');
            const ds = try std.fmt.allocPrint(a, "{d}", .{ot.shape[di]});
            defer a.free(ds);
            try hdr.appendSlice(a, ds);
        }
        const post = try std.fmt.allocPrint(a, "],\"data_offsets\":[{d},{d}]}}", .{ offset, end_off });
        defer a.free(post);
        try hdr.appendSlice(a, post);
        offset = end_off;
    }
    try hdr.append(a, '}');
    // Pad header with spaces to 8-byte alignment for fast mmap alignment
    while (hdr.items.len % 8 != 0) try hdr.append(a, ' ');

    const zpath = try a.dupeSentinel(u8, path, 0);
    defer a.free(zpath);
    const fd = open(zpath.ptr, O_WRONLY | O_CREAT | O_TRUNC, @as(c_int, 0o644));
    if (fd < 0) return error.FileCreateFailed;
    defer _ = close(fd);

    var len_le: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_le, @intCast(hdr.items.len), .little);
    try writeAllFd(fd, &len_le);
    try writeAllFd(fd, hdr.items);
    for (tensors) |ot| {
        if (ot.data.len > 0) try writeAllFd(fd, ot.data);
    }
}
