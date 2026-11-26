//! Disk-layout loader, refusal controls, independent serial oracle and planted fault tests for Inkling 2 Small.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const st = core.safetensors;
const exl3 = core.exl3_format;
const exl3_proj = core.exl3_projection;
const cfg = @import("../config.zig");
const exl3_pack = @import("../exl3_pack.zig");
const names = @import("../names.zig");
const state = @import("small_state.zig");
const model_file = @import("small_model.zig");
const fwd = @import("small_forward.zig");
const oracle = @import("small_oracle_ref_test.zig");

const t = std.testing;
const TensorSpec = struct { name: []const u8, dtype: st.DType, dims: []const u64, begin: usize, end: usize };
const Marker = struct { off: usize, val: u32 };
const StreamSpec = struct { name: []const u8, dtype: st.DType, dims: []const u64, seed: u64, scale: f32 };
const RefusalFault = enum { none, mcg_codebook, bad_marker, width_1, width_5, width_half, size_mismatch };

fn dtypeStr(d: st.DType) []const u8 {
    return switch (d) {
        .bf16 => "BF16",
        .f16 => "F16",
        .f32 => "F32",
        .i16 => "I16",
        .i32 => "I32",
        else => unreachable,
    };
}

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

fn buildHeader(a: std.mem.Allocator, specs: []const TensorSpec) ![]u8 {
    var hdr: std.ArrayList(u8) = .empty;
    errdefer hdr.deinit(a);
    try hdr.append(a, '{');
    for (specs, 0..) |sp, i| {
        if (i > 0) try hdr.append(a, ',');
        try hdr.print(a, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ sp.name, dtypeStr(sp.dtype) });
        for (sp.dims, 0..) |d, j| {
            if (j > 0) try hdr.append(a, ',');
            try hdr.print(a, "{d}", .{d});
        }
        try hdr.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ sp.begin, sp.end });
    }
    try hdr.append(a, '}');
    return hdr.toOwnedSlice(a);
}

fn openSafetensorsWithHeader(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, sub: []const u8, specs: []const TensorSpec) !struct { file: std.Io.File, dstart: usize } {
    if (std.fs.path.dirname(sub)) |p| try dir.createDirPath(io, p);
    const hdr = try buildHeader(a, specs);
    defer a.free(hdr);
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, hdr.len, .little);
    const f = try dir.createFile(io, sub, .{});
    errdefer f.close(io);
    try f.writePositionalAll(io, &prefix, 0);
    try f.writePositionalAll(io, hdr, 8);
    return .{ .file = f, .dstart = 8 + hdr.len };
}

fn writeSparseHeaderFile(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, sub: []const u8, specs: []const TensorSpec, data_len: usize, markers: []const Marker) !u64 {
    const out = try openSafetensorsWithHeader(a, io, dir, sub, specs);
    defer out.file.close(io);
    if (data_len > 0) try out.file.writePositionalAll(io, &.{0}, out.dstart + data_len - 1);
    for (markers) |m| {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, m.val, .little);
        try out.file.writePositionalAll(io, &b, out.dstart + m.off);
    }
    return out.dstart + data_len;
}

fn writeSingleExpertShard(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, sub: []const u8, bits: u16, fault: RefusalFault) !u64 {
    const h: usize = state.hidden;
    const inter: usize = state.intermediate;
    const w_dim: usize = switch (fault) {
        .width_1 => 16,
        .width_5 => 80,
        .width_half => 40,
        else => @as(usize, bits) * 16,
    };
    const t13_b = (h / 16) * ((2 * inter) / 16) * w_dim * 2;
    const t2_b = (inter / 16) * (h / 16) * w_dim * 2;
    const m13_off = t13_b + h * 2 + 2 * inter * 2;
    const t2_off = std.mem.alignForward(usize, m13_off + 4, 16);
    const m2_off = t2_off + t2_b + inter * 2 + h * 2;
    const m13 = if (fault == .mcg_codebook) "experts.0.w13.mcg" else "experts.0.w13.mul1";
    const m2 = if (fault == .mcg_codebook) "experts.0.w2.mcg" else "experts.0.w2.mul1";
    const m_val: u32 = if (fault == .bad_marker) 0x12345678 else exl3.Mul1Mul;
    const specs = [_]TensorSpec{
        .{ .name = "experts.0.w13.trellis", .dtype = .i16, .dims = &.{ h / 16, (2 * inter) / 16, w_dim }, .begin = 0, .end = t13_b },
        .{ .name = "experts.0.w13.suh", .dtype = .f16, .dims = &.{h}, .begin = t13_b, .end = t13_b + h * 2 },
        .{ .name = "experts.0.w13.svh", .dtype = .f16, .dims = &.{2 * inter}, .begin = t13_b + h * 2, .end = m13_off },
        .{ .name = m13, .dtype = .i32, .dims = &.{}, .begin = m13_off, .end = m13_off + 4 },
        .{ .name = "experts.0.w2.trellis", .dtype = .i16, .dims = &.{ inter / 16, h / 16, w_dim }, .begin = t2_off, .end = t2_off + t2_b },
        .{ .name = "experts.0.w2.suh", .dtype = .f16, .dims = &.{inter}, .begin = t2_off + t2_b, .end = t2_off + t2_b + inter * 2 },
        .{ .name = "experts.0.w2.svh", .dtype = .f16, .dims = &.{h}, .begin = t2_off + t2_b + inter * 2, .end = m2_off },
        .{ .name = m2, .dtype = .i32, .dims = &.{}, .begin = m2_off, .end = m2_off + 4 },
    };
    return writeSparseHeaderFile(a, io, dir, sub, &specs, m2_off + 4, &.{ .{ .off = m13_off, .val = m_val }, .{ .off = m2_off, .val = m_val } });
}

fn smallConfigJson(a: std.mem.Allocator, layers: u32, vocab: u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const unpadded: u32 = if (vocab == 201024) 200058 else vocab;
    try out.print(a,
        \\{{"architectures":["Inkling2ForConditionalGeneration"],"model_type":"inkling2","eos_token_id":2,"image_token_id":1,
        \\"quantization_config":{{"quant_method":"exl3","format":"kortexa-inkling2-exl3-v1"}},
        \\"text_config":{{"model_max_length":1048576,"hidden_size":4608,"num_hidden_layers":{d},"vocab_size":{d},"unpadded_vocab_size":{d},
        \\"num_attention_heads":48,"num_key_value_heads":8,"head_dim":128,"swa_num_attention_heads":48,"swa_num_key_value_heads":8,"swa_head_dim":128,
        \\"sliding_window_size":1024,"d_rel":16,"rel_extent":1024,"log_scaling_n_floor":8192,"log_scaling_alpha":0.1,"rms_norm_eps":1e-6,
        \\"qk_l2_clip_threshold":22.627416997969522,"rel_l2_clip_threshold":8.0,"sconv_kernel_size":4,"n_layers_per_attn_res_block":6,
        \\"n_routed_experts":384,"num_experts_per_tok":8,"n_shared_experts":0,"intermediate_size":2304,
        \\"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"route_scale":8.0,"final_logit_softcapping":30.0,"use_gate_bias":true,"local_layer_ids":[
    , .{ layers, vocab, unpadded });
    var first = true;
    for (0..layers) |li| {
        if (li % 6 == 5) continue;
        if (!first) try out.append(a, ',');
        first = false;
        try out.print(a, "{d}", .{li});
    }
    try out.appendSlice(a, "]},\"vision_config\":{\"hidden_size\":32,\"num_hidden_layers\":1,\"num_attention_heads\":2,\"native_patch_size\":2,\"adapter_merge_size\":2,\"posemb_grid_size\":2,\"mlp_ratio\":2.0,\"projection_hidden_dim\":32,\"projector_bias\":false}}");
    return out.toOwnedSlice(a);
}

const SparseSink = struct {
    a: std.mem.Allocator,
    specs: std.ArrayList(TensorSpec) = .empty,
    cursor: usize = 0,

    pub fn put(s: *SparseSink, name: []const u8, dtype: st.DType, dims: []const usize) !void {
        var bytes: usize = dtype.size();
        const d64 = try s.a.alloc(u64, dims.len);
        for (dims, 0..) |d, i| {
            d64[i] = d;
            bytes *= d;
        }
        const begin = s.cursor;
        s.cursor += bytes;
        try s.specs.append(s.a, .{ .name = try s.a.dupe(u8, name), .dtype = dtype, .dims = d64, .begin = begin, .end = s.cursor });
    }
};

fn buildSparseFullSmallPack(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, fault: RefusalFault) !void {
    const cfg_txt = try smallConfigJson(a, 48, 201024);
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg_txt });
    var why: cfg.Why = .{};
    const parsed = try cfg.parse(a, cfg_txt, &why);
    var ds = SparseSink{ .a = a };
    try names.all(&parsed, &.{}, .dense_text, &ds);
    _ = try writeSparseHeaderFile(a, io, dir, "model-dense.safetensors", ds.specs.items, ds.cursor, &.{});

    const w_bad: usize = switch (fault) {
        .width_1 => 16,
        .width_5 => 80,
        .width_half => 40,
        else => 32,
    };
    const h: usize = state.hidden;
    const inter: usize = state.intermediate;
    const addShard = struct {
        fn run(alloc: std.mem.Allocator, io_: std.Io, dir_: std.Io.Dir, sub: []const u8, bad_e0: bool, f: RefusalFault, wb: usize) !u64 {
            var specs: std.ArrayList(TensorSpec) = .empty;
            var marks: std.ArrayList(Marker) = .empty;
            var cur: usize = 0;
            for (0..state.expert_count) |e| {
                const is_bad = bad_e0 and e == 0;
                const wd: usize = if (is_bad) wb else 32;
                const mk = if (is_bad and f == .mcg_codebook) "mcg" else "mul1";
                const mv: u32 = if (is_bad and f == .bad_marker) 0xdeadbeef else exl3.Mul1Mul;
                inline for (.{ .{ "w13", h, 2 * inter }, .{ "w2", inter, h } }) |p| {
                    const tb = (p[1] / 16) * (p[2] / 16) * wd * 2;
                    try specs.append(alloc, .{ .name = try std.fmt.allocPrint(alloc, "experts.{d}.{s}.trellis", .{ e, p[0] }), .dtype = .i16, .dims = try alloc.dupe(u64, &.{ p[1] / 16, p[2] / 16, wd }), .begin = cur, .end = cur + tb });
                    cur += tb;
                    try specs.append(alloc, .{ .name = try std.fmt.allocPrint(alloc, "experts.{d}.{s}.suh", .{ e, p[0] }), .dtype = .f16, .dims = try alloc.dupe(u64, &.{p[1]}), .begin = cur, .end = cur + p[1] * 2 });
                    cur += p[1] * 2;
                    try specs.append(alloc, .{ .name = try std.fmt.allocPrint(alloc, "experts.{d}.{s}.svh", .{ e, p[0] }), .dtype = .f16, .dims = try alloc.dupe(u64, &.{p[2]}), .begin = cur, .end = cur + p[2] * 2 });
                    cur += p[2] * 2;
                    try specs.append(alloc, .{ .name = try std.fmt.allocPrint(alloc, "experts.{d}.{s}.{s}", .{ e, p[0], mk }), .dtype = .i32, .dims = &.{}, .begin = cur, .end = cur + 4 });
                    try marks.append(alloc, .{ .off = cur, .val = mv });
                    cur += 4;
                }
            }
            return writeSparseHeaderFile(alloc, io_, dir_, sub, specs.items, cur, marks.items);
        }
    }.run;
    const sz_bad = try addShard(a, io, dir, "experts/layers/00/experts-000-383.safetensors", true, fault, w_bad);
    const sz_ok = try addShard(a, io, dir, "experts/layers/ok/experts-000-383.safetensors", false, .none, 32);

    var man: std.ArrayList(u8) = .empty;
    try man.appendSlice(a, "{\"format\":\"kortexa-inkling2-exl3-v1\",\"status\":\"complete\",\"layers\":{");
    for (0..48) |li| {
        if (li > 0) try man.append(a, ',');
        const rel = if (li == 0) "layers/00/experts-000-383.safetensors" else "layers/ok/experts-000-383.safetensors";
        const sz = if (li == 0) (if (fault == .size_mismatch) sz_bad + 1 else sz_bad) else sz_ok;
        try man.print(a, "\"{d}\":{{\"layer\":{d},\"status\":\"complete\",\"batches\":[{{\"start\":0,\"end\":384,\"file\":\"{s}\",\"bytes\":{d}}}]}}", .{ li, li, rel, sz });
    }
    try man.appendSlice(a, "}}");
    try dir.writeFile(io, .{ .sub_path = "experts/checkpoint.json", .data = man.items });
}

fn fillPattern(tile: []u8, dtype: st.DType, seed: u64, scale: f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    if (dtype == .bf16) {
        for (std.mem.bytesAsSlice(u16, tile)) |*v| v.* = toBf16((rng.float(f32) * 2.0 - 1.0) * scale);
    } else {
        for (std.mem.bytesAsSlice(f32, tile)) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * scale;
    }
}

fn writeStreamedDense(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, sub: []const u8, items: []const StreamSpec) !void {
    const specs = try a.alloc(TensorSpec, items.len);
    var cur: usize = 0;
    for (items, 0..) |it, i| {
        cur = std.mem.alignForward(usize, cur, 16);
        var nb: usize = it.dtype.size();
        for (it.dims) |d| nb *= @intCast(d);
        specs[i] = .{ .name = it.name, .dtype = it.dtype, .dims = it.dims, .begin = cur, .end = cur + nb };
        cur += nb;
    }
    const out = try openSafetensorsWithHeader(a, io, dir, sub, specs);
    defer out.file.close(io);
    const tile_bytes: usize = 65536;
    const chunk_bytes: usize = 4 << 20;
    const tile = try a.alloc(u8, tile_bytes);
    const chunk = try a.alloc(u8, chunk_bytes);
    for (items, specs) |it, sp| {
        const nb = sp.end - sp.begin;
        if (nb <= tile_bytes) {
            fillPattern(tile[0..nb], it.dtype, it.seed, it.scale);
            try out.file.writePositionalAll(io, tile[0..nb], out.dstart + sp.begin);
        } else {
            fillPattern(tile, it.dtype, it.seed, it.scale);
            var off: usize = 0;
            while (off < chunk_bytes) : (off += tile_bytes) @memcpy(chunk[off .. off + tile_bytes], tile);
            var written: usize = 0;
            while (written < nb) {
                const take = @min(nb - written, chunk_bytes);
                try out.file.writePositionalAll(io, chunk[0..take], out.dstart + sp.begin + written);
                written += take;
            }
        }
    }
}

fn expertBits(e: usize) usize {
    return if (e % 8 == 0) 4 else if (e % 8 == 1) 3 else 2;
}

fn writeSharedExpertShard(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, sub: []const u8) !u64 {
    const h: usize = state.hidden;
    const inter: usize = state.intermediate;
    var specs: std.ArrayList(TensorSpec) = .empty;
    var cur: usize = 0;
    for (0..state.expert_count) |e| {
        const wd = expertBits(e) * 16;
        inline for (.{ .{ "w13", h, 2 * inter }, .{ "w2", inter, h } }) |p| {
            cur = std.mem.alignForward(usize, cur, 16);
            const tb = (p[1] / 16) * (p[2] / 16) * wd * 2;
            try specs.append(a, .{ .name = try std.fmt.allocPrint(a, "experts.{d}.{s}.trellis", .{ e, p[0] }), .dtype = .i16, .dims = try a.dupe(u64, &.{ p[1] / 16, p[2] / 16, wd }), .begin = cur, .end = cur + tb });
            cur += tb;
            try specs.append(a, .{ .name = try std.fmt.allocPrint(a, "experts.{d}.{s}.suh", .{ e, p[0] }), .dtype = .f16, .dims = try a.dupe(u64, &.{p[1]}), .begin = cur, .end = cur + p[1] * 2 });
            cur += p[1] * 2;
            try specs.append(a, .{ .name = try std.fmt.allocPrint(a, "experts.{d}.{s}.svh", .{ e, p[0] }), .dtype = .f16, .dims = try a.dupe(u64, &.{p[2]}), .begin = cur, .end = cur + p[2] * 2 });
            cur += p[2] * 2;
            try specs.append(a, .{ .name = try std.fmt.allocPrint(a, "experts.{d}.{s}.mul1", .{ e, p[0] }), .dtype = .i32, .dims = &.{}, .begin = cur, .end = cur + 4 });
            cur += 4;
        }
    }
    const out = try openSafetensorsWithHeader(a, io, dir, sub, specs.items);
    defer out.file.close(io);
    var prng = std.Random.DefaultPrng.init(0x5105_E001);
    const rng = prng.random();
    const trellis_pool = try a.alloc(u8, 16 << 20);
    rng.bytes(trellis_pool);
    const scale_pool = try a.alloc(u8, 2 * inter * 2 + 4096);
    for (std.mem.bytesAsSlice(f16, scale_pool)) |*v| v.* = @floatCast((rng.float(f32) * 2.0 - 1.0) * 0.08);
    var mb: [4]u8 = undefined;
    std.mem.writeInt(u32, &mb, exl3.Mul1Mul, .little);

    var si: usize = 0;
    for (0..state.expert_count) |e| {
        inline for (.{ 0, 1 }) |pi| {
            const sp_t = specs.items[si];
            const sp_su = specs.items[si + 1];
            const sp_sv = specs.items[si + 2];
            const sp_m = specs.items[si + 3];
            si += 4;
            const shift = ((e * 4099 + pi * 1009) % 2048) * 16;
            const t_len = sp_t.end - sp_t.begin;
            try out.file.writePositionalAll(io, trellis_pool[shift .. shift + t_len], out.dstart + sp_t.begin);
            const sc_off = ((e * 17 + pi * 7) % 1024) * 2;
            try out.file.writePositionalAll(io, scale_pool[sc_off .. sc_off + (sp_su.end - sp_su.begin)], out.dstart + sp_su.begin);
            try out.file.writePositionalAll(io, scale_pool[sc_off .. sc_off + (sp_sv.end - sp_sv.begin)], out.dstart + sp_sv.begin);
            try out.file.writePositionalAll(io, &mb, out.dstart + sp_m.begin);
        }
    }
    return out.dstart + cur;
}

fn appendGlobalSpecs(a: std.mem.Allocator, dst: *std.ArrayList(StreamSpec), layers: u32, vocab: u32) !void {
    const h: u64 = state.hidden;
    const add = struct {
        fn f(al: std.mem.Allocator, list: *std.ArrayList(StreamSpec), name: []const u8, dt: st.DType, dims: []const u64, seed: u64, scale: f32) !void {
            try list.append(al, .{ .name = name, .dtype = dt, .dims = try al.dupe(u64, dims), .seed = seed, .scale = scale });
        }
    }.f;
    try add(a, dst, "model.llm.embed.weight", .bf16, &.{ vocab, h }, 1, 0.2);
    try add(a, dst, "model.llm.embed_norm.weight", .bf16, &.{h}, 2, 0.5);
    try add(a, dst, "model.llm.norm.weight", .bf16, &.{h}, 3, 0.5);
    try add(a, dst, "model.llm.unembed.weight", .bf16, &.{ vocab, h }, 4, 0.05);
    try add(a, dst, "model.llm.attn_res.output_query", .f32, &.{h}, 5, 0.1);
    for (0..layers - 1) |i| {
        const dims: []const u64 = if (cfg.isGlobal(@intCast(i + 1))) &.{ 3, h } else &.{h};
        try add(a, dst, try std.fmt.allocPrint(a, "model.llm.attn_res.attn_queries.{d}", .{i}), .f32, dims, 100 + i, 0.1);
    }
    for (0..layers) |i| try add(a, dst, try std.fmt.allocPrint(a, "model.llm.attn_res.mlp_queries.{d}", .{i}), .f32, &.{h}, 200 + i, 0.1);
}

fn appendLayerSpecs(a: std.mem.Allocator, dst: *std.ArrayList(StreamSpec), li: usize) !void {
    const h: u64 = state.hidden;
    const q: u64 = state.query_width;
    const kv: u64 = state.kv_width;
    const rel: u64 = state.rel_width;
    const global = cfg.isGlobal(@intCast(li));
    const base: u64 = 1000 + li * 64;
    const add = struct {
        fn f(al: std.mem.Allocator, list: *std.ArrayList(StreamSpec), name: []const u8, dt: st.DType, dims: []const u64, seed: u64, scale: f32) !void {
            try list.append(al, .{ .name = name, .dtype = dt, .dims = try al.dupe(u64, dims), .seed = seed, .scale = scale });
        }
    }.f;
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.wq_du.weight", .{li}), .bf16, &.{ q, h }, base + 1, 0.02);
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.wk_dv.weight", .{li}), .bf16, &.{ kv, h }, base + 2, 0.02);
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.wv_dv.weight", .{li}), .bf16, &.{ kv, h }, base + 3, 0.02);
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.wr_du.weight", .{li}), .bf16, &.{ rel, h }, base + 4, 0.02);
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.wo_ud.weight", .{li}), .bf16, &.{ h, q }, base + 5, 0.02);
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.gate_attn.weight", .{li}), .bf16, &.{ q, h }, base + 6, 0.02);
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.attn_sink_proj.weight", .{li}), .bf16, &.{ state.heads, h }, base + 7, 0.02);
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.q_l2_clip.scale", .{li}), .bf16, &.{ state.heads, 128 }, base + 8, 0.5);
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.rel_l2_clip.scale", .{li}), .bf16, &.{ state.heads, 16 }, base + 9, 0.5);
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.rel_logits_proj.proj", .{li}), .bf16, &.{ 16, 1024 }, base + 10, 0.05);
    if (!global) {
        try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.k_sconv.weight", .{li}), .bf16, &.{ kv, 1, 4 }, base + 11, 0.1);
        try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.v_sconv.weight", .{li}), .bf16, &.{ kv, 1, 4 }, base + 12, 0.1);
        try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn_norm.weight", .{li}), .bf16, &.{h}, base + 13, 0.5);
    } else {
        try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.k_sconv.static_w", .{li}), .bf16, &.{ 4, kv }, base + 14, 0.1);
        try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.k_sconv.weight_proj", .{li}), .bf16, &.{ 32, h }, base + 15, 0.02);
        try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.v_sconv.static_w", .{li}), .bf16, &.{ 4, kv }, base + 16, 0.1);
        try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn.v_sconv.weight_proj", .{li}), .bf16, &.{ 32, h }, base + 17, 0.02);
        try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.q_input_norm.weight", .{li}), .bf16, &.{h}, base + 18, 0.5);
        try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.k_input_norm.weight", .{li}), .bf16, &.{h}, base + 19, 0.5);
        try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.v_input_norm.weight", .{li}), .bf16, &.{h}, base + 20, 0.5);
    }
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.attn_sconv.weight", .{li}), .bf16, &.{ h, 1, 4 }, base + 21, 0.1);
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.mlp_norm.weight", .{li}), .bf16, &.{h}, base + 22, 0.5);
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.mlp_sconv.weight", .{li}), .bf16, &.{ h, 1, 4 }, base + 23, 0.1);
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.mlp.gate.weight", .{li}), .bf16, &.{ state.expert_count, h }, base + 24, 0.02);
    try add(a, dst, try std.fmt.allocPrint(a, "model.llm.layers.{d}.mlp.gate.bias", .{li}), .f32, &.{state.expert_count}, base + 25, 0.1);
}

fn writeSyntheticPackPair(a: std.mem.Allocator, io: std.Io, root_dir: std.Io.Dir) !void {
    const layers: u32 = 6;
    const vocab: u32 = 64;
    try root_dir.createDirPath(io, "single");
    try root_dir.createDirPath(io, "split/auxiliary");
    try root_dir.createDirPath(io, "split/layers/00");
    var single_dir = try root_dir.openDir(io, "single", .{});
    defer single_dir.close(io);
    var split_dir = try root_dir.openDir(io, "split", .{});
    defer split_dir.close(io);

    const cfg_txt = try smallConfigJson(a, layers, vocab);
    try single_dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg_txt });
    try split_dir.writeFile(io, .{ .sub_path = "auxiliary/config.json", .data = cfg_txt });

    var g_specs: std.ArrayList(StreamSpec) = .empty;
    try appendGlobalSpecs(a, &g_specs, layers, vocab);
    try writeStreamedDense(a, io, split_dir, "dense/global.safetensors", g_specs.items);

    var all_specs: std.ArrayList(StreamSpec) = .empty;
    try all_specs.appendSlice(a, g_specs.items);
    for (0..layers) |li| {
        var l_specs: std.ArrayList(StreamSpec) = .empty;
        try appendLayerSpecs(a, &l_specs, li);
        try writeStreamedDense(a, io, split_dir, try std.fmt.allocPrint(a, "dense/layer-{d:0>2}.safetensors", .{li}), l_specs.items);
        try all_specs.appendSlice(a, l_specs.items);
    }
    try writeStreamedDense(a, io, single_dir, "model-dense.safetensors", all_specs.items);

    const shard_rel = "layers/00/experts-000-383.safetensors";
    const shard_sz = try writeSharedExpertShard(a, io, single_dir, "experts/" ++ shard_rel);
    try root_dir.hardLink("single/experts/" ++ shard_rel, root_dir, "split/" ++ shard_rel, io, .{});

    var man: std.ArrayList(u8) = .empty;
    try man.appendSlice(a, "{\"format\":\"kortexa-inkling2-exl3-v1\",\"status\":\"complete\",\"layers\":{");
    for (0..layers) |li| {
        if (li > 0) try man.append(a, ',');
        try man.print(a, "\"{d}\":{{\"layer\":{d},\"status\":\"complete\",\"K_by_expert\":[", .{ li, li });
        for (0..state.expert_count) |e| {
            if (e > 0) try man.append(a, ',');
            try man.print(a, "{d}", .{expertBits(e)});
        }
        try man.print(a, "],\"batches\":[{{\"start\":0,\"end\":384,\"file\":\"{s}\",\"bytes\":{d}}}]}}", .{ shard_rel, shard_sz });
    }
    try man.appendSlice(a, "}}");
    try single_dir.writeFile(io, .{ .sub_path = "experts/checkpoint.json", .data = man.items });
    try split_dir.writeFile(io, .{ .sub_path = "checkpoint.json", .data = man.items });
}

fn verifyAgainstOracle(m: *model_file.Model, cap: *const oracle.Capture, prompt: []const u32, decode_steps: usize) !void {
    const vocab = cap.vocab;
    const total = prompt.len + decode_steps;
    const chunk = @max(prompt.len, 16);
    var s = try state.Scratch.initFor(m.device, chunk, vocab);
    defer s.deinit();
    var c = try state.Cache.initFor(m.device, total + 16, &m.weight_plan, chunk, vocab);
    defer c.deinit();
    const taps = try m.device.buffer(state.captureBytes(total), mtl.ResourceOptions.shared);
    defer taps.deinit();
    const tap_words = state.feature_taps * state.hidden;
    @memset(taps.slice(u16, total * tap_words), 0);

    try fwd.run(m, &s, &c, prompt, .{ .all = true, .taps = .{ .buf = taps, .off = 0 } });
    c.len += prompt.len;
    const prompt_logits = s.logits_all.slice(u32, prompt.len * vocab);
    for (0..prompt.len) |i| {
        try t.expectEqualSlices(u32, cap.logits[i * vocab ..][0..vocab], prompt_logits[i * vocab ..][0..vocab]);
    }
    for (0..decode_steps) |di| {
        const step = prompt.len + di;
        const prev = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(cap.logits[(step - 1) * vocab ..][0..vocab]));
        var best: usize = 0;
        for (prev[1..], 1..) |v, j| if (v > prev[best]) {
            best = j;
        };
        try t.expectEqual(cap.tokens[step], @as(u32, @intCast(best)));
        try fwd.run(m, &s, &c, &.{@intCast(best)}, .{ .all = true, .taps = .{ .buf = taps, .off = step * state.feature_row_bytes } });
        c.len += 1;
        try t.expectEqualSlices(u32, cap.logits[step * vocab ..][0..vocab], s.logits_all.slice(u32, vocab));
    }
    try t.expectEqualSlices(u16, cap.taps, taps.slice(u16, total * tap_words));
}

fn verifyPlantedCompositionFault(m: *model_file.Model, cap: *const oracle.Capture, prompt: []const u32) !void {
    const saved = m.weight_plan.visits[1].layer;
    m.weight_plan.visits[1].layer = m.weight_plan.visits[2].layer;
    m.weight_plan.visits[2].layer = saved;
    defer {
        m.weight_plan.visits[2].layer = m.weight_plan.visits[1].layer;
        m.weight_plan.visits[1].layer = saved;
    }
    const vocab = cap.vocab;
    const chunk = @max(prompt.len, 16);
    var s = try state.Scratch.initFor(m.device, chunk, vocab);
    defer s.deinit();
    var c_chunk = try state.Cache.initFor(m.device, prompt.len + 16, &m.weight_plan, chunk, vocab);
    defer c_chunk.deinit();
    const taps_c = try m.device.buffer(state.captureBytes(prompt.len), mtl.ResourceOptions.shared);
    defer taps_c.deinit();
    const tap_words = state.feature_taps * state.hidden;
    @memset(taps_c.slice(u16, prompt.len * tap_words), 0);
    try fwd.run(m, &s, &c_chunk, prompt, .{ .all = true, .taps = .{ .buf = taps_c, .off = 0 } });
    const got_logits = try t.allocator.dupe(u32, s.logits_all.slice(u32, prompt.len * vocab));
    defer t.allocator.free(got_logits);
    const got_taps = try t.allocator.dupe(u16, taps_c.slice(u16, prompt.len * tap_words));
    defer t.allocator.free(got_taps);

    // Self-comparison still passes on the faulted composition, proving it cannot catch a shared composition bug.
    var c_seq = try state.Cache.initFor(m.device, prompt.len + 16, &m.weight_plan, chunk, vocab);
    defer c_seq.deinit();
    const taps_s = try m.device.buffer(state.captureBytes(prompt.len), mtl.ResourceOptions.shared);
    defer taps_s.deinit();
    @memset(taps_s.slice(u16, prompt.len * tap_words), 0);
    for (prompt, 0..) |tok, i| {
        try fwd.run(m, &s, &c_seq, &.{tok}, .{ .all = true, .taps = .{ .buf = taps_s, .off = i * state.feature_row_bytes } });
        c_seq.len += 1;
        try t.expectEqualSlices(u32, got_logits[i * vocab ..][0..vocab], s.logits_all.slice(u32, vocab));
    }
    try t.expectEqualSlices(u16, got_taps, taps_s.slice(u16, prompt.len * tap_words));

    // The independent capture must reject the planted composition fault on both FP32 logits and BF16 taps.
    try t.expect(!std.mem.eql(u32, cap.logits[0 .. prompt.len * vocab], got_logits));
    try t.expect(!std.mem.eql(u16, cap.taps[0 .. prompt.len * tap_words], got_taps));
}

test "inkling2 small: Model.load and loadExpertShard refuse non-mul1 codebooks, widths outside 2-4 bits and shard size drift" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", t.allocator);
    defer t.allocator.free(root);

    var w13: [1]exl3_proj.ExpertDescriptor = undefined;
    var w2: [1]exl3_proj.ExpertDescriptor = undefined;
    for ([_]u16{ 2, 3, 4 }) |bits| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const rel = try std.fmt.allocPrint(arena.allocator(), "ok_{d}.safetensors", .{bits});
        const sz = try writeSingleExpertShard(arena.allocator(), io, tmp.dir, rel, bits, .none);
        const full = try std.fs.path.joinZ(arena.allocator(), &.{ root, rel });
        const buf = try model_file.Model.loadExpertShardForTest(t.allocator, device, full, .{ .start = 0, .end = 1, .file = rel, .bytes = sz }, &.{@intCast(bits)}, &w13, &w2);
        buf.deinit();
    }
    const cases = [_]struct { fault: RefusalFault, shard_err: anyerror, load_err: anyerror }{
        .{ .fault = .mcg_codebook, .shard_err = error.UnsupportedCodebook, .load_err = error.UnsupportedCodebook },
        .{ .fault = .bad_marker, .shard_err = error.UnsupportedCodebook, .load_err = error.UnsupportedCodebook },
        .{ .fault = .width_1, .shard_err = error.UnsupportedWidth, .load_err = error.UnsupportedWidth },
        .{ .fault = .width_5, .shard_err = error.UnsupportedWidth, .load_err = error.UnsupportedWidth },
        .{ .fault = .width_half, .shard_err = error.WrongTensor, .load_err = error.UnsupportedWidth },
        .{ .fault = .size_mismatch, .shard_err = error.ShardSizeMismatch, .load_err = error.ShardSizeMismatch },
    };
    for (cases, 0..) |c, ci| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const rel = try std.fmt.allocPrint(a, "bad_shard_{d}.safetensors", .{ci});
        const sz = try writeSingleExpertShard(a, io, tmp.dir, rel, 2, c.fault);
        const full = try std.fs.path.joinZ(a, &.{ root, rel });
        const decl_sz = if (c.fault == .size_mismatch) sz + 1 else sz;
        try t.expectError(c.shard_err, model_file.Model.loadExpertShardForTest(t.allocator, device, full, .{ .start = 0, .end = 1, .file = rel, .bytes = decl_sz }, null, &w13, &w2));
        const sub = try std.fmt.allocPrint(a, "pack_{d}", .{ci});
        try tmp.dir.createDirPath(io, sub);
        var sub_dir = try tmp.dir.openDir(io, sub, .{});
        defer sub_dir.close(io);
        try buildSparseFullSmallPack(a, io, sub_dir, c.fault);
        try t.expectError(c.load_err, model_file.Model.load(t.allocator, io, try std.fs.path.join(a, &.{ root, sub })));
    }
}

test "inkling2 small: disk loader admits single_dense and split_dense, matches independent serial oracle bit-for-bit, and rejects planted composition fault" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    device.deinit();
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", t.allocator);
    defer t.allocator.free(root);

    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try writeSyntheticPackPair(a, io, tmp.dir);

    const single_path = try std.fs.path.join(a, &.{ root, "single" });
    const split_path = try std.fs.path.join(a, &.{ root, "split" });
    const prompt = [_]u32{ 3, 17, 42, 7, 55, 12, 1, 29, 8, 60, 14, 33, 22, 5, 49, 19 };
    const decode_steps: usize = 3;

    var cap = try oracle.capture(t.allocator, io, single_path, &prompt, decode_steps);
    defer cap.deinit();

    const m_single = try model_file.Model.loadForTest(t.allocator, io, single_path);
    defer m_single.deinit();
    try verifyAgainstOracle(m_single, &cap, &prompt, decode_steps);
    try verifyPlantedCompositionFault(m_single, &cap, &prompt);

    const m_split = try model_file.Model.loadForTest(t.allocator, io, split_path);
    defer m_split.deinit();
    try verifyAgainstOracle(m_split, &cap, &prompt, decode_steps);
}

test "inkling2 small: real 48-layer pack via Model.load matches independent serial oracle bit-for-bit and rejects planted composition fault (TF_INKLING2_EXL3_DIR)" {
    const dir = std.mem.span(std.c.getenv("TF_INKLING2_EXL3_DIR") orelse return error.SkipZigTest);
    const device = mtl.Device.init() catch return error.SkipZigTest;
    device.deinit();
    const io = t.io;
    const prompt = [_]u32{ 1, 791, 6864, 315, 9822, 374, 279, 2115, 315, 12366, 13, 198, 791, 1401, 374, 220 };
    const decode_steps: usize = 4;

    var cap = try oracle.capture(t.allocator, io, dir, &prompt, decode_steps);
    defer cap.deinit();

    const m = try model_file.Model.load(t.allocator, io, dir);
    defer m.deinit();
    try verifyAgainstOracle(m, &cap, &prompt, decode_steps);
    try verifyPlantedCompositionFault(m, &cap, &prompt);

    if (std.c.getenv("TF_INKLING2_EXL3_ALT_DIR")) |alt_c| {
        const m_alt = try model_file.Model.load(t.allocator, io, std.mem.span(alt_c));
        defer m_alt.deinit();
        try verifyAgainstOracle(m_alt, &cap, &prompt, decode_steps);
    }
}
