//! Pinned metadata, SHA-256 digest verification, and pre-allocation header admission for Inkling 2 Mini affine4 packs.
const std = @import("std");
const st = @import("core").safetensors;
const mlx_affine = @import("core").quant.mlx_affine;
const cfg = @import("../config.zig");
const image = @import("../image.zig");
const layer_plan = @import("../plan.zig");

const vision_hidden: u32 = 1024;
const vision_layers: u32 = 23;
const vision_heads: u32 = 16;
const vision_mlp_ratio: u32 = 4;
const vision_proj_hidden: u32 = 4096;
const vision_text_hidden: u32 = 2304;

pub const PinnedFile = struct {
    name: []const u8,
    sha256: [64]u8,
};

pub const pinned_pack = struct {
    pub const target_repo = "mlx-community/Inkling-2-Mini-4bit";
    pub const target_revision = "9856dc53edead71405f8348930937c3638ff21a2";
    pub const source_repo = "thinkingmachines/Inkling-2-Mini";
    pub const source_revision = "ed40e7da625f594fc9c4990f5f66be24873c22d7";
    pub const converter = "mlx-lm PR ml-explore/mlx-lm#2086";
    pub const mode = "affine";
    pub const bits: u8 = 4;
    pub const group_size: u32 = 64;
    pub const config_sha256: [64]u8 = "0c72c706ad94ff52d4e4c8585e15bcc98eaa964db6143e2a62dc3d809acd8a71".*;
    pub const index_sha256: [64]u8 = "6e2d0fbb83f4ec2488f22f521e57855375ea93b473a704da10871398f952d9f5".*;
    pub const shards = [_]PinnedFile{
        .{ .name = "model-00001-of-00004.safetensors", .sha256 = "2e51f82d978af1ee86e899f6a20ab1e13ea16256d2f7c435cc200680d688d8ec".* },
        .{ .name = "model-00002-of-00004.safetensors", .sha256 = "34055bc7f80403eae61911ea47ede7fb3ccf5bce098472f940606bf9945a73a3".* },
        .{ .name = "model-00003-of-00004.safetensors", .sha256 = "99fff9b3a485daee7dbf309caf77b2805cab6d7e529efee2b9f090ec439c8eb7".* },
        .{ .name = "model-00004-of-00004.safetensors", .sha256 = "760cbed0fa51b69be20427da7cb1953043b664da717cc4413836468c8dbe3685".* },
    };
    pub const files = [_]PinnedFile{
        .{ .name = "config.json", .sha256 = config_sha256 },
        .{ .name = "model.safetensors.index.json", .sha256 = index_sha256 },
    } ++ shards;
};

pub const HeaderTensor = struct {
    dtype: st.DType,
    shape: [st.max_rank]usize = @splat(1),
    rank: usize = 0,
    bytes: usize,
};

pub const HeaderIndex = struct {
    arena: std.heap.ArenaAllocator,
    map: std.StringHashMapUnmanaged(HeaderTensor) = .empty,

    pub fn init(gpa: std.mem.Allocator) HeaderIndex {
        return .{ .arena = .init(gpa) };
    }

    pub fn deinit(self: *HeaderIndex) void {
        self.map.deinit(self.arena.allocator());
        self.arena.deinit();
    }

    pub fn put(self: *HeaderIndex, name: []const u8, t: HeaderTensor) !void {
        const aa = self.arena.allocator();
        const key = try aa.dupe(u8, name);
        const slot = try self.map.getOrPut(aa, key);
        if (slot.found_existing) return error.DuplicateTensor;
        slot.value_ptr.* = t;
    }

    pub fn get(self: *const HeaderIndex, name: []const u8) !HeaderTensor {
        return self.map.get(name) orelse error.MissingTensor;
    }

    pub fn has(self: *const HeaderIndex, name: []const u8) bool {
        return self.map.contains(name);
    }

    pub fn keyIterator(self: *const HeaderIndex) std.StringHashMapUnmanaged(HeaderTensor).KeyIterator {
        return self.map.keyIterator();
    }
};

/// Compute lowercase hex SHA-256 of one file by streaming in 256 KiB blocks.
pub fn fileSha256Hex(io: std.Io, path: []const u8) ![64]u8 {
    const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.MissingPinnedFile,
        else => return err,
    };
    defer f.close(io);
    const total = try f.length(io);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [256 * 1024]u8 = undefined;
    var off: u64 = 0;
    while (off < total) {
        const want: usize = @intCast(@min(@as(u64, buf.len), total - off));
        const got = try f.readPositionalAll(io, buf[0..want], off);
        if (got != want) return error.UnexpectedEof;
        hasher.update(buf[0..got]);
        off += got;
    }
    const digest = hasher.finalResult();
    return std.fmt.bytesToHex(digest, .lower);
}

/// Verify one file matches its expected 64-char lowercase SHA-256 hex digest.
pub fn verifyFileDigest(io: std.Io, path: []const u8, expected: [64]u8) !void {
    const got = try fileSha256Hex(io, path);
    if (!std.mem.eql(u8, &got, &expected)) return error.PinnedDigestMismatch;
}

/// Verify all pinned config, index, and 4 shard SHA-256 digests under `dir`.
pub fn verifyPinnedDigests(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    for (pinned_pack.files) |pf| {
        const path = try std.fs.path.join(gpa, &.{ dir, pf.name });
        defer gpa.free(path);
        try verifyFileDigest(io, path, pf.sha256);
    }
}

fn requireTensor(t: anytype, dtype: st.DType, dims: []const usize) !void {
    if (t.dtype != dtype or t.rank != dims.len) return error.UnexpectedInkling2Tensor;
    var count: usize = 1;
    for (dims, 0..) |d, i| {
        if (t.shape[i] != d) return error.UnexpectedInkling2Tensor;
        count = std.math.mul(usize, count, d) catch return error.UnexpectedInkling2Tensor;
    }
    const bytes = std.math.mul(usize, count, dtype.size()) catch return error.UnexpectedInkling2Tensor;
    if (t.bytes != bytes) return error.UnexpectedInkling2Tensor;
}

fn checkTensor(c: anytype, comptime fmt: []const u8, args: anytype, dtype: st.DType, dims: []const usize) !void {
    var buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, args);
    const t = try c.get(name);
    try requireTensor(t, dtype, dims);
}

fn refuseCompanions(c: anytype, module: []const u8) !void {
    var buf: [280]u8 = undefined;
    if (c.has(try std.fmt.bufPrint(&buf, "{s}.scales", .{module})) or
        c.has(try std.fmt.bufPrint(&buf, "{s}.biases", .{module})))
        return error.UnexpectedAffineCompanion;
}

fn checkPlain(c: anytype, comptime fmt: []const u8, args: anytype, dtype: st.DType, dims: []const usize) !void {
    var mbuf: [256]u8 = undefined;
    const mod = try std.fmt.bufPrint(&mbuf, fmt, args);
    try refuseCompanions(c, mod);
    var wbuf: [280]u8 = undefined;
    const t = try c.get(try std.fmt.bufPrint(&wbuf, "{s}.weight", .{mod}));
    try requireTensor(t, dtype, dims);
}

fn checkBf16OrF32(c: anytype, comptime fmt: []const u8, args: anytype, dims: []const usize) !void {
    var mbuf: [256]u8 = undefined;
    const mod = try std.fmt.bufPrint(&mbuf, fmt, args);
    try refuseCompanions(c, mod);
    var wbuf: [280]u8 = undefined;
    const t = try c.get(try std.fmt.bufPrint(&wbuf, "{s}.weight", .{mod}));
    if (t.dtype == .bf16) return requireTensor(t, .bf16, dims);
    try requireTensor(t, .f32, dims);
}

fn checkAffine4(c: anytype, comptime fmt: []const u8, args: anytype, rows: u32, cols: u32) !void {
    var mbuf: [256]u8 = undefined;
    const mod = try std.fmt.bufPrint(&mbuf, fmt, args);
    var buf: [280]u8 = undefined;
    const w = try c.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{mod}));
    const s = try c.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{mod}));
    const z = try c.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{mod}));
    _ = try mlx_affine.check(w, s, z, .{ .n = rows, .k = cols, .bits = pinned_pack.bits, .group = pinned_pack.group_size });
}

fn checkExpert(c: anytype, layer: u32, comptime role: []const u8, precision: layer_plan.Precision, rows: usize, k: usize) !void {
    const base = "model.language_model.layers.{d}.mlp.experts." ++ role;
    var bbuf: [256]u8 = undefined;
    const mod = try std.fmt.bufPrint(&bbuf, base, .{layer});
    var zbuf: [280]u8 = undefined;
    if (c.has(try std.fmt.bufPrint(&zbuf, "{s}.biases", .{mod}))) return error.UnexpectedAffineCompanion;
    const e: usize = 128;
    switch (precision) {
        .nvfp4 => {
            try checkTensor(c, base ++ ".weight", .{layer}, .u32, &.{ e, rows, k / 8 });
            try checkTensor(c, base ++ ".scales", .{layer}, .u8, &.{ e, rows, k / 16 });
            try checkTensor(c, base ++ ".global_scale", .{layer}, .f32, &.{e});
        },
        .mxfp8 => {
            if (c.has(try std.fmt.bufPrint(&zbuf, "{s}.global_scale", .{mod}))) return error.UnexpectedInkling2Tensor;
            try checkTensor(c, base ++ ".weight", .{layer}, .u32, &.{ e, rows, k / 4 });
            try checkTensor(c, base ++ ".scales", .{layer}, .u8, &.{ e, rows, k / 32 });
        },
        .exl3 => return error.UnsupportedInkling2ExpertFormat,
    }
}

fn checkConv(c: anytype, layer: u32, comptime role: []const u8, global: bool) !usize {
    if (!global) {
        try checkPlain(c, "model.language_model.layers.{d}.self_attn." ++ role ++ "_sconv.conv1d", .{layer}, .bf16, &.{ 1024, 4, 1 });
        return 0;
    }
    try checkTensor(c, "model.language_model.layers.{d}.self_attn." ++ role ++ "_sconv.static_weight", .{layer}, .bf16, &.{ 4, 1024 });
    var mbuf: [256]u8 = undefined;
    const mod = try std.fmt.bufPrint(&mbuf, "model.language_model.layers.{d}.self_attn." ++ role ++ "_sconv.weight_proj", .{layer});
    var sbuf: [280]u8 = undefined;
    const has_comp = c.has(try std.fmt.bufPrint(&sbuf, "{s}.scales", .{mod})) or
        c.has(try std.fmt.bufPrint(&sbuf, "{s}.biases", .{mod}));
    if (has_comp) {
        try checkAffine4(c, "{s}", .{mod}, 32, 2304);
        return 2;
    }
    try checkPlain(c, "{s}", .{mod}, .bf16, &.{ 32, 2304 });
    return 0;
}

fn mulVision(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch return error.UnsupportedVisionGeometry;
}

fn checkVisionGeometry(v: cfg.Vision) !void {
    if (v.hidden != vision_hidden or v.layers != vision_layers or v.heads != vision_heads or
        v.patch != image.patch or v.merge != image.merge or v.grid != image.grid or
        v.mlp_ratio != vision_mlp_ratio or v.proj_hidden != vision_proj_hidden or v.text_hidden != vision_text_hidden)
        return error.UnsupportedVisionGeometry;
}

fn checkVision(c: anytype, v: cfg.Vision) !void {
    try checkVisionGeometry(v);
    const h: usize = v.hidden;
    const ff = try mulVision(h, v.mlp_ratio);
    const patch_sq = try mulVision(v.patch, v.patch);
    const patch_width = try mulVision(3, patch_sq);
    const grid_positions = try mulVision(v.grid, v.grid);
    const merge_sq = try mulVision(v.merge, v.merge);
    const merged_width = try mulVision(h, merge_sq);
    try checkPlain(c, "model.vision_tower.embeddings.patch_embedding", .{}, .bf16, &.{ h, patch_width });
    try checkPlain(c, "model.vision_tower.embeddings.position_embedding", .{}, .bf16, &.{ grid_positions, h });
    try checkTensor(c, "model.vision_tower.embeddings.class_embedding", .{}, .bf16, &.{h});
    try checkTensor(c, "model.vision_tower.embeddings.cls_position_embedding", .{}, .bf16, &.{h});
    try checkPlain(c, "model.vision_tower.pre_layernorm", .{}, .bf16, &.{h});
    try checkTensor(c, "model.vision_tower.pre_layernorm.bias", .{}, .bf16, &.{h});
    try checkPlain(c, "model.vision_tower.merger.linear_1", .{}, .bf16, &.{ v.proj_hidden, merged_width });
    try checkPlain(c, "model.vision_tower.merger.linear_2", .{}, .bf16, &.{ v.text_hidden, v.proj_hidden });
    for (0..v.layers) |i| {
        try checkPlain(c, "model.vision_tower.layers.{d}.layernorm_before", .{i}, .bf16, &.{h});
        try checkTensor(c, "model.vision_tower.layers.{d}.layernorm_before.bias", .{i}, .bf16, &.{h});
        try checkPlain(c, "model.vision_tower.layers.{d}.layernorm_after", .{i}, .bf16, &.{h});
        try checkTensor(c, "model.vision_tower.layers.{d}.layernorm_after.bias", .{i}, .bf16, &.{h});
        try checkTensor(c, "model.vision_tower.layers.{d}.lambda_1", .{i}, .bf16, &.{h});
        try checkTensor(c, "model.vision_tower.layers.{d}.lambda_2", .{i}, .bf16, &.{h});
        inline for (.{ "q_proj", "k_proj", "v_proj", "out_proj" }) |proj| {
            try checkPlain(c, "model.vision_tower.layers.{d}.self_attn." ++ proj, .{i}, .bf16, &.{ h, h });
            try checkTensor(c, "model.vision_tower.layers.{d}.self_attn." ++ proj ++ ".bias", .{i}, .bf16, &.{h});
        }
        try checkPlain(c, "model.vision_tower.layers.{d}.mlp.fc1", .{i}, .bf16, &.{ ff, h });
        try checkTensor(c, "model.vision_tower.layers.{d}.mlp.fc1.bias", .{i}, .bf16, &.{ff});
        try checkPlain(c, "model.vision_tower.layers.{d}.mlp.fc2", .{i}, .bf16, &.{ h, ff });
        try checkTensor(c, "model.vision_tower.layers.{d}.mlp.fc2.bias", .{i}, .bf16, &.{h});
    }
}

/// Admit every required tensor's presence, dtype, shape, and companions before allocating any GPU buffer.
pub fn admitHeaders(c: anytype, config: *const cfg.Config, wp: *const layer_plan.Plan) !void {
    if (config.quant != .mlx_affine4) return error.UnsupportedInkling2Quant;
    if (wp.n != 24 or wp.n_visits != 48) return error.UnsupportedInkling2Loop;
    const lp = config.text.loop orelse return error.UnsupportedInkling2Loop;
    if (lp.passes != 3 or lp.core_start != 6 or lp.core_end != 18 or lp.gate_rank != 64)
        return error.UnsupportedInkling2Loop;

    const T = if (@typeInfo(@TypeOf(c)) == .pointer) @typeInfo(@TypeOf(c)).pointer.child else @TypeOf(c);
    var has_vision = false;
    var seen_companions: usize = 0;
    if (@hasDecl(T, "keyIterator")) {
        var it = c.keyIterator();
        while (it.next()) |key_ptr| {
            const name = key_ptr.*;
            if (std.mem.startsWith(u8, name, "model.llm.") or std.mem.startsWith(u8, name, "model.visual."))
                return error.MixedCheckpointLayout;
            if (std.mem.startsWith(u8, name, "model.vision_tower.")) has_vision = true;
            if (std.mem.endsWith(u8, name, ".scales") or std.mem.endsWith(u8, name, ".biases")) seen_companions += 1;
        }
        if (has_vision) try checkVisionGeometry(config.vision);
    }

    var expected_companions: usize = 0;

    try checkAffine4(c, "model.language_model.embed_tokens", .{}, 201024, 2304);
    try checkAffine4(c, "lm_head", .{}, 201024, 2304);
    expected_companions += 4;
    try checkPlain(c, "model.language_model.embed_norm", .{}, .bf16, &.{2304});
    try checkPlain(c, "model.language_model.norm", .{}, .bf16, &.{2304});
    try checkTensor(c, "model.language_model.final_input.query", .{}, .f32, &.{2304});
    try checkTensor(c, "model.language_model.message_input.query", .{}, .f32, &.{ 12, 4, 2304 });
    for (1..24) |layer| {
        const dims: []const usize = if (cfg.isGlobal(@intCast(layer))) &.{ 3, 2304 } else &.{2304};
        try checkTensor(c, "model.language_model.layers.{d}.attention_input.query", .{layer}, .f32, dims);
    }
    for (0..24) |layer| try checkTensor(c, "model.language_model.layers.{d}.mlp_input.query", .{layer}, .f32, &.{2304});

    for (0..24) |i| {
        const layer: u32 = @intCast(i);
        const p = wp.at(layer);
        const global = p.kind == .global;
        if (p.heads.q != 24 or p.heads.kv != 8 or p.heads.dim != 128 or p.extent != 1024)
            return error.UnsupportedInkling2Geometry;
        const q_width = std.math.mul(u32, p.heads.q, p.heads.dim) catch return error.UnsupportedInkling2Geometry;
        try checkAffine4(c, "model.language_model.layers.{d}.self_attn.q_proj", .{layer}, q_width, 2304);
        try checkAffine4(c, "model.language_model.layers.{d}.self_attn.k_proj", .{layer}, 1024, 2304);
        try checkAffine4(c, "model.language_model.layers.{d}.self_attn.v_proj", .{layer}, 1024, 2304);
        try checkAffine4(c, "model.language_model.layers.{d}.self_attn.r_proj", .{layer}, 384, 2304);
        try checkAffine4(c, "model.language_model.layers.{d}.self_attn.o_proj", .{layer}, 2304, q_width);
        try checkAffine4(c, "model.language_model.layers.{d}.self_attn.gate_proj", .{layer}, q_width, 2304);
        try checkAffine4(c, "model.language_model.layers.{d}.self_attn.sink_proj", .{layer}, 24, 2304);
        expected_companions += 14;
        try checkTensor(c, "model.language_model.layers.{d}.self_attn.q_l2_clip.scale", .{layer}, .bf16, &.{ 24, 128 });
        try checkTensor(c, "model.language_model.layers.{d}.self_attn.rel_l2_clip.scale", .{layer}, .bf16, &.{ 24, 16 });
        try checkTensor(c, "model.language_model.layers.{d}.self_attn.rel_logits_proj.proj", .{layer}, .bf16, &.{ 16, p.extent });
        expected_companions += try checkConv(c, layer, "k", global);
        expected_companions += try checkConv(c, layer, "v", global);
        if (global) {
            try checkPlain(c, "model.language_model.layers.{d}.q_input_layernorm", .{layer}, .bf16, &.{2304});
            try checkPlain(c, "model.language_model.layers.{d}.k_input_layernorm", .{layer}, .bf16, &.{2304});
            try checkPlain(c, "model.language_model.layers.{d}.v_input_layernorm", .{layer}, .bf16, &.{2304});
        } else {
            try checkPlain(c, "model.language_model.layers.{d}.input_layernorm", .{layer}, .bf16, &.{2304});
        }
        try checkPlain(c, "model.language_model.layers.{d}.post_attention_convolution.conv1d", .{layer}, .bf16, &.{ 2304, 4, 1 });
        try checkPlain(c, "model.language_model.layers.{d}.post_attention_layernorm", .{layer}, .bf16, &.{2304});
        try checkPlain(c, "model.language_model.layers.{d}.post_mlp_convolution.conv1d", .{layer}, .bf16, &.{ 2304, 4, 1 });
        try checkBf16OrF32(c, "model.language_model.layers.{d}.mlp.gate", .{layer}, &.{ 128, 2304 });
        try checkTensor(c, "model.language_model.layers.{d}.mlp.gate.e_score_correction_bias", .{layer}, .f32, &.{128});
        try checkExpert(c, layer, "gate_up_proj", p.precision, 2304, 2304);
        try checkExpert(c, layer, "down_proj", p.precision, 2304, 1152);
        expected_companions += 2; // expert .scales for gate_up_proj and down_proj
        if (layer >= lp.core_start and layer < lp.core_end) {
            inline for (.{ "q", "k", "v", "mlp" }) |role| {
                try checkPlain(c, "model.language_model.layers.{d}." ++ role ++ "_message_gate.down_proj", .{layer}, .bf16, &.{ 64, 4608 });
                try checkBf16OrF32(c, "model.language_model.layers.{d}." ++ role ++ "_message_gate.forget_gate_proj", .{layer}, &.{ 2304, 64 });
                try checkBf16OrF32(c, "model.language_model.layers.{d}." ++ role ++ "_message_gate.message_gate_proj", .{layer}, &.{ 2304, 64 });
            }
        }
    }

    if (@hasDecl(T, "keyIterator")) {
        if (seen_companions != expected_companions) return error.UnexpectedAffineCompanion;
        if (has_vision) try checkVision(c, config.vision);
    }
}

/// Read only safetensors headers across `files` and admit the exact declared layout before any GPU allocation.
pub fn admitPackFiles(gpa: std.mem.Allocator, io: std.Io, files: []const [:0]const u8, config: *const cfg.Config, wp: *const layer_plan.Plan) !void {
    if (files.len == 0) return error.MissingTensor;
    var index = HeaderIndex.init(gpa);
    defer index.deinit();
    for (files) |path| {
        const f = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer f.close(io);
        const size = try f.length(io);
        const size_usize = std.math.cast(usize, size) orelse return error.BadSafetensors;
        var prefix: [8]u8 = undefined;
        if (try f.readPositionalAll(io, &prefix, 0) != 8) return error.BadSafetensors;
        const hlen = try st.headerSize(std.mem.readInt(u64, &prefix, .little), size_usize);
        var hdr_arena = std.heap.ArenaAllocator.init(gpa);
        defer hdr_arena.deinit();
        const json = try hdr_arena.allocator().alloc(u8, hlen);
        if (try f.readPositionalAll(io, json, 8) != hlen) return error.BadSafetensors;
        const header = try st.parseHeader(hdr_arena.allocator(), json, size_usize - 8 - hlen);
        var it = header.iterator();
        while (it.next()) |kv| {
            const e = kv.value_ptr.*;
            var t: HeaderTensor = .{ .dtype = e.dtype, .rank = e.rank, .bytes = e.end - e.begin };
            for (0..e.rank) |i| t.shape[i] = e.shape[i];
            try index.put(kv.key_ptr.*, t);
        }
    }
    try admitHeaders(&index, config, wp);
}
