//! Drama backbone checkpoint tensors on the GPU as stored; fused projections concatenate rows, none is requantized.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("backbone_config.zig");
const Stager = @import("stager.zig").Stager;

const Io = std.Io;

pub const prefix = "model.language_model.";
pub const head_name = "lm_head.weight";

/// Rows of the GDN input projection: in_proj_qkv (8192) | in_proj_z (4096) | in_proj_b (32) | in_proj_a (32).
pub const gdn_in_rows = cfg.conv_dim + cfg.value_dim + 2 * cfg.linear_value_heads;
pub const gdn_z_col = cfg.conv_dim;
pub const gdn_b_col = cfg.conv_dim + cfg.value_dim;
/// Rows of the attention input projection: q_proj (16 x [q 256 | gate 256]) | k_proj (512) | v_proj (512).
pub const attn_in_rows = 2 * cfg.query_heads * cfg.head_dim + 2 * cfg.kv_heads * cfg.head_dim;
/// Rows of the MoE input projection: shared gate_proj (512) | shared up_proj (512) | router gate (256).
pub const moe_in_rows = 2 * cfg.shared_inner + cfg.experts;
pub const router_col = 2 * cfg.shared_inner;
/// Semantic head rows: EOS, the 4096-code band, zero rows to a multiple of 8.
pub const head_cols = 1 + 4096;
pub const head_rows = (head_cols + 7) / 8 * 8;
/// One expert's gate | up rows and their scale blocks.
pub const gu_rows = 2 * cfg.expert_inner;
pub const gu_scales = (gu_rows / cfg.block) * (cfg.hidden / cfg.block);
pub const dn_scales = (cfg.hidden / cfg.block) * (cfg.expert_inner / cfg.block);

pub const Layer = struct {
    ln1: u64 = 0, // bf16 [hidden] input_layernorm (zero-centered)
    ln2: u64 = 0, // bf16 [hidden] post_attention_layernorm (zero-centered)
    // Gated DeltaNet
    gdn_in: u64 = 0, // bf16 [gdn_in_rows][hidden]
    conv: u64 = 0, // bf16 [conv_dim][4]
    a_log: u64 = 0, // f32 [32], widened from the stored bf16
    dt_bias: u64 = 0, // f32 [32], widened from the stored bf16
    gnorm: u64 = 0, // bf16 [128] RMSNormGated weight (plain, not zero-centered)
    gdn_out: u64 = 0, // bf16 [hidden][value_dim]
    // full attention
    attn_in: u64 = 0, // bf16 [attn_in_rows][hidden]
    q_norm: u64 = 0, // bf16 [256] (zero-centered)
    k_norm: u64 = 0, // bf16 [256] (zero-centered)
    attn_out: u64 = 0, // bf16 [hidden][query_heads * head_dim]
    // MoE
    moe_in: u64 = 0, // bf16 [moe_in_rows][hidden]
    shared_down: u64 = 0, // bf16 [hidden][shared_inner]
    shared_gate: u64 = 0, // bf16 [hidden]
    gu: u64 = 0, // e4m3 [experts][gu_rows][hidden]
    gu_s: u64 = 0, // f32 [experts][gu_rows / 128][hidden / 128]
    dn: u64 = 0, // e4m3 [experts][hidden][expert_inner]
    dn_s: u64 = 0, // f32 [experts][hidden / 128][expert_inner / 128]
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    embed: u64 = 0, // bf16 [vocab][hidden]
    final_norm: u64 = 0, // bf16 [hidden] (zero-centered)
    head: u64 = 0, // bf16 [head_rows][hidden]
    inv_freq: u64 = 0, // f32 [rotary_dim / 2]
    layers: [cfg.layers]Layer = @splat(.{}),
    bytes: usize = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.* = undefined;
    }

    fn alloc(w: *Weights, d: *const cuda.Driver, bytes: usize) !u64 {
        try w.buffers.ensureUnusedCapacity(w.gpa, 1);
        const b = try cuda.DeviceBuffer.alloc(d, bytes);
        w.buffers.appendAssumeCapacity(b);
        w.bytes += bytes;
        return b.ptr;
    }
};

const DType = enum { bf16, f32, f8_e4m3 };

fn parseDType(s: []const u8) ?DType {
    if (std.mem.eql(u8, s, "BF16")) return .bf16;
    if (std.mem.eql(u8, s, "F32")) return .f32;
    if (std.mem.eql(u8, s, "F8_E4M3")) return .f8_e4m3;
    return null;
}

fn size(t: DType) usize {
    return switch (t) {
        .bf16 => 2,
        .f32 => 4,
        .f8_e4m3 => 1,
    };
}

const Entry = struct { shard: usize, dtype: ?DType, rank: usize, shape: [3]usize, begin: usize, end: usize };

const Shard = struct { path: [:0]const u8, file: Io.File, map: Io.File.MemoryMap, data: usize };

/// The checkpoint's shards mapped read-only, every header entry indexed by name.
const Source = struct {
    gpa: std.mem.Allocator,
    io: Io,
    arena: std.heap.ArenaAllocator,
    shards: std.ArrayList(Shard) = .empty,
    names: std.StringHashMapUnmanaged(Entry) = .empty,
    backbone: usize = 0, // entries under `prefix`
    used: usize = 0, // of those, consumed

    fn open(gpa: std.mem.Allocator, io: Io, dir: []const u8) !Source {
        var s: Source = .{ .gpa = gpa, .io = io, .arena = .init(gpa) };
        errdefer s.close();
        const a = s.arena.allocator();
        const index = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        const text = try Io.Dir.cwd().readFileAlloc(io, index, a, .limited(1 << 26));
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        if (doc != .object) return error.BadDramaCheckpoint;
        const map = doc.object.get("weight_map") orelse return error.BadDramaCheckpoint;
        if (map != .object) return error.BadDramaCheckpoint;
        var files: std.StringArrayHashMapUnmanaged(void) = .empty;
        var it = map.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .string) return error.BadDramaCheckpoint;
            try files.put(a, e.value_ptr.string, {});
        }
        for (files.keys(), 0..) |name, i| {
            const path = try std.fs.path.joinZ(a, &.{ dir, name });
            try s.mapShard(path, i);
        }
        // every name the index lists must be in its shard's header
        it = map.object.iterator();
        while (it.next()) |e| {
            const got = s.names.get(e.key_ptr.*) orelse return error.BadDramaCheckpoint;
            if (!std.mem.eql(u8, files.keys()[got.shard], e.value_ptr.string)) return error.BadDramaCheckpoint;
        }
        return s;
    }

    fn mapShard(s: *Source, path: [:0]const u8, index: usize) !void {
        const a = s.arena.allocator();
        // the file and its map belong to `shards` once appended; the errdefers cover only the steps before
        const sh = blk: {
            var file = try Io.Dir.cwd().openFile(s.io, path, .{});
            errdefer file.close(s.io);
            const len: usize = @intCast(try file.length(s.io));
            if (len < 8) return error.BadDramaCheckpoint;
            var map = try Io.File.MemoryMap.create(s.io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
            errdefer map.destroy(s.io);
            const header_len: usize = @intCast(std.mem.readInt(u64, map.memory[0..8], .little));
            if (header_len > len - 8) return error.BadDramaCheckpoint;
            try s.shards.append(s.gpa, .{ .path = path, .file = file, .map = map, .data = 8 + header_len });
            break :blk s.shards.items[s.shards.items.len - 1];
        };
        const data_len = sh.map.memory.len - sh.data;
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, sh.map.memory[8..sh.data], .{ .allocate = .alloc_always });
        if (doc != .object) return error.BadDramaCheckpoint;
        var it = doc.object.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            const o = if (kv.value_ptr.* == .object) kv.value_ptr.object else return error.BadDramaCheckpoint;
            const dt = o.get("dtype") orelse return error.BadDramaCheckpoint;
            const shape = o.get("shape") orelse return error.BadDramaCheckpoint;
            const offs = o.get("data_offsets") orelse return error.BadDramaCheckpoint;
            if (dt != .string or shape != .array or offs != .array or offs.array.items.len != 2) return error.BadDramaCheckpoint;
            var e: Entry = .{ .shard = index, .dtype = parseDType(dt.string), .rank = shape.array.items.len, .shape = @splat(1), .begin = 0, .end = 0 };
            if (e.rank > 3) e.dtype = null; // not a backbone layout; refused if anything asks for it
            for (shape.array.items, 0..) |d, i| {
                if (d != .integer or d.integer < 0) return error.BadDramaCheckpoint;
                if (i < 3) e.shape[i] = @intCast(d.integer);
            }
            for (offs.array.items, 0..) |d, i| {
                if (d != .integer or d.integer < 0) return error.BadDramaCheckpoint;
                if (i == 0) e.begin = @intCast(d.integer) else e.end = @intCast(d.integer);
            }
            if (e.end < e.begin or e.end > data_len) return error.BadDramaCheckpoint;
            if (std.mem.startsWith(u8, kv.key_ptr.*, prefix)) s.backbone += 1;
            const got = try s.names.getOrPut(s.gpa, kv.key_ptr.*);
            if (got.found_existing) return error.BadDramaCheckpoint;
            got.value_ptr.* = e;
        }
    }

    fn close(s: *Source) void {
        for (s.shards.items) |*sh| {
            sh.map.destroy(s.io);
            sh.file.close(s.io);
        }
        s.shards.deinit(s.gpa);
        s.names.deinit(s.gpa);
        s.arena.deinit();
        s.* = undefined;
    }

    /// The bytes of `name`, refused unless its dtype and shape are exactly these.
    fn get(s: *Source, name: []const u8, dtype: DType, shape: []const usize) ![]const u8 {
        const e = s.names.get(name) orelse {
            std.log.err("drama backbone: checkpoint has no {s}", .{name});
            return error.MissingDramaTensor;
        };
        var ok = e.dtype != null and e.dtype.? == dtype and e.rank == shape.len;
        var n: usize = size(dtype);
        for (shape, 0..) |d, i| {
            ok = ok and e.shape[i] == d;
            n *= d;
        }
        if (!ok or e.end - e.begin != n) {
            std.log.err("drama backbone: {s} is not {s} {any}", .{ name, @tagName(dtype), shape });
            return error.UnexpectedDramaTensor;
        }
        if (std.mem.startsWith(u8, name, prefix)) s.used += 1;
        const sh = s.shards.items[e.shard];
        return sh.map.memory[sh.data + e.begin .. sh.data + e.end];
    }
};

fn widenBf16(out: []f32, bytes: []const u8) void {
    for (out, 0..) |*o, i| o.* = @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16);
}

const Loader = struct {
    src: *Source,
    st: *Stager,
    w: *Weights,
    d: *const cuda.Driver,
    name_buf: [192]u8 = undefined,

    fn name(L: *Loader, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&L.name_buf, fmt, args);
    }

    /// A fresh device tensor holding `name` as stored.
    fn raw(L: *Loader, n: []const u8, dtype: DType, shape: []const usize) !u64 {
        const bytes = try L.src.get(n, dtype, shape);
        const p = try L.w.alloc(L.d, bytes.len);
        try L.st.put(p, bytes);
        return p;
    }

    /// `name` as stored at `dst` (part of a fused tensor).
    fn into(L: *Loader, dst: u64, n: []const u8, dtype: DType, shape: []const usize) !void {
        try L.st.put(dst, try L.src.get(n, dtype, shape));
    }

    fn layerRaw(L: *Loader, i: usize, comptime suffix: []const u8, dtype: DType, shape: []const usize) !u64 {
        return L.raw(try L.name(prefix ++ "layers.{d}." ++ suffix, .{i}), dtype, shape);
    }

    fn layerInto(L: *Loader, dst: u64, i: usize, comptime suffix: []const u8, dtype: DType, shape: []const usize) !void {
        try L.into(dst, try L.name(prefix ++ "layers.{d}." ++ suffix, .{i}), dtype, shape);
    }

    fn widened(L: *Loader, i: usize, comptime suffix: []const u8) !u64 {
        const n = cfg.linear_value_heads;
        const bytes = try L.src.get(try L.name(prefix ++ "layers.{d}." ++ suffix, .{i}), .bf16, &.{n});
        var host: [n]f32 = undefined;
        widenBf16(&host, bytes);
        const p = try L.w.alloc(L.d, n * 4);
        try L.st.put(p, std.mem.sliceAsBytes(&host));
        return p;
    }
};

const H = cfg.hidden;

fn loadLinear(L: *Loader, i: usize, l: *Layer) !void {
    const row = H * 2;
    l.gdn_in = try L.w.alloc(L.d, gdn_in_rows * row);
    try L.layerInto(l.gdn_in, i, "linear_attn.in_proj_qkv.weight", .bf16, &.{ cfg.conv_dim, H });
    try L.layerInto(l.gdn_in + gdn_z_col * row, i, "linear_attn.in_proj_z.weight", .bf16, &.{ cfg.value_dim, H });
    try L.layerInto(l.gdn_in + gdn_b_col * row, i, "linear_attn.in_proj_b.weight", .bf16, &.{ cfg.linear_value_heads, H });
    try L.layerInto(l.gdn_in + (gdn_b_col + cfg.linear_value_heads) * row, i, "linear_attn.in_proj_a.weight", .bf16, &.{ cfg.linear_value_heads, H });
    l.conv = try L.layerRaw(i, "linear_attn.conv1d.weight", .bf16, &.{ cfg.conv_dim, 1, cfg.conv_taps });
    l.a_log = try L.widened(i, "linear_attn.A_log");
    l.dt_bias = try L.widened(i, "linear_attn.dt_bias");
    l.gnorm = try L.layerRaw(i, "linear_attn.norm.weight", .bf16, &.{cfg.linear_dim});
    l.gdn_out = try L.layerRaw(i, "linear_attn.out_proj.weight", .bf16, &.{ H, cfg.value_dim });
}

fn loadFull(L: *Loader, i: usize, l: *Layer) !void {
    const row = H * 2;
    const q_rows = 2 * cfg.query_heads * cfg.head_dim;
    const kv_rows = cfg.kv_heads * cfg.head_dim;
    l.attn_in = try L.w.alloc(L.d, attn_in_rows * row);
    try L.layerInto(l.attn_in, i, "self_attn.q_proj.weight", .bf16, &.{ q_rows, H });
    try L.layerInto(l.attn_in + q_rows * row, i, "self_attn.k_proj.weight", .bf16, &.{ kv_rows, H });
    try L.layerInto(l.attn_in + (q_rows + kv_rows) * row, i, "self_attn.v_proj.weight", .bf16, &.{ kv_rows, H });
    l.q_norm = try L.layerRaw(i, "self_attn.q_norm.weight", .bf16, &.{cfg.head_dim});
    l.k_norm = try L.layerRaw(i, "self_attn.k_norm.weight", .bf16, &.{cfg.head_dim});
    l.attn_out = try L.layerRaw(i, "self_attn.o_proj.weight", .bf16, &.{ H, cfg.query_heads * cfg.head_dim });
}

fn loadMoe(L: *Loader, i: usize, l: *Layer) !void {
    const row = H * 2;
    const I = cfg.expert_inner;
    l.moe_in = try L.w.alloc(L.d, moe_in_rows * row);
    try L.layerInto(l.moe_in, i, "mlp.shared_expert.gate_proj.weight", .bf16, &.{ cfg.shared_inner, H });
    try L.layerInto(l.moe_in + cfg.shared_inner * row, i, "mlp.shared_expert.up_proj.weight", .bf16, &.{ cfg.shared_inner, H });
    try L.layerInto(l.moe_in + router_col * row, i, "mlp.gate.weight", .bf16, &.{ cfg.experts, H });
    l.shared_down = try L.layerRaw(i, "mlp.shared_expert.down_proj.weight", .bf16, &.{ H, cfg.shared_inner });
    l.shared_gate = try L.layerRaw(i, "mlp.shared_expert_gate.weight", .bf16, &.{ 1, H });
    const E = cfg.experts;
    l.gu = try L.w.alloc(L.d, E * gu_rows * H);
    l.gu_s = try L.w.alloc(L.d, E * gu_scales * 4);
    l.dn = try L.w.alloc(L.d, E * H * I);
    l.dn_s = try L.w.alloc(L.d, E * dn_scales * 4);
    const gs = &[_]usize{ I / cfg.block, H / cfg.block };
    const ds = &[_]usize{ H / cfg.block, I / cfg.block };
    for (0..E) |e| {
        const gu = l.gu + e * gu_rows * H;
        const gu_s = l.gu_s + e * gu_scales * 4;
        try L.into(gu, try L.name(prefix ++ "layers.{d}.mlp.experts.{d}.gate_proj.weight", .{ i, e }), .f8_e4m3, &.{ I, H });
        try L.into(gu + I * H, try L.name(prefix ++ "layers.{d}.mlp.experts.{d}.up_proj.weight", .{ i, e }), .f8_e4m3, &.{ I, H });
        try L.into(gu_s, try L.name(prefix ++ "layers.{d}.mlp.experts.{d}.gate_proj.weight_scale_inv", .{ i, e }), .f32, gs);
        try L.into(gu_s + gu_scales * 2, try L.name(prefix ++ "layers.{d}.mlp.experts.{d}.up_proj.weight_scale_inv", .{ i, e }), .f32, gs);
        try L.into(l.dn + e * H * I, try L.name(prefix ++ "layers.{d}.mlp.experts.{d}.down_proj.weight", .{ i, e }), .f8_e4m3, &.{ H, I });
        try L.into(l.dn_s + e * dn_scales * 4, try L.name(prefix ++ "layers.{d}.mlp.experts.{d}.down_proj.weight_scale_inv", .{ i, e }), .f32, ds);
    }
}

/// Uploads the backbone tensors and lm_head's semantic rows; refuses missing, mis-shaped or unconsumed tensors.
pub fn load(gpa: std.mem.Allocator, io: Io, d: *const cuda.Driver, stream: cuda.Stream, dir: []const u8, c: cfg.Config) !Weights {
    var src = try Source.open(gpa, io, dir);
    defer src.close();
    var st: Stager = .init(gpa, d, stream);
    defer st.deinit();
    for (src.shards.items) |sh| try st.addFile(sh.path, sh.map.memory);
    var w: Weights = .{ .gpa = gpa };
    errdefer {
        stream.synchronize() catch {};
        w.deinit();
    }
    var L: Loader = .{ .src = &src, .st = &st, .w = &w, .d = d };
    w.embed = try L.raw(prefix ++ "embed_tokens.weight", .bf16, &.{ cfg.vocab, H });
    w.final_norm = try L.raw(prefix ++ "norm.weight", .bf16, &.{H});
    for (&w.layers, 0..) |*l, i| {
        l.ln1 = try L.layerRaw(i, "input_layernorm.weight", .bf16, &.{H});
        l.ln2 = try L.layerRaw(i, "post_attention_layernorm.weight", .bf16, &.{H});
        if (cfg.linear(i)) try loadLinear(&L, i, l) else try loadFull(&L, i, l);
        try loadMoe(&L, i, l);
    }
    // lm_head: only the rows the semantic logits read (EOS, then semantic_start..semantic_end), zero padded.
    const head = try src.get(head_name, .bf16, &.{ cfg.vocab, H });
    const row = H * 2;
    w.head = try w.alloc(d, head_rows * row);
    try st.zero(w.head, head_rows * row);
    try st.put(w.head, head[@as(usize, c.eos) * row ..][0..row]);
    try st.put(w.head + row, head[@as(usize, c.semantic_start) * row ..][0 .. (head_cols - 1) * row]);
    // RoPE inverse frequencies as SGLang computes them in fp32: 1 / theta^(2i / rotary_dim).
    var inv: [cfg.rotary_dim / 2]f32 = undefined;
    for (&inv, 0..) |*f, i| {
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, cfg.rotary_dim);
        f.* = 1.0 / std.math.pow(f32, @floatCast(cfg.theta), e);
    }
    w.inv_freq = try w.alloc(d, inv.len * 4);
    try st.put(w.inv_freq, std.mem.sliceAsBytes(&inv));
    if (src.used != src.backbone) {
        std.log.err("drama backbone: {d} of {d} {s}* tensors left unused", .{ src.backbone - src.used, src.backbone, prefix });
        return error.UnusedDramaTensors;
    }
    try st.finish();
    return w;
}
