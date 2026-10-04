//! A checkpoint's tensor table from its config and the Hub's file list, so a cluster can be planned before the weights land.
const std = @import("std");
const checkpoint = @import("checkpoint.zig");
const model = @import("model.zig");
const roles = @import("roles.zig");

const Allocator = std.mem.Allocator;
const Tensor = checkpoint.Tensor;
const File = checkpoint.File;

/// The Hub file list as `name<TAB>size<TAB>sha256` lines (sha256 empty for files stored in git).
pub fn manifest(a: Allocator, text: []const u8) ![]File {
    var out: std.ArrayList(File) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var cols = std.mem.splitScalar(u8, line, '\t');
        const name = cols.next() orelse continue;
        if (!std.mem.endsWith(u8, name, ".safetensors")) continue;
        var f: File = .{ .name = try a.dupe(u8, name), .size = std.fmt.parseInt(u64, cols.next() orelse continue, 10) catch continue };
        if (cols.next()) |hex| if (hex.len == 64) {
            var sha: [32]u8 = undefined;
            if (std.fmt.hexToBytes(&sha, hex)) |_| {
                f.sha256 = sha;
            } else |_| {}
        };
        try out.append(a, f);
    }
    std.mem.sort(File, out.items, {}, struct {
        fn less(_: void, x: File, y: File) bool {
            return std.mem.order(u8, x.name, y.name) == .lt;
        }
    }.less);
    return out.items;
}

/// Bytes a safetensors header takes per tensor entry, roughly (only start offsets depend on it).
const entry_bytes = 110;

const Builder = struct {
    a: Allocator,
    out: std.ArrayList(Tensor) = .empty,
    file: u32 = 0,
    at: u64 = 0,
    layer: u32 = 0,
    prefix: []const u8 = "language_model.model.layers",

    fn begin(b: *Builder, file: u32, layer: u32, entries: u64) void {
        b.file = file;
        b.layer = layer;
        b.at = 8 + entry_bytes * entries;
    }

    fn add(b: *Builder, name: []const u8, dtype: checkpoint.DType, shape: []const u64) !void {
        var t: Tensor = .{ .name = name, .file = b.file, .start = b.at, .bytes = checkpoint.size(dtype), .dtype = dtype, .rank = @intCast(shape.len) };
        for (shape, 0..) |d, i| {
            t.shape[i] = d;
            t.bytes *= d;
        }
        t.class = roles.classify(name);
        b.at += t.bytes;
        try b.out.append(b.a, t);
    }

    fn in(b: *Builder, suffix: []const u8, dtype: checkpoint.DType, shape: []const u64) !void {
        try b.add(try std.fmt.allocPrint(b.a, "{s}.{d}.{s}", .{ b.prefix, b.layer, suffix }), dtype, shape);
    }

    /// An FP8 matrix and its fp32 scales, one per 128 x 128 block (DeepSeek's `weight_scale_inv`).
    fn fp8(b: *Builder, stem: []const u8, rows: u64, cols: u64) !void {
        try b.in(try sfx(b.a, "{s}.weight", stem), .f8_e4m3, &.{ rows, cols });
        try b.in(try sfx(b.a, "{s}.weight_scale_inv", stem), .f32, &.{ std.math.divCeil(u64, rows, 128) catch unreachable, std.math.divCeil(u64, cols, 128) catch unreachable });
    }
};

fn kda(b: *Builder, s: *const model.Shape) !void {
    const h = s.hidden;
    const d: u64 = s.linear_head_dim;
    const p = @as(u64, s.linear_heads) * d;
    for ([_][]const u8{ "q_proj", "k_proj", "v_proj", "f_proj" }) |w| try b.in(try sfx(b.a, "self_attn.{s}.weight", w), .bf16, &.{ p, h });
    try b.in("self_attn.g_a_proj.weight", .bf16, &.{ d, h });
    try b.in("self_attn.g_b_proj.weight", .bf16, &.{ p, d });
    try b.in("self_attn.b_proj.weight", .bf16, &.{ s.linear_heads, h });
    for ([_][]const u8{ "q_conv1d", "k_conv1d", "v_conv1d" }) |w| try b.in(try sfx(b.a, "self_attn.{s}.weight", w), .f32, &.{ p, 1, s.conv_kernel });
    try b.in("self_attn.A_log", .f32, &.{s.linear_heads});
    try b.in("self_attn.dt_bias", .f32, &.{p});
    try b.in("self_attn.o_norm.weight", .bf16, &.{d});
    try b.in("self_attn.o_proj.weight", .bf16, &.{ h, p });
}

fn mla(b: *Builder, s: *const model.Shape) !void {
    const h = s.hidden;
    const heads: u64 = s.heads;
    try b.in("self_attn.q_a_proj.weight", .bf16, &.{ s.q_lora, h });
    try b.in("self_attn.q_a_layernorm.weight", .bf16, &.{s.q_lora});
    try b.in("self_attn.q_b_proj.weight", .bf16, &.{ heads * (s.nope_dim + s.rope_dim), s.q_lora });
    try b.in("self_attn.kv_a_proj_with_mqa.weight", .bf16, &.{ s.kv_lora + s.rope_dim, h });
    try b.in("self_attn.kv_a_layernorm.weight", .bf16, &.{s.kv_lora});
    try b.in("self_attn.kv_b_proj.weight", .bf16, &.{ heads * (s.nope_dim + s.v_dim), s.kv_lora });
    try b.in("self_attn.o_proj.weight", .bf16, &.{ h, heads * s.v_dim });
    try b.in("self_attn.g_proj.weight", .bf16, &.{ heads * s.v_dim, h });
}

fn moe(b: *Builder, s: *const model.Shape) !void {
    const h = s.hidden;
    const lat = s.latent;
    const i = s.moe_inter;
    const g: u64 = s.quant_group;
    try b.in("mlp.gate.weight", .bf16, &.{ s.experts, h });
    try b.in("mlp.gate.e_score_correction_bias", .f32, &.{s.experts});
    for ([_][]const u8{ "gate_proj", "up_proj" }) |w| try b.in(try sfx(b.a, "mlp.shared_experts.{s}.weight", w), .bf16, &.{ s.shared * i, h });
    try b.in("mlp.shared_experts.down_proj.weight", .bf16, &.{ h, s.shared * i });
    try b.in("mlp.latent_in_proj.weight", .bf16, &.{ lat, h });
    try b.in("mlp.latent_out_proj.weight", .bf16, &.{ h, lat });
    try b.in("mlp.latent_norm.weight", .bf16, &.{lat});
    for (0..s.experts) |e| {
        const projs = [_]struct { []const u8, u64, u64 }{ .{ "gate_proj", i, lat }, .{ "up_proj", i, lat }, .{ "down_proj", lat, i } };
        for (projs) |pr| {
            var buf: [96]u8 = undefined;
            const stem = std.fmt.bufPrint(&buf, "mlp.experts.{d}.{s}", .{ e, pr[0] }) catch unreachable;
            try b.in(try sfx(b.a, "{s}.weight_packed", stem), .u8, &.{ pr[1], pr[2] / 2 });
            try b.in(try sfx(b.a, "{s}.weight_scale", stem), .u8, &.{ pr[1], pr[2] / g });
            try b.in(try sfx(b.a, "{s}.weight_shape", stem), .i64, &.{2});
        }
    }
}

fn sfx(a: Allocator, comptime fmt: []const u8, x: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, fmt, .{x});
}

/// Kimi Linear (K3): KDA with one full-rank gate, MLA with an output gate, latent MXFP4 experts; layer L is file L.
pub fn kimi(a: Allocator, s: *const model.Shape, files: []const File) !checkpoint.Checkpoint {
    if (files.len < s.layers + 1) return error.Inconsistent;
    var b: Builder = .{ .a = a };
    for (0..s.layers) |l| {
        const layer: u32 = @intCast(l);
        b.begin(layer, layer, if (s.moe(layer)) 9 * @as(u64, s.experts) + 40 else 40);
        try b.in("input_layernorm.weight", .bf16, &.{s.hidden});
        try b.in("post_attention_layernorm.weight", .bf16, &.{s.hidden});
        if (s.kind(layer) == .linear) try kda(&b, s) else try mla(&b, s);
        if (s.moe(layer)) {
            try moe(&b, s);
        } else {
            for ([_][]const u8{ "gate_proj", "up_proj" }) |w| try b.in(try sfx(a, "mlp.{s}.weight", w), .bf16, &.{ s.inter, s.hidden });
            try b.in("mlp.down_proj.weight", .bf16, &.{ s.hidden, s.inter });
        }
    }
    b.begin(s.layers, 0, 3);
    try b.add("language_model.model.embed_tokens.weight", .bf16, &.{ s.vocab, s.hidden });
    try b.add("language_model.lm_head.weight", .bf16, &.{ s.vocab, s.hidden });
    try b.add("language_model.model.norm.weight", .bf16, &.{s.hidden});
    for (files[s.layers + 1 ..], s.layers + 1..) |f, i| {
        b.begin(@intCast(i), 0, 1);
        try b.add(try std.fmt.allocPrint(a, "vision_tower.file{d}", .{i}), .u8, &.{f.size -| b.at});
    }
    return .{ .files = try a.dupe(File, files), .tensors = b.out.items };
}

/// One GLM layer: MLA with an output projection of heads x v_dim, the DSA indexer, then a dense or MoE MLP (all FP8).
fn glmLayer(b: *Builder, s: *const model.Shape, moe_layer: bool) !void {
    const h = s.hidden;
    const heads: u64 = s.heads;
    try b.in("input_layernorm.weight", .bf16, &.{h});
    try b.in("post_attention_layernorm.weight", .bf16, &.{h});
    try b.fp8("self_attn.q_a_proj", s.q_lora, h);
    try b.in("self_attn.q_a_layernorm.weight", .bf16, &.{s.q_lora});
    try b.fp8("self_attn.q_b_proj", heads * (s.nope_dim + s.rope_dim), s.q_lora);
    try b.fp8("self_attn.kv_a_proj_with_mqa", s.kv_lora + s.rope_dim, h);
    try b.in("self_attn.kv_a_layernorm.weight", .bf16, &.{s.kv_lora});
    try b.fp8("self_attn.kv_b_proj", heads * (s.nope_dim + s.v_dim), s.kv_lora);
    try b.fp8("self_attn.o_proj", h, heads * s.v_dim);
    try b.fp8("self_attn.indexer.wq_b", @as(u64, s.index_heads) * s.index_dim, s.q_lora);
    try b.fp8("self_attn.indexer.wk", s.index_dim, h);
    try b.in("self_attn.indexer.k_norm.weight", .bf16, &.{s.index_dim});
    try b.in("self_attn.indexer.k_norm.bias", .bf16, &.{s.index_dim});
    try b.in("self_attn.indexer.weights_proj.weight", .bf16, &.{ s.index_heads, h });
    if (!moe_layer) {
        try b.fp8("mlp.gate_proj", s.inter, h);
        try b.fp8("mlp.up_proj", s.inter, h);
        try b.fp8("mlp.down_proj", h, s.inter);
        return;
    }
    try b.in("mlp.gate.weight", .bf16, &.{ s.experts, h });
    try b.in("mlp.gate.e_score_correction_bias", .f32, &.{s.experts});
    const wide = s.shared * s.moe_inter;
    try b.fp8("mlp.shared_experts.gate_proj", wide, h);
    try b.fp8("mlp.shared_experts.up_proj", wide, h);
    try b.fp8("mlp.shared_experts.down_proj", h, wide);
    for (0..s.experts) |e| {
        var buf: [64]u8 = undefined;
        for ([_][]const u8{ "gate_proj", "up_proj" }) |w| try b.fp8(try b.a.dupe(u8, std.fmt.bufPrint(&buf, "mlp.experts.{d}.{s}", .{ e, w }) catch unreachable), s.moe_inter, h);
        try b.fp8(try b.a.dupe(u8, std.fmt.bufPrint(&buf, "mlp.experts.{d}.down_proj", .{e}) catch unreachable), h, s.moe_inter);
    }
}

/// GLM-5.3 (glm_moe_dsa) in FP8 with block scales: one file per layer (the Hub's split is unknown), MTP, then the top.
pub fn glm(a: Allocator, s: *const model.Shape) !checkpoint.Checkpoint {
    var b: Builder = .{ .a = a, .prefix = "model.layers" };
    const count = s.layers + s.mtp_layers + 1;
    var files = try a.alloc(File, count);
    for (0..s.layers + s.mtp_layers) |l| {
        const layer: u32 = @intCast(l);
        b.begin(layer, layer, if (s.moe(layer)) 6 * @as(u64, s.experts) + 40 else 40);
        try glmLayer(&b, s, s.moe(layer));
        if (layer >= s.layers) {
            try b.fp8("eh_proj", s.hidden, 2 * s.hidden);
            for ([_][]const u8{ "enorm.weight", "hnorm.weight", "shared_head.norm.weight" }) |n| try b.in(n, .bf16, &.{s.hidden});
        }
        files[l] = .{ .name = try std.fmt.allocPrint(a, "model-{d:0>5}-of-{d:0>5}.safetensors", .{ l + 1, count }), .size = b.at };
    }
    b.begin(count - 1, 0, 3);
    try b.add("model.embed_tokens.weight", .bf16, &.{ s.vocab, s.hidden });
    try b.add("lm_head.weight", .bf16, &.{ s.vocab, s.hidden });
    try b.add("model.norm.weight", .bf16, &.{s.hidden});
    files[count - 1] = .{ .name = try std.fmt.allocPrint(a, "model-{d:0>5}-of-{d:0>5}.safetensors", .{ count, count }), .size = b.at };
    return .{ .files = files, .tensors = b.out.items };
}

/// Each file's bytes the table holds, to compare with the real file sizes (the difference is the header).
pub fn fileBytes(a: Allocator, c: checkpoint.Checkpoint) ![]u64 {
    const out = try a.alloc(u64, c.files.len);
    @memset(out, 0);
    for (c.tensors) |t| out[t.file] += t.bytes;
    return out;
}

const k3_sizes = struct {
    const kda_layer = 16_990_911_504;
    const mla_layer = 16_567_501_776;
    const first = 2_341_216_112;
    const top = 4_697_664_072;
    const vision = [_]u64{ 92_289_328, 802_448_352 };
};

fn k3Files(a: Allocator, s: *const model.Shape) ![]File {
    const out = try a.alloc(File, s.layers + 3);
    for (out, 0..) |*f, i| {
        const size: u64 = if (i == 0) k3_sizes.first else if (i < s.layers) (if (s.kind(@intCast(i)) == .linear) k3_sizes.kda_layer else k3_sizes.mla_layer) else if (i == s.layers) k3_sizes.top else k3_sizes.vision[i - s.layers - 1];
        f.* = .{ .name = try std.fmt.allocPrint(a, "model-{d:0>5}-of-{d:0>6}.safetensors", .{ i + 1, s.layers + 3 }), .size = size };
    }
    return out;
}

/// Every file's header is what the table leaves over: a few hundred kilobytes, never negative.
fn expectHeaders(a: Allocator, c: checkpoint.Checkpoint) !void {
    const got = try fileBytes(a, c);
    for (c.files, got) |f, b| {
        try std.testing.expect(b <= f.size);
        try std.testing.expect(f.size - b <= @max(f.size / 10_000, 64 << 10));
    }
}

test "GLM-5.3's table: 753.8 B parameters and about 756 GB in FP8, the MTP layer included" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s = model.glm53();
    const c = try glm(a, &s);
    try expectHeaders(a, c);
    var params: u64 = 0;
    var experts: u64 = 0;
    for (c.tensors) |t| {
        if (t.dtype != .f32 or std.mem.indexOf(u8, t.name, "scale") == null) params += t.bytes / checkpoint.size(t.dtype);
        if (t.class.role == .expert) experts += t.bytes;
    }
    try std.testing.expect(params > 753_000_000_000 and params < 754_500_000_000);
    try std.testing.expect(c.bytes() > 755_000_000_000 and c.bytes() < 757_000_000_000);
    try std.testing.expectEqual(@as(u64, 76 * 256 * 3 * (6144 * 2048 + 48 * 16 * 4)), experts);
    try std.testing.expectEqual(@as(usize, 80), c.files.len);
}

test "the K3 table matches every shard size to within its header" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s = model.k3();
    const c = try kimi(a, &s, try k3Files(a, &s));
    try expectHeaders(a, c);
    var experts: u64 = 0;
    for (c.tensors) |t| experts += if (t.class.role == .expert) t.bytes else 0;
    try std.testing.expectEqual(@as(u64, 92 * 896 * (3 * 5_849_088 + 48)), experts);
}

test "the real K3 file list (TF_K3_DIR): 96 files, 1.561 TB, every header under 0.01% of its file" {
    const dir = std.testing.environ.getPosix("TF_K3_DIR") orelse return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try std.fs.path.join(a, &.{ dir, "files.tsv" }), a, .limited(1 << 20));
    const files = try manifest(a, text);
    try std.testing.expectEqual(@as(usize, 96), files.len);
    var total: u64 = 0;
    for (files) |f| total += f.size;
    try std.testing.expectEqual(@as(u64, 1_560_936_091_448), total);
    try std.testing.expect(files[0].sha256 != null);
    const s = model.k3();
    try expectHeaders(a, try kimi(a, &s, files));
}
