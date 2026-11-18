//! The Qwen3.6 MTP layer's weights (mtp-4bit.safetensors beside the checkpoint) onto the device, their own arena.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const engine = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");

const Proj = engine.Proj;
const DeviceBuffer = cuda.DeviceBuffer;
const Engine = engine.Engine;
const head_dim = geometry.head_dim;

const arena_alignment = 256;

pub fn checkedAdd(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.MtpMemorySizeOverflow;
}

pub fn checkedMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.MtpMemorySizeOverflow;
}

pub fn checkedAligned(n: usize) !usize {
    const rounded = try checkedAdd(n, arena_alignment - 1);
    return rounded & ~@as(usize, arena_alignment - 1);
}

fn addTo(total_: *usize, value: usize) !void {
    total_.* = try checkedAdd(total_.*, value);
}

fn addProduct(total_: *usize, count: usize, value: usize) !void {
    try addTo(total_, try checkedMul(count, value));
}

fn paddedRows(n: usize) !usize {
    return checkedMul(try checkedAdd(n, 127) / 128, 128);
}

fn packedBytes(n: usize, k: usize) !usize {
    const npad = try paddedRows(n);
    const max_c_int: usize = @intCast(std.math.maxInt(c_int));
    if (npad > max_c_int) return error.UnsupportedMtpGeometry;
    const words = try checkedMul(npad, k) / 2;
    const scale_words = try checkedMul(k / 64, npad);
    const scales = try checkedMul(scale_words, 2);
    return checkedAdd(try checkedAligned(words), try checkedMul(2, try checkedAligned(scales)));
}

fn rawBytes(n: usize, k: usize) !usize {
    const elements = try checkedMul(n, k);
    const words = try checkedMul(elements / 8, 4);
    const scale_bytes = try checkedMul(elements / 64, 2);
    return checkedAdd(words, try checkedMul(2, scale_bytes));
}

fn expertBytes(count: usize, n: usize, k: usize) !usize {
    var bytes = try checkedMul(count, n / 32);
    bytes = try checkedMul(bytes, k / 64);
    return checkedMul(bytes, 288 * 4);
}

fn routedBytes(experts: usize, intermediate: usize, hidden: usize) !usize {
    const count = try checkedAdd(experts, 1);
    const pair = try checkedAligned(try checkedMul(2, try expertBytes(count, intermediate, hidden)));
    const down = try checkedAligned(try expertBytes(count, hidden, intermediate));
    const rows = try checkedAligned(try checkedMul(try checkedMul(count, hidden), 2));
    return checkedAdd(try checkedAdd(pair, down), rows);
}

pub fn validateGeometry(g: geometry.Geometry) !void {
    const max_c_int: usize = @intCast(std.math.maxInt(c_int));
    if (g.hidden == 0 or g.hidden > 8192 or g.intermediate == 0 or g.layers == 0 or g.layers > geometry.max_layers or g.vocab == 0 or g.q_heads == 0 or g.kv_heads == 0)
        return error.UnsupportedMtpGeometry;
    if (g.hidden % 64 != 0 or g.intermediate % 64 != 0) return error.UnsupportedMtpGeometry;
    if (g.q_heads > geometry.max_query_heads or g.q_heads % g.kv_heads != 0 or g.experts > 256)
        return error.UnsupportedMtpGeometry;
    if (g.hidden > max_c_int or g.intermediate > max_c_int or g.vocab > max_c_int or g.experts > max_c_int)
        return error.UnsupportedMtpGeometry;
    const aw = try checkedMul(g.q_heads, head_dim);
    const kw = try checkedMul(g.kv_heads, head_dim);
    const q_width = try checkedMul(2, aw);
    if (aw > max_c_int or kw > max_c_int or q_width > max_c_int)
        return error.UnsupportedMtpGeometry;
    for ([_]usize{ g.hidden, g.intermediate, aw, kw, q_width }) |n| {
        if (try paddedRows(n) > max_c_int) return error.UnsupportedMtpGeometry;
    }
}

pub const file = "mtp-4bit.safetensors";

pub const AllocationPlan = struct { resident: usize, staging: usize };
pub const Source = enum { standalone_affine4, indexed_bf16 };

/// Whether `dir` holds `file`.
fn hasFile(io: std.Io, dir: []const u8) bool {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, file }) catch return false;
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

/// Choose only formats whose loader matches this target and the stored MTP tensors.
pub fn sourceForTarget(g: geometry.Geometry, standalone: bool, indexed_head: bool, indexed_scales: bool) !Source {
    if (!g.moe() and standalone) return error.UnsupportedMtpWeightFormat;
    if (standalone) return .standalone_affine4;
    if (!indexed_head) return error.NoMtpLayer;
    if (indexed_scales) return error.UnsupportedMtpWeightFormat;
    return .indexed_bf16;
}

/// Whether this target can use a standalone or checkpoint-indexed MTP head.
pub fn hasHead(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    if (hasFile(io, dir)) return true;
    var ck = core.Checkpoint.openModelPrefix(gpa, io, dir, "mtp.") catch return false;
    defer core.checkpoint.cache.close(&ck);
    return ck.has("mtp.fc.weight");
}

pub const Weights = struct {
    arena: DeviceBuffer,
    arena_at: usize = 0,
    fc_e: Proj, // fc's halves: the embedding's inputs, then the hidden state's
    fc_h: Proj,
    main_norm: ?u64 = null,
    raw: struct { norm_e: bool = false, norm_h: bool = false, norm: bool = false, main: bool = false, input: bool = false, post: bool = false } = .{},
    norm_e: u64,
    norm_h: u64,
    input_norm: u64,
    post_norm: u64,
    q_norm: u64,
    k_norm: u64,
    norm: u64,
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    mlp: engine.Mlp,

    pub fn alloc(w: *Weights, bytes: usize) !u64 {
        const at = try checkedAligned(w.arena_at);
        const end = try checkedAdd(at, bytes);
        if (end > w.arena.len) return error.ArenaExhausted;
        const offset = std.math.cast(u64, at) orelse return error.MtpMemorySizeOverflow;
        const ptr = std.math.add(u64, w.arena.ptr, offset) catch return error.MtpMemorySizeOverflow;
        w.arena_at = end;
        return ptr;
    }

    pub fn deinit(w: *Weights) void {
        w.arena.free();
    }
};

/// Device bytes of everything `load` packs.
pub fn total(g: geometry.Geometry) !usize {
    try validateGeometry(g);
    var t: usize = 0;
    try addProduct(&t, 2, try packedBytes(g.hidden, g.hidden));
    try addProduct(&t, 7, try checkedAligned(try checkedMul(@max(g.hidden, head_dim), 2)));
    const aw = try checkedMul(g.q_heads, head_dim);
    const kw = try checkedMul(g.kv_heads, head_dim);
    try addTo(&t, try packedBytes(try checkedMul(2, aw), g.hidden));
    try addProduct(&t, 2, try packedBytes(kw, g.hidden));
    try addTo(&t, try packedBytes(g.hidden, aw));
    if (g.moe()) {
        try addTo(&t, try routedBytes(g.experts, g.intermediate, g.hidden));
    } else {
        try addProduct(&t, 2, try packedBytes(g.intermediate, g.hidden));
        try addTo(&t, try packedBytes(g.hidden, g.intermediate));
    }
    return t;
}

/// The largest tensor `load` stages, as stored: fc's whole rows, an attention projection or an expert table.
pub fn stageBytes(g: geometry.Geometry) !usize {
    try validateGeometry(g);
    const aw = try checkedMul(g.q_heads, head_dim);
    var mlp: usize = undefined;
    if (g.moe()) {
        mlp = @max(
            try rawBytes(try checkedMul(g.experts, g.intermediate), g.hidden),
            try rawBytes(try checkedMul(g.experts, g.hidden), g.intermediate),
        );
    } else {
        mlp = @max(try rawBytes(g.intermediate, g.hidden), try rawBytes(g.hidden, g.intermediate));
    }
    return @max(
        @max(try rawBytes(g.hidden, try checkedMul(2, g.hidden)), try rawBytes(try checkedMul(2, aw), g.hidden)),
        mlp,
    );
}

/// Peak device bytes needed while loading the MTP weights. The arena and staging buffer coexist.
pub fn allocationPlan(gpa: std.mem.Allocator, io: std.Io, g: geometry.Geometry, dir: []const u8) !AllocationPlan {
    if (g.loop != null) return @import("cuda_mtp_dense.zig").allocationPlan(gpa, io, g, dir);
    const resident = try total(g);
    const staging = try stageBytes(g);
    if (hasFile(io, dir)) {
        _ = try sourceForTarget(g, true, false, false);
        return .{ .resident = resident, .staging = staging };
    }
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, "mtp.");
    defer core.checkpoint.cache.close(&ck);
    const indexed_head = ck.has("mtp.fc.weight");
    const indexed_scales = hasIndexedScales(&ck, g);
    _ = try sourceForTarget(g, false, indexed_head, indexed_scales);
    if (!g.moe()) try validateDenseBf16(&ck, g);
    return .{ .resident = resident, .staging = staging };
}

fn hasIndexedScales(ck: *const core.Checkpoint, g: geometry.Geometry) bool {
    const common = [_][]const u8{
        "mtp.fc.weight_scale",
        "mtp.layers.0.self_attn.q_proj.weight_scale",
        "mtp.layers.0.self_attn.k_proj.weight_scale",
        "mtp.layers.0.self_attn.v_proj.weight_scale",
        "mtp.layers.0.self_attn.o_proj.weight_scale",
    };
    for (common) |name| if (ck.has(name)) return true;
    if (!g.moe()) {
        for ([_][]const u8{
            "mtp.layers.0.mlp.gate_proj.weight_scale",
            "mtp.layers.0.mlp.up_proj.weight_scale",
            "mtp.layers.0.mlp.down_proj.weight_scale",
        }) |name| if (ck.has(name)) return true;
    }
    return false;
}

pub const DenseTensor = struct { name: []const u8, shape: [2]usize, rank: usize, split_fc: bool = false };

pub fn denseTensors(g: geometry.Geometry) [15]DenseTensor {
    const hidden = g.hidden;
    const aw = g.attnWidth();
    const kvw = g.kv_heads * head_dim;
    return .{
        .{ .name = "mtp.fc.weight", .shape = .{ hidden, 2 * hidden }, .rank = 2, .split_fc = true },
        .{ .name = "mtp.pre_fc_norm_embedding.weight", .shape = .{ hidden, 0 }, .rank = 1 },
        .{ .name = "mtp.pre_fc_norm_hidden.weight", .shape = .{ hidden, 0 }, .rank = 1 },
        .{ .name = "mtp.norm.weight", .shape = .{ hidden, 0 }, .rank = 1 },
        .{ .name = "mtp.layers.0.input_layernorm.weight", .shape = .{ hidden, 0 }, .rank = 1 },
        .{ .name = "mtp.layers.0.post_attention_layernorm.weight", .shape = .{ hidden, 0 }, .rank = 1 },
        .{ .name = "mtp.layers.0.self_attn.q_norm.weight", .shape = .{ head_dim, 0 }, .rank = 1 },
        .{ .name = "mtp.layers.0.self_attn.k_norm.weight", .shape = .{ head_dim, 0 }, .rank = 1 },
        .{ .name = "mtp.layers.0.self_attn.q_proj.weight", .shape = .{ 2 * aw, hidden }, .rank = 2 },
        .{ .name = "mtp.layers.0.self_attn.k_proj.weight", .shape = .{ kvw, hidden }, .rank = 2 },
        .{ .name = "mtp.layers.0.self_attn.v_proj.weight", .shape = .{ kvw, hidden }, .rank = 2 },
        .{ .name = "mtp.layers.0.self_attn.o_proj.weight", .shape = .{ hidden, aw }, .rank = 2 },
        .{ .name = "mtp.layers.0.mlp.gate_proj.weight", .shape = .{ g.intermediate, hidden }, .rank = 2 },
        .{ .name = "mtp.layers.0.mlp.up_proj.weight", .shape = .{ g.intermediate, hidden }, .rank = 2 },
        .{ .name = "mtp.layers.0.mlp.down_proj.weight", .shape = .{ hidden, g.intermediate }, .rank = 2 },
    };
}

/// Indexed dense MTP is supported only when every required tensor is BF16 and has the expected shape.
fn validateDenseBf16(ck: *core.Checkpoint, g: geometry.Geometry) !void {
    for (denseTensors(g)) |tensor| _ = try ck.expect(tensor.name, .bf16, tensor.shape[0..tensor.rank]);
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, e: *Engine, dir: []const u8) !Weights {
    const g = e.g;
    if (g.loop != null) return @import("cuda_mtp_dense.zig").load(gpa, io, e, dir);
    if (hasFile(io, dir)) {
        _ = try sourceForTarget(g, true, false, false);
    } else {
        {
            var indexed = try core.Checkpoint.openModelPrefix(gpa, io, dir, "mtp.");
            defer core.checkpoint.cache.close(&indexed);
            const indexed_head = indexed.has("mtp.fc.weight");
            const indexed_scales = hasIndexedScales(&indexed, g);
            _ = try sourceForTarget(g, false, indexed_head, indexed_scales);
        }
        return @import("cuda_mtp_bf16.zig").load(gpa, io, e, dir);
    }
    var ck: core.Checkpoint = .{ .gpa = gpa, .io = io };
    defer core.checkpoint.cache.close(&ck);
    try ck.add(dir, file);
    var w: Weights = .{ .arena = try DeviceBuffer.alloc(e.d, try total(g)), .fc_e = undefined, .fc_h = undefined, .norm_e = 0, .norm_h = 0, .input_norm = 0, .post_norm = 0, .q_norm = 0, .k_norm = 0, .norm = 0, .q = undefined, .k = undefined, .v = undefined, .o = undefined, .mlp = undefined };
    errdefer w.arena.free();
    try w.arena.fill8(0, null);
    var stage = try DeviceBuffer.alloc(e.d, try stageBytes(g)); // the load's staging, freed once the layer is packed
    defer stage.free();
    const m = "language_model.mtp.";
    const p = "language_model.mtp.layers.0.";
    w.fc_e = .{ .a4 = try weights.linearSlice(e, &w, &ck, &stage, m, "fc", g.hidden, 2 * g.hidden, 0, g.hidden) };
    w.fc_h = .{ .a4 = try weights.linearSlice(e, &w, &ck, &stage, m, "fc", g.hidden, 2 * g.hidden, g.hidden, g.hidden) };
    w.norm_e = try weights.tensor(&w, &ck, "{s}pre_fc_norm_embedding.weight", .{m}, .bf16, &.{g.hidden});
    w.norm_h = try weights.tensor(&w, &ck, "{s}pre_fc_norm_hidden.weight", .{m}, .bf16, &.{g.hidden});
    w.norm = try weights.tensor(&w, &ck, "{s}norm.weight", .{m}, .bf16, &.{g.hidden});
    w.input_norm = try weights.tensor(&w, &ck, "{s}input_layernorm.weight", .{p}, .bf16, &.{g.hidden});
    w.post_norm = try weights.tensor(&w, &ck, "{s}post_attention_layernorm.weight", .{p}, .bf16, &.{g.hidden});
    w.q_norm = try weights.tensor(&w, &ck, "{s}self_attn.q_norm.weight", .{p}, .bf16, &.{head_dim});
    w.k_norm = try weights.tensor(&w, &ck, "{s}self_attn.k_norm.weight", .{p}, .bf16, &.{head_dim});
    w.q = try weights.linear(e, &w, &ck, &stage, p, "self_attn.q_proj", 2 * g.attnWidth(), g.hidden, 4);
    w.k = try weights.linear(e, &w, &ck, &stage, p, "self_attn.k_proj", g.kv_heads * head_dim, g.hidden, 4);
    w.v = try weights.linear(e, &w, &ck, &stage, p, "self_attn.v_proj", g.kv_heads * head_dim, g.hidden, 4);
    w.o = try weights.linear(e, &w, &ck, &stage, p, "self_attn.o_proj", g.hidden, g.attnWidth(), 4);
    w.mlp = try weights.routed(e, &w, &ck, &stage, p, 4);
    return w;
}
