//! Read Inkling's published pack in place, preserving EXL3 bytes and the original dense tensor precision.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("exl3_config.zig");
const Tensor = core.checkpoint.Tensor;
const File = core.safetensors.File;

pub const Linear = struct { ptr: u64, k: usize, n: usize };
pub const Matrix = struct { ptrs: u64, suh: u64, svh: u64, bits: u64, k: usize, n: usize };
pub const Conv = struct { weight: u64, dynamic: ?Linear = null, width: usize };
pub const Layer = struct {
    norms: u64,
    mlp_norm: u64,
    attn_query: u64,
    mlp_query: u64,
    q: Linear,
    k: Linear,
    v: Linear,
    r: Linear,
    o: Linear,
    gate: Linear,
    sink: Linear,
    router: Linear,
    router_bias: u64,
    q_scale: u64,
    r_scale: u64,
    relative: u64,
    k_conv: Conv,
    v_conv: Conv,
    attn_conv: Conv,
    mlp_conv: Conv,
    w13: Matrix,
    w2: Matrix,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    share: cfg.Share,
    source_config: @import("config.zig").Config,
    layout: @import("plan.zig").Plan,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    layer: []Layer = &.{},
    embed: u64 = 0,
    embed_first: ?usize = null,
    embed_norm: u64 = 0,
    norm: u64 = 0,
    output_query: u64 = 0,
    head: Linear = undefined,
    bytes: usize = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layer);
        w.* = undefined;
    }

    pub fn put(w: *Weights, data: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(w.d, data);
        errdefer {
            var owned = b;
            owned.free();
        }
        try w.buffers.append(w.gpa, b);
        w.bytes += b.len;
        return b.ptr;
    }
};

pub fn get(f: *const File, name: []const u8, dtype: core.checkpoint.DType, shape: []const usize) !Tensor {
    const t = f.get(name) orelse return error.InklingMissingTensor;
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
        std.log.err("inkling2: {s} is not {s}{any}", .{ name, @tagName(dtype), shape });
        return error.InklingTensorShape;
    }
    return t;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !File {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    return File.open(gpa, io, path);
}

/// Cut a BF16 projection by output rows or input columns; the dense bytes themselves never change precision.
pub fn linear(w: *Weights, f: *const File, name: []const u8, n: usize, k: usize, rows: [2]usize, cols: [2]usize) !Linear {
    const t = try get(f, name, .bf16, &.{ n, k });
    if (rows[0] >= rows[1] or rows[1] > n or cols[0] >= cols[1] or cols[1] > k) return error.InklingTensorShape;
    const nn = rows[1] - rows[0];
    const kk = cols[1] - cols[0];
    if (kk == k) return .{ .ptr = try w.put(t.bytes[rows[0] * k * 2 .. rows[1] * k * 2]), .k = k, .n = nn };
    const bytes = try w.gpa.alloc(u8, nn * kk * 2);
    defer w.gpa.free(bytes);
    for (0..nn) |r| @memcpy(bytes[r * kk * 2 ..][0 .. kk * 2], t.bytes[((rows[0] + r) * k + cols[0]) * 2 ..][0 .. kk * 2]);
    return .{ .ptr = try w.put(bytes), .k = kk, .n = nn };
}

const DenseCheck = struct {
    file: *const File,
    prefix: ?[]const u8,

    pub fn put(c: *DenseCheck, name: []const u8, dtype: @import("header.zig").Dtype, dims: []const u64) !void {
        if (c.prefix) |p| {
            if (!std.mem.startsWith(u8, name, p)) return;
        } else if (!std.mem.startsWith(u8, name, "model.llm.") or std.mem.indexOf(u8, name, ".layers.") != null) return;
        const t = c.file.get(name) orelse return error.InklingMissingTensor;
        const kind = core.checkpoint.DType.parse(dtype.tag()) orelse return error.InklingTensorShape;
        if (t.dtype != kind or t.rank != dims.len) return error.InklingTensorShape;
        for (dims, 0..) |n, i| if (t.dim(i) != n) return error.InklingTensorShape;
    }
};

fn sharedDense(w: *const Weights, f: *const File, prefix: ?[]const u8) !void {
    const formats: [@import("config.zig").max_layers]@import("plan.zig").Precision = @splat(.exl3);
    var check: DenseCheck = .{ .file = f, .prefix = prefix };
    try @import("names.zig").all(&w.source_config, formats[0..w.source_config.text.layers], .model, &check);
}

const Loader = struct {
    w: *Weights,
    f: *const File,
    prefix: []const u8,
    buf: [256]u8 = undefined,

    fn name(l: *Loader, suffix: []const u8) ![]const u8 {
        return std.fmt.bufPrint(&l.buf, "{s}{s}", .{ l.prefix, suffix });
    }
    fn tensor(l: *Loader, suffix: []const u8, dtype: core.checkpoint.DType, shape: []const usize) !u64 {
        return l.w.put((try get(l.f, try l.name(suffix), dtype, shape)).bytes);
    }
    fn projection(l: *Loader, suffix: []const u8, n: usize, k: usize, rows: [2]usize, cols: [2]usize) !Linear {
        return linear(l.w, l.f, try l.name(suffix), n, k, rows, cols);
    }
    fn full(l: *Loader, suffix: []const u8, n: usize) !Linear {
        return l.projection(suffix, n, cfg.hidden, .{ 0, n }, .{ 0, cfg.hidden });
    }
    fn convolution(l: *Loader, suffix: []const u8, dynamic: bool, kv: bool) !Conv {
        var key: [160]u8 = undefined;
        const s = l.w.share;
        const stored_width: usize = if (kv) 1024 else cfg.hidden;
        const width = if (kv) stored_width / s.world else stored_width;
        const first = if (kv) s.rank * width else 0;
        const weight_name = try std.fmt.bufPrint(&key, "{s}.{s}", .{ suffix, if (dynamic) "static_w" else "weight" });
        const t = try get(l.f, try l.name(weight_name), .bf16, if (dynamic) &.{ 4, stored_width } else &.{ stored_width, 1, 4 });
        var result: Conv = .{ .weight = 0, .width = width };
        if (dynamic) {
            const bytes = try l.w.gpa.alloc(u8, 4 * width * 2);
            defer l.w.gpa.free(bytes);
            for (0..4) |j| @memcpy(bytes[j * width * 2 ..][0 .. width * 2], t.bytes[(j * stored_width + first) * 2 ..][0 .. width * 2]);
            result.weight = try l.w.put(bytes);
            const proj = try std.fmt.bufPrint(&key, "{s}.weight_proj", .{suffix});
            result.dynamic = try l.projection(proj, 32, cfg.hidden, .{ s.rank * 32 / s.world, (s.rank + 1) * 32 / s.world }, .{ 0, cfg.hidden });
        } else result.weight = try l.w.put(t.bytes[first * 8 .. (first + width) * 8]);
        return result;
    }
};

fn loadLayer(w: *Weights, io: std.Io, dir: []const u8, i: usize, globals: *const File) !Layer {
    var name: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&name, "dense/layer-{d:0>2}.safetensors", .{i});
    var f = try open(w.gpa, io, dir, path);
    defer f.closeCopied(io);
    var prefix: [96]u8 = undefined;
    var l: Loader = .{ .w = w, .f = &f, .prefix = try std.fmt.bufPrint(&prefix, "model.llm.layers.{d}.", .{i}) };
    try sharedDense(w, &f, l.prefix);
    const h = cfg.hidden;
    const s = w.share;
    const q0 = s.rank * s.queryHeads();
    const q1 = q0 + s.queryHeads();
    const k0 = s.rank * s.keyHeads();
    const k1 = k0 + s.keyHeads();
    const is_local = w.layout.at(@intCast(i)).kind == .local;
    var norms: std.ArrayList(u8) = .empty;
    defer norms.deinit(w.gpa);
    const norm_names: []const []const u8 = if (is_local) &.{"attn_norm.weight"} else &.{ "q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight" };
    for (norm_names) |n| try norms.appendSlice(w.gpa, (try get(&f, try l.name(n), .bf16, &.{h})).bytes);
    const mlp_q = try get(globals, try std.fmt.bufPrint(&name, "model.llm.attn_res.mlp_queries.{d}", .{i}), .f32, &.{h});
    var aq: u64 = 0;
    if (i > 0) {
        const t = try get(globals, try std.fmt.bufPrint(&name, "model.llm.attn_res.attn_queries.{d}", .{i - 1}), .f32, if (is_local) &.{h} else &.{ 3, h });
        aq = try w.put(t.bytes);
    }
    const qs = try get(&f, try l.name("attn.q_l2_clip.scale"), .bf16, &.{ 48, 128 });
    const rs = try get(&f, try l.name("attn.rel_l2_clip.scale"), .bf16, &.{ 48, 16 });
    return .{
        .norms = try w.put(norms.items),
        .mlp_norm = try l.tensor("mlp_norm.weight", .bf16, &.{h}),
        .attn_query = aq,
        .mlp_query = try w.put(mlp_q.bytes),
        .q = try l.projection("attn.wq_du.weight", 6144, h, .{ q0 * 128, q1 * 128 }, .{ 0, h }),
        .k = try l.projection("attn.wk_dv.weight", 1024, h, .{ k0 * 128, k1 * 128 }, .{ 0, h }),
        .v = try l.projection("attn.wv_dv.weight", 1024, h, .{ k0 * 128, k1 * 128 }, .{ 0, h }),
        .r = try l.projection("attn.wr_du.weight", 768, h, .{ q0 * 16, q1 * 16 }, .{ 0, h }),
        .o = try l.projection("attn.wo_ud.weight", h, 6144, .{ 0, h }, .{ q0 * 128, q1 * 128 }),
        .gate = try l.projection("attn.gate_attn.weight", 6144, h, .{ q0 * 128, q1 * 128 }, .{ 0, h }),
        .sink = try l.projection("attn.attn_sink_proj.weight", 48, h, .{ q0, q1 }, .{ 0, h }),
        .router = try l.full("mlp.gate.weight", cfg.experts),
        .router_bias = try l.tensor("mlp.gate.bias", .f32, &.{cfg.experts}),
        .q_scale = try w.put(qs.bytes[q0 * 128 * 2 .. q1 * 128 * 2]),
        .r_scale = try w.put(rs.bytes[q0 * 16 * 2 .. q1 * 16 * 2]),
        .relative = try l.tensor("attn.rel_logits_proj.proj", .bf16, &.{ 16, 1024 }),
        .k_conv = try l.convolution("attn.k_sconv", !is_local, true),
        .v_conv = try l.convolution("attn.v_sconv", !is_local, true),
        .attn_conv = try l.convolution("attn_sconv", false, false),
        .mlp_conv = try l.convolution("mlp_sconv", false, false),
        .w13 = undefined,
        .w2 = undefined,
    };
}

fn loadExperts(w: *Weights, io: std.Io, dir: []const u8, layer: usize, comptime proj: []const u8, k: usize, n: usize) !Matrix {
    var ptrs: std.ArrayList(u64) = .empty;
    defer ptrs.deinit(w.gpa);
    var bits: std.ArrayList(i32) = .empty;
    defer bits.deinit(w.gpa);
    var su: std.ArrayList(u8) = .empty;
    defer su.deinit(w.gpa);
    var sv: std.ArrayList(u8) = .empty;
    defer sv.deinit(w.gpa);
    var name: [128]u8 = undefined;
    const start = w.share.expertStart();
    const end = start + w.share.expertCount();
    var first = start;
    while (first < end) : (first += 8) {
        const path = try std.fmt.bufPrint(&name, "layers/{d:0>2}/experts-{d:0>3}-{d:0>3}.safetensors", .{ layer, first, first + 7 });
        var f = try open(w.gpa, io, dir, path);
        defer f.closeCopied(io);
        for (first..first + 8) |e| {
            const key = try std.fmt.bufPrint(&name, "experts.{d}." ++ proj ++ ".trellis", .{e});
            const t = f.get(key) orelse return error.InklingMissingTensor;
            if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != k / 16 or t.dim(1) != n / 16 or
                (t.dim(2) != 32 and t.dim(2) != 48)) return error.InklingExl3;
            try ptrs.append(w.gpa, try w.put(t.bytes));
            try bits.append(w.gpa, @intCast(t.dim(2) / 8));
            const marker = try get(&f, try std.fmt.bufPrint(&name, "experts.{d}." ++ proj ++ ".mul1", .{e}), .i32, &.{});
            if (std.mem.readInt(u32, marker.bytes[0..4], .little) != 0x83dcd12d) return error.InklingExl3;
            const suh = try get(&f, try std.fmt.bufPrint(&name, "experts.{d}." ++ proj ++ ".suh", .{e}), .f16, &.{k});
            try su.appendSlice(w.gpa, suh.bytes);
            const svh = try get(&f, try std.fmt.bufPrint(&name, "experts.{d}." ++ proj ++ ".svh", .{e}), .f16, &.{n});
            try sv.appendSlice(w.gpa, svh.bytes);
        }
    }
    return .{ .ptrs = try w.put(std.mem.sliceAsBytes(ptrs.items)), .bits = try w.put(std.mem.sliceAsBytes(bits.items)), .suh = try w.put(su.items), .svh = try w.put(sv.items), .k = k, .n = n };
}

/// `count` is 48 when serving; the independent layer dump tool may load a shorter prefix for a bounded canary.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, share: cfg.Share, count: usize) !Weights {
    return loadProfile(gpa, io, d, dir, share, count, false);
}

pub fn loadProfile(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, share: cfg.Share, count: usize, drafts: bool) !Weights {
    if (count == 0 or count > cfg.layers) return error.InklingGeometry;
    const path = try std.fs.path.join(gpa, &.{ dir, "checkpoint.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    defer gpa.free(text);
    const source_path = try std.fs.path.join(gpa, &.{ dir, "auxiliary/config.json" });
    defer gpa.free(source_path);
    const source_bytes = try std.Io.Dir.cwd().readFileAlloc(io, source_path, gpa, .limited(1 << 24));
    defer gpa.free(source_bytes);
    const source_config = try cfg.admitSource(gpa, text, source_bytes);
    var w: Weights = .{ .gpa = gpa, .d = d, .share = share, .source_config = source_config, .layout = try cfg.layerPlan(&source_config) };
    errdefer w.deinit();
    var globals = try open(gpa, io, dir, "dense/global.safetensors");
    defer globals.closeCopied(io);
    try sharedDense(&w, &globals, null);
    const embed = try get(&globals, "model.llm.embed.weight", .bf16, &.{ cfg.vocab, cfg.hidden });
    if (drafts) {
        const first = share.rank * cfg.vocab / share.world;
        const end = (share.rank + 1) * cfg.vocab / share.world;
        w.embed_first = first;
        w.embed = try w.put(embed.bytes[first * cfg.hidden * 2 .. end * cfg.hidden * 2]);
    } else w.embed = try w.put(embed.bytes);
    w.embed_norm = try w.put((try get(&globals, "model.llm.embed_norm.weight", .bf16, &.{cfg.hidden})).bytes);
    w.norm = try w.put((try get(&globals, "model.llm.norm.weight", .bf16, &.{cfg.hidden})).bytes);
    w.output_query = try w.put((try get(&globals, "model.llm.attn_res.output_query", .f32, &.{cfg.hidden})).bytes);
    const vn = cfg.vocab / share.world;
    w.head = try linear(&w, &globals, "model.llm.unembed.weight", cfg.vocab, cfg.hidden, .{ share.rank * vn, (share.rank + 1) * vn }, .{ 0, cfg.hidden });
    w.layer = try gpa.alloc(Layer, count);
    for (w.layer, 0..) |*l, i| {
        l.* = try loadLayer(&w, io, dir, i, &globals);
        l.w13 = try loadExperts(&w, io, dir, i, "w13", cfg.hidden, cfg.hidden);
        l.w2 = try loadExperts(&w, io, dir, i, "w2", cfg.intermediate, cfg.hidden);
        std.log.info("inkling2 rank {d}: layer {d}/{d}, {d:.3} GiB loaded", .{ share.rank, i + 1, count, @as(f64, @floatFromInt(w.bytes)) / (1 << 30) });
    }
    return w;
}
