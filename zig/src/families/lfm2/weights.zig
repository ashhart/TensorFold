//! LFM2 weights on the GPU: the checkpoint's bf16 tensors in one allocation, each layer's projections stacked.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");

const Config = config.Config;
const Kind = config.Kind;
const Header = core.safetensors.Header;
const dio = core.direct_io;

/// Bytes a staging read moves at once.
const staging_bytes = 64 << 20;

pub const Layer = struct {
    kind: Kind,
    op_norm: u64,
    ffn_norm: u64,
    up: u64, // [2 ffn, hidden]: w1 rows, then w3 rows
    down: u64,
    proj: u64, // conv: in_proj [3 hidden, hidden]; attention: q, k, v rows [(heads + 2 kv_heads) dim, hidden]
    conv: u64 = 0, // [hidden, taps], conv layers only
    out: u64,
    q_norm: u64 = 0, // attention layers only
    k_norm: u64 = 0,
};

/// A SigLIP2 encoder layer; `qkv` stacks q, k, v rows as `Layer.proj` does, and `qkv_bias` their biases.
pub const VisionLayer = struct {
    norm1: u64,
    norm1_bias: u64,
    qkv: u64,
    qkv_bias: u64,
    out: u64,
    out_bias: u64,
    norm2: u64,
    norm2_bias: u64,
    fc1: u64,
    fc1_bias: u64,
    fc2: u64,
    fc2_bias: u64,
};

/// The SigLIP2 tower (patch embedding as a linear over channels x patch x patch) and multi_modal_projector.
pub const VisionWeights = struct {
    patch: u64,
    patch_bias: u64,
    position: u64,
    layers: []VisionLayer,
    post_norm: u64,
    post_norm_bias: u64,
    linear_1: u64,
    linear_1_bias: u64,
    linear_2: u64,
    linear_2_bias: u64,
};

/// One checkpoint tensor's copy: `src` counts from the safetensors data region, `dst` from the allocation's start.
pub const Piece = struct { name: []const u8, src: usize, len: usize, dst: u64 };

/// Weights with offsets from the allocation's start in place of addresses, and the copies that fill it.
pub const Plan = struct {
    embed: u64 = 0,
    norm: u64 = 0,
    layers: []Layer = &.{},
    vision: ?VisionWeights = null,
    bytes: u64 = 0,
    pieces: std.ArrayList(Piece) = .empty,

    pub fn deinit(p: *Plan, a: std.mem.Allocator) void {
        a.free(p.layers);
        if (p.vision) |v| a.free(v.layers);
        p.pieces.deinit(a);
        p.* = undefined;
    }
};

pub const Weights = struct {
    embed: u64,
    norm: u64,
    layers: []Layer,
    vision: ?VisionWeights,
    bytes: u64,
    memory: cuda.DeviceBuffer,

    /// `dir`'s model.safetensors on the device as `plan` lays it out; a refusal is logged with its reason.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, cfg: *const Config) !Weights {
        const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
        defer gpa.free(path);
        var file = try core.safetensors.File.open(gpa, io, path);
        defer file.close(io);
        var problem: []const u8 = "";
        var p = try plan(gpa, cfg, &file.names, &problem) orelse {
            defer gpa.free(problem);
            std.log.err("{s}", .{problem});
            return error.UnreadableLfm2Checkpoint;
        };
        errdefer p.deinit(gpa);
        var memory = try cuda.DeviceBuffer.alloc(d, p.bytes);
        errdefer memory.free();
        try copy(gpa, file.path, file.data, p.pieces.items, memory);
        p.pieces.deinit(gpa);
        const base = memory.ptr;
        for (p.layers) |*l| place(l, base);
        if (p.vision) |*v| {
            shift(v, base);
            for (v.layers) |*l| shift(l, base);
        }
        return .{ .embed = base + p.embed, .norm = base + p.norm, .layers = p.layers, .vision = p.vision, .bytes = p.bytes, .memory = memory };
    }

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        w.memory.free();
        gpa.free(w.layers);
        if (w.vision) |v| gpa.free(v.layers);
        w.* = undefined;
    }
};

/// Each piece read past the page cache into a staging buffer, then copied to its place.
fn copy(gpa: std.mem.Allocator, path: [:0]const u8, data: usize, pieces: []const Piece, memory: cuda.DeviceBuffer) !void {
    var file = try dio.File.open(path);
    defer file.close();
    const staging = try gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), staging_bytes);
    defer gpa.free(staging);
    for (pieces) |piece| {
        var done: usize = 0;
        while (done < piece.len) {
            const n = @min(piece.len - done, dio.fits(staging_bytes));
            try memory.upload(piece.dst + done, try file.read(staging, data + piece.src + done, n));
            done += n;
        }
    }
}

fn place(l: *Layer, base: u64) void {
    for ([_]*u64{ &l.op_norm, &l.ffn_norm, &l.up, &l.down, &l.proj, &l.out }) |f| f.* += base;
    switch (l.kind) {
        .conv => l.conv += base,
        .attention => for ([_]*u64{ &l.q_norm, &l.k_norm }) |f| {
            f.* += base;
        },
    }
}

/// Every u64 field of `s` moved by `base`.
fn shift(s: anytype, base: u64) void {
    const info = @typeInfo(@TypeOf(s.*)).@"struct";
    inline for (info.field_names, info.field_types) |name, T| if (T == u64) {
        @field(s, name) += base;
    };
}

const language = [_][]const u8{ "model.language_model.", "model.layers.", "model.embed" };
const projector = [_][]const u8{"model.multi_modal_projector."};

/// Where every tensor `cfg` reads goes (host only), or null with `problem` (from `a`) saying why the header is refused.
pub fn plan(a: std.mem.Allocator, cfg: *const Config, header: *const Header, problem: *[]const u8) !?Plan {
    var p: Planner = .{ .a = a, .header = header, .problem = problem };
    p.build(cfg) catch |e| {
        p.out.deinit(a);
        return if (e == error.Refused) null else e;
    };
    return p.out;
}

const Planner = struct {
    a: std.mem.Allocator,
    header: *const Header,
    problem: *[]const u8,
    out: Plan = .{},
    part: enum { language, vision } = .language,
    prefix_buf: [128]u8 = undefined,
    prefix: []const u8 = "",

    fn build(p: *Planner, cfg: *const Config) !void {
        if (p.count(&language) == 0) return p.refuse("no language-model tensors in this checkpoint (a complete d1 checkpoint is needed)", .{});
        const pre = if (p.count(&.{"model.language_model."}) != 0) "model.language_model." else "model.";
        const h, const f = .{ cfg.hidden, cfg.ffn };
        try p.within("{s}", .{pre});
        p.out.embed = try p.one("embed_tokens.weight", &.{ cfg.vocab, h });
        p.out.norm = try p.one("embedding_norm.weight", &.{h});
        p.out.layers = try p.a.alloc(Layer, cfg.layers.len);
        for (p.out.layers, cfg.layers, 0..) |*l, kind, i| {
            try p.within("{s}layers.{d}.", .{ pre, i });
            const op_norm = try p.one("operator_norm.weight", &.{h});
            const ffn_norm = try p.one("ffn_norm.weight", &.{h});
            const up = try p.stack(&.{ .{ "feed_forward.w1.weight", &.{ f, h } }, .{ "feed_forward.w3.weight", &.{ f, h } } });
            const down = try p.one("feed_forward.w2.weight", &.{ h, f });
            l.* = .{ .kind = kind, .op_norm = op_norm, .ffn_norm = ffn_norm, .up = up, .down = down, .proj = undefined, .out = undefined };
            switch (kind) {
                .conv => {
                    l.proj = try p.one("conv.in_proj.weight", &.{ 3 * h, h });
                    l.out = try p.one("conv.out_proj.weight", &.{ h, h });
                    l.conv = try p.one("conv.conv.weight", &.{ h, 1, cfg.taps });
                },
                .attention => {
                    const q, const kv = .{ cfg.heads * cfg.dim, cfg.kv_heads * cfg.dim };
                    l.proj = try p.stack(&.{ .{ "self_attn.q_proj.weight", &.{ q, h } }, .{ "self_attn.k_proj.weight", &.{ kv, h } }, .{ "self_attn.v_proj.weight", &.{ kv, h } } });
                    l.out = try p.one("self_attn.out_proj.weight", &.{ h, q });
                    l.q_norm = try p.one("self_attn.q_layernorm.weight", &.{cfg.dim});
                    l.k_norm = try p.one("self_attn.k_layernorm.weight", &.{cfg.dim});
                },
            }
        }
        try p.unread(&language, p.out.pieces.items.len, "language tensor(s) the LFM2 kernels do not read");
        if (cfg.vision) |v| p.out.vision = try p.vision(cfg, v);
    }

    fn vision(p: *Planner, cfg: *const Config, v: config.Vision) !VisionWeights {
        p.part = .vision;
        for ([_][]const u8{ "model.vision_tower.", projector[0] }) |prefix| for (p.header.keys(), p.header.values()) |k, e| {
            if (std.mem.startsWith(u8, k, prefix) and e.dtype != .bf16) return p.refuse("{s} is stored as {t}, and the LFM2-VL tower reads bf16 weights", .{ k[prefix.len..], e.dtype });
        };
        if (p.count(&.{"model.vision_tower."}) == 0) return p.refuse("no vision tower tensors in this checkpoint, which its config's vision_config needs", .{});
        const tower = if (p.count(&.{"model.vision_tower.vision_model."}) != 0) "model.vision_tower.vision_model." else "model.vision_tower.";
        const h, const f = .{ v.hidden, v.ffn };
        try p.within("{s}", .{tower});
        const patch = try p.one("embeddings.patch_embedding.weight", &.{ h, v.channels * v.patch * v.patch });
        const patch_bias = try p.one("embeddings.patch_embedding.bias", &.{h});
        const position = try p.one("embeddings.position_embedding.weight", &.{ v.num_patches, h });
        const layers = try p.a.alloc(VisionLayer, v.layers);
        errdefer p.a.free(layers);
        for (layers, 0..) |*l, i| {
            try p.within("{s}encoder.layers.{d}.", .{ tower, i });
            l.norm1 = try p.one("layer_norm1.weight", &.{h});
            l.norm1_bias = try p.one("layer_norm1.bias", &.{h});
            l.qkv = try p.stack(&.{ .{ "self_attn.q_proj.weight", &.{ h, h } }, .{ "self_attn.k_proj.weight", &.{ h, h } }, .{ "self_attn.v_proj.weight", &.{ h, h } } });
            l.qkv_bias = try p.stack(&.{ .{ "self_attn.q_proj.bias", &.{h} }, .{ "self_attn.k_proj.bias", &.{h} }, .{ "self_attn.v_proj.bias", &.{h} } });
            l.out = try p.one("self_attn.out_proj.weight", &.{ h, h });
            l.out_bias = try p.one("self_attn.out_proj.bias", &.{h});
            l.norm2 = try p.one("layer_norm2.weight", &.{h});
            l.norm2_bias = try p.one("layer_norm2.bias", &.{h});
            l.fc1 = try p.one("mlp.fc1.weight", &.{ f, h });
            l.fc1_bias = try p.one("mlp.fc1.bias", &.{f});
            l.fc2 = try p.one("mlp.fc2.weight", &.{ h, f });
            l.fc2_bias = try p.one("mlp.fc2.bias", &.{h});
        }
        try p.within("{s}", .{tower});
        const post_norm = try p.one("post_layernorm.weight", &.{h});
        const post_norm_bias = try p.one("post_layernorm.bias", &.{h});
        try p.within("{s}", .{projector[0]});
        const taken = p.out.pieces.items.len;
        const merged = h * v.downsample * v.downsample;
        const w: VisionWeights = .{
            .patch = patch,
            .patch_bias = patch_bias,
            .position = position,
            .layers = layers,
            .post_norm = post_norm,
            .post_norm_bias = post_norm_bias,
            .linear_1 = try p.one("linear_1.weight", &.{ v.projector_hidden, merged }),
            .linear_1_bias = try p.one("linear_1.bias", &.{v.projector_hidden}),
            .linear_2 = try p.one("linear_2.weight", &.{ cfg.hidden, v.projector_hidden }),
            .linear_2_bias = try p.one("linear_2.bias", &.{cfg.hidden}),
        };
        try p.unread(&projector, p.out.pieces.items.len - taken, "projector tensor(s) the LFM2-VL tower does not read");
        return w;
    }

    fn within(p: *Planner, comptime fmt: []const u8, args: anytype) !void {
        p.prefix = try std.fmt.bufPrint(&p.prefix_buf, fmt, args);
    }

    fn one(p: *Planner, suffix: []const u8, shape: []const usize) !u64 {
        return p.stack(&.{.{ suffix, shape }});
    }

    /// The offset of `parts` laid back to back from the next 256-byte boundary, each refused unless bf16 of its shape.
    fn stack(p: *Planner, parts: []const struct { []const u8, []const usize }) !u64 {
        const at = std.mem.alignForward(u64, p.out.bytes, 256);
        p.out.bytes = at;
        for (parts) |part| {
            var buf: [192]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ p.prefix, part[0] });
            const entry = p.header.getEntry(name) orelse return switch (p.part) {
                .language => p.refuse("the checkpoint has no {s}, which an LFM2 forward needs", .{name}),
                .vision => p.refuse("the checkpoint has no {s}, which the LFM2-VL tower needs", .{name}),
            };
            const e = entry.value_ptr.*;
            if (e.dtype != .bf16) return p.refuse("{s} is stored as {t}, and LFM2 d1 reads bf16 weights", .{ name, e.dtype });
            if (!std.mem.eql(usize, e.shape[0..e.rank], part[1])) return p.refuse("{s} is shaped {any}, and the config needs {any}", .{ name, e.shape[0..e.rank], part[1] });
            try p.out.pieces.append(p.a, .{ .name = entry.key_ptr.*, .src = e.begin, .len = e.end - e.begin, .dst = p.out.bytes });
            p.out.bytes += e.end - e.begin;
        }
        return at;
    }

    /// Header names under any of `prefixes`.
    fn count(p: *const Planner, prefixes: []const []const u8) usize {
        var n: usize = 0;
        for (p.header.keys()) |k| n += @intFromBool(under(k, prefixes));
        return n;
    }

    /// Refuses names under `prefixes` beyond the `taken` the plan copies, naming the first four in name order.
    fn unread(p: *Planner, prefixes: []const []const u8, taken: usize, comptime what: []const u8) !void {
        const n = p.count(prefixes);
        if (n == taken) return;
        var left: std.ArrayList([]const u8) = .empty;
        defer left.deinit(p.a);
        for (p.header.keys()) |k| if (under(k, prefixes) and !p.copies(k)) try left.append(p.a, k);
        std.mem.sort([]const u8, left.items, {}, lessThan);
        const shown = try std.mem.join(p.a, ", ", left.items[0..@min(4, left.items.len)]);
        defer p.a.free(shown);
        return p.refuse("{d} " ++ what ++ ": {s}", .{ left.items.len, shown });
    }

    fn copies(p: *const Planner, name: []const u8) bool {
        for (p.out.pieces.items) |piece| if (piece.name.ptr == name.ptr) return true;
        return false;
    }

    fn refuse(p: *Planner, comptime fmt: []const u8, args: anytype) error{ Refused, OutOfMemory } {
        p.problem.* = try std.fmt.allocPrint(p.a, fmt, args);
        return error.Refused;
    }
};

fn under(name: []const u8, prefixes: []const []const u8) bool {
    for (prefixes) |prefix| if (std.mem.startsWith(u8, name, prefix)) return true;
    return false;
}

fn lessThan(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.lessThan(u8, x, y);
}

const tiny_vision: config.Vision = .{ .hidden = 4, .ffn = 6, .layers = 1, .heads = 2, .channels = 3, .patch = 2, .num_patches = 4, .eps = 1e-6, .downsample = 2, .projector_hidden = 5 };
const tiny: Config = .{ .hidden = 4, .ffn = 6, .layers = &.{ .conv, .attention }, .heads = 2, .kv_heads = 1, .dim = 2, .taps = 3, .eps = 1e-5, .theta = 1e6, .vocab = 8, .bos = 1, .eos = 2, .image_token = 7, .vision = tiny_vision };

const Fake = struct { []const u8, []const usize };

/// The tiny model's checkpoint tensors, a stack's parts out of plan order (w3 before w1, v before k before q).
const tiny_tensors = [_]Fake{
    .{ "model.language_model.layers.0.feed_forward.w3.weight", &.{ 6, 4 } },
    .{ "model.language_model.embed_tokens.weight", &.{ 8, 4 } },
    .{ "model.language_model.embedding_norm.weight", &.{4} },
    .{ "model.language_model.layers.0.operator_norm.weight", &.{4} },
    .{ "model.language_model.layers.0.ffn_norm.weight", &.{4} },
    .{ "model.language_model.layers.0.feed_forward.w1.weight", &.{ 6, 4 } },
    .{ "model.language_model.layers.0.feed_forward.w2.weight", &.{ 4, 6 } },
    .{ "model.language_model.layers.0.conv.in_proj.weight", &.{ 12, 4 } },
    .{ "model.language_model.layers.0.conv.out_proj.weight", &.{ 4, 4 } },
    .{ "model.language_model.layers.0.conv.conv.weight", &.{ 4, 1, 3 } },
    .{ "model.language_model.layers.1.operator_norm.weight", &.{4} },
    .{ "model.language_model.layers.1.ffn_norm.weight", &.{4} },
    .{ "model.language_model.layers.1.feed_forward.w1.weight", &.{ 6, 4 } },
    .{ "model.language_model.layers.1.feed_forward.w3.weight", &.{ 6, 4 } },
    .{ "model.language_model.layers.1.feed_forward.w2.weight", &.{ 4, 6 } },
    .{ "model.language_model.layers.1.self_attn.v_proj.weight", &.{ 2, 4 } },
    .{ "model.language_model.layers.1.self_attn.k_proj.weight", &.{ 2, 4 } },
    .{ "model.language_model.layers.1.self_attn.q_proj.weight", &.{ 4, 4 } },
    .{ "model.language_model.layers.1.self_attn.out_proj.weight", &.{ 4, 4 } },
    .{ "model.language_model.layers.1.self_attn.q_layernorm.weight", &.{2} },
    .{ "model.language_model.layers.1.self_attn.k_layernorm.weight", &.{2} },
    .{ "model.vision_tower.vision_model.embeddings.patch_embedding.weight", &.{ 4, 12 } },
    .{ "model.vision_tower.vision_model.embeddings.patch_embedding.bias", &.{4} },
    .{ "model.vision_tower.vision_model.embeddings.position_embedding.weight", &.{ 4, 4 } },
    .{ "model.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight", &.{4} },
    .{ "model.vision_tower.vision_model.encoder.layers.0.layer_norm1.bias", &.{4} },
    .{ "model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.weight", &.{ 4, 4 } },
    .{ "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight", &.{ 4, 4 } },
    .{ "model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.weight", &.{ 4, 4 } },
    .{ "model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.bias", &.{4} },
    .{ "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.bias", &.{4} },
    .{ "model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.bias", &.{4} },
    .{ "model.vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.weight", &.{ 4, 4 } },
    .{ "model.vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.bias", &.{4} },
    .{ "model.vision_tower.vision_model.encoder.layers.0.layer_norm2.weight", &.{4} },
    .{ "model.vision_tower.vision_model.encoder.layers.0.layer_norm2.bias", &.{4} },
    .{ "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight", &.{ 6, 4 } },
    .{ "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.bias", &.{6} },
    .{ "model.vision_tower.vision_model.encoder.layers.0.mlp.fc2.weight", &.{ 4, 6 } },
    .{ "model.vision_tower.vision_model.encoder.layers.0.mlp.fc2.bias", &.{4} },
    .{ "model.vision_tower.vision_model.post_layernorm.weight", &.{4} },
    .{ "model.vision_tower.vision_model.post_layernorm.bias", &.{4} },
    .{ "model.multi_modal_projector.linear_1.weight", &.{ 5, 16 } },
    .{ "model.multi_modal_projector.linear_1.bias", &.{5} },
    .{ "model.multi_modal_projector.linear_2.weight", &.{ 4, 5 } },
    .{ "model.multi_modal_projector.linear_2.bias", &.{4} },
};

/// A safetensors header of `tensors` back to back, bf16 but for `f32_name`.
fn fakeHeader(a: std.mem.Allocator, tensors: []const Fake, f32_name: ?[]const u8) !Header {
    var json: std.Io.Writer.Allocating = .init(a);
    var at: usize = 0;
    for (tensors, 0..) |t, i| {
        const wide = f32_name != null and std.mem.eql(u8, t[0], f32_name.?);
        var n: usize = if (wide) 4 else 2;
        for (t[1]) |d| n *= d;
        try json.writer.print("{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (i == 0) "{" else ",", t[0], if (wide) "F32" else "BF16" });
        for (t[1], 0..) |d, j| try json.writer.print("{s}{d}", .{ if (j == 0) "" else ",", d });
        try json.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ at, at + n });
        at += n;
    }
    try json.writer.writeByte('}');
    return core.safetensors.parseHeader(a, json.written(), at);
}

fn pieceOf(p: Plan, name: []const u8) Piece {
    for (p.pieces.items) |piece| if (std.mem.eql(u8, piece.name, name)) return piece;
    unreachable;
}

test "plan stacks w1;w3 and q;k;v rows back to back on 256-byte boundaries, whatever the file order" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const header = try fakeHeader(a, &tiny_tensors, null);
    var problem: []const u8 = "";
    const p = (try plan(a, &tiny, &header, &problem)).?;
    try std.testing.expectEqual(tiny_tensors.len, p.pieces.items.len);
    try std.testing.expectEqualSlices(u64, &.{ 0, 256 }, &.{ p.embed, p.norm });
    const conv, const attention = .{ p.layers[0], p.layers[1] };
    const w1, const w3 = .{ pieceOf(p, tiny_tensors[5][0]), pieceOf(p, tiny_tensors[0][0]) };
    try std.testing.expectEqualSlices(u64, &.{ conv.up, conv.up + 48 }, &.{ w1.dst, w3.dst });
    try std.testing.expectEqualSlices(usize, &.{ header.get(tiny_tensors[5][0]).?.begin, 0 }, &.{ w1.src, w3.src });
    const q, const k, const v = .{ pieceOf(p, tiny_tensors[17][0]), pieceOf(p, tiny_tensors[16][0]), pieceOf(p, tiny_tensors[15][0]) };
    try std.testing.expectEqualSlices(u64, &.{ attention.proj, attention.proj + 32, attention.proj + 48 }, &.{ q.dst, k.dst, v.dst });
    try std.testing.expectEqualSlices(u64, &.{ 0, 0, 0 }, &.{ conv.q_norm, conv.k_norm, attention.conv });
    try std.testing.expect(conv.conv != 0 and attention.q_norm != 0 and attention.k_norm != 0);
    const vision = p.vision.?;
    const vq = pieceOf(p, tiny_tensors[27][0]);
    const vk_bias = pieceOf(p, tiny_tensors[31][0]);
    try std.testing.expectEqualSlices(u64, &.{ vision.layers[0].qkv, vision.layers[0].qkv_bias + 8 }, &.{ vq.dst, vk_bias.dst });
    for ([_]u64{ conv.up, conv.down, conv.proj, conv.conv, attention.proj, attention.k_norm, vision.layers[0].qkv, vision.linear_2_bias }) |at| try std.testing.expectEqual(0, at % 256);
    var end: u64 = 0;
    for (p.pieces.items) |piece| {
        try std.testing.expect(piece.dst >= end);
        try std.testing.expectEqual(header.get(piece.name).?.end - header.get(piece.name).?.begin, piece.len);
        end = piece.dst + piece.len;
    }
    try std.testing.expectEqual(end, p.bytes);
    const base: u64 = 1 << 40;
    var placed = conv;
    place(&placed, base);
    try std.testing.expectEqualSlices(u64, &.{ conv.up + base, conv.conv + base, 0, 0 }, &.{ placed.up, placed.conv, placed.q_norm, placed.k_norm });
    var tower = vision;
    shift(&tower, base);
    var layer = vision.layers[0];
    shift(&layer, base);
    try std.testing.expectEqualSlices(u64, &.{ vision.patch + base, vision.linear_2_bias + base, vision.layers[0].fc2_bias + base }, &.{ tower.patch, tower.linear_2_bias, layer.fc2_bias });
}

test "a text-only checkpoint's model. names plan without a vision tower" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: std.ArrayList(Fake) = .empty;
    for (tiny_tensors[0..21]) |t| try text.append(a, .{ try std.mem.replaceOwned(u8, a, t[0], "model.language_model.", "model."), t[1] });
    var cfg = tiny;
    cfg.vision = null;
    var problem: []const u8 = "";
    const p = (try plan(a, &cfg, &try fakeHeader(a, text.items, null), &problem)).?;
    try std.testing.expect(p.pieces.items.len == 21 and p.vision == null);
    try std.testing.expectEqualStrings("model.embed_tokens.weight", p.pieces.items[0].name);
}

/// The refusal `plan` gives for the tiny tensors less `drop`, plus `add`, with `f32_name` stored as f32.
fn refusal(a: std.mem.Allocator, drop: ?[]const u8, add: ?Fake, f32_name: ?[]const u8) ![]const u8 {
    var tensors: std.ArrayList(Fake) = .empty;
    for (tiny_tensors) |t| if (drop == null or !std.mem.eql(u8, t[0], drop.?)) try tensors.append(a, t);
    if (add) |t| try tensors.append(a, t);
    var problem: []const u8 = "";
    try std.testing.expect(try plan(a, &tiny, &try fakeHeader(a, tensors.items, f32_name), &problem) == null);
    return problem;
}

test "plan refuses a missing, mistyped, misshapen or unread tensor by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const k_proj = "model.language_model.layers.1.self_attn.k_proj.weight";
    try std.testing.expectEqualStrings("the checkpoint has no " ++ k_proj ++ ", which an LFM2 forward needs", try refusal(a, k_proj, null, null));
    const norm = "model.language_model.embedding_norm.weight";
    try std.testing.expectEqualStrings(norm ++ " is stored as f32, and LFM2 d1 reads bf16 weights", try refusal(a, null, null, norm));
    const conv = "model.language_model.layers.0.conv.conv.weight";
    try std.testing.expectEqualStrings(conv ++ " is shaped { 4, 3 }, and the config needs { 4, 1, 3 }", try refusal(a, conv, .{ conv, &.{ 4, 3 } }, null));
    const head = "model.language_model.lm_head.weight";
    try std.testing.expectEqualStrings("1 language tensor(s) the LFM2 kernels do not read: " ++ head, try refusal(a, null, .{ head, &.{ 8, 4 } }, null));
    const fc2_bias = "model.vision_tower.vision_model.encoder.layers.0.mlp.fc2.bias";
    try std.testing.expectEqualStrings("the checkpoint has no " ++ fc2_bias ++ ", which the LFM2-VL tower needs", try refusal(a, fc2_bias, null, null));
    try std.testing.expectEqualStrings("vision_model.encoder.layers.0.mlp.fc2.bias is stored as f32, and the LFM2-VL tower reads bf16 weights", try refusal(a, null, null, fc2_bias));
    const probe = "model.vision_tower.vision_model.head.probe.weight";
    try std.testing.expectEqualStrings("vision_model.head.probe.weight is stored as f32, and the LFM2-VL tower reads bf16 weights", try refusal(a, null, .{ probe, &.{4} }, probe));
    const projector_norm = "model.multi_modal_projector.layer_norm.weight";
    try std.testing.expectEqualStrings("1 projector tensor(s) the LFM2-VL tower does not read: " ++ projector_norm, try refusal(a, null, .{ projector_norm, &.{16} }, null));
    try std.testing.expectEqualStrings("layer_norm.weight is stored as f32, and the LFM2-VL tower reads bf16 weights", try refusal(a, null, .{ projector_norm, &.{16} }, projector_norm));
    var problem: []const u8 = "";
    try std.testing.expect(try plan(a, &tiny, &try fakeHeader(a, tiny_tensors[21..], null), &problem) == null);
    try std.testing.expectEqualStrings("no language-model tensors in this checkpoint (a complete d1 checkpoint is needed)", problem);
    try std.testing.expect(try plan(a, &tiny, &try fakeHeader(a, tiny_tensors[0..21], null), &problem) == null);
    try std.testing.expectEqualStrings("no vision tower tensors in this checkpoint, which its config's vision_config needs", problem);
}
