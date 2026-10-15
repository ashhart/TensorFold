//! The RADIO tower's and projector's tensors as stored (bf16) in one device allocation; the position table on the host.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Config = @import("config.zig").Config;
const Allocator = std.mem.Allocator;
const st = core.safetensors;

const tower = "vision_model.radio_model.model.";
/// Every name the image side owns; the rest of these prefixes (video, the conditioner, summary indices) is left unread.
const prefixes = [_][]const u8{ "vision_model.", "vision_projector.", "mlp1." };

pub const Layer = struct {
    norm1: u64,
    norm1_bias: u64,
    qkv: u64, // [3 hidden, hidden]: q rows, then k, then v, each head by head
    qkv_bias: u64,
    proj: u64,
    proj_bias: u64,
    norm2: u64,
    norm2_bias: u64,
    fc1: u64,
    fc1_bias: u64,
    fc2: u64,
    fc2_bias: u64,
};

pub const Weights = struct {
    memory: cuda.DeviceBuffer,
    patch: u64, // [hidden, 3 * patch * patch], no bias
    cls: u64, // [prefix, hidden]: cls tokens, then registers
    final: u64, // vision_final_layernorm (unset without an MTP layer: Config.final_norm)
    final_bias: u64,
    rms: u64, // the projector's RMSNorm weight as f32 [4 hidden]
    lin1: u64, // [projector_hidden, 4 hidden]
    lin2: u64, // [text_hidden, projector_hidden]
    layers: []Layer,
    table: []f32, // the position table [side, side, hidden]
    bytes: u64,

    /// `dir`'s image tensors on the device (the shards the index names for them), checked against `c`.
    pub fn load(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: *const Config) !Weights {
        var files = try open(gpa, io, dir);
        defer {
            for (files.items) |*f| {
                f.release();
                f.close(io);
            }
            files.deinit(gpa);
        }
        var p: Planner = .{ .gpa = gpa, .files = files.items, .c = c };
        defer p.pieces.deinit(gpa);
        const layers = try gpa.alloc(Layer, c.layers);
        errdefer gpa.free(layers);
        var w: Weights = undefined;
        w.layers = layers;
        try p.plan(&w);
        var memory = try cuda.DeviceBuffer.alloc(d, p.bytes);
        errdefer memory.free();
        for (p.pieces.items) |piece| try upload(gpa, memory, piece);
        const table = try gpa.alloc(f32, c.side * c.side * c.hidden);
        errdefer gpa.free(table);
        const pos = try p.get(tower ++ "patch_generator.pos_embed", &.{ 1, c.side * c.side, c.hidden });
        for (table, std.mem.bytesAsSlice(u16, pos.bytes)) |*t, v| t.* = @bitCast(@as(u32, v) << 16);
        for (table) |t| if (!std.math.isFinite(t)) return error.NonFinitePositions; // the resize would carry it
        w.memory = memory;
        w.table = table;
        w.bytes = p.bytes;
        shift(&w, memory.ptr);
        for (w.layers) |*l| shift(l, memory.ptr);
        return w;
    }

    pub fn deinit(w: *Weights, gpa: Allocator) void {
        w.memory.free();
        gpa.free(w.layers);
        gpa.free(w.table);
        w.* = undefined;
    }
};

/// One tensor's bytes and where they go; `widen` turns its bf16 values into f32 on the way.
const Piece = struct { bytes: []const u8, dst: u64, widen: bool = false };

const Planner = struct {
    gpa: Allocator,
    files: []const st.File,
    c: *const Config,
    pieces: std.ArrayList(Piece) = .empty,
    bytes: u64 = 0,

    fn get(p: *Planner, name: []const u8, shape: []const usize) !st.Tensor {
        for (p.files) |*f| if (f.get(name)) |t| {
            if (t.is(.bf16, shape)) return t;
            std.log.err("{s}: {t} {any}, expected bf16 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
            return error.UnexpectedTensor;
        };
        std.log.err("checkpoint has no tensor {s}", .{name});
        return error.MissingTensor;
    }

    /// Places `name` (an offset from the allocation's start), f32 when `widen`.
    fn take(p: *Planner, name: []const u8, shape: []const usize, widen: bool) !u64 {
        const t = try p.get(name, shape);
        const at = std.mem.alignForward(u64, p.bytes, 256);
        try p.pieces.append(p.gpa, .{ .bytes = t.bytes, .dst = at, .widen = widen });
        p.bytes = at + t.bytes.len * @as(u64, if (widen) 2 else 1);
        return at;
    }

    fn plan(p: *Planner, w: *Weights) !void {
        const c = p.c;
        const h = c.hidden;
        var buf: [160]u8 = undefined;
        w.patch = try p.take(tower ++ "patch_generator.embedder.weight", &.{ h, c.patchWidth() }, false);
        w.cls = try p.take(tower ++ "patch_generator.cls_token.token", &.{ c.prefix, h }, false);
        for (w.layers, 0..) |*l, i| {
            const n = struct {
                fn of(b: []u8, li: usize, s: []const u8) []const u8 {
                    return std.fmt.bufPrint(b, tower ++ "blocks.{d}.{s}", .{ li, s }) catch unreachable;
                }
            }.of;
            l.norm1 = try p.take(n(&buf, i, "norm1.weight"), &.{h}, false);
            l.norm1_bias = try p.take(n(&buf, i, "norm1.bias"), &.{h}, false);
            l.qkv = try p.take(n(&buf, i, "attn.qkv.weight"), &.{ 3 * h, h }, false);
            l.qkv_bias = try p.take(n(&buf, i, "attn.qkv.bias"), &.{3 * h}, false);
            l.proj = try p.take(n(&buf, i, "attn.proj.weight"), &.{ h, h }, false);
            l.proj_bias = try p.take(n(&buf, i, "attn.proj.bias"), &.{h}, false);
            l.norm2 = try p.take(n(&buf, i, "norm2.weight"), &.{h}, false);
            l.norm2_bias = try p.take(n(&buf, i, "norm2.bias"), &.{h}, false);
            l.fc1 = try p.take(n(&buf, i, "mlp.fc1.weight"), &.{ c.ffn, h }, false);
            l.fc1_bias = try p.take(n(&buf, i, "mlp.fc1.bias"), &.{c.ffn}, false);
            l.fc2 = try p.take(n(&buf, i, "mlp.fc2.weight"), &.{ h, c.ffn }, false);
            l.fc2_bias = try p.take(n(&buf, i, "mlp.fc2.bias"), &.{h}, false);
        }
        if (c.final_norm) {
            w.final = try p.take("vision_projector.vision_final_layernorm.weight", &.{h}, false);
            w.final_bias = try p.take("vision_projector.vision_final_layernorm.bias", &.{h}, false);
        } else {
            w.final = 0;
            w.final_bias = 0;
        }
        const m = Config.shuffle * Config.shuffle * h;
        w.rms = try p.take("mlp1.0.weight", &.{m}, true);
        w.lin1 = try p.take("mlp1.1.weight", &.{ c.projector_hidden, m }, false);
        w.lin2 = try p.take("mlp1.3.weight", &.{ c.text_hidden, c.projector_hidden }, false);
        for (p.files) |*f| for (f.names.keys()) |k| {
            if (std.mem.startsWith(u8, k, tower ++ "blocks.") and !known(k)) {
                std.log.err("the tower has a tensor these kernels do not read: {s}", .{k});
                return error.UnreadTensor;
            }
        };
    }

    /// Whether a block tensor is one `plan` placed (a layer scale or SwiGLU weight is not).
    fn known(name: []const u8) bool {
        const rest = name[tower.len + "blocks.".len ..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return false;
        const parts = [_][]const u8{ "norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias" };
        for (parts) |s| if (std.mem.eql(u8, rest[dot + 1 ..], s)) return true;
        return false;
    }
};

/// Every u64 offset of `s` (not its byte count) moved by `base`.
fn shift(s: anytype, base: u64) void {
    const info = @typeInfo(@TypeOf(s.*)).@"struct";
    inline for (info.field_names, info.field_types) |name, T| if (T == u64 and comptime !std.mem.eql(u8, name, "bytes")) {
        @field(s, name) += base;
    };
}

/// A piece to the device, widened to f32 first when it asks.
fn upload(gpa: Allocator, memory: cuda.DeviceBuffer, piece: Piece) !void {
    if (!piece.widen) return memory.upload(piece.dst, piece.bytes);
    const wide = try gpa.alloc(f32, piece.bytes.len / 2);
    defer gpa.free(wide);
    for (wide, std.mem.bytesAsSlice(u16, piece.bytes)) |*o, v| o.* = @bitCast(@as(u32, v) << 16);
    try memory.upload(piece.dst, std.mem.sliceAsBytes(wide));
}

/// The shards holding the image tensors: those the index names for them, else every shard that has one.
fn open(gpa: Allocator, io: std.Io, dir: []const u8) !std.ArrayList(st.File) {
    var out: std.ArrayList(st.File) = .empty;
    errdefer {
        for (out.items) |*f| f.close(io);
        out.deinit(gpa);
    }
    const paths = try core.checkpoint.shardFiles(gpa, io, dir);
    defer core.checkpoint.freeShardFiles(gpa, paths);
    var named = try indexed(gpa, io, dir);
    defer if (named) |*n| freeNames(gpa, n);
    for (paths) |path| {
        if (named) |n| if (!n.contains(std.fs.path.basename(path))) continue;
        var f = try st.File.open(gpa, io, path);
        const has = for (f.names.keys()) |k| {
            if (ours(k)) break true;
        } else false;
        if (!has) {
            f.close(io);
            continue;
        }
        try out.append(gpa, f);
    }
    if (out.items.len == 0) return error.NoVisionTower;
    return out;
}

fn ours(name: []const u8) bool {
    for (prefixes) |p| if (std.mem.startsWith(u8, name, p)) return true;
    return false;
}

/// The shard names model.safetensors.index.json gives the image tensors; null without an index.
fn indexed(gpa: Allocator, io: std.Io, dir: []const u8) !?std.StringHashMapUnmanaged(void) {
    const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 27)) catch return null;
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    var out: std.StringHashMapUnmanaged(void) = .empty;
    errdefer freeNames(gpa, &out);
    var it = (parsed.value.object.get("weight_map") orelse return error.BadIndex).object.iterator();
    while (it.next()) |e| if (ours(e.key_ptr.*) and !out.contains(e.value_ptr.string)) {
        try out.put(gpa, try gpa.dupe(u8, e.value_ptr.string), {});
    };
    return out;
}

fn freeNames(gpa: Allocator, names: *std.StringHashMapUnmanaged(void)) void {
    var it = names.keyIterator();
    while (it.next()) |k| gpa.free(k.*);
    names.deinit(gpa);
}

test "a layer's tensors are known; a layer scale is not" {
    try std.testing.expect(Planner.known(tower ++ "blocks.31.mlp.fc2.bias"));
    try std.testing.expect(!Planner.known(tower ++ "blocks.3.ls1.gamma"));
}
