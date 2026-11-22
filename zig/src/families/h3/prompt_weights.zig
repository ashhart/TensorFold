//! The prompt path's weights, bf16 read in place: the text encoder's first layers and the transformer's front end.
const std = @import("std");
const mtl = @import("metal");
const tensors = @import("tensors.zig");
const rules = @import("picture_prompt.zig");

const Ref = tensors.Ref;
pub const Layer = rules.Layer(Ref);
pub const Block = rules.Block(Ref);

/// The shards of a checkpoint folder that hold the tensors it was opened for, each mapped once.
pub const Shards = struct {
    files: []tensors.Tensors,

    pub fn deinit(s: *Shards, gpa: std.mem.Allocator) void {
        for (s.files) |*f| f.deinit();
        gpa.free(s.files);
    }

    /// The tensor `name`: bf16 of `shape`, starting on 8 bytes in its file, read in place.
    pub fn take(s: *const Shards, name: []const u8, shape: rules.Shape) !Ref {
        for (s.files) |*f| if (f.names.get(name)) |e| {
            const rank: usize = if (shape[1] == 0) 1 else 2;
            if (e.dtype != .bf16 or e.rank != rank or !std.mem.eql(usize, e.shape[0..rank], shape[0..rank])) {
                std.log.err("h3 prompt: {s} is {t} {any}; the prompt path reads bf16 {any}", .{ name, e.dtype, e.shape[0..e.rank], shape[0..rank] });
                return error.H3UnexpectedTensor;
            }
            const offset = f.data + e.begin;
            if (offset % 8 != 0) {
                std.log.err("h3 prompt: {s} starts at byte {d}, not on 8", .{ name, offset });
                return error.H3UnalignedTensor;
            }
            return .{ .buffer = f.buffer, .offset = offset };
        };
        std.log.err("h3 prompt: no tensor {s}", .{name});
        return error.MissingTensor;
    }

    /// The bytes of a tensor `take` returned, on the host.
    pub fn bytes(ref: Ref, n: usize) []const u8 {
        return ref.buffer.contents()[ref.offset..][0..n];
    }
};

/// `dir`'s shards (from its `model.safetensors.index.json`) that hold a name `wanted` accepts.
pub fn open(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, wanted: anytype) !Shards {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const path = try std.fs.path.join(arena, &.{ dir, "model.safetensors.index.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(64 << 20));
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch return error.H3BadShardIndex;
    if (doc != .object) return error.H3BadShardIndex;
    const map = doc.object.get("weight_map") orelse return error.H3BadShardIndex;
    if (map != .object) return error.H3BadShardIndex;
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (map.object.keys(), map.object.values()) |name, file| {
        if (file != .string or std.mem.indexOf(u8, file.string, "/") != null) return error.H3BadShardIndex;
        if (wanted.accepts(name)) try files.put(arena, file.string, {});
    }
    if (files.count() == 0) return error.MissingTensor;
    const out = try gpa.alloc(tensors.Tensors, files.count());
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |*f| f.deinit();
        gpa.free(out);
    }
    for (files.keys()) |file| {
        out[made] = try tensors.Tensors.open(gpa, device, try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ dir, file }, 0));
        made += 1;
    }
    return .{ .files = out };
}

/// The index's names, for the text model's prefix.
pub fn names(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![]const []const u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    const doc = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return error.H3BadShardIndex;
    defer doc.deinit();
    const map = if (doc.value == .object) doc.value.object.get("weight_map") else null;
    if (map == null or map.? != .object) return error.H3BadShardIndex;
    const out = try gpa.alloc([]const u8, map.?.object.count());
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |n| gpa.free(n);
        gpa.free(out);
    }
    for (map.?.object.keys()) |k| {
        out[made] = try gpa.dupe(u8, k);
        made += 1;
    }
    return out;
}

/// The text model's first `g.layers` layers and its embedding table, read in place.
pub const Text = struct {
    g: rules.Text,
    shards: Shards,
    embed: Ref,
    layers: []Layer,

    pub fn deinit(t: *Text, gpa: std.mem.Allocator) void {
        gpa.free(t.layers);
        t.shards.deinit(gpa);
    }

    /// Row `id` of the embedding table, its bf16 bits.
    pub fn row(t: *const Text, id: u32) ![]const u8 {
        if (id >= t.g.vocab) return error.H3TokenOutOfRange;
        return Shards.bytes(t.embed, t.g.vocab * t.g.hidden * 2)[@as(usize, id) * t.g.hidden * 2 ..][0 .. t.g.hidden * 2];
    }
};

/// The names of the text model's embedding and of its layers below `layers`, under `prefix`.
const TextNames = struct {
    prefix: []const u8,
    layers: usize,

    pub fn accepts(n: TextNames, name: []const u8) bool {
        if (!std.mem.startsWith(u8, name, n.prefix)) return false;
        const rest = name[n.prefix.len..];
        if (std.mem.eql(u8, rest, "embed_tokens.weight")) return true;
        if (!std.mem.startsWith(u8, rest, "layers.")) return false;
        const dot = std.mem.indexOfScalar(u8, rest["layers.".len..], '.') orelse return false;
        const i = std.fmt.parseInt(usize, rest["layers.".len..][0..dot], 10) catch return false;
        return i < n.layers;
    }
};

/// A Qwen3-VL checkpoint folder's text model at `g`: the shards of its first `g.layers` layers and embedding.
pub fn loadText(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, g: rules.Text) !Text {
    const all = try names(gpa, io, dir);
    defer {
        for (all) |n| gpa.free(n);
        gpa.free(all);
    }
    const prefix = try rules.textPrefix(all);
    var t: Text = .{ .g = g, .shards = try open(gpa, io, device, dir, TextNames{ .prefix = prefix, .layers = g.layers }), .embed = undefined, .layers = &.{} };
    errdefer t.shards.deinit(gpa);
    var name: [160]u8 = undefined;
    t.embed = try t.shards.take(try std.fmt.bufPrint(&name, "{s}embed_tokens.weight", .{prefix}), .{ g.vocab, g.hidden });
    t.layers = try gpa.alloc(Layer, g.layers);
    errdefer gpa.free(t.layers);
    const specs = rules.layerTensors(g);
    for (t.layers, 0..) |*l, i| {
        inline for (comptime std.meta.fieldNames(Layer), 0..) |field, f| {
            const s = specs[f];
            std.debug.assert(std.mem.eql(u8, field, s[0]));
            @field(l, field) = try t.shards.take(try std.fmt.bufPrint(&name, "{s}layers.{d}.{s}", .{ prefix, i, s[1] }), s[2]);
        }
    }
    return t;
}

/// condition_proj, the token refiner's blocks and its final norm, read in place.
pub const Refiner = struct {
    g: rules.Refiner,
    shards: Shards,
    cond_w: Ref,
    cond_b: Ref,
    final_norm: Ref,
    blocks: []Block,

    pub fn deinit(r: *Refiner, gpa: std.mem.Allocator) void {
        gpa.free(r.blocks);
        r.shards.deinit(gpa);
    }
};

const FrontNames = struct {
    pub fn accepts(_: FrontNames, name: []const u8) bool {
        return std.mem.startsWith(u8, name, "condition_proj.") or std.mem.startsWith(u8, name, "token_refiner.");
    }
};

/// A MiniMax-H3 transformer folder's text front end at `g`.
pub fn loadRefiner(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, g: rules.Refiner) !Refiner {
    var r: Refiner = .{ .g = g, .shards = try open(gpa, io, device, dir, FrontNames{}), .cond_w = undefined, .cond_b = undefined, .final_norm = undefined, .blocks = &.{} };
    errdefer r.shards.deinit(gpa);
    inline for (.{ "cond_w", "cond_b", "final_norm" }, rules.frontTensors(g)) |field, s| {
        std.debug.assert(std.mem.eql(u8, field, s[0]));
        @field(r, field) = try r.shards.take(s[1], s[2]);
    }
    r.blocks = try gpa.alloc(Block, g.blocks);
    errdefer gpa.free(r.blocks);
    const specs = rules.blockTensors(g);
    var name: [96]u8 = undefined;
    for (r.blocks, 0..) |*b, i| {
        inline for (comptime std.meta.fieldNames(Block), 0..) |field, f| {
            const s = specs[f];
            std.debug.assert(std.mem.eql(u8, field, s[0]));
            @field(b, field) = try r.shards.take(try std.fmt.bufPrint(&name, "token_refiner.blocks.{d}.{s}", .{ i, s[1] }), s[2]);
        }
    }
    return r;
}
