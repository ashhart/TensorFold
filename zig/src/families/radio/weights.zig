//! The RADIO tower's and projector's tensors as stored (bf16) in one device allocation.
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
    video_patch: u64, // [hidden, video_temporal * 3 * patch * patch] (0 without a video projection)
    table: u64, // the position table [side * side, hidden]
    cls: u64, // [prefix, hidden]: cls tokens, then registers
    final: u64, // vision_final_layernorm (unset without an MTP layer: Config.final_norm)
    final_bias: u64,
    rms: u64, // the projector's RMSNorm weight as f32 [4 hidden]
    lin1: u64, // [projector_hidden, 4 hidden]
    lin2: u64, // [text_hidden, projector_hidden]
    layers: []Layer,
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
        var local = true;
        errdefer if (local) gpa.free(layers);
        var w: Weights = undefined;
        w.layers = layers;
        try p.plan(&w);
        var memory = try cuda.DeviceBuffer.alloc(d, p.bytes);
        errdefer if (local) memory.free();
        w.memory = memory;
        w.bytes = p.bytes;
        const owner = try gpa.create(Failed);
        owner.* = .{ .gpa = gpa, .w = w };
        local = false;
        w = try transfer(owner, p.pieces.items);
        shift(&w, memory.ptr);
        for (w.layers) |*l| shift(l, memory.ptr);
        return w;
    }

    pub fn deinit(w: *Weights, gpa: Allocator) void {
        w.memory.free();
        gpa.free(w.layers);
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
        w.video_patch = if (c.video_temporal > 0) try p.take(tower ++ "patch_generator.video_embedder.weight", &.{ h, c.videoWidth() }, false) else 0;
        w.table = try p.take(tower ++ "patch_generator.pos_embed", &.{ 1, c.side * c.side, h }, false);
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

const Failed = struct {
    gpa: Allocator,
    w: Weights,
    next: ?*Failed = null,
    fn close(f: *Failed) !void {
        try f.w.memory.d.synchronize();
        f.w.deinit(f.gpa);
        f.gpa.destroy(f);
    }
};
var failed: ?*Failed = null;
var failed_mutex: std.atomic.Mutex = .unlocked;
fn keep(f: *Failed) void {
    while (!failed_mutex.tryLock()) std.Thread.yield() catch {};
    defer failed_mutex.unlock();
    f.next = failed;
    failed = f;
}
fn transfer(owner: *Failed, pieces: []const Piece) !Weights {
    for (pieces) |piece| upload(owner.gpa, owner.w.memory, piece) catch |err| {
        owner.close() catch {
            keep(owner);
            return error.LoadRetained;
        };
        return err;
    };
    const out = owner.w;
    owner.gpa.destroy(owner);
    return out;
}
var recovery: std.atomic.Mutex = .unlocked;
pub fn recover(d: *const cuda.Driver) usize {
    // A detached transaction remains a pending owner at every parent/admission boundary.
    if (!recovery.tryLock()) return 1;
    defer recovery.unlock();
    while (!failed_mutex.tryLock()) std.Thread.yield() catch {};
    var at = failed;
    failed = null;
    failed_mutex.unlock();
    if (comptime @import("builtin").is_test) if (Testing.detached) |hook| {
        Testing.detached = null;
        hook();
    };
    while (at) |f| {
        at = f.next;
        f.next = null;
        if (f.w.memory.d == d) {
            f.close() catch {
                keep(f);
            };
        } else keep(f);
    }
    while (!failed_mutex.tryLock()) std.Thread.yield() catch {};
    defer failed_mutex.unlock();
    var n: usize = 0;
    at = failed;
    while (at) |f| {
        if (f.w.memory.d == d) n += 1;
        at = f.next;
    }
    return n;
}

test "RADIO planned upload retains its device destination when recovery fails" {
    const F = struct {
        var pending = false;
        var fail = true;
        var frees: usize = 0;
        var unsafe: usize = 0;
        fn alloc(out: *u64, _: usize) callconv(.c) c_int {
            out.* = 256;
            return 0;
        }
        fn free(_: u64) callconv(.c) c_int {
            if (pending) unsafe += 1;
            frees += 1;
            return 0;
        }
        fn copy(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
            pending = true;
            return 0;
        }
        fn sync() callconv(.c) c_int {
            if (fail) return cuda.abi.error_not_ready;
            pending = false;
            return 0;
        }
        fn stream(_: cuda.abi.Stream) callconv(.c) c_int {
            return sync();
        }
    };
    F.pending = false;
    F.fail = true;
    F.frees = 0;
    F.unsafe = 0;
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = F.alloc;
    d.api.cuMemFree_v2 = F.free;
    d.api.cuMemcpyHtoD_v2 = F.copy;
    d.api.cuCtxSynchronize = F.sync;
    d.api.cuStreamSynchronize = F.stream;
    cuda.copy_source.setCurrent(@ptrFromInt(1));
    const owner = try std.testing.allocator.create(Failed);
    owner.gpa = std.testing.allocator;
    owner.next = null;
    owner.w = undefined;
    owner.w.memory = try cuda.DeviceBuffer.alloc(&d, 4);
    owner.w.layers = try std.testing.allocator.alloc(Layer, 0);
    try std.testing.expectError(error.LoadRetained, transfer(owner, &.{.{ .dst = 0, .bytes = &.{ 1, 2, 3, 4 } }}));
    try std.testing.expectEqual(@as(usize, 1), recover(&d));
    try std.testing.expect(F.pending);
    try std.testing.expectEqual(@as(usize, 0), F.frees);
    F.fail = false;
    try std.testing.expectEqual(@as(usize, 0), recover(&d));
    try std.testing.expectEqual(@as(usize, 1), F.frees);
    try std.testing.expectEqual(@as(usize, 0), F.unsafe);
}

pub const Testing = if (@import("builtin").is_test) struct {
    pub fn empty() bool {
        while (!failed_mutex.tryLock()) std.Thread.yield() catch {};
        defer failed_mutex.unlock();
        return failed == null;
    }
    pub var detached: ?*const fn () void = null;
    pub fn tiny(gpa: Allocator, d: *const cuda.Driver) !void {
        const f = try gpa.create(Failed);
        f.gpa = gpa;
        f.next = null;
        f.w = undefined;
        f.w.memory = try cuda.DeviceBuffer.alloc(d, 4);
        f.w.layers = try gpa.alloc(Layer, 0);
        keep(f);
    }
} else struct {};
