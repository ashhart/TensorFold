//! A Qwen-VL tower's bf16 tensors (vision_tower.*) from a checkpoint's shards into one Metal buffer, 256-aligned.
const std = @import("std");
const mtl = @import("metal");
const st = @import("../safetensors.zig");
const Config = @import("config.zig").Config;

pub const prefix = "vision_tower.";

/// A place in a buffer.
pub const Ref = struct {
    buf: mtl.Buffer,
    off: usize = 0,

    pub fn at(r: Ref, n: usize) Ref {
        return .{ .buf = r.buf, .off = r.off + n };
    }

    pub fn addr(r: Ref) [*]u8 {
        return r.buf.contents() + r.off;
    }
};

pub const Block = struct { norm1_w: Ref, norm1_b: Ref, qkv_w: Ref, qkv_b: Ref, proj_w: Ref, proj_b: Ref, norm2_w: Ref, norm2_b: Ref, fc1_w: Ref, fc1_b: Ref, fc2_w: Ref, fc2_b: Ref };

pub const Weights = struct {
    buf: mtl.Buffer,
    patch_w: Ref = undefined, // [hidden, channel x frame x y x x]: the patches' own order
    patch_b: Ref = undefined,
    pos: Ref = undefined, // [positions, hidden]
    blocks: []Block = &.{},
    norm_w: Ref = undefined, // the merger's
    norm_b: Ref = undefined,
    fc1_w: Ref = undefined,
    fc1_b: Ref = undefined,
    fc2_w: Ref = undefined, // [out_hidden, merged]
    fc2_b: Ref = undefined,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        gpa.free(w.blocks);
        w.buf.deinit();
    }
};

const Want = struct { name: []const u8, shape: []const usize, ref: *Ref };

/// Every tower tensor in buffer order with its shape (the patch weight as stored by MLX) and the place it fills.
fn wanted(arena: std.mem.Allocator, c: Config, w: *Weights) !std.ArrayList(Want) {
    const h: usize = c.hidden;
    const m: usize = c.merged();
    var out: std.ArrayList(Want) = .empty;
    try out.append(arena, .{ .name = "patch_embed.proj.weight", .shape = try arena.dupe(usize, &.{ h, c.temporal, c.patch, c.patch, 3 }), .ref = &w.patch_w });
    try out.append(arena, .{ .name = "patch_embed.proj.bias", .shape = try arena.dupe(usize, &.{h}), .ref = &w.patch_b });
    try out.append(arena, .{ .name = "pos_embed.weight", .shape = try arena.dupe(usize, &.{ c.positions, h }), .ref = &w.pos });
    for (w.blocks, 0..) |*b, i| {
        const fields = .{ .{ "norm1.weight", &b.norm1_w, &[_]usize{h} }, .{ "norm1.bias", &b.norm1_b, &[_]usize{h} }, .{ "attn.qkv.weight", &b.qkv_w, &[_]usize{ 3 * h, h } }, .{ "attn.qkv.bias", &b.qkv_b, &[_]usize{3 * h} }, .{ "attn.proj.weight", &b.proj_w, &[_]usize{ h, h } }, .{ "attn.proj.bias", &b.proj_b, &[_]usize{h} }, .{ "norm2.weight", &b.norm2_w, &[_]usize{h} }, .{ "norm2.bias", &b.norm2_b, &[_]usize{h} }, .{ "mlp.linear_fc1.weight", &b.fc1_w, &[_]usize{ c.intermediate, h } }, .{ "mlp.linear_fc1.bias", &b.fc1_b, &[_]usize{c.intermediate} }, .{ "mlp.linear_fc2.weight", &b.fc2_w, &[_]usize{ h, c.intermediate } }, .{ "mlp.linear_fc2.bias", &b.fc2_b, &[_]usize{h} } };
        inline for (fields) |f| try out.append(arena, .{ .name = try std.fmt.allocPrint(arena, "blocks.{d}.{s}", .{ i, f[0] }), .shape = try arena.dupe(usize, f[2]), .ref = f[1] });
    }
    const tail = .{ .{ "merger.norm.weight", &w.norm_w, &[_]usize{h} }, .{ "merger.norm.bias", &w.norm_b, &[_]usize{h} }, .{ "merger.linear_fc1.weight", &w.fc1_w, &[_]usize{ m, m } }, .{ "merger.linear_fc1.bias", &w.fc1_b, &[_]usize{m} }, .{ "merger.linear_fc2.weight", &w.fc2_w, &[_]usize{ c.out_hidden, m } }, .{ "merger.linear_fc2.bias", &w.fc2_b, &[_]usize{c.out_hidden} } };
    inline for (tail) |f| try out.append(arena, .{ .name = f[0], .shape = try arena.dupe(usize, f[2]), .ref = f[1] });
    return out;
}

fn count(shape: []const usize) usize {
    var n: usize = 1;
    for (shape) |d| n *= d;
    return n;
}

/// The buffer's bytes: every tensor in bf16, each 256-aligned.
pub fn bytes(gpa: std.mem.Allocator, c: Config) !usize {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var w: Weights = .{ .buf = undefined, .blocks = try arena.allocator().alloc(Block, c.depth) };
    var total: usize = 0;
    for ((try wanted(arena.allocator(), c, &w)).items) |want| total = std.mem.alignForward(usize, total, 256) + 2 * count(want.shape);
    return total;
}

const Src = struct { path: [:0]const u8, off: u64, e: st.Entry };

/// The tower's tensors from `dir` into one shared buffer; the MLX patch kernel [O, T, Y, X, C] turned to [O, C, T, Y, X].
pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, c: Config) !Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const found = try headers(arena, io, dir);
    var w: Weights = .{ .buf = undefined, .blocks = try gpa.alloc(Block, c.depth) };
    errdefer gpa.free(w.blocks);
    const wants = (try wanted(arena, c, &w)).items;
    if (found.count() != wants.len) {
        std.log.err("qwen vision: the checkpoint has {d} {s}* tensors; this tower reads {d}", .{ found.count(), prefix, wants.len });
        return error.UnexpectedVisionTensors;
    }
    const srcs = try arena.alloc(Src, wants.len);
    var total: usize = 0;
    for (wants, srcs, 0..) |want, *s, i| {
        const name = try std.fmt.allocPrint(arena, prefix ++ "{s}", .{want.name});
        s.* = found.get(name) orelse {
            std.log.err("qwen vision: the checkpoint has no tensor {s}", .{name});
            return error.MissingVisionTensor;
        };
        const torch_patch = i == 0 and s.e.rank == 5 and std.mem.eql(usize, s.e.shape[0..5], &.{ c.hidden, 3, c.temporal, c.patch, c.patch });
        if (s.e.dtype != .bf16 or !(torch_patch or (s.e.rank == want.shape.len and std.mem.eql(usize, s.e.shape[0..s.e.rank], want.shape)))) {
            std.log.err("qwen vision: {s} is {t} {any}; the tower reads bf16 {any}", .{ name, s.e.dtype, s.e.shape[0..s.e.rank], want.shape });
            return error.UnexpectedVisionTensor;
        }
        total = std.mem.alignForward(usize, total, 256) + (s.e.end - s.e.begin);
    }
    w.buf = try device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer w.buf.deinit();
    var at: usize = 0;
    for (wants, srcs) |want, s| {
        at = std.mem.alignForward(usize, at, 256);
        want.ref.* = .{ .buf = w.buf, .off = at };
        try readInto(s.path, w.buf.contents()[at..][0 .. s.e.end - s.e.begin], s.off);
        at += s.e.end - s.e.begin;
    }
    if (srcs[0].e.shape[4] == 3) try turnPatch(gpa, w.patch_w, c); // an HF kernel is already [O, C, T, Y, X]
    return w;
}

/// Every vision tensor's file, offset and header entry, found through the shard index (or the one file).
fn headers(arena: std.mem.Allocator, io: std.Io, dir: []const u8) !std.StringArrayHashMapUnmanaged(Src) {
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    const index = try std.fs.path.join(arena, &.{ dir, "model.safetensors.index.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, index, arena, .limited(64 << 20))) |text| {
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{});
        const map = (doc.object.get("weight_map") orelse return error.BadIndex).object;
        var it = map.iterator();
        while (it.next()) |kv| if (std.mem.startsWith(u8, kv.key_ptr.*, prefix)) try files.put(arena, kv.value_ptr.string, {});
    } else |_| try files.put(arena, "model.safetensors", {});
    var out: std.StringArrayHashMapUnmanaged(Src) = .empty;
    for (files.keys()) |name| {
        const path = try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ dir, name }, 0);
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.OpenFailed;
        defer _ = std.c.close(fd);
        var head: [8]u8 = undefined;
        try readAll(fd, &head, 0);
        const hlen = std.mem.readInt(u64, &head, .little);
        const end: u64 = @intCast(std.c.lseek(fd, 0, std.c.SEEK.END));
        if (hlen > end - 8) return error.BadSafetensors;
        const header = try arena.alloc(u8, @intCast(hlen));
        try readAll(fd, header, 8);
        const parsed = try st.parseHeaderPrefix(arena, header, @intCast(end - 8 - hlen), prefix);
        var it = parsed.iterator();
        while (it.next()) |kv| try out.put(arena, kv.key_ptr.*, .{ .path = path, .off = 8 + hlen + kv.value_ptr.begin, .e = kv.value_ptr.* });
    }
    if (out.count() == 0) return error.NoVisionTower;
    return out;
}

fn readInto(path: [:0]const u8, dst: []u8, off: u64) !void {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);
    _ = std.c.fcntl(fd, 48, @as(c_int, 1)); // F_NOCACHE, as the language model's shards are read
    try readAll(fd, dst, off);
}

fn readAll(fd: std.c.fd_t, dst: []u8, at: u64) !void {
    var done: usize = 0;
    while (done < dst.len) {
        const n = std.c.pread(fd, dst.ptr + done, dst.len - done, @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

/// [O, T, Y, X, C] (MLX's Conv3d) to [O, C, T, Y, X], each output channel's row in place.
fn turnPatch(gpa: std.mem.Allocator, r: Ref, c: Config) !void {
    const width: usize = c.patchWidth();
    const row = try gpa.alloc(u16, width);
    defer gpa.free(row);
    const all: [*]u16 = @ptrCast(@alignCast(r.addr()));
    const plane: usize = c.patch * c.patch;
    for (0..c.hidden) |o| {
        const dst = all[o * width ..][0..width];
        @memcpy(row, dst);
        for (0..3) |ch| for (0..c.temporal) |t| for (0..plane) |yx| {
            dst[(ch * c.temporal + t) * plane + yx] = row[(t * plane + yx) * 3 + ch];
        };
    }
}

test "the buffer holds the 27B tower's 333 tensors in bf16" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const sample = @import("config.zig");
    const c = (try Config.parse(arena.allocator(), sample.sample_config, null, null)).?;
    var w: Weights = .{ .buf = undefined, .blocks = try arena.allocator().alloc(Block, c.depth) };
    const wants = try wanted(arena.allocator(), c, &w);
    try std.testing.expectEqual(@as(usize, 333), wants.items.len);
    const n = try bytes(std.testing.allocator, c);
    try std.testing.expect(n > 900_000_000 and n < 930_000_000); // about 460 million weights
}
