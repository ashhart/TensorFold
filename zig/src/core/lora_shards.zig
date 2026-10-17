//! A checkpoint folder read for the merge tools: its shards, quantization table, projection layouts and dense weights.
const std = @import("std");
const st = @import("safetensors.zig");
const mlx = @import("quant/mlx.zig");
const lq = @import("lora_quant.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const Refused = error{Refused};

/// Says why on `w`, naming the tensor, and returns `Refused`.
pub fn refuse(w: *Io.Writer, name: []const u8, why: []const u8) Refused {
    w.print("merge: {s}: {s}\n", .{ name, why }) catch {};
    return error.Refused;
}

/// Whether directory entry `e` is a regular file, or a symlink to one; a dangling symlink is not.
pub fn isRegular(dir: Io.Dir, io: Io, e: Io.Dir.Entry) bool {
    const kind = if (e.kind == .sym_link) (dir.statFile(io, e.name, .{}) catch return false).kind else e.kind;
    return kind == .file;
}

/// Refuses `path` unless it is a regular file or a symlink to one; false when nothing is there.
pub fn requireRegular(io: Io, w: *Io.Writer, path: []const u8) !bool {
    const cwd = Io.Dir.cwd();
    const link = cwd.statFile(io, path, .{ .follow_symlinks = false }) catch |e| switch (e) {
        error.FileNotFound => return false,
        else => return e,
    };
    const kind = if (link.kind == .sym_link) (cwd.statFile(io, path, .{}) catch return refuse(w, std.fs.path.basename(path), "a dangling symlink")).kind else link.kind;
    if (kind != .file) return refuse(w, std.fs.path.basename(path), "not a regular file");
    return true;
}

/// A non-empty file name with no separator, not `.` or `..`: a path that stays inside its folder.
pub fn plainName(name: []const u8) bool {
    if (name.len == 0 or std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) return false;
    return std.mem.indexOfAny(u8, name, "/\\\x00") == null;
}

/// `path` as a JSON object, or null when it is missing or not an object.
pub fn readObject(arena: Allocator, io: Io, w: *Io.Writer, path: []const u8) !?std.json.ObjectMap {
    if (!try requireRegular(io, w, path)) return null;
    const text = try Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 26));
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{});
    return if (v == .object) v.object else null;
}

/// A checkpoint folder's shards: the index's `weight_map`, or the one `model.safetensors`.
pub const Shards = struct {
    gpa: Allocator,
    io: Io,
    w: *Io.Writer,
    dir: []const u8,
    weight_map: ?std.json.ObjectMap,
    files: std.StringArrayHashMapUnmanaged(st.File) = .empty,

    pub fn init(gpa: Allocator, arena: Allocator, io: Io, w: *Io.Writer, dir: []const u8) !Shards {
        const path = try std.fs.path.join(arena, &.{ dir, "model.safetensors.index.json" });
        const index = (try readObject(arena, io, w, path)) orelse return .{ .gpa = gpa, .io = io, .w = w, .dir = dir, .weight_map = null };
        const map = index.get("weight_map");
        if (map == null or map.? != .object) return refuse(w, "model.safetensors.index.json", "no weight_map object");
        for (map.?.object.keys(), map.?.object.values()) |name, v| {
            if (v != .string or !plainName(v.string)) return refuse(w, name, "its shard is not a plain file name inside the folder");
        }
        return .{ .gpa = gpa, .io = io, .w = w, .dir = dir, .weight_map = map.?.object };
    }

    pub fn deinit(s: *Shards) void {
        for (s.files.values()) |*f| f.close(s.io);
        s.files.deinit(s.gpa);
    }

    /// The file name of the shard holding `name`, or null.
    pub fn shardOf(s: *const Shards, name: []const u8) ?[]const u8 {
        const map = s.weight_map orelse return "model.safetensors";
        const v = map.get(name) orelse return null;
        return if (v == .string) v.string else null;
    }

    /// The shard file `shard`, opened once, or null when the folder has none.
    pub fn open(s: *Shards, shard: []const u8) !?*st.File {
        if (s.files.getPtr(shard)) |f| return f;
        const path = try std.fs.path.join(s.gpa, &.{ s.dir, shard });
        defer s.gpa.free(path);
        if (!try requireRegular(s.io, s.w, path)) return null;
        var file = try st.File.open(s.gpa, s.io, path);
        errdefer file.close(s.io);
        try s.files.put(s.gpa, shard, file);
        return s.files.getPtr(shard);
    }

    /// The tensor `name` as stored, or null when the folder has none.
    pub fn get(s: *Shards, name: []const u8) !?st.Tensor {
        const shard = s.shardOf(name) orelse return null;
        const f = (try s.open(shard)) orelse return null;
        return f.get(name);
    }
};

/// One projection a tool works on, with its layout in the shipped checkpoint; `a` and `b` name adapter tensors.
pub const Job = struct {
    module: []const u8,
    shipped: ?[]const u8 = null, // the module's name in the shipped checkpoint, when it differs from the dense one
    a: []const u8 = "",
    b: []const u8 = "",
    n: usize,
    k: usize,
    group: usize,
    dtype: st.DType,

    /// The module's name in the shipped checkpoint (and so in OUT).
    pub fn shippedName(j: Job) []const u8 {
        return j.shipped orelse j.module;
    }
};

/// Module prefix rewrites between a dense source and an MLX conversion: (dense, shipped), tried both ways.
const prefix_rewrites = [_][2][]const u8{.{ "model.language_model.", "language_model.model." }};

/// One module's name in the dense source and in the shipped checkpoint.
pub const Names = struct { dense: []const u8, shipped: []const u8 };

/// `module`'s dense and shipped names, in either namespace: the pair both folders hold a weight for, or SHIPPED's.
pub fn resolve(arena: Allocator, w: *Io.Writer, dense: ?*Shards, shipped: *Shards, module: []const u8) !Names {
    var cands: std.ArrayList(Names) = .empty;
    try cands.append(arena, .{ .dense = module, .shipped = module });
    for (prefix_rewrites) |r| {
        if (std.mem.startsWith(u8, module, r[0])) try cands.append(arena, .{ .dense = module, .shipped = try std.fmt.allocPrint(arena, "{s}{s}", .{ r[1], module[r[0].len..] }) });
        if (dense != null and std.mem.startsWith(u8, module, r[1])) try cands.append(arena, .{ .dense = try std.fmt.allocPrint(arena, "{s}{s}", .{ r[0], module[r[1].len..] }), .shipped = module });
    }
    var found: std.ArrayList(Names) = .empty;
    var list: std.ArrayList(u8) = .empty;
    for (cands.items) |c| {
        const in_shipped = (try shipped.get(try std.fmt.allocPrint(arena, "{s}.weight", .{c.shipped}))) != null;
        const d = dense orelse {
            try list.print(arena, "{s}{s}.weight {s}", .{ if (list.items.len == 0) "" else "; ", c.shipped, if (in_shipped) "found" else "missing" });
            if (in_shipped) try found.append(arena, c);
            continue;
        };
        const in_dense = (try d.get(try std.fmt.allocPrint(arena, "{s}.weight", .{c.dense}))) != null;
        try list.print(arena, "{s}dense {s}.weight {s}, shipped {s}.weight {s}", .{ if (list.items.len == 0) "" else "; ", c.dense, if (in_dense) "found" else "missing", c.shipped, if (in_shipped) "found" else "missing" });
        if (in_dense and in_shipped) try found.append(arena, c);
    }
    if (found.items.len == 0 and dense == null) return refuse(w, module, try std.fmt.allocPrint(arena, "not found in the checkpoint (tried: {s}); a fused projection stores another name", .{list.items}));
    if (found.items.len == 0) return refuse(w, module, try std.fmt.allocPrint(arena, "not found in both the dense base and the shipped checkpoint (tried: {s}); a fused projection stores another name", .{list.items}));
    if (found.items.len > 1) return refuse(w, module, try std.fmt.allocPrint(arena, "ambiguous: more than one name pair resolves (tried: {s})", .{list.items}));
    return found.items[0];
}

/// The quantization table of `dir`'s config.json, refused unless it is MLX affine.
pub fn quantConfig(arena: Allocator, io: Io, w: *Io.Writer, dir: []const u8) !mlx.Config {
    const cfg = (try readObject(arena, io, w, try std.fs.path.join(arena, &.{ dir, "config.json" }))) orelse return refuse(w, "config.json", "missing or not an object");
    const qobj = cfg.get("quantization") orelse cfg.get("quantization_config") orelse return refuse(w, "config.json", "no quantization: the checkpoint is not MLX affine");
    if (qobj != .object) return refuse(w, "config.json", "quantization is not an object");
    return (try mlx.detect(arena, .{ .root = cfg, .quantization = qobj.object })) orelse refuse(w, "config.json", "quantization is not MLX affine");
}

/// A shipped projection's shape: `n` rows of `k` values in groups of `group`, its weight words as `dtype`.
pub const Layout = struct { n: usize, k: usize, group: usize, dtype: st.DType };

/// `module`'s stored layout in `shipped`, with every leaf present, in one shard, 4-bit and bf16 tables.
pub fn layoutOf(arena: Allocator, w: *Io.Writer, shipped: *Shards, quant: mlx.Config, module: []const u8) !Layout {
    const leaves = [3][]const u8{ try std.fmt.allocPrint(arena, "{s}.weight", .{module}), try std.fmt.allocPrint(arena, "{s}.scales", .{module}), try std.fmt.allocPrint(arena, "{s}.biases", .{module}) };
    const shard = shipped.shardOf(leaves[0]) orelse return refuse(w, leaves[0], "not in the checkpoint: no such quantized tensor (a fused projection stores another name)");
    const weight = (try shipped.get(leaves[0])) orelse return refuse(w, leaves[0], "not in the checkpoint: no such tensor");
    if ((try shipped.get(leaves[1])) == null or (try shipped.get(leaves[2])) == null) return refuse(w, module, "not quantized in the shipped checkpoint: it has no .scales and .biases (norms and embeddings are not merged)");
    if (weight.rank != 2) return refuse(w, module, try std.fmt.allocPrint(arena, "a rank {d} tensor: stacked (expert) tensors are not merged, only rank-2 quantized linears", .{weight.rank}));
    var t: [3]st.Tensor = undefined;
    for (&t, leaves) |*x, leaf| {
        if (!std.mem.eql(u8, shipped.shardOf(leaf) orelse "", shard)) return refuse(w, leaf, "not in the shard of its weight");
        x.* = (try shipped.get(leaf)) orelse return refuse(w, leaf, "not in the checkpoint: no such quantized tensor");
    }
    const width = quant.width(module) catch return refuse(w, module, "quantization the merge does not read");
    if (width.bits != 4) return refuse(w, module, "only 4-bit affine is merged");
    const group: usize = width.group;
    if ((t[0].dtype != .u32 and t[0].dtype != .i32) or t[0].rank != 2) return refuse(w, leaves[0], "not packed 32-bit words");
    if (t[1].dtype != .bf16 or t[2].dtype != .bf16 or t[1].rank != 2 or t[2].rank != 2 or !std.mem.eql(usize, t[1].shape[0..2], t[2].shape[0..2])) return refuse(w, leaves[1], "scales and biases are not bf16 tables of rank 2 and one shape");
    const n = t[0].dim(0);
    const k = std.math.mul(usize, t[1].dim(1), group) catch return refuse(w, module, "dimensions overflow");
    _ = std.math.mul(usize, n, k) catch return refuse(w, module, "dimensions overflow");
    if (n == 0 or k == 0) return refuse(w, module, "an empty projection");
    if (t[1].dim(0) != n or t[0].dim(1) != k / 8) return refuse(w, module, "packed shape does not match its scales and group");
    return .{ .n = n, .k = k, .group = group, .dtype = t[0].dtype };
}

/// The dense weight of `module`, refused unless an `n` x `k` f32, bf16 or f16 matrix with that many bytes.
pub fn denseTensor(w: *Io.Writer, dense: *Shards, module: []const u8, n: usize, k: usize) !st.Tensor {
    var buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}.weight", .{module});
    const t = (try dense.get(name)) orelse return refuse(w, name, "not in the dense base");
    if (t.rank != 2 or t.dim(0) != n or t.dim(1) != k) return refuse(w, name, "dense shape does not match the shipped projection");
    if (t.dtype != .f32 and t.dtype != .bf16 and t.dtype != .f16) return refuse(w, name, "dense dtype is not f32, bf16 or f16");
    if (t.bytes.len != n * k * t.dtype.size()) return refuse(w, name, "dense payload does not match its shape");
    return t;
}

fn allFinite(v: []const f32) bool {
    for (v) |x| if (!std.math.isFinite(x)) return false;
    return true;
}

/// The dense weight of `j` as f32, checked against its layout and finite; the caller frees it.
pub fn denseOf(gpa: Allocator, w: *Io.Writer, dense: *Shards, j: Job) ![]f32 {
    const t = try denseTensor(w, dense, j.module, j.n, j.k);
    const out = try gpa.alloc(f32, j.n * j.k);
    errdefer gpa.free(out);
    try lq.toF32(t.dtype, t.bytes, out);
    if (!allFinite(out)) return refuse(w, j.module, "the dense weight holds a non-finite value");
    return out;
}

/// A projection's leaves copied into aligned memory; `freeQuant` returns it.
pub fn quantOf(gpa: Allocator, s: *Shards, j: Job) !lq.Quant {
    var buf: [256]u8 = undefined;
    const q = try lq.allocQuant(gpa, j.n, j.k, j.group);
    errdefer lq.freeQuant(gpa, q);
    inline for (.{ .{ "weight", q.words }, .{ "scales", q.scales }, .{ "biases", q.biases } }) |leaf| {
        const t = (try s.get(try std.fmt.bufPrint(&buf, "{s}." ++ leaf[0], .{j.shippedName()}))) orelse return error.MissingTensor;
        if (t.bytes.len != std.mem.sliceAsBytes(leaf[1]).len) return error.UnexpectedTensor;
        @memcpy(std.mem.sliceAsBytes(leaf[1]), t.bytes);
    }
    return q;
}

pub fn free(gpa: Allocator, q: lq.Quant) void {
    lq.freeQuant(gpa, q);
}
