//! A GLP control vector: the GGUF container's GLP subset read whole, written whole, and the rules that accept it.

const std = @import("std");
const gguf = @import("gguf.zig");

/// Why a file was refused, in one sentence the server prints.
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const Error = error{BadGlpVector} || std.mem.Allocator.Error;

/// How a direction is applied to the residual stream.
pub const Mode = enum {
    /// h -= alpha*(h.d)d: the component along the direction leaves the stream.
    project,
    /// h += alpha*d: llama.cpp's control vector, a bias rather than a removal.
    add,
};

/// One layer's direction: its zero-based layer id and the vector itself.
pub const Layer = struct {
    id: u32,
    dir: []const f32,
};

/// A vector read whole and accepted.
pub const Vector = struct {
    mode: Mode,
    alpha: f32,
    hook_point: []const u8,
    derived_at: []const u8,
    base_model: []const u8,
    base_revision: []const u8,
    method: []const u8,
    contrast: []const u8,
    width: usize,
    layers: []const Layer,
    sha256: [32]u8,
    transferred: bool,
};

/// The residual site this reader applies at, the one llama.cpp's control vectors name.
pub const residual_site = "residual_stream_post_layer";

fn refuse(why: *Why, comptime fmt: []const u8, args: anytype) Error {
    why.set(fmt, args);
    return error.BadGlpVector;
}

/// The direction's layer id, or null when the tensor is not one.
fn directionId(name: []const u8) ?u32 {
    if (!std.mem.startsWith(u8, name, "direction.")) return null;
    const tail = name["direction.".len..];
    if (tail.len == 0) return null;
    for (tail) |ch| if (ch < '0' or ch > '9') return null;
    return std.fmt.parseInt(u32, tail, 10) catch null;
}

/// The digest over the tensor bytes; the file's own key names the vectors it covers.
fn digest(file: gguf.File, only_directions: bool) ?[32]u8 {
    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    for (file.tensors) |t| {
        if (only_directions and directionId(t.name) == null) continue;
        const raw = file.bytesOf(t) orelse return null;
        h.update(raw);
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

/// Read a vector whole and accept it, or refuse it with a reason; the strings and layers live in `a`.
pub fn read(a: std.mem.Allocator, bytes: []const u8, why: *Why) Error!Vector {
    if (bytes.len < 8) return refuse(why, "the file is too short to be a control vector", .{});
    const file = try parseContainer(a, bytes, why);

    var glp_keys: usize = 0;
    for (file.entries) |e| {
        // The keys whose meaning the spec version decides; alpha and the digest mean the same in every version.
        const interpretation = std.mem.eql(u8, e.key, "glp.mode") or std.mem.eql(u8, e.key, "glp.rank") or
            std.mem.eql(u8, e.key, "glp.orthonormal") or std.mem.eql(u8, e.key, "glp.hook_point") or
            std.mem.eql(u8, e.key, "glp.derived_at") or std.mem.eql(u8, e.key, "glp.dir_scales") or
            std.mem.eql(u8, e.key, "glp.layer_scales");
        if (interpretation) glp_keys += 1;
    }

    const mode_key = if (file.str("glp.mode") != null) "glp.mode" else if (file.str("dspark.mode") != null) "dspark.mode" else "";
    const mode: Mode = if (mode_key.len == 0) .add else if (std.mem.eql(u8, file.str(mode_key).?, "project")) .project else if (std.mem.eql(u8, file.str(mode_key).?, "add")) .add else return refuse(why, "the file's mode is not one this reader applies", .{});

    if (file.num("glp.spec_version")) |version| {
        if (version != 1) return refuse(why, "glp.spec_version {d}; this reader implements version 1", .{version});
    } else if (glp_keys > 0) {
        return refuse(why, "the file carries glp keys but no glp.spec_version", .{});
    }

    if (glp_keys > 0) {
        const rank = file.num("glp.rank") orelse 1;
        if (rank != 1) return refuse(why, "glp.rank {d}; this reader implements rank 1", .{rank});
        const site = file.str("glp.hook_point") orelse return refuse(why, "the file names no glp.hook_point", .{});
        if (!std.mem.eql(u8, site, residual_site)) return refuse(why, "the file's hook point is {s}; this reader applies at {s}", .{ site, residual_site });
    }

    var layers: std.ArrayList(Layer) = .empty;
    var width: usize = 0;
    for (file.tensors) |t| {
        const id = directionId(t.name) orelse continue;
        if (id == 0) return refuse(why, "the file holds direction.0; layer 0 is not steerable", .{});
        if (t.kind != gguf.f32_type) return refuse(why, "{s} is not fp32", .{t.name});
        if (t.dims.len != 1) return refuse(why, "{s} is not one dimensional", .{t.name});
        if (width == 0) width = @intCast(t.dims[0]) else if (width != t.dims[0]) return refuse(why, "{s} is {d} wide; the other directions are {d}", .{ t.name, t.dims[0], width });
        const raw = file.bytesOf(t) orelse return refuse(why, "{s} points past the file", .{t.name});
        const dir = try a.alloc(f32, t.dims[0]);
        for (dir, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
        try layers.append(a, .{ .id = id, .dir = dir });
    }
    if (layers.items.len == 0) return refuse(why, "the file holds no direction tensors", .{});
    std.mem.sort(Layer, layers.items, {}, struct {
        fn lessThan(_: void, x: Layer, y: Layer) bool {
            return x.id < y.id;
        }
    }.lessThan);

    if (file.str("glp.layer_ids_zero_based")) |list| {
        var it = std.mem.splitScalar(u8, list, ',');
        var i: usize = 0;
        while (it.next()) |part| {
            if (i >= layers.items.len) return refuse(why, "the file names more layers than it holds", .{});
            const id = std.fmt.parseInt(u32, std.mem.trim(u8, part, " "), 10) catch return refuse(why, "the file's layer id list is not a number list", .{});
            if (id != layers.items[i].id) return refuse(why, "the file's layer id {d} does not match its tensor's {d}", .{ id, layers.items[i].id });
            i += 1;
        }
        if (i != layers.items.len) return refuse(why, "the file names {d} layers and holds {d}", .{ i, layers.items.len });
    }

    if (mode == .project) {
        for (layers.items) |l| {
            var sum: f64 = 0;
            for (l.dir) |v| sum += @as(f64, v) * @as(f64, v);
            const norm = @sqrt(sum);
            if (!std.math.isFinite(norm) or @abs(norm - 1.0) > 1e-3) return refuse(why, "layer {d}'s direction has length {d:.4}; a projective direction must be a unit vector", .{ l.id, norm });
        }
    }

    var sha: [32]u8 = undefined;
    if (file.str("glp.content_sha256")) |claimed| {
        const all = digest(file, false) orelse return refuse(why, "the file's tensors point past its end", .{});
        const dirs = digest(file, true) orelse return refuse(why, "the file's tensors point past its end", .{});
        // The file may claim the digest over every tensor or over the directions alone; the match decides.
        const hex_all = std.fmt.bytesToHex(all, .lower);
        const hex_dirs = std.fmt.bytesToHex(dirs, .lower);
        if (std.mem.eql(u8, hex_all[0..], claimed)) {
            sha = all;
        } else if (std.mem.eql(u8, hex_dirs[0..], claimed)) {
            sha = dirs;
        } else {
            return refuse(why, "the file's content digest does not match its vectors", .{});
        }
    } else sha = digest(file, true) orelse std.mem.zeroes([32]u8);

    const hook = file.str("glp.hook_point") orelse residual_site;
    const derived = file.str("glp.derived_at") orelse hook;

    return .{
        .mode = mode,
        .alpha = if (file.find("glp.alpha_default")) |v| switch (v) {
            .f32v => |x| x,
            .f64v => |x| @floatCast(x),
            else => 1.0,
        } else 1.0,
        .hook_point = hook,
        .derived_at = derived,
        .base_model = file.str("glp.base_model") orelse file.str("general.base_model.0.repo_url") orelse "",
        .base_revision = file.str("glp.base_revision") orelse "",
        .method = file.str("glp.method") orelse "",
        .contrast = file.str("glp.contrast") orelse "",
        .width = width,
        .layers = layers.items,
        .sha256 = sha,
        .transferred = !std.mem.eql(u8, derived, hook),
    };
}

/// Check the vector against the model it would steer: its width and the checkpoint it came from.
pub fn checkModel(v: Vector, hidden: usize, base_model: []const u8, why: *Why) Error!void {
    if (v.width != hidden) return refuse(why, "the vector's directions are {d} wide; the model is {d}", .{ v.width, hidden });
    if (base_model.len > 0 and v.base_model.len > 0 and !sameModel(v.base_model, base_model)) {
        return refuse(why, "the vector is for {s}; the model is {s}", .{ v.base_model, base_model });
    }
}

/// The served checkpoint is the vector's model when it is that model or a conversion of it.
fn sameModel(base: []const u8, served: []const u8) bool {
    if (std.mem.eql(u8, base, served)) return true;
    return std.mem.startsWith(u8, nameOf(served), nameOf(base));
}

/// A repo id's model name: what follows its owner.
fn nameOf(repo: []const u8) []const u8 {
    if (std.mem.lastIndexOfScalar(u8, repo, '/')) |i| return repo[i + 1 ..];
    return repo;
}

/// The container's metadata and tensor headers; the arena keeps every allocation.
fn parseContainer(a: std.mem.Allocator, bytes: []const u8, why: *Why) Error!gguf.File {
    var problem: gguf.Problem = .{};
    const file = gguf.parse(a, bytes, &problem) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return refuse(why, "{s}", .{problem.text()});
    };
    return file;
}

/// What a producer writes: the vector's fields and one direction per layer.
pub const WriteSpec = struct {
    mode: Mode = .project,
    alpha: f32 = 1.0,
    hook_point: []const u8 = residual_site,
    derived_at: ?[]const u8 = null,
    base_model: []const u8 = "",
    base_revision: []const u8 = "",
    method: []const u8 = "",
    contrast: []const u8 = "",
    layers: []const Layer,
};

/// Write a vector file the way the spec names it; the returned bytes are owned by the caller.
pub fn write(a: std.mem.Allocator, spec: WriteSpec) Error![]u8 {
    if (spec.layers.len == 0) return error.BadGlpVector;
    if (!std.mem.eql(u8, spec.hook_point, residual_site)) return error.BadGlpVector;

    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const tmp = arena.allocator();

    // The file lists its layers in ascending order, whatever order the caller hands them over in.
    const sorted = try tmp.alloc(Layer, spec.layers.len);
    @memcpy(sorted, spec.layers);
    std.mem.sort(Layer, sorted, {}, struct {
        fn lessThan(_: void, x: Layer, y: Layer) bool {
            return x.id < y.id;
        }
    }.lessThan);

    const width = sorted[0].dir.len;
    for (sorted, 0..) |l, i| {
        if (i > 0 and l.id == sorted[i - 1].id) return error.BadGlpVector;
        if (l.id == 0 or l.dir.len == 0 or l.dir.len != width) return error.BadGlpVector;
        var sum: f64 = 0;
        for (l.dir) |v| sum += @as(f64, v) * @as(f64, v);
        const norm = @sqrt(sum);
        if (!std.math.isFinite(norm) or @abs(norm - 1.0) > 1e-3) return error.BadGlpVector;
    }

    var payload: std.ArrayList(u8) = .empty;
    defer payload.deinit(tmp);
    for (sorted) |l| {
        for (l.dir) |v| {
            var buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &buf, @bitCast(v), .little);
            try payload.appendSlice(tmp, &buf);
        }
    }

    // The digest covers the concatenated tensor bytes, which is what the reader hashes to check the file.
    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    h.update(payload.items);
    var sha: [32]u8 = undefined;
    h.final(&sha);
    const hex = std.fmt.bytesToHex(sha, .lower);

    var ids: std.ArrayList(u8) = .empty;
    defer ids.deinit(tmp);
    for (sorted, 0..) |l, i| {
        if (i > 0) try ids.append(tmp, ',');
        try ids.print(tmp, "{d}", .{l.id});
    }

    const dims = try tmp.alloc(u64, 1);
    dims[0] = width;
    const names = try tmp.alloc([32]u8, sorted.len);
    const tensors = try tmp.alloc(gguf.OutTensor, sorted.len);
    var start: usize = 0;
    for (sorted, tensors, 0..) |l, *t, i| {
        const name = std.fmt.bufPrint(names[i][0..], "direction.{d}", .{l.id}) catch return error.BadGlpVector;
        t.* = .{ .name = name, .dims = dims, .kind = gguf.f32_type, .data = payload.items[start..][0 .. l.dir.len * 4] };
        start += l.dir.len * 4;
    }

    const entries = [_]gguf.Entry{
        .{ .key = "general.architecture", .value = .{ .string = "controlvector" } },
        .{ .key = "general.alignment", .value = .{ .u32v = 32 } },
        .{ .key = "glp.spec_version", .value = .{ .u32v = 1 } },
        .{ .key = "glp.mode", .value = .{ .string = if (spec.mode == .project) "project" else "add" } },
        .{ .key = "glp.rank", .value = .{ .u32v = 1 } },
        .{ .key = "glp.orthonormal", .value = .{ .boolv = true } },
        .{ .key = "glp.alpha_default", .value = .{ .f32v = spec.alpha } },
        .{ .key = "glp.hook_point", .value = .{ .string = spec.hook_point } },
        .{ .key = "glp.derived_at", .value = .{ .string = spec.derived_at orelse spec.hook_point } },
        .{ .key = "glp.base_model", .value = .{ .string = spec.base_model } },
        .{ .key = "glp.base_revision", .value = .{ .string = spec.base_revision } },
        .{ .key = "glp.method", .value = .{ .string = spec.method } },
        .{ .key = "glp.contrast", .value = .{ .string = spec.contrast } },
        .{ .key = "glp.layer_ids_zero_based", .value = .{ .string = ids.items } },
        .{ .key = "glp.content_sha256", .value = .{ .string = hex[0..] } },
    };
    return gguf.write(a, &entries, tensors) catch |err| switch (err) {
        error.BadGguf => error.BadGlpVector,
        error.OutOfMemory => error.OutOfMemory,
    };
}

/// Write a vector read earlier back out, so a file can be rewritten with the same fields.
pub fn writeVector(a: std.mem.Allocator, v: Vector) Error![]u8 {
    return write(a, .{
        .mode = v.mode,
        .alpha = v.alpha,
        .hook_point = v.hook_point,
        .derived_at = v.derived_at,
        .base_model = v.base_model,
        .base_revision = v.base_revision,
        .method = v.method,
        .contrast = v.contrast,
        .layers = v.layers,
    });
}

test {
    _ = @import("glp_test.zig");
}
