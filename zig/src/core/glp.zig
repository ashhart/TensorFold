//! A GLP control vector: the GGUF container's GLP subset, read and written whole. Matt Suiche's format (MIT).

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

/// What the file says about itself: the keys a consumer reads before it reads the directions.
pub const Publication = struct {
    /// What the file calls itself: general.name.
    name: ?[]const u8 = null,
    /// What the file says it is: general.description.
    description: ?[]const u8 = null,
    /// Who derived it: general.author.
    author: ?[]const u8 = null,
    /// The licence the file is under: general.license.
    license: ?[]const u8 = null,
    /// The model family a consumer matches against: controlvector.model_hint.
    model_hint: ?[]const u8 = null,
    /// What the strength does, as it was measured: glp.measured_effect.
    measured_effect: ?[]const u8 = null,
    /// The day the file was made: glp.created.
    created: ?[]const u8 = null,
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
    publication: Publication = .{},
};

/// The control vector a serve applies to the residual stream (--glp).
pub const Glp = struct {
    /// The file's vector: one direction per layer, the mode and the checkpoint it was derived from.
    vector: Vector,
    /// The strength in force: --glp-alpha, or the file's own alpha_default.
    alpha: f32,
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

/// The digest over the tensor bytes, in the container's physical tensor order; the file's own key names the vectors it covers.
fn digest(doc: *const gguf.Document, only_directions: bool) ?[32]u8 {
    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    for (doc.tensors) |t| {
        if (only_directions and directionId(t.name) == null) continue;
        h.update(t.bytes);
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

/// A metadata value the vector reads as text.
fn textOf(doc: *const gguf.Document, key: []const u8) ?[]const u8 {
    const v = doc.get(key) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

/// A metadata value the vector reads as an unsigned count.
fn countOf(doc: *const gguf.Document, key: []const u8) ?u64 {
    const v = doc.get(key) orelse return null;
    return v.unsigned() catch null;
}

/// A metadata value the vector reads as a real number.
fn realOf(doc: *const gguf.Document, key: []const u8) ?f64 {
    const v = doc.get(key) orelse return null;
    return v.float() catch null;
}

/// Read a vector whole and accept it, or refuse it with a reason; the strings and layers live in `a`.
pub fn read(a: std.mem.Allocator, bytes: []const u8, why: *Why) Error!Vector {
    if (bytes.len < 8) return refuse(why, "the file is too short to be a control vector", .{});
    var doc = try parseContainer(a, bytes, why);
    defer doc.deinit();
    const file = &doc;

    var glp_keys: usize = 0;
    for (file.metadata) |e| {
        // The keys whose meaning the spec version decides; alpha and the digest mean the same in every version.
        const interpretation = std.mem.eql(u8, e.key, "glp.mode") or std.mem.eql(u8, e.key, "glp.rank") or
            std.mem.eql(u8, e.key, "glp.orthonormal") or std.mem.eql(u8, e.key, "glp.hook_point") or
            std.mem.eql(u8, e.key, "glp.derived_at") or std.mem.eql(u8, e.key, "glp.dir_scales") or
            std.mem.eql(u8, e.key, "glp.layer_scales");
        if (interpretation) glp_keys += 1;
    }

    const mode_key = if (textOf(file, "glp.mode") != null) "glp.mode" else if (textOf(file, "dspark.mode") != null) "dspark.mode" else "";
    const mode: Mode = if (mode_key.len == 0) .add else blk: {
        const named = textOf(file, mode_key).?;
        if (std.mem.eql(u8, named, "project")) break :blk .project;
        if (std.mem.eql(u8, named, "add")) break :blk .add;
        return refuse(why, "the file's mode is not one this reader applies", .{});
    };

    if (countOf(file, "glp.spec_version")) |version| {
        if (version != 1) return refuse(why, "glp.spec_version {d}; this reader implements version 1", .{version});
    } else if (glp_keys > 0) {
        return refuse(why, "the file carries glp keys but no glp.spec_version", .{});
    }

    if (glp_keys > 0) {
        const rank = countOf(file, "glp.rank") orelse 1;
        if (rank != 1) return refuse(why, "glp.rank {d}; this reader implements rank 1", .{rank});
        const site = textOf(file, "glp.hook_point") orelse return refuse(why, "the file names no glp.hook_point", .{});
        if (!std.mem.eql(u8, site, residual_site)) return refuse(why, "the file's hook point is {s}; this reader applies at {s}", .{ site, residual_site });
    }

    var layers: std.ArrayList(Layer) = .empty;
    var width: usize = 0;
    for (file.tensors) |t| {
        const id = directionId(t.name) orelse continue;
        if (id == 0) return refuse(why, "the file holds direction.0; layer 0 is not steerable", .{});
        if (t.kind != .f32) return refuse(why, "{s} is not fp32", .{t.name});
        if (t.rank != 1) return refuse(why, "{s} is not one dimensional", .{t.name});
        const dim = t.shape[0];
        if (width == 0) width = @intCast(dim) else if (width != dim) return refuse(why, "{s} is {d} wide; the other directions are {d}", .{ t.name, dim, width });
        const raw = t.bytes;
        const dir = try a.alloc(f32, @intCast(dim));
        for (dir, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
        try layers.append(a, .{ .id = id, .dir = dir });
    }
    if (layers.items.len == 0) return refuse(why, "the file holds no direction tensors", .{});
    std.mem.sort(Layer, layers.items, {}, struct {
        fn lessThan(_: void, x: Layer, y: Layer) bool {
            return x.id < y.id;
        }
    }.lessThan);
    // "direction.5" and "direction.05" both parse as layer 5; a file that names one layer twice has no single direction to apply.
    for (layers.items[1..], 1..) |l, i| {
        if (l.id == layers.items[i - 1].id) return refuse(why, "the file holds two directions for layer {d}", .{l.id});
    }

    if (textOf(file, "glp.layer_ids_zero_based")) |list| {
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
    if (textOf(file, "glp.content_sha256")) |claimed| {
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

    const hook = textOf(file, "glp.hook_point") orelse residual_site;
    const derived = textOf(file, "glp.derived_at") orelse hook;

    return .{
        .mode = mode,
        .alpha = if (realOf(file, "glp.alpha_default")) |x| @floatCast(x) else 1.0,
        .hook_point = hook,
        .derived_at = derived,
        .base_model = textOf(file, "glp.base_model") orelse textOf(file, "general.base_model.0.repo_url") orelse "",
        .base_revision = textOf(file, "glp.base_revision") orelse "",
        .method = textOf(file, "glp.method") orelse "",
        .contrast = textOf(file, "glp.contrast") orelse "",
        .width = width,
        .layers = layers.items,
        .sha256 = sha,
        .transferred = !std.mem.eql(u8, derived, hook),
        .publication = .{
            .name = textOf(file, "general.name"),
            .description = textOf(file, "general.description"),
            .author = textOf(file, "general.author"),
            .license = textOf(file, "general.license"),
            .model_hint = textOf(file, "controlvector.model_hint"),
            .measured_effect = textOf(file, "glp.measured_effect"),
            .created = textOf(file, "glp.created"),
        },
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

/// The container's metadata and tensor headers, read with the engine's own GGUF reader; the caller's allocator keeps every allocation.
fn parseContainer(a: std.mem.Allocator, bytes: []const u8, why: *Why) Error!gguf.Document {
    return gguf.Document.parse(a, bytes, .{}) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return refuse(why, "the file is not a GGUF v3 container: {s}", .{@errorName(err)});
    };
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
    /// What the file says about itself; the keys a consumer reads before it reads the directions.
    publication: Publication = .{},
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
    const tensors = try tmp.alloc(OutTensor, sorted.len);
    var start: usize = 0;
    for (sorted, tensors, 0..) |l, *t, i| {
        const name = std.fmt.bufPrint(names[i][0..], "direction.{d}", .{l.id}) catch return error.BadGlpVector;
        t.* = .{ .name = name, .dims = dims, .data = payload.items[start..][0 .. l.dir.len * 4] };
        start += l.dir.len * 4;
    }

    const entries = [_]OutEntry{
        .{ .key = "general.architecture", .value = .{ .string = "controlvector" } },
        .{ .key = "general.alignment", .value = .{ .u32 = 32 } },
        .{ .key = "controlvector.layer_count", .value = .{ .u32 = @intCast(sorted.len) } },
        .{ .key = "glp.spec_version", .value = .{ .u32 = 1 } },
        .{ .key = "glp.mode", .value = .{ .string = if (spec.mode == .project) "project" else "add" } },
        .{ .key = "glp.rank", .value = .{ .u32 = 1 } },
        .{ .key = "glp.orthonormal", .value = .{ .boolean = true } },
        .{ .key = "glp.alpha_default", .value = .{ .f32 = spec.alpha } },
        .{ .key = "glp.hook_point", .value = .{ .string = spec.hook_point } },
        .{ .key = "glp.derived_at", .value = .{ .string = spec.derived_at orelse spec.hook_point } },
        .{ .key = "glp.base_model", .value = .{ .string = spec.base_model } },
        .{ .key = "glp.base_revision", .value = .{ .string = spec.base_revision } },
        .{ .key = "glp.method", .value = .{ .string = spec.method } },
        .{ .key = "glp.contrast", .value = .{ .string = spec.contrast } },
        .{ .key = "glp.layer_ids_zero_based", .value = .{ .string = ids.items } },
        .{ .key = "glp.content_sha256", .value = .{ .string = hex[0..] } },
    };
    var written = std.ArrayList(OutEntry).empty;
    defer written.deinit(tmp);
    try written.appendSlice(tmp, &entries);
    // The model the vector came from, in the shape a consumer reads: the repo it was derived from.
    if (spec.base_model.len > 0) {
        const is_url = std.mem.startsWith(u8, spec.base_model, "http");
        const owner = std.mem.lastIndexOfScalar(u8, spec.base_model, '/');
        const url = if (is_url)
            spec.base_model
        else
            try std.fmt.allocPrint(tmp, "https://huggingface.co/{s}", .{spec.base_model});
        try written.append(tmp, .{ .key = "general.base_model.count", .value = .{ .u32 = 1 } });
        if (!is_url) {
            const org = if (owner) |i| spec.base_model[0..i] else "";
            const repo_name = if (owner) |i| spec.base_model[i + 1 ..] else spec.base_model;
            try written.append(tmp, .{ .key = "general.base_model.0.name", .value = .{ .string = repo_name } });
            try written.append(tmp, .{ .key = "general.base_model.0.organization", .value = .{ .string = org } });
        }
        try written.append(tmp, .{ .key = "general.base_model.0.version", .value = .{ .string = spec.base_revision } });
        try written.append(tmp, .{ .key = "general.base_model.0.repo_url", .value = .{ .string = url } });
    }
    // What the file says about itself, when the producer named it.
    const about = spec.publication;
    try appendText(&written, tmp, "general.name", about.name);
    try appendText(&written, tmp, "general.description", about.description);
    try appendText(&written, tmp, "general.author", about.author);
    try appendText(&written, tmp, "general.license", about.license);
    try appendText(&written, tmp, "controlvector.model_hint", about.model_hint);
    try appendText(&written, tmp, "glp.measured_effect", about.measured_effect);
    try appendText(&written, tmp, "glp.created", about.created);
    return writeContainer(a, written.items, tensors);
}

/// One metadata value the producer writes: a count, a real, a flag or text.
const OutValue = union(enum) {
    u32: u32,
    f32: f32,
    boolean: bool,
    string: []const u8,
};

/// One metadata key and the value it holds.
const OutEntry = struct { key: []const u8, value: OutValue };

/// One tensor the producer writes: its name, its fp32 shape and its bytes.
const OutTensor = struct { name: []const u8, dims: []const u64, data: []const u8 };

/// The ggml type a written tensor carries: plain fp32.
const f32_type: u32 = 0;

/// The alignment every written container pads its tensor data to.
const written_alignment: usize = 32;

/// One more metadata key, when the producer named one.
fn appendText(written: *std.ArrayList(OutEntry), tmp: std.mem.Allocator, key: []const u8, text: ?[]const u8) !void {
    const value = text orelse return;
    try written.append(tmp, .{ .key = key, .value = .{ .string = value } });
}

/// The GGUF v3 bytes for a vector, laid out the way llama.cpp's convention lays a container out.
fn writeContainer(a: std.mem.Allocator, entries: []const OutEntry, tensors: []const OutTensor) Error![]u8 {
    // The tensor offsets are relative to the data section, so their place follows from the bytes alone.
    const offsets = try a.alloc(u64, tensors.len);
    defer a.free(offsets);
    var data_len: usize = 0;
    for (tensors, offsets) |t, *off| {
        data_len = alignUp(data_len, written_alignment);
        off.* = data_len;
        data_len = std.math.add(usize, data_len, t.data.len) catch return error.BadGlpVector;
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "GGUF");
    try putU32(a, &out, 3);
    try putU64(a, &out, tensors.len);
    try putU64(a, &out, entries.len);
    for (entries) |e| try writeEntry(a, &out, e);
    for (tensors, offsets) |t, off| {
        try putString(a, &out, t.name);
        try putU32(a, &out, @intCast(t.dims.len));
        for (t.dims) |d| try putU64(a, &out, d);
        try putU32(a, &out, f32_type);
        try putU64(a, &out, off);
    }
    while (out.items.len % written_alignment != 0) try out.append(a, 0);
    for (tensors) |t| {
        while (out.items.len % written_alignment != 0) try out.append(a, 0);
        try out.appendSlice(a, t.data);
    }
    return out.toOwnedSlice(a);
}

fn writeEntry(a: std.mem.Allocator, out: *std.ArrayList(u8), e: OutEntry) Error!void {
    try putString(a, out, e.key);
    switch (e.value) {
        .u32 => |x| {
            try putU32(a, out, 4);
            try putU32(a, out, x);
        },
        .f32 => |x| {
            try putU32(a, out, 6);
            try putU32(a, out, @bitCast(x));
        },
        .boolean => |x| {
            try putU32(a, out, 7);
            try out.append(a, if (x) 1 else 0);
        },
        .string => |s| {
            try putU32(a, out, 8);
            try putString(a, out, s);
        },
    }
}

fn alignUp(x: usize, a: usize) usize {
    return (x + a - 1) / a * a;
}

fn putU32(a: std.mem.Allocator, out: *std.ArrayList(u8), x: u32) Error!void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, x, .little);
    try out.appendSlice(a, &buf);
}

fn putU64(a: std.mem.Allocator, out: *std.ArrayList(u8), x: u64) Error!void {
    var buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &buf, x, .little);
    try out.appendSlice(a, &buf);
}

fn putString(a: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) Error!void {
    try putU64(a, out, s.len);
    try out.appendSlice(a, s);
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
        .publication = v.publication,
        .layers = v.layers,
    });
}

test {
    _ = @import("glp_test.zig");
    _ = @import("glp_publication_test.zig");
}
