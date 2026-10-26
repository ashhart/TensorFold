//! A GLP control vector: the GGUF container's GLP subset read whole, and the metadata rules that accept or refuse it.

const std = @import("std");

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

/// The gguf value types the container may hold.
const Type = enum(u32) {
    u8v = 0,
    i8v = 1,
    u16v = 2,
    i16v = 3,
    u32v = 4,
    i32v = 5,
    f32v = 6,
    boolv = 7,
    string = 8,
    array = 9,
    u64v = 10,
    i64v = 11,
    f64v = 12,
};

/// A metadata value; strings and arrays borrow from the file's bytes or the arena.
const Value = union(Type) {
    u8v: u8,
    i8v: i8,
    u16v: u16,
    i16v: i16,
    u32v: u32,
    i32v: i32,
    f32v: f32,
    boolv: bool,
    string: []const u8,
    array: []Value,
    u64v: u64,
    i64v: i64,
    f64v: f64,
};

const Entry = struct { key: []const u8, value: Value };

/// One tensor's header: its name, its shape, its ggml type and its offset in the data section.
const Tensor = struct { name: []const u8, dims: []const u64, kind: u32, offset: u64 };

/// The ggml type a GLP direction uses.
const f32_type = 0;

fn refuse(why: *Why, comptime fmt: []const u8, args: anytype) Error {
    why.set(fmt, args);
    return error.BadGlpVector;
}

/// A cursor over the file's bytes that refuses to read past its end.
const Cursor = struct {
    b: []const u8,
    o: usize = 0,

    fn need(self: *Cursor, n: usize) Error!void {
        if (n > self.b.len or self.o + n > self.b.len) return error.BadGlpVector;
    }

    fn u8v(self: *Cursor) Error!u8 {
        try self.need(1);
        defer self.o += 1;
        return self.b[self.o];
    }

    fn u32v(self: *Cursor) Error!u32 {
        try self.need(4);
        defer self.o += 4;
        return std.mem.readInt(u32, self.b[self.o..][0..4], .little);
    }

    fn u64v(self: *Cursor) Error!u64 {
        try self.need(8);
        defer self.o += 8;
        return std.mem.readInt(u64, self.b[self.o..][0..8], .little);
    }

    fn f32v(self: *Cursor) Error!f32 {
        try self.need(4);
        defer self.o += 4;
        return @bitCast(std.mem.readInt(u32, self.b[self.o..][0..4], .little));
    }

    fn string(self: *Cursor) Error![]const u8 {
        const n = try self.u64v();
        if (n > self.b.len) return error.BadGlpVector;
        try self.need(@intCast(n));
        defer self.o += @intCast(n);
        return self.b[self.o..][0..@intCast(n)];
    }
};

fn alignUp(x: usize, a: usize) usize {
    return (x + a - 1) / a * a;
}

fn find(kv: []const Entry, key: []const u8) ?Value {
    for (kv) |e| if (std.mem.eql(u8, e.key, key)) return e.value;
    return null;
}

fn str(kv: []const Entry, key: []const u8) ?[]const u8 {
    const v = find(kv, key) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn num(kv: []const Entry, key: []const u8) ?u64 {
    const v = find(kv, key) orelse return null;
    return switch (v) {
        .u8v => |x| x,
        .u16v => |x| x,
        .u32v => |x| x,
        .u64v => |x| x,
        .i8v => |x| @intCast(x),
        .i16v => |x| @intCast(x),
        .i32v => |x| @intCast(x),
        .i64v => |x| @intCast(x),
        else => null,
    };
}

/// Read the container's metadata and tensor headers; the arena keeps every allocation.
const Container = struct {
    kv: []Entry,
    tensors: []Tensor,
    data_start: usize,
    alignment: usize,
};

fn parseContainer(a: std.mem.Allocator, bytes: []const u8, why: *Why) Error!Container {
    var c: Cursor = .{ .b = bytes };
    const magic = c.b[0..4];
    if (!std.mem.eql(u8, magic, "GGUF")) return refuse(why, "not a GGUF container", .{});
    c.o = 4;
    const version = try c.u32v();
    if (version != 3) return refuse(why, "GGUF version {d}; the GLP container is version 3", .{version});
    const tensor_count = try c.u64v();
    const kv_count = try c.u64v();
    if (tensor_count > 1 << 20 or kv_count > 1 << 16) return refuse(why, "the container's counts are out of range", .{});

    var kv: std.ArrayList(Entry) = .empty;
    for (0..kv_count) |_| {
        const key = try c.string();
        const kind = std.enums.fromInt(Type, try c.u32v()) orelse return refuse(why, "the container holds an unknown metadata type", .{});
        const value = try readValue(a, &c, kind);
        try kv.append(a, .{ .key = key, .value = value });
    }

    var tensors: std.ArrayList(Tensor) = .empty;
    for (0..tensor_count) |_| {
        const name = try c.string();
        const n_dims = try c.u32v();
        if (n_dims > 8) return refuse(why, "a tensor has {d} dimensions", .{n_dims});
        const dims = try a.alloc(u64, n_dims);
        for (dims) |*d| d.* = try c.u64v();
        const kind = try c.u32v();
        const offset = try c.u64v();
        try tensors.append(a, .{ .name = name, .dims = dims, .kind = kind, .offset = offset });
    }

    const alignment: usize = if (find(kv.items, "general.alignment")) |v| switch (v) {
        .u32v => |x| x,
        .u64v => |x| @intCast(x),
        else => 32,
    } else 32;
    if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) return refuse(why, "the container's alignment is {d}", .{alignment});

    return .{
        .kv = kv.items,
        .tensors = tensors.items,
        .data_start = alignUp(c.o, alignment),
        .alignment = alignment,
    };
}

fn readValue(a: std.mem.Allocator, c: *Cursor, kind: Type) Error!Value {
    return switch (kind) {
        .u8v => .{ .u8v = try c.u8v() },
        .i8v => .{ .i8v = @bitCast(try c.u8v()) },
        .u16v => .{ .u16v = @truncate(try c.u32v()) },
        .i16v => .{ .i16v = @bitCast(@as(u16, @truncate(try c.u32v()))) },
        .u32v => .{ .u32v = try c.u32v() },
        .i32v => .{ .i32v = @bitCast(try c.u32v()) },
        .f32v => .{ .f32v = try c.f32v() },
        .boolv => .{ .boolv = (try c.u8v()) != 0 },
        .string => .{ .string = try c.string() },
        .u64v => .{ .u64v = try c.u64v() },
        .i64v => .{ .i64v = @bitCast(try c.u64v()) },
        .f64v => .{ .f64v = @bitCast(try c.u64v()) },
        .array => blk: {
            const elem = std.enums.fromInt(Type, try c.u32v()) orelse return error.BadGlpVector;
            const n = try c.u64v();
            if (n > 1 << 20) return error.BadGlpVector;
            const items = try a.alloc(Value, @intCast(n));
            for (items) |*it| it.* = try readValue(a, c, elem);
            break :blk .{ .array = items };
        },
    };
}

/// The tensor's bytes, by its offset in the data section.
fn tensorBytes(bytes: []const u8, c: Container, t: Tensor) ?[]const u8 {
    var n: u64 = 1;
    for (t.dims) |d| {
        if (d == 0 or n > (1 << 40) / d) return null;
        n *= d;
    }
    const width: usize = if (t.kind == f32_type) 4 else 2;
    const size = n * width;
    const start = c.data_start + @as(usize, @intCast(t.offset));
    if (start < c.data_start or size > bytes.len or start + size > bytes.len) return null;
    return bytes[start..][0..size];
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
fn digest(bytes: []const u8, c: Container, only_directions: bool) ?[32]u8 {
    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    for (c.tensors) |t| {
        if (only_directions and directionId(t.name) == null) continue;
        const raw = tensorBytes(bytes, c, t) orelse return null;
        h.update(raw);
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

/// Read a vector whole and accept it, or refuse it with a reason; the strings and layers live in `a`.
pub fn read(a: std.mem.Allocator, bytes: []const u8, why: *Why) Error!Vector {
    if (bytes.len < 8) return refuse(why, "the file is too short to be a control vector", .{});
    const c = try parseContainer(a, bytes, why);

    var glp_keys: usize = 0;
    for (c.kv) |e| {
        // The keys whose meaning the spec version decides; alpha and the digest mean the same in every version.
        const interpretation = std.mem.eql(u8, e.key, "glp.mode") or std.mem.eql(u8, e.key, "glp.rank") or
            std.mem.eql(u8, e.key, "glp.orthonormal") or std.mem.eql(u8, e.key, "glp.hook_point") or
            std.mem.eql(u8, e.key, "glp.derived_at") or std.mem.eql(u8, e.key, "glp.dir_scales") or
            std.mem.eql(u8, e.key, "glp.layer_scales");
        if (interpretation) glp_keys += 1;
    }

    const mode_key = if (str(c.kv, "glp.mode") != null) "glp.mode" else if (str(c.kv, "dspark.mode") != null) "dspark.mode" else "";
    const mode: Mode = if (mode_key.len == 0) .add else if (std.mem.eql(u8, str(c.kv, mode_key).?, "project")) .project else if (std.mem.eql(u8, str(c.kv, mode_key).?, "add")) .add else return refuse(why, "the file's mode is not one this reader applies", .{});

    if (num(c.kv, "glp.spec_version")) |version| {
        if (version != 1) return refuse(why, "glp.spec_version {d}; this reader implements version 1", .{version});
    } else if (glp_keys > 0) {
        return refuse(why, "the file carries glp keys but no glp.spec_version", .{});
    }

    if (glp_keys > 0) {
        const rank = num(c.kv, "glp.rank") orelse 1;
        if (rank != 1) return refuse(why, "glp.rank {d}; this reader implements rank 1", .{rank});
        const site = str(c.kv, "glp.hook_point") orelse return refuse(why, "the file names no glp.hook_point", .{});
        if (!std.mem.eql(u8, site, residual_site)) return refuse(why, "the file's hook point is {s}; this reader applies at {s}", .{ site, residual_site });
    }

    var layers: std.ArrayList(Layer) = .empty;
    var width: usize = 0;
    for (c.tensors) |t| {
        const id = directionId(t.name) orelse continue;
        if (id == 0) return refuse(why, "the file holds direction.0; layer 0 is not steerable", .{});
        if (t.kind != f32_type) return refuse(why, "{s} is not fp32", .{t.name});
        if (t.dims.len != 1) return refuse(why, "{s} is not one dimensional", .{t.name});
        if (width == 0) width = @intCast(t.dims[0]) else if (width != t.dims[0]) return refuse(why, "{s} is {d} wide; the other directions are {d}", .{ t.name, t.dims[0], width });
        const raw = tensorBytes(bytes, c, t) orelse return refuse(why, "{s} points past the file", .{t.name});
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

    if (str(c.kv, "glp.layer_ids_zero_based")) |list| {
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
    if (str(c.kv, "glp.content_sha256")) |claimed| {
        const all = digest(bytes, c, false) orelse return refuse(why, "the file's tensors point past its end", .{});
        const dirs = digest(bytes, c, true) orelse return refuse(why, "the file's tensors point past its end", .{});
        if (!std.mem.eql(u8, &std.fmt.bytesToHex(all, .lower), claimed) and !std.mem.eql(u8, &std.fmt.bytesToHex(dirs, .lower), claimed)) {
            return refuse(why, "the file's content digest does not match its vectors", .{});
        }
        sha = all;
    } else sha = digest(bytes, c, true) orelse std.mem.zeroes([32]u8);

    const hook = str(c.kv, "glp.hook_point") orelse residual_site;
    const derived = str(c.kv, "glp.derived_at") orelse hook;

    return .{
        .mode = mode,
        .alpha = if (find(c.kv, "glp.alpha_default")) |v| switch (v) {
            .f32v => |x| x,
            .f64v => |x| @floatCast(x),
            else => 1.0,
        } else 1.0,
        .hook_point = hook,
        .derived_at = derived,
        .base_model = str(c.kv, "glp.base_model") orelse str(c.kv, "general.base_model.0.repo_url") orelse "",
        .base_revision = str(c.kv, "glp.base_revision") orelse "",
        .method = str(c.kv, "glp.method") orelse "",
        .contrast = str(c.kv, "glp.contrast") orelse "",
        .width = width,
        .layers = layers.items,
        .sha256 = sha,
        .transferred = !std.mem.eql(u8, derived, hook),
    };
}

/// Check the vector against the model it would steer: its width and the checkpoint it came from.
pub fn checkModel(v: Vector, hidden: usize, base_model: []const u8, why: *Why) Error!void {
    if (v.width != hidden) return refuse(why, "the vector's directions are {d} wide; the model is {d}", .{ v.width, hidden });
    if (base_model.len > 0 and v.base_model.len > 0 and !std.mem.eql(u8, base_model, v.base_model)) {
        return refuse(why, "the vector is for {s}; the model is {s}", .{ v.base_model, base_model });
    }
}

test {
    _ = @import("glp_test.zig");
}
