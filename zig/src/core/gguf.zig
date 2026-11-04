//! Checked GGUF v3 containers; metadata, tokenizer strings and tensor bytes borrow the input until deinit.
const std = @import("std");
const meta = @import("gguf_value.zig");
const tensor = @import("gguf_tensor.zig");
pub const Limits = meta.Limits;
pub const Value = meta.Value;
pub const Array = meta.Array;
pub const MetadataKind = meta.Kind;
pub const TensorKind = tensor.Kind;
pub const Tensor = tensor.Tensor;
pub const views = @import("gguf_views.zig");
pub const stops = @import("gguf_stops.zig");
pub const affine = @import("gguf_repack.zig");
pub const Entry = struct { key: []const u8, value: Value };
pub const UnsupportedTensor = meta.UnsupportedTensor;

const Header = struct { tensors: usize, metadata: usize, data: usize, alignment: usize };
fn prefix(c: *meta.Cursor) !Header {
    if (!std.mem.eql(u8, try c.take(4), "GGUF")) return error.BadGgufMagic;
    if (try c.int(u32) != 3) return error.UnsupportedGgufVersion;
    const nt = try c.int(u64);
    const nk = try c.int(u64);
    if (nt > (c.bytes.len - c.pos) / 33) return error.TensorCountExceedsFile;
    if (nk > (c.bytes.len - c.pos) / 13) return error.MetadataCountExceedsFile;
    if (nt > c.limits.tensors) return error.TooManyTensors;
    if (nk > c.limits.metadata_entries) return error.TooManyMetadataEntries;
    return .{ .tensors = @intCast(nt), .metadata = @intCast(nk), .data = 0, .alignment = 32 };
}
fn entry(c: *meta.Cursor) !Entry {
    const key = try c.string();
    if (key.len == 0 or std.mem.indexOfScalar(u8, key, 0) != null) return error.InvalidMetadataKey;
    const kind = try c.kind();
    return .{ .key = key, .value = try c.value(kind, 0) };
}
fn alignEntry(e: Entry, h: *Header) !void {
    if (!std.mem.eql(u8, e.key, "general.alignment")) return;
    if (e.value != .uint32) return error.InvalidAlignmentType;
    const a = e.value.uint32;
    if (a == 0 or !std.math.isPowerOfTwo(a)) return error.InvalidAlignment;
    h.alignment = a;
}
fn dataStart(c: *meta.Cursor, h: *Header) !void {
    const add = (h.alignment - c.pos % h.alignment) % h.alignment;
    if (add > c.bytes.len - c.pos) return error.TruncatedAlignmentPadding;
    h.data = c.pos + add;
}
fn range(t: Tensor, h: Header, len: usize) !void {
    if (t.offset % h.alignment != 0) return error.UnalignedTensorOffset;
    const available = len - h.data;
    if (t.offset > available or t.len > available - t.offset) return error.TensorOutsideFile;
}
fn cursor(bytes: []const u8, limits: Limits) meta.Cursor {
    return .{ .bytes = bytes, .ceiling = bytes.len, .limits = limits };
}
fn metadataCeiling(c: *meta.Cursor) void {
    c.ceiling = c.pos + @min(c.limits.metadata_bytes, c.bytes.len - c.pos);
}
/// The entire serialized index and every tensor range are checked before allocating an index or array.
fn validate(bytes: []const u8, limits: Limits, unsupported: ?*UnsupportedTensor) !Header {
    var c = cursor(bytes, limits);
    c.unsupported = unsupported;
    var h = try prefix(&c);
    metadataCeiling(&c);
    for (0..h.metadata) |_| try alignEntry(try entry(&c), &h);
    c.ceiling = bytes.len;
    const tensor_start = c.pos;
    for (0..h.tensors) |_| _ = try Tensor.read(&c);
    try dataStart(&c, &h);
    c.pos = tensor_start;
    for (0..h.tensors) |_| try range(try Tensor.read(&c), h, bytes.len);
    return h;
}
pub const Document = struct {
    gpa: std.mem.Allocator,
    bytes: []const u8,
    metadata: []Entry,
    /// Tensors are ordered by physical offset after checking overlap, independently of their serialized order.
    tensors: []Tensor,
    names: std.StringHashMapUnmanaged(usize) = .empty,
    keys: std.StringHashMapUnmanaged(usize) = .empty,
    data_offset: usize,
    alignment: usize,

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, limits: Limits) !Document {
        return parseDiagnosed(gpa, bytes, limits, null);
    }
    /// Optional unsupported-type details borrow the input and survive a refused parse without allocation.
    pub fn parseDiagnosed(gpa: std.mem.Allocator, bytes: []const u8, limits: Limits, unsupported: ?*UnsupportedTensor) !Document {
        if (unsupported) |p| p.* = .{};
        const h = try validate(bytes, limits, unsupported);
        const metadata = try gpa.alloc(Entry, h.metadata);
        errdefer gpa.free(metadata);
        const tensors = try gpa.alloc(Tensor, h.tensors);
        errdefer gpa.free(tensors);
        var keys: std.StringHashMapUnmanaged(usize) = .empty;
        errdefer keys.deinit(gpa);
        var names: std.StringHashMapUnmanaged(usize) = .empty;
        errdefer names.deinit(gpa);
        var c = cursor(bytes, limits);
        c.unsupported = unsupported;
        _ = try prefix(&c);
        for (metadata, 0..) |*m, i| {
            m.* = try entry(&c);
            const pos = try keys.getOrPut(gpa, m.key);
            if (pos.found_existing) return error.DuplicateMetadataKey;
            pos.value_ptr.* = i;
        }
        for (tensors) |*t| {
            t.* = try Tensor.read(&c);
            try range(t.*, h, bytes.len);
            t.bytes = bytes[h.data + t.offset ..][0..t.len];
        }
        std.mem.sort(Tensor, tensors, {}, struct {
            fn less(_: void, a: Tensor, b: Tensor) bool {
                return a.offset < b.offset;
            }
        }.less);
        var end: usize = 0;
        for (tensors, 0..) |t, i| {
            if (t.offset < end) return error.OverlappingTensors;
            end = t.offset + t.len;
            const pos = try names.getOrPut(gpa, t.name);
            if (pos.found_existing) return error.DuplicateTensorName;
            pos.value_ptr.* = i;
        }
        return .{ .gpa = gpa, .bytes = bytes, .metadata = metadata, .tensors = tensors, .names = names, .keys = keys, .data_offset = h.data, .alignment = h.alignment };
    }
    pub fn deinit(d: *Document) void {
        d.names.deinit(d.gpa);
        d.keys.deinit(d.gpa);
        d.gpa.free(d.tensors);
        d.gpa.free(d.metadata);
        d.* = undefined;
    }
    pub fn get(d: *const Document, key: []const u8) ?Value {
        return d.metadata[d.keys.get(key) orelse return null].value;
    }
    pub fn getTensor(d: *const Document, name: []const u8) ?Tensor {
        return d.tensors[d.names.get(name) orelse return null];
    }
    pub fn config(d: *const Document) !views.Config {
        return views.config(d);
    }
    pub fn tokenizer(d: *const Document) !views.Tokenizer {
        return views.tokenizer(d);
    }
};
/// File diagnostics own their tensor name because refused mappings are immediately released.
pub const FileProblem = struct {
    name: ?[]u8 = null,
    kind: u32 = 0,
    pub fn deinit(p: *FileProblem, a: std.mem.Allocator) void {
        if (p.name) |name| a.free(name);
        p.* = .{};
    }
};
pub const File = struct {
    file: std.Io.File,
    map: std.Io.File.MemoryMap,
    document: Document,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8, limits: Limits) !File {
        return openDiagnosed(gpa, io, path, limits, null);
    }
    pub fn openDiagnosed(gpa: std.mem.Allocator, io: std.Io, path: []const u8, limits: Limits, problem: ?*FileProblem) !File {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const len = std.math.cast(usize, try file.length(io)) orelse return error.GgufFileTooLarge;
        if (len < 24) return error.TruncatedInput;
        var map = try std.Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        errdefer map.destroy(io);
        var unsupported: UnsupportedTensor = .{};
        var doc = Document.parseDiagnosed(gpa, map.memory, limits, &unsupported) catch |err| {
            if (problem) |p| if (err == error.UnsupportedTensorType) {
                p.* = .{ .name = try gpa.dupe(u8, unsupported.name), .kind = unsupported.kind };
            };
            return err;
        };
        errdefer doc.deinit();
        if (try file.length(io) != len) return error.GgufFileChanged;
        return .{ .file = file, .map = map, .document = doc };
    }
    pub fn close(f: *File, io: std.Io) void {
        f.document.deinit();
        f.map.destroy(io);
        f.file.close(io);
        f.* = undefined;
    }
};

test {
    _ = @import("gguf_test.zig");
    _ = affine;
    _ = stops;
}
