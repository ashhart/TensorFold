//! A DeepSeek-V4.1 pack folder by its safetensors headers alone, read with pread and no mmap.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const log = std.log.scoped(.dsv41);
const core = @import("core");
const exl3 = @import("core").exl3_layout;
const st = core.safetensors;
const Io = std.Io;

pub const DType = st.DType;

/// One stored tensor: its shard, dtype, shape and where its bytes start in that file.
pub const Info = struct {
    file: u32,
    dtype: DType,
    rank: u8,
    shape: [st.max_rank]usize,
    /// absolute byte offset of the first byte in the shard file
    start: u64,
    nbytes: u64,

    pub fn dims(i: *const Info) []const usize {
        return i.shape[0..i.rank];
    }

    pub fn is(i: *const Info, dtype: DType, shape: []const usize) bool {
        return i.dtype == dtype and std.mem.eql(usize, i.dims(), shape);
    }
};

pub const Pack = struct {
    arena: std.heap.ArenaAllocator,
    /// the folder, else "" for header-only packs (tests, header dumps)
    dir: []const u8 = "",
    /// shards the index lists that the folder does not hold
    missing: u32 = 0,
    files: std.ArrayList([]const u8) = .empty,
    names: std.StringArrayHashMapUnmanaged(Info) = .empty,
    /// the index's weight map (name -> shard); when set, names another shard owns are skipped
    owner: std.StringHashMapUnmanaged([]const u8) = .empty,

    pub fn init(gpa: std.mem.Allocator) Pack {
        return .{ .arena = std.heap.ArenaAllocator.init(gpa) };
    }

    pub fn deinit(p: *Pack) void {
        p.arena.deinit();
        p.* = undefined;
    }

    /// Every shard's header in `dir`: the index's weight map if present, else every *.safetensors file.
    pub fn open(gpa: std.mem.Allocator, io: Io, dir: []const u8) !Pack {
        var p = init(gpa);
        errdefer p.deinit();
        const a = p.arena.allocator();
        p.dir = try a.dupe(u8, dir);
        const cwd = Io.Dir.cwd();
        var shards: std.ArrayList([]const u8) = .empty;
        const index = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        if (cwd.readFileAlloc(io, index, a, .limited(1 << 28))) |text| {
            try p.useIndex(text);
            var it = p.owner.valueIterator();
            while (it.next()) |f| {
                for (shards.items) |s| {
                    if (std.mem.eql(u8, s, f.*)) break;
                } else try shards.append(a, f.*);
            }
        } else |err| switch (err) {
            error.FileNotFound => {
                var d = try cwd.openDir(io, dir, .{ .iterate = true });
                defer d.close(io);
                var it = d.iterate();
                while (try it.next(io)) |e| if (std.mem.endsWith(u8, e.name, ".safetensors")) try shards.append(a, try a.dupe(u8, e.name));
            },
            else => return err,
        }
        if (shards.items.len == 0) return error.NoSafetensors;
        std.mem.sort([]const u8, shards.items, {}, lessStr);
        for (shards.items) |name| {
            const path = try std.fs.path.join(a, &.{ dir, name });
            // a shard the index lists but the folder lacks: its tensors stay unknown (a plan needing one fails)
            const file = cwd.openFile(io, path, .{}) catch |e| switch (e) {
                error.FileNotFound => {
                    p.missing += 1;
                    continue;
                },
                else => return e,
            };
            defer file.close(io);
            var head: [8]u8 = undefined;
            if (try file.readPositionalAll(io, &head, 0) != 8) return error.BadSafetensors;
            const n = std.mem.readInt(u64, &head, .little);
            const len = try file.length(io);
            if (n > len - 8 or n > 1 << 30) return error.BadSafetensors;
            const json = try a.alloc(u8, @intCast(n));
            if (try file.readPositionalAll(io, json, 8) != json.len) return error.BadSafetensors;
            try p.addHeader(name, json, len - 8 - n);
        }
        return p;
    }

    /// model.safetensors.index.json's weight map, before any header is added.
    pub fn useIndex(p: *Pack, json: []const u8) !void {
        const a = p.arena.allocator();
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{});
        if (parsed != .object) return error.BadIndex;
        const map = (parsed.object.get("weight_map") orelse return error.BadIndex);
        if (map != .object) return error.BadIndex;
        if (p.files.items.len != 0 or p.owner.count() != 0 or map.object.count() == 0) return error.BadIndex;
        var check = map.object.iterator();
        while (check.next()) |entry| {
            if (entry.value_ptr.* != .string) return error.BadIndex;
            const file = entry.value_ptr.string;
            if (file.len == 0 or std.mem.indexOfAny(u8, file, "/\\") != null or
                std.mem.eql(u8, file, ".") or std.mem.eql(u8, file, "..")) return error.BadIndex;
        }
        try p.owner.ensureTotalCapacity(a, @intCast(map.object.count()));
        var it = map.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .string) return error.BadIndex;
            p.owner.putAssumeCapacity(e.key_ptr.*, e.value_ptr.string);
        }
    }

    /// A shard's header JSON (`data_len` bytes follow; maxInt(u64) for dumps); a name already indexed is refused.
    pub fn addHeader(p: *Pack, file_name: []const u8, json: []const u8, data_len: u64) !void {
        const a = p.arena.allocator();
        const id: u32 = @intCast(p.files.items.len);
        const file_copy = try a.dupe(u8, file_name);
        const header = try st.parseHeader(a, json, @intCast(data_len));
        var check = header.iterator();
        while (check.next()) |e| {
            if (p.owner.count() > 0) {
                const owner = p.owner.get(e.key_ptr.*) orelse continue;
                if (!std.mem.eql(u8, owner, file_name)) continue;
            }
            if (p.names.contains(e.key_ptr.*)) return error.DuplicateTensor;
            _ = std.math.add(u64, 8 + json.len, e.value_ptr.begin) catch return error.BadSafetensors;
        }
        try p.names.ensureUnusedCapacity(a, header.count());
        try p.files.ensureUnusedCapacity(a, 1);
        p.files.appendAssumeCapacity(file_copy);
        var it = header.iterator();
        while (it.next()) |e| {
            const v = e.value_ptr.*;
            if (p.owner.count() > 0) {
                const f = p.owner.get(e.key_ptr.*) orelse continue;
                if (!std.mem.eql(u8, f, file_name)) continue;
            }
            const gop = p.names.getOrPutAssumeCapacity(e.key_ptr.*);
            if (gop.found_existing) return error.DuplicateTensor;
            gop.value_ptr.* = .{ .file = id, .dtype = v.dtype, .rank = v.rank, .shape = v.shape, .start = 8 + json.len + v.begin, .nbytes = v.end - v.begin };
        }
    }

    /// A header dump (the 8-byte length, then the JSON; nothing after) as shard `file_name`.
    pub fn addHeaderDump(p: *Pack, file_name: []const u8, bytes: []const u8) !void {
        if (bytes.len < 8) return error.BadSafetensors;
        const n = std.mem.readInt(u64, bytes[0..8], .little);
        if (n > bytes.len - 8) return error.BadSafetensors;
        try p.addHeader(file_name, bytes[8..][0..@intCast(n)], std.math.maxInt(u64));
    }

    pub fn get(p: *const Pack, name: []const u8) ?Info {
        return p.names.get(name);
    }

    pub fn has(p: *const Pack, name: []const u8) bool {
        return p.names.contains(name);
    }

    /// A required tensor; a missing name is logged and refused.
    pub fn need(p: *const Pack, name: []const u8) !Info {
        return p.get(name) orelse {
            if (!quiet) log.warn("pack has no tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    pub const Projection = @import("projection.zig").Projection;
    pub fn projection(p: *const Pack, prefix: []const u8) !Projection {
        return @import("projection.zig").read(p, prefix);
    }
    /// Current CUDA rank adapter. Complete dense/packed storage remains available through projection().
    pub fn group(p: *const Pack, prefix: []const u8, buf: []u8) !exl3.Group {
        _ = buf;
        return (try p.projection(prefix)).cudaGroup();
    }

    /// The group's marker tensor name (mul1 or mcg) if it has one; its value picks the codebook at load time.
    pub fn marker(p: *const Pack, prefix: []const u8, buf: []u8) !?[]const u8 {
        var found: ?[]const u8 = null;
        for ([_][]const u8{ "mul1", "mcg" }) |m| {
            const name = try std.fmt.bufPrint(buf, "{s}.{s}", .{ prefix, m });
            if (p.names.getKey(name)) |k| {
                if (found != null) return error.BadMarker;
                const info = p.get(k).?;
                if (!info.is(.i32, &.{}) or info.nbytes != 4) return error.BadMarker;
                found = k;
            }
        }
        return found;
    }

    /// `span` of tensor `info`'s bytes into `out` (span.bytes() long), run by run.
    pub fn read(p: *const Pack, io: Io, info: Info, span: exl3.Span, out: []u8) !void {
        const bytes = std.math.mul(u64, span.len, span.count) catch return error.BadSpan;
        if (out.len != bytes) return error.BadSpan;
        const last = std.math.mul(u64, span.count -| 1, span.stride) catch return error.BadSpan;
        const tail = std.math.add(u64, span.offset, last) catch return error.BadSpan;
        const end = std.math.add(u64, tail, span.len) catch return error.BadSpan;
        if (span.count > 0 and end > info.nbytes) return error.BadSpan;
        _ = std.math.add(u64, info.start, end) catch return error.BadSpan;
        if (bytes == 0) return;
        if (p.dir.len == 0) return error.HeaderOnlyPack;
        var path_buf: [Io.Dir.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ p.dir, p.files.items[info.file] });
        const file = try Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        if (span.count <= 1) {
            if (try file.readPositionalAll(io, out, info.start + span.offset) != out.len) return error.ShortRead;
            dropCache(file, info.start + span.offset, out.len);
            return;
        }
        // a strided part (a column split): one read covering every run, then the runs gathered
        const first = span.at(0);
        const cover: usize = @intCast(span.at(span.count - 1) + span.len - first);
        const tmp = try std.heap.page_allocator.alloc(u8, cover);
        defer std.heap.page_allocator.free(tmp);
        if (try file.readPositionalAll(io, tmp, info.start + first) != cover) return error.ShortRead;
        dropCache(file, info.start + first, cover);
        const len: usize = @intCast(span.len);
        const stride: usize = @intCast(span.stride);
        for (0..@intCast(span.count)) |i| @memcpy(out[i * len ..][0..len], tmp[i * stride ..][0..len]);
    }

    /// Drops the page cache of bytes just read (POSIX_FADV_DONTNEED): on unified memory it competes with the device.
    fn dropCache(file: Io.File, offset: u64, len: usize) void {
        if (@import("builtin").os.tag != .linux) return;
        const linux = std.os.linux;
        _ = linux.fadvise(file.handle, @intCast(offset), @intCast(len), linux.POSIX_FADV.DONTNEED);
    }

    /// The codebook of the group at `prefix`: its marker's value, else the 3-instruction default.
    pub fn codebook(p: *const Pack, io: Io, prefix: []const u8) !exl3.Codebook {
        var buf: [256]u8 = undefined;
        const marker_name = try p.marker(prefix, &buf);
        if (p.get(try std.fmt.bufPrint(&buf, "{s}.trellis", .{prefix}))) |t| {
            const g = try exl3.Group.fromShape(t.dims());
            if (g.k2() % 2 != 0 and (marker_name == null or !std.mem.endsWith(u8, marker_name.?, ".mul1"))) return error.BadMarker;
        }
        const name = marker_name orelse return .inst3;
        const info = p.get(name).?;
        if (info.nbytes < 4) return error.BadMarker;
        var v: [4]u8 = undefined;
        try p.read(io, info, .{ .offset = 0, .len = 4 }, &v);
        return exl3.fromMarker(std.mem.readInt(u32, &v, .little));
    }
};

fn lessStr(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.order(u8, x, y) == .lt;
}

const testing = std.testing;

/// One tensor of an in-memory image.
pub const Spec = struct { name: []const u8, dtype: []const u8, shape: []const usize, fill: u8 = 0 };

/// A safetensors image in memory: tensors in order, each filled with `fill`.
pub fn image(a: std.mem.Allocator, specs: []const Spec) ![]u8 {
    var json: std.Io.Writer.Allocating = .init(a);
    defer json.deinit();
    try json.writer.writeAll("{\"__metadata__\":{\"format\":\"pt\"}");
    var at: usize = 0;
    for (specs) |s| {
        var n = (DType.parse(s.dtype) orelse return error.UnsupportedDType).size();
        for (s.shape) |d| n *= d;
        try json.writer.print(",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ s.name, s.dtype });
        for (s.shape, 0..) |d, i| try json.writer.print("{s}{d}", .{ if (i == 0) "" else ",", d });
        try json.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ at, at + n });
        at += n;
    }
    try json.writer.writeAll("}");
    const head = json.written();
    const out = try a.alloc(u8, 8 + head.len + at);
    std.mem.writeInt(u64, out[0..8], head.len, .little);
    @memcpy(out[8..][0..head.len], head);
    var pos = 8 + head.len;
    for (specs) |s| {
        var n = DType.parse(s.dtype).?.size();
        for (s.shape) |d| n *= d;
        @memset(out[pos..][0..n], s.fill);
        pos += n;
    }
    return out;
}

test "a folder by its headers: index order, groups checked, strided reads, codebook markers" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const a = testing.allocator;
    const one = try image(a, &.{
        .{ .name = "x.trellis", .dtype = "I16", .shape = &.{ 2, 16, 16 }, .fill = 0 },
        .{ .name = "x.suh", .dtype = "F16", .shape = &.{32} },
        .{ .name = "x.svh", .dtype = "F16", .shape = &.{256} },
        .{ .name = "x.mcg", .dtype = "I32", .shape = &.{}, .fill = 0 },
    });
    defer a.free(one);
    // tile (kt, nt) holds bytes of value 16 kt + nt, so a column part's runs are checkable
    const hl = std.mem.readInt(u64, one[0..8], .little);
    const data = one[8 + hl ..];
    for (0..2) |kt| for (0..16) |nt| @memset(data[(kt * 16 + nt) * 32 ..][0..32], @intCast(16 * kt + nt));
    std.mem.writeInt(u32, data[2 * 16 * 32 + 64 + 512 ..][0..4], exl3.marker_mcg, .little);
    const two = try image(a, &.{.{ .name = "y.weight", .dtype = "BF16", .shape = &.{ 3, 2 }, .fill = 7 }});
    defer a.free(two);
    try tmp.dir.writeFile(io, .{ .sub_path = "b.safetensors", .data = one });
    try tmp.dir.writeFile(io, .{ .sub_path = "a.safetensors", .data = two });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var p = try Pack.open(a, io, dir);
    defer p.deinit();
    try testing.expectEqualStrings("a.safetensors", p.files.items[0]);
    var buf: [128]u8 = undefined;
    const g = try p.group("x", &buf);
    try testing.expectEqual(@as(u32, 256), g.n());
    const part = try g.part(.col, 1, 2);
    const out = try a.alloc(u8, @intCast(part.trellis.bytes()));
    defer a.free(out);
    try p.read(io, p.get("x.trellis").?, part.trellis, out);
    for (0..2) |kt| for (0..8) |j| try testing.expectEqual(@as(u8, @intCast(16 * kt + 8 + j)), out[(kt * 8 + j) * 32]);
    try testing.expectEqual(exl3.Codebook.mcg, try p.codebook(io, "x"));
    try testing.expectEqual(exl3.Codebook.inst3, try p.codebook(io, "y"));
    try testing.expectError(error.MissingTensor, p.group("y", &buf));
    try testing.expectError(error.BadSpan, p.read(io, p.get("y.weight").?, .{ .offset = 8, .len = 8 }, out[0..8]));
}
