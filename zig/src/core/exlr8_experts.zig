//! An EXLR8 MoE layer's index (offsets, side data, checksums) and its K2/K3/K4 trellis files, mapped read-only.

const std = @import("std");
const Io = std.Io;
const st = @import("safetensors.zig");
const codec = @import("exlr8_codec.zig");
const xxh = @import("xxhash64.zig");

pub const hidden = 6144;
pub const halves = 16;
pub const routed = 256;
/// The shared expert's number in index slices and in this API.
pub const shared = 256;

pub const Proj = enum(u2) {
    gate,
    up,
    down,

    pub fn name(p: Proj) []const u8 {
        return @tagName(p);
    }
};

/// Bytes of one 128-channel half at width k: 3072 tiles.
pub fn halfBytes(k: u8) usize {
    return 3072 * codec.tileBytes(k);
}

/// One half's trellis bytes and side data, decoded on its own with its width's vectors.
pub const Half = struct {
    k: u8,
    proj: Proj,
    trellis: []const u8,
    /// gate/up: svh of the half's 128 output channels; down: suh of its 128 input channels.
    chan: []align(1) const f16,
    /// gate/up: su_block, down: sv_block; one scale per 128 hidden channels.
    block: []align(1) const f16,
    /// gate/up: signs.in; down: signs.out.
    signs: []const i8,
    checksum: u64,

    pub fn rows(h: Half) usize {
        return if (h.proj == .down) 128 else hidden;
    }
    pub fn cols(h: Half) usize {
        return if (h.proj == .down) hidden else 128;
    }
    pub fn order(h: Half) codec.Order {
        return if (h.proj == .down) .in_major else .out_major;
    }

    /// xxhash64 seed 0 of the trellis bytes, the 128 channel values, then the 48 block scales.
    pub fn verify(h: Half) bool {
        var x = xxh.Hasher.init(0);
        x.update(h.trellis);
        x.update(std.mem.sliceAsBytes(h.chan));
        x.update(std.mem.sliceAsBytes(h.block));
        return x.final() == h.checksum;
    }

    /// The hidden side's scales: the layer's signs times the expert's block scale (exact in fp16).
    fn hiddenScales(h: Half, out: []f32) void {
        for (out, 0..) |*v, i| v.* = @as(f32, @floatFromInt(h.signs[i])) * @as(f32, h.block[i / 128]);
    }
    pub fn suh(h: Half, out: []f32) void {
        if (h.proj == .down) {
            for (out, h.chan) |*v, c| v.* = c;
        } else h.hiddenScales(out);
    }
    pub fn svh(h: Half, out: []f32) void {
        if (h.proj == .down) {
            h.hiddenScales(out);
        } else for (out, h.chan) |*v, c| {
            v.* = c;
        }
    }
};

const Side = struct { index: st.Tensor, checksum: st.Tensor, block: [3]st.Tensor, chan: [3]st.Tensor };

/// A half's place in its width file and what its checksum covers after the trellis bytes: chan, then block.
pub const Entry = struct {
    off: usize,
    len: usize,
    chan: []const u8,
    block: []const u8,
    checksum: u64,

    /// xxhash64 seed 0 of `trellis` (the half's bytes), its 128 channel values, then its 48 block scales.
    pub fn verify(en: Entry, trellis: []const u8) bool {
        var x = xxh.Hasher.init(0);
        x.update(trellis);
        x.update(en.chan);
        x.update(en.block);
        return x.final() == en.checksum;
    }
};

/// experts/exlr8-layer-LLL.kK.safetensors under `root`.
pub fn widthPath(gpa: std.mem.Allocator, root: []const u8, number: u16, k: u8) ![]u8 {
    return std.fmt.allocPrint(gpa, "{s}/experts/exlr8-layer-{d:0>3}.k{d}.safetensors", .{ root, number, k });
}

pub const Layer = struct {
    number: u16,
    index: st.File,
    files: [5]?st.File = @splat(null),
    widths: [5]bool = @splat(false),
    shared_widths: [5]bool = @splat(false),
    signs_in: []const i8,
    signs_out: []const i8,

    /// experts/exlr8-layer-LLL.index.safetensors and the trellis files of every width it names.
    pub fn open(gpa: std.mem.Allocator, io: Io, root: []const u8, number: u16) !Layer {
        var l = try openIndex(gpa, io, root, number);
        errdefer l.close(io);
        for (2..5) |k| if (l.widths[k] or l.shared_widths[k]) {
            const tp = try widthPath(gpa, root, number, @intCast(k));
            defer gpa.free(tp);
            // A node may hold only the widths its manifest or rank needs: a missing file is a width it cannot load.
            l.files[k] = st.File.open(gpa, io, tp) catch |e| if (e == error.FileNotFound) null else return e;
        };
        return l;
    }

    /// The index alone: offsets, side data and checksums, for a node that plans what to fetch.
    pub fn openIndex(gpa: std.mem.Allocator, io: Io, root: []const u8, number: u16) !Layer {
        const path = try std.fmt.allocPrint(gpa, "{s}/experts/exlr8-layer-{d:0>3}.index.safetensors", .{ root, number });
        defer gpa.free(path);
        var index = try st.File.open(gpa, io, path);
        errdefer index.close(io);
        var l: Layer = .{ .number = number, .index = index, .signs_in = try signs(&index, "signs.in"), .signs_out = try signs(&index, "signs.out") };
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const meta = try metadata(arena.allocator(), &index);
        if (!std.mem.eql(u8, meta.get("format") orelse "", "exlr8-v2") or !std.mem.eql(u8, meta.get("contents") orelse "", "index"))
            return error.NotExlr8Index;
        try parseWidths(meta.get("widths") orelse return error.NotExlr8Index, &l.widths);
        try parseWidths(meta.get("shared_widths") orelse "", &l.shared_widths);
        return l;
    }

    pub fn close(l: *Layer, io: Io) void {
        for (&l.files) |*f| if (f.*) |*file| file.close(io);
        l.index.close(io);
        l.* = undefined;
    }

    fn side(l: *const Layer, expert: u16, k: u8) !Side {
        var buf: [48]u8 = undefined;
        const pre = if (expert == shared) "shared." else "";
        const rows: usize = if (expert == shared) 1 else routed;
        var s: Side = undefined;
        s.index = try l.tensor(try std.fmt.bufPrint(&buf, "{s}k{d}.index", .{ pre, k }), .i64, &.{ rows, 3, halves, 2 });
        s.checksum = try l.tensor(try std.fmt.bufPrint(&buf, "{s}k{d}.checksum", .{ pre, k }), .u64, &.{ rows, 3, halves });
        for ([_]Proj{ .gate, .up, .down }, 0..) |p, i| {
            const b = if (p == .down) "sv_block" else "su_block";
            const c = if (p == .down) "suh" else "svh";
            s.block[i] = try l.tensor(try std.fmt.bufPrint(&buf, "{s}k{d}.{s}.{s}", .{ pre, k, p.name(), b }), .f16, &.{ rows, 48 });
            s.chan[i] = try l.tensor(try std.fmt.bufPrint(&buf, "{s}k{d}.{s}.{s}", .{ pre, k, p.name(), c }), .f16, &.{ rows, 2048 });
        }
        return s;
    }

    fn tensor(l: *const Layer, name: []const u8, dtype: st.DType, shape: []const usize) !st.Tensor {
        const t = l.index.get(name) orelse return error.MissingExlr8Tensor;
        if (!t.is(dtype, shape)) return error.BadExlr8Tensor;
        return t;
    }

    /// Whether the layer stores the routed experts (or the shared expert) at width k.
    pub fn stored(l: *const Layer, expert: u16, k: u8) bool {
        if (k > 4) return false;
        return if (expert == shared) l.shared_widths[k] else l.widths[k];
    }

    /// Where half h of projection p of an expert (0-255, or `shared`) sits at width k, from the index alone.
    pub fn entry(l: *const Layer, expert: u16, proj: Proj, h: usize, k: u8) !Entry {
        if (h >= halves or expert > shared) return error.BadExlr8Half;
        if (!l.stored(expert, k)) return error.WidthNotStored;
        const s = try l.side(expert, k);
        const e: usize = if (expert == shared) 0 else expert;
        const p = @backingInt(proj);
        const at = ((e * 3 + p) * halves + h);
        const off = std.math.cast(usize, std.mem.readInt(i64, s.index.bytes[at * 16 ..][0..8], .little)) orelse return error.BadExlr8Offset;
        const len = std.math.cast(usize, std.mem.readInt(i64, s.index.bytes[at * 16 + 8 ..][0..8], .little)) orelse return error.BadExlr8Offset;
        if (len != halfBytes(k)) return error.BadExlr8Offset;
        return .{
            .off = off,
            .len = len,
            .chan = s.chan[p].bytes[(e * 2048 + 128 * h) * 2 ..][0..256],
            .block = s.block[p].bytes[e * 96 ..][0..96],
            .checksum = std.mem.readInt(u64, s.checksum.bytes[at * 8 ..][0..8], .little),
        };
    }

    /// Half h of projection p of an expert (0-255, or `shared`) at width k.
    pub fn half(l: *const Layer, expert: u16, proj: Proj, h: usize, k: u8) !Half {
        const en = try l.entry(expert, proj, h, k);
        const file: *const st.File = &(l.files[k] orelse return error.WidthFileMissing);
        if (en.off < file.data or en.off > file.map.memory.len or en.len > file.map.memory.len - en.off) return error.BadExlr8Offset;
        return .{
            .k = k,
            .proj = proj,
            .trellis = file.map.memory[en.off..][0..en.len],
            .chan = std.mem.bytesAsSlice(f16, en.chan),
            .block = std.mem.bytesAsSlice(f16, en.block),
            .signs = if (proj == .down) l.signs_out else l.signs_in,
            .checksum = en.checksum,
        };
    }
};

fn signs(f: *const st.File, name: []const u8) ![]const i8 {
    const t = f.get(name) orelse return error.MissingExlr8Tensor;
    if (!t.is(.i8, &.{hidden})) return error.BadExlr8Tensor;
    const v = std.mem.bytesAsSlice(i8, t.bytes);
    for (v) |x| if (x != 1 and x != -1) return error.BadExlr8Signs;
    return v;
}

/// The header's __metadata__ strings (the safetensors index skips them).
pub fn metadata(arena: std.mem.Allocator, f: *const st.File) !std.StringArrayHashMapUnmanaged([]const u8) {
    const json = f.map.memory[8..f.data];
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{});
    var out: std.StringArrayHashMapUnmanaged([]const u8) = .empty;
    if (v != .object) return error.BadSafetensors;
    const m = v.object.get("__metadata__") orelse return out;
    if (m != .object) return error.BadSafetensors;
    var it = m.object.iterator();
    while (it.next()) |kv| if (kv.value_ptr.* == .string) try out.put(arena, kv.key_ptr.*, kv.value_ptr.string);
    return out;
}

fn parseWidths(text: []const u8, out: *[5]bool) !void {
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |w| {
        const k = std.fmt.parseInt(u8, std.mem.trim(u8, w, " "), 10) catch return error.BadExlr8Widths;
        if (k < 2 or k > 4) return error.BadExlr8Widths;
        out[k] = true;
    }
}

test "widths parse from the index metadata and refuse what the format does not store" {
    var w: [5]bool = @splat(false);
    try parseWidths("2,3,4", &w);
    try std.testing.expect(w[2] and w[3] and w[4] and !w[1]);
    try std.testing.expectError(error.BadExlr8Widths, parseWidths("3,5", &w));
    try std.testing.expectEqual(@as(usize, 3 * 98304), halfBytes(3));
}
