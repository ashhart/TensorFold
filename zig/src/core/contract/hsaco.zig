//! What hipcc made of each kernel in an AMDGPU code object: the msgpack metadata note's registers, LDS and bounds.
const std = @import("std");
const limits = @import("limits.zig");

/// One kernel's metadata: its name (the symbol launches resolve) and its usage.
pub const Kernel = struct { name: []const u8, usage: limits.HipUsage };

pub const Error = error{ NotACodeObject, BadMetadata, OutOfMemory };

const bundle_magic = "__CLANG_OFFLOAD_BUNDLE__";
const nt_amdgpu_metadata = 32;

/// The device ELF inside `bytes`: a clang offload bundle's amdgcn entry, or `bytes` when it is the ELF itself.
pub fn elf(bytes: []const u8) Error![]const u8 {
    if (std.mem.startsWith(u8, bytes, "\x7fELF")) return bytes;
    if (!std.mem.startsWith(u8, bytes, bundle_magic)) return error.NotACodeObject;
    var at: usize = bundle_magic.len;
    const n = try int(u64, bytes, at);
    at += 8;
    for (0..n) |_| {
        const off = try int(u64, bytes, at);
        const size = try int(u64, bytes, at + 8);
        const len = try int(u64, bytes, at + 16);
        at += 24;
        if (at + len > bytes.len) return error.NotACodeObject;
        const id = bytes[at..][0..len];
        at += len;
        if (std.mem.indexOf(u8, id, "amdgcn-amd-amdhsa") == null) continue;
        if (off + size > bytes.len) return error.NotACodeObject;
        return bytes[off..][0..size];
    }
    return error.NotACodeObject;
}

fn int(comptime T: type, bytes: []const u8, at: usize) Error!T {
    if (at + @sizeOf(T) > bytes.len) return error.NotACodeObject;
    return std.mem.readInt(T, bytes[at..][0..@sizeOf(T)], .little);
}

/// The msgpack descriptor of the code object's AMDGPU metadata note.
pub fn metadata(bytes: []const u8) Error![]const u8 {
    const e = try elf(bytes);
    const shoff = try int(u64, e, 0x28);
    const shnum = try int(u16, e, 0x3c);
    for (0..shnum) |i| {
        const sh = shoff + i * 64;
        if (try int(u32, e, sh + 4) != 7) continue; // SHT_NOTE
        const off = try int(u64, e, sh + 24);
        const size = try int(u64, e, sh + 32);
        var at = off;
        while (at + 12 <= off + size) {
            const namesz = try int(u32, e, at);
            const descsz = try int(u32, e, at + 4);
            const kind = try int(u32, e, at + 8);
            const name = at + 12;
            const desc = name + std.mem.alignForward(usize, namesz, 4);
            if (desc + descsz > e.len) return error.NotACodeObject;
            if (kind == nt_amdgpu_metadata and std.mem.startsWith(u8, e[name..][0..namesz], "AMDGPU"))
                return e[desc..][0..descsz];
            at = desc + std.mem.alignForward(usize, descsz, 4);
        }
    }
    return error.BadMetadata;
}

/// A msgpack value as far as the metadata needs: containers by length, strings, integers, the rest skipped.
const Value = union(enum) { map: usize, array: usize, str: []const u8, uint: u64, other };

const Reader = struct {
    b: []const u8,
    at: usize = 0,

    fn take(r: *Reader, n: usize) Error![]const u8 {
        if (r.at + n > r.b.len) return error.BadMetadata;
        defer r.at += n;
        return r.b[r.at..][0..n];
    }

    fn be(r: *Reader, comptime T: type) Error!T {
        return std.mem.readInt(T, (try r.take(@sizeOf(T)))[0..@sizeOf(T)], .big);
    }

    fn next(r: *Reader) Error!Value {
        const t = (try r.take(1))[0];
        return switch (t) {
            0x00...0x7f => .{ .uint = t },
            0x80...0x8f => .{ .map = t & 0x0f },
            0x90...0x9f => .{ .array = t & 0x0f },
            0xa0...0xbf => .{ .str = try r.take(t & 0x1f) },
            0xc0, 0xc2, 0xc3, 0xe0...0xff => .other,
            0xc4, 0xd9 => r.blob(try r.be(u8), t),
            0xc5, 0xda => r.blob(try r.be(u16), t),
            0xc6, 0xdb => r.blob(try r.be(u32), t),
            0xca => skip(try r.take(4)),
            0xcb => skip(try r.take(8)),
            0xcc => .{ .uint = try r.be(u8) },
            0xcd => .{ .uint = try r.be(u16) },
            0xce => .{ .uint = try r.be(u32) },
            0xcf => .{ .uint = try r.be(u64) },
            0xd0 => skip(try r.take(1)),
            0xd1 => skip(try r.take(2)),
            0xd2 => skip(try r.take(4)),
            0xd3 => skip(try r.take(8)),
            0xdc => .{ .array = try r.be(u16) },
            0xdd => .{ .array = try r.be(u32) },
            0xde => .{ .map = try r.be(u16) },
            0xdf => .{ .map = try r.be(u32) },
            else => error.BadMetadata,
        };
    }

    fn blob(r: *Reader, n: usize, t: u8) Error!Value {
        const s = try r.take(n);
        return if (t >= 0xd9) .{ .str = s } else .other;
    }

    fn skip(_: []const u8) Value {
        return .other;
    }

    /// Skips one whole value, containers included.
    fn pass(r: *Reader) Error!void {
        var left: usize = 1;
        while (left > 0) : (left -= 1) switch (try r.next()) {
            .map => |n| left += 2 * n,
            .array => |n| left += n,
            else => {},
        };
    }

    fn str(r: *Reader) Error![]const u8 {
        return switch (try r.next()) {
            .str => |s| s,
            else => error.BadMetadata,
        };
    }
};

/// Every kernel in the code object `bytes` and what hipcc made of it; names point into `bytes`.
pub fn kernels(a: std.mem.Allocator, bytes: []const u8) Error![]Kernel {
    var r = Reader{ .b = try metadata(bytes) };
    var out: std.ArrayList(Kernel) = .empty;
    errdefer out.deinit(a);
    const top = switch (try r.next()) {
        .map => |n| n,
        else => return error.BadMetadata,
    };
    for (0..top) |_| {
        if (!std.mem.eql(u8, try r.str(), "amdhsa.kernels")) {
            try r.pass();
            continue;
        }
        const n = switch (try r.next()) {
            .array => |n| n,
            else => return error.BadMetadata,
        };
        for (0..n) |_| try out.append(a, try kernel(&r));
    }
    return out.toOwnedSlice(a);
}

fn kernel(r: *Reader) Error!Kernel {
    const n = switch (try r.next()) {
        .map => |n| n,
        else => return error.BadMetadata,
    };
    var k = Kernel{ .name = "", .usage = .{ .vgprs = 0, .bound = 0, .wave = 0 } };
    const fields = .{
        .{ ".vgpr_count", "vgprs" },
        .{ ".sgpr_count", "sgprs" },
        .{ ".group_segment_fixed_size", "lds" },
        .{ ".private_segment_fixed_size", "private" },
        .{ ".max_flat_workgroup_size", "bound" },
        .{ ".wavefront_size", "wave" },
    };
    for (0..n) |_| {
        const key = try r.str();
        if (std.mem.eql(u8, key, ".name")) {
            k.name = try r.str();
            continue;
        }
        const v = try r.next();
        inline for (fields) |f| if (std.mem.eql(u8, key, f[0])) {
            @field(k.usage, f[1]) = switch (v) {
                .uint => |x| @intCast(x),
                else => return error.BadMetadata,
            };
        };
        switch (v) {
            .map => |m| for (0..2 * m) |_| try r.pass(),
            .array => |m| for (0..m) |_| try r.pass(),
            else => {},
        }
    }
    return k;
}

/// The kernel named `name` in a code object's list.
pub fn find(list: []const Kernel, name: []const u8) ?Kernel {
    for (list) |k| if (std.mem.eql(u8, k.name, name)) return k;
    return null;
}

/// A test's msgpack writer: just the forms the metadata holds.
const Pack = struct {
    out: std.ArrayList(u8) = .empty,
    a: std.mem.Allocator,

    fn map(p: *Pack, n: u8) !void {
        try p.out.append(p.a, 0x80 | n);
    }

    fn array(p: *Pack, n: u8) !void {
        try p.out.append(p.a, 0x90 | n);
    }

    fn str(p: *Pack, s: []const u8) !void {
        const n: u8 = @intCast(s.len);
        if (n < 32) try p.out.append(p.a, 0xa0 | n) else try p.out.appendSlice(p.a, &.{ 0xd9, n });
        try p.out.appendSlice(p.a, s);
    }

    fn uint(p: *Pack, v: u32) !void {
        if (v < 128) return p.out.append(p.a, @intCast(v));
        try p.out.append(p.a, 0xcd);
        try p.out.appendSlice(p.a, &std.mem.toBytes(std.mem.nativeToBig(u16, @intCast(v))));
    }
};

/// A minimal ELF64 holding one note section with `desc` as its AMDGPU metadata.
fn testElf(a: std.mem.Allocator, desc: []const u8) ![]u8 {
    const note_at = 64;
    const note_len = 12 + 8 + std.mem.alignForward(usize, desc.len, 4);
    const sh_at = note_at + note_len;
    const e = try a.alloc(u8, sh_at + 2 * 64);
    @memset(e, 0);
    @memcpy(e[0..4], "\x7fELF");
    std.mem.writeInt(u64, e[0x28..][0..8], sh_at, .little);
    std.mem.writeInt(u16, e[0x3c..][0..2], 2, .little);
    std.mem.writeInt(u32, e[note_at..][0..4], 7, .little);
    std.mem.writeInt(u32, e[note_at + 4 ..][0..4], @intCast(desc.len), .little);
    std.mem.writeInt(u32, e[note_at + 8 ..][0..4], nt_amdgpu_metadata, .little);
    @memcpy(e[note_at + 12 ..][0..7], "AMDGPU\x00");
    @memcpy(e[note_at + 20 ..][0..desc.len], desc);
    const sh = sh_at + 64; // section 0 stays null
    std.mem.writeInt(u32, e[sh + 4 ..][0..4], 7, .little);
    std.mem.writeInt(u64, e[sh + 24 ..][0..8], note_at, .little);
    std.mem.writeInt(u64, e[sh + 32 ..][0..8], note_len, .little);
    return e;
}

test "a code object's metadata gives each kernel's VGPRs, SGPRs, LDS, scratch, bound and wave size" {
    const a = std.testing.allocator;
    var p = Pack{ .a = a };
    defer p.out.deinit(a);
    try p.map(2);
    try p.str("amdhsa.target");
    try p.str("amdgcn-amd-amdhsa--gfx1100");
    try p.str("amdhsa.kernels");
    try p.array(2);
    const ks = [_]struct { []const u8, [6]u32 }{
        .{ "tf_rms_rows", .{ 40, 18, 0, 0, 256, 32 } },
        .{ "tf_ctl_big_lds", .{ 8, 10, 40960, 16, 1024, 32 } },
    };
    const keys = [_][]const u8{
        ".vgpr_count",                 ".sgpr_count",              ".group_segment_fixed_size",
        ".private_segment_fixed_size", ".max_flat_workgroup_size", ".wavefront_size",
    };
    for (ks) |k| {
        try p.map(9);
        try p.str(".args");
        try p.array(1);
        try p.map(1);
        try p.str(".size");
        try p.uint(8);
        try p.str(".name");
        try p.str(k[0]);
        for (keys, k[1]) |key, v| {
            try p.str(key);
            try p.uint(v);
        }
        try p.str(".symbol");
        try p.str("x.kd");
    }
    const e = try testElf(a, p.out.items);
    defer a.free(e);
    const got = try kernels(a, e);
    defer a.free(got);
    try std.testing.expectEqual(@as(usize, 2), got.len);
    const big = find(got, "tf_ctl_big_lds").?;
    const want = limits.HipUsage{ .vgprs = 8, .sgprs = 10, .lds = 40960, .private = 16, .bound = 1024, .wave = 32 };
    try std.testing.expectEqual(want, big.usage);
    try std.testing.expectEqual(@as(u32, 256), find(got, "tf_rms_rows").?.usage.bound);
    // the same ELF as the amdgcn entry of a clang offload bundle, after a host entry
    var b: std.ArrayList(u8) = .empty;
    defer b.deinit(a);
    const ids = [_][]const u8{ "host-x86_64-unknown-linux-gnu-", "hipv4-amdgcn-amd-amdhsa--gfx1100" };
    try b.appendSlice(a, bundle_magic);
    try b.appendSlice(a, &std.mem.toBytes(@as(u64, 2)));
    var head: usize = b.items.len;
    for (ids) |id| head += 24 + id.len;
    for (ids, 0..) |id, i| {
        try b.appendSlice(a, &std.mem.toBytes(@as(u64, head)));
        try b.appendSlice(a, &std.mem.toBytes(@as(u64, if (i == 0) 0 else e.len)));
        try b.appendSlice(a, &std.mem.toBytes(@as(u64, id.len)));
        try b.appendSlice(a, id);
    }
    try b.appendSlice(a, e);
    try std.testing.expectEqualSlices(u8, e, try elf(b.items));
    const again = try kernels(a, b.items);
    defer a.free(again);
    try std.testing.expectEqualDeep(got[1].usage, again[1].usage);
    try std.testing.expectError(error.NotACodeObject, elf("not a code object"));
}
