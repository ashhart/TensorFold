//! Release check: every CUDA fatbin given holds an ELF (SASS) image for each required SM target.
const std = @import("std");

const magic: u32 = 0xBA55ED50;
const elf_kind: u16 = 2;

/// The SM targets of the ELF images in one fatbin file's containers, as a set of SM numbers below 256.
pub fn targets(bytes: []const u8) !std.StaticBitSet(256) {
    const all = try profiles(bytes);
    var found = std.StaticBitSet(256).empty;
    for (0..256) |sm| if (all.isSet(2 * sm) or all.isSet(2 * sm + 1)) found.set(sm);
    return found;
}

/// An ELF entry's `sm_<n>a` flag: bit 20 of its u32 at offset 40 (nvcc 13.3; the cubin e_flags do not differ).
const arch_specific_word = 40;
const arch_specific_bit: u32 = 1 << 20;

/// The SASS profiles of a fatbin's ELF images: bit `2 * sm + a`, `a` 1 for an arch-specific image.
pub fn profiles(bytes: []const u8) !std.StaticBitSet(512) {
    var found = std.StaticBitSet(512).empty;
    var at: usize = 0;
    while (at < bytes.len) {
        if (bytes.len - at < 16 or read(u32, bytes, at) != magic) return error.NotAFatbin;
        const header: usize = read(u16, bytes, at + 6);
        const size = std.math.cast(usize, read(u64, bytes, at + 8)) orelse return error.BadFatbin;
        const end = std.math.add(usize, at + header, size) catch return error.BadFatbin;
        if (header < 16 or end > bytes.len) return error.BadFatbin;
        var entry = at + header;
        while (entry < end) {
            if (end - entry < 32) return error.BadFatbin;
            const entry_header: usize = read(u32, bytes, entry + 4);
            const payload = std.math.cast(usize, read(u64, bytes, entry + 8)) orelse return error.BadFatbin;
            const next = std.math.add(usize, entry + entry_header, payload) catch return error.BadFatbin;
            if (entry_header < 32 or next > end) return error.BadFatbin;
            const sm = read(u32, bytes, entry + 28);
            const a = entry_header >= arch_specific_word + 4 and read(u32, bytes, entry + arch_specific_word) & arch_specific_bit != 0;
            if (read(u16, bytes, entry) == elf_kind and sm < 256) found.set(2 * sm + @intFromBool(a));
            entry = next;
        }
        // containers are 8-byte aligned when several are concatenated
        at = std.mem.alignForward(usize, end, 8);
    }
    return found;
}

/// One fatbin entry: its kind (2 for an ELF/SASS image, 1 for PTX), SM and arch-specific flag.
pub const Entry = struct { kind: u16, sm: u32, arch_specific: bool };

/// Every entry of every container in `bytes`, in order, PTX included, into `buf`; error.TooMany past its length.
pub fn entries(bytes: []const u8, buf: []Entry) ![]Entry {
    var n: usize = 0;
    var at: usize = 0;
    while (at < bytes.len) {
        if (bytes.len - at < 16 or read(u32, bytes, at) != magic) return error.NotAFatbin;
        const header: usize = read(u16, bytes, at + 6);
        const size = std.math.cast(usize, read(u64, bytes, at + 8)) orelse return error.BadFatbin;
        const end = std.math.add(usize, at + header, size) catch return error.BadFatbin;
        if (header < 16 or end > bytes.len) return error.BadFatbin;
        var entry = at + header;
        while (entry < end) {
            if (end - entry < 32) return error.BadFatbin;
            const entry_header: usize = read(u32, bytes, entry + 4);
            const payload = std.math.cast(usize, read(u64, bytes, entry + 8)) orelse return error.BadFatbin;
            const next = std.math.add(usize, entry + entry_header, payload) catch return error.BadFatbin;
            if (entry_header < 32 or next > end) return error.BadFatbin;
            if (n == buf.len) return error.TooMany;
            const a = entry_header >= arch_specific_word + 4 and read(u32, bytes, entry + arch_specific_word) & arch_specific_bit != 0;
            buf[n] = .{ .kind = read(u16, bytes, entry), .sm = read(u32, bytes, entry + 28), .arch_specific = a };
            n += 1;
            entry = next;
        }
        at = std.mem.alignForward(usize, end, 8);
    }
    return buf[0..n];
}

fn read(comptime T: type, bytes: []const u8, at: usize) T {
    return std.mem.readInt(T, bytes[at..][0..@sizeOf(T)], .little);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.Usage;
    var failed = false;
    for (args[2..]) |path| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30)) catch |e| {
            std.debug.print("missing or unreadable CUDA fatbin: {s} ({t})\n", .{ path, e });
            failed = true;
            continue;
        };
        defer gpa.free(bytes);
        const have = targets(bytes) catch |e| {
            std.debug.print("not a CUDA fatbin: {s} ({t})\n", .{ path, e });
            failed = true;
            continue;
        };
        var it = std.mem.tokenizeScalar(u8, args[1], ',');
        while (it.next()) |text| {
            const sm = try std.fmt.parseInt(u8, text, 10);
            if (!have.isSet(sm)) {
                std.debug.print("{s} has no sm_{d} image: build the release fatbins with -Dsm={s}\n", .{ path, sm, args[1] });
                failed = true;
            }
        }
    }
    if (failed) std.process.exit(1);
}

fn appendEntry(out: *std.ArrayList(u8), kind: u16, sm: u32, payload: usize) !void {
    return appendProfile(out, kind, sm, false, payload);
}

fn appendProfile(out: *std.ArrayList(u8), kind: u16, sm: u32, arch_specific: bool, payload: usize) !void {
    var head: [64]u8 = @splat(0);
    std.mem.writeInt(u16, head[0..2], kind, .little);
    std.mem.writeInt(u32, head[4..8], 64, .little);
    std.mem.writeInt(u64, head[8..16], payload, .little);
    std.mem.writeInt(u32, head[28..32], sm, .little);
    std.mem.writeInt(u32, head[40..44], if (arch_specific) 0x01100011 else 0x01000011, .little);
    try out.appendSlice(std.testing.allocator, &head);
    try out.appendNTimes(std.testing.allocator, 0xee, payload);
}

test "an ELF image counts for its SM target, a PTX one does not, and truncation is refused" {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(std.testing.allocator);
    try appendEntry(&body, elf_kind, 86, 40);
    try appendEntry(&body, 1, 120, 24);
    try appendEntry(&body, elf_kind, 121, 8);
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(std.testing.allocator);
    var head: [16]u8 = @splat(0);
    std.mem.writeInt(u32, head[0..4], magic, .little);
    std.mem.writeInt(u16, head[4..6], 1, .little);
    std.mem.writeInt(u16, head[6..8], 16, .little);
    std.mem.writeInt(u64, head[8..16], body.items.len, .little);
    try file.appendSlice(std.testing.allocator, &head);
    try file.appendSlice(std.testing.allocator, body.items);
    const have = try targets(file.items);
    try std.testing.expect(have.isSet(86) and have.isSet(121) and !have.isSet(120));
    try std.testing.expectError(error.BadFatbin, targets(file.items[0 .. file.items.len - 1]));
    try std.testing.expectError(error.NotAFatbin, targets("not a fatbin at all"));
}

/// One entry of a test fatbin: an ELF image unless `kind` says otherwise.
pub const TestEntry = struct { sm: u32, arch_specific: bool, kind: u16 = elf_kind };

/// A fatbin file of the given entries, for tests.
pub fn testFile(list: []const TestEntry) !std.ArrayList(u8) {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(std.testing.allocator);
    for (list) |e| try appendProfile(&body, e.kind, e.sm, e.arch_specific, 16);
    var file: std.ArrayList(u8) = .empty;
    var head: [16]u8 = @splat(0);
    std.mem.writeInt(u32, head[0..4], magic, .little);
    std.mem.writeInt(u16, head[4..6], 1, .little);
    std.mem.writeInt(u16, head[6..8], 16, .little);
    std.mem.writeInt(u64, head[8..16], body.items.len, .little);
    try file.appendSlice(std.testing.allocator, &head);
    try file.appendSlice(std.testing.allocator, body.items);
    return file;
}

test "profiles: plain and arch-specific images of one SM are told apart, also in one file" {
    var plain = try testFile(&.{.{ .sm = 120, .arch_specific = false }});
    defer plain.deinit(std.testing.allocator);
    var specific = try testFile(&.{.{ .sm = 120, .arch_specific = true }});
    defer specific.deinit(std.testing.allocator);
    var two = try testFile(&.{ .{ .sm = 120, .arch_specific = false }, .{ .sm = 121, .arch_specific = true } });
    defer two.deinit(std.testing.allocator);
    const p = try profiles(plain.items);
    try std.testing.expect(p.isSet(240) and !p.isSet(241) and p.count() == 1);
    const s = try profiles(specific.items);
    try std.testing.expect(s.isSet(241) and !s.isSet(240) and s.count() == 1);
    const t = try profiles(two.items);
    try std.testing.expect(t.isSet(240) and t.isSet(243) and t.count() == 2);
    try std.testing.expect((try targets(specific.items)).isSet(120));
    try std.testing.expectError(error.BadFatbin, profiles(two.items[0 .. two.items.len - 1]));
}

test "entries: every entry in order, PTX and duplicates kept" {
    var file = try testFile(&.{ .{ .sm = 120, .arch_specific = true }, .{ .sm = 120, .arch_specific = true }, .{ .sm = 121, .arch_specific = false, .kind = 1 } });
    defer file.deinit(std.testing.allocator);
    var buf: [8]Entry = undefined;
    const got = try entries(file.items, &buf);
    try std.testing.expectEqual(@as(usize, 3), got.len);
    try std.testing.expect(got[0].kind == elf_kind and got[0].sm == 120 and got[0].arch_specific);
    try std.testing.expect(got[2].kind == 1 and got[2].sm == 121);
    try std.testing.expectError(error.TooMany, entries(file.items, buf[0..2]));
}
