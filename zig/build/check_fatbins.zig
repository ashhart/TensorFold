//! Release check: every CUDA fatbin given holds an ELF (SASS) image for each required SM target.
const std = @import("std");

const magic: u32 = 0xBA55ED50;
const elf_kind: u16 = 2;

/// The SM targets of the ELF images in one fatbin file's containers, as a set of SM numbers below 256.
pub fn targets(bytes: []const u8) !std.StaticBitSet(256) {
    var found = std.StaticBitSet(256).empty;
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
            if (read(u16, bytes, entry) == elf_kind and sm < 256) found.set(sm);
            entry = next;
        }
        // containers are 8-byte aligned when several are concatenated
        at = std.mem.alignForward(usize, end, 8);
    }
    return found;
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
    var head: [64]u8 = @splat(0);
    std.mem.writeInt(u16, head[0..2], kind, .little);
    std.mem.writeInt(u32, head[4..8], 64, .little);
    std.mem.writeInt(u64, head[8..16], payload, .little);
    std.mem.writeInt(u32, head[28..32], sm, .little);
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
