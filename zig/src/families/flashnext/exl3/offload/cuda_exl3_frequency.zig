const std = @import("std");
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Identity = struct {
    model: [32]u8,
    world: u32,
    rank: u32,
};

pub const decoder_layers = 48;
pub const experts = 512;
pub const format_version: u32 = 1;

pub const Counts = struct {
    decoder: [decoder_layers][experts]u32 = @splat(@splat(0)),
};

const magic = "TFEXL3FQ";
const header_bytes = 64;
const decoder_bytes = decoder_layers * experts * @sizeOf(u32);
pub const max_file_bytes = header_bytes + decoder_bytes + Sha256.digest_length;

pub fn read(gpa: std.mem.Allocator, io: Io, path: []const u8, identity: Identity) !Counts {
    try validateIdentity(identity);
    const file: Io.File = if (comptime @import("builtin").os.tag == .windows) return error.UnsupportedPlatform else .{
        .handle = try std.posix.openat(Io.Dir.cwd().handle, path, .{
            .ACCMODE = .RDONLY,
            .NONBLOCK = true,
            .CLOEXEC = true,
            .NOCTTY = true,
        }, 0),
        .flags = .{ .nonblocking = true },
    };
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.ProfileNotRegular;
    if (stat.size > max_file_bytes) return error.ProfileTooLarge;
    const buffer = try gpa.alloc(u8, max_file_bytes + 1);
    defer gpa.free(buffer);
    var reader = file.reader(io, &.{});
    const length = reader.interface.readSliceShort(buffer) catch return reader.err.?;
    if (length > max_file_bytes) return error.ProfileTooLarge;
    return decode(buffer[0..length], identity);
}

pub fn write(gpa: std.mem.Allocator, io: Io, path: []const u8, identity: Identity, counts: *const Counts) !void {
    try validateIdentity(identity);
    if (comptime @import("builtin").os.tag == .windows) return error.UnsupportedPlatform;
    const basename = std.fs.path.basename(path);
    if (path.len == 0 or std.mem.endsWith(u8, path, "/") or
        std.mem.eql(u8, basename, ".") or std.mem.eql(u8, basename, "..")) return error.InvalidPath;
    const bytes = try gpa.alloc(u8, max_file_bytes);
    defer gpa.free(bytes);
    encode(bytes, identity, counts);
    var atomic = try Io.Dir.cwd().createFileAtomic(io, path, .{
        .permissions = .fromMode(0o600),
        .make_path = false,
        .replace = true,
    });
    defer atomic.deinit(io);
    try atomic.file.writePositionalAll(io, bytes, 0);
    try atomic.file.sync(io);
    try atomic.replace(io);
}

fn validateIdentity(identity: Identity) !void {
    if (identity.world == 0 or identity.rank >= identity.world) return error.InvalidIdentity;
}

fn encode(bytes: []u8, identity: Identity, counts: *const Counts) void {
    @memcpy(bytes[0..8], magic);
    putWord(bytes, 8, format_version);
    putWord(bytes, 12, 0);
    @memcpy(bytes[16..48], &identity.model);
    putWord(bytes, 48, identity.world);
    putWord(bytes, 52, identity.rank);
    putWord(bytes, 56, decoder_layers);
    putWord(bytes, 60, experts);
    var offset: usize = header_bytes;
    for (&counts.decoder) |*layer| for (layer) |count| {
        putWord(bytes, offset, count);
        offset += 4;
    };
    var checksum: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(bytes[0..offset], &checksum, .{});
    @memcpy(bytes[offset..], &checksum);
}

fn decode(bytes: []const u8, identity: Identity) !Counts {
    if (bytes.len < header_bytes + Sha256.digest_length) return error.TruncatedProfile;
    if (!std.mem.eql(u8, bytes[0..8], magic)) return error.MalformedProfile;
    if (getWord(bytes, 8) != format_version) return error.UnsupportedVersion;
    if (getWord(bytes, 12) != 0 or getWord(bytes, 56) != decoder_layers or getWord(bytes, 60) != experts) return error.MalformedProfile;
    const payload_end = header_bytes + decoder_bytes;
    if (bytes.len < max_file_bytes) return error.TruncatedProfile;
    if (bytes.len != max_file_bytes) return error.MalformedProfile;
    var checksum: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(bytes[0..payload_end], &checksum, .{});
    if (!std.mem.eql(u8, bytes[payload_end..], &checksum)) return error.ChecksumMismatch;
    if (!std.mem.eql(u8, bytes[16..48], &identity.model) or
        getWord(bytes, 48) != identity.world or getWord(bytes, 52) != identity.rank) return error.IdentityMismatch;
    var counts: Counts = .{};
    var offset: usize = header_bytes;
    for (&counts.decoder) |*layer| for (layer) |*count| {
        count.* = getWord(bytes, offset);
        offset += 4;
    };
    return counts;
}

fn putWord(bytes: []u8, offset: usize, value: u32) void {
    std.mem.writeInt(u32, bytes[offset..][0..4], value, .little);
}

fn getWord(bytes: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, bytes[offset..][0..4], .little);
}

test {
    _ = @import("cuda_exl3_frequency_test.zig");
}
