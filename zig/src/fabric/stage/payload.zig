//! The stage protocol's payload prefix (RFC #2005 section 6.4): 8 bytes naming the message and its version.
const std = @import("std");

pub const magic: u32 = 0x5446_5350; // "TFSP"
pub const version: u16 = 1;
pub const size = 8;

/// What a protocol payload carries: a stage's boot description, its ready report, or rank 0's round plan.
pub const Message = enum(u8) { describe = 1, ready = 2, plan = 3 };

/// The backend a stage runs its layers on.
pub const Backend = enum(u8) { cuda = 1, metal = 2 };

/// A KV or row element type; a row dtype must have a whole number of bytes an element (not nvfp4).
pub const Dtype = enum(u8) {
    bf16 = 1,
    f16 = 2,
    f32 = 3,
    f8e4m3 = 4,
    nvfp4 = 5,

    /// Bytes an element, or null for a packed type without a whole byte an element.
    pub fn bytes(d: Dtype) ?u32 {
        return switch (d) {
            .bf16, .f16 => 2,
            .f32 => 4,
            .f8e4m3 => 1,
            .nvfp4 => null,
        };
    }
};

pub const Error = error{ NoRoom, BadLength, NotStagePayload, BadPayloadVersion, WrongMessage, BadReserved };

/// The prefix for `m` at `out[0..8]`.
pub fn put(out: []u8, m: Message) void {
    put32(out, 0, magic);
    std.mem.writeInt(u16, out[4..6], version, .little);
    out[6] = @backingInt(m);
    out[7] = 0;
}

/// The message `p` carries, refused unless it is this protocol's magic and version with its reserved byte 0.
pub fn message(p: []const u8) Error!Message {
    if (p.len < size) return error.BadLength;
    if (get32(p, 0) != magic) return error.NotStagePayload;
    if (std.mem.readInt(u16, p[4..6], .little) != version) return error.BadPayloadVersion;
    const m = std.enums.fromInt(Message, p[6]) orelse return error.WrongMessage;
    if (p[7] != 0) return error.BadReserved;
    return m;
}

/// The prefix of `p` checked, and refused unless it carries `want`.
pub fn expect(p: []const u8, want: Message) Error!void {
    if (try message(p) != want) return error.WrongMessage;
}

pub fn get32(p: []const u8, at: usize) u32 {
    return std.mem.readInt(u32, p[at..][0..4], .little);
}

pub fn put32(p: []u8, at: usize, v: u32) void {
    std.mem.writeInt(u32, p[at..][0..4], v, .little);
}

pub fn get64(p: []const u8, at: usize) u64 {
    return std.mem.readInt(u64, p[at..][0..8], .little);
}

pub fn put64(p: []u8, at: usize, v: u64) void {
    std.mem.writeInt(u64, p[at..][0..8], v, .little);
}

test "stage payload: a prefix names its message, and another magic, version, message or reserved byte is refused" {
    var p: [size]u8 = undefined;
    put(&p, .ready);
    try std.testing.expectEqual(Message.ready, try message(&p));
    try std.testing.expectError(error.WrongMessage, expect(&p, .plan));
    try std.testing.expectError(error.BadLength, message(p[0..7]));
    var bad = p;
    bad[0] ^= 1;
    try std.testing.expectError(error.NotStagePayload, message(&bad));
    bad = p;
    bad[4] = 2;
    try std.testing.expectError(error.BadPayloadVersion, message(&bad));
    bad = p;
    bad[6] = 9;
    try std.testing.expectError(error.WrongMessage, message(&bad));
    bad = p;
    bad[7] = 1;
    try std.testing.expectError(error.BadReserved, message(&bad));
}
