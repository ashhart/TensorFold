//! What the open /health names besides the model: the release, the GPU backend and the GPU's own name.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Read once at startup; an empty field is left out of /health, and a test server given none names none.
pub const Identity = struct { version: []const u8 = "", backend: []const u8 = "", device: []const u8 = "" };

/// A binary's identity: its release always, its engine module's first backend, and the GPU `device` picks, if any.
pub fn of(comptime Engines: type, a: Allocator, version: []const u8, device: ?u32) Identity {
    const name: ?[]const u8 = if (@hasDecl(Engines, "deviceName")) Engines.deviceName(a, device) else null;
    return .{ .version = version, .backend = if (Engines.backends.len > 0) Engines.backends[0] else "", .device = name orelse "" };
}
