//! A fixed-capacity test writer keeps hostile serialized lengths independent of fixture allocation.
const std = @import("std");
const g = @import("gguf.zig");
pub const Fixture = struct {
    bytes: [16384]u8 = @splat(0),
    len: usize = 0,
    pub fn init(nt: u64, nk: u64) Fixture {
        var f: Fixture = .{};
        f.put("GGUF");
        f.int(u32, 3);
        f.int(u64, nt);
        f.int(u64, nk);
        return f;
    }
    pub fn slice(f: *const Fixture) []const u8 {
        return f.bytes[0..f.len];
    }
    pub fn put(f: *Fixture, x: []const u8) void {
        @memcpy(f.bytes[f.len..][0..x.len], x);
        f.len += x.len;
    }
    pub fn int(f: *Fixture, comptime T: type, x: T) void {
        f.set(T, f.len, x);
        f.len += @sizeOf(T);
    }
    pub fn set(f: *Fixture, comptime T: type, at: usize, x: T) void {
        std.mem.writeInt(T, f.bytes[at..][0..@sizeOf(T)], x, .little);
    }
    pub fn string(f: *Fixture, x: []const u8) void {
        f.int(u64, x.len);
        f.put(x);
    }
    pub fn kv(f: *Fixture, key: []const u8, v: g.Value) void {
        f.string(key);
        f.int(u32, @backingInt(v));
        switch (v) {
            .string => |x| f.string(x),
            .float32 => |x| f.int(u32, @bitCast(x)),
            .float64 => |x| f.int(u64, @bitCast(x)),
            .boolean => |x| f.int(u8, @intFromBool(x)),
            .array => unreachable,
            inline else => |x| f.int(@TypeOf(x), x),
        }
    }
    pub fn array(f: *Fixture, key: []const u8, kind: g.MetadataKind, n: u64) void {
        f.string(key);
        f.int(u32, 9);
        f.int(u32, @backingInt(kind));
        f.int(u64, n);
    }
    pub fn tensor(f: *Fixture, name: []const u8, shape: []const u64, kind: g.TensorKind, offset: u64) void {
        f.string(name);
        f.int(u32, @intCast(shape.len));
        for (shape) |d| f.int(u64, d);
        f.int(u32, @backingInt(kind));
        f.int(u64, offset);
    }
    pub fn finish(f: *Fixture, alignment: usize, data: usize) void {
        f.len = std.mem.alignForward(usize, f.len, alignment);
        for (f.bytes[f.len..][0..data], 0..) |*v, i| v.* = @truncate(i *% 37 +% 13);
        f.len += data;
    }
    pub fn rich() Fixture {
        var f = Fixture.init(1, 2);
        f.kv("general.architecture", .{ .string = "test" });
        f.array("tokenizer.ggml.tokens", .string, 2);
        f.string("zero");
        f.string("one");
        f.tensor("w", &.{32}, .q8_0, 0);
        f.finish(32, 34);
        return f;
    }
};
