//! A recording loader has no device constructor, compiler, buffers, queue or dispatch implementation.
const std = @import("std");
const real = @import("real_metal");
pub const CompileOptions = real.CompileOptions;
pub const Buffer = real.Buffer;
pub const ComputeEncoder = real.ComputeEncoder;
pub const Queue = real.Queue;
pub const Size = real.Size;
pub const Device = struct {
    units: bool,
    gen: u32,
    pub fn tensorUnits(d: Device) bool {
        return d.units;
    }
    pub fn generation(d: Device) u32 {
        return d.gen;
    }
};
pub const objc = struct {
    pub const Pool = struct {
        pub fn push() Pool {
            return .{};
        }
        pub fn pop(_: Pool) void {}
    };
};
pub const Seen = struct { text: []const u8, options: CompileOptions };
pub const Library = struct {
    pub var allocator: std.mem.Allocator = undefined;
    pub var seen: [3]Seen = undefined;
    pub var count: usize = 0;
    pub fn fromSource(_: Device, text: []const u8, options: CompileOptions) !Library {
        if (std.mem.indexOf(u8, text, "lfm2_embed_b4") != null) {
            if (count == seen.len) return error.UnexpectedLfm2Library;
            seen[count] = .{ .text = try allocator.dupe(u8, text), .options = options };
            count += 1;
        }
        return .{};
    }
    pub fn deinit(_: Library) void {}
};
pub const Pipeline = struct {
    pub fn init(_: Device, _: Library, _: []const u8, _: bool) !Pipeline {
        return .{};
    }
    pub fn deinit(_: Pipeline) void {}
};
