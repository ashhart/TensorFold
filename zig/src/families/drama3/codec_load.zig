//! One FP32 codec schema for prepared safetensors and the restricted native PTH reader.
const std = @import("std");
const core = @import("core");
const pth = @import("codec_pth.zig");
const mapping = @import("codec_map.zig");
const schema = @import("codec_schema.zig");

pub const Source = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    data: union(enum) { prepared: core.safetensors.File, archive: pth.Archive },

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Source {
        return .{ .gpa = gpa, .io = io, .data = if (std.mem.endsWith(u8, path, ".pth"))
            .{ .archive = try pth.Archive.open(gpa, io, path) }
        else
            .{ .prepared = try core.safetensors.File.open(gpa, io, path) } };
    }

    pub fn close(s: *Source) void {
        switch (s.data) {
            .prepared => |*f| f.close(s.io),
            .archive => |*a| a.close(),
        }
    }

    pub fn has(s: *const Source, name: []const u8) bool {
        return switch (s.data) {
            .prepared => |*f| f.get(name) != null,
            .archive => |*a| mapping.has(a, name),
        };
    }

    pub fn read(s: *const Source, name: []const u8) ![]f32 {
        switch (s.data) {
            .prepared => |*f| {
                const t = f.get(name) orelse return error.MissingTensor;
                if (t.dtype != .f32 or t.bytes.len % 4 != 0) return error.BadTensor;
                try schema.validate(name, t.bytes.len / 4);
                const out = try s.gpa.alloc(f32, t.bytes.len / 4);
                @memcpy(std.mem.sliceAsBytes(out), t.bytes);
                return out;
            },
            .archive => |*a| {
                const values = try mapping.read(s.gpa, a, name);
                errdefer s.gpa.free(values);
                try schema.validate(name, values.len);
                return values;
            },
        }
    }
};
