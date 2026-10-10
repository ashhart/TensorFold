//! Oracle fixtures from tests/cuda/oracle/oracle.py: manifest.json plus one raw little-endian file per array.

const std = @import("std");

pub const Fixture = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    parsed: std.json.Parsed(std.json.Value),

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Fixture {
        const path = try std.fs.path.join(gpa, &.{ dir, "manifest.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        return .{ .gpa = gpa, .io = io, .dir = dir, .parsed = parsed };
    }

    pub fn deinit(self: *Fixture) void {
        self.parsed.deinit();
    }

    fn section(self: Fixture, name: []const u8) !std.json.ObjectMap {
        const v = self.parsed.value.object.get(name) orelse return error.MissingField;
        return v.object;
    }

    pub fn int(self: Fixture, name: []const u8) !i64 {
        const v = (try self.section("params")).get(name) orelse return error.MissingField;
        return v.integer;
    }

    pub fn float(self: Fixture, name: []const u8) !f64 {
        const v = (try self.section("params")).get(name) orelse return error.MissingField;
        return switch (v) {
            .float => |f| f,
            .integer => |i| @floatFromInt(i),
            else => error.WrongType,
        };
    }

    pub fn string(self: Fixture, name: []const u8) ![]const u8 {
        const v = (try self.section("params")).get(name) orelse return error.MissingField;
        return v.string;
    }

    /// The raw bytes of one array; the caller frees them.
    pub fn bytes(self: Fixture, name: []const u8) ![]u8 {
        const entry = (try self.section("arrays")).get(name) orelse return error.MissingField;
        const file = entry.object.get("file").?.string;
        const path = try std.fs.path.join(self.gpa, &.{ self.dir, file });
        defer self.gpa.free(path);
        return std.Io.Dir.cwd().readFileAlloc(self.io, path, self.gpa, .limited(1 << 34)); // 288 experts' tables pass 2 GiB
    }

    /// Any file in the fixture directory (cubins, metadata); the caller frees it.
    pub fn readFile(self: Fixture, name: []const u8) ![]u8 {
        const path = try std.fs.path.join(self.gpa, &.{ self.dir, name });
        defer self.gpa.free(path);
        return std.Io.Dir.cwd().readFileAlloc(self.io, path, self.gpa, .limited(1 << 28));
    }
};
