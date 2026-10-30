//! Stage tensors an encode pass hands back for checks against a reference (float32, named as the reference names them).
const std = @import("std");
const cuda = @import("cuda");
const mel = @import("mel.zig");
const Allocator = std.mem.Allocator;

pub const Item = struct { name: []u8, shape: [2]usize, data: []f32 };

pub const Dump = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    items: std.ArrayList(Item) = .empty,

    pub fn deinit(dm: *Dump) void {
        for (dm.items.items) |it| {
            dm.gpa.free(it.name);
            dm.gpa.free(it.data);
        }
        dm.items.deinit(dm.gpa);
    }

    fn add(dm: *Dump, tag: []const u8, name: []const u8, rows: usize, cols: usize) ![]f32 {
        const n = try std.fmt.allocPrint(dm.gpa, "{s}{s}", .{ tag, name });
        errdefer dm.gpa.free(n);
        const data = try dm.gpa.alloc(f32, rows * cols);
        errdefer dm.gpa.free(data);
        try dm.items.append(dm.gpa, .{ .name = n, .shape = .{ rows, cols }, .data = data });
        return data;
    }

    /// bf16 rows [rows, cols] from the device, after the stream's work so far.
    pub fn device(dm: *Dump, s: cuda.Stream, tag: []const u8, name: []const u8, ptr: u64, rows: usize, cols: usize) !void {
        const raw = try dm.gpa.alloc(u16, rows * cols);
        defer dm.gpa.free(raw);
        try s.synchronize();
        try dm.d.check(dm.d.api.cuMemcpyDtoH_v2(raw.ptr, ptr, raw.len * 2), "cuMemcpyDtoH");
        for (try dm.add(tag, name, rows, cols), raw) |*o, v| o.* = mel.widen(v);
    }

    /// bf16 bits from the host.
    pub fn bits(dm: *Dump, tag: []const u8, name: []const u8, v: []const u16, rows: usize, cols: usize) !void {
        for (try dm.add(tag, name, rows, cols), v[0 .. rows * cols]) |*o, x| o.* = mel.widen(x);
    }

    /// f32 values from the host.
    pub fn floats(dm: *Dump, tag: []const u8, name: []const u8, v: []const f32, rows: usize, cols: usize) !void {
        @memcpy(try dm.add(tag, name, rows, cols), v[0 .. rows * cols]);
    }
};
