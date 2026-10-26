//! Optional diagnostic taps copy queued tensors before scratch reuse; normal inference leaves the sink absent.
const std = @import("std");
pub const Kind = enum { bf16, f32, i32 };
pub fn Sink(comptime B: type) type {
    return struct {
        ctx: *anyopaque,
        emit: *const fn (*anyopaque, []const u8, B.Buffer, usize, usize, usize, Kind, usize) anyerror!void,
        pub fn tensor(s: @This(), name: []const u8, x: anytype) !void {
            return s.emit(s.ctx, name, x.buffer, x.offset, x.rows, x.width, if (x.kind == .bf16) .bf16 else .f32, 0);
        }
        pub fn view(s: @This(), name: []const u8, v: anytype, rows: usize, width: usize, kind: Kind, first: usize) !void {
            return s.emit(s.ctx, name, v.buffer, v.offset, rows, width, kind, first);
        }
    };
}
pub fn label(buffer: []u8, layer: ?usize, scope: []const u8, node: []const u8) ![]const u8 {
    return if (layer) |l| std.fmt.bufPrint(buffer, "model.layers.{d}.{s}.{s}", .{ l, scope, node }) else if (scope.len == 0) std.fmt.bufPrint(buffer, "{s}", .{node}) else std.fmt.bufPrint(buffer, "{s}.{s}", .{ scope, node });
}
