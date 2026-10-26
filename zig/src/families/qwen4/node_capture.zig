//! Diagnostic snapshots own independent buffers until completion; readback never changes the inference arithmetic.
const std = @import("std");
const tap = @import("node_trace.zig");
const quant = @import("core").quant.convert;
pub const Node = struct { name: []const u8, position: usize, kind: tap.Kind, values: []const f64 };
pub const Oracle = struct { nodes: []const Node };
pub const reference = @embedFile("fixtures/native-metal-nodes.json");
pub const Summary = struct { different: usize = 0, checked: usize = 0, first: [32]?usize = @splat(null) };
pub fn Capture(comptime B: type, comptime read: anytype) type {
    return struct {
        const Self = @This();
        const Entry = struct { name: []const u8, buffer: B.Buffer, rows: usize, width: usize, kind: tap.Kind, first: usize };
        a: std.mem.Allocator,
        backend: *B,
        entries: std.ArrayList(Entry) = .empty,
        pub fn sink(s: *Self) tap.Sink(B) {
            return .{ .ctx = s, .emit = emit };
        }
        fn emit(ctx: *anyopaque, name: []const u8, buffer: B.Buffer, offset: usize, rows: usize, width: usize, kind: tap.Kind, first: usize) !void {
            const s: *Self = @ptrCast(@alignCast(ctx));
            if (rows == 0 or width == 0 or first + rows > 32) return error.Qwen4TraceShape;
            const bytes = try std.math.mul(usize, try std.math.mul(usize, rows, width), if (kind == .bf16) 2 else 4);
            try s.entries.ensureUnusedCapacity(s.a, 1);
            const key = try s.a.dupe(u8, name);
            var published = false;
            errdefer if (!published) s.a.free(key);
            const own = try s.backend.allocate(bytes);
            s.entries.appendAssumeCapacity(.{ .name = key, .buffer = own, .rows = rows, .width = width, .kind = kind, .first = first });
            published = true;
            // The entry owns the destination even if encoding fails after a command has taken a reference.
            try s.backend.copy(own, 0, buffer, offset, bytes);
        }
        pub fn close(s: *Self) !void {
            try s.backend.drain();
            for (s.entries.items) |entry| {
                s.backend.release(entry.buffer);
                s.a.free(entry.name);
            }
            s.entries.deinit(s.a);
        }
        pub fn report(s: *Self, oracle: Oracle, verbose: bool) !Summary {
            try s.backend.drain();
            const seen = try s.a.alloc(bool, oracle.nodes.len);
            defer s.a.free(seen);
            @memset(seen, false);
            var summary: Summary = .{};
            var used_rows: usize = 0;
            for (s.entries.items, 0..) |entry, index| {
                const bytes = try s.a.alloc(u8, entry.rows * entry.width * @as(usize, if (entry.kind == .bf16) 2 else 4));
                defer s.a.free(bytes);
                try read(s.backend, entry.buffer, bytes);
                for (0..entry.rows) |r| {
                    const pos = entry.first + r;
                    used_rows = @max(used_rows, pos + 1);
                    var expected: ?Node = null;
                    for (oracle.nodes, 0..) |node, i| if (node.position == pos and std.mem.eql(u8, node.name, entry.name)) {
                        if (seen[i]) return error.Qwen4TraceDuplicate;
                        seen[i] = true;
                        expected = node;
                        break;
                    };
                    if (expected) |node| if (node.kind != entry.kind or node.values.len != entry.width) return error.Qwen4TraceShape;
                    var different: usize = 0;
                    if (verbose) std.debug.print("QWEN4_NODE row={d} name={s} kind={t} got=[", .{ pos, entry.name, entry.kind });
                    for (0..entry.width) |col| {
                        const at = (r * entry.width + col) * @as(usize, if (entry.kind == .bf16) 2 else 4);
                        if (entry.kind == .bf16) {
                            const actual = std.mem.readInt(u16, bytes[at..][0..2], .little);
                            if (expected) |node| if (actual != quant.bf16(@floatCast(node.values[col]))) {
                                different += 1;
                            };
                            if (verbose) std.debug.print("{s}{x:0>4}", .{ if (col == 0) "" else ",", actual });
                        } else {
                            const actual = std.mem.readInt(u32, bytes[at..][0..4], .little);
                            if (verbose) std.debug.print("{s}{x:0>8}", .{ if (col == 0) "" else ",", actual });
                        }
                    }
                    if (verbose) {
                        std.debug.print("] want=[", .{});
                        if (expected) |node| for (node.values, 0..) |value, col| {
                            if (entry.kind == .bf16) std.debug.print("{s}{x:0>4}", .{ if (col == 0) "" else ",", quant.bf16(@floatCast(value)) }) else std.debug.print("{s}{d}", .{ if (col == 0) "" else ",", value });
                        };
                        std.debug.print("] bf16_different={d}\n", .{different});
                    }
                    if (expected != null) summary.checked += 1;
                    summary.different += different;
                    if (different != 0 and summary.first[pos] == null) {
                        summary.first[pos] = index;
                        std.debug.print("QWEN4_FIRST_DIFFERENCE row={d} name={s} values={d}\n", .{ pos, entry.name, different });
                    }
                }
            }
            for (seen, oracle.nodes) |present, node| if (!present) {
                if (verbose) std.debug.print("FAIL QWEN4_MISSING_NODE row={d} name={s}\n", .{ node.position, node.name });
                return error.Qwen4TraceMissing;
            };
            if (verbose) for (summary.first[0..used_rows], 0..) |first, r| {
                if (first == null) std.debug.print("PASS QWEN4_NODES row={d}: no differing BF16 storage node\n", .{r});
            };
            return summary;
        }
    };
}
