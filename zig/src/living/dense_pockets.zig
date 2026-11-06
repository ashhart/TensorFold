//! Two-input dense pockets publish immutable device rows and retain the exact accepted host values.
const std = @import("std");
pub const neurons = 64;

pub fn Pockets(comptime B: type, comptime Loader: type) type {
    return struct {
        const Self = @This();
        pub const Rows = struct { buffer: B.Buffer, values: []f32, cap: usize, used: usize };
        a: std.mem.Allocator,
        b: *B,
        hidden: usize,
        layers: []?Rows,
        budget: u64 = std.math.maxInt(u64),
        bytes: u64 = 0,

        pub fn init(a: std.mem.Allocator, b: *B, hidden: usize, layers: usize) !Self {
            if (hidden == 0 or hidden % 64 != 0 or layers == 0 or layers > 64) return Loader.shape;
            _ = try std.math.mul(usize, hidden, neurons * 3 * 4);
            const list = try a.alloc(?Rows, layers);
            @memset(list, null);
            return .{ .a = a, .b = b, .hidden = hidden, .layers = list };
        }

        pub fn close(p: *Self) !void {
            try p.b.drain();
            p.deinitDrained();
        }

        pub fn deinitDrained(p: *Self) void {
            for (p.layers) |slot| if (slot) |rows| {
                p.b.release(rows.buffer);
                p.a.free(rows.values);
            };
            p.a.free(p.layers);
            p.* = undefined;
        }

        pub fn set(p: *Self, layer: usize, gate: []const f32, up: []const f32, down: []const f32, used: usize) !void {
            const n = try std.math.mul(usize, neurons, p.hidden);
            if (layer >= p.layers.len or gate.len != n or up.len != n or down.len != n or used > neurons)
                return Loader.shape;
            for ([_][]const f32{ gate, up, down }) |values| for (values) |v| if (!std.math.isFinite(v))
                return Loader.nonfinite;
            if (used == 0 and p.layers[layer] == null) return;
            const charge = try std.math.add(u64, try B.allocationBytes(3 * n * 4), 3 * n * 4);
            if (charge > p.budget -| p.bytes) return Loader.budget;
            const values = try p.a.alloc(f32, 3 * n);
            errdefer p.a.free(values);
            @memcpy(values[0..n], gate);
            @memcpy(values[n..][0..n], up);
            @memcpy(values[2 * n ..], down);
            try p.b.drain();
            const buffer = try p.b.upload(std.mem.sliceAsBytes(values));
            const old = p.layers[layer];
            p.layers[layer] = .{ .buffer = buffer, .values = values, .cap = neurons, .used = used };
            if (old == null) p.bytes += charge;
            if (old) |rows| {
                p.b.release(rows.buffer);
                p.a.free(rows.values);
            }
        }

        pub fn read(p: *const Self, layer: usize, gate: []f32, up: []f32, down: []f32) !usize {
            const n = try std.math.mul(usize, neurons, p.hidden);
            if (layer >= p.layers.len or gate.len != n or up.len != n or down.len != n) return Loader.shape;
            const rows = p.layers[layer] orelse {
                @memset(gate, 0);
                @memset(up, 0);
                @memset(down, 0);
                return 0;
            };
            @memcpy(gate, rows.values[0..n]);
            @memcpy(up, rows.values[n..][0..n]);
            @memcpy(down, rows.values[2 * n ..]);
            return rows.used;
        }

        pub fn mask(p: *const Self) u64 {
            var bits: u64 = 0;
            for (p.layers, 0..) |slot, l| if (slot) |rows| if (rows.used != 0) {
                bits |= @as(u64, 1) << @intCast(l);
            };
            return bits;
        }

        pub fn load(p: *Self, reader: anytype, c: anytype) !void {
            try Loader.load(p, reader, c);
        }
    };
}
