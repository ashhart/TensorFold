//! Per-engine bounded processor transport buffers, allocated on the first active request.
const std = @import("std");
const cuda = @import("cuda");
const rep = @import("cuda_repetition.zig");

pub const Processor = struct { bits: u64, factors: u64, words: u32 };

pub const Device = struct {
    table: rep.Table = .{},
    masks: ?cuda.DeviceBuffer = null,
    factors: ?cuda.DeviceBuffer = null,
    host_factors: [rep.max_rows]f32 = undefined,
    words: usize = 0,

    pub fn deinit(p: *Device) void {
        if (p.masks) |*b| b.free();
        if (p.factors) |*b| b.free();
    }

    pub fn prepare(p: *Device, d: *const cuda.Driver, stream: cuda.Stream, table: rep.Table, words: usize) !void {
        if (table.len == 0) {
            p.table = table;
            return;
        }
        if (p.masks == null) {
            var masks = try cuda.DeviceBuffer.alloc(d, rep.max_rows * words * 4);
            errdefer masks.free();
            const factors = try cuda.DeviceBuffer.alloc(d, rep.max_rows * 4);
            p.masks = masks;
            p.factors = factors;
            p.words = words;
        }
        if (words != p.words or table.len * words * 4 > p.masks.?.len) return error.BadProcessorTable;
        for (table.rows[0..table.len], 0..) |r, i| p.host_factors[i] = r.factor;
        try p.factors.?.uploadAsync(0, std.mem.sliceAsBytes(p.host_factors[0..table.len]), stream.handle);
        p.table = table;
    }

    pub fn view(p: *const Device, at: usize) Processor {
        std.debug.assert(at < p.table.len);
        return .{ .bits = p.masks.?.ptr + at * p.words * 4, .factors = p.factors.?.ptr + at * 4, .words = @intCast(p.words) };
    }

    pub fn find(p: *const Device, slot: u32, first: u64, rows: usize) !?Processor {
        const at = (try p.table.find(slot, first, rows)) orelse return null;
        return p.view(at);
    }
};
