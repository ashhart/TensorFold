//! Bounded host candidate dimensions; no engine, memory probe or allocation is made here.
const std = @import("std");

pub const quantum: u32 = 256;
pub const max_cells: u32 = 65536;
pub const Input = struct { upper: u32, exact: bool, streams: u32, fixed: bool };
pub const Cell = struct { configured_window: u32, streams: u32 };

/// Enumerate the exact upper endpoint, then lower 256-token tiers, with stream counts descending.
pub const Census = struct {
    input: Input,
    window: u32,
    streams: u32,

    pub fn init(input: Input) !Census {
        if (input.upper == 0 or input.streams == 0) return error.BadFitRequest;
        const windows: u64 = if (input.exact) 1 else (@as(u64, input.upper) + quantum - 1) / quantum;
        const counts: u64 = if (input.fixed) 1 else input.streams;
        if (windows * counts > max_cells) return error.FitCensusTooLarge;
        return .{ .input = input, .window = input.upper, .streams = input.streams };
    }

    pub fn next(c: *Census) ?Cell {
        if (c.window == 0) return null;
        const out = Cell{ .configured_window = c.window, .streams = c.streams };
        if (!c.input.fixed and c.streams > 1) {
            c.streams -= 1;
        } else {
            c.streams = c.input.streams;
            c.window = if (c.input.exact) 0 else if (c.window % quantum != 0)
                c.window / quantum * quantum
            else
                c.window - quantum;
        }
        return out;
    }
};

/// Caller supplies the existing constructor's reserve and pool constants, never another guessed geometry.
pub const Shape = struct { reserve: u32, max_rows: u32, pooled: bool, block: u32, kpool: u32 };
pub const Dimensions = struct { configured: u32, capacity: u32, advertised: u32 };

pub fn dimensions(configured: u32, shape: Shape) !Dimensions {
    if (configured == 0 or shape.max_rows == 0) return error.BadFitRequest;
    const capacity = try std.math.add(u32, configured, shape.reserve);
    const tail = try std.math.add(u32, shape.max_rows, 1);
    const usable = if (shape.pooled) blk: {
        if (shape.block == 0 or shape.kpool == 0) return error.BadFitRequest;
        break :blk (capacity / shape.block * shape.block) -| shape.kpool;
    } else capacity;
    const advertised = @min(configured, usable -| tail);
    if (advertised == 0) return error.NoContextCapacity;
    return .{ .configured = configured, .capacity = capacity, .advertised = advertised };
}

test "fit census preserves an unaligned endpoint then visits every lower tier once" {
    var c = try Census.init(.{ .upper = 513, .streams = 2, .exact = false, .fixed = false });
    const expected = [_]Cell{
        .{ .configured_window = 513, .streams = 2 }, .{ .configured_window = 513, .streams = 1 },
        .{ .configured_window = 512, .streams = 2 }, .{ .configured_window = 512, .streams = 1 },
        .{ .configured_window = 256, .streams = 2 }, .{ .configured_window = 256, .streams = 1 },
    };
    for (expected) |want| try std.testing.expectEqual(want, c.next().?);
    try std.testing.expect(c.next() == null);
    try std.testing.expect(c.next() == null);
}

test "fit census explicit window and fixed streams keep their exact counts" {
    var c = try Census.init(.{ .upper = 4097, .streams = 8, .exact = true, .fixed = true });
    try std.testing.expectEqual(Cell{ .configured_window = 4097, .streams = 8 }, c.next().?);
    try std.testing.expect(c.next() == null);
    c = try Census.init(.{ .upper = 1, .streams = 1, .exact = false, .fixed = true });
    try std.testing.expectEqual(@as(u32, 1), c.next().?.configured_window);
    try std.testing.expect(c.next() == null);
}

test "fit census is bounded before iteration and u32 extremes never wrap" {
    _ = try Census.init(.{ .upper = 131072, .streams = 128, .exact = false, .fixed = false });
    try std.testing.expectError(error.FitCensusTooLarge, Census.init(.{ .upper = 131073, .streams = 128, .exact = false, .fixed = false }));
    try std.testing.expectError(error.FitCensusTooLarge, Census.init(.{ .upper = std.math.maxInt(u32), .streams = std.math.maxInt(u32), .exact = false, .fixed = false }));
    try std.testing.expectError(error.BadFitRequest, Census.init(.{ .upper = 0, .streams = 1, .exact = false, .fixed = false }));
}

test "fit dimensions reproduce existing pooled and private cache advertisements" {
    var shape = Shape{ .reserve = 64, .max_rows = 16, .pooled = true, .block = 256, .kpool = 4 };
    try std.testing.expectEqual(Dimensions{ .configured = 131072, .capacity = 131136, .advertised = 131051 }, try dimensions(131072, shape));
    shape.pooled = false;
    try std.testing.expectEqual(@as(u32, 131072), (try dimensions(131072, shape)).advertised);
    shape.pooled = true;
    try std.testing.expectError(error.NoContextCapacity, dimensions(1, shape));
    try std.testing.expectError(error.Overflow, dimensions(std.math.maxInt(u32), shape));
    shape.block = 0;
    try std.testing.expectError(error.BadFitRequest, dimensions(4096, shape));
}
