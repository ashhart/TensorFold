const std = @import("std");
extern "c" fn erfc(f64) f64;

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 2) return error.ExpectedOutput;
    const file = try std.Io.Dir.cwd().createFile(init.io, args[1], .{ .exclusive = true });
    defer file.close(init.io);
    try file.writeStreamingAll(init.io, "constant float2 sam_erfc_coefficients[193][6] = {\n");
    for (0..193) |piece| {
        const center = @as(f64, @floatFromInt(piece)) / 32;
        const scale = -@as(f64, 2) / @sqrt(@as(f64, std.math.pi)) * @exp(-center * center);
        try file.writeStreamingAll(init.io, "    {");
        var previous: f64 = 0;
        var current: f64 = 1;
        for (0..6) |power| {
            const coefficient = if (power == 0) erfc(center) else scale * current / @as(f64, @floatFromInt(power));
            const high: f32 = @floatCast(coefficient);
            const low: f32 = @floatCast(coefficient - @as(f64, high));
            try file.writeStreamingAll(init.io, try std.fmt.allocPrint(allocator, "float2({e:.9}f, {e:.9}f){s}", .{ high, low, if (power == 5) "" else ", " }));
            if (power != 0) {
                const next = (-2 * center * current - 2 * previous) / @as(f64, @floatFromInt(power));
                previous = current;
                current = next;
            }
        }
        try file.writeStreamingAll(init.io, "},\n");
    }
    try file.writeStreamingAll(init.io, "};\n");
}
