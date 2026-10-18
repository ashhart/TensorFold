//! Live graph capacities and actual projection/scan launches, using a recording driver rather than GPU execution.
const std = @import("std");
const q = @import("qwen35");
const cuda = @import("cuda");
const Shape = q.engine.Engine.Shape;

test "a lone window does not launch the widest row and stream capacity" {
    const shape = Shape.init(17, 1, true, true, 1);
    try std.testing.expectEqual(@as(u8, 32), shape.total);
    try std.testing.expectEqual(@as(u8, 1), shape.nseg);
    for (1..129) |rows| {
        const s = Shape.init(rows, 1, true, false, 1);
        try std.testing.expect(s.total >= rows and s.total - rows < 16);
        try std.testing.expectEqual(@as(u8, 1), s.nseg);
        const prompt = Shape.init(rows, 1, false, true, 1);
        try std.testing.expectEqual(rows, prompt.total);
        try std.testing.expectEqual(@as(u8, 1), prompt.nseg);
    }
    for (1..129) |segments| {
        const s = Shape.init(segments, segments, true, false, 1);
        try std.testing.expect(s.nseg >= segments and s.nseg < 2 * segments);
        try std.testing.expect(std.math.isPowerOfTwo(s.nseg));
    }
    const wide = Shape.init(128, 128, true, true, 16);
    try std.testing.expectEqual(@as(u8, 128), wide.total);
    try std.testing.expectEqual(@as(u8, 128), wide.nseg);
    for ([_]u8{ 1, 2, 4, 8, 16 }) |slots| {
        const s = Shape.init(64, 1, true, true, slots);
        try std.testing.expectEqual(slots, s.slots);
        try std.testing.expect(s.record and !s.prompt and s.taps);
        try std.testing.expect(s.key() != Shape.init(64, 2, true, true, slots).key());
        try std.testing.expect(s.key() != Shape.init(64, 1, true, false, slots).key());
    }
}

const Trace = struct {
    var projections: usize = 0;
    var scan_segments: u32 = 0;
    var outputs: [8]u64 = undefined;
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, gz: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        if (@intFromPtr(f) == 2) {
            const part = @as(*const cuda.qlinear.GemvPart, @ptrCast(@alignCast(args.?[2].?))).*;
            outputs[projections] = part.out;
            projections += 1;
        }
        if (@intFromPtr(f) == 3) scan_segments = gz / 4;
        return cuda.abi.success;
    }
};
fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}

test "selected live shape reaches real tiled projections and DeltaNet scan dispatch" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Trace.launch;
    var e: q.engine.Engine = undefined;
    e.d = &d;
    e.stream = .{ .d = &d, .handle = null };
    e.fns.sums = function(&d, 1);
    e.fns.gdn_slots_1 = function(&d, 3);
    e.affine.gemv_fn = function(&d, 2);
    e.affine.gemv_blocks = 1;
    e.affine.pdl = false;
    e.affine.split = null;
    e.tables = .{ .d = &d, .ptr = 0x1000, .len = q.round.bytes };
    e.g.v_heads = 4;
    e.geo = .{ .hid = 512, .nk = 2, .nv = 4, .qh = 4, .kvh = 1, .nl = 1 };
    e.q = 1;
    e.k = 2;
    e.v = 3;
    e.g_ = 4;
    e.beta = 5;
    e.y = 6;
    const proj: q.engine.Proj = .{ .a4 = .{ .w = 10, .s = 20, .b = 30, .n = 4096, .k = 8192, .npad = 4096 } };
    Trace.projections = 0;
    const shape = Shape.init(17, 1, true, false, 1);
    try e.projSums(proj, 0x10000, 0x20000, 0x30000, shape.total, false, 0);
    try e.deltaScan(true, shape.slots, shape.nseg, 0);
    try std.testing.expectEqual(@as(usize, 2), Trace.projections);
    try std.testing.expectEqual(@as(u64, 0x30000), Trace.outputs[0]);
    try std.testing.expectEqual(@as(u64, 0x30000 + 16 * 4096 * 2), Trace.outputs[1]);
    try std.testing.expectEqual(@as(u32, 1), Trace.scan_segments);
}
