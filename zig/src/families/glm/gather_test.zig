//! Every shared affine format and independent gate/up/down layout in prompt gathers.
const std = @import("std");
const affine = @import("affine.zig");
const wide = @import("wide.zig");
const gather = @import("gather.zig");
const mtl = @import("metal");

fn q(bits: u8, group: u16) affine.Affine {
    const empty: affine.Ref = .{ .buf = undefined };
    return .{ .w = empty, .s = empty, .b = empty, .n = 64, .k = 128, .bits = bits, .group = group };
}

test "GLM gathers: every affine width and group has a prompt gather" {
    for (affine.row_bits) |bits| for ([_]u16{ 32, 64 }) |group| {
        const one = q(bits, group);
        const choice = wide.routed(one, one, one);
        try std.testing.expect(choice != .rows);
        if (group == 64 and bits == 4) try std.testing.expectEqual(wide.Routed.mm4, choice);
        if (group == 64 and bits == 2) try std.testing.expectEqual(wide.Routed.mm2, choice);
    };
}

test "GLM gathers: gate up and down choose their own widths and groups" {
    for (affine.row_bits) |gb| for (affine.row_bits) |ub| for (affine.row_bits) |db|
        for ([_]u16{ 32, 64 }) |gg| for ([_]u16{ 32, 64 }) |ug| for ([_]u16{ 32, 64 }) |dg| {
            try std.testing.expect(wide.routed(q(gb, gg), q(ub, ug), q(db, dg)) != .rows);
        };
}

test "GLM gathers: unsupported widths groups and split projections stay on the fallback" {
    const good = q(4, 64);
    var split = good;
    split.more = 1;
    for ([_]affine.Affine{ q(7, 64), q(4, 16), q(4, 128), split }) |bad| {
        try std.testing.expectEqual(wide.Routed.rows, wide.routed(bad, good, good));
        try std.testing.expectEqual(wide.Routed.rows, wide.routed(good, bad, good));
        try std.testing.expectEqual(wide.Routed.rows, wide.routed(good, good, bad));
    }
}

fn ref(id: usize) affine.Ref {
    return .{ .buf = .{ .id = @ptrFromInt(id * 16) }, .off = id * 32 };
}
const Recorder = struct {
    refs: [11]?affine.Ref = @splat(null),
    dims: gather.Dims = undefined,
    pipe: mtl.Pipeline = undefined,
    grid: mtl.Size = undefined,
    threads: mtl.Size = undefined,
    calls: usize = 0,
    pub fn setPipeline(r: *@This(), p: mtl.Pipeline) void {
        r.pipe = p;
    }
    pub fn setBuffer(r: *@This(), b: mtl.Buffer, off: usize, at: usize) void {
        r.refs[at] = .{ .buf = b, .off = off };
    }
    pub fn setValue(r: *@This(), d: anytype, at: usize) void {
        std.debug.assert(at == 4);
        r.dims = d;
    }
    pub fn dispatchGroups(r: *@This(), g: mtl.Size, t: mtl.Size) void {
        r.grid = g;
        r.threads = t;
        r.calls += 1;
    }
};

test "GLM gathers: the real encoder binds independent projection formats and every buffer offset" {
    for (affine.row_bits) |gb| for (affine.row_bits) |ub| for ([_]u16{ 32, 64 }) |gg| for ([_]u16{ 32, 64 }) |ug| {
        var gate = q(gb, gg);
        gate.w = ref(1);
        gate.s = ref(2);
        gate.b = ref(3);
        var up = q(ub, ug);
        up.w = ref(8);
        up.s = ref(9);
        up.b = ref(10);
        const io: gather.Io = .{ .x = ref(11), .out = ref(5), .offsets = ref(6), .order = ref(7) };
        var r: Recorder = .{};
        const p: mtl.Pipeline = .{ .id = @ptrFromInt(16) };
        gather.encode(&r, p, gate, up, io, 37, 5, 8, 80, 7.25);
        try std.testing.expectEqualDeep(gather.Dims{ .k = 128, .n = 64, .group = gg, .bits = gb, .up_group = ug, .up_bits = ub, .pairs = 37, .experts = 5, .repeat = 8, .ldo = 80, .lim = 7.25 }, r.dims);
        inline for (.{ io.x, gate.w, gate.s, gate.b }, 0..) |v, i| try std.testing.expectEqualDeep(v, r.refs[i].?);
        inline for (.{ io.out, io.offsets, io.order, up.w, up.s, up.b }, 5..) |v, i| try std.testing.expectEqualDeep(v, r.refs[i].?);
        try std.testing.expectEqual(p.id, r.pipe.id);
        try std.testing.expectEqualDeep(mtl.Size.of(8, 14, 1), r.grid);
        try std.testing.expectEqualDeep(mtl.Size.of(64, 1, 1), r.threads);
        try std.testing.expectEqual(@as(usize, 1), r.calls);
    };
}

test "GLM gathers: down uses its own format and scatters through the same order without another temporary" {
    for (affine.row_bits) |bits| for ([_]u16{ 32, 64 }) |group| {
        var down = q(bits, group);
        down.w = ref(1);
        down.s = ref(2);
        down.b = ref(3);
        const io: gather.Io = .{ .x = ref(4), .out = ref(5), .offsets = ref(6), .order = ref(7) };
        var r: Recorder = .{};
        gather.encode(&r, .{ .id = @ptrFromInt(16) }, down, null, io, 1, 256, 1, 256, 0);
        try std.testing.expectEqual(@as(i32, bits), r.dims.bits);
        try std.testing.expectEqual(@as(i32, group), r.dims.group);
        try std.testing.expectEqual(@as(i32, 1), r.dims.repeat);
        try std.testing.expectEqualDeep(io.out, r.refs[5].?);
        try std.testing.expectEqualDeep(io.order, r.refs[7].?);
        try std.testing.expectEqualDeep(mtl.Size.of(8, 1, 1), r.grid);
        gather.encode(&r, r.pipe, down, null, io, 0, 256, 1, 256, 0);
        try std.testing.expectEqual(@as(usize, 1), r.calls);
    };
}
