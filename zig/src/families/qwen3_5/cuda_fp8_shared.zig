//! One row quantization for compatible prompt projections, with no state cached across calls.
const std = @import("std");
const cuda = @import("cuda");
const capacity = 8;

const Range = struct {
    begin: u64,
    end: u64,
    fn of(address: u64, rows: usize, columns: usize) ?Range {
        if (address == 0) return null;
        const count = std.math.mul(usize, rows, columns) catch return null;
        const bytes = std.math.mul(usize, count, 2) catch return null;
        return .{ .begin = address, .end = std.math.add(u64, address, bytes) catch return null };
    }
    fn intersects(a: Range, b: Range) bool {
        return a.begin < b.end and b.begin < a.end;
    }
};

fn readSafe(output: Range, scratch: Range, address: u64, count: usize, bytes: usize) bool {
    const size = std.math.mul(usize, count, bytes) catch return false;
    const end = std.math.add(u64, address, size) catch return false;
    if (address == 0 or size == 0) return false;
    const source = Range{ .begin = address, .end = end };
    return !output.intersects(source) and !scratch.intersects(source);
}

fn affineSafe(output: Range, scratch: Range, w: cuda.qlinear.Affine4) bool {
    const codes = std.math.mul(usize, w.npad, w.k) catch return false;
    const groups = std.math.mul(usize, w.npad, w.k / 64) catch return false;
    return readSafe(output, scratch, w.w, codes / 2, 1) and
        readSafe(output, scratch, w.s, groups, 2) and readSafe(output, scratch, w.b, groups, 2);
}

fn weightsSafe(projection: anytype, output: Range, scratch: Range) bool {
    return switch (projection) {
        .a4 => |w| affineSafe(output, scratch, w),
        .bf => |w| readSafe(output, scratch, w.w, std.math.mul(usize, w.n, w.k) catch return false, 2),
        .ql => |q| switch (q) {
            .bf16 => |w| readSafe(output, scratch, w.codes, std.math.mul(usize, w.npad, w.k) catch return false, 2),
            .fp8g, .fp8 => |w| blk: {
                const codes = std.math.mul(usize, w.npad, w.k) catch return false;
                if (!readSafe(output, scratch, w.codes, codes, 1)) break :blk false;
                if (q == .fp8) break :blk true;
                const scales = std.math.mul(usize, w.npad, w.k / 64) catch return false;
                break :blk readSafe(output, scratch, w.scales, scales, 4);
            },
            else => false,
        },
        else => false,
    };
}

fn vendorUses(e: anytype, model: anytype, projection: anytype, rows: usize) bool {
    if (comptime !@hasField(@TypeOf(model.*), "vendor")) return false;
    const vendor = if (model.vendor) |*v| v else return false;
    const address = switch (projection) {
        .a4 => |w| w.w,
        .ql => |q| switch (q) {
            .fp8g, .fp8 => |w| w.codes,
            else => return false,
        },
        else => return false,
    };
    // The vendor path only accepts weights owned by this target engine's arena.
    if (address < e.arena.ptr or address - e.arena.ptr >= e.arena.len) return false;
    return switch (projection) {
        .a4 => |w| if (comptime @hasDecl(@TypeOf(vendor.*), "usesAffine4")) vendor.usesAffine4(rows, w) else true,
        .ql => |q| switch (q) {
            .fp8g, .fp8 => |w| vendor.usesFp8(rows, w),
            else => false,
        },
        else => false,
    };
}

pub fn run(e: anytype, projections: anytype, input: u64, outputs: []const u64, rows: usize, prompt: bool) !u64 {
    if (!prompt or e.learning != null or projections.len == 0 or projections.len > capacity or projections.len != outputs.len)
        return 0;
    const model = if (e.mo) |*m| m else return 0;
    const fast = if (model.fast8) |*n| n else return 0;
    const k = projections[0].inputs();
    const source = Range.of(input, rows, k) orelse return 0;
    const scratch = Range{ .begin = fast.scratch.ptr, .end = std.math.add(u64, fast.scratch.ptr, fast.scratch.len) catch return 0 };
    // Reordering a shared group is valid only when every output and all source weights stay disjoint.
    for (projections, outputs, 0..) |projection, out, index| {
        if (projection.inputs() != k) return 0;
        const span = Range.of(out, rows, projection.outputs()) orelse return 0;
        if (span.intersects(source) or span.intersects(scratch)) return 0;
        for (projections) |p| if (!weightsSafe(p, span, scratch)) return 0;
        for (projections[0..index], outputs[0..index]) |prior, address| {
            const previous = Range.of(address, rows, prior.outputs()) orelse return 0;
            if (span.intersects(previous)) return 0;
        }
    }
    var row_weights: [capacity]cuda.qmmf.Weight = undefined;
    var affine_weights: [capacity]cuda.qlinear.Affine4 = undefined;
    var row_outputs: [capacity]u64 = undefined;
    var affine_outputs: [capacity]u64 = undefined;
    var row_count: usize = 0;
    var affine_count: usize = 0;
    var row_mask: u64 = 0;
    var affine_mask: u64 = 0;
    for (projections, outputs, 0..) |projection, out, index| {
        if (vendorUses(e, model, projection, rows)) continue;
        const bit = @as(u64, 1) << @intCast(index);
        if (projection == .a4) {
            affine_weights[affine_count] = projection.a4;
            affine_outputs[affine_count] = out;
            affine_count += 1;
            affine_mask |= bit;
        } else if (e.g.kinds.ct and projection == .ql and projection.ql == .fp8g) {
            row_weights[row_count] = projection.ql.fp8g;
            row_outputs[row_count] = out;
            row_count += 1;
            row_mask |= bit;
        }
    }
    const row_shared = row_count >= 2 and fast.canShareRows(input, k, rows, row_weights[0..row_count], row_outputs[0..row_count], .bf16);
    const affine_shared = affine_count >= 2 and fast.canShareAffine4(input, k, rows, affine_weights[0..affine_count], affine_outputs[0..affine_count], .bf16);
    if (row_shared) try fast.rowScaledMany(e.stream, input, k, rows, row_weights[0..row_count], row_outputs[0..row_count], .bf16);
    if (affine_shared) try fast.affine4Many(e.stream, input, k, rows, affine_weights[0..affine_count], affine_outputs[0..affine_count], .bf16);
    return (if (row_shared) row_mask else 0) | (if (affine_shared) affine_mask else 0);
}

const TestProjection = union(enum) {
    a4: cuda.qlinear.Affine4,
    ql: union(enum) { fp8g: cuda.qmmf.Weight, fp8: cuda.qmmf.Weight, bf16: cuda.qmmf.Weight, other: cuda.qmmf.Weight },
    bf: struct { w: u64 = 0xe00000, n: usize, k: usize },
    other: struct { n: usize, k: usize },
    fn inputs(p: TestProjection) usize {
        return switch (p) {
            .bf => p.bf.k,
            .a4 => p.a4.k,
            .other => p.other.k,
            .ql => switch (p.ql) {
                inline else => |q| q.k,
            },
        };
    }
    fn outputs(p: TestProjection) usize {
        return switch (p) {
            .bf => p.bf.n,
            .a4 => p.a4.n,
            .other => p.other.n,
            .ql => switch (p.ql) {
                inline else => |q| q.n,
            },
        };
    }
};
const Recorder = struct {
    count: usize = 0,
    affine_count: usize = 0,
    projections: usize = 0,
    input: u64 = 0,
    scratch: struct { ptr: u64 = 0xf0000000, len: usize = 65536 } = .{},
    fn canShareRows(_: *Recorder, _: u64, _: usize, _: usize, _: []const cuda.qmmf.Weight, _: []const u64, _: cuda.qmmf.Out) bool {
        return true;
    }
    fn rowScaledMany(r: *Recorder, _: u8, input: u64, _: usize, _: usize, weights: []const cuda.qmmf.Weight, _: []const u64, _: cuda.qmmf.Out) !void {
        try std.testing.expect(weights.len >= 2);
        r.count += 1;
        r.projections = weights.len;
        r.input = input;
    }
    fn canShareAffine4(_: *Recorder, _: u64, _: usize, _: usize, _: []const cuda.qlinear.Affine4, _: []const u64, _: cuda.qmmf.Out) bool {
        return true;
    }
    fn affine4Many(r: *Recorder, _: u8, input: u64, _: usize, _: usize, weights: []const cuda.qlinear.Affine4, _: []const u64, _: cuda.qmmf.Out) !void {
        try std.testing.expect(weights.len >= 2);
        r.affine_count += 1;
        r.projections = weights.len;
        r.input = input;
    }
};
const Vendor = struct {
    pub fn usesAffine4(_: *const Vendor, _: usize, w: cuda.qlinear.Affine4) bool {
        return w.n == 48;
    }
    pub fn usesFp8(_: *const Vendor, _: usize, w: cuda.qmmf.Weight) bool {
        return w.n == 48;
    }
};
const Engine = struct {
    learning: ?u8 = null,
    g: struct { kinds: struct { ct: bool = true } = .{} } = .{},
    mo: ?struct { fast8: ?Recorder = .{}, vendor: ?Vendor = null } = .{},
    arena: struct { ptr: u64 = 0x100000, len: usize = 0x10000000 } = .{},
    stream: u8 = 0,
};
fn testProjection(n: u32) TestProjection {
    return .{ .ql = .{ .fp8g = .{ .mode = .fp8g, .n = n, .npad = (n + 127) / 128 * 128, .k = 5120, .codes = 0x100000, .scales = 0x200000 } } };
}

fn testAffine(n: usize) TestProjection {
    return .{ .a4 = .{ .w = 0x100000, .s = 0x2000000, .b = 0x2100000, .n = n, .npad = (n + 127) / 128 * 128, .k = 5120 } };
}

fn testBf16(n: u32, address: u64) TestProjection {
    return .{ .ql = .{ .bf16 = .{ .mode = .bf16, .codes = address, .scales = 0, .n = n, .npad = (n + 127) / 128 * 128, .k = 5120 } } };
}

test "FP8 shared full CT QKV and Z admit BF16 A and B and check their complete padded storage" {
    var e: Engine = .{};
    const ps = [_]TestProjection{ testProjection(10240), testProjection(6144), testBf16(48, 0x1f000000), testBf16(48, 0x20000000) };
    var outs = [_]u64{ 0x4000000, 0x8000000, 0xc000000, 0xd000000 };
    try std.testing.expectEqual(@as(u64, 3), try run(&e, &ps, 0x3000000, &outs, 128, true));
    try std.testing.expectEqual(@as(usize, 1), e.mo.?.fast8.?.count);
    try std.testing.expectEqual(@as(usize, 2), e.mo.?.fast8.?.projections);
    const weight = ps[2].ql.bf16;
    outs[2] = weight.codes + @as(u64, weight.n) * weight.k * 2 + 16;
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, 0x3000000, &outs, 128, true));
    try std.testing.expectEqual(@as(usize, 1), e.mo.?.fast8.?.count);
    outs[2] = weight.codes + @as(u64, weight.npad) * weight.k * 2;
    try std.testing.expectEqual(@as(u64, 3), try run(&e, &ps, 0x3000000, &outs, 128, true));
    try std.testing.expectEqual(@as(usize, 2), e.mo.?.fast8.?.count);
}

test "FP8 shared prompt selection groups QKV and Z while preserving other projections and decode" {
    var e: Engine = .{};
    const ps = [_]TestProjection{ testProjection(10240), testProjection(6144), .{ .bf = .{ .n = 48, .k = 5120 } }, .{ .bf = .{ .n = 48, .k = 5120 } } };
    const outs = [_]u64{ 0x4000000, 0x8000000, 0xc000000, 0xd000000 };
    try std.testing.expectEqual(@as(u64, 3), try run(&e, &ps, 0x3000000, &outs, 128, true));
    try std.testing.expectEqual(@as(usize, 1), e.mo.?.fast8.?.count);
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, 0x3000000, &outs, 128, false));
    try std.testing.expectEqual(@as(usize, 1), e.mo.?.fast8.?.count);
    try std.testing.expectEqual(@as(u64, 3), try run(&e, &ps, 0x3100000, &outs, 128, true));
    try std.testing.expectEqual(@as(u64, 0x3100000), e.mo.?.fast8.?.input);
}

test "FP8 shared prompt selection refuses reordering aliases, different widths and non-CT or learning paths" {
    var e: Engine = .{};
    var ps = [_]TestProjection{ testProjection(10240), testProjection(6144), .{ .bf = .{ .n = 48, .k = 5120 } } };
    const outs = [_]u64{ 0x4000000, 0x8000000, 0xc000000 };
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, outs[0], &outs, 128, true));
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, 0x3000000, &.{ outs[0], outs[1], outs[0] }, 128, true));
    ps[2].bf.k = 256;
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, 0x3000000, &outs, 128, true));
    ps[2].bf.k = 5120;
    e.g.kinds.ct = false;
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, 0x3000000, &outs, 128, true));
    e.g.kinds.ct = true;
    e.learning = 1;
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, 0x3000000, &outs, 128, true));
    try std.testing.expectEqual(@as(usize, 0), e.mo.?.fast8.?.count);
}

test "FP8 shared affine selection handles the non-CT mixed model and leaves vendor pieces separate" {
    var e: Engine = .{};
    e.g.kinds.ct = false;
    const ps = [_]TestProjection{ testAffine(10240), testAffine(6144), testAffine(48), testAffine(48) };
    const outs = [_]u64{ 0x4000000, 0x8000000, 0xc000000, 0xd000000 };
    try std.testing.expectEqual(@as(u64, 15), try run(&e, &ps, 0x3000000, &outs, 128, true));
    try std.testing.expectEqual(@as(usize, 1), e.mo.?.fast8.?.affine_count);
    try std.testing.expectEqual(@as(usize, 4), e.mo.?.fast8.?.projections);
    e.mo.?.vendor = .{};
    try std.testing.expectEqual(@as(u64, 3), try run(&e, &ps, 0x3100000, &outs, 128, true));
    try std.testing.expectEqual(@as(usize, 2), e.mo.?.fast8.?.projections);
    try std.testing.expectEqual(@as(u64, 0x3100000), e.mo.?.fast8.?.input);
    e.arena.ptr = 0x30000000;
    try std.testing.expectEqual(@as(u64, 15), try run(&e, &ps, 0x3100000, &outs, 128, true));
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, 0x3100000, &outs, 128, false));
    e.learning = 1;
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, 0x3100000, &outs, 128, true));
    try std.testing.expectEqual(@as(usize, 3), e.mo.?.fast8.?.affine_count);
    e.learning = null;
    e.mo.?.fast8 = null;
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, 0x3100000, &outs, 128, true));
}

test "FP8 shared affine selection rejects every reordering alias and incompatible width before launch" {
    var e: Engine = .{};
    const outs = [_]u64{ 0x4000000, 0x8000000, 0xc000000 };
    var ps = [_]TestProjection{ testAffine(12288), testAffine(1024), testAffine(1024) };
    ps[2].a4.k = 256;
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, 0x3000000, &outs, 128, true));
    ps[2].a4.k = 5120;
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, outs[0], &outs, 128, true));
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, 0x3000000, &.{ outs[0], outs[1], outs[0] }, 128, true));
    ps[2].a4.s = outs[0];
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, 0x3000000, &outs, 128, true));
    ps[2].a4.s = 0x2000000;
    ps[2].a4.b = e.mo.?.fast8.?.scratch.ptr;
    try std.testing.expectEqual(@as(u64, 0), try run(&e, &ps, 0x3000000, &outs, 128, true));
    ps[2].a4.b = 0x2100000;
    try std.testing.expectEqual(@as(u64, 7), try run(&e, &ps, 0x3000000, &outs, 128, true));
    try std.testing.expectEqual(@as(usize, 1), e.mo.?.fast8.?.affine_count);
}
