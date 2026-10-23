//! CPU census, independent fixtures and actual production dispatch recording for GLM window contracts.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const cases = @import("tests/contract/glm_rows_cases.zig");
const Fixture = @import("tests/contract/glm_rows_fixture.zig").Fixture;
const metal = @import("metal");
const tf = @import("tensorfold");
const wide = tf.glm.wide;
const Affine = tf.glm.affine.Affine;
const Probe = @import("tests/contract/metal_glm_rows.zig").Probe;
const sources = @import("kernel_sources");
const a = std.testing.allocator;
const Collector = struct {
    a: std.mem.Allocator,
    list: std.ArrayList(cases.Entry) = .empty,
    pub fn add(s: *@This(), v: cases.Entry, _: []const u8) !void {
        try s.list.append(s.a, v);
    }
};
test "GLM window contract: all twelve row symbols and both gather paths reach the actual collector" {
    var found: Collector = .{ .a = a };
    defer found.list.deinit(a);
    try cases.visit(&found, null);
    try std.testing.expectEqual(@as(usize, 78), found.list.items.len);
    var controls: usize = 0;
    for (found.list.items) |v| controls += @intFromBool(v.mode == .control);
    try std.testing.expectEqual(@as(usize, 24), controls);
    for ([_]cases.Kind{ .narrow, .wide, .down, .glu }) |kind| for (cases.bits) |bits| {
        var modes: [4]usize = @splat(0);
        for (found.list.items) |v| if (v.kind == kind and v.bits == bits) {
            modes[@backingInt(v.mode)] += 1;
        };
        try std.testing.expectEqualSlices(usize, if (kind == .glu) &.{ 1, 1, 1, 1 } else &.{ 1, 1, 1, 0 }, &modes);
    };
    const main = @embedFile("tests/contract/metal_main.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, main, "try glm_windows.collect(o.only);"));
    found.list.clearRetainingCapacity();
    try cases.visit(&found, "narrow q3");
    try std.testing.expectEqual(@as(usize, 3), found.list.items.len);
}
test "GLM window contract: real shapes and every window width are admitted, replacements name a frozen predecessor" {
    for (cases.entries) |v| {
        const served = cases.served(v.kind);
        const s = cases.spec(v, "test", &served);
        for (served) |shape| try std.testing.expect(Fixture.takes(v, shape));
        for (1..18) |m| {
            var found = false;
            for (served) |shape| if (shape.rows == m) {
                found = true;
            };
            try std.testing.expect(found);
        }
        try std.testing.expect((s.replaces != null) == (v.mode == .replacement or v.mode == .control));
        const shapes = try c.shapes(a, s);
        defer a.free(shapes);
        var saw_tail = false;
        for (shapes) |shape| if (shape.k == 544 or shape.k == 576) {
            saw_tail = true;
        };
        try std.testing.expect(saw_tail);
    }
}
test "GLM window contract: logical maps retain head strides, empty experts, permutation, repeated picks and padding" {
    for ([_]cases.Kind{ .narrow, .wide, .down, .glu }) |kind| {
        const v: cases.Entry = .{ .kind = kind, .bits = 3, .mode = .oracle };
        const f = try Fixture.init(a, v, .{ .rows = 7, .n = 8, .k = 576 }, .random, 1130);
        defer f.deinit();
        @memset(f.f.scales, c.toBf16(0.5));
        @memset(f.f.biases, c.toBf16(-1));
        for (0..f.f.shape.rows) |r| @memset(f.f.x[r * 576 ..][0..576], c.toBf16(@floatFromInt(r + 1)));
        for (0..f.f.shape.n) |j| @memset(f.f.codes[j * 576 ..][0..576], @intCast(j % 3 + 1));
        const want = try a.alloc(f64, f.count());
        defer a.free(want);
        const mag = try a.alloc(f64, f.count());
        defer a.free(mag);
        f.reference(want, mag);
        const used = try a.alloc(bool, f.outputElements());
        defer a.free(used);
        @memset(used, false);
        for (want, 0..) |w, i| {
            const out_row = i / 8;
            const sorted = if (kind == .down) 6 - out_row else out_row;
            const is_rows = kind == .narrow or kind == .wide;
            const input_row = if (is_rows) out_row else if (kind == .glu) (6 - sorted) / 2 else sorted;
            const column = if (is_rows) (out_row % 2) * 8 + i % 8 else (if (sorted < 3) @as(usize, 1) else 3) * 8 + i % 8;
            const at = f.coordinates(i);
            try std.testing.expectEqual(input_row, at.row);
            try std.testing.expectEqual(column, at.col);
            const expected_index = if (is_rows) (out_row / 2) * 32 + (out_row % 2) * 12 + i % 8 else out_row * 16 + i % 8;
            try std.testing.expectEqual(expected_index, f.outputIndex(i));
            const expected = 576 * @as(f64, @floatFromInt(input_row + 1)) * (0.5 * @as(f64, @floatFromInt(column % 3 + 1)) - 1);
            try std.testing.expectEqual(expected, w);
            try std.testing.expect(!used[f.outputIndex(i)]);
            used[f.outputIndex(i)] = true;
        }
        const x = try a.alloc(u16, f.inputElements());
        defer a.free(x);
        f.packX(x);
        for (0..f.f.shape.rows) |r| {
            const offset = if (kind == .narrow or kind == .wide) (r / 2) * 1168 + (r % 2) * 576 else r * 576;
            try std.testing.expectEqualSlices(u16, f.f.x[r * 576 ..][0..576], x[offset..][0..576]);
        }
        if (f.heads == 2) try std.testing.expectEqual(@as(u16, 0x7fc1), x[2 * 576]);
        if (kind == .down or kind == .glu) {
            try std.testing.expectEqualSlices(i32, &.{ 0, 0, 3, 3 }, &f.offsets);
            try std.testing.expectEqualSlices(i32, &.{ 6, 5, 4, 3, 2, 1, 0 }, f.order);
        }
    }
}
test "GLM window contract: every dispatched source control has a nontrivial oracle witness" {
    const text = sources.core_row_projection ++ sources.glm_rows ++ sources.glm_gather;
    for (cases.entries) |v| {
        if (v.mode != .control) continue;
        const changed = try cases.source(a, text, v);
        defer a.free(changed);
        try std.testing.expect(!std.mem.eql(u8, changed, text));
        const f = try Fixture.init(a, v, .{ .rows = 7, .n = 8, .k = 576 }, .random, 0x1130 + @as(u64, v.bits));
        defer f.deinit();
        try std.testing.expect(f.controlDiffers());
    }
    for ([_]cases.Mode{ .oracle, .up_oracle }) |mode| {
        const text_ = try cases.source(a, text, .{ .kind = .glu, .bits = 3, .mode = mode });
        defer a.free(text_);
        try std.testing.expect(std.mem.indexOf(u8, text_, cases.glu_store) == null);
        try std.testing.expect(std.mem.indexOf(u8, text_, if (mode == .oracle) "bfloat(acc[b][j])" else "bfloat(up[b][j])") != null);
    }
}
test "GLM window contract: each raw projection stress fixture rejects reduced sums in every order" {
    for (cases.bits) |bits| for ([_]cases.Kind{ .narrow, .down, .glu }) |kind| for ([_]cases.Mode{ .oracle, .up_oracle }) |mode| {
        if (mode == .up_oracle and kind != .glu) continue;
        for ([_]usize{ 4096, 16384 }) |k| for ([_]u64{ 0x6b63, 0xa11ce }) |seed| {
            const v: cases.Entry = .{ .kind = kind, .bits = bits, .mode = mode };
            const f = try Fixture.init(a, v, .{ .rows = 4, .n = 4, .k = k }, .stress, seed);
            defer f.deinit();
            var want: [16]f64 = undefined;
            var mag: [16]f64 = undefined;
            var got: [16]f64 = undefined;
            f.reference(&want, &mag);
            const served = cases.served(kind);
            const spec = cases.spec(v, "test", &served);
            for ([_]contract.gates.Order{ .sequential, .pairwise, .strided }) |order| {
                try f.simulate(.f32, order, &got);
                try std.testing.expectEqualSlices(f64, &want, &got);
                for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                    try f.simulate(acc, order, &got);
                    var most: f64 = 0;
                    for (got, want) |value, expected| most = @max(most, if (std.math.isFinite(value)) @abs(value - expected) / c.boundExact(spec, expected) else std.math.inf(f64));
                    if (most < 100) std.debug.print("weak stress q{d} {s} {s} k{d} {s}/{s}: {d}\n", .{ bits, @tagName(kind), @tagName(mode), k, @tagName(acc), @tagName(order), most });
                    try std.testing.expect(most >= 100);
                }
            }
        };
    };
}
test "GLM window contract: actual dense and per-head launchers select narrow/wide, strides and four-column grids" {
    for (cases.bits, 0..) |bits, i| for ([_]u32{ 1, 2, 3, 4, 5, 8, 16 }) |rows| {
        var p: wide.Pipes = undefined;
        p.rows[i] = 11;
        p.rows_wide[i] = 22;
        const q: Affine = .{ .w = .{ .buf = .{ .id = 2 }, .off = 24 }, .s = .{ .buf = .{ .id = 3 }, .off = 32 }, .b = .{ .buf = .{ .id = 4 }, .off = 48 }, .n = 24, .k = 576, .bits = bits, .group = 32 };
        var record: metal.Record = .{};
        wide.perHead(&p, .{ .r = &record }, q, 2, .{ .in = .{ .buf = .{ .id = 1 }, .off = 64 }, .out = .{ .buf = .{ .id = 5 }, .off = 80 }, .ldx = 1200, .ldo = 40, .rows = rows }, 576, 16);
        try std.testing.expectEqual(@as(usize, 1), record.n);
        const pass = record.passes[0];
        const dims = std.mem.bytesToValue(wide.RowDims, pass.values[4][0..@sizeOf(wide.RowDims)]);
        try std.testing.expectEqual(@as(u32, if (rows < 4) 11 else 22), pass.pipe);
        try std.testing.expectEqual(@as(usize, 3), pass.grid.width);
        try std.testing.expectEqual(@as(usize, 2), pass.grid.height);
        try std.testing.expectEqual(@as(usize, 64), pass.threads.width);
        try std.testing.expectEqual(@as(i32, 1200), dims.ldx);
        try std.testing.expectEqual(@as(i32, 16), dims.hso);
        try std.testing.expectEqual(@as(usize, 80), pass.buffers[5].off);
    };
}
test "GLM window contract: actual gather binds independent up width, empty-expert tiles and output stride" {
    const q: Affine = .{ .w = .{ .buf = .{ .id = 2 } }, .s = .{ .buf = .{ .id = 3 } }, .b = .{ .buf = .{ .id = 4 } }, .n = 12, .k = 576, .bits = 3, .group = 32 };
    var up = q;
    up.bits = 6;
    up.group = 64;
    var record: metal.Record = .{};
    wide.gather.encode(metal.ComputeEncoder{ .r = &record }, 99, q, up, .{ .x = .{ .buf = .{ .id = 1 } }, .offsets = .{ .buf = .{ .id = 6 } }, .order = .{ .buf = .{ .id = 7 } }, .out = .{ .buf = .{ .id = 5 } } }, 9, 4, 2, 20, 10);
    const pass = record.passes[0];
    const d = std.mem.bytesToValue(wide.gather.Dims, pass.values[4][0..@sizeOf(wide.gather.Dims)]);
    try std.testing.expectEqual(@as(i32, 3), d.bits);
    try std.testing.expectEqual(@as(i32, 6), d.up_bits);
    try std.testing.expectEqual(@as(i32, 64), d.up_group);
    try std.testing.expectEqual(@as(i32, 20), d.ldo);
    try std.testing.expectEqual(@as(usize, 2), pass.grid.width);
    try std.testing.expectEqual(@as(usize, 6), pass.grid.height);
    try std.testing.expectEqual(@as(usize, 6), pass.buffers[6].id);
    try std.testing.expectEqual(@as(usize, 7), pass.buffers[7].id);
}

test "GLM window contract: the real adapter encodes the new grid and original kernel independently" {
    const f = try Fixture.init(a, .{ .kind = .wide, .bits = 3, .mode = .replacement }, .{ .rows = 3, .n = 12, .k = 576 }, .random, 1130);
    defer f.deinit();
    var p: Probe = .{
        .a = a,
        .rig = .{ .a = a, .device = .{}, .queue = .{} },
        .v = f.v,
        .name = "test",
        .served = cases.served(.wide),
        .pipe = 11,
        .base_pipe = 33,
        .f = f,
        .x = .{ .buf = .{ .id = 1 }, .off = 64 },
        .w = .{ .w = .{ .buf = .{ .id = 2 }, .off = 80 }, .s = .{ .buf = .{ .id = 3 }, .off = 96 }, .b = .{ .buf = .{ .id = 4 }, .off = 112 } },
        .y = .{ .buf = .{ .id = 5 }, .off = 128 },
        .base = .{ .buf = .{ .id = 6 }, .off = 144 },
    };
    var record: metal.Record = .{};
    p.encode(metal.ComputeEncoder{ .r = &record }, p.pipe, p.y, false);
    try p.encodePredecessor(metal.ComputeEncoder{ .r = &record });
    try std.testing.expectEqual(@as(u32, 11), record.passes[0].pipe);
    try std.testing.expectEqual(@as(u32, 33), record.passes[1].pipe);
    try std.testing.expectEqual(@as(usize, 3), record.passes[0].grid.width);
    try std.testing.expectEqual(@as(usize, 2), record.passes[1].grid.width);
    try std.testing.expectEqual(@as(usize, 144), record.passes[1].buffers[5].off);
    p.base_pipe = null;
    try std.testing.expectError(error.MissingPredecessor, p.encodePredecessor(metal.ComputeEncoder{ .r = &record }));
}
test "GLM window contract: nonfinite raw taps retain mixed finite and nonfinite outputs" {
    for (cases.entries) |v| {
        if (v.mode != .oracle and v.mode != .up_oracle) continue;
        const f = try Fixture.init(a, v, .{ .rows = 5, .n = 8, .k = 576 }, .nonfinite, 0x6b63);
        defer f.deinit();
        const want = try a.alloc(f64, f.count());
        defer a.free(want);
        const mag = try a.alloc(f64, f.count());
        defer a.free(mag);
        f.reference(want, mag);
        var nonfinite: usize = 0;
        for (want) |value| nonfinite += @intFromBool(!std.math.isFinite(value));
        try std.testing.expect(nonfinite > 0 and nonfinite < want.len);
    }
}

test "GLM window contract: frozen kernels are the actual pre-change bytes, not the replacement oracle" {
    const files = .{
        .{ @embedFile("tests/contract/baselines/glm_row_projection.metal.txt"), "4631914a81bb4dfeaec7d09f8078ba7453b37935272061406ea59ecd2322547c" },
        .{ @embedFile("tests/contract/baselines/glm_rows.metal.txt"), "92f6e89d1162b07eff3868105900b2f3e72951db49e3d465a74eb99ad1aaaf1e" },
        .{ @embedFile("tests/contract/baselines/glm_gather.metal.txt"), "56b05e1cf8eef8a2f9d43a9de1eb3cac9dd5a2281b3d7f6d5a967e164c89488d" },
    };
    inline for (files) |file| {
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(file[0], &hash, .{});
        const hex = std.fmt.bytesToHex(hash, .lower);
        try std.testing.expectEqualStrings(file[1], &hex);
    }
    const original = @import("tests/contract/metal_glm_rows.zig").predecessor_source;
    try std.testing.expect(std.mem.startsWith(u8, original, files[0][0]));
    try std.testing.expect(std.mem.indexOf(u8, original, "decltype(glm_rows<B, true>)") == null);
    try std.testing.expect(std.mem.indexOf(u8, sources.glm_rows, "decltype(glm_rows<B, true>)") != null);
}
test "GLM window contract: clipped final SwiGLU retains its stress reference without pretending to prove sums" {
    for (cases.bits) |bits| for ([_]u64{ 0x6b63, 0xa11ce }) |seed| {
        const v: cases.Entry = .{ .kind = .glu, .bits = bits, .mode = .replacement };
        const f = try Fixture.init(a, v, .{ .rows = 4, .n = 4, .k = 4096 }, .stress, seed);
        defer f.deinit();
        var want: [16]f64 = undefined;
        var mag: [16]f64 = undefined;
        var got: [16]f64 = undefined;
        f.reference(&want, &mag);
        const served = cases.served(.glu);
        const spec = cases.spec(v, "test", &served);
        try std.testing.expectEqual(@as(f64, 0), spec.sums.f16);
        try std.testing.expectEqual(@as(f64, 0), spec.sums.bf16);
        try std.testing.expect(spec.sums.why.len != 0);
        for ([_]contract.gates.Order{ .sequential, .pairwise, .strided }) |order| {
            try f.simulate(.f32, order, &got);
            for (got, want, mag) |value, w, m| try std.testing.expect(@abs(value - w) <= c.bound(spec, f.shape, w, m));
        }
    };
}

test "GLM window contract: gate and up observers read different matrices at independently selected widths" {
    for (cases.bits) |bits| {
        const f = try Fixture.init(a, .{ .kind = .glu, .bits = bits, .mode = .up_oracle }, .{ .rows = 7, .n = 8, .k = 576 }, .random, 1130);
        defer f.deinit();
        @memset(f.f.codes, 1);
        @memset(f.up.?.codes, 3);
        @memset(f.f.scales, c.toBf16(1));
        @memset(f.up.?.scales, c.toBf16(1));
        @memset(f.f.biases, 0);
        @memset(f.up.?.biases, 0);
        @memset(f.f.x, c.toBf16(1));
        @memset(f.up.?.x, c.toBf16(1));
        try std.testing.expect(f.f.bits != f.up.?.bits and f.f.group != f.up.?.group);
        var want: [56]f64 = undefined;
        var mag: [56]f64 = undefined;
        f.reference(&want, &mag);
        for (want) |w| try std.testing.expectEqual(@as(f64, 3 * 576), w);
        var gate = f;
        gate.v.mode = .oracle;
        gate.reference(&want, &mag);
        for (want) |w| try std.testing.expectEqual(@as(f64, 576), w);
    }
}

fn metalScalarBf(v: f32) f32 {
    const normal = if (@abs(v) < 0x1p-126) std.math.copysign(@as(f32, 0), v) else v;
    return c.fromBf16(c.toBf16(normal));
}
fn metalScalarGlu(gate: f64, value: f64) f64 {
    const x = metalScalarBf(@min(metalScalarBf(@floatCast(gate)), 10));
    const up = metalScalarBf(std.math.clamp(metalScalarBf(@floatCast(value)), -10, 10));
    const e = metalScalarBf(@exp(@abs(x)));
    const y = metalScalarBf(1 / metalScalarBf(1 + e));
    return metalScalarBf(metalScalarBf(x * (if (x < 0) y else metalScalarBf(1 - y))) * up);
}
fn fixedGluDots(f: Fixture) void {
    for ([_]contract.affine.Affine{ f.f, f.up.? }, [_]u8{ 11, 2 }) |q, delta| {
        const half: u8 = @as(u8, 1) << @intCast(q.bits - 1);
        for (0..q.shape.n) |j| for (0..q.shape.k / 2) |i| {
            q.codes[j * q.shape.k + i] = half + delta;
            q.codes[j * q.shape.k + q.shape.k / 2 + i] = half - delta;
        };
    }
}
test "GLM window contract: exact FP32 dots cannot falsely fail final GLU at a subnormal sigmoid" {
    for ([_]u8{ 5, 6 }) |bits| {
        const shape: c.Shape = .{ .rows = 4, .n = 4, .k = 4096 };
        const v: cases.Entry = .{ .kind = .glu, .bits = bits, .mode = .replacement };
        const f = try Fixture.init(a, v, shape, .stress, 0x6b63);
        defer f.deinit();
        fixedGluDots(f);
        var want: [16]f64 = undefined;
        var mag: [16]f64 = undefined;
        f.reference(&want, &mag);
        const spec = cases.spec(v, "normal nonlinear stress", &.{shape});
        for (want, mag, 0..) |w, m, i| {
            const at = f.coordinates(i);
            const gate = f.f.one(at.row, at.col)[0];
            const up = f.up.?.one(at.row, at.col)[0];
            const got = metalScalarGlu(gate, up);
            try std.testing.expect(@abs(got - w) <= c.bound(spec, shape, w, m));
        }
        for ([_]cases.Mode{ .oracle, .up_oracle }) |mode| {
            const raw = try Fixture.init(a, .{ .kind = .glu, .bits = bits, .mode = mode }, shape, .stress, 0x6b63);
            defer raw.deinit();
            fixedGluDots(raw);
            const at = raw.coordinates(0);
            try std.testing.expectEqual(@as(f64, -88), raw.f.one(at.row, at.col)[0]);
            try std.testing.expectEqual(@as(f64, -16), raw.up.?.one(at.row, at.col)[0]);
            for ([_]contract.affine.Affine{ raw.f, raw.up.? }) |q| {
                for (q.scales) |s| try std.testing.expectEqual(c.toBf16(1), s);
                for (q.biases) |b| try std.testing.expectEqual(c.toBf16(-@as(f32, @floatFromInt(@as(u32, 1) << @intCast(q.bits - 1)))), b);
            }
        }
    }
}

test "GLM window contract: final nonlinear stress stays normal across widths, groups and seeds" {
    for (cases.bits) |bits| for ([_]usize{ 3, 4 }) |rows| for ([_]usize{ 576, 4096 }) |k| for ([_]u64{ 0x6b63, 0xa11ce }) |seed| {
        const shape: c.Shape = .{ .rows = rows, .n = 4, .k = k };
        const v: cases.Entry = .{ .kind = .glu, .bits = bits, .mode = .replacement };
        const f = try Fixture.init(a, v, shape, .stress, seed);
        defer f.deinit();
        var want: [16]f64 = undefined;
        var mag: [16]f64 = undefined;
        f.reference(want[0..f.count()], mag[0..f.count()]);
        const spec = cases.spec(v, "normal nonlinear stress", &.{shape});
        for (0..f.count()) |i| {
            const at = f.coordinates(i);
            const gate = f.f.one(at.row, at.col)[0];
            const up = f.up.?.one(at.row, at.col)[0];
            try std.testing.expect(gate < 0 and gate >= -16 and up < 0 and up >= -16);
            try std.testing.expect(want[i] >= 0x1p-126);
            const got = metalScalarGlu(gate, up);
            try std.testing.expect(@abs(got - want[i]) <= c.bound(spec, shape, want[i], mag[i]));
        }
    };
}
