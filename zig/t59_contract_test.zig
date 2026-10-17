//! The same family catalogue is required by the normal Metal runner, not a stand-alone comparison list.
const std = @import("std");
const family = @import("tensorfold").gemma4.contracts;
const contract = @import("contract");
test "T59 normal runner requires every Gemma geometry, fused kind and tap setting" {
    try family.requireComplete(family.entries);
    try std.testing.expectError(error.IncompleteGemmaContract, family.requireComplete(&.{}));
    for (family.entries) |v| {
        const launch = contract.limits.MetalLaunch{ .name = v.name, .threads = if (v.kind == 0) 128 else @intCast(v.layout().threads), .tg_memory = @intCast(v.threadgroupBytes()), .offline = false };
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(std.testing.allocator);
        for (contract.limits.apple) |chip| try std.testing.expectEqual(@as(usize, 0), try contract.limits.metal(launch, chip, &out, std.testing.allocator));
    }
}

test "T59 literal 31B and 12B launch settings and private staged math" {
    const ns = [12]usize{ 5376, 16384, 18432, 16384, 43008, 5376, 3840, 8192, 8704, 8192, 30720, 3840 };
    const sks = [12]usize{ 0, 2, 2, 2, 8, 4, 0, 4, 4, 4, 8, 4 };
    const bytes = [12]usize{ 9216, 19456, 19456, 19456, 25600, 7680, 9216, 20480, 20480, 20480, 22528, 7424 };
    for (family.entries, 0..) |v, i| {
        try std.testing.expectEqual(ns[i], v.n);
        try std.testing.expectEqual(if (v.kind == 3) (if (i < 6) @as(usize, 21504) else 15360) else (if (i < 6) @as(usize, 5376) else 3840), v.k);
        try std.testing.expectEqual(bytes[i], v.threadgroupBytes());
        if (v.kind != 0) {
            const l = v.layout();
            try std.testing.expectEqual(sks[i], l.projection.sk);
            try std.testing.expectEqual(if (v.kind == 3) @as(usize, 128) else 256, l.threads);
            try std.testing.expectEqual(if (v.kind == 3) @as(usize, 0) else 1, l.x_rows);
            try std.testing.expectEqual(v.kind == 2, l.epilogue);
            const source = try family.fused.source(std.testing.allocator, l);
            defer std.testing.allocator.free(source);
            const define = try std.fmt.allocPrint(std.testing.allocator, "#define TF_FUSE {d}\n", .{v.kind});
            defer std.testing.allocator.free(define);
            try std.testing.expect(std.mem.indexOf(u8, source, define) != null);
        }
    }
    var duplicate: [12]family.Variant = family.entries[0..12].*;
    duplicate[1] = duplicate[2];
    try std.testing.expectError(error.DuplicateGemmaContract, family.requireComplete(&duplicate));
    _ = @import("tests/contract/gemma4_math.zig");
}

test "T59 fused prologues preserve exact stress inputs and bound random norm stages" {
    const math = @import("tests/contract/gemma4_math.zig");
    const c = contract.spec;
    const a = std.testing.allocator;
    for ([_]usize{ 5376, 3840 }) |k| for ([_]u8{ 1, 2 }) |kind| {
        var aux: [5][]u16 = undefined;
        for (&aux) |*v| v.* = try a.alloc(u16, k);
        defer for (aux) |v| a.free(v);
        const out = try a.alloc(u16, k);
        defer a.free(out);
        const hidden = try a.alloc(u16, k);
        defer a.free(hidden);
        const bounds = try a.alloc(math.Interval, k);
        defer a.free(bounds);
        const hb = try a.alloc(math.Interval, k);
        defer a.free(hb);
        for (0..k) |i| {
            aux[0][i] = c.toBf16(1);
            aux[1][i] = 0;
            aux[2][i] = 0;
            aux[3][i] = if (kind == 1) c.toBf16(1) else c.toBf16(@floatFromInt(i % 251 + 1));
            aux[4][i] = c.toBf16(@floatFromInt(i % 251 + 1));
        }
        math.prologue(kind, .{ aux[0], aux[1], aux[2], aux[3], aux[4] }, out, hidden, bounds, hb, 1e-6);
        try std.testing.expectEqualSlices(u16, aux[4], out);
        for (hidden, out, bounds, hb) |h, x, range, hrange| {
            try std.testing.expectEqual(c.toBf16(1), h);
            try std.testing.expect(range.contains(c.fromBf16(x)));
            try std.testing.expect(hrange.contains(1));
        }
    };
}

test "T59 production numerical hooks reject corrupt main, hidden, tap and inner guards" {
    const mtl = @import("metal");
    const Probe = @import("tests/contract/metal_gemma4.zig").Probe;
    const a = std.testing.allocator;
    const c = contract.spec;
    for (family.entries) |v| {
        mtl.reset();
        var p = try Probe.init(a, .{}, .{}, v, false);
        defer p.deinit();
        try std.testing.expect(try p.prepare(.{ .rows = 1, .n = 128, .k = v.k }, .random, 59));
        const k = p.kernel();
        const n = p.count();
        const data = try a.alloc(f64, n * 2);
        defer a.free(data);
        try p.reference(data[0..n], data[n..]);
        const main = p.f.?.outputs() / (if (v.kind == 2) @as(usize, 2) else 1);
        for (data[0..main], 0..) |x, i| std.mem.writeInt(u16, p.rig.bytes(p.y, main * 2)[2 * i ..][0..2], c.toBf16(@floatCast(x)), .little);
        if (v.kind == 1 or v.kind == 2) {
            @memcpy(p.rig.bytes(p.h, p.hidden.len * 2), std.mem.sliceAsBytes(p.hidden));
            if (v.tap) @memcpy(p.rig.bytes(p.tap, p.hidden.len * 2), std.mem.sliceAsBytes(p.hidden));
        }
        try std.testing.expectEqual(@as(?[]const u8, null), try p.check(.shapes));
        const first = p.rig.bytes(p.y, 2);
        const saved = first[0..2].*;
        std.mem.writeInt(u16, first[0..2], c.toBf16(std.math.nan(f32)), .little);
        try std.testing.expect(try p.check(.shapes) != null);
        @memcpy(first, &saved);
        if (v.kind == 1 or v.kind == 2) {
            const h = p.rig.bytes(p.h, 2);
            const old = h[0..2].*;
            std.mem.writeInt(u16, h[0..2], c.toBf16(1000), .little);
            try std.testing.expect(try p.check(.shapes) != null);
            @memcpy(h, &old);
        }
        if (v.tap) {
            std.mem.writeInt(u16, p.rig.bytes(p.tap, 2)[0..2], c.toBf16(1000), .little);
            try std.testing.expect(try p.check(.shapes) != null);
        }
        if (v.kind == 2) {
            p.rig.bytes(p.y, p.f.?.outputs() * 2)[main * 2] = 0;
            try std.testing.expect(!try k.vt.intact(k.ctx));
        }
    }
}

test "T59 registered stress simulators prove their stated accumulator margins" {
    const mtl = @import("metal");
    const Probe = @import("tests/contract/metal_gemma4.zig").Probe;
    const a = std.testing.allocator;
    const c = contract.spec;
    for (family.entries) |v| {
        mtl.reset();
        var p = try Probe.init(a, .{}, .{}, v, false);
        defer p.deinit();
        const shape = c.Shape{ .rows = 1, .n = 128, .k = v.k };
        try std.testing.expect(try p.prepare(shape, .stress, 59));
        const n = p.count();
        const data = try a.alloc(f64, 3 * n);
        defer a.free(data);
        const want = data[0..n];
        const mag = data[n .. 2 * n];
        const got = data[2 * n ..];
        try p.reference(want, mag);
        const spec = p.kernel().spec;
        var margins = [2]f64{ std.math.inf(f64), std.math.inf(f64) };
        for ([_]contract.gates.Order{ .sequential, .pairwise, .strided }) |order| {
            try p.simulate(.f32, order, got);
            for (got, want) |x, y| try std.testing.expect(@abs(x - y) <= c.boundExact(spec, y));
            for ([_]c.Accum{ .f16, .bf16 }, &margins) |acc, *least| {
                try p.simulate(acc, order, got);
                var most: f64 = 0;
                for (got, want) |x, y| most = @max(most, if (std.math.isFinite(x)) @abs(x - y) / c.boundExact(spec, y) else std.math.inf(f64));
                least.* = @min(least.*, most);
            }
        }
        try std.testing.expect(margins[0] >= spec.sums.f16);
        try std.testing.expect(margins[1] >= spec.sums.bf16);
    }
}
