//! The unquantized embedding gather: BF16 rows byte-exact as stored, fp32 rows rounded once to the output's kind.

const std = @import("std");
const rig = @import("rig.zig");
const quant = @import("core").quant;
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const Ops = @import("../ops/ops.zig").Ops;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

const vocab = 1031;
const width = 4096;
const ids = [_]i32{ 0, vocab - 1, 7, 7, 512, 1030, 3, 999, 0 };

/// A table whose values span many binades, every bit pattern finite: row r's element c from a mixed hash.
fn table(rng: *rig.Rng, comptime T: type) ![]T {
    const out = try gpa.alloc(T, vocab * width);
    for (out) |*v| {
        const x: f32 = @floatCast(rng.normal() * std.math.pow(f64, 2, @floatFromInt(@as(i32, @intCast(rng.next() % 40)) - 20)));
        v.* = if (T == u16) quant.convert.bf16(x) else x;
    }
    return out;
}

fn gather(t: *Rig, tab: DeviceBuffer, kind: c_int, out_kind: c_int, out_size: usize) ![]u8 {
    var dev_ids = try t.upload(&ids);
    defer dev_ids.free();
    var out = try t.alloc(ids.len * width * out_size);
    defer out.free();
    try out.fill8(0xa5);
    try t.on.tf_embed_dense(@ptrFromInt(at(tab)), kind, @ptrFromInt(at(dev_ids)), ids.len, width, @ptrFromInt(at(out)), out_kind, t.stream.handle);
    try t.stream.synchronize();
    return rig.download(u8, out);
}

test "a bf16 table's rows come out byte-exact as bf16 and widen exactly to fp32" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xE3BD };
    const host = try table(&rng, u16);
    defer gpa.free(host);
    var tab = try t.upload(host);
    defer tab.free();
    const brain = try gather(t, tab, 2, 2, 2);
    defer gpa.free(brain);
    const wide = try gather(t, tab, 2, 0, 4);
    defer gpa.free(wide);
    for (ids, 0..) |id, r| {
        const row = host[@as(usize, @intCast(id)) * width ..][0..width];
        try rig.expect(std.mem.eql(u8, std.mem.sliceAsBytes(row), brain[r * width * 2 ..][0 .. width * 2]), "bf16 row {d} (id {d}) differs", .{ r, id });
        for (row, 0..) |b, c| {
            const got = std.mem.readInt(u32, wide[(r * width + c) * 4 ..][0..4], .little);
            try rig.expect(got == @as(u32, b) << 16, "fp32 row {d} col {d}: {x} for bf16 {x}", .{ r, c, got, b });
        }
    }
}

test "an fp32 table's rows round to nearest even into bf16, as before the table kind" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x5EED };
    const host = try table(&rng, f32);
    defer gpa.free(host);
    var tab = try t.upload(host);
    defer tab.free();
    const brain = try gather(t, tab, 0, 2, 2);
    defer gpa.free(brain);
    for (ids, 0..) |id, r| for (0..width) |c| {
        const want = quant.convert.bf16(host[@as(usize, @intCast(id)) * width + c]);
        const got = std.mem.readInt(u16, brain[(r * width + c) * 2 ..][0..2], .little);
        try rig.expect(got == want, "row {d} col {d}: {x}, round to nearest even gives {x}", .{ r, c, got, want });
    };
}

test "embedRows reads a bf16 projection's table through the gather" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (!t.bf16) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0xB16 };
    const host = try table(&rng, u16);
    defer gpa.free(host);
    var tab = try t.upload(host);
    defer tab.free();
    var dev_ids = try t.upload(&ids);
    defer dev_ids.free();
    var out = try t.alloc(ids.len * width * 2);
    defer out.free();
    const o: Ops = t.ops(&t.on);
    try o.embedRows(.{ .n = vocab, .k = width, .handle = .{ .bf16 = .{ .weight = at(tab) } } }, at(dev_ids), ids.len, .{ .ptr = at(out), .kind = .bf16 });
    try t.stream.synchronize();
    const got = try rig.download(u16, out);
    defer gpa.free(got);
    for (ids, 0..) |id, r| try rig.expect(std.mem.eql(u16, host[@as(usize, @intCast(id)) * width ..][0..width], got[r * width ..][0..width]), "row {d} differs", .{r});
}
