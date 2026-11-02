//! Raw GGUF gain and decay coefficients retain source bits instead of passing through BF16 or exp/log.
const std = @import("std");
const t = std.testing;
const gg = @import("../../core/gguf.zig");
const image = @import("gguf_image.zig");
const names = @import("gguf_names.zig");
const bf = @import("../../core/bf16_round.zig");
fn entry(a: std.mem.Allocator, d: *const gg.Document, name: []const u8) !names.Entry {
    const config = @import("gguf_config.zig");
    for (try names.plan(a, d, try config.parse(a, try config.build(a, d)))) |e| {
        if (std.mem.eql(u8, e.tensor.name, name)) return e;
    }
    return error.MissingTestTensor;
}
test "GGUF Qwen: unrounded F32 norm gain survives image preparation and distinguishes the narrowed control" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try @import("gguf_fixture.zig").makeTokenized(a);
    var d = try gg.Document.parse(t.allocator, bytes, .{});
    defer d.deinit();
    const config = @import("gguf_config.zig");
    const entries = try names.plan(a, &d, try config.parse(a, try config.build(a, &d)));
    var norm_count: usize = 0;
    for (entries) |e| {
        const role = std.mem.indexOf(u8, e.tensor.name, "norm.weight") != null and
            std.mem.indexOf(u8, e.tensor.name, "ssm_norm.weight") == null;
        if (!role) continue;
        norm_count += 1;
        try t.expectEqual(.f32, e.transform);
        const at = @intFromPtr(e.tensor.bytes.ptr) - @intFromPtr(bytes.ptr);
        std.mem.writeInt(u32, bytes[at..][0..4], 0x3fd68000, .little);
        try image.validateSmall(e, null);
    }
    try t.expectEqual(@as(usize, 11), norm_count);
    const e = try entry(a, &d, "output_norm.weight");
    const gains = try image.rawF32(a, e);
    try t.expectEqual(@as(u32, 0x3fd68000), @as(u32, @bitCast(gains[0])));
    const narrowed: f32 = @bitCast(@as(u32, bf.bits(gains[0])) << 16);
    try t.expectEqual(@as(u32, 0x3fd60000), @as(u32, @bitCast(narrowed)));
    // Independent binary64 RMS: this input puts the two gains on opposite sides of a BF16 rounding boundary.
    const inv = 1.0 / @sqrt(@as(f64, 515.0 / 512.0) + @as(f64, 1e-6));
    try t.expect(bf.bits(@floatCast(@as(f64, gains[0]) * inv)) != bf.bits(@floatCast(@as(f64, narrowed) * inv)));
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    try image.write(t.allocator, t.io, &d, root, "raw-f32-regression", .{});
    var ck = try @import("../../core/checkpoint.zig").Checkpoint.openModel(t.allocator, t.io, root);
    defer ck.close();
    for (entries) |candidate| {
        if (candidate.transform != .f32) continue;
        const gain = try ck.get(candidate.name);
        try t.expectEqual(.f32, gain.dtype);
        try t.expectEqual(@as(u32, 0x3fd68000), std.mem.readInt(u32, gain.bytes[0..4], .little));
    }
}
test "GGUF Qwen: stored ssm_a uses exact negative F32 including converter one-ulp witnesses" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try @import("gguf_fixture.zig").makeTokenized(a);
    var d = try gg.Document.parse(t.allocator, bytes, .{});
    defer d.deinit();
    var e = try entry(a, &d, "blk.0.ssm_a");
    e.heads = null;
    var raw: [4]u8 = undefined;
    e.tensor.bytes = &raw;
    // A converter may choose either neighbor of a correctly rounded exp; source bits win over this host's exp.
    var inverse_losses: usize = 0;
    for ([_]f32{ -2, -0.3125, -0.046875 }) |a_log| {
        const original: f32 = @floatCast(-@exp(@as(f64, a_log)));
        const center: u32 = @bitCast(original);
        for ([_]u32{ center - 1, center + 1 }) |bits| {
            std.mem.writeInt(u32, &raw, bits, .little);
            const weight = (try image.rawF32(a, e))[0];
            try t.expectEqual(bits, @as(u32, @bitCast(weight)));
            try t.expectEqual(bf.bits(a_log), bf.bits(@log(-weight)));
            const roundtrip = -@exp(@as(f32, @bitCast(@as(u32, bf.bits(@log(-weight))) << 16)));
            inverse_losses += @intFromBool(bits != @as(u32, @bitCast(roundtrip)));
            const smooth: f64 = @as(f32, @bitCast(@as(u32, bf.bits(@log(@as(f32, 2)))) << 16));
            const correct = @exp(@as(f64, weight) * smooth);
            const ignored_mode = @exp(-@exp(@as(f64, weight)) * smooth);
            try t.expect(@abs(correct - ignored_mode) > 1e-3);
        }
    }
    try t.expect(inverse_losses >= 3);
    for ([_]f32{ 0, -0.0, 1, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) }) |x| {
        std.mem.writeInt(u32, &raw, @bitCast(x), .little);
        try t.expectError(if (std.math.isFinite(x)) error.BadGgufSsmA else error.NonfiniteGgufTensor, image.rawF32(a, e));
    }
}

test "GGUF Qwen: direct decay mode requires F32, prices four bytes per head, and leaves MLX flags unchanged" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try @import("gguf_fixture.zig").make(a);
    var d = try gg.Document.parse(t.allocator, bytes, .{});
    defer d.deinit();
    const config = @import("gguf_config.zig");
    var c = try config.parse(a, try config.build(a, &d));
    c.k_heads = 16;
    c.v_heads = 48;
    c.dk = 128;
    c.dv = 128;
    const gdn = @import("gdn_contract.zig");
    const s = try gdn.Shape.init(c);
    var k: gdn.WeightKinds = .{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 };
    try t.expectEqual(@as(u32, 0), k.flags());
    k.stored_ssm_a = true;
    try t.expectError(error.BadGdnDecayStorage, s.params(3, 2, k));
    k.a_log = .f32;
    const p = try s.params(3, 2, k);
    try t.expectEqual(@as(u32, 18), p.weight_flags);
    try t.expectEqual(@as(usize, 48 * 4), (try gdn.Bytes.init(s, p, k)).a_log);
    k.stored_ssm_a = false;
    try t.expectEqual(@as(u32, 2), k.flags());
}
