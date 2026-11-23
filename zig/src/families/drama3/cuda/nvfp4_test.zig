//! Compact NVFP4 experts on the host, no CUDA context: staged as stored in held-slot order, decoded against FP64.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const cfg = @import("../config.zig");
const wts = @import("weights.zig");
const quant = @import("../metal/quant_format.zig");
const ref = @import("../metal/quant_reference.zig");
const Shards = @import("../shards.zig").Shards;

const testing = std.testing;
const io = testing.io;
const E = cuda.experts.Nvfp4Experts;
const held_ids = [_]u8{ 3, 9, 250 };
const projections = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
const shape = [3][2]usize{ .{ cfg.expert_width, cfg.hidden }, .{ cfg.expert_width, cfg.hidden }, .{ cfg.hidden, cfg.expert_width } };

/// Layer 0's held experts in NVFP4 as tf-drama-quant writes them; layer 1's expert 0 gate in FP8.
fn writeCheckpoint(arena: std.mem.Allocator, dir: []const u8) !void {
    var prng = std.Random.DefaultPrng.init(27);
    const rnd = prng.random();
    var tensors: std.ArrayList(core.shard_edit.New) = .empty;
    for (held_ids) |e| for (projections, shape) |p, s| {
        const name = try std.fmt.allocPrint(arena, "model.language_model.layers.0.mlp.experts.{d}.{s}.weight", .{ e, p });
        const codes = try arena.alloc(u8, s[0] * s[1] / 2);
        rnd.bytes(codes);
        const scales = try arena.alloc(u8, s[0] * s[1] / 16);
        for (scales) |*b| b.* = 0x30 + rnd.uintLessThan(u8, 0x30);
        const global = try arena.dupe(u8, std.mem.asBytes(&(0.5 + rnd.float(f32))));
        try tensors.append(arena, .{ .name = name, .dtype = .u8, .shape = try arena.dupe(usize, &.{ s[0], s[1] / 2 }), .bytes = codes });
        try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}_scale", .{name}), .dtype = .f8_e4m3, .shape = try arena.dupe(usize, &.{ s[0], s[1] / 16 }), .bytes = scales });
        try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}_scale_2", .{name}), .dtype = .f32, .shape = &.{}, .bytes = global });
    };
    const fp8 = try arena.alloc(u8, cfg.expert_width * cfg.hidden);
    rnd.bytes(fp8);
    try tensors.append(arena, .{ .name = "model.language_model.layers.1.mlp.experts.0.gate_proj.weight", .dtype = .f8_e4m3, .shape = &.{ cfg.expert_width, cfg.hidden }, .bytes = fp8 });
    try core.shard_edit.create(arena, try std.fs.path.join(arena, &.{ dir, "model.safetensors" }), &.{.{ "format", "pt" }}, tensors.items);
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(arena, "{\"metadata\": {}, \"weight_map\": {");
    for (tensors.items, 0..) |t, i| try index.print(arena, "{s}\"{s}\": \"model.safetensors\"", .{ if (i == 0) "" else ", ", t.name });
    try index.appendSlice(arena, "}}");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(arena, &.{ dir, "model.safetensors.index.json" }), .data = index.items });
}

/// Code and e4m3 scale of (row, input) in matrix `part` of `parts` packed blocks, as experts.zig documents the layout.
fn unpacked(blocks: []const u32, parts: usize, part: usize, k: usize, row: usize, input: usize) struct { u8, u8 } {
    const rr = row % E.cols;
    const ii = input % 32;
    const h = ii / 16;
    const block = blocks[(((row / E.cols) * (k / 32) + input / 32) * parts + part) * E.words ..][0..E.words];
    const word = block[((rr % 8) * 4 + (ii % 16) / 4) * 4 + rr / 8];
    const scale = std.mem.sliceAsBytes(block[128..])[((((rr % 8) / 2) * 2 + h) * 4 + rr / 8) * 2 + rr % 2];
    return .{ @intCast((word >> E.slot(h, ii % 4)) & 0xF), scale };
}

/// nvfp4_experts.cu in f32: each 16-group's dot from zero, fmaf by its e4m3 scale into the row sum, then the global.
fn servedDot(codes: []const u8, scales: []const u8, x: []const f32, global: f32) f32 {
    var sum: f32 = 0;
    for (scales, 0..) |scale, b| {
        var dot: f32 = 0;
        for (codes[16 * b ..][0..16], x[16 * b ..][0..16]) |code, v| dot = @mulAdd(f32, @floatCast(ref.fp4(code)), v, dot);
        sum = @mulAdd(f32, dot, @floatCast(ref.fp8(scale)), sum);
    }
    return sum * global;
}

/// A staged row through servedDot.
fn servedRow(blocks: []const u32, parts: usize, part: usize, k: usize, row: usize, x: []const f32, global: f32) f32 {
    var codes: [cfg.hidden]u8 = undefined;
    var scales: [cfg.hidden / 16]u8 = undefined;
    for (0..k) |i| codes[i], scales[i / 16] = unpacked(blocks, parts, part, k, row, i);
    return servedDot(codes[0..k], scales[0 .. k / 16], x[0..k], global);
}

/// Round to nearest even bf16 by the integer trick the kernels' __float2bfloat16_rn performs.
fn bf(x: f32) f32 {
    const b: u32 = @bitCast(x);
    return @bitCast((b +% 0x7fff +% ((b >> 16) & 1)) & 0xffff0000);
}

/// nvfp4_experts.cu's SwiGLU epilogue, stored as bf16: bf16(bf16(silu(bf16(g))) * bf16(u)).
fn servedSwiglu(g: f32, u: f32) f32 {
    const gv = bf(g);
    return bf(bf(gv / (1 + @exp(-gv))) * bf(u));
}

/// The CUDA rounding contract in FP64, as drama3's FP8 path (d3_swiglu) and bf16 torch round: silu rounds to bf16.
fn contractSwiglu(gate: f64, up: f64) f64 {
    const g = ref.roundBf16(gate);
    return ref.roundBf16(ref.roundBf16(g / (1 + @exp(-g))) * ref.roundBf16(up));
}

fn bits(v: f64) u16 {
    return @truncate(@as(u32, @bitCast(@as(f32, @floatCast(v)))) >> 16);
}

/// One bf16 step at |v|: the most an f32 sum's last bit can move a bf16-rounded result.
fn ulp(v: f64) f64 {
    return if (v == 0) 0x1p-133 else std.math.ldexp(@as(f64, 1), std.math.ilogb(@abs(v)) - 7);
}

test "SwiGLU rounds silu to bf16 before the up product: the served epilogue and the FP64 contract at a tie" {
    try testing.expectEqual(@as(f32, -0.00335693359375), servedSwiglu(-4, 0.046875));
    try testing.expectEqual(@as(f64, -0.00335693359375), contractSwiglu(-4, 0.046875));
    // unrounded silu (Metal's reference) lands one bf16 step away: this case pins the rounding point
    try testing.expectEqual(@as(f64, -0.0033721923828125), ref.swiglu(-4, 0.046875));
    try testing.expectEqual(contractSwiglu(1.0078125, -3), servedSwiglu(1.0078125, -3));
    try testing.expectEqual(contractSwiglu(-0.5, 0.0009765625), servedSwiglu(-0.5, 0.0009765625));
}

/// servedDot and the FP64 oracle on one crafted row of 32 inputs: codes, one e4m3 scale a group, bf16 inputs.
fn crafted(codes: [32]u8, scales: [2]u8, x: [32]f32) struct { f32, f64 } {
    var packed_codes: [16]u8 = undefined;
    for (&packed_codes, 0..) |*p, i| p.* = codes[2 * i] | codes[2 * i + 1] << 4;
    var xb: [32]u16 = undefined;
    for (&xb, x) |*b, v| b.* = bits(v);
    return .{ servedDot(&codes, &scales, &x, 1), ref.projection(&packed_codes, &scales, 1, 0, 32, &xb) };
}

test "served NVFP4 rows scale each 16-group's dot, not each term: overflow and cancellation the FP64 oracle lacks" {
    const six = 0x7; // fp4 6
    const one = 0x2; // fp4 1
    const neg_one = 0xA;
    var codes: [32]u8 = @splat(0);
    var x: [32]f32 = @splat(0);
    // 16 x (6 * 2^125) leaves f32 inside the group's dot, before e4m3 1/512 (0x01) would bring it back
    @memset(codes[0..16], six);
    @memset(x[0..16], 0x1p125);
    var got, var want = crafted(codes, .{ 0x01, 0 }, x);
    try testing.expect(std.math.isInf(got));
    try testing.expectEqual(@as(f64, 96 * 0x1p125 / 512), want);
    // 2^24 + 1 - 2^24 in a group: its f32 dot drops the 1 before e4m3 3 (0x44); per-term scaling gives 4, FP64 3
    codes = @splat(0);
    x = @splat(0);
    codes[0..3].* = .{ one, one, neg_one };
    x[0..3].* = .{ 0x1p24, 1, 0x1p24 };
    got, want = crafted(codes, .{ 0x44, 0 }, x);
    try testing.expectEqual(@as(f32, 0), got);
    try testing.expectEqual(@as(f64, 3), want);
}

test "compact NVFP4 experts stage as stored in held-slot order and decode gate-up, SwiGLU, down and combine to FP64" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", arena);
    try writeCheckpoint(arena, dir);
    var sh = try Shards.open(testing.allocator, io, dir);
    defer sh.close(io);

    var held: quant.Held = .{};
    held.counts[0] = held_ids.len;
    @memcpy(held.ids[0][0..held_ids.len], &held_ids);
    try testing.expect(try wts.nvfp4Layer(&sh, 0, &held));
    var stage = try wts.stageNvfp4(testing.allocator, &sh, 0, held.ids[0][0..held.counts[0]], 2);
    defer stage.deinit(testing.allocator);

    var prng = std.Random.DefaultPrng.init(5);
    var x: [cfg.hidden]u16 = undefined;
    for (&x) |*v| v.* = 0x3c00 + prng.random().uintLessThan(u16, 0x400) + (@as(u16, prng.random().int(u1)) << 15);
    var xf: [cfg.hidden]f32 = undefined;
    for (&xf, x) |*f, b| f.* = @floatCast(ref.fromBf16(b));
    const slots = wts.compactSlots(&held, 0).?;
    const per_up = stage.up.len / held_ids.len;
    const per_down = stage.down.len / held_ids.len;
    const route = [held_ids.len]f32{ 0.5, 0.3125, 0.1875 };
    var want_moe: [cfg.hidden]f64 = @splat(0);
    var got_moe: [cfg.hidden]f32 = @splat(0);
    for (slots, 0..) |slot, id| {
        // an original id the router can pick reaches the slot holding its own weights; the rest are masked
        const k_id = std.mem.indexOfScalar(u8, &held_ids, @intCast(id)) orelse {
            try testing.expectEqual(@as(i32, -1), slot);
            continue;
        };
        try testing.expectEqual(@as(i32, @intCast(k_id)), slot);
        const s: usize = @intCast(slot);
        var stored: [3]struct { w: []const u8, sc: []const u8, global: f32 } = undefined;
        for (projections, &stored, 0..) |p, *t, j| {
            var nb: [3][128]u8 = undefined;
            const base = "model.language_model.layers.0.mlp.experts.{d}.{s}.weight";
            const g = try sh.get(try std.fmt.bufPrint(&nb[2], base ++ "_scale_2", .{ id, p }));
            t.* = .{ .w = (try sh.get(try std.fmt.bufPrint(&nb[0], base, .{ id, p }))).bytes, .sc = (try sh.get(try std.fmt.bufPrint(&nb[1], base ++ "_scale", .{ id, p }))).bytes, .global = @bitCast(std.mem.readInt(u32, g.bytes[0..4], .little)) };
            try testing.expectEqual(t.global, if (j < 2) stage.up_scale[2 * s + j] else stage.down_scale[s]);
        }
        const up = stage.up[s * per_up ..][0..per_up];
        const down = stage.down[s * per_down ..][0..per_down];
        var want_act: [cfg.expert_width]u16 = undefined;
        var got_act: [cfg.expert_width]f32 = undefined;
        for (0..cfg.expert_width) |row| {
            const want = contractSwiglu(ref.projection(stored[0].w, stored[0].sc, stored[0].global, row, cfg.hidden, &x), ref.projection(stored[1].w, stored[1].sc, stored[1].global, row, cfg.hidden, &x));
            got_act[row] = servedSwiglu(servedRow(up, 2, 0, cfg.hidden, row, &xf, stored[0].global), servedRow(up, 2, 1, cfg.hidden, row, &xf, stored[1].global));
            // past gate -88 the served expf(-g) overflows and silu is 0 where FP64 keeps ~1e-37
            try testing.expect(@abs(got_act[row] - want) <= ulp(want) + 0x1p-100);
            want_act[row] = bits(want);
        }
        for (0..cfg.hidden) |row| {
            want_moe[row] += route[k_id] * ref.projection(stored[2].w, stored[2].sc, stored[2].global, row, cfg.expert_width, &want_act);
            got_moe[row] = @mulAdd(f32, route[k_id], servedRow(down, 1, 0, cfg.expert_width, row, &got_act, stored[2].global), got_moe[row]);
        }
    }
    // d3_moe_combine rounds the weighted sum of the fp32 down rows once; an activation's last-bit flip stays below 1%
    for (want_moe, got_moe) |want, got| try testing.expect(@abs(bf(got) - ref.roundBf16(want)) <= ulp(want) + @abs(want) / 128);
}

test "a compact map over FP8 experts is refused; a full FP8 layer keeps the FP8 path" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", arena);
    try writeCheckpoint(arena, dir);
    var sh = try Shards.open(testing.allocator, io, dir);
    defer sh.close(io);

    var held: quant.Held = .{};
    try testing.expect(!try wts.nvfp4Layer(&sh, 1, &held));
    held.counts[1] = cfg.top_k;
    try testing.expectError(error.CompactFp8Experts, wts.nvfp4Layer(&sh, 1, &held));
}

test "an unpruned NVFP4 checkpoint keeps the original router in every layer, so it cannot pass as compact" {
    const full = try quant.heldMetadata(null);
    for (0..cfg.layers) |li| try testing.expectEqual(@as(?[cfg.experts]i32, null), wts.compactSlots(&full, li));
    var pruned = full;
    pruned.counts[7] = cfg.experts - 1;
    try testing.expectEqual(@as(i32, -1), wts.compactSlots(&pruned, 7).?[cfg.experts - 1]);
    try testing.expectEqual(@as(?[cfg.experts]i32, null), wts.compactSlots(&pruned, 6));
}
