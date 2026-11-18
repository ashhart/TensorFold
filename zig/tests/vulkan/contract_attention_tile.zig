// The tiled prefill gate: non-finite V planted past a row's bound must be skipped and byte-match decode.
const std = @import("std");
const vk = @import("vulkan");
const ca = @import("contract_attention.zig");
const fx = @import("stage3_fixture.zig");
const main = @import("contract_attention_main.zig");
const contract = @import("contract");
const cx = contract.spec;

// Tiled prefill with non-finite V planted past a row's bound: the tile must skip future keys and byte-match decode.
pub fn sdpaTileGate(a: std.mem.Allocator, clock: main.Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, attn: *vk.Attention, case: ca.TileCase, seed: u64, timeout: u64, comptime wide: bool) !bool {
    const El = if (wide) u32 else u16;
    const arm: []const u8 = if (wide) "f32" else "bf16";
    const row_elems: u64 = fx.heads * fx.head_dim;
    const q_len: usize = @intCast(case.rows * row_elems);
    const cache_elems: usize = @intCast(fx.kv_heads * case.capacity * fx.head_dim);
    const q_bits = try a.alloc(u16, q_len);
    defer a.free(q_bits);
    const k_bits = try a.alloc(u16, cache_elems);
    defer a.free(k_bits);
    const v_bits = try a.alloc(u16, cache_elems);
    defer a.free(v_bits);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (q_bits) |*x| x.* = fx.sampleDyadic(rng, -5, 3);
    for (k_bits) |*x| x.* = fx.sampleDyadic(rng, -5, 3);
    for (v_bits) |*x| x.* = fx.sampleDyadic(rng, -5, 3);
    const poison = [_]u16{ 0x7fc0, 0x7f80, 0xff80 }; // NaN, +inf, -inf
    const cap: usize = @intCast(case.capacity);
    const hd: usize = @intCast(fx.head_dim);
    if (case.zero_qk) {
        @memset(q_bits, 0);
        @memset(k_bits, 0);
    }
    for (0..@as(usize, @intCast(fx.kv_heads))) |kh| {
        for (@as(usize, @intCast(case.poison_from))..cap) |j| {
            const at = (kh * cap + j) * hd;
            @memset(v_bits[at..][0..hd], poison[(j - @as(usize, @intCast(case.poison_from))) % poison.len]);
        }
        if (case.zero_qk) @memset(v_bits[(kh * cap) * hd ..][0..hd], 0x3f80); // V0 = 1.0
    }
    var keys = try main.upload16(dev, cmds, staging, k_bits, timeout);
    defer keys.deinit();
    var values = try main.upload16(dev, cmds, staging, v_bits, timeout);
    defer values.deinit();
    // Wide q rides the f32 path: bf16 widens exactly, so both arms see identical queries over the same bf16 cache.
    var q = if (wide) wq: {
        const words = try a.alloc(u32, q_len);
        defer a.free(words);
        for (q_bits, 0..) |bits, i| words[i] = @bitCast(@as(f32, @floatCast(cx.fromBf16(bits))));
        break :wq try main.upload32(dev, cmds, staging, words, timeout);
    } else try main.upload16(dev, cmds, staging, q_bits, timeout);
    defer q.deinit();
    const q_view = try vk.tensor.View.init(&q, if (wide) .f32 else .bf16, 0, @intCast(q_len));
    const keys_view = try vk.tensor.View.init(&keys, .bf16, 0, @intCast(k_bits.len));
    const values_view = try vk.tensor.View.init(&values, .bf16, 0, @intCast(v_bits.len));
    const tiled = try a.alloc(El, q_len);
    defer a.free(tiled);
    {
        const shell_poison: El = if (wide) 0x7fc00000 else fx.poison16;
        const shell = try a.alloc(El, q_len + fx.tail);
        defer a.free(shell);
        @memset(shell, shell_poison);
        var out = try main.uploadWords(El, dev, cmds, staging, shell, timeout);
        defer out.deinit();
        const y_view = try vk.tensor.View.init(&out, if (wide) .f32 else .bf16, 0, @intCast(q_len));
        // rows 2..64: attend() dispatches sdpa_scores_tile + sdpa_softmax_rows + sdpa_pv_tile, the PR's tiled path.
        try vk.orExit(attn.attend(cmds, y_view, q_view, keys_view, values_view, null, case.prefix, case.rows, case.capacity, timeout));
        try main.fetch(cmds, staging, out, std.mem.sliceAsBytes(tiled), timeout);
    }
    const old = try a.alloc(El, q_len);
    defer a.free(old);
    {
        const shell_poison: El = if (wide) 0x7fc00000 else fx.poison16;
        const shell = try a.alloc(El, @intCast(row_elems + fx.tail));
        defer a.free(shell);
        @memset(shell, shell_poison);
        var one = try main.uploadWords(El, dev, cmds, staging, shell, timeout);
        defer one.deinit();
        const y_view = try vk.tensor.View.init(&one, if (wide) .f32 else .bf16, 0, @intCast(row_elems));
        for (0..@as(usize, @intCast(case.rows))) |r| {
            const q_row = try q_view.slice(@intCast(r * row_elems), @intCast(row_elems));
            // The old path: the row decoded alone at q_len 1 by sdpa_rows_bf16(+f32q), which the PR replaced.
            try vk.orExit(vk.upload(cmds, one, 0, staging, std.mem.sliceAsBytes(shell), timeout));
            try vk.orExit(attn.attend(cmds, y_view, q_row, keys_view, values_view, null, case.prefix + @as(u64, r), 1, case.capacity, timeout));
            try main.fetch(cmds, staging, one, std.mem.sliceAsBytes(old[@intCast(r * row_elems)..][0..@intCast(row_elems)]), timeout);
        }
    }
    if (std.mem.indexOfDiff(El, tiled, old)) |at| {
        std.debug.print("FAIL sdpa-tile case=\"{s}\" {s}: tiled output differs from per-row decode at element {d}\n", .{ case.name, arm, at });
        return false;
    }
    const finite = ca.finiteRows(case);
    var plant_rows: usize = 0;
    for (0..@as(usize, @intCast(case.rows))) |r| {
        const row = tiled[@intCast(r * row_elems)..][0..@intCast(row_elems)];
        var nonfinite: usize = 0;
        for (row) |bits| {
            const v: f32 = if (wide) @bitCast(bits) else @floatCast(cx.fromBf16(bits));
            if (!std.math.isFinite(v)) nonfinite += 1;
        }
        if (r < finite) {
            if (nonfinite != 0) {
                std.debug.print("FAIL sdpa-tile case=\"{s}\" {s}: row {d} went non-finite past a bound it never sees\n", .{ case.name, arm, r });
                return false;
            }
        } else {
            if (nonfinite == 0) {
                std.debug.print("FAIL sdpa-tile case=\"{s}\" {s}: row {d} stayed finite though its window holds the plants\n", .{ case.name, arm, r });
                return false;
            }
            plant_rows += 1;
        }
    }
    if (case.zero_qk) {
        const one: El = if (wide) 0x3f800000 else 0x3f80;
        for (tiled[0..@intCast(row_elems)]) |bits| {
            if (bits != one) {
                std.debug.print("FAIL sdpa-tile case=\"{s}\" {s}: row 0 is not the exact 1.0 its single visible key forces\n", .{ case.name, arm });
                return false;
            }
        }
    }
    std.debug.print("MILESTONE sdpa-tile elapsed_ms={d} case=\"{s}\" {s} rows={d} prefix={d} poison_from={d} capacity={d}: " ++
        "tiled byte-identical to per-row decode, {d} finite rows finite, {d} plant rows non-finite{s}\n", .{
        clock.ms(), case.name,  arm,                                             case.rows, case.prefix, case.poison_from, case.capacity,
        finite,     plant_rows, if (case.zero_qk) ", row 0 exactly 1.0" else "",
    });
    return true;
}
