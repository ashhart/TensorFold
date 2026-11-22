//! TF_Q27_CAUSAL_SKIP: a prompt chunk wholly past a threadgroup's causal limit writes only its neutral partial.
const std = @import("std");

/// The prompt kernel's threads a group: eight simdgroups, both kernels (attention_prompt*.metal's SG = 8).
pub const prompt_threads = 256;
/// Q27PromptArgs in attention_prompt.metal; causal_skip is 1 under TF_Q27_CAUSAL_SKIP=1.
pub const PromptArgs = extern struct { start: i32, rows: i32, chunks: i32, rp: i32, key_stride: i32, value_stride: i32, scale2: f32, kv_heads: i32, causal_skip: i32 = 0 };

/// One past the last key a packed row of threadgroup `group` sees, as q27_prompt_group_keys computes it; 0 if none.
pub fn groupKeys(start: i32, rows: i32, g: i32, group: i32, tiles: i32) i32 {
    const first = group * tiles * 16;
    const end = @min(first + tiles * 16, rows * g);
    return if (end > first) start + @divFloor(end - 1, g) + 1 else 0;
}

const Partial = struct { m: f32, l: f32, o: f32 };

/// A row's chunk over its visible keys, in the kernel's order; a chunk it cannot see stays at -inf, 0 and +0.
fn partial(scores: []const f32, values: []const f32, kbeg: usize, kend: usize, n: usize) Partial {
    var r = Partial{ .m = -std.math.inf(f32), .l = 0, .o = 0 };
    for (kbeg..kend) |key| {
        if (key >= n) continue;
        const nm = @max(r.m, scores[key]);
        const f = if (r.m == -std.math.inf(f32)) @as(f32, 0) else std.math.pow(f32, 2, r.m - nm);
        const p = std.math.pow(f32, 2, scores[key] - nm);
        r = .{ .m = nm, .l = r.l * f + p, .o = r.o * f + p * values[key] };
    }
    return r;
}

/// q27_attn_prompt_merge's reduction of one row's chunks.
fn merge(parts: []const Partial) u32 {
    var m = -std.math.inf(f32);
    var l: f32 = 0;
    var o: f32 = 0;
    for (parts) |c| {
        if (c.m == -std.math.inf(f32)) continue;
        const nm = @max(m, c.m);
        const f1 = if (m == -std.math.inf(f32)) @as(f32, 0) else std.math.pow(f32, 2, m - nm);
        const f2 = std.math.pow(f32, 2, c.m - nm);
        l = l * f1 + c.l * f2;
        o = o * f1 + c.o * f2;
        m = nm;
    }
    return @bitCast(o / l);
}

test "causal skip: the limit bounds every row of the group and is tight" {
    for ([_]i32{ 2, 4, 6, 8, 16 }) |g| for ([_]i32{ 0, 57, 500, 511, 512, 1000 }) |start| for (1..129) |r| {
        const rows: i32 = @intCast(r);
        var group: i32 = 0;
        while (group * 64 < rows * g) : (group += 1) {
            var top: i32 = 0;
            var row = group * 64;
            while (row < @min((group + 1) * 64, rows * g)) : (row += 1) top = @max(top, start + @divFloor(row, g) + 1);
            try std.testing.expectEqual(top, groupKeys(start, rows, g, group, 4));
        }
        try std.testing.expectEqual(@as(i32, 0), groupKeys(start, rows, g, group, 4));
    };
}

test "causal skip: stale partials of skipped chunks merge to the same bits" {
    var prng = std.Random.DefaultPrng.init(33);
    const rng = prng.random();
    var scores: [1536]f32 = undefined;
    var values: [1536]f32 = undefined;
    for (&scores, &values) |*s, *v| {
        s.* = rng.float(f32) * 16 - 8;
        v.* = rng.float(f32) * 4 - 2;
    }
    var skipped: usize = 0;
    for ([_]i32{ 2, 6, 16 }) |g| for ([_]i32{ 0, 57, 500, 1000 }) |start| for ([_]i32{ 16, 128 }) |rows| {
        const total: usize = @intCast(start + rows);
        const chunks = (total + 511) / 512;
        var row: i32 = 0;
        while (row < rows * g) : (row += 1) {
            const n: usize = @intCast(start + @divFloor(row, g) + 1);
            const limit: usize = @intCast(groupKeys(start, rows, g, @divFloor(row, 64), 4));
            var full: [3]Partial = undefined;
            var cut: [3]Partial = undefined;
            for (0..chunks) |c| {
                const kbeg = c * 512;
                full[c] = partial(&scores, &values, kbeg, @min(kbeg + 512, total), n);
                cut[c] = if (kbeg >= limit) .{ .m = -std.math.inf(f32), .l = 0, .o = std.math.nan(f32) } else full[c];
                skipped += @intFromBool(kbeg >= limit);
            }
            try std.testing.expectEqual(merge(full[0..chunks]), merge(cut[0..chunks]));
        }
    };
    try std.testing.expect(skipped > 0);
}

test "causal skip: each prompt kernel, contiguous and paged, carries the guard and the args' last word" {
    const sources = @import("kernel_sources");
    const guard = "args.causal_skip != 0 && kbeg >= q27_prompt_group_keys(args, int(tg.z), SG / 2)";
    for ([_][]const u8{ sources.qwen27_paged.attention_prompt, sources.qwen27_paged.attention_prompt_sg }) |text| {
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, guard));
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "int kt = skip ? kend : kbeg;"));
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "if (!skip) {"));
    }
    try std.testing.expect(std.mem.indexOf(u8, sources.qwen27_paged.attention_prompt, "float scale2; int kv_heads, causal_skip; };") != null);
    try std.testing.expectEqual(@as(usize, 36), @sizeOf(PromptArgs));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(PromptArgs, "causal_skip"));
}
