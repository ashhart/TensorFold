//! GPU check: with a YaRN table on the engine, the MTP drafter's ring keys past 262,144 are the YaRN rope of its keys.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const mtp = @import("cuda_mtp.zig");
const geometry = @import("cuda_geometry.zig");

const c = core.contract.spec;
const head_dim = geometry.head_dim;

/// The engine's table for Qwen3.6's 1M config, uploaded for the check and restored after.
fn table() [33]f32 {
    const y = core.rope_yarn.yarn(10000000, 4, 262144);
    var t: [33]f32 = undefined;
    t[0] = y.scale;
    @memcpy(t[1..], &y.freq);
    return t;
}

/// Three rows absorbed at 300,000 under the table: each kv head's ring key against float64, then batched vs serial.
pub fn run(gpa: std.mem.Allocator, d: *mtp.Mtp, tap: u64, ids: []const u32) !bool {
    const e = d.e;
    const t = table();
    var buf = try cuda.DeviceBuffer.fromHost(e.d, std.mem.sliceAsBytes(&t));
    defer buf.free();
    const saved = e.rope_table;
    defer e.rope_table = saved;
    e.rope_table = buf.ptr;
    const ctx = try d.create();
    defer d.drop(ctx);
    const start: u64 = 300000;
    const rows = 3;
    try d.absorb(ctx, tap, start, rows, null, ids[1 .. 1 + rows]);
    try e.stream.synchronize();
    const kw = d.kvWidth();
    const raw = try gpa.alloc(u16, rows * kw);
    defer gpa.free(raw);
    try (cuda.DeviceBuffer{ .d = e.d, .ptr = d.kraw, .len = raw.len * 2 }).download(0, std.mem.sliceAsBytes(raw));
    var w: [head_dim]u16 = undefined;
    try (cuda.DeviceBuffer{ .d = e.d, .ptr = d.w.k_norm, .len = head_dim * 2 }).download(0, std.mem.sliceAsBytes(&w));
    var ok = true;
    var worst: f64 = 0;
    for (0..rows) |r| {
        const pos = start + r;
        var key: [2 * head_dim]u16 = undefined; // a ring slot's keys: kv_heads (2) of head_dim
        try (cuda.DeviceBuffer{ .d = e.d, .ptr = d.ringK(ctx) + (pos % d.capacity) * kw * 2, .len = kw * 2 }).download(0, std.mem.sliceAsBytes(key[0..kw]));
        for (0..kw / head_dim) |h| {
            const row = raw[r * kw + h * head_dim ..][0..head_dim];
            var ss: f64 = 0;
            for (row) |v| ss += c.fromBf16(v) * c.fromBf16(v);
            const inv = 1 / @sqrt(ss / head_dim + e.g.eps);
            var n: [head_dim]f64 = undefined;
            for (&n, row, w) |*x, v, g| x.* = c.fromBf16(c.toBf16(@floatCast(c.fromBf16(v) * inv * c.fromBf16(g))));
            for (0..64) |dim| {
                const a = n[dim % 32];
                const b = n[dim % 32 + 32];
                const angle: f32 = @as(f32, @floatFromInt(pos)) * t[1 + dim % 32];
                const cs = @cos(@as(f64, angle)) * t[0];
                const sn = @sin(@as(f64, angle)) * t[0];
                const want = if (dim < 32) a * cs - b * sn else b * cs + a * sn;
                const mag = (@abs(a) + @abs(b)) * t[0];
                const bound = c.unit(.bf16) * @abs(want) + (c.gamma(4, c.unit(.f32)) + 68 * c.unit(.f32) + 2 * c.unit(.bf16)) * mag + 0x1p-126;
                const got = c.fromBf16(key[h * head_dim + dim]);
                if (!(@abs(got - want) <= bound)) {
                    if (ok) std.debug.print("FAIL mtp YaRN: position {d} kv head {d} dim {d}: {d} vs {d} (bound {e})\n", .{ pos, h, dim, got, want, bound });
                    ok = false;
                }
                worst = @max(worst, @abs(got - want) / bound);
            }
        }
    }
    std.debug.print("{s} mtp YaRN: ring keys at 300,000 to 300,002 against float64 YaRN rope (worst {d:.3} of the bound)\n", .{ if (ok) "PASS" else "FAIL", worst });
    const batched = try @import("cuda_mtp_batch_check.zig").run(gpa, d, tap, ids);
    std.debug.print("{s} mtp YaRN: batched drafting equals serial with the table\n", .{if (batched) "PASS" else "FAIL"});
    return ok and batched;
}
