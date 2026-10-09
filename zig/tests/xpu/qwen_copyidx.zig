//! Copy-drafter index against the reference scan (draftN) on random histories, and the cost of a call at 123K tokens.
const std = @import("std");
const spec = @import("qwen_xpu").spec;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn main() !void {
    const gpa = std.heap.page_allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    var bad: usize = 0;
    var total: usize = 0;
    var hits: usize = 0;
    for ([_]struct { min_n: usize, max_n: usize, vocab: u32 }{ .{ .min_n = 3, .max_n = 6, .vocab = 6 }, .{ .min_n = 6, .max_n = 12, .vocab = 5 }, .{ .min_n = 6, .max_n = 12, .vocab = 40 } }) |cfg| {
        var hist: std.ArrayList(u32) = .empty;
        defer hist.deinit(gpa);
        var idx: spec.GramIndex = .{};
        defer idx.deinit();
        for (0..3000) |_| {
            // repeats of earlier spans so that long matches exist
            if (hist.items.len > 50 and rnd.uintLessThan(u32, 4) == 0) {
                const s = rnd.uintLessThan(usize, hist.items.len - 20);
                for (0..rnd.uintLessThan(usize, 15) + 1) |j| try hist.append(gpa, hist.items[s + j]);
            } else try hist.append(gpa, rnd.uintLessThan(u32, cfg.vocab));
            var a: [16]u32 = undefined;
            var b: [16]u32 = undefined;
            const ra = spec.draftN(hist.items, 3, cfg.max_n, cfg.min_n, &a);
            const rb = idx.draft(hist.items, 3, cfg.max_n, cfg.min_n, &b);
            total += 1;
            if (ra.cnt > 0) hits += 1;
            if (ra.cnt != rb.cnt or ra.n != rb.n or !std.mem.eql(u32, a[0..ra.cnt], b[0..rb.cnt])) bad += 1;
        }
    }
    std.debug.print("copy index vs reference scan: {d} of {d} calls differ ({d} with a match)\n", .{ bad, total, hits });
    // cost at depth: 123K random tokens, no match for the suffix (the reference scans everything)
    var big: std.ArrayList(u32) = .empty;
    defer big.deinit(gpa);
    for (0..122880) |_| try big.append(gpa, rnd.uintLessThan(u32, 150000));
    var idx: spec.GramIndex = .{};
    defer idx.deinit();
    var o: [16]u32 = undefined;
    _ = idx.draft(big.items, 3, 12, 6, &o); // builds the index once
    var t = nowNs();
    for (0..20) |_| {
        try big.append(gpa, rnd.uintLessThan(u32, 150000));
        _ = idx.draft(big.items, 3, 12, 6, &o);
    }
    std.debug.print("index at 123K: {d:.4} ms a call (incremental)\n", .{@as(f64, @floatFromInt(nowNs() - t)) / 20e6});
    t = nowNs();
    for (0..5) |_| _ = spec.draftN(big.items, 3, 12, 6, &o);
    std.debug.print("reference scan at 123K: {d:.2} ms a call\n", .{@as(f64, @floatFromInt(nowNs() - t)) / 5e6});
    if (bad != 0) return error.Mismatch;
}
