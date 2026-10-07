//! sample.cu's draws against the host references of the Metal engine's rule (lanes.gpu_rule, lanes.gpu_full).

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const check = @import("check.zig");
const Gpu = check.Gpu;

const vocab = 131072;
const subset = 32768;
const rows = 4;

/// sample.cu's Rule: seed, 1 / T, top_p, the near window, ln(min_p), top_k.
const Rule = extern struct { seed: u64, inv_t: f32, top_p: f32, near: f32, min_log: f32, top_k: u32, pad: u32 = 0 };

fn ruleOf(s: lanes.Sampling) Rule {
    return .{ .seed = s.seed, .inv_t = @floatCast(1.0 / @max(s.temperature, 1e-6)), .top_p = @floatCast(s.top_p), .near = 20.0, .min_log = @floatCast(s.minLog()), .top_k = s.top_k };
}

fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

fn widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

const Kind = enum { spread, peaked, ties, flat };

/// Columns by (value desc, column asc): the kernels' order at any temperature.
fn order(gpa: std.mem.Allocator, row: []const f32) ![]u32 {
    const cols = try gpa.alloc(u32, row.len);
    for (cols, 0..) |*c, i| c.* = @intCast(i);
    std.mem.sort(u32, cols, row, struct {
        fn before(r: []const f32, a: u32, b: u32) bool {
            return r[a] > r[b] or (r[a] == r[b] and a < b);
        }
    }.before);
    return cols;
}

/// The pick's share of the top_k mass at T (the whole row when top_k is off or past it), in f64.
fn share(row: []const f32, sorted: []const u32, s: lanes.Sampling, col: usize) f64 {
    const inv_t = 1.0 / @max(s.temperature, 1e-6);
    const m = @as(f64, row[sorted[0]]) * inv_t;
    const k = if (s.top_k == 0 or s.top_k > row.len) row.len else s.top_k;
    var z: f64 = 0;
    for (sorted[0..k]) |c| z += @exp(@as(f64, row[c]) * inv_t - m);
    return @exp(@as(f64, row[col]) * inv_t - m) / z;
}

/// A row of `n` logits by a fixed generator, rounded to bf16 as the engine's logits are.
fn fill(kind: Kind, salt: u64, out: []u16, wide: []f32) void {
    var x: u64 = 0x9E37_79B9_7F4A_7C15 ^ (salt *% 0xD1B5_4A32_D192_ED03);
    for (out, wide, 0..) |*h, *w, i| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        const r: f32 = @as(f32, @floatFromInt(x >> 40)) / 16777216.0;
        const v: f32 = switch (kind) {
            .spread => -3.0 * r,
            .peaked => if (i % 2711 == 7) 2.0 + 6.0 * r else -12.0 * r,
            .ties => -3.0 * @floor(r * 16.0) / 16.0,
            .flat => 0.0,
        };
        h.* = bf16(v);
        w.* = widen(h.*);
    }
}

/// Every rule on every row kind: the GPU's token against the host reference's, at row r's position.
pub fn draws(gpu: Gpu) !void {
    const gpa = gpu.gpa;
    var m = try cuda.Module.load(gpu.d, cuda.kernels.sample);
    defer m.unload();
    const full = try m.function("tf_draw");
    const ids_fn = try m.function("tf_draw_ids");
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const host = try gpa.alloc(u16, rows * vocab);
    defer gpa.free(host);
    const wide = try gpa.alloc(f32, rows * vocab);
    defer gpa.free(wide);
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, rows * vocab * 2);
    defer logits.free();
    var rule = try cuda.DeviceBuffer.alloc(gpu.d, @sizeOf(Rule));
    defer rule.free();
    var meta = try cuda.DeviceBuffer.alloc(gpu.d, 16);
    defer meta.free();
    var tok = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer tok.free();
    var prob = try cuda.DeviceBuffer.alloc(gpu.d, rows * 4);
    defer prob.free();
    // a draft head's subset vocabulary: every fourth id, so the noise keys differ from the columns
    const ids = try gpa.alloc(u32, subset);
    defer gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(4 * i + 1);
    const wide_ids = try gpa.alloc(i64, subset); // the engine's draft id table is int64
    defer gpa.free(wide_ids);
    for (wide_ids, ids) |*w, id| w.* = id;
    var ids_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(wide_ids));
    defer ids_dev.free();
    const temps = [_]f64{ 0.6, 1.0, 1.5 };
    const ks = [_]u32{ 1, 20, 40, 100, 1024, 0, 1500 };
    const ps = [_]f64{ 0.9, 0.95, 1.0 };
    const mins = [_]f64{ 0.0, 0.05 };
    var total: usize = 0;
    var differ: usize = 0;
    var by_full: [2]usize = .{ 0, 0 };
    var worst: f64 = 0; // the largest relative error of a pick's share
    for ([_]bool{ false, true }) |mapped| {
        const n: usize = if (mapped) subset else vocab;
        for (std.enums.values(Kind), 0..) |kind, ki| {
            for (0..rows) |r| fill(kind, ki * 131 + r + @as(u64, @intFromBool(mapped)) * 977, host[r * n ..][0..n], wide[r * n ..][0..n]);
            // uploads go on the draw's own stream: a legacy-stream copy can land after a non-blocking stream's launch
            try logits.uploadAsync(0, std.mem.sliceAsBytes(host[0 .. rows * n]), stream.handle);
            var sorted: [rows][]u32 = undefined;
            for (&sorted, 0..) |*o, r| o.* = try order(gpa, wide[r * n ..][0..n]);
            defer for (sorted) |o| gpa.free(o);
            for (temps) |t| for (ks) |k| for (ps) |p| for (mins) |mp| {
                if (mapped and (k == 1500 or mp > 0)) continue; // the draft head's draws: fewer rules suffice
                const s: lanes.Sampling = .{ .seed = 0x5EED_0000 + ki * 7 + total, .temperature = t, .top_k = k, .top_p = p, .min_p = mp };
                const base: i32 = @intCast(100 + total % 997);
                const offset: i32 = if (mapped) 2 else 0;
                try rule.uploadAsync(0, std.mem.asBytes(&ruleOf(s)), stream.handle);
                try meta.uploadAsync(0, std.mem.asBytes(&[4]i32{ base, 0, 0, 0 }), stream.handle);
                var args: cuda.Args = .{};
                args.add(logits.ptr);
                args.add(@as(u32, @intCast(n)));
                args.add(rule.ptr);
                args.add(meta.ptr);
                args.add(offset);
                args.add(tok.ptr);
                if (mapped) args.add(ids_dev.ptr);
                args.add(prob.ptr);
                try cuda.launch.launch(if (mapped) ids_fn else full, .{ .grid = .{ .x = rows }, .block = .{ .x = 1024 } }, stream, &args);
                try stream.synchronize();
                var got: [rows]u32 = undefined;
                try tok.download(0, std.mem.sliceAsBytes(&got));
                var shares: [rows]f32 = undefined;
                try prob.download(0, std.mem.sliceAsBytes(&shares));
                for (0..rows) |r| {
                    const position: u32 = @intCast(base + @as(i32, @intCast(r)) + 1 + offset);
                    const want = try lanes.gpu_full.draw(gpa, wide[r * n ..][0..n], s, position, if (mapped) ids else null);
                    total += 1;
                    const col: usize = if (mapped) (got[r] - 1) / 4 else got[r];
                    const ref = share(wide[r * n ..][0..n], sorted[r], s, col);
                    worst = @max(worst, @abs(@as(f64, shares[r]) - ref) / ref);
                    if (got[r] != want) {
                        differ += 1;
                        by_full[@intFromBool(lanes.gpu_full.fullVocabulary(s))] += 1;
                        std.debug.print("DIFFER {s}{s} row {d} T {d} top_k {d} top_p {d} min_p {d} position {d}: gpu {d} host {d}\n", .{ @tagName(kind), if (mapped) " ids" else "", r, t, k, p, mp, position, got[r], want });
                    }
                }
            };
        }
    }
    std.debug.print("RESULT sample: {d} draws, {d} differ from the host references ({d} tf_gpu_sample, {d} tf_sample_full); shares within {e} of f64\n", .{ total, differ, by_full[0], by_full[1], worst });
    try check.expect(differ == 0, "sample: {d} of {d} draws differ", .{ differ, total });
    try check.expect(worst < 1e-3, "sample: a pick's share is {e} off its f64 value", .{worst});
    check.pass("sample: {d} draws equal the Metal rule's host references", .{total});
}
