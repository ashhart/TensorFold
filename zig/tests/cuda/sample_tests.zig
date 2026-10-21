//! sample.cu's draws against the host references of the Metal engine's rule (lanes.gpu_rule, lanes.gpu_full).

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const check = @import("check.zig");
const Gpu = check.Gpu;

const vocab = 131072;
const subset = 32768;
const rows = 4;

const Rule = cuda.sample.Rule;

fn ruleOf(s: lanes.Sampling) Rule {
    return Rule.from(lanes.gpu_rule.rule(s));
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
    try @import("sample_proposal_tests.zig").run(gpu);
}

/// topkRows' two stages against tf_logprob_rows: the same k ids and logits, word for word, on every row kind and width.
pub fn topk(gpu: Gpu) !void {
    const gpa = gpu.gpa;
    var m = try cuda.Module.load(gpu.d, cuda.kernels.sample);
    defer m.unload();
    const d = try cuda.sample.Draws.resolve(m);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const n_rows = 7;
    const lp_words = 3 + 2 * 20;
    const widths = [_]usize{ 248320, 131072, 4097, 4096, 100, 10 };
    const ks = [_]u8{ 1, 16, 20 };
    const most = widths[0];
    const host = try gpa.alloc(u16, n_rows * most);
    defer gpa.free(host);
    const wide = try gpa.alloc(f32, n_rows * most);
    defer gpa.free(wide);
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, n_rows * most * 2);
    defer logits.free();
    var picks = try cuda.DeviceBuffer.alloc(gpu.d, n_rows * 4);
    defer picks.free();
    try picks.fill8(0, null);
    var parts = try cuda.DeviceBuffer.alloc(gpu.d, cuda.sample.Draws.topkPartsBytes(most, 20, n_rows));
    defer parts.free();
    var want_dev = try cuda.DeviceBuffer.alloc(gpu.d, n_rows * lp_words * 4);
    defer want_dev.free();
    var got_dev = try cuda.DeviceBuffer.alloc(gpu.d, n_rows * lp_words * 4);
    defer got_dev.free();
    var cases: usize = 0;
    var differ: usize = 0;
    for (widths) |n| for (std.enums.values(Kind), 0..) |kind, ki| for ([_]bool{ false, true }) |odd| {
        for (0..n_rows) |r| {
            const row = host[r * n ..][0..n];
            fill(kind, ki * 37 + r + n, row, wide[r * n ..][0..n]);
            // -inf and NaN entries: tf_logprob_rows never takes a NaN, and takes -inf only past every finite entry
            if (odd) for (row, 0..) |*h, i| {
                if (i % 5 == 1) h.* = 0xFF80 else if (i % 13 == 3) h.* = 0x7FC0;
            };
        }
        try logits.uploadAsync(0, std.mem.sliceAsBytes(host[0 .. n_rows * n]), stream.handle);
        for (ks) |k| {
            try d.logprobRows(stream, logits.ptr, n, picks.ptr, k, false, want_dev.ptr, n_rows);
            try d.topkRows(stream, logits.ptr, n, k, parts.ptr, got_dev.ptr, n_rows);
            try stream.synchronize();
            var want: [n_rows * lp_words]u32 = undefined;
            var got: [n_rows * lp_words]u32 = undefined;
            try want_dev.download(0, std.mem.sliceAsBytes(&want));
            try got_dev.download(0, std.mem.sliceAsBytes(&got));
            for (0..n_rows) |r| {
                const w = want[r * lp_words ..][0..lp_words];
                const g = got[r * lp_words ..][0..lp_words];
                cases += 1;
                if (!std.mem.eql(u32, w[2..][0..k], g[2..][0..k]) or !std.mem.eql(u32, w[22..][0..k], g[22..][0..k])) {
                    differ += 1;
                    std.debug.print("DIFFER {s}{s} width {d} k {d} row {d}: first ids {d} / {d}\n", .{ @tagName(kind), if (odd) " inf+nan" else "", n, k, r, w[2], g[2] });
                }
            }
        }
    };
    std.debug.print("RESULT topk: {d} rows, {d} differ from tf_logprob_rows\n", .{ cases, differ });
    try check.expect(differ == 0, "topk: {d} of {d} rows differ", .{ differ, cases });
    check.pass("topk: {d} rows equal tf_logprob_rows' k best", .{cases});
}

/// qwen35.cu's two-stage greedy argmax against the host rule: the highest logit, ties to the lower id, no NaN or -inf.
pub fn q35Argmax(gpu: Gpu) !void {
    const gpa = gpu.gpa;
    var m = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
    defer m.unload();
    const slices_fn = try m.function("tf_q35_argmax_slices");
    const merge_fn = try m.function("tf_q35_argmax_merge");
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const n_rows = 12;
    const widths = [_]usize{ 248320, 151936, 4097, 4096, 100 };
    const host = try gpa.alloc(u16, n_rows * widths[0]);
    defer gpa.free(host);
    const wide = try gpa.alloc(f32, n_rows * widths[0]);
    defer gpa.free(wide);
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, host.len * 2);
    defer logits.free();
    var parts = try cuda.DeviceBuffer.alloc(gpu.d, n_rows * ((widths[0] + 4095) / 4096) * 8);
    defer parts.free();
    var toks = try cuda.DeviceBuffer.alloc(gpu.d, n_rows * 4);
    defer toks.free();
    var cases: usize = 0;
    var differ: usize = 0;
    for (widths) |n| for (std.enums.values(Kind), 0..) |kind, ki| for (0..3) |variant| {
        for (0..n_rows) |r| {
            const row = host[r * n ..][0..n];
            fill(kind, ki * 53 + r + n + variant, row, wide[r * n ..][0..n]);
            for (row, 0..) |*h, i| switch (variant) {
                1 => if (i % 7 == 2) {
                    h.* = 0x7FC0; // NaN
                } else if (i % 11 == 5) {
                    h.* = 0xFF80; // -inf
                },
                2 => if (r == 3) {
                    h.* = 0xFF80; // a row of -inf: no token, 0x7fffffff
                } else if (i == n - 1) {
                    h.* = 0x7F80; // +inf at the last id
                },
                else => {},
            };
        }
        try logits.uploadAsync(0, std.mem.sliceAsBytes(host[0 .. n_rows * n]), stream.handle);
        const slices: usize = (n + 4095) / 4096;
        var a: cuda.Args = .{};
        a.add(logits.ptr);
        a.add(@as(c_int, @intCast(n)));
        a.add(parts.ptr);
        a.add(parts.ptr + n_rows * slices * 4);
        try cuda.launch.launch(slices_fn, .{ .grid = .{ .x = @intCast(slices), .y = n_rows }, .block = .{ .x = 256 } }, stream, &a);
        var b: cuda.Args = .{};
        b.add(parts.ptr);
        b.add(parts.ptr + n_rows * slices * 4);
        b.add(@as(c_int, @intCast(slices)));
        b.add(toks.ptr);
        try cuda.launch.launch(merge_fn, .{ .grid = .{ .x = n_rows }, .block = .{ .x = 32 } }, stream, &b);
        try stream.synchronize();
        var got: [n_rows]u32 = undefined;
        try toks.download(0, std.mem.sliceAsBytes(&got));
        for (0..n_rows) |r| {
            var best: f32 = -std.math.inf(f32);
            var want: u32 = 0x7fffffff;
            for (host[r * n ..][0..n], 0..) |h, i| {
                const v = widen(h);
                if (v > best) {
                    best = v;
                    want = @intCast(i);
                }
            }
            cases += 1;
            if (got[r] != want) {
                differ += 1;
                std.debug.print("DIFFER {s} variant {d} width {d} row {d}: gpu {d} host {d}\n", .{ @tagName(kind), variant, n, r, got[r], want });
            }
        }
    };
    std.debug.print("RESULT q35-argmax: {d} rows, {d} differ from the host rule\n", .{ cases, differ });
    try check.expect(differ == 0, "q35-argmax: {d} of {d} rows differ", .{ differ, cases });
    check.pass("q35-argmax: {d} rows equal the host rule's greedy token", .{cases});
}
