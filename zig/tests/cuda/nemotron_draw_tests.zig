//! Nemotron's batched draws and row kernels: every row equals the stream-at-a-time path, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const Gpu = check.Gpu;

const Rule = cuda.sample.Rule;
const subset = 32768;
const vocab = 131072;
const most_rows = 8;

fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

fn fill(salt: u64, out: []u16) void {
    var x: u64 = 0x9E37_79B9_7F4A_7C15 ^ (salt *% 0xD1B5_4A32_D192_ED03);
    for (out, 0..) |*h, i| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        const r: f32 = @as(f32, @floatFromInt(x >> 40)) / 16777216.0;
        const v: f32 = if (i % 2711 == 7) 2.0 + 6.0 * r else -3.0 * r;
        h.* = bf16(v);
    }
}

fn ruleOf(seed: u64) Rule {
    return cuda.sample.rule(lanes.Sampling{ .seed = seed, .temperature = 1, .top_k = 20, .top_p = 0.95 });
}

/// The batched draw over 1 to 8 rows equals the per-row draw under 8 rules, subset and whole head.
pub fn run(gpu: Gpu) !void {
    const gpa = gpu.gpa;
    var m = try cuda.Module.load(gpu.d, cuda.kernels.sample);
    defer m.unload();
    const d = try cuda.sample.Draws.resolve(m);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const host = try gpa.alloc(u16, most_rows * vocab);
    defer gpa.free(host);
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * vocab * 2);
    defer logits.free();
    var rules = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * @sizeOf(Rule));
    defer rules.free();
    var meta = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer meta.free();
    var tok = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer tok.free();
    var prob = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer prob.free();
    var want_tok = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer want_tok.free();
    var want_prob = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer want_prob.free();
    const ids = try gpa.alloc(u32, subset);
    defer gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(4 * i + 1);
    const wide_ids = try gpa.alloc(i64, subset);
    defer gpa.free(wide_ids);
    for (wide_ids, ids) |*w, id| w.* = id;
    var ids_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(wide_ids));
    defer ids_dev.free();
    var diffs: usize = 0;
    var checked: usize = 0;
    for ([_]bool{ true, false }) |mapped| {
        const n: usize = if (mapped) subset else vocab;
        for (1..most_rows + 1) |rows| {
            fill(rows, host[0 .. rows * n]);
            try logits.uploadAsync(0, std.mem.sliceAsBytes(host[0 .. rows * n]), stream.handle);
            var rule_bytes: [most_rows * @sizeOf(Rule)]u8 = undefined;
            var meta_words: [most_rows]i32 = undefined;
            for (0..rows) |r| {
                const rl = ruleOf(0xA5A5 + r);
                @memcpy(rule_bytes[r * @sizeOf(Rule) ..][0..@sizeOf(Rule)], std.mem.asBytes(&rl));
                meta_words[r] = @intCast(100 + r);
            }
            try rules.uploadAsync(0, rule_bytes[0 .. rows * @sizeOf(Rule)], stream.handle);
            try meta.uploadAsync(0, std.mem.sliceAsBytes(meta_words[0..rows]), stream.handle);
            try d.drawIdsRows(stream, logits.ptr, n, rules.ptr, meta.ptr, 1, tok.ptr, rows, if (mapped) ids_dev.ptr else null, prob.ptr);
            for (0..rows) |r| try d.draw(stream, logits.ptr + r * n * 2, n, rules.ptr + r * @sizeOf(Rule), meta.ptr + r * 4, 1, want_tok.ptr + r * 4, 1, if (mapped) ids_dev.ptr else null, want_prob.ptr + r * 4);
            try stream.synchronize();
            const got_tok = try gpa.alloc(u32, rows);
            defer gpa.free(got_tok);
            const want = try gpa.alloc(u32, rows);
            defer gpa.free(want);
            const got_prob = try gpa.alloc(f32, rows);
            defer gpa.free(got_prob);
            const ref_prob = try gpa.alloc(f32, rows);
            defer gpa.free(ref_prob);
            try tok.download(0, std.mem.sliceAsBytes(got_tok));
            try want_tok.download(0, std.mem.sliceAsBytes(want));
            try prob.download(0, std.mem.sliceAsBytes(got_prob));
            try want_prob.download(0, std.mem.sliceAsBytes(ref_prob));
            for (0..rows) |r| {
                diffs += @intFromBool(got_tok[r] != want[r] or @as(u32, @bitCast(got_prob[r])) != @as(u32, @bitCast(ref_prob[r])));
                checked += 1;
            }
        }
    }
    try check.expect(diffs == 0, "nemotron-draws: {d} of {d} rows differ from the per-row draw", .{ diffs, checked });
    check.pass("nemotron-draws: {d} batched rows equal the per-row draw", .{checked});
    try greedy(gpu, stream);
    try targetRows(gpu, stream);
    try argmax(gpu, stream);
    try rowKernels(gpu, stream);
}

/// The target's rows draw over 1 to 8 rows equals one tf_draw a row at the same rule and position.
fn targetRows(gpu: Gpu, stream: cuda.Stream) !void {
    const gpa = gpu.gpa;
    var m = try cuda.Module.load(gpu.d, cuda.kernels.sample);
    defer m.unload();
    const d = try cuda.sample.Draws.resolve(m);
    const host = try gpa.alloc(u16, most_rows * vocab);
    defer gpa.free(host);
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * vocab * 2);
    defer logits.free();
    var rules = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * @sizeOf(Rule));
    defer rules.free();
    var meta = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer meta.free();
    var tok = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer tok.free();
    var want = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer want.free();
    var diffs: usize = 0;
    var checked: usize = 0;
    for (1..most_rows + 1) |n| {
        fill(n, host[0 .. n * vocab]);
        try logits.uploadAsync(0, std.mem.sliceAsBytes(host[0 .. n * vocab]), stream.handle);
        var rule_bytes: [most_rows * @sizeOf(Rule)]u8 = undefined;
        var meta_words: [most_rows]i32 = undefined;
        for (0..n) |r| {
            const rl = ruleOf(0x5A5A + r);
            @memcpy(rule_bytes[r * @sizeOf(Rule) ..][0..@sizeOf(Rule)], std.mem.asBytes(&rl));
            meta_words[r] = @intCast(700 + r);
        }
        try rules.uploadAsync(0, rule_bytes[0 .. n * @sizeOf(Rule)], stream.handle);
        try meta.uploadAsync(0, std.mem.sliceAsBytes(meta_words[0..n]), stream.handle);
        try d.drawRows(stream, logits.ptr, vocab, rules.ptr, meta.ptr, tok.ptr, n, null);
        for (0..n) |r| try d.draw(stream, logits.ptr + r * vocab * 2, vocab, rules.ptr + r * @sizeOf(Rule), meta.ptr + r * 4, 0, want.ptr + r * 4, 1, null, null);
        try stream.synchronize();
        const got = try gpa.alloc(u32, n);
        defer gpa.free(got);
        const ref = try gpa.alloc(u32, n);
        defer gpa.free(ref);
        try tok.download(0, std.mem.sliceAsBytes(got));
        try want.download(0, std.mem.sliceAsBytes(ref));
        for (0..n) |r| {
            diffs += @intFromBool(got[r] != ref[r]);
            checked += 1;
        }
    }
    try check.expect(diffs == 0, "nemotron-draws: {d} of {d} target rows differ", .{ diffs, checked });
    check.pass("nemotron-draws: {d} target rows equal one tf_draw a row", .{checked});
}

fn greedy(gpu: Gpu, stream: cuda.Stream) !void {
    const gpa = gpu.gpa;
    var mods: [6]cuda.Module = undefined;
    mods[0] = try cuda.Module.load(gpu.d, cuda.kernels.torch_argmax);
    mods[1] = try cuda.Module.load(gpu.d, cuda.kernels.torch_topk);
    mods[2] = try cuda.Module.load(gpu.d, cuda.kernels.torch_pointwise);
    mods[3] = try cuda.Module.load(gpu.d, cuda.kernels.torch_indexing);
    mods[4] = try cuda.Module.load(gpu.d, cuda.kernels.torch_movement);
    mods[5] = try cuda.Module.load(gpu.d, cuda.kernels.torch_nemotron_constants);
    defer for (&mods) |*m| m.unload();
    const tf = try nemotron.torch_ops.Functions.resolve(&mods);
    const t: nemotron.torch_ops.Torch = .{ .f = &tf, .s = stream };
    var km = try cuda.Module.load(gpu.d, cuda.kernels.nemotron_keyed);
    defer km.unload();
    var fns: nemotron.glue.Fns = undefined;
    fns.keyed = try km.function("tf_nemo_keyed_greedy");
    const g: nemotron.glue.Glue = .{ .set = null, .f = &fns, .s = stream, .emb = undefined, .norms = undefined, .moe = undefined };
    const host = try gpa.alloc(u16, most_rows * subset);
    defer gpa.free(host);
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * subset * 2);
    defer logits.free();
    var flog = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * subset * 4);
    defer flog.free();
    var vals = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * cuda.sample.max_candidates * 4);
    defer vals.free();
    var cols = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * cuda.sample.max_candidates * 8);
    defer cols.free();
    var cand = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * cuda.sample.max_candidates * 8);
    defer cand.free();
    var scratch = try cuda.DeviceBuffer.alloc(gpu.d, nemotron.torch_ops.topkScratchBytes(most_rows, subset));
    defer scratch.free();
    var invalid = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer invalid.free();
    var tok = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer tok.free();
    var prob = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer prob.free();
    var want_tok = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer want_tok.free();
    var want_prob = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer want_prob.free();
    const ids = try gpa.alloc(i64, subset);
    defer gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(4 * i + 1);
    var ids_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ids));
    defer ids_dev.free();
    const count = cuda.sample.count(nemotron.sampler.greedy_draft.k, subset);
    var diffs: usize = 0;
    var checked: usize = 0;
    for (1..most_rows + 1) |rows| {
        fill(rows, host[0 .. rows * subset]);
        try logits.uploadAsync(0, std.mem.sliceAsBytes(host[0 .. rows * subset]), stream.handle);
        try t.toF32(logits.ptr, flog.ptr, rows * subset);
        try t.topk(flog.ptr, subset, rows, count, vals.ptr, cols.ptr, scratch.ptr);
        try t.lookup(ids_dev.ptr, subset, cols.ptr, cand.ptr, rows * count, invalid.ptr);
        try g.keyed(vals.ptr, cand.ptr, 0, tok.ptr, 0, 0, prob.ptr, 0, rows, count, nemotron.sampler.greedy_draft);
        for (0..rows) |r| {
            try t.toF32(logits.ptr + r * subset * 2, flog.ptr, subset);
            try t.topk(flog.ptr, subset, 1, count, vals.ptr, cols.ptr, scratch.ptr);
            try t.lookup(ids_dev.ptr, subset, cols.ptr, cand.ptr, count, invalid.ptr);
            try g.keyed(vals.ptr, cand.ptr, 0, want_tok.ptr + r * 4, 0, 0, want_prob.ptr + r * 4, 0, 1, count, nemotron.sampler.greedy_draft);
        }
        try stream.synchronize();
        const got = try gpa.alloc(u32, rows);
        defer gpa.free(got);
        const want = try gpa.alloc(u32, rows);
        defer gpa.free(want);
        const gp = try gpa.alloc(f32, rows);
        defer gpa.free(gp);
        const wp = try gpa.alloc(f32, rows);
        defer gpa.free(wp);
        try tok.download(0, std.mem.sliceAsBytes(got));
        try want_tok.download(0, std.mem.sliceAsBytes(want));
        try prob.download(0, std.mem.sliceAsBytes(gp));
        try want_prob.download(0, std.mem.sliceAsBytes(wp));
        for (0..rows) |r| {
            diffs += @intFromBool(got[r] != want[r] or @as(u32, @bitCast(gp[r])) != @as(u32, @bitCast(wp[r])));
            checked += 1;
        }
    }
    try check.expect(diffs == 0, "nemotron-draws: {d} of {d} greedy rows differ", .{ diffs, checked });
    check.pass("nemotron-draws: {d} greedy rows equal the per-row keyed draw", .{checked});
}

fn argmax(gpu: Gpu, stream: cuda.Stream) !void {
    const gpa = gpu.gpa;
    var m = try cuda.Module.load(gpu.d, cuda.kernels.torch_argmax);
    defer m.unload();
    const f = try m.function("tf_argmax_rows_i32_kernel");
    const host = try gpa.alloc(u16, most_rows * vocab);
    defer gpa.free(host);
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * vocab * 2);
    defer logits.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer out.free();
    var want = try cuda.DeviceBuffer.alloc(gpu.d, most_rows * 4);
    defer want.free();
    var diffs: usize = 0;
    var checked: usize = 0;
    for (1..most_rows + 1) |rows| {
        fill(rows, host[0 .. rows * vocab]);
        try logits.uploadAsync(0, std.mem.sliceAsBytes(host[0 .. rows * vocab]), stream.handle);
        var a: cuda.Args = .{};
        a.add(logits.ptr);
        a.add(out.ptr);
        for ([_]usize{ rows, vocab, vocab }) |v| a.add(@as(i64, @intCast(v)));
        a.add(@as(i32, 0));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
        for (0..rows) |r| {
            var b: cuda.Args = .{};
            b.add(logits.ptr + r * vocab * 2);
            b.add(want.ptr + r * 4);
            for ([_]usize{ 1, vocab, vocab }) |v| b.add(@as(i64, @intCast(v)));
            b.add(@as(i32, 0));
            try cuda.launch.launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, stream, &b);
        }
        try stream.synchronize();
        const got = try gpa.alloc(u32, rows);
        defer gpa.free(got);
        const ref = try gpa.alloc(u32, rows);
        defer gpa.free(ref);
        try out.download(0, std.mem.sliceAsBytes(got));
        try want.download(0, std.mem.sliceAsBytes(ref));
        for (0..rows) |r| {
            diffs += @intFromBool(got[r] != ref[r]);
            checked += 1;
        }
    }
    try check.expect(diffs == 0, "nemotron-draws: {d} of {d} argmax rows differ", .{ diffs, checked });
    check.pass("nemotron-draws: {d} argmax rows equal the per-row argmax", .{checked});
}

fn rowKernels(gpu: Gpu, stream: cuda.Stream) !void {
    const gpa = gpu.gpa;
    var m = try cuda.Module.load(gpu.d, cuda.kernels.nemotron_ops);
    defer m.unload();
    const gather = try m.function("tf_nemo_gather_rows");
    const last = try m.function("tf_nemo_gather_last");
    const scatter = try m.function("tf_nemo_scatter_rows");
    const words = 16;
    var a = try cuda.DeviceBuffer.alloc(gpu.d, 3 * words * 4);
    defer a.free();
    var b = try cuda.DeviceBuffer.alloc(gpu.d, 3 * words * 4);
    defer b.free();
    var dst_a = try cuda.DeviceBuffer.alloc(gpu.d, 3 * words * 4);
    defer dst_a.free();
    var dst_b = try cuda.DeviceBuffer.alloc(gpu.d, 3 * words * 4);
    defer dst_b.free();
    const ha = try gpa.alloc(u32, 3 * words);
    defer gpa.free(ha);
    const hb = try gpa.alloc(u32, 3 * words);
    defer gpa.free(hb);
    for (ha, hb, 0..) |*x, *y, i| {
        x.* = @intCast(i + 1);
        y.* = @intCast(i + 101);
    }
    try a.uploadAsync(0, std.mem.sliceAsBytes(ha), stream.handle);
    try b.uploadAsync(0, std.mem.sliceAsBytes(hb), stream.handle);
    var g: nemotron.kernels.GatherRows = std.mem.zeroes(nemotron.kernels.GatherRows);
    g.a[0] = a.ptr;
    g.b[0] = b.ptr;
    g.rows[0] = 1;
    g.a[1] = a.ptr + words * 4;
    g.b[1] = b.ptr + words * 4;
    g.rows[1] = 1;
    g.row0[1] = 1;
    g.wa = words * 4;
    g.wb = words * 4;
    var args: cuda.Args = .{};
    args.add(g);
    args.add(dst_a.ptr);
    args.add(dst_b.ptr);
    try cuda.launch.launch(gather, .{ .grid = .{ .x = 2 }, .block = .{ .x = 256 } }, stream, &args);
    try stream.synchronize();
    const gathered = try gpa.alloc(u32, 2 * words);
    defer gpa.free(gathered);
    var diffs: usize = 0;
    // the gather, read before the last-row gather overwrites both rows: part k's row lands at dst row k
    try dst_a.download(0, std.mem.sliceAsBytes(gathered));
    if (!std.mem.eql(u32, gathered, ha[0 .. 2 * words])) diffs += 1;
    try dst_b.download(0, std.mem.sliceAsBytes(gathered));
    if (!std.mem.eql(u32, gathered, hb[0 .. 2 * words])) diffs += 1;
    var l: nemotron.kernels.LastRows = std.mem.zeroes(nemotron.kernels.LastRows);
    l.a[0] = a.ptr + words * 4;
    l.b[0] = b.ptr + words * 4;
    l.a[1] = a.ptr;
    l.b[1] = b.ptr;
    var largs: cuda.Args = .{};
    largs.add(l);
    largs.add(dst_a.ptr);
    largs.add(dst_b.ptr);
    for ([_]usize{ words * 4, words * 4, 2 }) |v| largs.add(@as(c_int, @intCast(v)));
    try cuda.launch.launch(last, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, stream, &largs);
    var s: nemotron.kernels.ScatterRows = std.mem.zeroes(nemotron.kernels.ScatterRows);
    s.tok[0] = dst_a.ptr;
    s.prob[0] = dst_b.ptr;
    s.rows[0] = 1;
    s.tok_at = 2;
    s.prob_at = 1;
    var sargs: cuda.Args = .{};
    sargs.add(a.ptr);
    sargs.add(b.ptr);
    sargs.add(s);
    try cuda.launch.launch(scatter, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, stream, &sargs);
    try stream.synchronize();
    const got_a = try gpa.alloc(u32, 2 * words);
    defer gpa.free(got_a);
    const got_b = try gpa.alloc(u32, 2 * words);
    defer gpa.free(got_b);
    try dst_a.download(0, std.mem.sliceAsBytes(got_a));
    try dst_b.download(0, std.mem.sliceAsBytes(got_b));
    // dst row 0 is source row 1 and row 1 row 0; row 0's token then at tok_at, its probability at prob_at (j, j - 1)
    const want_a = try gpa.alloc(u32, 2 * words);
    defer gpa.free(want_a);
    const want_b = try gpa.alloc(u32, 2 * words);
    defer gpa.free(want_b);
    for ([_][]u32{ want_a, want_b }, [_][]const u32{ ha, hb }) |w, h| {
        @memcpy(w[0..words], h[words .. 2 * words]);
        @memcpy(w[words .. 2 * words], h[0..words]);
    }
    want_a[@intCast(s.tok_at)] = ha[0];
    want_b[@intCast(s.prob_at)] = hb[0];
    if (!std.mem.eql(u32, got_a, want_a)) diffs += 1;
    if (!std.mem.eql(u32, got_b, want_b)) diffs += 1;
    try check.expect(diffs == 0, "nemotron-draws: {d} row kernels differ", .{diffs});
    check.pass("nemotron-draws: gather, ordered last-row gather and scatter keep their rows", .{});
}
