//! TF_DSV41_HANDOFF_CHECK: lands a hand-off (`roundtrip` or a file) offline and compares it to a local prefill.
const std = @import("std");
const tf = @import("tensorfold");
const ds = tf.deepseek_v41;
const ho = tf.deepseek_v41_handoff;
const remote = @import("deepseek_v41_remote.zig");

const vocab = 129280;
const log = std.log.scoped(.handoff_check);

fn envText(name: [:0]const u8) ?[]const u8 {
    return (@import("sys").environment.Process{}).get(name);
}

fn argmax(row: []const f32) u32 {
    var best: u32 = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = @intCast(i);
    };
    return best;
}

/// ‖a - b‖ / ‖a‖ over a vocabulary row.
fn nrms(a: []const f32, b: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (a, b) |x, y| {
        const d: f64 = @as(f64, x) - y;
        num += d * d;
        den += @as(f64, x) * x;
    }
    return @sqrt(num / @max(den, 1e-30));
}

fn readTokens(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    return remote.parseIds(a, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)));
}

/// `ids` through slot 0 in workspace-sized chunks; the last chunk's final row (copied) when `last`.
fn feed(m: *ds.Model, a: std.mem.Allocator, ids: []const u32, last: bool) !?[]f32 {
    const rows = m.workspace.plan.rows;
    var at: usize = 0;
    while (at < ids.len) {
        const end = @min(at + rows, ids.len);
        if (last and end == ids.len) return try a.dupe(f32, try m.forward(ids[at..end], .last));
        _ = try m.forward(ids[at..end], .none);
        at = end;
    }
    return null;
}

/// `steps` greedy tokens after `first_logits` (`forced` replaces own picks); returns steps + 1 logit rows and picks.
fn decode(m: *ds.Model, a: std.mem.Allocator, first_logits: []const f32, forced: ?[]const u32, steps: usize, picks: []u32) ![]f32 {
    const rows = try a.alloc(f32, (steps + 1) * vocab);
    @memcpy(rows[0..vocab], first_logits);
    for (0..steps) |k| {
        picks[k] = argmax(rows[k * vocab ..][0..vocab]);
        const feed_id = [1]u32{if (forced) |f| f[k] else picks[k]};
        @memcpy(rows[(k + 1) * vocab ..][0..vocab], try m.forward(&feed_id, .last));
    }
    return rows;
}

fn compare(name: []const u8, want: []const f32, got: []const f32, steps: usize) void {
    var agree: usize = 0;
    var worst: f64 = 0;
    var total: f64 = 0;
    for (0..steps + 1) |k| {
        const r = want[k * vocab ..][0..vocab];
        const g = got[k * vocab ..][0..vocab];
        agree += @intFromBool(argmax(r) == argmax(g));
        const e = nrms(r, g);
        worst = @max(worst, e);
        total += e;
    }
    log.info("{s}: top-1 agrees on {d}/{d} steps, NRMS mean {d:.5} worst {d:.5}", .{ name, agree, steps + 1, total / @as(f64, @floatFromInt(steps + 1)), worst });
}

fn streamAgree(want: []const u32, got: []const u32) usize {
    var n: usize = 0;
    while (n < want.len and want[n] == got[n]) n += 1;
    return n;
}

/// Runs the check named by `spec` against slot 0 of `m`; every failure is logged and returned.
pub fn run(gpa: std.mem.Allocator, io: std.Io, m: *ds.Model, spec: []const u8, checkpoint: []const u8) !void {
    if (try @import("deepseek_v41_handoff_synth.zig").run(gpa, io, checkpoint, spec)) return;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const steps: usize = if (envText("TF_DSV41_HANDOFF_STEPS")) |s| try std.fmt.parseInt(usize, s, 10) else 16;
    const round = std.mem.eql(u8, spec, "roundtrip");
    const file: ?ho.blob.Blob = if (round) null else try remote.loadFile(a, io, spec, checkpoint);
    var prompt: []u32 = if (envText("TF_DSV41_HANDOFF_TOKENS")) |p| try readTokens(a, io, p) else blk: {
        const n: usize = if (file) |f| f.tokens.len + 64 else 1500;
        const t = try a.alloc(u32, n);
        for (t, 0..) |*id, i| id.* = @intCast(1200 + (i * 37) % 9000);
        break :blk t;
    };
    if (file) |f| {
        if (prompt.len <= f.tokens.len or !std.mem.eql(u32, prompt[0..f.tokens.len], f.tokens)) {
            if (envText("TF_DSV41_HANDOFF_TOKENS") != null) return error.PromptDoesNotExtendTheFile;
            const joined = try a.alloc(u32, f.tokens.len + 64);
            @memcpy(joined[0..f.tokens.len], f.tokens);
            for (joined[f.tokens.len..], 0..) |*id, i| id.* = @intCast(1200 + (i * 37) % 9000);
            prompt = joined;
        }
    }
    const upto: u32 = if (file) |f| @intCast(f.tokens.len) else if (envText("TF_DSV41_HANDOFF_UPTO")) |u| try std.fmt.parseInt(u32, u, 10) else @intCast(@max(1, (prompt.len - 1) / 128 * 128));
    if (prompt.len < upto + 1 or prompt.len + steps + 1 > m.caches[0].plan.capacity) return error.PromptTooShortOrLong;
    log.info("{d} prompt tokens, prefix {d}, {d} steps, {s}", .{ prompt.len, upto, steps, if (round) "Metal round trip" else spec });

    // the local prefill, with the native slot at `upto` kept for the round trip's export
    try m.reset();
    const t0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    _ = try feed(m, a, prompt[0..upto], false);
    const native = try m.snapshotSave(0);
    defer m.snapshotDrop(native);
    const first = (try feed(m, a, prompt[upto..], true)).?;
    const t1 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    const ref_picks = try a.alloc(u32, steps);
    const ref = try decode(m, a, first, null, steps, ref_picks);
    log.info("local prefill of {d} tokens: {d:.2} s", .{ prompt.len, @as(f64, @floatFromInt(t1 - t0)) / 1e9 });

    // the control: the native snapshot restored into the slot continues exactly as the local run did
    try m.reset();
    try m.snapshotRestore(0, native);
    const control = try decode(m, a, (try feed(m, a, prompt[upto..], true)).?, ref_picks, steps, try a.alloc(u32, steps));
    compare("control (the native snapshot restored)", ref, control, steps);

    // the hand-off: a file (or the round trip's own export) landed in a blank Snapshot as a pull lands it
    const snap = try m.snapshotBlank();
    defer m.snapshotDrop(snap);
    const parts = if (file != null) null else try exportNative(a, m, native, prompt[0..upto]);
    const bytes = if (parts) |p| try ho.blob.write(a, checkpoint, prompt[0..upto], p) else null;
    if (parts) |p| if (envText("TF_DSV41_HANDOFF_DUMP")) |path| {
        const as_stream = std.mem.endsWith(u8, path, ho.file.extension);
        const data = if (as_stream) try stream(a, checkpoint, prompt[0..upto], p) else bytes.?;
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
        log.info("wrote the round trip's hand-off {s} ({d} bytes, {d} tokens) to {s}", .{ if (as_stream) "stream, as a producer serves it," else "file", data.len, upto, path });
    };
    const landed = if (file) |f| f else try ho.blob.Blob.parse(a, bytes.?);
    var landing: ho.landing.Landing = undefined;
    const t2 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    try landing.init(gpa, upto, .{ .@"packed" = remote.packedOf(m, snap) });
    defer landing.deinit();
    try landed.feed(landing.decoderSink());
    _ = try landing.complete();
    try m.snapshotSeal(snap, prompt, upto);
    const t3 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    log.info("landed {d} tokens ({d} ring parts dropped) in {d:.1} ms", .{ upto, landing.dropped, @as(f64, @floatFromInt(t3 - t2)) / 1e6 });

    try m.reset();
    try m.snapshotRestore(0, snap);
    const tail_t0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    const got_first = (try feed(m, a, prompt[upto..], true)).?;
    log.info("local tail of {d} tokens after the restore: {d:.1} ms", .{ prompt.len - upto, @as(f64, @floatFromInt(std.Io.Timestamp.now(io, .awake).toNanoseconds() - tail_t0)) / 1e6 });
    const got = try decode(m, a, got_first, ref_picks, steps, try a.alloc(u32, steps));
    compare("hand-off, the reference's tokens fed", ref, got, steps);

    if (envText("TF_DSV41_HANDOFF_BISECT") != null) try bisect(m, a, native, snap, prompt, upto, ref_picks, ref, steps);

    // free-running greedy on the imported prefix
    try m.reset();
    try m.snapshotRestore(0, snap);
    const own_picks = try a.alloc(u32, steps);
    _ = try decode(m, a, (try feed(m, a, prompt[upto..], true)).?, null, steps, own_picks);
    log.info("hand-off, free-running: {d}/{d} tokens equal the local run's, ids {any}", .{ streamAgree(ref_picks, own_picks), steps, own_picks });
    log.info("local run's ids {any}", .{ref_picks});
    try m.reset();
}

/// The native slot's state through the library's Metal-to-neutral export: the parts a producer would send.
fn exportNative(a: std.mem.Allocator, m: *ds.Model, native: *const ds.Model.Snapshot, tokens: []const u32) ![]const ho.export_cuda.Part {
    const view = remote.packedOf(m, @constCast(native));
    return ho.export_metal.build(a, .{ .upto = tokens.len }, view);
}

/// `parts` as a producer's replies (`GET /v1/handoff/<id>` body): MANIFEST for `checkpoint`/`tokens`, then DATA frames.
fn stream(a: std.mem.Allocator, checkpoint: []const u8, tokens: []const u32, parts: []const ho.export_cuda.Part) ![]const u8 {
    const built = try tf.deepseek_v41_handoff_p2.layers(a, parts);
    const exp: tf.fabric.producer.Export = .{ .handoff = @splat(0x5a), .tokens = tokens, .first_token = 0, .tokens_per_row = 1, .layers = built.layers, .sources = built.sources };
    var out: std.Io.Writer.Allocating = .init(a);
    try ho.file.write(a, &out.writer, &exp, checkpoint, 0, 1);
    return out.written();
}

const Part = enum { swa, carry, comp, ixk };

/// `native` with only `part`'s regions replaced by the landed hand-off's: shows which part carries the difference.
fn graft(m: *ds.Model, native: *const ds.Model.Snapshot, landed: *ds.Model.Snapshot, part: Part, upto: u32) !*ds.Model.Snapshot {
    const out = try m.snapshotBlank();
    errdefer m.snapshotDrop(out);
    @memcpy(out.storage, native.storage);
    @memcpy(out.live, native.live);
    @memcpy(out.taps, native.taps);
    out.position = native.position;
    out.history = native.history;
    out.taps_row = native.taps_row;
    out.taps_n = native.taps_n;
    out.taps_rows_last = native.taps_rows_last;
    out.taps_rows_valid = native.taps_rows_valid;
    out.taps_base = native.taps_base;
    const pk = remote.packedOf(m, landed);
    const P = ho.metal.Packed;
    switch (part) {
        .swa => @memcpy(out.live[0..pk.ringBytes()], landed.live[0..pk.ringBytes()]),
        .carry => @memcpy(out.live[pk.ringBytes()..pk.liveBytes()], landed.live[pk.ringBytes()..pk.liveBytes()]),
        .comp, .ixk => for (P.ratios, 0..) |ratio, i| {
            const rows = @as(usize, upto) / ratio;
            const main = pk.mainOff(i);
            const index = main + @as(usize, pk.capacity / ratio) * P.main_stride;
            if (part == .comp) @memcpy(out.storage[main..][0 .. rows * P.main_stride], landed.storage[main..][0 .. rows * P.main_stride]) else @memcpy(out.storage[index..][0 .. rows * P.index_stride], landed.storage[index..][0 .. rows * P.index_stride]);
        },
    }
    return out;
}

/// TF_DSV41_HANDOFF_BISECT: one part kind from the hand-off, the rest from the local prefill, compared on logits.
fn bisect(m: *ds.Model, a: std.mem.Allocator, native: *const ds.Model.Snapshot, landed: *ds.Model.Snapshot, prompt: []const u32, upto: u32, ref_picks: []const u32, ref: []const f32, steps: usize) !void {
    try compareRows(m, a, native, landed, upto);
    inline for (comptime std.meta.tags(Part)) |part| {
        const hybrid = try graft(m, native, landed, part, upto);
        defer m.snapshotDrop(hybrid);
        try m.reset();
        try m.snapshotRestore(0, hybrid);
        const first = (try feed(m, a, prompt[upto..], true)).?;
        const rows = try decode(m, a, first, ref_picks, steps, try a.alloc(u32, steps));
        compare("only " ++ @tagName(part) ++ " from the hand-off", ref, rows, steps);
    }
}

fn bf(h: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, h) << 16)));
}

/// Relative RMS error of each landed part vs the native one (bf16 parts), per kind, with mean, worst and its layer.
fn compareRows(m: *ds.Model, a: std.mem.Allocator, native: *const ds.Model.Snapshot, landed: *ds.Model.Snapshot, upto: u32) !void {
    const want = try ho.export_metal.build(a, .{ .upto = upto }, remote.packedOf(m, @constCast(native)));
    const got = try ho.export_metal.build(a, .{ .upto = upto }, remote.packedOf(m, landed));
    inline for (comptime std.meta.tags(Part)) |kind| {
        var sum: f64 = 0;
        var worst: f64 = 0;
        var worst_layer: u32 = 0;
        var n: usize = 0;
        var each: std.Io.Writer.Allocating = .init(a);
        for (want, got) |w, g| {
            if (!std.mem.eql(u8, @tagName(w.spec.kind), @tagName(kind))) continue;
            var err: f64 = 0;
            var ref_sq: f64 = 0;
            if (w.spec.dtype == .f32) {
                for (0..w.bytes.len / 4) |i| {
                    const x: f64 = @as(f32, @bitCast(std.mem.readInt(u32, w.bytes[i * 4 ..][0..4], .little)));
                    const y: f64 = @as(f32, @bitCast(std.mem.readInt(u32, g.bytes[i * 4 ..][0..4], .little)));
                    err += (x - y) * (x - y);
                    ref_sq += x * x;
                }
            } else for (0..w.bytes.len / 2) |i| {
                const x = bf(std.mem.readInt(u16, w.bytes[i * 2 ..][0..2], .little));
                const y = bf(std.mem.readInt(u16, g.bytes[i * 2 ..][0..2], .little));
                err += (x - y) * (x - y);
                ref_sq += x * x;
            }
            const r = if (ref_sq > 0) @sqrt(err / ref_sq) else 0;
            sum += r;
            n += 1;
            try each.writer.print(" {d}:{d:.3}", .{ w.spec.layer, r });
            if (r > worst) {
                worst = r;
                worst_layer = w.spec.layer;
            }
        }
        if (n > 0) log.info("rows {s}: {d} parts, relative RMS vs the local prefill's: mean {d:.4}, worst {d:.4} (layer {d}); by layer{s}", .{ @tagName(kind), n, sum / @as(f64, @floatFromInt(n)), worst, worst_layer, each.written() });
    }
}
