//! `gen` and `teacher`: greedy decoding and teacher-forced logits of raw id files, for a host reference to score.
const std = @import("std");
const hip = @import("hip");
const nem = @import("nemotron");
const group = @import("group.zig");

const Tensor = hip.ops.Tensor;

pub const usage = "       tf-nemotron-hip-test gen <model dir> <ids.bin> <out stem> [--steps N] " ++
    "[--no-mtp]\n       tf-nemotron-hip-test teacher <model dir> <ids.bin> <logits.bin> [--no-mtp] [--chunk " ++
    "N] [--tp W --rank R --master H --port P]\n       tf-nemotron-hip-test window <model dir> <ids.bin> " ++
    "<unused> [--steps K] [--no-mtp] [--tp W --rank R --master H --port P]\n       tf-nemotron-hip-test " ++
    "rounds <model dir> <ids.bin> <unused> [--steps K] [--streams N] [--no-mtp] [--tp W --rank R --master H " ++
    "--port P]\n       tf-nemotron-hip-test trace <model dir> <ids.bin> <residuals.bin> [--tp W --rank R " ++
    "--master H --port P]\n";

const Args = struct {
    dir: []const u8,
    ids: []const u8,
    out: []const u8,
    steps: usize = 32,
    streams: usize = 3,
    mtp: bool = true,
    chunk: usize = 512,
    g: group.Group = .{},
};

fn parse(args: []const [:0]const u8) !Args {
    if (args.len < 3) return error.MissingArgument;
    var a: Args = .{ .dir = args[0], .ids = args[1], .out = args[2] };
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--no-mtp")) {
            a.mtp = false;
        } else if (std.mem.eql(u8, args[i], "--steps")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            a.steps = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--chunk")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            a.chunk = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--streams")) {
            i += 1;
            if (i >= args.len) return error.MissingArgument;
            a.streams = try std.fmt.parseInt(usize, args[i], 10);
        } else if (!try a.g.option(args, &i)) return error.UnknownOption;
    }
    return a;
}

/// The u32 ids of a raw little-endian file.
pub fn readIds(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24));
    defer a.free(bytes);
    const out = try a.alloc(u32, bytes.len / 4);
    for (out, 0..) |*v, i| v.* = std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
    return out;
}

/// `n` values of `t` read back as fp32 into `out`, after the stream's work.
pub fn readBack(e: *nem.engine.Engine, a: std.mem.Allocator, t: Tensor, out: []f32) !void {
    const bytes = try a.alloc(u8, out.len * t.kind.size());
    defer a.free(bytes);
    try e.stream.synchronize();
    try hip.runtime.check(e.driver.api.hipMemcpyDtoH(bytes.ptr, @ptrFromInt(t.ptr), bytes.len));
    for (out, 0..) |*v, i| v.* = switch (t.kind) {
        .f32 => @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)))),
    };
}

pub fn argmax(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |x, i| if (x > row[best]) {
        best = i;
    };
    return @intCast(best);
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Timestamp.now(io, .awake).nanoseconds;
}

/// Greedy decoding after the prompt: `<out>.ids.bin` the new ids, `<out>.logits.bin` the fp32 row behind each.
pub fn gen(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    const a = try parse(args);
    const prompt = try readIds(gpa, io, a.ids);
    defer gpa.free(prompt);
    const e = try nem.engine.Engine.open(gpa, io, a.dir, .{ .capacity = 4096, .streams = 1, .mtp = a.mtp });
    defer e.deinit();
    const m = e.model_();
    const v = m.dims.vocab;
    var s = try e.openStream();
    defer e.closeStream(&s);
    const row = try gpa.alloc(f32, v);
    defer gpa.free(row);
    const ids_path = try std.fmt.allocPrint(gpa, "{s}.ids.bin", .{a.out});
    defer gpa.free(ids_path);
    const logits_path = try std.fmt.allocPrint(gpa, "{s}.logits.bin", .{a.out});
    defer gpa.free(logits_path);
    var rows: std.ArrayList(f32) = .empty;
    defer rows.deinit(gpa);
    const t0 = nowNs(io);
    const last = try e.prompt(&s, prompt);
    var o = e.ops(&e.arena);
    var y = try e.logits(o, nem.forward.at(last, (prompt.len - 1) * m.dims.hidden), 1);
    try readBack(e, gpa, y, row);
    const t1 = nowNs(io);
    std.debug.print(
        "prompt {d} tokens in {d:.3} s ({t} rows)\n",
        .{ prompt.len, @as(f64, @floatFromInt(t1 - t0)) / 1e9, e.act },
    );
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    var finite = true;
    for (0..a.steps) |_| {
        for (row) |x| finite = finite and std.math.isFinite(x);
        try rows.appendSlice(gpa, row);
        const next = argmax(row);
        try out.append(gpa, next);
        try s.cover(&e.pool, s.len + 1);
        const h = try e.round(&.{.{ .s = &s, .tokens = &.{next} }}, null);
        o = e.ops(&e.arena);
        s.len += 1;
        y = try e.roundLogits(o, h, 1);
        try readBack(e, gpa, y, row);
    }
    const t2 = nowNs(io);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ids_path, .data = std.mem.sliceAsBytes(out.items) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = logits_path, .data = std.mem.sliceAsBytes(rows.items) });
    std.debug.print(
        "decoded {d} tokens in {d:.3} s ({d:.1} tok/s), finite {}\n",
        .{
            a.steps,
            @as(f64, @floatFromInt(t2 - t1)) / 1e9,
            @as(f64, @floatFromInt(a.steps)) * 1e9 / @as(f64, @floatFromInt(t2 - t1)),
            finite,
        },
    );
    std.debug.print("ids:", .{});
    for (out.items) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
}

/// Every row's logits of the ids run as one prompt (chunked as the engine chunks), fp32 [rows][vocab] raw.
pub fn teacher(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    const a = try parse(args);
    const ids = try readIds(gpa, io, a.ids);
    defer gpa.free(ids);
    var j = try a.g.join(gpa, io, a.dir);
    defer j.close();
    const e = try a.g.engine(gpa, io, a.dir, j, .{ .capacity = 4096, .streams = 1, .mtp = a.mtp, .chunk = a.chunk });
    defer e.deinit();
    const m = e.model_();
    const v = m.dims.vocab;
    var s = try e.openStream();
    defer e.closeStream(&s);
    // only the last chunk's rows are kept
    const n = ids.len - (ids.len - 1) / a.chunk * a.chunk;
    const rows = try e.prompt(&s, ids);
    const o = e.ops(&e.arena);
    const y = try e.logits(o, rows, n);
    const all = try gpa.alloc(f32, n * v);
    defer gpa.free(all);
    try readBack(e, gpa, y, all);
    const out = try a.g.outPath(gpa, a.out);
    defer if (a.g.rank > 0) gpa.free(out);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out, .data = std.mem.sliceAsBytes(all) });
    var finite = true;
    for (all) |x| finite = finite and std.math.isFinite(x);
    std.debug.print(
        "teacher (rank {d} of {d}): the last {d} of {d} rows ({d}-row chunks), {d} logits, finite {}; greedy:",
        .{ a.g.rank, a.g.world, n, ids.len, a.chunk, v, finite },
    );
    for (0..n) |r| std.debug.print(" {d}", .{argmax(all[r * v ..][0..v])});
    std.debug.print("\n", .{});
}

/// The ids after the prompt once as one window of rows and once a row at a time: the same logits bits, or not.
pub fn window(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    const a = try parse(args);
    const ids = try readIds(gpa, io, a.ids);
    defer gpa.free(ids);
    var j = try a.g.join(gpa, io, a.dir);
    defer j.close();
    const e = try a.g.engine(gpa, io, a.dir, j, .{ .capacity = 4096, .streams = 2, .mtp = a.mtp });
    defer e.deinit();
    const m = e.model_();
    const v = m.dims.vocab;
    const k = a.steps;
    if (ids.len <= k) return error.MissingArgument;
    const prompt = ids[0 .. ids.len - k];
    const tail = ids[ids.len - k ..];
    const once = try gpa.alloc(f32, k * v);
    defer gpa.free(once);
    const each = try gpa.alloc(f32, k * v);
    defer gpa.free(each);
    const snapped = try gpa.alloc(f32, k * v);
    defer gpa.free(snapped);
    const back = try gpa.alloc(f32, 2 * v);
    defer gpa.free(back);
    const state_bytes = nem.pages.Stream.stateBytes(m.dims, e.act, e.cfg.count(.mamba));
    var snaps = try hip.DeviceBuffer.alloc(&e.driver, (k + 1) * state_bytes);
    defer snaps.free();
    // TF_NEMOTRON_PAGE_SHIFT=1: the single-row pass runs on other pages (a check that pages are interchangeable)
    const shift = if (std.c.getenv("TF_NEMOTRON_PAGE_SHIFT")) |flag| flag[0] == '1' else false;
    for ([_]u8{ 0, 1, 2 }) |mode| {
        var holder = try e.openStream();
        defer e.closeStream(&holder);
        if (shift and mode == 0) try holder.cover(&e.pool, 1);
        var s = try e.openStream();
        defer e.closeStream(&s);
        _ = try e.prompt(&s, prompt);
        var o = e.ops(&e.arena);
        if (mode > 0) {
            try s.cover(&e.pool, s.len + k);
            const sn: ?nem.round.Snaps = if (mode == 2) .{ .buf = snaps, .stride = state_bytes } else null;
            const base = s.len;
            const h = try e.round(&.{.{ .s = &s, .tokens = tail, .snaps = sn }}, null);
            o = e.ops(&e.arena);
            try readBack(e, gpa, try e.roundLogits(o, h, k), if (mode == 2) snapped else once);
            if (mode == 2 and k >= 2) {
                // roll back to the state after k - 2 rows and run the last two rows one at a time
                try hip.raw.copy(s.mem, 0, snaps.base() + (k - 2) * state_bytes, state_bytes, e.stream.handle);
                s.len = base + k - 2;
                for (0..2) |i| {
                    const h1 = try e.round(&.{.{ .s = &s, .tokens = tail[k - 2 + i ..][0..1] }}, null);
                    o = e.ops(&e.arena);
                    s.len += 1;
                    try readBack(e, gpa, try e.roundLogits(o, h1, 1), back[i * v ..][0..v]);
                }
            } else s.len += k;
        } else for (tail, 0..) |t, i| {
            try s.cover(&e.pool, s.len + 1);
            const h = try e.round(&.{.{ .s = &s, .tokens = &.{t} }}, null);
            o = e.ops(&e.arena);
            s.len += 1;
            try readBack(e, gpa, try e.roundLogits(o, h, 1), each[i * v ..][0..v]);
        }
    }
    var worst: f32 = 0;
    var first: ?usize = null;
    for (once, each, 0..) |x, y, i| if (x != y) {
        worst = @max(worst, @abs(x - y));
        if (first == null) first = i / v;
    };
    std.debug.print(
        "window (rank {d} of {d}): {d} rows at once and one at a time: {s}; first differing row {?d}, " ++
            "largest difference {e}\n",
        .{ a.g.rank, a.g.world, k, if (first == null) "the same bits" else "DIFFERENT", first, worst },
    );
    const snap_same = std.mem.eql(f32, snapped, once);
    const back_same = k < 2 or std.mem.eql(f32, back, each[(k - 2) * v ..][0 .. 2 * v]);
    std.debug.print(
        "snapshots: the snapshotted window {s} the plain one; rows after a rollback {s} the single steps\n",
        .{ if (snap_same) "equals" else "DIFFERS from", if (back_same) "equal" else "DIFFER from" },
    );
    if (first != null or !snap_same or !back_same) return error.WindowMismatch;
}

/// N streams of different prompts, each a window of its own width, in one round and each alone a row at a time.
pub fn rounds(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    const a = try parse(args);
    const ids = try readIds(gpa, io, a.ids);
    defer gpa.free(ids);
    var j = try a.g.join(gpa, io, a.dir);
    defer j.close();
    const n = a.streams;
    const k = a.steps;
    if (n == 0 or k == 0 or k > nem.round.max_window or ids.len <= k + n) return error.MissingArgument;
    const e = try a.g.engine(gpa, io, a.dir, j, .{ .capacity = 4096, .streams = n, .mtp = a.mtp });
    defer e.deinit();
    const m = e.model_();
    const v = m.dims.vocab;
    const state_bytes = nem.pages.Stream.stateBytes(m.dims, e.act, e.cfg.count(.mamba));
    var snaps = try hip.DeviceBuffer.alloc(&e.driver, (k + 1) * state_bytes);
    defer snaps.free();
    const streams = try gpa.alloc(nem.pages.Stream, n);
    defer gpa.free(streams);
    const rows = try gpa.alloc(nem.round.Rows, n);
    defer gpa.free(rows);
    // stream i's prompt ends i tokens earlier and its window holds k - (i mod k) rows, so positions and widths differ
    var total: usize = 0;
    var opened: usize = 0;
    defer for (streams[0..opened]) |*s| e.closeStream(s);
    for (streams, 0..) |*s, i| {
        s.* = try e.openStream();
        opened += 1;
        const width = k - i % k;
        _ = try e.prompt(s, ids[0 .. ids.len - k - i]);
        try s.cover(&e.pool, s.len + width);
        rows[i] = .{
            .s = s,
            .tokens = ids[ids.len - k - i ..][0..width],
            .snaps = if (i == 0) .{ .buf = snaps, .stride = state_bytes } else null,
        };
        total += width;
    }
    const once = try gpa.alloc(f32, total * v);
    defer gpa.free(once);
    const h = try e.round(rows, null);
    var o = e.ops(&e.arena);
    try readBack(e, gpa, try e.roundLogits(o, h, total), once);
    const each = try gpa.alloc(f32, total * v);
    defer gpa.free(each);
    // the same windows one stream and one row at a time, from the same prompts on fresh streams
    var at: usize = 0;
    var bad: usize = 0;
    for (rows, 0..) |r, i| {
        var s = try e.openStream();
        defer e.closeStream(&s);
        _ = try e.prompt(&s, ids[0 .. ids.len - k - i]);
        for (r.tokens, 0..) |t, row| {
            try s.cover(&e.pool, s.len + 1);
            const h1 = try e.round(&.{.{ .s = &s, .tokens = &.{t} }}, null);
            o = e.ops(&e.arena);
            s.len += 1;
            try readBack(e, gpa, try e.roundLogits(o, h1, 1), each[(at + row) * v ..][0..v]);
            if (!std.mem.eql(f32, once[(at + row) * v ..][0..v], each[(at + row) * v ..][0..v])) {
                bad += 1;
                if (bad == 1) std.debug.print(
                    "rounds: stream {d} row {d} differs between the shared round and its own step\n",
                    .{ i, row },
                );
            }
        }
        at += r.tokens.len;
    }
    std.debug.print(
        "rounds (rank {d} of {d}): {d} streams, {d} rows in one round and each stream alone a row at a " ++
            "time: {s} ({d} rows differ)\n",
        .{ a.g.rank, a.g.world, n, total, if (bad == 0) "the same bits" else "DIFFERENT", bad },
    );
    if (bad > 0) return error.WindowMismatch;
}

/// The rows of the residual after every block, fp32 [layers][rows][hidden] raw: a layer by layer view of two runs.
const Probe = struct {
    e: *nem.engine.Engine,
    gpa: std.mem.Allocator,
    into: []f32,
    width: usize,

    fn layerFn(ctx: *anyopaque, layer: usize, x: Tensor, rows: usize) anyerror!void {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        try readBack(p.e, p.gpa, x, p.into[layer * p.width ..][0 .. rows * p.e.model_().dims.hidden]);
    }
};

/// Every row of the residual after each block of the ids run as one prompt chunk.
pub fn trace(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    const a = try parse(args);
    const ids = try readIds(gpa, io, a.ids);
    defer gpa.free(ids);
    var j = try a.g.join(gpa, io, a.dir);
    defer j.close();
    const e = try a.g.engine(gpa, io, a.dir, j, .{ .capacity = 4096, .streams = 1, .mtp = false, .chunk = ids.len });
    defer e.deinit();
    const m = e.model_();
    const width = ids.len * m.dims.hidden;
    const all = try gpa.alloc(f32, e.cfg.layers * width);
    defer gpa.free(all);
    var probe: Probe = .{ .e = e, .gpa = gpa, .into = all, .width = width };
    var s = try e.openStream();
    defer e.closeStream(&s);
    try s.cover(&e.pool, ids.len);
    try e.fresh();
    const o = e.ops(&e.arena);
    const dev = try e.ids(o, ids);
    _ = try nem.forward.run(
        o,
        m,
        e.place[0..e.cfg.layers],
        .{ .pool = &e.pool, .stream = &s, .trace = .{ .ctx = &probe, .layer = Probe.layerFn } },
        dev,
        ids.len,
        0,
    );
    const out = try a.g.outPath(gpa, a.out);
    defer if (a.g.rank > 0) gpa.free(out);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out, .data = std.mem.sliceAsBytes(all) });
    std.debug.print(
        "trace (rank {d} of {d}): {d} blocks x {d} rows x {d}\n",
        .{ a.g.rank, a.g.world, e.cfg.layers, ids.len, m.dims.hidden },
    );
}
