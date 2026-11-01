//! LFM2's DSpark drafter on the GPU against an fp32 host reference of the released model, on the real weights.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.lfm2_moe;
const c = q.config;
const st = q.state;
const ck = tf.checkpoint;
const quant = q.dspark_quant;
const controls = @import("lfm2_dspark_controls.zig");

const hd = c.head_dim;
const kw = c.kv_heads * hd;
const qw = c.query_heads * hd;

/// The drafter's tensors as fp32 on the host, in the checkpoint's own layout (interleaved RoPE pairs).
const Ref = struct {
    gpa: std.mem.Allocator,
    check: ck.Checkpoint,
    cfg: q.dspark_config.Config,

    fn get(r: *const Ref, name: []const u8) ![]f32 {
        const t = try r.check.get(name);
        const src = t.host(u16);
        const out = try r.gpa.alloc(f32, src.len);
        for (out, src) |*o, v| o.* = quant.toF32(v);
        return out;
    }

    fn layer(r: *const Ref, buf: []u8, i: usize, part: []const u8) ![]f32 {
        return r.get(try std.fmt.bufPrint(buf, "layers.{d}.{s}.weight", .{ i, part }));
    }
};

/// y[n] = W[n, k] x[k] for each of `rows` rows, on every core.
fn matmul(gpa: std.mem.Allocator, w: []const f32, x: []const f32, y: []f32, n: usize, k: usize, rows: usize) !void {
    const Job = struct {
        fn run(w_: []const f32, x_: []const f32, y_: []f32, n_: usize, k_: usize, rows_: usize, from: usize, to: usize) void {
            for (0..rows_) |r| for (from..to) |col| {
                var s: f32 = 0;
                const wr = w_[col * k_ ..][0..k_];
                const xr = x_[r * k_ ..][0..k_];
                for (wr, xr) |a, b| s += a * b;
                y_[r * n_ + col] = s;
            };
        }
    };
    const threads = @min(std.Thread.getCpuCount() catch 1, 16);
    const per = (n + threads - 1) / threads;
    const hs = try gpa.alloc(std.Thread, threads);
    defer gpa.free(hs);
    var made: usize = 0;
    defer for (hs[0..made]) |h| h.join(); // a failed spawn still joins the workers already running
    var at: usize = 0;
    while (at < n) : (at += per) {
        hs[made] = try std.Thread.spawn(.{}, Job.run, .{ w, x, y, n, k, rows, at, @min(n, at + per) });
        made += 1;
    }
}

fn rms(x: []f32, gain: []const f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + c.eps);
    for (x, gain) |*v, g| v.* = v.* * inv * g;
}

/// Interleaved RoPE at `pos` over each head of `x` (pairs 2i, 2i + 1).
fn rope(x: []f32, pos: u64, theta: f32) void {
    var h: usize = 0;
    while (h < x.len) : (h += hd) for (0..hd / 2) |i| {
        const angle = @as(f32, @floatFromInt(pos)) * std.math.pow(f32, theta, -@as(f32, @floatFromInt(i)) / (hd / 2));
        const a = x[h + 2 * i];
        const b = x[h + 2 * i + 1];
        x[h + 2 * i] = a * @cos(angle) - b * @sin(angle);
        x[h + 2 * i + 1] = a * @sin(angle) + b * @cos(angle);
    };
}

/// The target's embedding row `id`, dequantized from its codes at the checkpoint's own width (scale * q + bias).
fn embedRow(m: *q.Model, id: u32, out: []f32) void {
    const e = m.weights.embedding;
    const bits: usize = e.bits;
    const row_bytes = c.hidden * bits / 8;
    const codes = (e.w.contents() + e.w_off)[@as(usize, id) * row_bytes ..][0..row_bytes];
    const sc = @as([*]const u16, @ptrCast(@alignCast(e.scales.contents() + e.s_off)))[@as(usize, id) * c.hidden / 64 ..][0 .. c.hidden / 64];
    const bi = @as([*]const u16, @ptrCast(@alignCast(e.biases.contents() + e.b_off)))[@as(usize, id) * c.hidden / 64 ..][0 .. c.hidden / 64];
    for (0..c.hidden) |j| {
        // codes are a little-endian bit stream: code j takes bits [j * bits, (j + 1) * bits)
        const at = j * bits;
        var word: u32 = codes[at / 8];
        if (at / 8 + 1 < row_bytes) word |= @as(u32, codes[at / 8 + 1]) << 8;
        const qv = (word >> @intCast(at % 8)) & ((@as(u32, 1) << @intCast(bits)) - 1);
        out[j] = quant.toF32(sc[j / 64]) * @as(f32, @floatFromInt(qv)) + quant.toF32(bi[j / 64]);
    }
}

/// GPU bf16 row against the reference's row + bias: cosine and worst difference.
fn compare(label: []const u8, gpu: []const u16, row: []const f32, bias: []const f32) void {
    var dot: f64 = 0;
    var gg: f64 = 0;
    var rr: f64 = 0;
    var worst: f32 = 0;
    for (gpu, row, bias) |g, b, m2| {
        const x = quant.toF32(g);
        const y = b + m2;
        dot += x * y;
        gg += x * x;
        rr += y * y;
        worst = @max(worst, @abs(x - y));
    }
    std.debug.print("{s}: cosine {d:.5}, worst difference {d:.3}\n", .{ label, dot / @sqrt(gg * rr), worst });
}

/// A hold's GPU ms: the blocks held in a signalled command buffer, waited on and read for its GPU times.
fn gpuHold(d: *q.dspark.Drafter, blocks: []const q.dspark_block.Block) !f64 {
    d.signalled += 1;
    const cb = (try q.dspark_block.hold(d, blocks, .{ .event = d.event, .value = d.signalled })).?;
    defer cb.release();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    return (cb.gpuEnd() - cb.gpuStart()) * 1000;
}

fn absorb(d: *q.dspark.Drafter, piece: q.dspark_block.Piece) !void {
    try q.dspark_block.absorbNow(d, &.{piece});
}

/// `rows` tapped rows from row `row` of `buf` (the target's rows), also starting at `row` in the context.
fn ownAbsorb(key: *tf.lanes.Stream, buf: mtl.Buffer, row_bytes: usize, row: usize, rows: u32, prompt: bool) tf.lanes.drafter.Absorb {
    return .{
        .stream = key,
        .features = .{ .buffer = @intFromPtr(buf.id), .offset = row * row_bytes, .rows = rows, .row_bytes = @intCast(row_bytes), .space = .device, .dtype = .bf16 },
        .start = row,
        .follow = &.{},
        .prompt = prompt,
    };
}


/// Served rounds through the drafter's interface: after four warm rounds, no host allocation and one hold pass a round.
fn steady(gpa: std.mem.Allocator, d: *q.dspark.Drafter, taps: mtl.Buffer, ids: []const u32, ctx: usize) !void {
    const lanes = tf.lanes;
    var s = try lanes.Stream.init(gpa, .{ .id = "steady", .prompt = ids[0 .. ctx + 1], .max_new = 160 });
    defer s.deinit(gpa);
    const dr = d.drafter();
    try dr.open(&s);
    defer dr.release(&s);
    const width: u32 = @intCast(d.tapBytes());
    const feats = struct {
        fn at(buf: mtl.Buffer, row: usize, rows: usize, w: u32) lanes.backend.Features {
            return .{ .buffer = @intFromPtr(buf.id), .offset = row * w, .rows = @intCast(rows), .row_bytes = w, .space = .device, .dtype = .bf16 };
        }
    };
    try dr.absorb(&.{.{ .stream = &s, .features = feats.at(taps, 0, ctx, width), .start = 0, .follow = &.{} }});
    var counted = std.testing.FailingAllocator.init(gpa, .{});
    defer d.gpa = gpa;
    var anchor: u64 = ctx;
    const rounds = 24;
    const warm = 4;
    var holds: u64 = 0;
    for (0..rounds) |r| {
        if (r == warm) {
            d.gpa = counted.allocator();
            holds = d.signalled;
        }
        try dr.hold(&.{.{ .stream = &s, .pending = ids[@intCast(anchor % ctx)], .position = anchor + 1, .depth = 3 }});
        var chances: [3]f64 = undefined;
        _ = try dr.chances(&s, &chances);
        var drafts: [3]u32 = undefined;
        try dr.held(&.{&s}, &.{&drafts});
        const rows = r % 3 + 1;
        const at: usize = @intCast(anchor % (ctx - 3));
        try dr.absorb(&.{.{ .stream = &s, .features = feats.at(taps, at, rows, width), .start = anchor, .follow = &.{}, .prompt = false }});
        anchor += rows;
    }
    const n = rounds - warm;
    std.debug.print("steady rounds: {d} rounds of hold, chances, held and a verified absorb: {d} host allocations, {d} resizes, {d} unwaited holds\n", .{ n, counted.allocations, counted.resize_index, d.signalled - holds });
    if (counted.allocations != 0 or counted.resize_index != 0 or d.signalled - holds != n) return error.SteadyRoundsAllocate;
}

/// Replay owned context after long zero-grant stretches, tap overwrites and a rewind inside the pending tail.
fn deferred(gpa: std.mem.Allocator, d: *q.dspark.Drafter, taps: mtl.Buffer, ids: []const u32, ctx: usize) !void {
    const skip = q.dspark.stage_rows + 33;
    const anchor = ctx + skip;
    const width = d.tapBytes();
    const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const history = try d.target.device.buffer(anchor * width, shared);
    defer history.deinit();
    @memcpy(history.contents()[0 .. ctx * width], taps.contents()[0 .. ctx * width]);
    const source = try d.target.device.buffer(width, shared);
    defer source.deinit();
    const reference = try d.newLane(anchor + 64);
    defer d.dropLane(reference);
    var s = try tf.lanes.Stream.init(gpa, .{ .id = "deferred", .prompt = ids[0 .. ctx + 1], .max_new = skip + 64 });
    defer s.deinit(gpa);
    const dr = d.drafter();
    try dr.open(&s);
    defer dr.release(&s);
    const live = d.lanes.get(&s).?;
    try dr.absorb(&.{.{ .stream = &s, .features = .{ .buffer = @intFromPtr(taps.id), .rows = @intCast(ctx), .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 }, .start = 0, .follow = &.{}, .propose = false }});
    var at: usize = 0;
    while (at < live.end) {
        const n = @min(q.dspark.absorb_rows, live.end - at);
        try absorb(d, .{ .lane = reference, .src = .{ .buf = history, .off = at * width }, .rows = n, .p0 = at });
        at += n;
    }
    var rewound: usize = 0;
    for (0..2) |phase| {
        const count: usize = if (phase == 0) skip else @min(29, @as(usize, live.staged));
        const first = if (phase == 0) ctx else anchor - count;
        if (phase == 1) rewound = count;
        for (0..count) |i| {
            const pos = first + i;
            const row = ((phase * skip + i) * 71 + 17) % ctx;
            @memcpy(source.contents()[0..width], (taps.contents() + row * width)[0..width]);
            @memcpy((history.contents() + pos * width)[0..width], source.contents()[0..width]);
            const before = live.end;
            const staged = live.staged;
            try dr.absorb(&.{.{ .stream = &s, .features = .{ .buffer = @intFromPtr(source.id), .rows = 1, .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 }, .start = pos, .follow = &.{}, .prompt = false, .propose = false }});
            @memset(source.slice(u16, width / 2), 0x7fc0); // the target's next forward overwrites borrowed taps
            if (live.end != before) {
                // Match projection batches, not source storage: the oracle reads the immutable complete history.
                if (staged > 0) try absorb(d, .{ .lane = reference, .src = .{ .buf = history, .off = before * width }, .rows = staged, .p0 = before });
                try absorb(d, .{ .lane = reference, .src = .{ .buf = history, .off = pos * width }, .rows = 1, .p0 = pos });
            }
        }
    }
    if (live.staged > 0) try absorb(d, .{ .lane = reference, .src = .{ .buf = history, .off = live.end * width }, .rows = live.staged, .p0 = live.end });
    reference.end = anchor;
    _ = try q.dspark_block.hold(d, &.{.{ .lane = reference, .pending = ids[ctx], .anchor = anchor, .depth = 3 }}, null);
    const want = reference.held[0..3].*;
    try dr.hold(&.{.{ .stream = &s, .pending = ids[ctx], .position = anchor + 1, .depth = 3 }});
    var got: [3]u32 = undefined;
    try dr.held(&.{&s}, &.{&got});
    if (!std.mem.eql(u32, &want, &got)) return error.DeferredDraftMismatch;
    const bytes = anchor * c.head_dim * 2;
    for (0..d.config.layers) |layer| for ([_]bool{ false, true }) |values| for (0..c.kv_heads) |head| {
        const actual = live.kv(layer, values);
        const expected = reference.kv(layer, values);
        const a = (live.mem.contents() + actual.off + head * live.cap * c.head_dim * 2)[0..bytes];
        const b = (reference.mem.contents() + expected.off + head * reference.cap * c.head_dim * 2)[0..bytes];
        if (!std.mem.eql(u8, a, b)) return error.DeferredContextMismatch;
    };
    std.debug.print("deferred context: {d} zero-grant rows, {d} overwritten pending rows, {d} complete context rows and 3 later drafts match the independent replay\n", .{ skip, rewound, anchor });
}


pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4 or args.len > 6) return error.ExpectedModelDrafterPromptLengthRowsBits;
    const bits: u8 = if (args.len == 6) try std.fmt.parseInt(u8, args[5], 10) else 0;
    const length = try std.fmt.parseInt(usize, args[3], 10);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.Model.load(gpa, init.io, args[1]);
    defer m.deinit();
    const d = try q.dspark.Drafter.init(gpa, init.io, m, args[2], .{ .capacity = length + q.dspark.stage_rows + 512, .bits = bits });
    defer d.deinit();
    const cfg = d.config;
    // the compared block's rows (a hold of depth d runs a d-row block)
    const block: u32 = if (args.len >= 5) try std.fmt.parseInt(u32, args[4], 10) else cfg.block;
    if (block == 0 or block > cfg.block) return error.BadBlockRows;
    var scratch = try st.Scratch.init(gpa, m.device, &m.config.attention, length);
    defer scratch.deinit();
    const width = cfg.tap_count * c.hidden;
    var taps: st.Taps = .{ .count = cfg.tap_count, .buf = try scratch.memory.alloc(length * width * 2) };
    @memcpy(taps.layers[0..cfg.tap_count], cfg.taps[0..cfg.tap_count]);
    scratch.taps = taps;
    var cache = try st.Cache.init(gpa, m.device, &m.config.attention, length + 16);
    defer cache.deinit();
    const ids = try gpa.alloc(u32, length);
    defer gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast((i * 7919 + 13) % 120000);
    try controls.targetTaps(gpa, m, &scratch, ids);
    try q.forward.run(m, &scratch, &.{.{ .cache = &cache, .rows = length }}, ids, false, .last);
    // Repeated target layers are legal: every requested column must receive the same hidden row.
    const tapped = taps.buf.slice(u16, length * width);
    for (cfg.taps[0..cfg.tap_count], 0..) |layer, i| for (cfg.taps[0..i], 0..) |other, j| {
        if (layer != other) continue;
        for (0..length) |row| {
            const a = tapped[row * width + i * c.hidden ..][0..c.hidden];
            const b = tapped[row * width + j * c.hidden ..][0..c.hidden];
            if (!std.mem.eql(u16, a, b)) return error.RepeatedTapMismatch;
        }
        std.debug.print("ok repeated tap {d}, columns {d} and {d}, {d} rows\n", .{ layer, j, i, length });
    };
    // the context: rows [0, length - 1); the pending token is the last id, at position length - 1
    const ctx = length - 1;
    const lane = try d.newLane(length + 32);
    defer d.dropLane(lane);
    var at: usize = 0;
    while (at < ctx) : (at += q.dspark.absorb_rows) {
        const rows = @min(q.dspark.absorb_rows, ctx - at);
        try absorb(d, .{ .lane = lane, .src = .{ .buf = taps.buf, .off = at * width * 2 }, .rows = rows, .p0 = at });
    }
    lane.end = ctx;
    _ = try q.dspark_block.hold(d, &.{.{ .lane = lane, .pending = ids[ctx], .anchor = ctx, .depth = block }}, null);
    var held: [16]u32 = undefined;
    @memcpy(held[0..block], lane.held[0..block]);
    var conf: [16]f32 = undefined;
    @memcpy(conf[0..block], lane.conf[0..block]);
    const gpu = held[0..block];
    // The owned stage takes uneven absorbs (prompt rows first, last rows taken again) and holds them on the host.
    // One hold absorbs them as one piece of ctx rows from position 0; those staged bytes come from the same taps.
    // The direct absorbs above are also one piece while ctx <= absorb_rows, so fc, k, v and qk see the same row count
    // and run the same kernels: bf16 uses 16-row kernels up to 32 rows, then 64x64 tiles; 8-bit uses twin kernels up to
    // 32 rows, then tensor or tile kernels. Keys, values and picks must be exactly equal, without tolerance.
    if (ctx > 3 and ctx <= @min(q.dspark.absorb_rows, @min(q.dspark.stage_rows, lane.cap))) {
        const key: *tf.lanes.Stream = @ptrFromInt(0x1000);
        const own = try d.newLane(length + 32);
        d.lanes.put(gpa, key, own) catch |err| {
            d.dropLane(own);
            return err;
        };
        const staged = d.drafter();
        const sizes = [_]u32{ 5, 1, 9, 33, 2 };
        const row_bytes = width * 2;
        var pos: usize = 0;
        var step: usize = 0;
        while (pos < ctx) : (step += 1) {
            const n: u32 = @intCast(@min(sizes[step % sizes.len], ctx - pos));
            try staged.absorb(&.{ownAbsorb(key, taps.buf, row_bytes, pos, n, step == 0)});
            pos += n;
        }
        try staged.absorb(&.{ownAbsorb(key, taps.buf, row_bytes, ctx - 3, 3, false)});
        if (own.staged != ctx or own.end != 0) return error.OwnedStageMissesRows;
        try staged.hold(&.{.{ .stream = key, .pending = ids[ctx], .position = ctx + 1, .depth = block }});
        var own_held: [16]u32 = undefined;
        try staged.held(&.{key}, &.{own_held[0..block]});
        if (!std.mem.eql(u32, gpu, own_held[0..block])) return error.OwnedStagePicksDiffer;
        const span = ctx * hd * 2;
        for (0..cfg.layers) |li| for ([_]bool{ false, true }) |values| {
            const reference = lane.kv(li, values);
            const staged_kv = own.kv(li, values);
            for (0..c.kv_heads) |h| {
                const reference_off = reference.off + h * lane.cap * hd * 2;
                const staged_off = staged_kv.off + h * own.cap * hd * 2;
                if (!std.mem.eql(u8, (reference.buf.contents() + reference_off)[0..span], (staged_kv.buf.contents() + staged_off)[0..span])) return error.OwnedStageContextDiffers;
            }
        };
        std.debug.print("the owned stage: {d} rows staged in uneven absorbs, absorbed as one {d}-row piece like the direct absorbs' ({s} regime): keys, values and picks equal\n", .{ ctx, ctx, if (ctx <= 32) "<= 32-row" else "> 32-row" });
    }
    try @import("lfm2_dspark_lifecycle.zig").run(gpa, d, taps.buf, ids, ctx);
    // settled GPU time is taken once: a standalone absorb adds its time and leaves the grid as it was; a synchronous
    // one-block hold adds its own and reprices exactly the passes at one block (other block counts stay as timed)
    if (ctx > 0) {
        const grid = d.holds;
        const grid_len = d.holds_len;
        const step_ms = d.step_ms;
        const saved_hold_ms = d.hold_ms;
        const spent_ms = d.hold_spent_ms;
        defer {
            d.holds = grid;
            d.holds_len = grid_len;
            d.step_ms = step_ms;
            d.hold_ms = saved_hold_ms;
            d.hold_spent_ms = spent_ms;
        }
        _ = q.dspark_block.takeHoldMs(d);
        try absorb(d, .{ .lane = lane, .src = .{ .buf = taps.buf, .off = (ctx - 1) * width * 2 }, .rows = 1, .p0 = ctx - 1 });
        const absorb_ms = q.dspark_block.takeHoldMs(d);
        if (!(absorb_ms > 0)) return error.AbsorbTimeNotTaken;
        for (grid[0..grid_len], d.holds[0..grid_len]) |before, now| if (before.ms != now.ms) return error.AbsorbRepricedAHold;
        _ = try q.dspark_block.hold(d, &.{.{ .lane = lane, .pending = ids[ctx], .anchor = ctx, .depth = block }}, null);
        const hold_ms = q.dspark_block.takeHoldMs(d);
        if (!(hold_ms > 0) or q.dspark_block.takeHoldMs(d) != 0) return error.HoldTimeNotTakenOnce;
        if (d.holds_len != grid_len) return error.GridResized;
        var moved = false;
        for (grid[0..grid_len], d.holds[0..grid_len]) |before, now| {
            if (before.streams != 1 and before.ms != now.ms) return error.OtherBlockCountsRepriced;
            if (before.streams == 1 and before.ms != now.ms) moved = true;
        }
        if (!moved) return error.SyncHoldNotRepriced;
        std.debug.print("a synchronous hold of one block: {d:.2} ms GPU taken once, the one-block passes repriced toward it; a standalone absorb {d:.2} ms priced none\n", .{ hold_ms, absorb_ms });
    }
    if (cfg.block < 9) {
        std.debug.print("split-hold fixture skipped: this drafter's block is {d} rows and its depths [9, 1 x 7] need 9\n", .{cfg.block});
    } else try controls.splitHold(gpa, init.io, d, ids);
    for ([_]usize{ 1, 4, 16, 128, q.dspark.absorb_rows }) |rows| {
        if (rows > ctx) continue;
        var best: f64 = std.math.floatMax(f64);
        for (0..5) |_| {
            const t0 = std.Io.Clock.awake.now(init.io);
            try absorb(d, .{ .lane = lane, .src = .{ .buf = taps.buf, .off = (ctx - rows) * width * 2 }, .rows = rows, .p0 = ctx - rows });
            best = @min(best, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds())) / 1e6);
        }
        std.debug.print("absorb of {d} rows: {d:.2} ms (best of 5)\n", .{ rows, best });
    }
    const depths = [_]u32{ 1, 3, cfg.block };
    for (depths) |depth| {
        var best: f64 = std.math.floatMax(f64);
        for (0..5) |_| {
            const t0 = std.Io.Clock.awake.now(init.io);
            _ = try q.dspark_block.hold(d, &.{.{ .lane = lane, .pending = ids[ctx], .anchor = ctx, .depth = depth }}, null);
            best = @min(best, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds())) / 1e6);
        }
        std.debug.print("hold at depth {d}: {d:.2} ms (best of 5)\n", .{ depth, best });
    }
    // a shared round's hold: S streams' blocks of d rows in one pass (S x d at most pass_rows), each on this context
    var lanes_: [16]*q.dspark.Lane = undefined;
    lanes_[0] = lane;
    var made: usize = 1;
    defer for (lanes_[1..made]) |l| d.dropLane(l);
    while (made < lanes_.len) : (made += 1) {
        const l = try d.newLane(length + 32);
        lanes_[made] = l;
        var from: usize = 0;
        while (from < ctx) : (from += q.dspark.absorb_rows) {
            const rows = @min(q.dspark.absorb_rows, ctx - from);
            try absorb(d, .{ .lane = l, .src = .{ .buf = taps.buf, .off = from * width * 2 }, .rows = rows, .p0 = from });
        }
        l.end = ctx;
    }
    for ([_]usize{ 2, 4, 8, 16 }) |streams| for ([_]u32{ 1, 2, 3, 4, 8 }) |depth| {
        if (streams * depth > q.dspark.pass_rows) continue;
        var blocks: [16]q.dspark_block.Block = undefined;
        for (blocks[0..streams], lanes_[0..streams]) |*b, l| b.* = .{ .lane = l, .pending = ids[ctx], .anchor = ctx, .depth = depth };
        var best: f64 = std.math.floatMax(f64);
        var gpu_best: f64 = std.math.floatMax(f64);
        for (0..5) |_| {
            const t0 = std.Io.Clock.awake.now(init.io);
            _ = try q.dspark_block.hold(d, blocks[0..streams], null);
            best = @min(best, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds())) / 1e6);
            gpu_best = @min(gpu_best, try gpuHold(d, blocks[0..streams]));
        }
        std.debug.print("hold of {d} streams at depth {d}: {d:.2} ms, GPU {d:.2} ms (best of 5)\n", .{ streams, depth, best, gpu_best });
    };
    try steady(gpa, d, taps.buf, ids, ctx);
    try deferred(gpa, d, taps.buf, ids, ctx);
    _ = try q.dspark_block.hold(d, &.{.{ .lane = lane, .pending = ids[ctx], .anchor = ctx, .depth = block }}, null);
    // the host reference, fp32 throughout
    var r: Ref = .{ .gpa = gpa, .check = ck.Checkpoint.init(gpa), .cfg = cfg };
    defer r.check.deinit();
    const path = try std.fs.path.joinZ(gpa, &.{ args[2], "model.safetensors" });
    defer gpa.free(path);
    try r.check.addFile(m.device, path, "");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    r.gpa = arena.allocator();
    const a = r.gpa;
    const n = block;
    const tap_rows = taps.buf.slice(u16, length * width);
    const feats = try a.alloc(f32, ctx * width);
    for (feats, tap_rows[0 .. ctx * width]) |*f, v| f.* = quant.toF32(v);
    const hctx = try a.alloc(f32, ctx * c.hidden);
    try matmul(gpa, try r.get("fc.weight"), feats, hctx, c.hidden, width, ctx);
    const hn = try r.get("hidden_norm.weight");
    for (0..ctx) |j| rms(hctx[j * c.hidden ..][0..c.hidden], hn);
    const h = try a.alloc(f32, n * c.hidden);
    for (0..n) |i| embedRow(m, if (i == 0) ids[ctx] else cfg.mask, h[i * c.hidden ..][0..c.hidden]);
    const keys = ctx + n;
    var name: [96]u8 = undefined;
    const x = try a.alloc(f32, n * c.hidden);
    for (0..cfg.layers) |li| {
        const kmat = try r.layer(&name, li, "self_attn.k_proj");
        const vmat = try r.layer(&name, li, "self_attn.v_proj");
        const qn = try r.layer(&name, li, "self_attn.q_norm");
        const kn = try r.layer(&name, li, "self_attn.k_norm");
        const kk = try a.alloc(f32, keys * kw);
        const vv = try a.alloc(f32, keys * kw);
        try matmul(gpa, kmat, hctx, kk, kw, c.hidden, ctx);
        try matmul(gpa, vmat, hctx, vv, kw, c.hidden, ctx);
        @memcpy(x, h);
        for (0..n) |i| rms(x[i * c.hidden ..][0..c.hidden], try r.layer(&name, li, "input_layernorm"));
        const qq = try a.alloc(f32, n * qw);
        try matmul(gpa, try r.layer(&name, li, "self_attn.q_proj"), x, qq, qw, c.hidden, n);
        try matmul(gpa, kmat, x, kk[ctx * kw ..], kw, c.hidden, n);
        try matmul(gpa, vmat, x, vv[ctx * kw ..], kw, c.hidden, n);
        for (0..keys) |j| for (0..c.kv_heads) |kh| {
            const row = kk[j * kw + kh * hd ..][0..hd];
            rms(row, kn);
            rope(row, j, cfg.theta);
        };
        for (0..n) |i| for (0..c.query_heads) |qh| {
            const row = qq[i * qw + qh * hd ..][0..hd];
            rms(row, qn);
            rope(row, ctx + i, cfg.theta);
        };
        const o = try a.alloc(f32, n * qw);
        const sc = try a.alloc(f32, keys);
        for (0..n) |i| for (0..c.query_heads) |qh| {
            const kh = qh / (c.query_heads / c.kv_heads);
            var top: f32 = -std.math.inf(f32);
            for (0..keys) |j| {
                var s: f32 = 0;
                for (0..hd) |t| s += qq[i * qw + qh * hd + t] * kk[j * kw + kh * hd + t];
                sc[j] = s / @sqrt(@as(f32, hd));
                top = @max(top, sc[j]);
            }
            var sum: f32 = 0;
            for (sc) |*s| {
                s.* = @exp(s.* - top);
                sum += s.*;
            }
            for (0..hd) |t| {
                var acc: f32 = 0;
                for (0..keys) |j| acc += sc[j] * vv[j * kw + kh * hd + t];
                o[i * qw + qh * hd + t] = acc / sum;
            }
        };
        const y = try a.alloc(f32, n * c.hidden);
        try matmul(gpa, try r.layer(&name, li, "self_attn.o_proj"), o, y, c.hidden, qw, n);
        for (h, y) |*v, w| v.* += w;
        @memcpy(x, h);
        for (0..n) |i| rms(x[i * c.hidden ..][0..c.hidden], try r.layer(&name, li, "post_attention_layernorm"));
        const gate = try a.alloc(f32, n * cfg.intermediate);
        const up = try a.alloc(f32, n * cfg.intermediate);
        try matmul(gpa, try r.layer(&name, li, "mlp.gate_proj"), x, gate, cfg.intermediate, c.hidden, n);
        try matmul(gpa, try r.layer(&name, li, "mlp.up_proj"), x, up, cfg.intermediate, c.hidden, n);
        for (gate, up) |*g, u| g.* = g.* / (1 + @exp(-g.*)) * u;
        try matmul(gpa, try r.layer(&name, li, "mlp.down_proj"), gate, y, c.hidden, cfg.intermediate, n);
        for (h, y) |*v, w| v.* += w;
    }
    const fin = try r.get("norm.weight");
    for (0..n) |i| rms(h[i * c.hidden ..][0..c.hidden], fin);
    const head = try a.alloc(f32, c.vocab * c.hidden);
    for (0..c.vocab) |v| embedRow(m, @intCast(v), head[v * c.hidden ..][0..c.hidden]);
    const base = try a.alloc(f32, n * c.vocab);
    try matmul(gpa, head, h, base, c.vocab, c.hidden, n);
    const w1 = try r.get("markov_head.markov_w1.weight");
    const w2 = try r.get("markov_head.markov_w2.weight");
    const bias = try a.alloc(f32, c.vocab);
    const proj = try r.get("confidence_head.proj.weight");
    const proj_bias = (try r.get("confidence_head.proj.bias"))[0];
    var conf_worst: f32 = 0;
    var prev = ids[ctx];
    var same: usize = 0;
    var worst: f32 = 0;
    for (0..n) |i| {
        try matmul(gpa, w2, w1[@as(usize, prev) * cfg.rank ..][0..cfg.rank], bias, c.vocab, cfg.rank, 1);
        const row = base[i * c.vocab ..][0..c.vocab];
        var best: usize = 0;
        var second: usize = 1;
        for (row, bias, 0..) |b, m2, v| {
            const s = b + m2;
            if (s > row[best] + bias[best]) {
                second = best;
                best = v;
            } else if (v != best and s > row[second] + bias[second]) second = v;
        }
        const margin = (row[best] + bias[best]) - (row[second] + bias[second]);
        if (i == 0) { // the GPU keeps the base logits (the Markov step adds its bias on the way to the pick)
            const zero = try a.alloc(f32, c.vocab);
            @memset(zero, 0);
            compare("row 0 base logits", d.s.logits.slice(u16, c.vocab), row, zero);
        }
        const ok = gpu[i] == best;
        same += @intFromBool(ok);
        std.debug.print("step {d}: reference {d} (margin {d:.3}, second {d}), GPU {d} {s}\n", .{ i, best, margin, second, gpu[i], if (ok) "equal" else "DIFFERENT" });
        if (!ok) worst = @max(worst, margin);
        // the confidence head: [final norm row i, W1 of the token before draft i] . proj + bias, through a sigmoid
        var logit: f32 = proj_bias;
        for (h[i * c.hidden ..][0..c.hidden], proj[0..c.hidden]) |x0, w0| logit += x0 * w0;
        for (w1[@as(usize, prev) * cfg.rank ..][0..cfg.rank], proj[c.hidden..]) |x0, w0| logit += x0 * w0;
        const want = 1 / (1 + @exp(-logit));
        conf_worst = @max(conf_worst, @abs(want - conf[i]));
        std.debug.print("step {d}: confidence reference {d:.4}, GPU {d:.4}\n", .{ i, want, conf[i] });
        prev = gpu[i]; // teacher-forced: step i + 1 follows the GPU's pick, so each step is compared alone
    }
    std.debug.print("the reference's largest margin over a different GPU pick: {d:.3}\n", .{worst});
    std.debug.print("the confidences' worst difference from the fp32 reference: {d:.5}\n", .{conf_worst});
    std.debug.print("{d} of {d} drafts equal the fp32 reference's picks on the same chain ({d}-token context)\n", .{ same, n, ctx });
}
