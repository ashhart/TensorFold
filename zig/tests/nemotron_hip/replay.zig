//! `replay`: a traced lane run (windows, keeps, drafts) replayed on the engine; each window's first draw checked.
const std = @import("std");
const hip = @import("hip");
const nem = @import("nemotron");
const gen = @import("gen.zig");
const group = @import("group.zig");

const Tensor = hip.ops.Tensor;

pub const usage = "       tf-nemotron-hip-test replay <model dir> <prompt ids.bin> <script.u32> [--no-mtp] " ++
    "[--tp W --rank R --master H --port P]\n";

/// The script's records: 1 pos n tokens... expected; 2 k; 3 npairs pairs... depth.
pub fn replay(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 3) return error.MissingArgument;
    var use_head = true;
    var g: group.Group = .{};
    var at: usize = 3;
    while (at < args.len) : (at += 1) {
        if (std.mem.eql(
            u8,
            args[at],
            "--no-mtp",
        )) use_head = false else if (!try g.option(args, &at)) return error.UnknownOption;
    }
    // the head is rank 0's alone: under tensor parallelism the replay runs without it on every rank
    if (g.world > 1 and use_head) return error.HeadNotShared;
    const prompt = try gen.readIds(gpa, io, args[1]);
    defer gpa.free(prompt);
    const script = try gen.readIds(gpa, io, args[2]);
    defer gpa.free(script);
    var joined = try g.join(gpa, io, args[0]);
    defer joined.close();
    const e = try g.engine(gpa, io, args[0], joined, .{ .capacity = 4096, .streams = 1, .mtp = use_head });
    defer e.deinit();
    const m = e.model_();
    const d = m.dims;
    const v = d.vocab;
    const width = d.hidden * e.act.size();
    const max_window = nem.lanes.max_window;
    const state_bytes = nem.pages.Stream.stateBytes(d, e.act, e.cfg.count(.mamba));
    var s = try e.openStream();
    defer e.closeStream(&s);
    var snaps = try hip.DeviceBuffer.alloc(&e.driver, (max_window + 1) * state_bytes);
    defer snaps.free();
    var hid = try hip.DeviceBuffer.alloc(&e.driver, max_window * width);
    defer hid.free();
    const head = if (use_head) try nem.head.Head.init(gpa, e) else null;
    defer if (head) |h| h.deinit();
    const row = try gpa.alloc(f32, max_window * v);
    defer gpa.free(row);
    // the prompt as the lane runs it: one chunk, the head absorbing each row with the next token
    try s.cover(&e.pool, prompt.len);
    try e.fresh();
    var o = e.ops(&e.arena);
    const dev = try e.ids(o, prompt);
    var hidden = try nem.forward.run(
        o,
        m,
        e.place[0..e.cfg.layers],
        .{ .pool = &e.pool, .stream = &s },
        dev,
        prompt.len,
        0,
    );
    s.len = prompt.len;
    if (head) |h| _ = try h.absorb(o, &s, hidden, prompt[1..]);
    try hip.raw.copy(hid, 0, hidden.ptr + (prompt.len - 1) * width, width, o.stream);
    try gen.readBack(e, gpa, try e.logits(o, nem.forward.at(hidden, (prompt.len - 1) * d.hidden), 1), row[0..v]);
    std.debug.print(
        "replay: prompt {d} tokens, first token {d}, head {s}\n",
        .{ prompt.len, gen.argmax(row[0..v]), if (head != null) "on" else "off" },
    );
    // the reference stream: after each keep it takes the kept rows one at a time, then the two states are compared
    var r = try e.openStream();
    defer e.closeStream(&r);
    try r.cover(&e.pool, prompt.len);
    try e.fresh();
    var ro = e.ops(&e.arena);
    const rdev = try e.ids(ro, prompt);
    _ = try nem.forward.run(ro, m, e.place[0..e.cfg.layers], .{ .pool = &e.pool, .stream = &r }, rdev, prompt.len, 0);
    r.len = prompt.len;
    const reference = try gpa.alloc(f32, max_window * v);
    defer gpa.free(reference);
    const mine = try gpa.alloc(u8, state_bytes);
    defer gpa.free(mine);
    const theirs = try gpa.alloc(u8, state_bytes);
    defer gpa.free(theirs);
    const layer_bytes = nem.mamba.convBytes(d, e.act) + nem.mamba.ssmBytes(d);
    // the residual's row 0 after every block of a window and of the reference step, for the first differing layer
    var probe: Probe = .{
        .e = e,
        .rows = try gpa.alloc(f32, e.cfg.layers * d.hidden),
        .other = try gpa.alloc(f32, e.cfg.layers * d.hidden),
    };
    defer gpa.free(probe.rows);
    defer gpa.free(probe.other);
    var i: usize = 0;
    var pending: usize = 0;
    var windows: usize = 0;
    var bad: usize = 0;
    var first_bits: ?usize = null;
    var held: [nem.head.max_depth]u32 = undefined;
    var last_rows: [nem.lanes.max_window]u32 = undefined;
    var last_pos: usize = 0;
    var last_n: usize = 0;
    while (i < script.len) {
        const kind = script[i];
        i += 1;
        switch (kind) {
            1 => {
                const pos = script[i];
                const n = script[i + 1];
                const tokens = script[i + 2 ..][0..n];
                const expected = script[i + 2 + n];
                i += 3 + n;
                if (pos != s.len) {
                    std.debug.print("replay: window at {d}, the stream holds {d}\n", .{ pos, s.len });
                    return error.PositionMismatch;
                }
                try s.cover(&e.pool, s.len + n);
                probe.into = probe.rows;
                hidden = try e.round(
                    &.{.{ .s = &s, .tokens = tokens, .snaps = .{ .buf = snaps, .stride = state_bytes } }},
                    probe.trace(),
                );
                o = e.ops(&e.arena);
                try hip.raw.copy(hid, 0, hidden.ptr, n * width, o.stream);
                try gen.readBack(e, gpa, try e.roundLogits(o, hidden, n), row[0 .. n * v]);
                const got = gen.argmax(row[0..v]);
                windows += 1;
                pending = n;
                last_pos = pos;
                last_n = n;
                @memcpy(last_rows[0..n], tokens);
                if (got != expected) {
                    bad += 1;
                    if (bad == 1) std.debug.print(
                        "replay: FIRST MISMATCH at {d}: window of {d} rows {any} draws {d}, the serial run drew {d}\n",
                        .{ pos, n, tokens, got, expected },
                    );
                }
            },
            2 => {
                const k = script[i];
                i += 1;
                if (k < pending) try hip.raw.copy(
                    s.mem,
                    0,
                    snaps.base() + k * state_bytes,
                    state_bytes,
                    e.stream.handle,
                );
                s.len += k;
                pending = 0;
                // the reference takes the kept rows one at a time; its logits and its state must match
                for (0..k) |j| {
                    try r.cover(&e.pool, r.len + 1);
                    probe.into = if (j == 0) probe.other else null;
                    const rh = try e.round(&.{.{ .s = &r, .tokens = last_rows[j..][0..1] }}, probe.trace());
                    ro = e.ops(&e.arena);
                    r.len += 1;
                    try gen.readBack(e, gpa, try e.roundLogits(ro, rh, 1), reference[0..v]);
                    if (first_bits == null and !std.mem.eql(f32, row[j * v ..][0..v], reference[0..v])) {
                        first_bits = last_pos + j;
                        var worst: f32 = 0;
                        for (row[j * v ..][0..v], reference[0..v]) |x, y| worst = @max(worst, @abs(x - y));
                        std.debug.print(
                            "replay: first logits difference at {d} (row {d} of the window of {d} at {d}, " ++
                                "tokens {any}), largest {e}\n",
                            .{ last_pos + j, j, last_n, last_pos, last_rows[0..last_n], worst },
                        );
                        if (j == 0) probe.report(&e.cfg, d.hidden);
                    }
                }
                try e.stream.synchronize();
                if (try kvDiffers(e, gpa, &s, &r, s.len)) |where| {
                    std.debug.print(
                        "replay: K/V differ after the keep of {d} of {d} at {d} (now {d}): attention slot " ++
                            "{d}, position {d}, {s}\n",
                        .{ k, last_n, last_pos, s.len, where.slot, where.pos, if (where.key) "keys" else "values" },
                    );
                    return error.KvMismatch;
                }
                try s.mem.download(0, mine);
                try r.mem.download(0, theirs);
                if (!std.mem.eql(u8, mine, theirs)) {
                    var layers_off: usize = 0;
                    var first_layer: ?usize = null;
                    var part: []const u8 = "";
                    for (0..e.cfg.count(.mamba)) |l| {
                        const a = mine[l * layer_bytes ..][0..layer_bytes];
                        const b = theirs[l * layer_bytes ..][0..layer_bytes];
                        if (!std.mem.eql(u8, a, b)) {
                            layers_off += 1;
                            if (first_layer == null) {
                                first_layer = l;
                                const cb = nem.mamba.convBytes(d, e.act);
                                part = if (!std.mem.eql(
                                    u8,
                                    a[0..cb],
                                    b[0..cb],
                                )) (if (std.mem.eql(u8, a[cb..], b[cb..])) "conv" else "conv and ssm") else "ssm";
                            }
                        }
                    }
                    std.debug.print(
                        "replay: Mamba state differs after the keep of {d} of {d} at {d} (now {d}): {d} " ++
                            "layers, first Mamba layer {?d} ({s})\n",
                        .{ k, last_n, last_pos, s.len, layers_off, first_layer, part },
                    );
                    return error.StateMismatch;
                }
            },
            3 => {
                const np = script[i];
                const pairs = script[i + 1 ..][0..np];
                const depth = script[i + 1 + np];
                i += 2 + np;
                if (head) |h| {
                    try s.cover(&e.pool, s.len + nem.head.max_depth);
                    try e.fresh();
                    o = e.ops(&e.arena);
                    var jobs = [_]nem.head.Job{.{
                        .s = &s,
                        .hidden = hid.base(),
                        .next = pairs,
                        .out = held[0..@min(depth, nem.head.max_depth)],
                    }};
                    try h.chains(o, &jobs);
                }
            },
            else => return error.BadScript,
        }
    }
    std.debug.print(
        "replay: {d} windows, {d} with a first draw unlike the serial run; first logits difference {?d}; " ++
            "states equal throughout\n",
        .{ windows, bad, first_bits },
    );
    if (bad > 0) return error.ReplayMismatch;
}

/// Row 0 of the residual after each block, read back as fp32.
const Probe = struct {
    e: *nem.engine.Engine,
    rows: []f32,
    other: []f32,
    into: ?[]f32 = null,

    fn trace(p: *Probe) nem.forward.Trace {
        return .{ .ctx = p, .layer = layerFn };
    }

    fn layerFn(ctx: *anyopaque, layer: usize, x: Tensor, rows: usize) anyerror!void {
        _ = rows;
        const p: *Probe = @ptrCast(@alignCast(ctx));
        const into = p.into orelse return;
        const width = p.e.model_().dims.hidden;
        const gpa = std.heap.page_allocator;
        try gen.readBack(p.e, gpa, x, into[layer * width ..][0..width]);
    }

    /// The first block after which the window's row 0 and the reference step's differ, and its kind.
    fn report(p: *Probe, c: *const nem.Config, width: usize) void {
        for (0..c.layers) |l| {
            const a = p.rows[l * width ..][0..width];
            const b = p.other[l * width ..][0..width];
            if (!std.mem.eql(f32, a, b)) {
                var worst: f32 = 0;
                for (a, b) |x, y| worst = @max(worst, @abs(x - y));
                std.debug.print(
                    "replay: the residual first differs after block {d} ({s}), largest {e}\n",
                    .{ l, @tagName(c.kinds[l]), worst },
                );
                return;
            }
        }
        std.debug.print(
            "replay: every block's residual row 0 matches; the difference is in the final norm or the head\n",
            .{},
        );
    }
};

const KvWhere = struct { slot: usize, pos: usize, key: bool };

/// The first attention slot and position below `len` whose keys or values differ between the two streams.
fn kvDiffers(
    e: *nem.engine.Engine,
    gpa: std.mem.Allocator,
    a: *nem.pages.Stream,
    b: *nem.pages.Stream,
    len: usize,
) !?KvWhere {
    const pool = &e.pool;
    const d = e.model_().dims;
    const row = d.head_dim * e.act.size();
    const page_bytes = nem.pages.tokens * row;
    const x = try gpa.alloc(u8, row);
    defer gpa.free(x);
    const y = try gpa.alloc(u8, row);
    defer gpa.free(y);
    const slots = e.cfg.count(.attention);
    for (0..slots) |slot| {
        for (0..len) |pos| {
            const pa = a.ids.items[pos / nem.pages.tokens];
            const pb = b.ids.items[pos / nem.pages.tokens];
            for (0..pool.kv_heads) |h| {
                for ([_]bool{ true, false }) |key| {
                    const bufs = if (key) pool.keys else pool.values;
                    const oa = (h * pool.count + pa) * page_bytes + (pos % nem.pages.tokens) * row;
                    const ob = (h * pool.count + pb) * page_bytes + (pos % nem.pages.tokens) * row;
                    try bufs[slot].download(oa, x);
                    try bufs[slot].download(ob, y);
                    if (!std.mem.eql(u8, x, y)) return .{ .slot = slot, .pos = pos, .key = key };
                }
            }
        }
    }
    return null;
}
