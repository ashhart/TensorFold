//! GLM-5.3-Flash's dev-time checks on a loaded engine: captures, traces, teacher forcing and knock-out profiles.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const mtp = @import("mtp.zig");
const prompt_mod = @import("prompt.zig");
const tape_mod = @import("prompt_tape.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const Engine = @import("engine.zig").Engine;

/// The first window of `prompt`: each sublayer's input and output, then the logits, raw bf16 in glm_ref.py's order.
pub fn capture(e: *Engine, prompt: []const u32, path: []const u8) !void {
    const c = &e.c;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const n: u32 = @intCast(@min(prompt.len, st.max_rows));
    e.sync();
    e.s.reset();
    @memcpy(Engine.u32s(e.prompt_ids, n), prompt[0..n]);
    const plane = @as(usize, n) * c.hidden * 2;
    const bytes = plane * (2 + 4 * @as(usize, c.run)) + @as(usize, n) * c.vocab * 2;
    const dump = try e.arena.buffer(bytes);
    var x = e.ctx();
    x.dump = dump;
    const b = e.begin();
    fwd.backbone(&x, b.enc, e.prompt_ids, n, 0);
    fwd.head(&x, b.enc, e.sc.hidden, dump.at(x.dump_at), e.sc.picks, n);
    try e.finish(b.cb, b.enc);
    fwd.flipKda(&x);
    const file = platform.c.fopen(try std.fmt.allocPrintSentinel(e.gpa, "{s}", .{path}, 0), "wb") orelse return error.OpenFailed;
    defer _ = platform.c.fclose(file);
    if (platform.c.fwrite(dump.addr(), 1, bytes, file) != bytes) return error.WriteFailed;
    std.debug.print("captured {d} rows ({d} bytes) in {s}\n", .{ n, bytes, path });
}

/// Each call of a plain reply (prompt windows, then `steps` rows of `reply`) by sublayer: twin of glm_ref.py --trace.
pub fn trace(e: *Engine, prompt: []const u32, reply: []const u32, steps: usize, path: []const u8) !void {
    const c = &e.c;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const P: u32 = @intCast(prompt.len);
    const total: u32 = P + @as(u32, @intCast(@min(steps, reply.len)));
    if (total + 1 > e.s.cap) return error.ContextFull;
    e.sync();
    e.s.reset();
    @memcpy(Engine.u32s(e.prompt_ids, P), prompt);
    @memcpy(Engine.u32s(e.prompt_ids.at(@as(usize, P) * 4), total - P), reply[0 .. total - P]);
    const most = @as(usize, st.max_rows) * c.hidden * 2 * (2 + 4 * @as(usize, c.run)) + @as(usize, st.max_rows) * c.vocab * 2;
    const dump = try e.arena.buffer(most);
    const file = platform.c.fopen(try std.fmt.allocPrintSentinel(e.gpa, "{s}", .{path}, 0), "wb") orelse return error.OpenFailed;
    defer _ = platform.c.fclose(file);
    var x = e.ctx();
    var at: u32 = 0;
    var calls: usize = 0;
    while (at < total) : (calls += 1) {
        const n: u32 = if (at < P) @min(st.max_rows, P - at) else 1;
        x.dump = dump;
        x.dump_at = 0;
        const b = e.begin();
        fwd.backbone(&x, b.enc, e.prompt_ids.at(@as(usize, at) * 4), n, at);
        fwd.head(&x, b.enc, e.sc.hidden, dump.at(x.dump_at), e.sc.picks, n);
        try e.finish(b.cb, b.enc);
        fwd.flipKda(&x);
        const bytes = x.dump_at + @as(usize, n) * c.vocab * 2;
        if (platform.c.fwrite(dump.addr(), 1, bytes, file) != bytes) return error.WriteFailed;
        at += n;
    }
    std.debug.print("traced {d} calls ({d} prompt rows, {d} steps) in {s}\n", .{ calls, P, total - P, path });
}

/// Teacher-forced agreement: greedy picks with `reply`'s tokens fed back; how many equal them, the first that differs.
pub fn forced(e: *Engine, prompt: []const u32, reply: []const u32) !struct { same: usize, first: ?usize } {
    const c = &e.c;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const P: u32 = @intCast(prompt.len);
    if (reply.len == 0) return .{ .same = 0, .first = null };
    const total: u32 = P + @as(u32, @intCast(reply.len - 1));
    if (total + 1 > e.s.cap) return error.ContextFull;
    e.sync();
    e.s.reset();
    @memcpy(Engine.u32s(e.prompt_ids, P), prompt);
    @memcpy(Engine.u32s(e.prompt_ids.at(@as(usize, P) * 4), total - P), reply[0 .. total - P]);
    var x = e.ctx();
    var at: u32 = 0;
    var same: usize = 0;
    var first: ?usize = null;
    while (at < total) {
        const n: u32 = if (at < P) @min(st.max_rows, P - at) else 1;
        const b = e.begin();
        fwd.backbone(&x, b.enc, e.prompt_ids.at(@as(usize, at) * 4), n, at);
        fwd.head(&x, b.enc, e.sc.hidden.at(@as(usize, n - 1) * c.hidden * 2), e.sc.logits, e.sc.picks, 1);
        try e.finish(b.cb, b.enc);
        fwd.flipKda(&x);
        at += n;
        if (at >= P) { // this call's last row predicts reply[at - P]
            const i = at - P;
            if (Engine.u32s(e.sc.picks, 1)[0] == reply[i]) same += 1 else if (first == null) first = i;
        }
    }
    return .{ .same = same, .first = first };
}

/// Knock-out profile: a `depth`-draft round replayed `reps` times, each launch class left out in turn; median GPU time.
pub fn profile(e: *Engine, depth: u32, reps: usize, only: bool, parts: bool) !void {
    const d: u32 = if (e.w.mtp == null) 0 else @min(depth, st.max_rows - 1);
    if (d > 0) return profileRows(e, d, d + 1, reps, only, parts);
    var it = std.mem.tokenizeScalar(u8, if (platform.c.getenv("GLM_PROFILE_ROWS")) |v| std.mem.span(v) else "1", ','); // copy windows' rows, without the MTP head
    while (it.next()) |v| try profileRows(e, 0, std.math.clamp(std.fmt.parseInt(u32, v, 10) catch 1, 1, st.max_rows), reps, only, parts);
}

fn profileRows(e: *Engine, d: u32, R: u32, reps: usize, only: bool, parts: bool) !void {
    const c = &e.c;
    const D = c.hidden;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (e.s.pos + R + 1 > e.s.cap) return error.ContextFull;
    e.sync();
    const ids = Engine.u32s(e.sc.ids, R);
    for (ids, 0..) |*t, i| t.* = @intCast(1000 + 37 * i);
    const n_class = if (parts) fwd.Part.names.len else fwd.Class.names.len;
    var masks_buf: [34]u32 = undefined;
    masks_buf[0] = 0;
    for (0..n_class) |i| masks_buf[i + 1] = @as(u32, 1) << @intCast(i);
    masks_buf[n_class + 1] = 0xffff_ffff; // every class left out: what remains
    masks_buf[n_class + 2] = 0;
    const masks = masks_buf[0 .. n_class + 3];
    const times = try e.gpa.alloc(f64, reps);
    defer e.gpa.free(times);
    var full: f64 = 0;
    const all: u32 = (@as(u32, 1) << @intCast(n_class)) - 1;
    for (masks, 0..) |mask, mi| {
        var x = e.ctx();
        const m = if (mask == 0xffff_ffff) all else if (only and mask != 0) all & ~mask else mask;
        if (parts) x.pskip = m else x.skip = m;
        for (0..reps + 1) |rep| {
            const b = e.begin();
            if (d > 0 and x.skip & fwd.Class.mtp == 0) {
                mtp.run(&x, b.enc, e.sc.hidden, e.sc.picks, R, e.s.mtp_pos, e.sc.ids.at(4));
                for (1..d) |j| mtp.chain(&x, b.enc, if (j == 1) e.sc.m_x.at(@as(usize, R - 1) * D * 2) else e.sc.m_x, e.sc.ids.at(j * 4), e.s.mtp_pos + R + @as(u32, @intCast(j)) - 1, e.sc.ids.at((j + 1) * 4));
            }
            fwd.backbone(&x, b.enc, e.sc.ids, R, e.s.pos);
            fwd.head(&x, b.enc, e.sc.hidden, e.sc.logits, e.sc.picks, R);
            try e.finish(b.cb, b.enc);
            if (rep > 0) times[rep - 1] = (e.gpu[1] - e.gpu[0]) * 1e3; // the first run warms the class's state
        }
        std.mem.sort(f64, times, {}, std.sort.asc(f64));
        const med = times[reps / 2];
        if (mi == 0) full = med;
        if (mi == 0) std.debug.print("profile {d} rows: the last MoE layer's unique experts {d}\n", .{ R, if (R == 1) 8 else Engine.u32s(e.sc.ucount, 1)[0] }); // one row: its picks (no groups)
        const name = if (mask == 0) (if (mi == 0) "full" else "full again") else if (mask == 0xffff_ffff) "none" else if (parts) fwd.Part.names[@ctz(mask)] else fwd.Class.names[@ctz(mask)];
        std.debug.print("profile {d} rows{s}: {s:<10} {d:7.3} ms (min {d:.3}, max {d:.3}){s}", .{ R, if (only) " only" else "", name, med, times[0], times[reps - 1], if (mask == 0 or only or mask == 0xffff_ffff) "\n" else "" });
        if (mask != 0 and !only and mask != 0xffff_ffff) std.debug.print("  class {d:6.3} ms {d:5.1}%\n", .{ full - med, 100 * (full - med) / full });
    }
}

/// The chunk matmul vs the decode row kernel on layer 0's real KDA in-projection input: the share that differs.
pub fn checkMatmul(e: *Engine, prompt: []const u32) !void {
    const pr = &(e.pr orelse return error.NoPromptPath);
    const rows: u32 = @intCast(@min(prompt.len, st.max_rows));
    const L = &e.w.layers[0];
    const a = switch (L.attn) {
        .kda => |*a| a,
        .mla => return error.NotKda,
    };
    if (!@import("wide.zig").native(a.in_proj)) return error.UnsupportedWidth; // the 4/64 product's comparison only
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    e.sync();
    e.s.reset();
    @memcpy(Engine.u32s(e.prompt_ids, rows), prompt[0..rows]);
    const N = a.in_proj.n;
    const out = try e.arena.buffer(@as(usize, rows) * N * 2);
    var x = e.ctx();
    const b = e.begin();
    fwd.embed(&x, b.enc, e.prompt_ids, rows, 0);
    fwd.boundary(&x, b.enc, rows, false, L.hc.?[0], L.in_norm);
    fwd.qmv(&x, b.enc, e.k.qmv_kda_in, e.sc.normed, a.in_proj, out, rows);
    affine_mm.rowSums(b.enc, e.k.mm_bf16, 64, e.sc.normed, pr.sums, rows, a.in_proj.k);
    affine_mm.denseBm(b.enc, e.k.mm_bf16, e.sc.normed, pr.sums, a.in_proj, pr.proj, rows, if (e.k.m3_fast) 32 else 64);
    try e.finish(b.cb, b.enc);
    const want: [*]const u16 = @ptrCast(@alignCast(out.addr()));
    const got: [*]const u16 = @ptrCast(@alignCast(pr.proj.addr()));
    var differ: usize = 0;
    var worst: u32 = 0;
    for (0..@as(usize, rows) * N) |i| if (want[i] != got[i]) {
        differ += 1;
        const d = @as(i32, @as(i16, @bitCast(want[i]))) - @as(i32, @as(i16, @bitCast(got[i])));
        worst = @max(worst, @abs(d));
    };
    std.debug.print("matmul check (layer 0 KDA in-projection, {d} rows x {d}): {d} of {d} bf16 outputs differ ({d:.3}%), at most {d} steps\n", .{ rows, N, differ, @as(usize, rows) * N, 100 * @as(f64, @floatFromInt(differ)) / @as(f64, @floatFromInt(@as(usize, rows) * N)), worst });
}

/// Prompt chunk GPU time, median of `reps`: whole, then each class left out (alone with `only`); GLM_PROFILE_AT=N.
pub fn profilePrompt(e: *Engine, prompt: []const u32, reps: usize, only: bool, parts: bool) !void {
    const pr = &(e.pr orelse return error.NoPromptPath);
    const C = fwd.Class;
    // GLM_PROFILE_AT=N: the chunk at position N, the prompt's rows before it run first (the caches a long prompt has)
    const at: u32 = if (platform.c.getenv("GLM_PROFILE_AT")) |v| @intCast(@min(try std.fmt.parseInt(usize, std.mem.span(v), 10), prompt.len - 1)) else 0;
    const rows: u32 = @intCast(@min(prompt.len - at, pr.rows));
    if (at + rows + 1 > e.s.cap) return error.ContextFull;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    e.sync();
    @memcpy(Engine.u32s(e.prompt_ids, at + rows), prompt[0 .. at + rows]);
    if (at > 0) e.s.reset();
    var done: u32 = 0;
    while (done < at) { // the rows before the chunk, in chunks, as a prompt pass runs them
        const n = @min(pr.rows, at - done);
        var x = e.ctx();
        x.sc = &pr.streams;
        const b = e.begin();
        prompt_mod.backbone(pr, &x, b.enc, e.prompt_ids.at(@as(usize, done) * 4), n, done);
        try e.finish(b.cb, b.enc);
        fwd.flipKda(&x);
        done += n;
    }
    const times = try e.gpa.alloc(f64, reps);
    defer e.gpa.free(times);
    const run: Run = .{ .e = e, .pr = pr, .rows = rows, .at = at, .times = times };
    if (parts and platform.c.getenv("GLM_PROFILE_TAPE") != null) return profileTaped(run); // routes and lists played
    const P = fwd.Part;
    const classes = if (parts) [_]u32{ P.bit("mla_proj"), P.bit("mla_cache"), P.bit("mla_absorb"), P.bit("mla_select"), P.bit("mla_attn"), P.bit("mla_unabs"), P.bit("mla_out"), P.bit("kda_in"), P.bit("kda_pre"), P.bit("kda_scan"), P.bit("kda_post"), P.bit("kda_out"), P.bit("hc_mix"), P.bit("r_router"), P.bit("r_topk"), P.bit("e_gateup"), P.bit("e_down"), P.bit("s_gateup"), P.bit("s_down"), P.bit("combine") } else [_]u32{ C.hc, C.kda, C.mla, C.dense, C.route, C.routed, C.shared, C.exchange, C.combine, C.ends, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    var all: u32 = 0;
    for (classes) |m| all |= m;
    var full: f64 = 0;
    for (0..classes.len + 3) |mi| {
        const mask: u32 = if (mi == 0 or mi == classes.len + 2) 0 else if (mi == classes.len + 1) all else classes[mi - 1];
        if (mask == 0 and mi > 0 and mi <= classes.len) continue;
        var x = e.ctx();
        x.sc = &pr.streams;
        const m = if (mask == all) all else if (only and mask != 0) all & ~mask else mask;
        if (parts) x.pskip = m else x.skip = m;
        const med = try run.median(&x);
        if (mi == 0) full = med;
        const name = if (mask == 0) (if (mi == 0) "full" else "full again") else if (mask == all) "none" else if (parts) fwd.Part.names[@ctz(mask)] else fwd.Class.names[@ctz(mask)];
        std.debug.print("prompt profile {d} rows at {d}{s}: {s:<10} {d:9.2} ms (min {d:.2}, max {d:.2}){s}", .{ rows, at, if (only) " only" else "", name, med, times[0], times[reps - 1], if (mask == 0 or only or mask == all) "\n" else "" });
        if (mask != 0 and !only and mask != all) std.debug.print("  class {d:8.2} ms {d:5.1}%\n", .{ full - med, 100 * (full - med) / full });
    }
}

/// One profiled chunk: the median GPU time of a pass over it.
const Run = struct {
    e: *Engine,
    pr: *const prompt_mod.Prompt,
    rows: u32,
    at: u32,
    times: []f64,

    fn median(r: Run, x: *fwd.Ctx) !f64 {
        for (0..r.times.len + 1) |rep| {
            if (r.at == 0) r.e.s.reset(); // later chunks keep the caches (each rep reads and writes the same rows)
            const b = r.e.begin();
            prompt_mod.backbone(r.pr, x, b.enc, r.e.prompt_ids.at(@as(usize, r.at) * 4), r.rows, r.at);
            try r.e.finish(b.cb, b.enc);
            if (rep > 0) r.times[rep - 1] = (r.e.gpu[1] - r.e.gpu[0]) * 1e3;
        }
        std.mem.sort(f64, r.times, {}, std.sort.asc(f64));
        return r.times[r.times.len / 2];
    }
};

/// GLM_PROFILE_TAPE: parts against a pass whose routes and key lists play back from a recorded whole pass.
fn profileTaped(r: Run) !void {
    const e = r.e;
    const c = &e.c;
    var moes: u32 = 0;
    for (e.w.layers[0..c.run]) |*l| moes += @intFromBool(l.mlp == .moe);
    var t = try tape_mod.Tape.init(&e.arena, moes, c.countKind(.mla), r.rows, c.topk, c.experts, c.keyWidth());
    const pr: *prompt_mod.Prompt = @constCast(r.pr);
    pr.tape = &t;
    defer pr.tape = null;
    const P = fwd.Part;
    const parts = [_]u32{ P.bit("mla_proj"), P.bit("mla_cache"), P.bit("mla_absorb"), P.bit("mla_select"), P.bit("mla_attn"), P.bit("mla_unabs"), P.bit("mla_out"), P.bit("kda_in"), P.bit("kda_pre"), P.bit("kda_scan"), P.bit("kda_post"), P.bit("kda_out"), P.bit("hc_mix"), P.bit("r_router"), P.bit("r_topk"), P.bit("e_gateup"), P.bit("e_down"), P.bit("s_gateup"), P.bit("s_down"), P.bit("combine") };
    var all: u32 = 0;
    for (parts) |m| all |= m;
    const M = tape_mod.Mode;
    const Step = struct { name: []const u8, skip: u32 = 0, route: M = .play, keys: M = .play };
    var steps: [parts.len + 6]Step = undefined;
    steps[0] = .{ .name = "full", .route = .off, .keys = .off };
    steps[1] = .{ .name = "taped" };
    steps[2] = .{ .name = "route", .route = .off };
    steps[3] = .{ .name = "select", .keys = .off };
    for (parts, 0..) |m, i| steps[4 + i] = .{ .name = P.names[@ctz(m)], .skip = m };
    steps[parts.len + 4] = .{ .name = "none", .skip = all };
    steps[parts.len + 5] = .{ .name = "full again", .route = .off, .keys = .off };
    { // the tape: one whole pass recorded
        var x = e.ctx();
        x.sc = &pr.streams;
        t.route_mode = .record;
        t.keys_mode = .record;
        if (r.at == 0) e.s.reset();
        const b = e.begin();
        prompt_mod.backbone(pr, &x, b.enc, e.prompt_ids.at(@as(usize, r.at) * 4), r.rows, r.at);
        try e.finish(b.cb, b.enc);
    }
    var base: f64 = 0;
    var full: f64 = 0;
    for (steps, 0..) |st_, i| {
        var x = e.ctx();
        x.sc = &pr.streams;
        x.pskip = st_.skip;
        t.route_mode = st_.route;
        t.keys_mode = st_.keys;
        const med = try r.median(&x);
        if (i == 0) full = med;
        if (i == 1) base = med;
        std.debug.print("prompt profile {d} rows at {d} taped: {s:<10} {d:9.2} ms (min {d:.2}, max {d:.2})", .{ r.rows, r.at, st_.name, med, r.times[0], r.times[r.times.len - 1] });
        const d = if (i == 2 or i == 3) med - base else base - med; // live parts add, knock-outs take away
        if (i >= 2 and i < steps.len - 2) std.debug.print("  part {d:8.2} ms {d:5.1}%", .{ d, 100 * d / full });
        std.debug.print("\n", .{});
    }
}

/// Per-token loss via the prompt path in `rows`-row chunks: the head over every row, -log p(ids[i + 1]) into out[i].
pub fn promptNll(e: *Engine, ids: []const u32, rows: u32, out: []f32) !void {
    const c = &e.c;
    const V: usize = c.vocab;
    const pr = &(e.pr orelse return error.NoPromptChunks);
    const P: u32 = @intCast(ids.len);
    if (P < 2 or out.len < P - 1) return error.BadArguments;
    if (P + 1 > e.s.cap) return error.ContextFull;
    const R: u32 = @min(rows, e.chunk_rows);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const logits = try e.device.buffer(@as(usize, R) * V * 2, mtl.ResourceOptions.shared);
    defer logits.deinit();
    e.sync();
    e.s.reset();
    @memcpy(Engine.u32s(e.prompt_ids, P), ids);
    var x = e.ctx();
    var at: u32 = 0;
    while (at < P - 1) {
        const n: u32 = @min(R, P - at);
        const b = e.begin();
        var px = x;
        px.sc = &pr.streams;
        prompt_mod.backbone(pr, &px, b.enc, e.prompt_ids.at(@as(usize, at) * 4), n, at);
        fwd.flipKda(&x);
        fwd.qmv(&x, b.enc, x.k.qmv_head, pr.streams.hidden, x.w.head, .{ .buf = logits }, n);
        try e.finish(b.cb, b.enc);
        const l = logits.slice(u16, @as(usize, n) * V);
        for (0..n) |r| {
            const i = at + @as(u32, @intCast(r)); // row i predicts ids[i + 1]
            if (i + 1 >= P) break;
            const row = l[r * V ..][0..V];
            var m: f32 = -std.math.inf(f32);
            for (row) |v| m = @max(m, bf16f(v));
            var s: f64 = 0;
            for (row) |v| s += @exp(@as(f64, bf16f(v) - m));
            out[i] = @floatCast(@as(f64, m) + @log(s) - @as(f64, bf16f(row[ids[i + 1]])));
        }
        at += n;
    }
}

fn bf16f(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}
