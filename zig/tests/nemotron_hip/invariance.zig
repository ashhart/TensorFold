//! Within-path HIP invariance checks: prompt cuts + 1-row head, decode window/grouping, and prompt-cache resume.
const std = @import("std");
const hip = @import("hip");
const nem = @import("nemotron");
const gen = @import("gen.zig");
const replay = @import("replay.zig");

pub const usage = "       tf-nemotron-hip-test invariance <model dir> [ids.bin] [--no-mtp]\n";

const Snap = struct {
    state: []u8,
    hidden: []u8,
    logits: []f32,

    fn alloc(gpa: std.mem.Allocator, state_bytes: usize, hidden_bytes: usize, logits_len: usize) !Snap {
        return .{
            .state = try gpa.alloc(u8, state_bytes),
            .hidden = try gpa.alloc(u8, hidden_bytes),
            .logits = try gpa.alloc(f32, logits_len),
        };
    }

    fn deinit(s: Snap, gpa: std.mem.Allocator) void {
        gpa.free(s.state);
        gpa.free(s.hidden);
        gpa.free(s.logits);
    }
};

fn buildPrompt(gpa: std.mem.Allocator, io: std.Io, path: ?[]const u8, vocab: usize) ![]u32 {
    if (path) |p| return gen.readIds(gpa, io, p);
    const ids = try gpa.alloc(u32, 193);
    const mod: usize = @min(vocab, 50000);
    for (ids, 0..) |*t, i| t.* = @intCast((1000 + i * 37) % mod);
    return ids;
}

fn checkKv(e: *nem.engine.Engine, gpa: std.mem.Allocator, a: *nem.pages.Stream, b: *nem.pages.Stream, len: usize, label: []const u8) !void {
    if (try replay.kvDiffers(e, gpa, a, b, len)) |w| {
        std.debug.print("invariance ({s}): K/V differ at slot {d} pos {d} ({s})\n", .{ label, w.slot, w.pos, if (w.key) "keys" else "values" });
        return error.KvMismatch;
    }
}

fn captureHidden(e: *nem.engine.Engine, t: hip.ops.Tensor, dst: []u8) !void {
    try e.stream.synchronize();
    try hip.runtime.check(e.driver.api.hipMemcpyDtoH(dst.ptr, @ptrFromInt(t.ptr), dst.len));
}

/// Synchronizes `e.stream` (hipStreamNonBlocking) before synchronous null-stream Mamba state downloads.
fn downloadState(e: *nem.engine.Engine, s: *nem.pages.Stream, dst: []u8) !void {
    try e.stream.synchronize();
    try s.mem.download(0, dst);
}

fn compareLogits(a: []const f32, b: []const f32, what: []const u8) !void {
    const ba = std.mem.sliceAsBytes(a);
    const bb = std.mem.sliceAsBytes(b);
    if (!std.mem.eql(u8, ba, bb)) {
        var worst: f32 = 0;
        for (a, b) |x, y| worst = @max(worst, @abs(x - y));
        std.debug.print("invariance ({s}): logits bits differ (max diff {e})\n", .{ what, worst });
        return error.LogitsMismatch;
    }
}

/// Checks prompt chunk-cut invariance (trunk + 1-row head) across trailing chunks {1, 8, 15, 16, 17} and multi-chunk.
fn checkPromptCuts(gpa: std.mem.Allocator, e: *nem.engine.Engine, prompt: []const u32) !void {
    const m = e.model_();
    const d = m.dims;
    const width = d.hidden * e.act.size();
    const state_bytes = nem.pages.Stream.stateBytes(d, e.act, e.cfg.count(.mamba));
    const saved_chunk = e.o.chunk;
    defer e.o.chunk = saved_chunk;

    var base_s = try e.openStream();
    defer e.closeStream(&base_s);
    e.o.chunk = prompt.len;
    const base_last = try e.prompt(&base_s, prompt);
    const ref = try Snap.alloc(gpa, state_bytes, prompt.len * width, d.vocab);
    defer ref.deinit(gpa);
    try downloadState(e, &base_s, ref.state);
    try captureHidden(e, base_last, ref.hidden);
    const o0 = e.ops(&e.arena);
    try gen.readBack(e, gpa, try e.logits(o0, nem.forward.at(base_last, (prompt.len - 1) * d.hidden), 1), ref.logits);

    const tails = [_]usize{ 1, 8, 15, 16, 17, 65 };
    const got = try Snap.alloc(gpa, state_bytes, prompt.len * width, d.vocab);
    defer got.deinit(gpa);
    for (tails) |tail| {
        if (tail >= prompt.len) continue;
        const cut = prompt.len - tail;
        e.o.chunk = cut;
        var s = try e.openStream();
        defer e.closeStream(&s);
        const last = try e.prompt(&s, prompt);
        const rem = prompt.len - ((prompt.len - 1) / cut) * cut;
        try downloadState(e, &s, got.state);
        if (!std.mem.eql(u8, ref.state, got.state)) {
            std.debug.print("invariance (prompt cut {d}, tail {d}): Mamba state differs\n", .{ cut, rem });
            return error.StateMismatch;
        }
        const chunk_bytes = rem * width;
        try captureHidden(e, last, got.hidden[0..chunk_bytes]);
        if (!std.mem.eql(u8, ref.hidden[(prompt.len - rem) * width ..][0..chunk_bytes], got.hidden[0..chunk_bytes])) {
            std.debug.print("invariance (prompt cut {d}, tail {d}): hidden chunk differs\n", .{ cut, rem });
            return error.HiddenMismatch;
        }
        try checkKv(e, gpa, &base_s, &s, prompt.len, "prompt cut");
        const o = e.ops(&e.arena);
        try gen.readBack(e, gpa, try e.logits(o, nem.forward.at(last, (rem - 1) * d.hidden), 1), got.logits);
        try compareLogits(ref.logits, got.logits, "prompt cut 1-row head");
    }
    std.debug.print("invariance: prompt cuts ({d} cuts) + 1-row head: 0 bits differ\n", .{tails.len});
}

/// Checks decode window width (1 vs 2/3/4) and multi-stream grouping invariance on identical input rows.
fn checkDecodeGrouping(gpa: std.mem.Allocator, e: *nem.engine.Engine, base_prompt: []const u32, tail: []const u32) !void {
    const m = e.model_();
    const d = m.dims;
    const v = d.vocab;
    const k = tail.len;
    const state_bytes = nem.pages.Stream.stateBytes(d, e.act, e.cfg.count(.mamba));

    var ref_s = try e.openStream();
    defer e.closeStream(&ref_s);
    _ = try e.prompt(&ref_s, base_prompt);
    const serial_logits = try gpa.alloc(f32, k * v);
    defer gpa.free(serial_logits);
    for (tail, 0..) |t, step| {
        try ref_s.cover(&e.pool, ref_s.len + 1);
        const h = try e.round(&.{.{ .s = &ref_s, .tokens = &.{t} }}, null);
        const o = e.ops(&e.arena);
        ref_s.len += 1;
        try gen.readBack(e, gpa, try e.roundLogits(o, h, 1), serial_logits[step * v ..][0..v]);
    }
    const ref_state = try gpa.alloc(u8, state_bytes);
    defer gpa.free(ref_state);
    try downloadState(e, &ref_s, ref_state);

    // 2-row and 4-row windows across the same 8 tokens, then a grouped 2-stream round
    const win_logits = try gpa.alloc(f32, k * v);
    defer gpa.free(win_logits);
    const win_state = try gpa.alloc(u8, state_bytes);
    defer gpa.free(win_state);
    for ([_]usize{ 2, 4 }) |w| {
        var s = try e.openStream();
        defer e.closeStream(&s);
        _ = try e.prompt(&s, base_prompt);
        var at: usize = 0;
        while (at < k) : (at += w) {
            try s.cover(&e.pool, s.len + w);
            const h = try e.round(&.{.{ .s = &s, .tokens = tail[at..][0..w] }}, null);
            const o = e.ops(&e.arena);
            s.len += w;
            try gen.readBack(e, gpa, try e.roundLogits(o, h, w), win_logits[at * v ..][0 .. w * v]);
        }
        try compareLogits(serial_logits, win_logits, "decode window");
        try downloadState(e, &s, win_state);
        if (!std.mem.eql(u8, ref_state, win_state)) return error.StateMismatch;
        try checkKv(e, gpa, &ref_s, &s, ref_s.len, "decode window");
    }

    // Grouped multi-stream round: stream 0 takes tail[0..4], stream 1 takes a shorter prompt + 3 rows alongside it
    var s0 = try e.openStream();
    defer e.closeStream(&s0);
    _ = try e.prompt(&s0, base_prompt);
    var s1 = try e.openStream();
    defer e.closeStream(&s1);
    _ = try e.prompt(&s1, base_prompt[0 .. base_prompt.len - 13]);
    try s0.cover(&e.pool, s0.len + 4);
    try s1.cover(&e.pool, s1.len + 3);
    const h_grp = try e.round(&.{
        .{ .s = &s0, .tokens = tail[0..4] },
        .{ .s = &s1, .tokens = base_prompt[base_prompt.len - 13 ..][0..3] },
    }, null);
    const grp_logits = try gpa.alloc(f32, 4 * v);
    defer gpa.free(grp_logits);
    const o_grp = e.ops(&e.arena);
    try gen.readBack(e, gpa, try e.roundLogits(o_grp, h_grp, 4), grp_logits);
    try compareLogits(serial_logits[0 .. 4 * v], grp_logits, "grouped 2-stream round");
    std.debug.print("invariance: decode windows (1 vs 2 vs 4) and 2-stream grouping: 0 bits differ\n", .{});
}

/// Checks served prompt-cache resume-equals-fresh at a prompt state (state, KV pages, logits and 4 decode steps).
fn checkResumePrefix(gpa: std.mem.Allocator, e: *nem.engine.Engine, prompt: []const u32, decode_tail: []const u32) !void {
    const m = e.model_();
    const d = m.dims;
    const width = d.hidden * e.act.size();
    const h = try nem.lanes.Hip.init(gpa, e);
    defer h.deinit();

    // Fresh reference lane without prefix cache
    const l_fresh = try h.make();
    defer h.free(l_fresh);
    try l_fresh.s.cover(&e.pool, prompt.len);
    const y_fresh_t = try h.chunk(l_fresh, prompt, 0, prompt.len, null);
    const ref = try Snap.alloc(gpa, h.state_bytes, width, d.vocab);
    defer ref.deinit(gpa);
    try downloadState(e, &l_fresh.s, ref.state);
    try captureHidden(e, .{ .ptr = l_fresh.hidden.base(), .kind = e.act }, ref.hidden);
    try gen.readBack(e, gpa, y_fresh_t.?, ref.logits);

    // Seed the prefix cache at mark 128 on lane 1, free it so only the cached prefix stays, then resume on lane 2
    h.prefix.keepPages(16, 4);
    {
        const l_seed = try h.make();
        defer h.free(l_seed);
        l_seed.key = prompt;
        var adopt_seed: std.ArrayList(u32) = .empty;
        defer adopt_seed.deinit(gpa);
        const plan_seed = try h.prefix.begin(prompt, null, 128, &.{}, l_seed, &adopt_seed);
        defer gpa.free(plan_seed.marks);
        if (plan_seed.marks.len == 0) return error.NoPrefixMark;
        var at: usize = 0;
        for (plan_seed.marks) |mark| {
            try l_seed.s.cover(&e.pool, mark);
            _ = try h.chunk(l_seed, prompt, at, mark, null);
            try h.keepCut(l_seed, prompt, mark);
            at = mark;
        }
        try l_seed.s.cover(&e.pool, prompt.len);
        _ = try h.chunk(l_seed, prompt, at, prompt.len, null);
    }

    const l_res = try h.make();
    defer h.free(l_res);
    var adopt_res: std.ArrayList(u32) = .empty;
    defer adopt_res.deinit(gpa);
    const plan_res = try h.prefix.begin(prompt, null, 0, &.{}, l_res, &adopt_res);
    defer gpa.free(plan_res.marks);
    if (plan_res.from == 0) return error.PrefixNotReused;
    if (adopt_res.items.len > 0) try l_res.s.adopt(&e.pool, adopt_res.items, false);
    l_res.s.resumeAt(plan_res.from);
    try l_res.s.cover(&e.pool, prompt.len);
    const y_res_t = try h.chunk(l_res, prompt, plan_res.from, prompt.len, null);

    const got = try Snap.alloc(gpa, h.state_bytes, width, d.vocab);
    defer got.deinit(gpa);
    try downloadState(e, &l_res.s, got.state);
    if (!std.mem.eql(u8, ref.state, got.state)) return error.StateMismatch;
    try captureHidden(e, .{ .ptr = l_res.hidden.base(), .kind = e.act }, got.hidden);
    if (!std.mem.eql(u8, ref.hidden, got.hidden)) return error.HiddenMismatch;
    try checkKv(e, gpa, &l_fresh.s, &l_res.s, prompt.len, "resume prompt");
    try gen.readBack(e, gpa, y_res_t.?, got.logits);
    try compareLogits(ref.logits, got.logits, "resume prompt head");

    // Step 4 decode tokens on both lanes through Hip.windows + keepRows
    try l_fresh.s.cover(&e.pool, prompt.len + decode_tail.len);
    try l_res.s.cover(&e.pool, prompt.len + decode_tail.len);
    const dec_fresh = try gpa.alloc(f32, decode_tail.len * d.vocab);
    defer gpa.free(dec_fresh);
    const dec_res = try gpa.alloc(f32, decode_tail.len * d.vocab);
    defer gpa.free(dec_res);
    try gen.readBack(e, gpa, try h.windows(&.{l_fresh}, &.{decode_tail}), dec_fresh);
    try h.keepRows(l_fresh, decode_tail.len);
    try gen.readBack(e, gpa, try h.windows(&.{l_res}, &.{decode_tail}), dec_res);
    try h.keepRows(l_res, decode_tail.len);
    try compareLogits(dec_fresh, dec_res, "resume decode");
    try checkKv(e, gpa, &l_fresh.s, &l_res.s, l_fresh.s.len, "resume decode");
    std.debug.print(
        "invariance: served prompt-cache resume at {d}/{d} (+{d} decode steps): 0 bits differ\n",
        .{ plan_res.from, prompt.len, decode_tail.len },
    );
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 1) return error.MissingArgument;
    const dir = args[0];
    var ids_path: ?[]const u8 = null;
    var mtp = true;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--no-mtp")) {
            mtp = false;
        } else if (ids_path == null) {
            ids_path = arg;
        } else return error.UnknownOption;
    }
    const e = try nem.engine.Engine.open(gpa, io, dir, .{ .capacity = 4096, .streams = 2, .extra_pages = 32, .mtp = mtp });
    defer e.deinit();
    const prompt = try buildPrompt(gpa, io, ids_path, e.model_().dims.vocab);
    defer gpa.free(prompt);
    if (prompt.len < 193) return error.PromptTooShort;
    try checkPromptCuts(gpa, e, prompt);
    const tail = [_]u32{ 1001, 2003, 3005, 4007, 5009, 6011, 7013, 8015 };
    try checkDecodeGrouping(gpa, e, prompt[0 .. prompt.len - 8], &tail);
    try checkResumePrefix(gpa, e, prompt, tail[0..4]);
    std.debug.print("invariance: PASS\n", .{});
}
