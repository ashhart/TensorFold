//! Dense MTP CUDA regressions and prompt tap-history equivalence.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const E = q.engine;
const solo = q.solo;
const tap_window = q.cache.tap_window;

/// Validate dense MTP drafting and retained prompt tap history on CUDA.
pub fn run(gpa: std.mem.Allocator, io: std.Io, e: *E.Engine, dir: []const u8, ids: []const u32) !u8 {
    if (ids.len == 0) return error.EmptyMtpTestPrompt;
    const tap_rows = e.tapRows();
    const prompt_rows = solo.passRows(e);
    // Always cross the drafter's retained tap boundary, even when the supplied seed prompt is short.
    const test_ids = try gpa.alloc(u32, @max(ids.len, tap_rows + 1));
    defer gpa.free(test_ids);
    for (test_ids, 0..) |*id, i| id.* = ids[i % ids.len];
    var mtp_ok = false;
    const prompt_len = 8192;
    const tap_from = prompt_len - tap_window - 1;
    const boundary_ids = try gpa.alloc(u32, prompt_len);
    defer gpa.free(boundary_ids);
    for (boundary_ids, 0..) |*id, i| id.* = ids[i % ids.len];
    var tap_ok = false;
    {
        std.debug.print("[mtp-cuda] initializing drafter\n", .{});
        const d = try q.mtp.Mtp.init(gpa, io, e, dir, 4096);
        defer d.deinit();
        std.debug.print("[mtp-cuda] drafter initialized\n", .{});
        try e.setTaps(d.taps());
        // The target cache keeps this tap-boundary prefill; the drafter uses its bounded ring.
        var cache = try E.Cache.init(gpa, e, test_ids.len);
        defer cache.deinit(gpa);
        const prior_mode = e.mtp_prompt.mode;
        const prior_from = e.mtp_prompt.from;
        e.mtp_prompt.mode = .normal;
        e.mtp_prompt.from = null;
        defer {
            e.mtp_prompt.mode = prior_mode;
            e.mtp_prompt.from = prior_from;
        }
        std.debug.print("[mtp-cuda] filling initial target prompt: {d} rows\n", .{test_ids.len});
        try fill(e, &cache, test_ids, test_ids.len);
        std.debug.print("[mtp-cuda] initial target prompt filled\n", .{});
        mtp_ok = true;
        if (!e.g.bf16) {
            // e.x holds only the latest chunk; the final prompt row is absorbed after the first target token.
            const last_start = (test_ids.len - 1) / prompt_rows * prompt_rows;
            const last_rows = test_ids.len - last_start;
            const comparable_rows = last_rows - 1;
            if (comparable_rows > 0) {
                const seg = cache.seg(e, last_start, 0, comparable_rows);
                std.debug.print("  tap check setup: n_taps={d} layer={d}/{d} mode={s} cache={d}/{d} tap_ptr=0x{x} seg.tap=0x{x} seg.taps={d} seg.len={d} rows={d}\n", .{
                    e.n_taps,
                    e.taps[0],
                    e.layers.len,
                    @tagName(e.mtp_prompt.mode),
                    cache.len,
                    cache.written,
                    cache.tap.?.ptr,
                    seg.tap,
                    seg.taps,
                    seg.len,
                    seg.rows,
                });
                std.debug.print("[mtp-cuda] checking normalized input: start={d} rows={d}\n", .{ last_start, comparable_rows });
                mtp_ok = try q.mtp_check.normalizedInputCheck(
                    gpa,
                    d,
                    cache.tap.?.ptr,
                    e.x,
                    last_start,
                    comparable_rows,
                    test_ids[last_start + 1 .. last_start + 1 + comparable_rows],
                );
                std.debug.print("[mtp-cuda] normalized input check complete\n", .{});
            }
        }
        std.debug.print("[mtp-cuda] checking proposal and tree equivalence\n", .{});
        mtp_ok = (try q.mtp_check.run(gpa, d, cache.tap.?.ptr, test_ids)) and mtp_ok;
        std.debug.print("[mtp-cuda] proposal and tree equivalence complete\n", .{});
        std.debug.print("[mtp-cuda] checking prompt tap suffix boundaries\n", .{});
        tap_ok = try tapBoundaryCheck(gpa, e, d, d.taps(), boundary_ids, tap_from);
        std.debug.print("[mtp-cuda] tap suffix boundary check complete\n", .{});
    }
    return if (mtp_ok and tap_ok) 0 else 1;
}

/// Run one text prompt with an ordinary MTP tap pass or the filtered prefix path, then return its retained ring suffix.
fn captureTapSuffix(gpa: std.mem.Allocator, e: *E.Engine, d: *q.mtp.Mtp, ids: []const u32, from: usize, history_rows: usize, filtered: bool) ![]u8 {
    if (ids.len <= from or from + history_rows >= ids.len or e.n_taps == 0) return error.BadTapSuffixCheck;
    const prior_mode = e.mtp_prompt.mode;
    const prior_from = e.mtp_prompt.from;
    defer {
        e.mtp_prompt.mode = prior_mode;
        e.mtp_prompt.from = prior_from;
    }
    var cache = try E.Cache.init(gpa, e, ids.len);
    defer cache.deinit(gpa);
    // Poison the filtered ring so an unwritten tap cannot accidentally pass as valid data.
    if (filtered) try cache.tap.?.fill8(0xa5, e.stream.handle);
    var at: usize = 0;
    const prompt_rows = solo.passRows(e);
    while (at < ids.len) {
        const rows = @min(prompt_rows, ids.len - at);
        const end = at + rows;
        e.mtp_prompt.mode = if (filtered and end <= from) .none else if (filtered) .mtp else .normal;
        e.mtp_prompt.from = if (filtered) from else null;
        std.debug.print("[mtp-cuda] capture filtered={} start={d} rows={d} mode={s} tap_from={d}\n", .{ filtered, at, rows, @tagName(e.mtp_prompt.mode), e.mtp_prompt.from orelse 0 });
        try e.forward(&.{.{ .cache = &cache, .rows = rows }}, ids[at..end], .last);
        std.debug.print("[mtp-cuda] capture complete end={d}\n", .{end});
        at = end;
    }

    if (!filtered) {
        const last_start = (ids.len - 1) / prompt_rows * prompt_rows;
        if (from < last_start) return error.BadTapSuffixNormalizedRange;
        const normalized = e.x + (from - last_start) * e.g.hidden * 2;
        const direct_equal = try q.mtp_check.normalizedInputCheck(gpa, d, cache.tap.?.ptr, normalized, from, history_rows, ids[from + 1 ..][0..history_rows]);
        if (!direct_equal) return error.MtpNormalizedShortHistoryMismatch;
    }

    const row_bytes = e.n_taps * e.g.hidden * 2;
    var ring = try gpa.alloc(u8, e.tapBytes());
    defer gpa.free(ring);
    try cache.tap.?.download(0, ring);
    const suffix_rows = ids.len - from;
    const suffix = try gpa.alloc(u8, suffix_rows * row_bytes);
    for (0..suffix_rows) |i| {
        const pos = from + i;
        const slot = pos % e.tapRows();
        @memcpy(suffix[i * row_bytes ..][0..row_bytes], ring[slot * row_bytes ..][0..row_bytes]);
    }
    return suffix;
}

/// Each short initial history must preserve the same target tap bytes and MTP state as its full-capture reference.
fn tapBoundaryCheck(gpa: std.mem.Allocator, e: *E.Engine, d: *q.mtp.Mtp, taps: []const u32, ids: []const u32, from: usize) !bool {
    if (from != 6143 or ids.len != 8192) return error.BadTapSuffixBoundary;
    try e.setTaps(taps);
    const cases = [_]struct { prompt_len: usize, history_rows: usize }{
        .{ .prompt_len = 2048, .history_rows = 256 },
        .{ .prompt_len = 8192, .history_rows = 256 },
        .{ .prompt_len = 8192, .history_rows = 512 },
        .{ .prompt_len = 8192, .history_rows = tap_window },
    };
    var all_equal = true;
    for (cases) |case| {
        const prompt = ids[0..case.prompt_len];
        const case_from = case.prompt_len - case.history_rows - 1;
        std.debug.print("[mtp-cuda] suffix case prompt={d} history={d} from={d}\n", .{ case.prompt_len, case.history_rows, case_from });
        const reference = try captureTapSuffix(gpa, e, d, prompt, case_from, case.history_rows, false);
        defer gpa.free(reference);
        const filtered = try captureTapSuffix(gpa, e, d, prompt, case_from, case.history_rows, true);
        defer gpa.free(filtered);
        const tap_equal = std.mem.eql(u8, reference, filtered);
        const reference_populated = containsNonZero(reference);
        const state_equal = if (tap_equal) try mtpSuffixStateCheck(gpa, d, e, prompt, case_from, case.history_rows, reference, filtered) else false;
        const equal = tap_equal and reference_populated and state_equal;
        all_equal = all_equal and equal;
        if (!equal) {
            const row_bytes = e.n_taps * e.g.hidden * 2;
            for (reference, filtered, 0..) |want, got, byte| if (want != got) {
                std.debug.print("  first tap mismatch at prompt position {d}, row byte {d}: {x:0>2} != {x:0>2}\n", .{ case_from + byte / row_bytes, byte % row_bytes, want, got });
                break;
            };
        }
        std.debug.print("{s} filtered MTP tap suffix [{d}, {d}) with {d} history rows matches full capture; populated={}; K/V, final state and proposals {s}\n", .{ if (equal) "PASS" else "FAIL", case_from, prompt.len, case.history_rows, reference_populated, if (state_equal) "match" else "differ" });
    }
    return all_equal;
}

fn containsNonZero(bytes: []const u8) bool {
    for (bytes) |byte| if (byte != 0) return true;
    return false;
}

/// A tap-ring equivalence must survive the next consumer: initial MTP K/V, state and candidates.
fn mtpSuffixStateCheck(gpa: std.mem.Allocator, d: *q.mtp.Mtp, e: *E.Engine, ids: []const u32, from: usize, rows: usize, reference: []const u8, filtered: []const u8) !bool {
    if (from + rows >= ids.len or reference.len != filtered.len) return error.BadTapSuffixStateCheck;
    const trees = d.trees;
    defer d.trees = trees;
    // The boundary equivalence checks the default chain proposal, which is stored in held_dev.
    d.trees = false;
    const row_bytes = e.n_taps * e.g.hidden * 2;
    const host_bytes = e.tapBytes();
    if (reference.len != (ids.len - from) * row_bytes or host_bytes != e.tapRows() * row_bytes) return error.BadTapSuffixStateCheck;
    const a_host = try gpa.alloc(u8, host_bytes);
    defer gpa.free(a_host);
    const b_host = try gpa.alloc(u8, host_bytes);
    defer gpa.free(b_host);
    @memset(a_host, 0xa5);
    @memset(b_host, 0xa5);
    for (0..ids.len - from) |i| {
        const slot = (from + i) % e.tapRows();
        @memcpy(a_host[slot * row_bytes ..][0..row_bytes], reference[i * row_bytes ..][0..row_bytes]);
        @memcpy(b_host[slot * row_bytes ..][0..row_bytes], filtered[i * row_bytes ..][0..row_bytes]);
    }
    var a_tap = try cuda.DeviceBuffer.alloc(e.d, host_bytes);
    defer a_tap.free();
    var b_tap = try cuda.DeviceBuffer.alloc(e.d, host_bytes);
    defer b_tap.free();
    try a_tap.upload(0, a_host);
    try b_tap.upload(0, b_host);
    const a_ctx = try d.create();
    defer d.drop(a_ctx);
    const b_ctx = try d.create();
    defer d.drop(b_ctx);
    const follow = ids[from + 1 ..][0..rows];
    try d.absorb(a_ctx, a_tap.ptr, from, rows, null, follow);
    try d.absorb(b_ctx, b_tap.ptr, from, rows, null, follow);
    try e.stream.synchronize();
    const equal = try sameMtpSuffixState(gpa, d, a_ctx, b_ctx);
    if (!equal) std.debug.print("  MTP suffix state mismatch: begin/end {d}/{d} vs {d}/{d}, candidates equal={}\n", .{ a_ctx.begin, a_ctx.end, b_ctx.begin, b_ctx.end, std.mem.eql(u32, &a_ctx.cands, &b_ctx.cands) });
    if (!equal or a_ctx.held_n == 0) return false;

    // Prove the device-resident proposal comparison catches an actual changed token.
    var planted: u32 = undefined;
    try b_ctx.held_dev.download(0, std.mem.asBytes(&planted));
    planted ^= 1;
    try b_ctx.held_dev.upload(0, std.mem.asBytes(&planted));
    const mismatch_detected = !(try sameMtpSuffixState(gpa, d, a_ctx, b_ctx));
    std.debug.print("{s} MTP suffix checker rejects a planted held-token mismatch\n", .{if (mismatch_detected) "PASS" else "FAIL"});
    return mismatch_detected;
}

fn sameMtpSuffixState(gpa: std.mem.Allocator, d: *q.mtp.Mtp, a_ctx: *q.mtp.Ctx, b_ctx: *q.mtp.Ctx) !bool {
    if (a_ctx.begin != b_ctx.begin or a_ctx.end != b_ctx.end or a_ctx.held_n != b_ctx.held_n or
        !std.mem.eql(u32, &a_ctx.cands, &b_ctx.cands)) return false;
    if (a_ctx.held_n > q.mtp.Mtp.max_chain_tokens) return error.TooManyDrafts;
    const held_bytes = a_ctx.held_n * @sizeOf(u32);
    if (held_bytes > 0) {
        const a_held = try gpa.alloc(u8, held_bytes);
        defer gpa.free(a_held);
        const b_held = try gpa.alloc(u8, held_bytes);
        defer gpa.free(b_held);
        try d.e.stream.synchronize();
        try a_ctx.held_dev.download(0, a_held);
        try b_ctx.held_dev.download(0, b_held);
        if (!std.mem.eql(u8, a_held, b_held)) return false;
    }
    const row_bytes = d.kvWidth() * 2;
    const ring_part = d.capacity * row_bytes;
    const a_k = try gpa.alloc(u8, ring_part);
    defer gpa.free(a_k);
    const b_k = try gpa.alloc(u8, ring_part);
    defer gpa.free(b_k);
    const a_v = try gpa.alloc(u8, ring_part);
    defer gpa.free(a_v);
    const b_v = try gpa.alloc(u8, ring_part);
    defer gpa.free(b_v);
    try a_ctx.ring.download(0, a_k);
    try a_ctx.ring.download(ring_part, a_v);
    try b_ctx.ring.download(0, b_k);
    try b_ctx.ring.download(ring_part, b_v);
    const live_equal = blk: {
        var pos = a_ctx.begin;
        while (pos < a_ctx.end) : (pos += 1) {
            const slot: usize = @intCast(pos % d.capacity);
            const at = slot * row_bytes;
            if (!std.mem.eql(u8, a_k[at..][0..row_bytes], b_k[at..][0..row_bytes]) or
                !std.mem.eql(u8, a_v[at..][0..row_bytes], b_v[at..][0..row_bytes])) break :blk false;
        }
        break :blk true;
    };
    const state_bytes = d.e.g.hidden * 2;
    const a_state = try gpa.alloc(u8, state_bytes);
    defer gpa.free(a_state);
    const b_state = try gpa.alloc(u8, state_bytes);
    defer gpa.free(b_state);
    try a_ctx.last.download(0, a_state);
    try b_ctx.last.download(0, b_state);
    return live_equal and std.mem.eql(u8, a_state, b_state);
}

fn fill(e: *E.Engine, cache: *E.Cache, ids: []const u32, n: usize) !void {
    var buf: [E.max_rows]u32 = undefined;
    const prompt_rows = solo.passRows(e);
    var at: usize = 0;
    while (at < n) {
        const rows = @min(prompt_rows, n - at);
        for (buf[0..rows], at..) |*token, i| token.* = ids[i % ids.len];
        try e.forward(&.{.{ .cache = cache, .rows = rows }}, buf[0..rows], .last);
        at += rows;
    }
}
