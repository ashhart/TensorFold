//! Real-checkpoint Mini serving: serial target references, cache restore, DSpark and exact transactions.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const api = @import("engine_api");
const native = @import("inkling2.zig");
const Model = @import("../families/inkling2/metal/model.zig").Model;
const backend = @import("../families/inkling2/metal/backend.zig");
const st = @import("../families/inkling2/metal/state.zig");
const dspark = @import("../families/inkling2/metal/dspark.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
/// A history the prompt cache can end a state on: the family's chunk grid.
const history = st.prompt_rows;
const context = 4 * history;
const made = 24;
const taps = [_]u32{ 5, 23, 41, 48 };

fn checkpoint() ![]const u8 {
    return @import("sys").getenv("TF_INKLING2_METAL_DIR") orelse error.SkipZigTest;
}

fn prompt(comptime n: usize, salt: u32) [n]u32 {
    var ids: [n]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(1000 + (i * 37 + @as(usize, salt) * 101) % 20000);
    return ids;
}

/// Repeating 16-token span ending mid-span so greedy continuation triggers SuffixLookup (copy_match = 8).
fn copyPrompt() [6 * 16 + 8]u32 {
    const span = prompt(16, 1);
    var ids: [6 * 16 + 8]u32 = undefined;
    for (0..6) |rep| @memcpy(ids[rep * 16 ..][0..16], &span);
    @memcpy(ids[6 * 16 ..], span[0..8]);
    return ids;
}

const TapSink = struct {
    bytes: [st.feature_row_bytes]u8 = @splat(0),
    seen: usize = 0,

    fn take(ptr: *anyopaque, s: *lanes.Stream, f: lanes.backend.Features, start: u64) !void {
        const self: *TapSink = @ptrCast(@alignCast(ptr));
        try std.testing.expectEqual(lanes.backend.Features.Space.device, f.space);
        try std.testing.expectEqual(lanes.backend.Features.Dtype.bf16, f.dtype);
        try std.testing.expectEqual(@as(u32, st.feature_row_bytes), f.row_bytes);
        try std.testing.expect(f.rows > 0);
        if (start + f.rows != s.prompt_len) return;
        const buf: mtl.Buffer = .{ .id = @ptrFromInt(f.buffer) };
        const bytes = buf.slice(u8, @intCast(f.offset + f.rows * f.row_bytes));
        const last = bytes[@intCast(f.offset + (f.rows - 1) * f.row_bytes)..][0..st.feature_row_bytes];
        @memcpy(&self.bytes, last);
        self.seen += 1;
    }
};

fn featureRow(b: lanes.backend.Backend, s: *lanes.Stream, start: u64) ![st.feature_row_bytes]u8 {
    const f = try b.features(s, &taps, start, 1);
    try std.testing.expectEqual(lanes.backend.Features.Space.device, f.space);
    try std.testing.expectEqual(lanes.backend.Features.Dtype.bf16, f.dtype);
    try std.testing.expectEqual(@as(u32, 1), f.rows);
    try std.testing.expectEqual(@as(u32, st.feature_row_bytes), f.row_bytes);
    const buf: mtl.Buffer = .{ .id = @ptrFromInt(f.buffer) };
    const bytes = buf.slice(u8, @intCast(f.offset + f.row_bytes));
    var row: [st.feature_row_bytes]u8 = undefined;
    @memcpy(&row, bytes[@intCast(f.offset)..][0..st.feature_row_bytes]);
    return row;
}

const Reference = struct {
    tokens: [made]u32,
    first: lanes.LogprobRow,
    tap: [st.feature_row_bytes]u8,
    full_tap: [st.feature_row_bytes]u8 = undefined,
    partial_tap: [st.feature_row_bytes]u8 = undefined,
    continuation_tap: [st.feature_row_bytes]u8 = undefined,
};

fn serial(b: lanes.backend.Backend, ids: []const u32) !Reference {
    var sink: TapSink = .{};
    var s = try lanes.Stream.init(gpa, .{ .id = "host-reference", .prompt = ids, .max_new = made, .drafts = false, .logprobs = 3 });
    defer s.deinit(gpa);
    try b.vtable.prefill_features.?(b.ptr, &s, &taps, .{ .ptr = &sink, .take = TapSink.take });
    defer b.release(&s);
    try std.testing.expectEqual(@as(usize, 1), sink.seen);
    var next = try b.read(try b.first(&s, ids.len));
    const first = try b.firstRow(&s);
    var out: Reference = .{ .tokens = undefined, .first = first, .tap = sink.bytes };
    out.tokens[0] = next;
    try std.testing.expectEqual(next, first.token);
    for (1..made) |i| {
        next = try b.read(try b.queue(&s, .{ .value = next }, ids.len + i));
        out.tokens[i] = next;
        switch (i) {
            8 => out.full_tap = try featureRow(b, &s, ids.len + 7),
            15 => out.partial_tap = try featureRow(b, &s, ids.len + 14),
            16 => out.continuation_tap = try featureRow(b, &s, ids.len + 15),
            else => {},
        }
    }
    return out;
}

fn withReference(dir: []const u8, ids: []const u32, extended: ?[]const u32) !struct { first: Reference, second: ?Reference } {
    const model = try Model.load(gpa, io, dir);
    defer model.deinit();
    const target = try backend.Metal.init(gpa, model, .{ .capacity = context, .streams = 1, .budget = 2 * st.cacheBytes(context) });
    defer target.deinit();
    try target.warm();
    const b = target.backend();
    try b.vtable.prepare_features.?(b.ptr, &taps);
    const first = try serial(b, ids);
    const repeated = try serial(b, ids);
    try std.testing.expectEqualSlices(u8, &first.tap, &repeated.tap);
    try std.testing.expectEqualSlices(u32, &first.tokens, &repeated.tokens);
    try expectFirstRow(first.first, repeated.first);
    const second = if (extended) |p| try serial(b, p) else null;
    return .{ .first = first, .second = second };
}

fn expectFirstRow(want: lanes.LogprobRow, got: lanes.LogprobRow) !void {
    try std.testing.expectEqual(want.token, got.token);
    try std.testing.expectEqual(@as(u32, @bitCast(want.logprob)), @as(u32, @bitCast(got.logprob)));
    try std.testing.expectEqual(want.count, got.count);
    try std.testing.expectEqualSlices(u32, want.ids[0..want.count], got.ids[0..got.count]);
    try std.testing.expectEqualSlices(f32, want.logprobs[0..want.count], got.logprobs[0..got.count]);
}

const Reply = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    rows: std.ArrayList(lanes.LogprobRow) = .empty,
    cached: ?u32 = null,
    drafted: u64 = 0,
    accepted: u64 = 0,
    done: ?api.Reason = null,

    fn event(ptr: *anyopaque, _: api.Id, item: *const api.Event) void {
        const self: *Reply = @ptrCast(@alignCast(ptr));
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        switch (item.*) {
            .tokens => |ids| self.tokens.appendSlice(gpa, ids) catch {},
            .logprobs => |rows| self.rows.appendSlice(gpa, rows) catch {},
            .prefilled => |n| self.cached = n,
            .finished => |f| {
                self.done = f.reason;
                self.drafted = f.stats.drafted;
                self.accepted = f.stats.accepted;
            },
            else => {},
        }
    }

    fn wait(self: *Reply) !api.Reason {
        for (0..600_000) |_| {
            self.mutex.lockUncancelable(io);
            const done = self.done;
            self.mutex.unlock(io);
            if (done) |reason| return reason;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }

    fn deinit(self: *Reply) void {
        self.tokens.deinit(gpa);
        self.rows.deinit(gpa);
    }
};

fn serve(native_opened: api.Opened, id: api.Id, request: *const api.Request) !Reply {
    var reply: Reply = .{};
    errdefer reply.deinit();
    try native_opened.engine.submit(id, request, .{ .ctx = &reply, .event = Reply.event });
    try std.testing.expectEqual(api.Reason.length, try reply.wait());
    try std.testing.expectEqual(@as(usize, made), reply.tokens.items.len);
    return reply;
}

fn opened(dir: []const u8, drafter: ?[]const u8, cache_gib: ?f64) !api.Opened {
    var why: []const u8 = "";
    const result = try native.open(gpa, gpa, io, .{
        .dir = dir,
        .model_type = "inkling2_loop",
        .context = context,
        .lanes = 1,
        .drafts = drafter != null,
        .drafter = drafter,
        .prompt_cache_gib = cache_gib,
        .decode_share = 0.5,
    }, &why);
    if (result == null) {
        std.debug.print("Mini Metal open refused: {s}\n", .{why});
        return error.MiniMetalOpenRefused;
    }
    return result.?;
}

fn expectTokens(want: *const Reference, got: *const Reply) !void {
    try std.testing.expectEqualSlices(u32, &want.tokens, got.tokens.items);
    try std.testing.expect(got.rows.items.len >= 1);
    try expectFirstRow(want.first, got.rows.items[0]);
}

fn expectCopyParity(plain: *const Reply, copied: *const Reply) !void {
    try std.testing.expectEqualSlices(u32, plain.tokens.items, copied.tokens.items);
    try std.testing.expectEqual(plain.rows.items.len, copied.rows.items.len);
    for (plain.rows.items, copied.rows.items) |want, got| try expectFirstRow(want, got);
    try std.testing.expectEqual(@as(u64, 0), plain.drafted);
    try std.testing.expectEqual(@as(u64, 0), plain.accepted);
    try std.testing.expect(copied.drafted > 0);
    try std.testing.expect(copied.accepted > 0);
}

fn transactions(dir: []const u8, drafter_dir: []const u8, ids: []const u32, want: *const Reference) !void {
    const model = try Model.load(gpa, io, dir);
    defer model.deinit();
    const target = try backend.Metal.init(gpa, model, .{ .capacity = context, .streams = 1, .budget = st.cacheBytes(context) });
    defer target.deinit();
    const draft_head = try dspark.Drafter.init(gpa, io, model.device, model, drafter_dir);
    defer draft_head.deinit();
    try draft_head.setContext(context);
    var wrapped = try lanes.drafted.Drafted.init(gpa, target.backend(), draft_head.drafter());
    defer wrapped.deinit();
    wrapped.first_rows = true;
    const b = wrapped.backend();
    const target_b = target.backend();
    var s = try lanes.Stream.init(gpa, .{ .id = "transactions", .prompt = ids, .max_new = made, .drafts = true, .logprobs = 3 });
    defer s.deinit(gpa);
    try b.prefill(&s);
    defer b.release(&s);
    try std.testing.expectEqual(want.tokens[0], try b.read(try b.first(&s, ids.len)));
    try expectFirstRow(want.first, try b.firstRow(&s));
    try std.testing.expectEqualSlices(u8, &want.tap, &(try featureRow(target_b, &s, ids.len - 1)));

    const full_drafts = want.tokens[1..8];
    const full_positions = [_]u64{ ids.len + 1, ids.len + 2, ids.len + 3, ids.len + 4, ids.len + 5, ids.len + 6, ids.len + 7, ids.len + 8 };
    const full_path = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const full = [_]lanes.backend.Window{.{ .stream = &s, .pending = want.tokens[0], .held = 0, .tokens = full_drafts, .parents = null, .positions = &full_positions }};
    var sampled: [8]u32 = undefined;
    var echo: [7]u32 = undefined;
    var full_out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &echo }};
    try b.verify(&full, &full_out);
    try std.testing.expectEqualSlices(u32, want.tokens[1..9], &sampled);
    try std.testing.expectEqualSlices(u32, full_drafts, &echo);
    try b.keep(&full, &.{&full_path});
    try std.testing.expectEqualSlices(u8, &want.full_tap, &(try featureRow(target_b, &s, ids.len + 7)));
    try b.draft(&.{.{ .stream = &s, .follow = want.tokens[1..9], .rows = &full_path, .start = ids.len, .position = ids.len + 9, .depth = 7 }});

    var partial_drafts = want.tokens[9..16].*;
    partial_drafts[6] = (want.tokens[15] + 1) % model.config.text.unpadded_vocab;
    const partial_positions = [_]u64{ ids.len + 9, ids.len + 10, ids.len + 11, ids.len + 12, ids.len + 13, ids.len + 14, ids.len + 15, ids.len + 16 };
    const partial = [_]lanes.backend.Window{.{ .stream = &s, .pending = want.tokens[8], .held = 0, .tokens = &partial_drafts, .parents = null, .positions = &partial_positions }};
    const path = [_]u32{ 0, 1, 2, 3, 4, 5, 6 };
    var partial_out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &echo }};
    try b.verify(&partial, &partial_out);
    try std.testing.expectEqualSlices(u32, want.tokens[9..16], sampled[0..7]);
    try b.keep(&partial, &.{&path});
    try std.testing.expectEqualSlices(u8, &want.partial_tap, &(try featureRow(target_b, &s, ids.len + 14)));
    try b.draft(&.{.{ .stream = &s, .follow = want.tokens[9..16], .rows = &path, .start = ids.len + 8, .position = ids.len + 16, .depth = 7 }});
    const next = try b.read(try b.queue(&s, .{ .value = want.tokens[15] }, ids.len + 16));
    try std.testing.expectEqual(want.tokens[16], next);
    try std.testing.expectEqualSlices(u8, &want.continuation_tap, &(try featureRow(target_b, &s, ids.len + 15)));
}

test "Mini Metal serving matches the host reference token for token" {
    const dir = try checkpoint();
    const ids = prompt(history, 1);
    var turn: [history + 16]u32 = undefined;
    @memcpy(turn[0..history], &ids);
    @memcpy(turn[history..], &prompt(16, 2));
    const refs = try withReference(dir, &ids, &turn);
    var tap_nonzero = false;
    for (refs.first.tap) |byte| tap_nonzero = tap_nonzero or byte != 0;
    try std.testing.expect(tap_nonzero);

    const opened_once = try opened(dir, null, 1);
    defer opened_once.close(opened_once.ctx);
    try std.testing.expectEqual(@as(?f64, 0.5), opened_once.engine.info().decode_share);
    const keepalive = opened_once.engine.keepaliveTarget() orelse return error.MissingMiniKeepalive;
    keepalive.tick(keepalive.ctx);
    var first = try serve(opened_once, 1, &.{ .prompt = &ids, .max_tokens = made, .drafts = false, .logprobs = 3, .history_len = ids.len });
    defer first.deinit();
    try expectTokens(&refs.first, &first);
    var fresh = try serve(opened_once, 2, &.{ .prompt = &turn, .max_tokens = made, .drafts = false, .logprobs = 3, .history_len = ids.len });
    defer fresh.deinit();
    try std.testing.expectEqual(@as(?u32, 0), fresh.cached);
    try expectTokens(&refs.second.?, &fresh);
    var restored = try serve(opened_once, 3, &.{ .prompt = &turn, .max_tokens = made, .drafts = false, .logprobs = 3, .history_len = ids.len });
    defer restored.deinit();
    try std.testing.expectEqual(@as(?u32, ids.len), restored.cached);
    try expectTokens(&refs.second.?, &restored);
}

test "Mini Metal opt-in DSpark drafting preserves target tokens and rolls back exactly" {
    const dir = try checkpoint();
    const drafter = @import("sys").getenv("TF_INKLING2_DSPARK_DIR") orelse return error.SkipZigTest;
    const ids = prompt(history, 3);
    var turn: [history + 16]u32 = undefined;
    @memcpy(turn[0..history], &ids);
    @memcpy(turn[history..], &prompt(16, 2));
    const refs = try withReference(dir, &ids, &turn);
    try transactions(dir, drafter, &ids, &refs.first);

    const opened_once = try opened(dir, drafter, 1);
    defer opened_once.close(opened_once.ctx);
    var plain = try serve(opened_once, 10, &.{ .prompt = &ids, .max_tokens = made, .drafts = false, .logprobs = 3, .history_len = ids.len });
    defer plain.deinit();
    try expectTokens(&refs.first, &plain);
    try std.testing.expectEqual(@as(u64, 0), plain.drafted);
    var drafted = try serve(opened_once, 11, &.{ .prompt = &ids, .max_tokens = made, .drafts = true, .logprobs = 3, .history_len = ids.len });
    defer drafted.deinit();
    try expectTokens(&refs.first, &drafted);
    try std.testing.expect(drafted.drafted > 0);
    try std.testing.expect(drafted.accepted > 0);
    var fresh = try serve(opened_once, 12, &.{ .prompt = &turn, .max_tokens = made, .drafts = true, .logprobs = 3, .history_len = ids.len });
    defer fresh.deinit();
    try std.testing.expectEqual(@as(?u32, 0), fresh.cached);
    try expectTokens(&refs.second.?, &fresh);
    var restored = try serve(opened_once, 13, &.{ .prompt = &turn, .max_tokens = made, .drafts = true, .logprobs = 3, .history_len = ids.len });
    defer restored.deinit();
    try std.testing.expectEqual(@as(?u32, ids.len), restored.cached);
    try expectTokens(&refs.second.?, &restored);
}

test "Mini Metal verify tree matches serial target bit-for-bit across all branches" {
    const dir = try checkpoint();
    const model = try Model.load(gpa, io, dir);
    defer model.deinit();
    const target = try backend.Metal.init(gpa, model, .{ .capacity = context, .streams = 3, .budget = 4 * st.cacheBytes(context) });
    defer target.deinit();
    try target.warm();
    const b = target.backend();
    try b.vtable.prepare_features.?(b.ptr, &taps);
    const ids = prompt(history + 4, 5);
    const vocab = model.config.text.unpadded_vocab;
    const parents = [_]i32{ -1, 0, 0, 1, 2, 4 };
    const positions = [_]u64{ ids.len + 1, ids.len + 2, ids.len + 2, ids.len + 3, ids.len + 3, ids.len + 4 };
    const branch_paths = [_][]const u32{ &.{0}, &.{ 0, 2 }, &.{ 0, 2, 4, 5 } };

    var base = try lanes.Stream.init(gpa, .{ .id = "tree-base", .prompt = &ids, .max_new = made });
    defer base.deinit(gpa);
    try b.prefill(&base);
    defer b.release(&base);
    const root_tok = try b.read(try b.first(&base, ids.len));
    const snap = try backend.Metal.snapSave(target, &base, @intCast(ids.len));
    defer backend.Metal.snapDrop(target, snap);

    var draft_toks: [5]u32 = undefined;
    for (&draft_toks, 0..) |*t, i| t.* = @intCast((root_tok + 13 + i * 29) % vocab);
    const tree_inputs = [_]u32{ root_tok, draft_toks[0], draft_toks[1], draft_toks[2], draft_toks[3], draft_toks[4] };

    for (branch_paths) |path| {
        var ts = try lanes.Stream.init(gpa, .{ .id = "tree", .prompt = &ids, .max_new = made, .reuse = .{ .saved = snap, .at = @intCast(ids.len) } });
        defer ts.deinit(gpa);
        var rs = try lanes.Stream.init(gpa, .{ .id = "serial", .prompt = &ids, .max_new = made, .reuse = .{ .saved = snap, .at = @intCast(ids.len) } });
        defer rs.deinit(gpa);
        try b.prefill(&ts);
        defer b.release(&ts);
        try b.prefill(&rs);
        defer b.release(&rs);

        var sampled: [6]u32 = undefined;
        var echoed: [5]u32 = undefined;
        var out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &echoed }};
        const win = [_]lanes.backend.Window{.{ .stream = &ts, .pending = root_tok, .held = 0, .tokens = &draft_toks, .parents = &parents, .positions = &positions }};
        try b.verify(&win, &out);
        const verify_logits = try gpa.dupe(f32, target.scratch.logits_all.slice(f32, 6 * vocab));
        defer gpa.free(verify_logits);

        try b.keep(&win, &.{path});
        const tc = target.caches.get(&ts).?;
        const tree_kept_logits = try gpa.dupe(f32, tc.logits.slice(f32, vocab));
        defer gpa.free(tree_kept_logits);

        const rc = target.caches.get(&rs).?;
        for (path, 0..) |node, j| {
            const got_tok = try b.read(try b.queue(&rs, .{ .value = tree_inputs[node] }, ids.len + 1 + j));
            try std.testing.expectEqual(got_tok, sampled[node]);
            for (rc.logits.slice(f32, vocab), verify_logits[node * vocab ..][0..vocab]) |a, c| {
                try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(c)));
            }
        }
        for (rc.logits.slice(f32, vocab), tree_kept_logits) |a, c| {
            try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(c)));
        }
        const next_t = try b.read(try b.queue(&ts, .{ .value = draft_toks[0] }, ids.len + path.len + 1));
        const next_r = try b.read(try b.queue(&rs, .{ .value = draft_toks[0] }, ids.len + path.len + 1));
        try std.testing.expectEqual(next_r, next_t);
        for (rc.logits.slice(f32, vocab), tc.logits.slice(f32, vocab)) |a, c| {
            try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(c)));
        }
    }
}

test "Mini Metal SuffixLookup copy drafting reports draft_sources.copies, accepts copies, and matches plain tokens" {
    const dir = try checkpoint();
    var why: []const u8 = "";
    const with_copies = (try native.open(gpa, gpa, io, .{
        .dir = dir,
        .model_type = "inkling2_loop",
        .context = context,
        .lanes = 1,
        .drafts = true,
        .drafter = null,
        .decode_share = 0.5,
    }, &why)) orelse return error.MiniMetalOpenRefused;
    defer with_copies.close(with_copies.ctx);
    const info_on = with_copies.engine.info().draft_sources;
    try std.testing.expect(info_on.copies);
    try std.testing.expect(!info_on.mtp);
    try std.testing.expect(info_on.drafter == null);

    why = "";
    const no_drafts = (try native.open(gpa, gpa, io, .{
        .dir = dir,
        .model_type = "inkling2_loop",
        .context = context,
        .lanes = 1,
        .drafts = false,
        .drafter = null,
        .decode_share = 0.5,
    }, &why)) orelse return error.MiniMetalOpenRefused;
    defer no_drafts.close(no_drafts.ctx);
    const info_off = no_drafts.engine.info().draft_sources;
    try std.testing.expect(!info_off.copies);
    try std.testing.expect(!info_off.mtp);
    try std.testing.expect(info_off.drafter == null);

    const ids = copyPrompt();
    var plain = try serve(no_drafts, 1, &.{ .prompt = &ids, .max_tokens = made, .drafts = true, .logprobs = 3 });
    defer plain.deinit();
    var copied = try serve(with_copies, 2, &.{ .prompt = &ids, .max_tokens = made, .drafts = true, .logprobs = 3 });
    defer copied.deinit();
    try expectCopyParity(&plain, &copied);
}

test "Mini Metal SuffixLookup copy parity guard rejects idle drafting and copyPrompt triggers copy_match=8" {
    const span = prompt(16, 1);
    const ids = copyPrompt();
    var ctx: [ids.len + 1]u32 = undefined;
    @memcpy(ctx[0..ids.len], &ids);
    ctx[ids.len] = span[8];
    var lookup = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer lookup.deinit();
    const got = try lookup.propose(&ctx, st.window_rows - 1);
    try std.testing.expect(lookup.last_match >= 8);
    try std.testing.expectEqual(@as(usize, st.window_rows - 1), got.len);
    try std.testing.expectEqualSlices(u32, span[9..16], got[0..7]);
    try std.testing.expectEqualSlices(u32, span[0..8], got[7..15]);

    var plain: Reply = .{};
    defer plain.deinit();
    try plain.tokens.appendSlice(gpa, span[8..16]);
    var idle: Reply = .{};
    defer idle.deinit();
    try idle.tokens.appendSlice(gpa, span[8..16]);
    try std.testing.expectError(error.TestUnexpectedResult, expectCopyParity(&plain, &idle));
    idle.drafted = 7;
    try std.testing.expectError(error.TestUnexpectedResult, expectCopyParity(&plain, &idle));
    idle.accepted = 7;
    try expectCopyParity(&plain, &idle);
}
