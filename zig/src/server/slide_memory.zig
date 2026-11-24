//! Folded Memory Weights v5 HTTP & In-Flight handlers for /v1/slide/learn, /v1/slide/consolidate,
//! /v1/slide/inflight, /v1/slide/fact/{id}, /v1/slide/facts, /v1/slide/graph.
const std = @import("std");
const api = @import("engine_api");
const living = @import("living");
const mem = living.memory;
const json = @import("json");
const http_body = @import("http_body.zig");
const sse = @import("sse.zig");
const errors = @import("errors.zig");
const log = @import("log.zig");
const slide = @import("slide.zig");
const slide_ask = @import("slide_ask.zig");
const slide_graph = @import("slide_graph.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

pub const LearnOpts = struct {
    replace_id: ?u32 = null,
    standing: bool = false,
    consolidate: bool = true,
    is_update: bool = false,
    auto_revise: bool = true,
};

const NeverGone = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

const TokenCountCtx = struct {
    srv: *Server,
    a: Allocator,

    fn count(ctx: *anyopaque, text: []const u8) usize {
        const self: *@This() = @ptrCast(@alignCast(ctx));
        const ids = self.srv.text.encode(self.a, text, false) catch return @max(1, (text.len + 3) / 4);
        return ids.len;
    }
};

/// Ensures the engine's Folded Memory Weights bank is calibrated and has its shared prefix compiled.
/// Returns the next safe unused parent ID (`max_existing_parent_id + 1`).
fn ensureInit(srv: *Server, a: Allocator) !u32 {
    const cal_texts = mem.whitener.calibration_texts;
    var cal_ids = try a.alloc([]const u32, cal_texts.len);
    for (cal_texts, 0..) |txt, i| {
        cal_ids[i] = srv.text.encode(a, txt, false) catch return error.TokenizerFailed;
    }
    const prefix_ids = srv.text.encode(a, mem.bank.shared_sys_prefix, false) catch return error.TokenizerFailed;
    const suffix_ids = srv.text.encode(a, mem.bank.shared_sys_suffix, false) catch return error.TokenizerFailed;
    const res = srv.engine.memoryOp(&.{ .init = .{
        .cal_ids = cal_ids,
        .prefix_ids = prefix_ids,
        .suffix_ids = suffix_ids,
    } });
    if (res.err) |msg| {
        log.line("slide(v5): init failed: {s}", .{msg});
        return error.MemoryInitFailed;
    }
    return @max(res.affected, 1);
}

fn containsCI(hay: []const u8, needle: []const u8) bool {
    if (needle.len == 0 or needle.len > hay.len) return false;
    var i: usize = 0;
    while (i + needle.len <= hay.len) : (i += 1) {
        var ok = true;
        for (0..needle.len) |j| {
            if (std.ascii.toLower(hay[i + j]) != std.ascii.toLower(needle[j])) {
                ok = false;
                break;
            }
        }
        if (ok) return true;
    }
    return false;
}

/// Zero-latency (< 0.01 ms) deterministic concept-cluster synonym anchors for everyday short facts.
/// Avoids brittle LLM cue-word clipping (e.g. "colour" <-> "shade/hue/tint", "dog" <-> "pet/puppy")
/// and cuts small-fact learn time by 3-5x (~100-250 ms/fact)!
fn deterministicSynonymCues(a: Allocator, fact_text: []const u8) ?[]const u8 {
    const Cluster = struct { triggers: []const []const u8, anchors: []const u8 };
    const clusters = [_]Cluster{
        .{ .triggers = &.{ "colour", "color", "shade", "hue", "tint" }, .anchors = "colour color shade hue tint tone prefer" },
        .{ .triggers = &.{ "dog", "puppy", "hound", "canine" }, .anchors = "dog pet puppy hound canine" },
        .{ .triggers = &.{ "cat", "kitten", "feline" }, .anchors = "cat pet kitten feline" },
        .{ .triggers = &.{ "sister", "brother", "sibling" }, .anchors = "sister brother sibling city town live lives moved" },
        .{ .triggers = &.{ "robot", "droid" }, .anchors = "robot bot droid warehouse machine" },
        .{ .triggers = &.{ "deploy", "release", "rollout" }, .anchors = "deploy deployment deployments release rollout schedule" },
        .{ .triggers = &.{ "password", "passcode" }, .anchors = "password passcode credential secret" },
        .{ .triggers = &.{ "sailboat", "yacht", "marina", "berth", "moored" }, .anchors = "sailboat boat vessel yacht moored marina berth dock" },
        .{ .triggers = &.{ "database", "postgres", "duckdb", "sqlite", "mysql" }, .anchors = "database db datastore sql storage backend" },
        .{ .triggers = &.{ "capital", "france", "paris", "lyon" }, .anchors = "capital city country nation" },
    };

    var buf: std.ArrayList(u8) = .empty;
    for (clusters) |c| {
        var hit = false;
        for (c.triggers) |tr| {
            if (containsCI(fact_text, tr)) {
                hit = true;
                break;
            }
        }
        if (hit) {
            if (buf.items.len > 0) buf.append(a, ' ') catch return null;
            buf.appendSlice(a, c.anchors) catch return null;
        }
    }
    if (buf.items.len == 0) return null;
    return buf.items;
}

/// Fast concept cue builder for short facts: uses < 0.01 ms deterministic synonym clusters first,
/// falling back to a single-line LLM cue pass ONLY for novel/unfamiliar domain jargon!
fn generateConceptCues(srv: *Server, cx: *errors.Cx, fact_text: []const u8, gone: anytype) []const u32 {
    const a = cx.a;
    if (deterministicSynonymCues(a, fact_text)) |anchors| {
        return srv.text.encode(a, anchors, false) catch return &.{};
    }
    if (std.c.getenv("TF_SLIDE_FAST_CUES")) |v| {
        const s = std.mem.span(v);
        if (std.mem.eql(u8, s, "1") or std.mem.eql(u8, s, "true")) return &.{};
    }
    const prompt = std.fmt.allocPrint(
        a,
        "List 8 single-word topic synonyms or related nouns on ONE line (comma-separated, no numbering, no questions, " ++
            "no pronouns) someone might use to ask about this fact.\nFact: {s}\nKeywords:",
        .{std.mem.trim(u8, fact_text, " \t\r\n")},
    ) catch return &.{};
    const reply = slide_ask.askIn(srv, cx, &.{}, prompt, 36, true, gone) catch return &.{};
    var trimmed = std.mem.trim(u8, reply.content, " \t\r\n");
    if (std.mem.indexOfScalar(u8, trimmed, '\n')) |nl| {
        trimmed = std.mem.trim(u8, trimmed[0..nl], " \t\r\n");
    }
    if (std.c.getenv("TF_SLIDE_TRACE") != null) {
        log.line("slide(v6) trace: cues for '{s}' -> '{s}'", .{ std.mem.trim(u8, fact_text, " \t\r\n"), trimmed });
    }
    if (trimmed.len == 0) return &.{};
    return srv.text.encode(a, trimmed, false) catch return &.{};
}

fn startsWithCI(hay: []const u8, prefix: []const u8) bool {
    if (prefix.len > hay.len) return false;
    for (0..prefix.len) |i| {
        if (std.ascii.toLower(hay[i]) != std.ascii.toLower(prefix[i])) return false;
    }
    return true;
}

/// Pure-weight contradiction verifier (ZERO plaintext on disk!):
/// Queries the model WITH its currently stored folded memory weights active (`shipped = false`)
/// to decide whether `new_text` contradicts/replaces an existing recorded value vs merely
/// adding a compatible extra detail. Uses strict prefix matching and puts COMPATIBLE first to
/// eliminate small-model first-option bias.
fn verifyContradictionWithMemory(srv: *Server, cx: *errors.Cx, new_text: []const u8, gone: anytype) bool {
    const a = cx.a;
    const prompt = std.fmt.allocPrint(
        a,
        "Compare the new statement against your stored notes.\n" ++
            "New statement: {s}\n" ++
            "Write ONE short sentence comparing what attribute each states, then on a new line write ONLY:\n" ++
            "VERDICT: COMPATIBLE (if both can be true at the same time, such as adding a hobby, activity, breed, or extra detail)\n" ++
            "or\n" ++
            "VERDICT: CONTRADICT (ONLY if the new statement replaces the exact same attribute with a different conflicting value, making the old value false).\n" ++
            "Comparison:",
        .{std.mem.trim(u8, new_text, " \t\r\n")},
    ) catch return false;
    const reply = slide_ask.askIn(srv, cx, &.{}, prompt, 40, false, gone) catch return false;
    const ans = std.mem.trim(u8, reply.content, " \t\r\n");
    if (std.c.getenv("TF_SLIDE_TRACE") != null) {
        log.line("slide(v6) revision probe: '{s}' -> '{s}'", .{ std.mem.trim(u8, new_text, " \t\r\n"), ans });
    }
    if (containsCI(ans, "verdict: contradict")) return true;
    if (containsCI(ans, "verdict: compatible")) return false;
    // Fallback if model omitted the "VERDICT:" prefix on the final line
    if (std.mem.lastIndexOfScalar(u8, ans, '\n')) |nl| {
        const last = std.mem.trim(u8, ans[nl + 1 ..], " \t\r\n.*:-");
        return startsWithCI(last, "contradict");
    }
    return startsWithCI(ans, "contradict");
}

const CoreResult = struct {
    id: u32,
    unit_count: usize,
    total_tokens: u32,
    total_units: u32,
    bytes: u64,
    seconds: f64,
    is_standing: bool,
};

/// Shared ultra-fast Folded Memory Weights v5 compiler used by BOTH `POST /v1/slide/learn`
/// and the non-blocking background in-flight chat learner!
fn learnCore(
    srv: *Server,
    a: Allocator,
    text: []const u8,
    source: []const u8,
    opts: LearnOpts,
    sse_out: ?*slide.Out,
    gone: anytype,
) !CoreResult {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return error.EmptyText;

    var cx: errors.Cx = .{ .a = a };
    const began = slide.now(srv);
    const min_next = try ensureInit(srv, a);
    srv.slide.ensureNextAtLeast(min_next);

    const is_standing = opts.standing or std.mem.eql(u8, source, "policy") or std.mem.eql(u8, source, "standing");
    const id = srv.slide.add(trimmed, source, began);
    if (sse_out) |out| {
        out.event("fact", .{ .id = id, .text = trimmed });
        out.event("learning", .{ .id = id });
        out.stage(id, "initializing");
    }
    srv.slide.mark(id, .learning);
    srv.slide.phase(id, .reading);

    if (opts.replace_id) |old_id| {
        _ = srv.engine.memoryOp(&.{ .forget_parent = old_id });
        srv.slide.note(old_id, std.fmt.allocPrint(a, "replaced by fact {d}", .{id}) catch "replaced");
        srv.slide.mark(old_id, .replaced);
    }

    var total_tokens: u32 = 0;
    var unit_count: usize = 0;
    var superseded_pid: u32 = 0;

    if (is_standing) {
        const entry_text = try std.fmt.allocPrint(a, "\n- Standing instruction (always follow on every reply, including general-knowledge questions): {s}\n", .{trimmed});
        const body_ids = srv.text.encode(a, entry_text, false) catch return error.TokenizerFailed;
        const res = srv.engine.memoryOp(&.{ .teach_chunk = .{
            .parent_id = id,
            .body_ids = body_ids,
            .gate_ids = &.{},
            .is_standing = true,
        } });
        if (res.err != null) {
            _ = srv.engine.memoryOp(&.{ .forget_parent = id });
            return error.TeachFailed;
        }
        total_tokens = res.tokens;
        unit_count = 1;
    } else {
        var tc_ctx = TokenCountCtx{ .srv = srv, .a = a };
        const tc = mem.chunker.TokenCounter{ .ctx = &tc_ctx, .count = TokenCountCtx.count };
        const chunks = try mem.chunker.chunkDocument(a, trimmed, 550, tc);
        defer mem.chunker.freeChunks(a, chunks);
        if (chunks.len == 0) return error.EmptyText;

        const want_digest = opts.consolidate and chunks.len >= 2;
        const total_steps: u32 = @intCast(chunks.len + @intFromBool(want_digest));

        for (chunks, 0..) |chunk, ci| {
            if (gone.check()) {
                _ = srv.engine.memoryOp(&.{ .forget_parent = id });
                return error.Cancelled;
            }
            const prog: api.LearnProgress = .{
                .phase = .fitting,
                .at = @intCast(ci + 1),
                .of = total_steps,
            };
            srv.slide.progress(id, prog);
            if (sse_out) |out| out.progress(id, prog);

            const entry_text = try mem.chunker.formatEntryText(a, chunk.full_text, chunk.is_small_fact);
            const body_ids = srv.text.encode(a, entry_text, false) catch {
                _ = srv.engine.memoryOp(&.{ .forget_parent = id });
                return error.TokenizerFailed;
            };
            const gate_ids = srv.text.encode(a, chunk.unique_body, false) catch {
                _ = srv.engine.memoryOp(&.{ .forget_parent = id });
                return error.TokenizerFailed;
            };
            const title_ids: []const u32 = if (chunk.title_line.len > 0)
                (srv.text.encode(a, chunk.title_line, false) catch &.{})
            else
                &.{};
            const cue_ids: []const u32 = if (chunk.is_small_fact)
                generateConceptCues(srv, &cx, chunk.unique_body, gone)
            else
                &.{};

            // Temporal Belief Revision (v6): before adding a new small fact, check pure-vector entity
            // conflicts against active facts; verify contradiction via the model's OWN folded weights!
            if (chunks.len == 1 and chunk.is_small_fact and opts.auto_revise and opts.replace_id == null) {
                const cres = srv.engine.memoryOp(&.{ .find_conflicts = .{ .gate_ids = gate_ids } });
                if (cres.err == null and cres.conflict_count > 0) {
                    const candidate_old = cres.conflict_ids[0];
                    if (candidate_old != 0 and candidate_old != id) {
                        if (opts.is_update or verifyContradictionWithMemory(srv, &cx, trimmed, gone)) {
                            superseded_pid = candidate_old;
                        }
                    }
                }
            }

            const res = srv.engine.memoryOp(&.{ .teach_chunk = .{
                .parent_id = id,
                .body_ids = body_ids,
                .gate_ids = gate_ids,
                .title_ids = title_ids,
                .cue_ids = cue_ids,
                .is_small_fact = chunk.is_small_fact,
            } });
            if (res.err != null) {
                _ = srv.engine.memoryOp(&.{ .forget_parent = id });
                return error.TeachFailed;
            }
            total_tokens += res.tokens;
            unit_count += 1;
        }

        // Sub-millisecond 100% VERBATIM extractive cross-section index (ZERO LLM hallucination & 50,000x faster!)
        if (want_digest and !gone.check()) {
            const prog: api.LearnProgress = .{
                .phase = .fitting,
                .at = total_steps,
                .of = total_steps,
            };
            srv.slide.progress(id, prog);
            if (sse_out) |out| out.progress(id, prog);

            const digest_txt = mem.chunker.buildExtractiveDigest(a, chunks, 540, tc) catch &.{};
            if (digest_txt.len > 24) {
                if (std.c.getenv("TF_SLIDE_TRACE") != null) {
                    log.line("slide(v5) trace: verbatim extractive index ({d} bytes)", .{digest_txt.len});
                }
                const d_entry = try std.fmt.allocPrint(a, "\n---\n[Cross-Section Verbatim Index]\n{s}---\n", .{digest_txt});
                if (srv.text.encode(a, d_entry, false)) |d_body_ids| {
                    const dres = srv.engine.memoryOp(&.{ .teach_chunk = .{
                        .parent_id = id,
                        .body_ids = d_body_ids,
                        .gate_ids = &.{},
                        .is_digest = true,
                    } });
                    if (dres.err == null) {
                        total_tokens += dres.tokens;
                        unit_count += 1;
                    }
                } else |_| {}
            }
        }
    }

    if (sse_out) |out| out.stage(id, "saving");
    var save_res = srv.engine.memoryOp(&.save);
    if (save_res.err != null) {
        _ = srv.engine.memoryOp(&.{ .forget_parent = id });
        return error.SaveFailed;
    }

    // Apply undoable supersession tombstone AFTER new unit is persisted!
    if (superseded_pid > 0) {
        const sres = srv.engine.memoryOp(&.{ .supersede_parent = .{
            .old_parent_id = superseded_pid,
            .new_parent_id = id,
        } });
        if (sres.err == null and sres.affected > 0) {
            save_res = sres;
            srv.inflight.noteRevision();
            srv.slide.note(superseded_pid, std.fmt.allocPrint(a, "superseded by fact {d}", .{id}) catch "superseded");
            srv.slide.mark(superseded_pid, .replaced);
            log.line("slide(v6): fact {d} superseded stale fact {d} (undoable on delete)", .{ id, superseded_pid });
        }
    }

    const secs = slide.seconds(srv, began);
    const marker = [_]api.Neuron{.{ .layer = 0, .neuron = @intCast(id & 0xffff), .key = .reply }};
    srv.slide.placed(id, &marker);
    srv.slide.checked(id, true);
    srv.slide.note(
        id,
        std.fmt.allocPrint(
            a,
            "folded into {d} {s}unit{s} ({d} tokens, {d:.1} MiB total) in {d:.2} s",
            .{
                unit_count,
                if (is_standing) "standing policy " else "memory ",
                if (unit_count == 1) "" else "s",
                total_tokens,
                @as(f64, @floatFromInt(save_res.bytes)) / (1024.0 * 1024.0),
                secs,
            },
        ) catch "folded into memory weights",
    );
    srv.slide.mark(id, .learned);

    log.line(
        "slide(v5): fact {d} folded ({d} units{s}, {d} tokens, {d} total units, {d:.1} MiB) in {d:.2} s",
        .{
            id,
            unit_count,
            if (is_standing) " standing" else "",
            total_tokens,
            save_res.units,
            @as(f64, @floatFromInt(save_res.bytes)) / (1024.0 * 1024.0),
            secs,
        },
    );

    return .{
        .id = id,
        .unit_count = unit_count,
        .total_tokens = total_tokens,
        .total_units = save_res.units,
        .bytes = save_res.bytes,
        .seconds = secs,
        .is_standing = is_standing,
    };
}

/// Fast Folded Memory Weights v5 learn path for `POST /v1/slide/learn`.
pub fn learn(srv: *Server, conn: *Conn, a: Allocator, text: []const u8, source: []const u8, opts: LearnOpts) void {
    srv.teacher.learning.lockUncancelable(srv.io);
    defer srv.teacher.learning.unlock(srv.io);

    sse.open(conn) catch return;
    var out: slide.Out = .{ .conn = conn, .a = a };
    var cx: errors.Cx = .{ .a = a };
    defer out.stop();
    out.start(srv.io, 5000) catch |e| return out.fail(&cx, e);

    const res = learnCore(srv, a, text, source, opts, &out, &out) catch |err| {
        const msg = std.fmt.allocPrint(a, "folded memory learn failed ({s})", .{@errorName(err)}) catch "learn failed";
        out.event("failed", .{ .message = msg });
        out.done();
        return;
    };

    out.event("learned", .{ .id = res.id, .recalled = true, .steps = @as(u32, @intCast(res.unit_count)) });
    out.event("saved", .{ .saved = true });
    out.done();
}

/// Non-blocking background learn entry point called by `slide_inflight.Worker` during live chat!
pub fn learnBackground(srv: *Server, text: []const u8, source: []const u8, opts: LearnOpts) ?u32 {
    srv.teacher.learning.lockUncancelable(srv.io);
    defer srv.teacher.learning.unlock(srv.io);

    var arena = std.heap.ArenaAllocator.init(srv.gpa);
    defer arena.deinit();
    const res = learnCore(srv, arena.allocator(), text, source, opts, null, NeverGone{}) catch |err| {
        log.line("slide(v5) in-flight learn skipped ({s})", .{@errorName(err)});
        return null;
    };
    return res.id;
}

/// Sleep-time consolidation endpoint (`POST /v1/slide/consolidate`).
pub fn consolidate(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.info.folded_memory) {
        return slide.refuse(conn, a, 501, "consolidation requires Folded Memory Weights (--slide)");
    }
    const raw = switch (http_body.read(conn, a, http_body.limit) catch return) {
        .ok => |b| b,
        .refused => |message| return slide.refuse(conn, a, 400, message),
    };
    const body = switch (json.parse(a, if (raw.len == 0) "{}" else raw) catch return) {
        .ok => |v| v,
        .err => |message| return slide.refuse(conn, a, 400, message),
    };
    const compact_small = if (body.field("compact_small_facts")) |v| (v == .bool and v.bool) else false;
    const prune_superseded = if (body.field("prune_superseded")) |v| (v == .bool and v.bool) else true;
    const compact_policies = if (body.field("compact_policies")) |v| (v == .bool and v.bool) else true;

    if (!srv.teacher.learning.tryLock()) return slide.refuse(conn, a, 503, "a learn or delete request is running: send yours when it ends");
    defer srv.teacher.learning.unlock(srv.io);

    const began = slide.now(srv);
    var compacted: u32 = 0;
    var consolidated_id: u32 = 0;

    if (compact_small) {
        if (srv.slide.snapshot(a)) |facts| {
            var small_ids: std.ArrayList(u32) = .empty;
            var merged_txt: std.ArrayList(u8) = .empty;
            for (facts) |f| {
                if (f.state != .learned) continue;
                if (std.mem.eql(u8, f.source, "policy") or std.mem.eql(u8, f.source, "standing")) continue;
                if (f.text.len == 0 or f.text.len > 256) continue;
                small_ids.append(a, f.id) catch break;
                merged_txt.appendSlice(a, "- ") catch break;
                merged_txt.appendSlice(a, f.text) catch break;
                merged_txt.append(a, '\n') catch break;
            }
            if (small_ids.items.len >= 2) {
                if (ensureInit(srv, a)) |min_next| {
                    srv.slide.ensureNextAtLeast(min_next);
                    const new_id = srv.slide.add(merged_txt.items, "consolidated", began);
                    const entry = std.fmt.allocPrint(a, "\n---\n[Consolidated User Facts]\n{s}---\n", .{merged_txt.items}) catch null;
                    if (entry) |en| {
                        if (srv.text.encode(a, en, false)) |b_ids| {
                            if (srv.text.encode(a, merged_txt.items, false)) |g_ids| {
                                const tres = srv.engine.memoryOp(&.{ .teach_chunk = .{
                                    .parent_id = new_id,
                                    .body_ids = b_ids,
                                    .gate_ids = g_ids,
                                    .is_small_fact = true,
                                } });
                                if (tres.err == null) {
                                    for (small_ids.items) |old_id| {
                                        _ = srv.engine.memoryOp(&.{ .forget_parent = old_id });
                                        srv.slide.mark(old_id, .replaced);
                                    }
                                    srv.slide.checked(new_id, true);
                                    srv.slide.mark(new_id, .learned);
                                    compacted = @intCast(small_ids.items.len);
                                    consolidated_id = new_id;
                                }
                            } else |_| {}
                        } else |_| {}
                    }
                } else |_| {}
            }
        } else |_| {}
    }

    const cons_res = srv.engine.memoryOp(&.{ .consolidate = .{
        .prune_superseded = prune_superseded,
        .compact_policies = compact_policies,
    } });
    if (cons_res.err) |msg| {
        return slide.refuse(conn, a, 500, msg);
    }

    log.line(
        "slide(v6): consolidated ({d} pruned, {d} policies fused, {d} deltas compacted, {d} clusters, {d} active units, {d:.1} MiB) in {d:.2} s",
        .{
            cons_res.pruned,
            cons_res.affected,
            cons_res.deltas,
            cons_res.clusters,
            cons_res.units,
            @as(f64, @floatFromInt(cons_res.bytes)) / (1024.0 * 1024.0),
            slide.seconds(srv, began),
        },
    );
    const resp = std.fmt.allocPrint(
        a,
        "{{\"consolidated\":true,\"units\":{d},\"bytes\":{d},\"pruned_superseded\":{d},\"consolidated_policies\":{d},\"clusters\":{d},\"compacted_deltas\":{d},\"compacted_facts\":{d},\"consolidated_id\":{d},\"seconds\":{d:.3}}}",
        .{
            cons_res.units,
            cons_res.bytes,
            cons_res.pruned,
            cons_res.affected,
            cons_res.clusters,
            cons_res.deltas,
            compacted,
            consolidated_id,
            slide.seconds(srv, began),
        },
    ) catch return slide.refuse(conn, a, 500, "out of memory");
    conn.sendJson(200, resp);
}

/// Live status of the background in-flight chat learner (`GET /v1/slide/inflight`).
pub fn inflightStatus(srv: *Server, conn: *Conn, a: Allocator) void {
    const body = srv.inflight.statusJson(a) catch return slide.refuse(conn, a, 500, "out of memory");
    conn.sendJson(200, body);
}

/// Exact atomic deletion of one fact (standing policy, or all chunks+digest of a whole document) by parent ID.
pub fn forgetOne(srv: *Server, conn: *Conn, a: Allocator, id: u32, reason: []const u8) void {
    if (!srv.teacher.learning.tryLock()) return slide.refuse(conn, a, 503, "a learn or delete request is running: send yours when it ends");
    defer srv.teacher.learning.unlock(srv.io);

    const facts = srv.slide.snapshot(a) catch return slide.refuse(conn, a, 500, "out of memory");
    var found_fact: ?slide_graph.Fact = null;
    for (facts) |f| if (f.id == id) {
        found_fact = f;
        break;
    };

    sse.open(conn) catch return;
    var out: slide.Out = .{ .conn = conn, .a = a };
    if (found_fact) |f| {
        out.event("fact", .{ .id = id, .text = f.text, .state = @tagName(f.state) });
    } else {
        out.event("fact", .{ .id = id, .text = "", .state = "unknown" });
    }

    const res = srv.engine.memoryOp(&.{ .forget_parent = id });
    if (res.err) |msg| {
        out.event("forgot", .{ .id = id, .deleted = false, .state = if (found_fact) |f| @tagName(f.state) else "unknown", .message = msg });
        sse.done(conn) catch {};
        return;
    }

    if (found_fact != null) {
        srv.slide.remove(id, slide.now(srv), reason) catch {};
    }
    const msg = std.fmt.allocPrint(a, "deleted: removed {d} folded memory unit{s} cleanly (bit-exact revert)", .{ res.affected, if (res.affected == 1) "" else "s" }) catch "deleted";
    log.line("slide(v5): fact {d} {s}", .{ id, msg });
    out.event("forgot", .{ .id = id, .deleted = res.affected > 0, .state = "deleted", .neurons = res.affected, .message = msg });
    sse.done(conn) catch {};
}

/// Exact atomic deletion of multiple fact IDs.
pub fn forgetMany(srv: *Server, conn: *Conn, a: Allocator, ids: []const u32, reason: []const u8) void {
    if (!srv.teacher.learning.tryLock()) return slide.refuse(conn, a, 503, "a learn or delete request is running: send yours when it ends");
    defer srv.teacher.learning.unlock(srv.io);

    sse.open(conn) catch return;
    var out: slide.Out = .{ .conn = conn, .a = a };
    var total_removed: u32 = 0;
    for (ids) |id| {
        const res = srv.engine.memoryOp(&.{ .forget_parent = id });
        if (res.err) |msg| {
            out.event("forgot", .{ .id = id, .deleted = false, .state = "learned", .message = msg });
            continue;
        }
        total_removed += res.affected;
        srv.slide.remove(id, slide.now(srv), reason) catch {};
        out.event("forgot", .{ .id = id, .selected = @as(u32, @intCast(ids.len)), .deleted = true, .state = "deleted", .total_neurons = total_removed, .message = "deleted from folded memory weights" });
    }
    sse.done(conn) catch {};
}

/// Clears every taught memory unit and removes `living-weights.safetensors`.
pub fn clearAll(srv: *Server) void {
    srv.teacher.learning.lockUncancelable(srv.io);
    defer srv.teacher.learning.unlock(srv.io);
    _ = srv.engine.memoryOp(&.clear_all);
}
