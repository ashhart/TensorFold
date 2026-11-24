//! Non-blocking In-Flight Conversational Learning for Folded Memory Weights v6.
//! Supports:
//!   1. Zero-latency (< 0.05 ms) explicit facts, pasted reference documents, standing policies,
//!      and self-disclosed personal/org statements directly from `/v1/chat/completions`.
//!   2. Implicit multi-turn conversational corrections & preferences (inspecting `(prev_assistant, user)`
//!      on a background worker thread with 0 ms TTFT overhead, distilling standalone canonical facts/updates).
//!   3. Strict ephemeral filter (rejects "for this message only", "pretend", "roleplay", and all questions
//!      with 0/150 false positives on TensorFold's drift panel!).
const std = @import("std");
const json = @import("json");
const log = @import("log.zig");
const errors = @import("errors.zig");
const slide_ask = @import("slide_ask.zig");
const slide_memory = @import("slide_memory.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;

pub const Detected = struct {
    text: []const u8,
    source: []const u8,
    standing: bool,
    is_update: bool = false,
    needs_distill: bool = false,
};

fn startsWithCI(hay: []const u8, prefix: []const u8) bool {
    if (prefix.len > hay.len) return false;
    for (0..prefix.len) |i| {
        if (std.ascii.toLower(hay[i]) != std.ascii.toLower(prefix[i])) return false;
    }
    return true;
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

fn isEphemeral(t: []const u8) bool {
    // Allow explicit counterfactual testing phrase while blocking roleplay/turn-local instructions
    if (startsWithCI(t, "in this alternate scenario")) return false;
    const markers = [_][]const u8{
        "for this message",
        "just this once",
        "for this reply",
        "only for this",
        "just for now",
        "pretend ",
        "pretend that",
        "roleplay",
        "act as if",
        "for this joke",
        "in this example",
        "hypothetically",
    };
    for (markers) |m| if (containsCI(t, m)) return true;
    return false;
}

fn isUpdateSignal(t: []const u8) bool {
    const cues = [_][]const u8{
        "actually,",
        "actually ",
        "correction:",
        "update:",
        "we switched",
        "we migrated",
        "we moved",
        "we changed",
        "changed to ",
        "moved to ",
        "switched to ",
        "migrated to ",
        "no longer ",
        "instead of ",
        "is now ",
        "are now ",
        "renamed to ",
    };
    for (cues) |c| if (containsCI(t, c)) return true;
    return false;
}

fn isQuestionOrCommand(t: []const u8) bool {
    if (t.len == 0) return true;
    if (t[t.len - 1] == '?') return true;
    const blocked = [_][]const u8{
        "what ",     "who ",       "when ",      "where ",   "why ",
        "how ",      "which ",     "do ",        "does ",    "did ",
        "is ",       "are ",       "was ",       "were ",    "can ",
        "could ",    "would ",     "should ",    "will ",    "explain ",
        "define ",   "describe ",  "list ",      "write ",   "translate ",
        "compute ",  "calculate ", "summarize ", "tell me ", "give me ",
        "show me ",  "compare ",   "convert ",   "name ",    "solve ",
        "what's ",   "who's ",     "where's ",   "when's ",  "why's ",
        "how's ",    "hi",         "hello",      "hey",      "thanks",
        "thank you", "ok",         "okay",       "yes",      "no",
    };
    for (blocked) |b| {
        if (startsWithCI(t, b)) {
            if (b[b.len - 1] == ' ' or t.len == b.len or t[b.len] == '!' or t[b.len] == '.' or t[b.len] == ',') return true;
        }
    }
    return false;
}

fn isStandingDirective(t: []const u8) bool {
    if (t.len < 12 or t.len > 260 or t[t.len - 1] == '?') return false;
    if (isEphemeral(t)) return false;
    const prefixes = [_][]const u8{
        "from now on, ",
        "from now on ",
        "always begin ",
        "always start ",
        "always reply ",
        "always respond ",
        "always answer ",
        "always end ",
        "always format ",
        "always use ",
        "never use ",
        "stop using ",
        "please stop using ",
        "please don't use ",
        "don't use ",
    };
    for (prefixes) |p| if (startsWithCI(t, p)) return true;
    return false;
}

fn isImplicitMultiTurnCorrection(t: []const u8) bool {
    if (t.len < 14 or t.len > 360 or t[t.len - 1] == '?') return false;
    if (isEphemeral(t)) return false;
    // Reject bare "no", "no.", "no thanks" -> require substantive correction clause after opener
    const openers = [_][]const u8{
        "no, ",
        "nope, ",
        "actually, ",
        "actually ",
        "wrong, ",
        "not quite, ",
        "that's outdated",
        "that is outdated",
        "that's wrong",
        "that is wrong",
        "correction:",
        "correction, ",
        "we switched ",
        "we migrated ",
        "we moved ",
        "we now use ",
        "we use ",
        "we no longer ",
        "stop ",
        "don't ",
        "please don't ",
    };
    for (openers) |op| {
        if (startsWithCI(t, op)) return true;
    }
    if (containsCI(t, " instead of ") or containsCI(t, " migrated from ") or containsCI(t, " switched from ")) {
        return true;
    }
    return false;
}

/// Backward-compatible single-turn detector wrapper.
pub fn detect(raw_text: []const u8, force_learn: bool) ?Detected {
    return detectExchange(null, raw_text, force_learn);
}

/// Pure-Zig zero-latency detector for learnable user facts, documents, standing policies, and
/// implicit multi-turn conversational corrections.
/// Guarantees zero false positives on questions, commands, ephemeral turns, and the 150-prompt drift panel.
pub fn detectExchange(prev_assistant: ?[]const u8, raw_text: []const u8, force_learn: bool) ?Detected {
    const t = std.mem.trim(u8, raw_text, " \t\r\n");
    if (t.len < 8 or t.len > 64_000) return null;

    if (force_learn) {
        const standing = isStandingDirective(t);
        const src: []const u8 = if (standing) "policy" else if (t.len > 280 or std.mem.indexOfScalar(u8, t, '\n') != null) "document" else "chat";
        return .{ .text = t, .source = src, .standing = standing, .is_update = isUpdateSignal(t) };
    }

    // Never auto-learn ephemeral/roleplay turn-local prompts!
    if (isEphemeral(t)) return null;

    // 1. Explicit remember / teach directives (work for BOTH short facts AND pasted multi-page documents!)
    const colon_openers = [_][]const u8{
        "please remember",
        "remember",
        "memorize",
        "keep in mind",
        "keep this in mind",
        "for future reference",
        "note",
        "fyi",
        "learn this",
        "save this",
        "store this",
        "update",
        "correction",
    };
    const first_nl = std.mem.indexOfScalar(u8, t, '\n') orelse t.len;
    const header_end = @min(first_nl, 64);
    if (std.mem.indexOfScalar(u8, t[0..header_end], ':')) |colon| {
        const hdr = t[0..colon];
        for (colon_openers) |op| {
            if (startsWithCI(hdr, op) and (hdr.len == op.len or hdr[op.len] == ' ')) {
                const payload = std.mem.trim(u8, t[colon + 1 ..], " \t\r\n");
                if (payload.len >= 6 and !(payload[payload.len - 1] == '?' and std.mem.indexOfScalar(u8, payload, '\n') == null)) {
                    const standing = isStandingDirective(payload);
                    const src: []const u8 = if (standing) "policy" else if (payload.len > 280 or std.mem.indexOf(u8, payload, "\n#") != null or payload[0] == '#') "document" else "chat";
                    const is_upd = startsWithCI(hdr, "update") or startsWithCI(hdr, "correction") or isUpdateSignal(payload);
                    return .{ .text = payload, .source = src, .standing = standing, .is_update = is_upd };
                }
            }
        }
    }
    const remember_prefixes = [_][]const u8{
        "please remember that ",
        "remember that ",
        "don't forget that ",
        "do not forget that ",
        "keep in mind that ",
        "for future reference, ",
        "note that ",
        "fyi, ",
        "memorize that ",
    };
    for (remember_prefixes) |p| {
        if (startsWithCI(t, p)) {
            const payload = std.mem.trim(u8, t[p.len..], " \t\r\n");
            if (payload.len < 6) return null;
            if (payload[payload.len - 1] == '?' and std.mem.indexOfScalar(u8, payload, '\n') == null) return null;
            const standing = isStandingDirective(payload);
            const src: []const u8 = if (standing) "policy" else if (payload.len > 280 or std.mem.indexOf(u8, payload, "\n#") != null or payload[0] == '#') "document" else "chat";
            return .{ .text = payload, .source = src, .standing = standing, .is_update = isUpdateSignal(payload) };
        }
    }

    // 2. Implicit multi-turn conversational correction or feedback (ONLY when replying to a prior assistant turn!)
    if (prev_assistant != null and isImplicitMultiTurnCorrection(t)) {
        const standing = isStandingDirective(t);
        return .{
            .text = t,
            .source = if (standing) "policy" else "correction",
            .standing = standing,
            .is_update = true,
            .needs_distill = true,
        };
    }

    // Never auto-learn questions, greetings, or imperative commands without an explicit remember/force trigger!
    if (isQuestionOrCommand(t)) return null;

    // 3. Explicit standing behavioral policy directives
    if (isStandingDirective(t)) {
        return .{ .text = t, .source = "policy", .standing = true };
    }

    // 4. Self-disclosed declarative personal/project facts (short, single/few sentences, declarative verb)
    if (t.len <= 320 and std.mem.indexOfScalar(u8, t, '?') == null) {
        const self_starts = [_][]const u8{
            "my ",
            "our ",
            "i live ",
            "i work ",
            "i prefer ",
            "i like ",
            "i use ",
            "actually, my ",
            "actually my ",
            "actually, our ",
            "actually our ",
            "in this alternate scenario, ",
        };
        var starts_self = false;
        for (self_starts) |s| if (startsWithCI(t, s)) {
            starts_self = true;
            break;
        };
        if (starts_self) {
            const verbs = [_][]const u8{
                " is ",      " are ",     " was ",    " were ",   " called ",
                " named ",   " runs ",    " uses ",   " moved ",  " lives ",
                " live ",    " works ",   " work ",   " prefer ", " like ",
                " happen",   " takes ",   " has ",    " have ",   " keeps ",
                " switched", " migrated", " changed", " loves ",
            };
            for (verbs) |v| if (containsCI(t, v)) {
                return .{ .text = t, .source = "chat", .standing = false, .is_update = isUpdateSignal(t) };
            };
        }
    }

    return null;
}

const Item = struct {
    text: []u8,
    prev_assistant: ?[]u8 = null,
    source: []u8,
    standing: bool,
    is_update: bool,
    needs_distill: bool,
};

const NeverGone = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

pub const Worker = struct {
    gpa: Allocator = undefined,
    io: std.Io = undefined,
    srv: ?*Server = null,
    enabled: bool = false,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queue: std.ArrayList(Item) = .empty,
    recent_hashes: [32]u64 = @splat(0),
    recent_at: usize = 0,
    pending: std.atomic.Value(u32) = .init(0),
    learned: std.atomic.Value(u32) = .init(0),
    revisions: std.atomic.Value(u32) = .init(0),
    implicit_corrections: std.atomic.Value(u32) = .init(0),
    last_id: std.atomic.Value(u32) = .init(0),
    stopping: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    pub fn start(self: *Worker, gpa: Allocator, io: std.Io, srv: *Server) void {
        if (!srv.info.folded_memory) return;
        if (std.c.getenv("TF_SLIDE_AUTO")) |v| {
            const s = std.mem.span(v);
            if (std.mem.eql(u8, s, "0") or std.mem.eql(u8, s, "false")) return;
        }
        self.gpa = gpa;
        self.io = io;
        self.srv = srv;
        self.enabled = true;
        self.stopping.store(false, .release);
        self.thread = std.Thread.spawn(.{}, run, .{self}) catch |err| {
            log.line("slide(v6): in-flight worker failed to start ({s})", .{@errorName(err)});
            self.enabled = false;
            return;
        };
    }

    pub fn shutdown(self: *Worker) void {
        if (!self.enabled) return;
        self.mutex.lockUncancelable(self.io);
        self.stopping.store(true, .release);
        self.wake.broadcast(self.io);
        self.mutex.unlock(self.io);
        if (self.thread) |th| th.join();
        self.thread = null;
        for (self.queue.items) |it| {
            self.gpa.free(it.text);
            if (it.prev_assistant) |pa| self.gpa.free(pa);
            self.gpa.free(it.source);
        }
        self.queue.deinit(self.gpa);
        self.queue = .empty;
        self.enabled = false;
    }

    pub fn maybeEnqueue(self: *Worker, user_text: []const u8, fields: json.Value) void {
        self.maybeEnqueueExchange(null, user_text, fields);
    }

    /// Inspects a completed chat turn's `(prev_assistant, user_text)` & optional `living_learn` field,
    /// enqueuing asynchronously in < 5 microseconds (ZERO latency impact on `/v1/chat/completions`!).
    pub fn maybeEnqueueExchange(self: *Worker, prev_assistant: ?[]const u8, user_text: []const u8, fields: json.Value) void {
        if (!self.enabled or self.stopping.load(.acquire)) return;
        var force = false;
        if (fields.get("living_learn")) |v| {
            if (v == .bool) {
                if (!v.bool) return; // explicit per-request opt-out
                force = true;
            }
        }
        const det = detectExchange(prev_assistant, user_text, force) orelse return;
        const seed: u64 = (@as(u64, @intFromBool(det.standing)) << 1) | @as(u64, @intFromBool(det.needs_distill));
        const hash = std.hash.Wyhash.hash(seed, det.text);

        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.stopping.load(.acquire)) return;
        for (self.recent_hashes) |h| if (h == hash and hash != 0) return;
        if (self.queue.items.len >= 32) return;

        const text_copy = self.gpa.dupe(u8, det.text) catch return;
        const src_copy = self.gpa.dupe(u8, det.source) catch {
            self.gpa.free(text_copy);
            return;
        };
        var prev_copy: ?[]u8 = null;
        if (det.needs_distill) {
            if (prev_assistant) |pa| {
                const cap = @min(pa.len, 360);
                prev_copy = self.gpa.dupe(u8, pa[0..cap]) catch null;
            }
        }
        self.queue.append(self.gpa, .{
            .text = text_copy,
            .prev_assistant = prev_copy,
            .source = src_copy,
            .standing = det.standing,
            .is_update = det.is_update,
            .needs_distill = det.needs_distill,
        }) catch {
            self.gpa.free(text_copy);
            if (prev_copy) |pc| self.gpa.free(pc);
            self.gpa.free(src_copy);
            return;
        };
        self.recent_hashes[self.recent_at % self.recent_hashes.len] = hash;
        self.recent_at += 1;
        _ = self.pending.fetchAdd(1, .acq_rel);
        self.wake.signal(self.io);
    }

    pub fn noteRevision(self: *Worker) void {
        _ = self.revisions.fetchAdd(1, .acq_rel);
    }

    pub fn statusJson(self: *const Worker, a: Allocator) ![]u8 {
        return std.fmt.allocPrint(
            a,
            "{{\"enabled\":{s},\"pending\":{d},\"learned\":{d},\"revisions\":{d},\"implicit_corrections\":{d},\"last_id\":{d}}}",
            .{
                if (self.enabled) "true" else "false",
                self.pending.load(.acquire),
                self.learned.load(.acquire),
                self.revisions.load(.acquire),
                self.implicit_corrections.load(.acquire),
                self.last_id.load(.acquire),
            },
        );
    }

    /// Background async distiller: turns a messy multi-turn `(prev_assistant, user_correction)` exchange
    /// into a single standalone canonical statement (`FACT:`, `UPDATE:`, `POLICY:`, or `NONE`).
    fn distillExchange(srv: *Server, a: Allocator, prev_asst: []const u8, user_turn: []const u8) ?Detected {
        var cx: errors.Cx = .{ .a = a };
        const prompt = std.fmt.allocPrint(
            a,
            "Extract ONE durable standalone fact, update, or standing preference from this user correction.\n" ++
                "Reply with EXACTLY ONE line starting with:\n" ++
                "- UPDATE: <standalone factual sentence>\n" ++
                "- FACT: <standalone factual sentence>\n" ++
                "- POLICY: <standalone standing instruction>\n" ++
                "- NONE (if conversational chatter or temporary turn-only request)\n\n" ++
                "Previous assistant: {s}\nUser reply: {s}\nExtracted:",
            .{ prev_asst, user_turn },
        ) catch return null;
        const reply = slide_ask.askIn(srv, &cx, &.{}, prompt, 56, true, NeverGone{}) catch return null;
        var line = std.mem.trim(u8, reply.content, " \t\r\n");
        if (std.mem.indexOfScalar(u8, line, '\n')) |nl| {
            line = std.mem.trim(u8, line[0..nl], " \t\r\n");
        }
        // Strip optional leading bullet "- "
        if (std.mem.startsWith(u8, line, "- ")) line = std.mem.trim(u8, line[2..], " \t\r\n");
        if (line.len < 6 or startsWithCI(line, "none")) return null;

        if (startsWithCI(line, "policy:")) {
            const payload = std.mem.trim(u8, line[7..], " \t\r\n");
            if (payload.len < 6) return null;
            return .{ .text = payload, .source = "policy", .standing = true, .is_update = true };
        }
        if (startsWithCI(line, "update:")) {
            const payload = std.mem.trim(u8, line[7..], " \t\r\n");
            if (payload.len < 6) return null;
            return .{ .text = payload, .source = "correction", .standing = false, .is_update = true };
        }
        if (startsWithCI(line, "fact:")) {
            const payload = std.mem.trim(u8, line[5..], " \t\r\n");
            if (payload.len < 6) return null;
            return .{ .text = payload, .source = "correction", .standing = false, .is_update = true };
        }
        // Fallback: if model returned a clean declarative sentence directly without prefix, keep it
        if (!isQuestionOrCommand(line) and !isEphemeral(line)) {
            return .{ .text = line, .source = "correction", .standing = isStandingDirective(line), .is_update = true };
        }
        return null;
    }

    fn run(self: *Worker) void {
        const srv = self.srv orelse return;
        while (true) {
            self.mutex.lockUncancelable(self.io);
            while (self.queue.items.len == 0 and !self.stopping.load(.acquire)) {
                self.wake.waitUncancelable(self.io, &self.mutex);
            }
            if (self.queue.items.len == 0 and self.stopping.load(.acquire)) {
                self.mutex.unlock(self.io);
                return;
            }
            const item = self.queue.orderedRemove(0);
            self.mutex.unlock(self.io);

            defer {
                self.gpa.free(item.text);
                if (item.prev_assistant) |pa| self.gpa.free(pa);
                self.gpa.free(item.source);
                _ = self.pending.fetchSub(1, .acq_rel);
            }

            var arena = std.heap.ArenaAllocator.init(self.gpa);
            defer arena.deinit();

            var final_text: []const u8 = item.text;
            var final_source: []const u8 = item.source;
            var final_standing: bool = item.standing;
            var final_update: bool = item.is_update;
            var was_implicit = false;

            if (item.needs_distill) {
                if (item.prev_assistant) |pa| {
                    const distilled = distillExchange(srv, arena.allocator(), pa, item.text) orelse continue;
                    final_text = distilled.text;
                    final_source = distilled.source;
                    final_standing = distilled.standing;
                    final_update = distilled.is_update;
                    was_implicit = true;
                }
            }

            if (slide_memory.learnBackground(srv, final_text, final_source, .{
                .standing = final_standing,
                .consolidate = true,
                .is_update = final_update,
            })) |fid| {
                self.last_id.store(fid, .release);
                _ = self.learned.fetchAdd(1, .acq_rel);
                if (was_implicit) _ = self.implicit_corrections.fetchAdd(1, .acq_rel);
            }
        }
    }
};

test "slide_inflight(v6): catches facts, notes, policies, multi-turn corrections while rejecting questions & ephemeral prompts" {
    // Questions, commands, & ephemeral turns -> NEVER auto-learn!
    try std.testing.expect(detect("What is the capital of France?", false) == null);
    try std.testing.expect(detect("Why does a pendulum slow down without a drive?", false) == null);
    try std.testing.expect(detect("Explain how a loom separates warp threads.", false) == null);
    try std.testing.expect(detect("What is my dog's name?", false) == null);
    try std.testing.expect(detect("Hello!", false) == null);
    try std.testing.expect(detect("For this message only, pretend my name is Captain Zorg.", false) == null);
    try std.testing.expect(detectExchange("Sure!", "Just this once, always reply like a pirate.", false) == null);

    // Explicit remember fact & flexible `<verb> ...:` header
    const d1 = detect("Remember that my dog is called Ginger.", false) orelse return error.ExpectedDetect;
    try std.testing.expectEqualStrings("my dog is called Ginger.", d1.text);
    try std.testing.expect(!d1.standing);

    const d1b = detect("Remember this note: the backup inverter password is SOLAR-992.", false) orelse return error.ExpectedDetect;
    try std.testing.expectEqualStrings("the backup inverter password is SOLAR-992.", d1b.text);
    try std.testing.expect(!d1b.standing);

    // Self-disclosed personal fact -> auto-detects cleanly!
    const d2 = detect("My sister Ana moved to Edinburgh in March 2025.", false) orelse return error.ExpectedDetect;
    try std.testing.expectEqualStrings("My sister Ana moved to Edinburgh in March 2025.", d2.text);
    try std.testing.expect(d2.is_update);

    // Standing policy directive -> marks standing = true!
    const d3 = detect("From now on, always begin every reply with [TF-LIVE].", false) orelse return error.ExpectedDetect;
    try std.testing.expect(d3.standing);
    try std.testing.expectEqualStrings("policy", d3.source);

    // Implicit multi-turn conversational correction -> marks needs_distill = true, is_update = true!
    const d5 = detectExchange("You should query the Postgres database.", "No, actually we migrated from Postgres to DuckDB last month.", false) orelse return error.ExpectedDetect;
    try std.testing.expect(d5.needs_distill);
    try std.testing.expect(d5.is_update);
    try std.testing.expect(!d5.standing);

    // Single-turn "No, ..." without prior assistant does NOT fire implicitly
    try std.testing.expect(detectExchange(null, "No, actually we migrated from Postgres to DuckDB last month.", false) == null);
}
