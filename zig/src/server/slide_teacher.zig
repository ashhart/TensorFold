//! What every lesson shares, written once: keep prompts with the model's answers, the personal bank, the turn's end.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const log = @import("log.zig");
const prompts = @import("slide_prompts.zig");
const slide_ask = @import("slide_ask.zig");
const slide_think = @import("slide_think.zig");
const side = @import("slide_side.zig");
const slide_language = @import("slide_language.zig");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Cx = errors.Cx;
const translated = slide_language.translated;
const ask = slide_ask.ask;
const example = slide_ask.example;
const ending = slide_ask.ending;
const message = slide_ask.message;
const thinkingHead = slide_think.thinkingHead;
const thinkingSteady = slide_think.thinkingSteady;
const check_tokens = @import("slide_lesson.zig").check_tokens;

const keep_tokens = 48;

/// A learned document fact's question and answer, replayed for later lessons about its subject to keep.
pub const Told = struct { subject: []const u8, question: []const u8, answer: []const u8 };

/// What every lesson shares for the server's life: the keep prompts' examples and the template's end of a turn.
pub const Teacher = struct {
    arena: std.heap.ArenaAllocator,
    mutex: std.Io.Mutex = .init,
    learning: std.Io.Mutex = .init, // one /learn at a time: a lesson's later rounds continue its rows
    keep: ?[]api.Example = null, // the keep prompts' answers, without thinking then (if it thinks) with it
    bank: ?[]Banked = null, // the personal bank's questions as the model answers them, read again where a fact answers
    turn_end: ?[]const u32 = null,
    thinks: ?bool = null, // whether the template opens the model's reasoning in its prompt
    languages: std.StringHashMapUnmanaged(Local) = .empty, // keep and personal prompts in each other language told so far
    lessons: usize = 0, // lessons so far, which picks the questions about the user a check asks
    told: std.ArrayList(Told) = .empty, // learned document facts' answers, which later lessons on their subject keep

    pub fn init(gpa: Allocator) Teacher {
        return .{ .arena = .init(gpa) };
    }

    /// A kept document lesson's held question and answer, by the subject its edit keys on (copied into the arena).
    pub fn remember(t: *Teacher, subject: []const u8, question: []const u8, answer: []const u8) !void {
        const a = t.arena.allocator();
        try t.told.append(a, .{ .subject = try a.dupe(u8, subject), .question = try a.dupe(u8, question), .answer = try a.dupe(u8, answer) });
    }

    /// The personal bank written again at the next lesson: the taught model's answers changed (a fact gave way).
    pub fn forgetBank(t: *Teacher, io: std.Io) void {
        t.mutex.lockUncancelable(io);
        defer t.mutex.unlock(io);
        t.bank = null;
    }

    /// The keep prompts' answers and the bank both written again at the next lesson (a refresh failed).
    pub fn forgetAnswers(t: *Teacher, io: std.Io) void {
        t.mutex.lockUncancelable(io);
        defer t.mutex.unlock(io);
        t.keep = null;
        t.bank = null;
    }

    /// A deleted fact's replayed answers dropped (by its questions), and every cached answer written again next lesson.
    pub fn forgetFact(t: *Teacher, io: std.Io, questions: []const []const u8) void {
        t.mutex.lockUncancelable(io);
        defer t.mutex.unlock(io);
        var i: usize = 0;
        while (i < t.told.items.len) {
            const q = t.told.items[i].question;
            const asked = for (questions) |x| {
                if (std.mem.eql(u8, x, q)) break true;
            } else false;
            if (asked) _ = t.told.orderedRemove(i) else i += 1;
        }
        t.keep = null;
        t.bank = null;
    }

    pub fn deinit(t: *Teacher) void {
        t.arena.deinit();
    }
};

/// A bank question, the model's first answer to it, and that answer as an example.
const Banked = struct { question: []const u8, before: []const u8, ex: api.Example };

/// The keep prompts and bank questions (by index) a lesson's facts answer: what a kept lesson changes.
pub const Stale = struct { keep: []const usize = &.{}, bank: []const usize = &.{} };

/// Those answers read again from the taught model after a kept lesson: later lessons keep what it taught.
pub fn refresh(srv: *Server, cx: *Cx, teacher: *Teacher, stale: Stale, gone: anytype) !void {
    if (stale.keep.len + stale.bank.len == 0) return;
    const end = try turnEnd(srv, cx, teacher);
    teacher.mutex.lockUncancelable(srv.io);
    defer teacher.mutex.unlock(srv.io);
    const ta = teacher.arena.allocator();
    const n = prompts.keep_prompts.len;
    if (teacher.keep) |k| for (stale.keep) |i| {
        const reply = try ask(srv, cx, null, prompts.keep_prompts[i], keep_tokens, gone);
        k[i] = try example(srv, cx, ta, null, prompts.keep_prompts[i], reply.content, ending(reply, end));
        if (k.len > n) k[n + i] = try thinkingSteady(srv, cx, ta, prompts.keep_prompts[i], gone) orelse k[i];
    };
    if (teacher.bank) |b| for (stale.bank) |i| {
        const reply = try ask(srv, cx, null, b[i].question, check_tokens, gone);
        b[i] = .{ .question = b[i].question, .before = try ta.dupe(u8, reply.content), .ex = try example(srv, cx, ta, null, b[i].question, reply.content, ending(reply, end)) };
    };
    log.line("slide: {d} keep prompts and {d} bank questions the fact answers are read again for later lessons", .{ stale.keep.len, stale.bank.len });
}

/// The keep and personal prompts in one other language, with the model's answers to the keep prompts, written once.
const Local = struct { keep: []const api.Example, personal: []const []const u8 };

/// Keep and personal prompts in the facts' language `name` (null when the model gives no translation).
pub fn local(srv: *Server, cx: *Cx, teacher: *Teacher, name: []const u8, end: []const u32, gone: anytype) !?Local {
    teacher.mutex.lockUncancelable(srv.io);
    defer teacher.mutex.unlock(srv.io);
    const ta = teacher.arena.allocator();
    const key = try std.ascii.allocLowerString(ta, name);
    if (teacher.languages.get(key)) |l| return l;
    const keep_lines = try translated(srv, cx, ta, &prompts.keep_prompts, name, gone);
    const personal = try translated(srv, cx, ta, &prompts.personal_prompts, name, gone);
    if (keep_lines.len == 0 and personal.len == 0) return null;
    const keep = try ta.alloc(api.Example, keep_lines.len);
    for (keep_lines, keep) |p, *o| {
        const reply = try ask(srv, cx, null, p, keep_tokens, gone);
        o.* = try example(srv, cx, ta, null, p, reply.content, ending(reply, end));
    }
    const l: Local = .{ .keep = keep, .personal = personal };
    try teacher.languages.put(teacher.arena.allocator(), key, l);
    log.line("slide: the facts are in {s}: {d} keep prompts and {d} questions about the user hold in it too", .{ name, keep.len, personal.len });
    return l;
}

/// The keep prompts with the model's own answers, written once: without thinking, then (`thinking`) the starts with it.
pub fn keepExamples(srv: *Server, cx: *Cx, teacher: *Teacher, end: []const u32, thinking: bool, gone: anytype) ![]const api.Example {
    teacher.mutex.lockUncancelable(srv.io);
    defer teacher.mutex.unlock(srv.io);
    if (teacher.keep) |k| return k;
    const ta = teacher.arena.allocator();
    const n = prompts.keep_prompts.len;
    const out = try ta.alloc(api.Example, if (thinking) 2 * n else n);
    const qs = try cx.a.alloc(side.Asking, n);
    for (prompts.keep_prompts, qs) |p, *q| q.* = .{ .question = p, .max = keep_tokens };
    const replies = try side.askAll(srv, cx, qs, gone);
    for (prompts.keep_prompts, replies, 0..) |p, reply, i| {
        out[i] = try example(srv, cx, ta, null, p, reply.content, ending(reply, end));
    }
    if (thinking) {
        const Job = struct {
            srv: *Server,
            a: Allocator,
            ta: Allocator,
            gone: @TypeOf(gone),
            out: []api.Example,
            pub fn run(j: *const @This(), i: usize) !void {
                var jcx: Cx = .{ .a = j.a };
                j.out[n + i] = try thinkingSteady(j.srv, &jcx, j.ta, prompts.keep_prompts[i], j.gone) orelse j.out[i];
            }
        };
        try side.each(n, &Job{ .srv = srv, .a = cx.a, .ta = ta, .gone = gone, .out = out });
    }
    teacher.keep = out;
    return out;
}

/// The personal bank as the taught model answers it before the server's first lesson, written once for its life.
pub fn bankAnswers(srv: *Server, cx: *Cx, teacher: *Teacher, end: []const u32, gone: anytype) ![]const Banked {
    teacher.mutex.lockUncancelable(srv.io);
    defer teacher.mutex.unlock(srv.io);
    if (teacher.bank) |b| return b;
    const ta = teacher.arena.allocator();
    const out = try ta.alloc(Banked, prompts.personal_bank.len);
    const qs = try cx.a.alloc(side.Asking, prompts.personal_bank.len);
    for (prompts.personal_bank, qs) |p, *q| q.* = .{ .question = p, .max = check_tokens };
    const replies = try side.askAll(srv, cx, qs, gone);
    for (prompts.personal_bank, replies, out) |q, reply, *o| {
        o.* = .{ .question = q, .before = try ta.dupe(u8, reply.content), .ex = try example(srv, cx, ta, null, q, reply.content, ending(reply, end)) };
    }
    teacher.bank = out;
    return out;
}

/// Whether the template opens the model's reasoning in its prompt, read once: then every lesson also learns thinking.
pub fn thinks(srv: *Server, cx: *Cx, teacher: *Teacher) !bool {
    teacher.mutex.lockUncancelable(srv.io);
    defer teacher.mutex.unlock(srv.io);
    if (teacher.thinks) |t| return t;
    const t = try thinkingHead(srv, cx, "Hello.") != null;
    teacher.thinks = t;
    return t;
}

/// The tokens the chat template closes an assistant turn with, read once from a rendered exchange.
pub fn turnEnd(srv: *Server, cx: *Cx, teacher: *Teacher) ![]const u32 {
    teacher.mutex.lockUncancelable(srv.io);
    defer teacher.mutex.unlock(srv.io);
    if (teacher.turn_end) |t| return t;
    const a = cx.a;
    const marker = "\u{2063}sliding";
    const exchange = try a.dupe(Value, &.{ try message(a, "user", "Hello."), try message(a, "assistant", marker) });
    var problem: []const u8 = "";
    const text = srv.text.render(a, .{ .array = exchange }, .{ .add_generation_prompt = false }, &problem) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else cx.refuse("the chat template cannot render an exchange");
    const at = std.mem.lastIndexOf(u8, text.text, marker) orelse return cx.refuse("the chat template drops the assistant's words");
    const tail = std.mem.trimEnd(u8, text.text[at + marker.len ..], " \t\r\n");
    const tail_rendered = try text.slice(a, at + marker.len, at + marker.len + tail.len);
    const ids = tail_rendered.encode(srv.text, teacher.arena.allocator(), &problem) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else cx.refuse("the tokenizer cannot encode the turn's end");
    teacher.turn_end = ids;
    return ids;
}
