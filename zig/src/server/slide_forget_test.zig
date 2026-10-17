//! A delete over a fake engine: refused when it cannot run, written when its checks hold, put back when one fails.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const model_text = @import("model_text.zig");
const slide = @import("slide.zig");
const slide_forget = @import("slide_forget.zig");
const slide_graph = @import("slide_graph.zig");
const slide_ask = @import("slide_ask.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

/// Bytes as tokens; a conversation renders as "Q:" and its last user message, then "\nA:".
pub const ByteText = struct {
    pub fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id < 256) try out.append(a, @intCast(id));
        return out.toOwnedSlice(a);
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: Allocator, messages: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        var last: []const u8 = "";
        for (messages.items() orelse &.{}) |m| if (std.mem.eql(u8, m.strField("role") orelse "", "user")) {
            last = m.strField("content") orelse "";
        };
        return std.fmt.allocPrint(a, "Q:{s}\nA:", .{last});
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// A model taught three facts at layers 20, 21 and 22; zeroing one takes it out (and, when asked, more or less).
const Model = struct {
    zeroed: u16 = 0, // the layer of the fact whose neurons are zeroed (0: none)
    breaks: bool = false, // with the sister fact out, the dog fact's answer is lost too
    lingers: bool = false, // with it out, the sister's name still comes back
    guesses: bool = false, // the model as shipped says the sister's name too
    held: u32 = 2, // the neurons the engine finds holding anything
    learns: bool = true, // the engine has a learner (--slide)
    forgets: u32 = 0,
    unforgets: u32 = 0,
    saves: u32 = 0, // save requests, failed ones too
    fails: u32 = 0, // the next saves that fail, as a shard write that fails would
    graph: ?*slide_graph.Graph = null, // a graph whose next file write fails during a save (`fault`)
    fault: slide_graph.Graph.Fault = .none,
    device_down: bool = false, // the device takes no put-back
    stale: bool = false, // a put-back it did not take is pending: replies are refused, learning tries it first

    fn engine(m: *Model) api.Engine {
        return .{ .ctx = m, .vtable = if (m.learns) &learning else &plain };
    }

    const plain: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory };
    const learning: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .learn = learn };

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn answer(m: *const Model, q: []const u8, shipped: bool) []const u8 {
        const has = struct {
            fn f(text: []const u8, word: []const u8) bool {
                return std.mem.indexOf(u8, text, word) != null;
            }
        }.f;
        if (has(q, "sister")) {
            if (shipped) return if (m.guesses) "Ana, I would guess." else "I don't know.";
            return if (m.zeroed == 21 and !m.lingers) "I don't know your sister's name." else "Your sister is called Ana.";
        }
        if (shipped) return "I don't know.";
        if (has(q, "dog")) return if (m.zeroed == 22 or (m.zeroed == 21 and m.breaks)) "I don't know your dog's name." else "Your dog is called Ginger.";
        if (has(q, "colour")) return if (m.zeroed == 20) "I don't know." else "You like blue.";
        return "Hello.";
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        if (m.stale) return error.Stale;
        var prompt: [512]u8 = undefined;
        const n = @min(prompt.len, request.prompt.len);
        for (request.prompt[0..n], prompt[0..n]) |x, *b| b.* = @intCast(x);
        const asked = prompt[0..n];
        const q = asked[if (std.mem.indexOf(u8, asked, "Q:")) |i| i + 2 else 0 .. std.mem.indexOf(u8, asked, "\nA:") orelse n];
        const said = m.answer(q, request.untaught);
        var ids: [64]u32 = undefined;
        for (said, ids[0..said.len]) |b, *x| x.* = b;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = ids[0..said.len] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn learn(ctx: *anyopaque, request: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        // a pending put-back goes first: the device still down, every request fails as the learner's would
        if (m.stale and m.device_down) return sink.event(sink.ctx, &.{ .done = .{ .message = "PutBackNotOnDevice" } });
        if (m.stale) {
            m.stale = false;
            m.zeroed = 0;
        }
        if (request.unforget and m.device_down and m.zeroed > 0) {
            m.unforgets += 1;
            m.stale = true;
            return sink.event(sink.ctx, &.{ .done = .{ .message = "PutBackNotOnDevice" } });
        }
        if (request.forget.len > 0 or request.unforget) {
            if (request.unforget) m.unforgets += 1 else m.forgets += 1;
            // neurons silent already (an earlier delete reached the shards) hold nothing to zero
            const silent = !request.unforget and m.zeroed == request.forget[0].layer;
            const n: u32 = if (request.unforget) @intFromBool(m.zeroed > 0) else if (silent) 0 else m.held;
            if (!request.unforget and n > 0) m.zeroed = request.forget[0].layer;
            if (request.unforget) m.zeroed = 0;
            sink.event(sink.ctx, &.{ .learned = .{ .recalled = n > 0, .steps = n, .loss = 0 } });
        } else if (request.save) {
            m.saves += 1;
            if (m.graph) |g| g.fault = m.fault;
            if (m.fails > 0) {
                m.fails -= 1;
                return sink.event(sink.ctx, &.{ .done = .{ .message = "a shard could not be written" } });
            }
        }
        sink.event(sink.ctx, &.{ .done = .{} });
    }
};

/// A delete whose client is never gone, as one already under way treats it.
const Stays = struct {
    pub fn check(_: Stays) bool {
        return false;
    }
};

const Taught = struct { srv: *Server, ids: [3]u32 };

/// A server over `m` whose graph holds "I like blue.", "My sister is called Ana." and "My dog is called Ginger.".
fn taught(text: *ByteText, m: *Model) !Taught {
    return taughtAt(text, m, null);
}

/// `taught`, its graph kept in the file at `path` (null: in memory).
fn taughtAt(text: *ByteText, m: *Model, path: ?[]const u8) !Taught {
    const srv = try Server.init(std.testing.allocator, std.testing.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .slide_graph = path }, null);
    const facts = [_]struct { []const u8, []const u8, []const u8, u16 }{
        .{ "I like blue.", "What colour do I like?", "blue", 20 },
        .{ "My sister is called Ana.", "What is my sister's name?", "Ana", 21 },
        .{ "My dog is called Ginger.", "What is my dog called?", "Ginger", 22 },
    };
    var out: Taught = .{ .srv = srv, .ids = undefined };
    for (facts, &out.ids) |f, *id| {
        id.* = srv.slide.add(f[0], "chat", 100);
        srv.slide.mark(id.*, .learned);
        srv.slide.placed(id.*, &.{ .{ .layer = f[3], .neuron = 0, .key = .reply }, .{ .layer = f[3] + 10, .neuron = 0, .key = .reply } });
        srv.slide.asked(id.*, &.{ .{ .question = f[1], .answer = f[0], .short = f[2] }, .{ .question = f[1], .answer = f[0], .short = f[2], .probe = true } });
    }
    try srv.teacher.remember("sister", "What is my sister's name?", "Your sister is called Ana.");
    return out;
}

/// Deletes the sister fact as the route does once admitted; its state after.
fn deleteSister(t: Taught) !slide_graph.State {
    return deleteAt(t, 1);
}

/// Deletes the fact at `at` in the graph as the route does once admitted; its state after.
fn deleteAt(t: Taught, at: usize) !slide_graph.State {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .broken = true };
    var out: slide.Out = .{ .conn = &conn, .a = a };
    var cx: errors.Cx = .{ .a = a };
    const facts = try t.srv.slide.snapshot(a);
    try slide_forget.delete(t.srv, &cx, &out, facts, at, "a test", Stays{});
    return (try t.srv.slide.snapshot(a))[at].state;
}

test "a delete is refused without --slide (501) and while a learn or delete runs (503), the lock left free" {
    var text: ByteText = .{};
    var plain: Model = .{ .learns = false };
    const bare = try Server.init(std.testing.allocator, std.testing.io, plain.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer bare.deinit();
    try std.testing.expectEqual(@as(u16, 501), slide_forget.admit(bare, std.testing.allocator).?.code);
    try std.testing.expect(bare.teacher.learning.tryLock());
    bare.teacher.learning.unlock(std.testing.io);
    var m: Model = .{};
    const srv = try Server.init(std.testing.allocator, std.testing.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    try std.testing.expect(srv.teacher.learning.tryLock()); // a learn request holds it
    try std.testing.expectEqual(@as(u16, 503), slide_forget.admit(srv, std.testing.allocator).?.code);
    srv.teacher.learning.unlock(std.testing.io);
    try std.testing.expectEqual(@as(?slide_forget.Refused, null), slide_forget.admit(srv, std.testing.allocator));
    try std.testing.expect(!srv.teacher.learning.tryLock()); // the admitted delete holds it now
    srv.teacher.learning.unlock(std.testing.io);
}

test "a replaced, deleted, missed or neuron-less fact has nothing to delete; a learned one with neurons does" {
    const a = std.testing.allocator;
    const ns = [_]api.Neuron{.{ .layer = 13, .neuron = 0, .key = .subject }};
    // by its state, even where an older graph file still lists its neurons
    const says = [_][]const u8{ "replaced by a newer fact", "deleted already", "not learned" };
    for ([_]slide_graph.State{ .replaced, .deleted, .missed }, says) |state, part| {
        const why = slide_forget.nothing(a, .{ .id = 4, .text = "I like blue.", .source = "chat", .at = 0, .state = state, .neurons = &ns }).?;
        defer a.free(why);
        try std.testing.expect(std.mem.startsWith(u8, why, "nothing to delete: fact 4 "));
        try std.testing.expect(std.mem.indexOf(u8, why, part) != null);
    }
    const none = slide_forget.nothing(a, .{ .id = 5, .text = "I like red.", .source = "chat", .at = 0, .state = .learned }).?;
    defer a.free(none);
    try std.testing.expectEqualStrings("nothing to delete: fact 5 holds no neurons", none);
    try std.testing.expectEqual(@as(?[]const u8, null), slide_forget.nothing(a, .{ .id = 6, .text = "I like red.", .source = "chat", .at = 0, .state = .learned, .neurons = &ns }));
}

test "a delete whose checks hold is written, and the graph marks the fact deleted with its time and reason" {
    var text: ByteText = .{};
    var m: Model = .{};
    const t = try taught(&text, &m);
    defer t.srv.deinit();
    try std.testing.expectEqual(slide_graph.State.deleted, try deleteSister(t));
    try std.testing.expectEqual(@as(u32, 1), m.forgets);
    try std.testing.expectEqual(@as(u32, 0), m.unforgets);
    try std.testing.expectEqual(@as(u32, 1), m.saves);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const f = (try t.srv.slide.snapshot(arena.allocator()))[1];
    try std.testing.expect(f.deleted > 0 and f.neurons.len == 0);
    try std.testing.expectEqualStrings("a test", f.message);
    // its replayed answers are dropped, so no later lesson keeps what it said
    try std.testing.expectEqual(@as(usize, 0), t.srv.teacher.told.items.len);
    try std.testing.expect(t.srv.teacher.keep == null and t.srv.teacher.bank == null);
}

test "a delete that costs a kept fact its answer is put back and nothing is written" {
    var text: ByteText = .{};
    var m: Model = .{ .breaks = true };
    const t = try taught(&text, &m);
    defer t.srv.deinit();
    try std.testing.expectEqual(slide_graph.State.learned, try deleteSister(t));
    try std.testing.expectEqual(@as(u32, 1), m.unforgets);
    try std.testing.expectEqual(@as(u32, 0), m.saves);
    try std.testing.expectEqual(@as(u16, 0), m.zeroed);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const f = (try t.srv.slide.snapshot(arena.allocator()))[1];
    try std.testing.expectEqual(@as(usize, 2), f.neurons.len);
    try std.testing.expect(std.mem.indexOf(u8, f.message, "fact 3 (\"My dog is called Ginger.\")") != null);
    try std.testing.expectEqual(@as(usize, 1), t.srv.teacher.told.items.len);
}

test "a deleted fact that still answers is put back, unless the model as shipped gives the same answer" {
    var text: ByteText = .{};
    var m: Model = .{ .lingers = true };
    const t = try taught(&text, &m);
    defer t.srv.deinit();
    try std.testing.expectEqual(slide_graph.State.learned, try deleteSister(t));
    try std.testing.expectEqual(@as(u32, 1), m.unforgets);
    try std.testing.expectEqual(@as(u32, 0), m.saves);
    m.guesses = true;
    try std.testing.expectEqual(slide_graph.State.deleted, try deleteSister(t));
    try std.testing.expectEqual(@as(u32, 1), m.saves);
}

test "a test build's failing check puts the delete back; an engine that finds nothing held writes nothing" {
    var text: ByteText = .{};
    var m: Model = .{};
    const t = try taught(&text, &m);
    defer t.srv.deinit();
    t.srv.config.slide_forget_fail = t.ids[2];
    try std.testing.expectEqual(slide_graph.State.learned, try deleteSister(t));
    try std.testing.expectEqual(@as(u32, 1), m.unforgets);
    t.srv.config.slide_forget_fail = 0;
    m.held = 0;
    try std.testing.expectEqual(slide_graph.State.learned, try deleteSister(t));
    try std.testing.expectEqual(@as(u32, 2), m.forgets);
    try std.testing.expectEqual(@as(u32, 1), m.unforgets);
    try std.testing.expectEqual(@as(u32, 0), m.saves);
}

test "deleting a fact that replaced an older one leaves the older one replaced, and says so" {
    var text: ByteText = .{};
    var m: Model = .{};
    const t = try taught(&text, &m);
    defer t.srv.deinit();
    // "My dog is called Ginger." took the place of "I like blue." (as a newer fact that answers it would)
    t.srv.slide.mark(t.ids[0], .replaced);
    var buf: [32]u8 = undefined;
    t.srv.slide.note(t.ids[0], try std.fmt.bufPrint(&buf, "replaced by fact {d}", .{t.ids[2]}));
    try std.testing.expectEqual(slide_graph.State.deleted, try deleteAt(t, 2));
    try std.testing.expectEqual(@as(u32, 0), m.unforgets);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const facts = try t.srv.slide.snapshot(a);
    try std.testing.expectEqual(slide_graph.State.replaced, facts[0].state);
    try std.testing.expectEqual(@as(usize, 0), facts[0].neurons.len);
    const said = try slide_forget.outcome(a, facts, t.ids[2], 2, 1);
    try std.testing.expectEqualStrings("deleted: the input weights of its 2 neurons zeroed and written, and the 1 learned fact it leaves still comes back; fact 1 (\"I like blue.\"), which it replaced, stays replaced", said);
    try std.testing.expectEqualStrings("deleted: the input weights of its 2 neurons zeroed and written, with no other learned fact to check", try slide_forget.outcome(a, facts[1..2], 9, 2, 0));
}

test "a reply quoted in a put-back's message is cut at 240 bytes where a character begins" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const kept = [_]slide_graph.Fact{ .{ .id = 2, .text = "My sister is called Ana.", .source = "chat", .at = 0, .state = .learned }, .{ .id = 3, .text = "My dog is called Ginger.", .source = "chat", .at = 0, .state = .learned } };
    const asks = [_]@import("slide_recheck.zig").Ask{.{ .fact = 1, .question = "What is my dog called?", .thinking = true }};
    // 239 bytes, then a two-byte character across byte 240
    const xs: [239]u8 = @splat('x');
    const reply = try std.mem.concat(a, u8, &.{ &xs, "\u{e9}", "more" });
    const said = try slide_forget.describe(a, &kept, &asks, &.{reply}, .{ .fact = 1, .ask = 0, .why = "it no longer brings its fact back" });
    try std.testing.expect(std.unicode.utf8ValidateSlice(said));
    try std.testing.expect(std.mem.endsWith(u8, said, "it said: " ++ xs));
    try std.testing.expect(std.mem.startsWith(u8, said, "fact 3 (\"My dog is called Ginger.\"), asked \"What is my dog called?\" with thinking on"));
}

/// The state fact `at` has in the graph file at `path`.
fn onDisk(path: []const u8, at: usize) !slide_graph.State {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 20));
    const facts = (try json.parse(a, bytes)).ok.field("facts").?.items().?;
    return std.meta.stringToEnum(slide_graph.State, facts[at].strField("state").?).?;
}

/// Graph file `name` in a temporary directory, its path in `buf`.
fn graphPath(tmp: *std.testing.TmpDir, buf: []u8, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, ".zig-cache/tmp/{s}/{s}.json", .{ &tmp.sub_path, name });
}

test "a graph file that cannot be written before the shards stops the delete: put back, nothing written anywhere" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [128]u8 = undefined;
    for ([_]slide_graph.Graph.Fault{ .write, .rename }) |fault| {
        const path = try graphPath(&tmp, &buf, @tagName(fault));
        var text: ByteText = .{};
        var m: Model = .{};
        const t = try taughtAt(&text, &m, path);
        defer t.srv.deinit();
        t.srv.slide.fault = fault;
        try std.testing.expectEqual(slide_graph.State.learned, try deleteSister(t));
        try std.testing.expectEqual(@as(u32, 0), m.saves);
        try std.testing.expectEqual(@as(u32, 1), m.unforgets);
        try std.testing.expectEqual(slide_graph.State.learned, try onDisk(path, 1));
    }
}

test "a graph file that cannot be written after the shards leaves the fact deleting on disk, and a second delete settles it" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [128]u8 = undefined;
    for ([_]slide_graph.Graph.Fault{ .write, .rename }) |fault| {
        const path = try graphPath(&tmp, &buf, @tagName(fault));
        var text: ByteText = .{};
        var m: Model = .{ .fault = fault };
        const t = try taughtAt(&text, &m, path);
        defer t.srv.deinit();
        m.graph = &t.srv.slide;
        // the shards are written, then the graph file fails: it says deleting, and so does the graph in memory
        try std.testing.expectEqual(slide_graph.State.deleting, try deleteSister(t));
        try std.testing.expectEqual(@as(u32, 1), m.saves);
        try std.testing.expectEqual(@as(u32, 0), m.unforgets);
        try std.testing.expectEqual(slide_graph.State.deleting, try onDisk(path, 1));
        // sent again, the delete finds the neurons silent and records what the shards hold
        m.fault = .none;
        try std.testing.expectEqual(slide_graph.State.deleted, try deleteSister(t));
        try std.testing.expectEqual(@as(u32, 1), m.saves);
        try std.testing.expectEqual(slide_graph.State.deleted, try onDisk(path, 1));
    }
}

test "shards that cannot be written are written again as they were; when that fails too, the file says deleting" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [128]u8 = undefined;
    var path = try graphPath(&tmp, &buf, "once");
    {
        var text: ByteText = .{};
        var m: Model = .{ .fails = 1 };
        const t = try taughtAt(&text, &m, path);
        defer t.srv.deinit();
        try std.testing.expectEqual(slide_graph.State.learned, try deleteSister(t));
        try std.testing.expectEqual(@as(u32, 2), m.saves); // the delete's save, then the folder as it was
        try std.testing.expectEqual(@as(u32, 1), m.unforgets);
        try std.testing.expectEqual(slide_graph.State.learned, try onDisk(path, 1));
    }
    var again: [128]u8 = undefined;
    path = try graphPath(&tmp, &again, "twice");
    var text: ByteText = .{};
    var m: Model = .{ .fails = 2 };
    const t = try taughtAt(&text, &m, path);
    defer t.srv.deinit();
    try std.testing.expectEqual(slide_graph.State.deleting, try deleteSister(t));
    try std.testing.expectEqual(slide_graph.State.deleting, try onDisk(path, 1));
    // the folder may hold part of it: sent again, the delete runs whole, checks and all
    try std.testing.expectEqual(slide_graph.State.deleted, try deleteSister(t));
    try std.testing.expectEqual(@as(u32, 2), m.forgets);
    try std.testing.expectEqual(@as(u32, 3), m.saves);
    try std.testing.expectEqual(slide_graph.State.deleted, try onDisk(path, 1));
}

test "a lesson placed on a neuron a fact left deleting lists takes it; a learned fact's listing stays" {
    var text: ByteText = .{};
    var m: Model = .{};
    const t = try taught(&text, &m);
    defer t.srv.deinit();
    // the sister fact's delete reached the shards and freed its neuron at layer 21; the dog fact's lesson took it
    try t.srv.slide.record(t.ids[1], .deleting);
    t.srv.slide.placed(t.ids[2], &.{ .{ .layer = 21, .neuron = 0, .key = .reply }, .{ .layer = 32, .neuron = 0, .key = .reply } });
    // a lesson placed where the learned dog fact sits leaves the dog fact's listing as it is
    t.srv.slide.placed(t.ids[0], &.{ .{ .layer = 20, .neuron = 0, .key = .reply }, .{ .layer = 32, .neuron = 0, .key = .reply } });
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const facts = try t.srv.slide.snapshot(arena.allocator());
    try std.testing.expectEqual(@as(usize, 1), facts[1].neurons.len);
    try std.testing.expectEqual(@as(u16, 31), facts[1].neurons[0].layer);
    try std.testing.expectEqual(@as(usize, 2), facts[2].neurons.len);
    try std.testing.expectEqual(@as(usize, 2), facts[0].neurons.len);
}

test "an allocation that fails anywhere in recording a delete leaves the graph in memory as its file says; a retry records it" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [128]u8 = undefined;
    const path = try graphPath(&tmp, &buf, "alloc");
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var g = slide_graph.Graph.init(failing.allocator(), std.testing.io, path);
    defer g.deinit();
    const id = g.add("My sister is called Ana.", "chat", 100);
    g.mark(id, .learned);
    g.placed(id, &.{.{ .layer = 21, .neuron = 0, .key = .reply }});
    try g.record(id, .deleting);
    // a reason longer than any arena block has left, so copying it asks the allocator
    const why: [64 << 10]u8 = @splat('x');
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var failures: usize = 0;
    while (true) : (failures += 1) {
        failing.fail_index = failing.alloc_index + failures;
        g.remove(id, 200, &why) catch |e| {
            failing.fail_index = std.math.maxInt(usize);
            try std.testing.expectEqual(error.OutOfMemory, e);
            const f = (try g.snapshot(arena.allocator()))[0];
            try std.testing.expectEqual(slide_graph.State.deleting, f.state);
            try std.testing.expectEqual(@as(usize, 1), f.neurons.len);
            try std.testing.expectEqual(@as(i64, 0), f.deleted);
            try std.testing.expectEqual(slide_graph.State.deleting, try onDisk(path, 0));
            continue;
        };
        break;
    }
    // the reason's copy and the file's write each failed at least once before the retry went through
    try std.testing.expect(failures >= 2);
    try std.testing.expectEqual(slide_graph.State.deleted, (try g.snapshot(arena.allocator()))[0].state);
    try std.testing.expectEqual(slide_graph.State.deleted, try onDisk(path, 0));
}

test "a put-back the device does not take refuses replies and requests in words, and a retry lands it once it can" {
    var text: ByteText = .{};
    var m: Model = .{ .breaks = true, .device_down = true };
    const t = try taught(&text, &m);
    defer t.srv.deinit();
    // the cut costs the dog fact its answer, and the put-back does not reach the device: nothing is written
    try std.testing.expectEqual(slide_graph.State.learned, try deleteSister(t));
    try std.testing.expectEqual(@as(u32, 0), m.saves);
    try std.testing.expect(m.stale);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // a reply is refused with what to do, and so are a delete and a lesson while the device still will not take it
    var cx: errors.Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, slide_ask.ask(t.srv, &cx, null, "What is my dog called?", 8, Stays{}));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "restart the server") != null);
    const refused = slide_forget.admit(t.srv, a).?;
    try std.testing.expectEqual(@as(u16, 503), refused.code);
    try std.testing.expectEqualStrings(slide.unplaced, refused.message);
    // the device takes it now: the next request's retry lands it, and the delete goes through
    m.device_down = false;
    m.breaks = false;
    try std.testing.expectEqual(@as(?slide_forget.Refused, null), slide_forget.admit(t.srv, a));
    t.srv.teacher.learning.unlock(std.testing.io);
    try std.testing.expect(!m.stale);
    try std.testing.expectEqual(slide_graph.State.deleted, try deleteSister(t));
    try std.testing.expectEqual(@as(u32, 1), m.saves);
}

/// A model folder a lesson could change: an empty shard and its config.
fn folderAt(a: Allocator, io: std.Io, tmp: std.testing.TmpDir, name: []const u8) ![]const u8 {
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, name });
    try std.Io.Dir.cwd().createDirPath(io, dir);
    const shard = "\x08\x00\x00\x00\x00\x00\x00\x00{}      "; // a header of 8 bytes, no tensors
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "model.safetensors" }), .data = shard });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "config.json" }), .data = "{}" });
    return dir;
}

fn present(io: std.Io, path: []const u8) bool {
    _ = std.Io.Dir.cwd().statFile(io, path, .{ .follow_symlinks = false }) catch return false;
    return true;
}

test "a delete keeps the folder's original files before anything changes, and a broken way back refuses it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var text: ByteText = .{};
    var m: Model = .{};
    const t = try taught(&text, &m);
    defer t.srv.deinit();
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .broken = true };
    // an unknown fact changes nothing, not even the snapshot
    const fresh = try folderAt(a, io, tmp, "fresh");
    t.srv.config.model_dir = fresh;
    slide_forget.route(t.srv, &conn, a, "99");
    try std.testing.expect(!present(io, try std.fs.path.join(a, &.{ fresh, ".tensorfold-original" })));
    // a real delete keeps it before its stream opens (this connection's stream never does)
    const id = try std.fmt.allocPrint(a, "{d}", .{t.ids[1]});
    slide_forget.route(t.srv, &conn, a, id);
    try std.testing.expect(present(io, try std.fs.path.join(a, &.{ fresh, ".tensorfold-original", "SHA256SUMS" })));
    try std.testing.expect(t.srv.snapshot_checked);
    // a snapshot that no longer matches its manifest refuses the delete before the engine is asked to cut
    const broken = try folderAt(a, io, tmp, "broken");
    const snap = try std.fs.path.join(a, &.{ broken, ".tensorfold-original" });
    try std.Io.Dir.cwd().createDirPath(io, snap);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ snap, "model.safetensors" }), .data = "not the shard" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ snap, "SHA256SUMS" }), .data = "0000000000000000000000000000000000000000000000000000000000000000  model.safetensors\n" });
    t.srv.config.model_dir = broken;
    t.srv.snapshot_checked = false;
    slide_forget.route(t.srv, &conn, a, id);
    try std.testing.expect(!t.srv.snapshot_checked); // refused at the way back, so never at the cut
    try std.testing.expectEqual(@as(u32, 0), m.forgets);
    try std.testing.expect(t.srv.teacher.learning.tryLock()); // the refused delete let go of the lock
    t.srv.teacher.learning.unlock(io);
}
