//! The DeepSeek serving side behind the server: tokenizer, encoding, thinking, tools, reasoning memory, images.
const std = @import("std");
const json = @import("json");
const serve = @import("dsv41_serve");
const errors = @import("errors.zig");
const family = @import("family.zig");
const model_text = @import("model_text.zig");
const ids_mod = @import("ids.zig");
const log = @import("log.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;
const tokenizer = serve.tokenizer;
const template = serve.template;
pub const tools_mod = serve.tools;
const deepseek_session = @import("deepseek_session.zig");
const Memory = deepseek_session.Memory;
const Reply = deepseek_session.Reply;

pub const model_type = "deepseek_v41";

pub const image_text = "[image omitted: this server does not pass images to the model, so the model cannot see this image. Do not guess what it shows; say that it could not be seen.]";

pub const Images = enum { placeholder, reject };

pub const Options = struct {
    images: Images = .placeholder,
    image_text: []const u8 = image_text,
    default_effort: []const u8 = "high",
    reasoning_fields: family.ReasoningFields = .both,
    schema_prompt: bool = true,
    memory_entries: usize = 1024,
    eos: []const u32 = &.{1},

    /// The TF_DSV41_* knobs from ``env``; ``problem`` says which one is wrong.
    pub fn fromEnv(env: ?*const std.process.Environ.Map, problem: *[]const u8) error{Invalid}!Options {
        var o: Options = .{};
        const m = env orelse return o;
        if (m.get("TF_DSV41_IMAGES")) |v| {
            const t = std.mem.trim(u8, v, " ");
            o.images = std.meta.stringToEnum(Images, if (t.len == 0) "placeholder" else t) orelse {
                problem.* = "TF_DSV41_IMAGES: expected placeholder or reject";
                return error.Invalid;
            };
        }
        if (m.get("TF_DSV41_IMAGE_TEXT")) |v| if (std.mem.trim(u8, v, " ").len > 0) {
            o.image_text = std.mem.trim(u8, v, " ");
        };
        if (m.get("TF_DSV41_DEFAULT_EFFORT")) |v| {
            const t = std.mem.trim(u8, v, " ");
            if (t.len > 0) {
                if (level(t) == null or level(t).? == null) {
                    problem.* = "TF_DSV41_DEFAULT_EFFORT: expected low (50), medium (75), high (75), max (100) or an integer 1-100";
                    return error.Invalid;
                }
                o.default_effort = t;
            }
        }
        if (m.get("TF_DSV41_REASONING_FIELDS")) |v| {
            o.reasoning_fields = std.meta.stringToEnum(family.ReasoningFields, std.mem.trim(u8, v, " ")) orelse {
                problem.* = "TF_DSV41_REASONING_FIELDS: expected both, reasoning_content or reasoning";
                return error.Invalid;
            };
        }
        if (m.get("TF_DSV41_SCHEMA_PROMPT")) |v| o.schema_prompt = !std.mem.eql(u8, std.mem.trim(u8, v, " "), "0");
        if (m.get("TF_DSV41_REASONING_MEMORY")) |v| o.memory_entries = std.fmt.parseInt(usize, std.mem.trim(u8, v, " "), 10) catch 1024;
        return o;
    }
};

/// A ``reasoning_effort`` value as a budget (``template.effortOf``'s names).
const level = template.effortOf;

/// The budget an effort label renders (defaults to high).
pub fn budget(label: []const u8) u8 {
    return ((level(label) orelse return template.default_budget) orelse template.default_budget);
}

/// The serving side of one checkpoint: its tokenizer, the encoding and the family rules.
pub const DeepSeek = struct {
    gpa: Allocator,
    tok: *tokenizer.Tokenizer,
    options: Options,
    memory: Memory,
    images_omitted: std.atomic.Value(u64) = .init(0),

    pub fn load(gpa: Allocator, io: std.Io, dir: []const u8, options: Options, problem: *[]const u8) !*DeepSeek {
        const d = try gpa.create(DeepSeek);
        errdefer gpa.destroy(d);
        const tok = tokenizer.Tokenizer.load(gpa, io, dir) catch |e| {
            problem.* = try std.fmt.allocPrint(gpa, "cannot read {s}/tokenizer.json ({s})", .{ dir, @errorName(e) });
            return error.Load;
        };
        d.* = .{ .gpa = gpa, .tok = tok, .options = options, .memory = .{ .gpa = gpa, .entries = options.memory_entries } };
        return d;
    }

    /// A DeepSeek side over an already loaded tokenizer (tests).
    pub fn init(gpa: Allocator, tok: *tokenizer.Tokenizer, options: Options) Allocator.Error!*DeepSeek {
        const d = try gpa.create(DeepSeek);
        d.* = .{ .gpa = gpa, .tok = tok, .options = options, .memory = .{ .gpa = gpa, .entries = options.memory_entries } };
        return d;
    }

    pub fn deinit(d: *DeepSeek, owns_tokenizer: bool) void {
        d.memory.deinit();
        if (owns_tokenizer) d.tok.deinit();
        d.gpa.destroy(d);
    }

    fn self(ctx: *anyopaque) *DeepSeek {
        return @ptrCast(@alignCast(ctx));
    }

    // -- model_text.Text --------------------------------------------------------------------------------------------

    pub fn text(d: *DeepSeek) model_text.Text {
        return .{ .ctx = d, .vtable = &.{
            .encode = encodeFn,
            .decode = decodeFn,
            .token_id = tokenIdFn,
            .token_string = tokenStringFn,
            .vocab_size = vocabFn,
            .eos_ids = eosFn,
            .render = renderTextFn,
            .template_source = sourceFn,
            .special = specialFn,
        } };
    }

    fn encodeFn(ctx: *anyopaque, a: Allocator, s: []const u8, _: bool) model_text.Error![]u32 {
        return self(ctx).tok.encodeAlloc(a, s); // the release tokenizer adds nothing for add_special_tokens
    }

    fn decodeFn(ctx: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        return self(ctx).tok.decodeAlloc(a, ids);
    }

    fn tokenIdFn(ctx: *anyopaque, piece: []const u8) ?u32 {
        return self(ctx).tok.tokenId(piece);
    }

    fn tokenStringFn(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return a.dupe(u8, self(ctx).tok.tokenString(id));
    }

    fn vocabFn(ctx: *anyopaque) u32 {
        return self(ctx).tok.vocabSize();
    }

    fn eosFn(ctx: *anyopaque) []const u32 {
        return self(ctx).options.eos;
    }

    fn sourceFn(_: *anyopaque) []const u8 {
        return ""; // the encoding is code, not a template: no effort names to read from it
    }

    fn specialFn(ctx: *anyopaque, id: u32) bool {
        return self(ctx).tok.isSpecial(id);
    }

    fn imageHook(ctx: ?*anyopaque, _: Value) []const u8 {
        const d: *DeepSeek = @ptrCast(@alignCast(ctx.?));
        _ = d.images_omitted.fetchAdd(1, .monotonic);
        return d.options.image_text;
    }

    fn templateOptions(d: *DeepSeek, offered: []const Value, thinking: bool, effort: []const u8, generation: bool) template.Options {
        return .{
            .tools = offered,
            .thinking = thinking,
            .budget = budget(effort),
            .add_generation_prompt = generation,
            .image = if (d.options.images == .placeholder) .{ .ctx = d, .text = imageHook } else null,
        };
    }

    /// The generic path's render (template probes, /tokenize without the family): the encoding with its options.
    fn renderTextFn(ctx: *anyopaque, a: Allocator, messages: Value, o: model_text.RenderOptions, problem: *[]const u8) model_text.Error![]u8 {
        const d = self(ctx);
        var p: template.Problem = .{};
        const offered: []const Value = if (o.tools) |t| (if (t == .array) t.array else &.{}) else &.{};
        return template.render(a, messages, d.templateOptions(offered, o.enable_thinking, o.reasoning_effort orelse d.options.default_effort, o.add_generation_prompt), &p) catch |e| switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.Refused => {
                problem.* = p.message;
                return error.Template;
            },
        };
    }

    // -- family.Family ------------------------------------------------------------------------------------------

    pub fn family_(d: *DeepSeek) family.Family {
        return .{ .ctx = d, .vt = switch (d.options.reasoning_fields) {
            inline else => |f| &struct {
                const vt: family.Family.VTable = .{
                    .name = model_type,
                    .thinking = thinkingFn,
                    .default_effort = defaultEffortFn,
                    .tools = toolsFn,
                    .render = renderFn,
                    .reader = readerFn,
                    .remember = rememberFn,
                    .reasoning_fields = f,
                    .reads_min_p = false,
                };
            }.vt,
        } };
    }

    /// ``options``: ``reasoning_effort`` (top level, then template kwargs), then ``enable_thinking`` / ``thinking``.
    fn thinkingFn(_: *anyopaque, cx: *Cx, body: Value) errors.Refused!family.Thinking {
        var out: family.Thinking = .{};
        const kwargs = body.get("chat_template_kwargs") orelse Value.null;
        if (kwargs != .null and kwargs != .object) return cx.refuse("chat_template_kwargs must be a JSON object or null");
        for ([_]?Value{ body.get("reasoning_effort"), if (kwargs == .object) kwargs.get("reasoning_effort") else null }) |raw| {
            const v = raw orelse continue;
            if (v == .null) continue;
            const t: []const u8 = switch (v) {
                .string => |s| s,
                .int => |s| s,
                else => return cx.refuse("reasoning_effort must be 1-100 or low / medium / high / max"),
            };
            const got = level(t) orelse return cx.refuse("reasoning_effort must be 1-100 or low / medium / high / max");
            if (got) |n| {
                out.enable = true;
                out.effort = try std.fmt.allocPrint(cx.a, "{d}", .{n});
            } else out.enable = false;
        }
        if (kwargs == .object) for ([_][]const u8{ "enable_thinking", "thinking" }) |name| if (kwargs.get(name)) |v| {
            if (v != .bool) return cx.fail(.request, "chat_template_kwargs.{s} must be true or false", .{name});
            out.enable = v.bool;
        };
        return out;
    }

    fn defaultEffortFn(ctx: *anyopaque) []const u8 {
        return self(ctx).options.default_effort;
    }

    fn toolsFn(_: *anyopaque, cx: *Cx, body: Value) errors.Refused![]const Value {
        const t = body.get("tools") orelse return &.{};
        if (t == .null) return &.{};
        if (t != .array) return cx.refuse("tools must be a list of objects");
        for (t.array) |x| if (x != .object) return cx.refuse("tools must be a list of objects");
        if (body.get("tool_choice")) |c| if (c == .string and std.mem.eql(u8, c.string, "none")) return &.{};
        return t.array;
    }

    /// ``schema_for_prompt``: what ``## Response Format`` shows for ``response_format``, or null.
    fn schemaForPrompt(a: Allocator, body: Value) Allocator.Error!?Value {
        const rf = body.get("response_format") orelse return null;
        if (rf != .object) return null;
        const kind = rf.get("type") orelse return null;
        if (kind != .string) return null;
        if (std.mem.eql(u8, kind.string, "json_object")) return (try json.parseText(a, "{\"type\": \"object\"}")).ok;
        if (!std.mem.eql(u8, kind.string, "json_schema")) return null;
        const js = rf.get("json_schema") orelse return null;
        if (!js.truthy() or js != .object) return null;
        var schema = js.get("schema") orelse js.get("json_schema") orelse return null;
        if (schema == .string) schema = switch (try json.parseText(a, schema.string)) {
            .ok => |v| v,
            .err => return null,
        };
        return schema;
    }

    fn renderFn(ctx: *anyopaque, cx: *Cx, p: family.Prompt) errors.Refused!family.Rendered {
        const d = self(ctx);
        const a = cx.a;
        if (p.messages != .array or p.messages.array.len == 0) return cx.refuse("messages must be a non-empty list");
        const messages = try d.memory.restore(a, p.messages);
        var o = d.templateOptions(p.tools, p.thinking, p.effort, p.generation);
        if (d.options.schema_prompt) o.response_format = try schemaForPrompt(a, p.body);
        var counter: Counter = .{ .d = d };
        if (o.image != null) o.image = .{ .ctx = &counter, .text = Counter.hook };
        var problem: template.Problem = .{};
        const prompt = template.render(a, messages, o, &problem) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Refused => return cx.refuse(problem.message),
        };
        const ids = try d.tok.encodeAlloc(a, prompt);
        if (counter.n > 0 and p.generation) log.line("images: {d} image part(s) replaced by the placeholder notice (TF_DSV41_IMAGES=placeholder)", .{counter.n});
        return .{ .ids = ids, .images_omitted = counter.n };
    }

    const Counter = struct {
        d: *DeepSeek,
        n: u32 = 0,
        fn hook(ctx: ?*anyopaque, _: Value) []const u8 {
            const c: *Counter = @ptrCast(@alignCast(ctx.?));
            c.n += 1;
            return c.d.options.image_text;
        }
    };

    fn rememberFn(ctx: *anyopaque, reasoning: []const u8, content: []const u8, calls: []const Value) void {
        self(ctx).memory.put(reasoning, content, calls);
    }

    fn readerFn(ctx: *anyopaque, a: Allocator, thinking: bool, offered: []const Value, single: bool) Allocator.Error!family.Reader {
        const d = self(ctx);
        const r = try a.create(Reply);
        r.* = .{ .a = a, .d = d, .detok = .{ .tok = d.tok }, .stream = tools_mod.Stream.init(a, thinking, try tools_mod.Tools.init(a, offered), .{ .make = callId }), .thinking = thinking, .offered = try tools_mod.Tools.init(a, offered), .single = single };
        return .{ .ctx = r, .vt = &.{ .push = Reply.push, .pending = Reply.pending, .feed = Reply.feed, .parse = Reply.parse, .calls = Reply.calls, .markup = Reply.markup } };
    }

    pub fn callId(_: ?*anyopaque, a: Allocator, _: usize) Allocator.Error![]const u8 {
        return ids_mod.make(a, "call_", 24);
    }
};

test "reasoning_effort and the thinking switches read as the Python app reads them" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const body = (try json.parseText(cx.a, "{\"reasoning_effort\": \"medium\", \"chat_template_kwargs\": {\"reasoning_effort\": 90}}")).ok;
    const got = try DeepSeek.thinkingFn(undefined, &cx, body);
    try std.testing.expectEqualStrings("90", got.effort.?);
    try std.testing.expect(got.enable.?);
    const off = try DeepSeek.thinkingFn(undefined, &cx, (try json.parseText(cx.a, "{\"reasoning_effort\": \"none\"}")).ok);
    try std.testing.expect(!off.enable.?);
    try std.testing.expectEqual(@as(u8, 75), budget("medium"));
    try std.testing.expectEqual(@as(u8, 100), budget("xhigh"));
    try std.testing.expectError(error.Refused, DeepSeek.thinkingFn(undefined, &cx, (try json.parseText(cx.a, "{\"reasoning_effort\": 101}")).ok));
    try std.testing.expectError(error.Refused, DeepSeek.thinkingFn(undefined, &cx, (try json.parseText(cx.a, "{\"chat_template_kwargs\": {\"thinking\": 1}}")).ok));
}
