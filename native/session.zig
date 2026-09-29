const std = @import("std");
const mx = @import("mlx.zig");
const tokenizer = @import("vendor/tokenizer.zig");
const qwen = @import("model.zig");
const sampling = @import("sampling.zig");
const text = @import("reply_text.zig");
pub const Options = @import("request_options.zig").Options;

pub const Backend = union(enum) {
    qwen: qwen.Model,
    nemotron: @import("nemotron.zig").Model,
    flash: @import("flash.zig").Model,
    gemma: @import("gemma.zig").Model,
    glm: @import("glm.zig").Model,
    deepseek: @import("deepseek.zig").Model,

    fn init(io: std.Io, dir: []const u8) !Backend {
        const path = try std.fmt.allocPrint(mx.allocator, "{s}/config.json", .{dir});
        defer mx.allocator.free(path);
        const bytes = try @import("weights.zig").readFile(io, path);
        defer mx.allocator.free(bytes);
        const cfg = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer cfg.deinit();
        if (cfg.value != .object) return error.InvalidConfig;
        const kind = cfg.value.object.get("model_type") orelse return error.UnsupportedModel;
        if (kind != .string) return error.UnsupportedModel;
        if (std.mem.eql(u8, kind.string, "nemotron_h")) return .{ .nemotron = try @import("nemotron.zig").Model.init(io, dir, false) };
        if (std.mem.eql(u8, kind.string, "qwen4_exp")) return .{ .flash = try @import("flash.zig").Model.init(io, dir, false) };
        if (std.mem.eql(u8, kind.string, "gemma4")) return .{ .gemma = try @import("gemma.zig").Model.init(io, dir) };
        if (std.mem.eql(u8, kind.string, "glm5_next")) return .{ .glm = try @import("glm.zig").Model.init(io, dir) };
        if (std.mem.eql(u8, kind.string, "deepseek_v4")) return .{ .deepseek = try @import("deepseek.zig").Model.init(io, dir) };
        return .{ .qwen = try qwen.Model.init(io, dir) };
    }
    fn deinit(b: *Backend) void {
        switch (b.*) {
            inline else => |*m| m.deinit(),
        }
    }
};

pub const Sink = struct {
    context: ?*anyopaque = null,
    emit: ?*const fn (?*anyopaque, []const u8) anyerror!void = null,
    cancelled: ?*const fn (?*anyopaque) bool = null,
    fn check(s: Sink) !void {
        if (s.cancelled) |call| if (call(s.context)) return error.RequestCancelled;
    }
};
pub const Reply = struct {
    tokens: std.ArrayList(u32) = .empty,
    content: []u8 = &.{},
    finish_reason: enum { stop, length } = .length,
    pub fn deinit(r: *Reply, a: std.mem.Allocator) void {
        r.tokens.deinit(a);
        a.free(r.content);
    }
};

pub const Session = struct {
    backend: Backend,
    tokenizer: tokenizer.Tokenizer,
    pub fn init(io: std.Io, dir: []const u8) !Session {
        var backend = try Backend.init(io, dir);
        errdefer backend.deinit();
        const path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir, mx.allocator);
        defer mx.allocator.free(path);
        return .{ .backend = backend, .tokenizer = try tokenizer.loadTokenizer(io, mx.allocator, path) };
    }
    pub fn deinit(s: *Session) void {
        s.tokenizer.deinit();
        s.backend.deinit();
    }
    pub fn generate(s: *Session, a: std.mem.Allocator, prompt: []const i32, options: Options, sink: Sink) !Reply {
        switch (s.backend) {
            inline else => |*m| return generateModel(m, &s.tokenizer, a, prompt, options, sink),
        }
    }
    pub fn validate(s: *Session, prompt: []const i32, options: Options) !void {
        const vocab: i32 = switch (s.backend) {
            .qwen => 248320,
            inline else => |m| if (@hasField(@TypeOf(m), "vocab")) m.vocab else @TypeOf(m).vocab,
        };
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len > 262144 or options.max_tokens > 262144 - prompt.len) return error.ContextLimitExceeded;
        for (prompt) |id| if (id < 0 or id >= vocab) return error.InvalidToken;
        try options.sampling.validate();
    }
};

fn generateModel(m: anytype, tok: *tokenizer.Tokenizer, a: std.mem.Allocator, prompt: []const i32, options: Options, sink: Sink) !Reply {
    const M = @TypeOf(m.*);
    const vocab: i32 = if (M == qwen.Model) 248320 else if (@hasField(M, "vocab")) m.vocab else M.vocab;
    if (prompt.len == 0) return error.EmptyPrompt;
    if (prompt.len > 262144 or options.max_tokens > 262144 - prompt.len) return error.ContextLimitExceeded;
    for (prompt) |id| if (id < 0 or id >= vocab) return error.InvalidToken;
    try options.sampling.validate();
    m.reset();
    defer m.reset();
    var reply = Reply{};
    errdefer reply.deinit(a);
    if (options.max_tokens == 0) return reply;
    var settings = options.sampling;
    settings.seed = options.seed orelse sampling.seedFor(prompt);
    var next: i32 = 0;
    var offset: usize = 0;
    while (offset < prompt.len) {
        try sink.check();
        const count = @min(if (@hasDecl(M, "prefill")) @as(usize, 2048) else 16, prompt.len - offset);
        var pass = if (@hasDecl(M, "prefill")) try m.prefill(prompt[offset..][0..count]) else try m.forward(prompt[offset..][0..count]);
        defer pass.deinit();
        const logits = try pass.scope.reshape(pass.logits, &.{ -1, vocab });
        const rows = mx.dim(logits, 0);
        const ids = try sampling.rows(&m.kernels, &pass.scope, try pass.scope.slice(logits, 0, rows - 1, rows), &.{@intCast(offset + count)}, settings);
        defer mx.allocator.free(ids);
        next = ids[0];
        if (M == qwen.Model) {
            var kept: [2048]i32 = undefined;
            for (kept[0..count], 0..) |*row, j| row.* = @intCast(j);
            try m.commit(&pass, kept[0..count]);
        } else try m.commit(&pass, count);
        offset += count;
    }
    var sent: usize = 0;
    while (reply.tokens.items.len < options.max_tokens) {
        try sink.check();
        const eos = if (M == qwen.Model) next == 248044 or next == 248046 else if (@hasDecl(M, "isEos")) m.isEos(next) else M.eos(next);
        if (eos and !options.ignore_eos) {
            reply.finish_reason = .stop;
            break;
        }
        try reply.tokens.append(a, @intCast(next));
        const decoded = try tok.decode(a, reply.tokens.items, false);
        defer a.free(decoded);
        const stopped = text.stopAt(decoded, options.stops) != null;
        const shown = text.visible(decoded, options.stops, !stopped);
        if (shown.len >= sent and std.unicode.utf8ValidateSlice(shown) and !std.mem.endsWith(u8, shown, "�")) {
            if (sink.emit) |emit| try emit(sink.context, shown[sent..]);
            sent = shown.len;
        }
        if (stopped) {
            reply.finish_reason = .stop;
            break;
        }
        if (reply.tokens.items.len == options.max_tokens) break;
        var pass = if (M == qwen.Model) try m.forward(&.{next}, &.{-1}) else try m.forward(&.{next});
        defer pass.deinit();
        const ids = try sampling.rows(&m.kernels, &pass.scope, pass.logits, &.{m.position + 1}, settings);
        defer mx.allocator.free(ids);
        next = ids[0];
        if (M == qwen.Model) try m.commit(&pass, &.{0}) else try m.commit(&pass, 1);
    }
    const decoded = try tok.decode(a, reply.tokens.items, false);
    defer a.free(decoded);
    reply.content = try a.dupe(u8, text.visible(decoded, options.stops, false));
    if (sink.emit) |emit| if (reply.content.len > sent) try emit(sink.context, reply.content[sent..]);
    return reply;
}
