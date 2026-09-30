const std = @import("std");
const mx = @import("mlx.zig");
const tokenizer = @import("vendor/tokenizer.zig");
const qwen = @import("model.zig");
const sampling = @import("sampling.zig");
const text = @import("reply_text.zig");
const prefill_plan = @import("prefill_plan.zig");
const neural = @import("neural_draft.zig");
pub const Options = @import("request_options.zig").Options;

pub const Backend = union(enum) {
    qwen: qwen.Model,
    nemotron: @import("nemotron.zig").Model,
    flash: @import("flash.zig").Model,
    gemma: @import("gemma.zig").Model,
    glm: @import("glm.zig").Model,
    deepseek: @import("deepseek.zig").Model,

    fn init(io: std.Io, dir: []const u8, drafts: bool) !Backend {
        const path = try std.fmt.allocPrint(mx.allocator, "{s}/config.json", .{dir});
        defer mx.allocator.free(path);
        const bytes = try @import("weights.zig").readFile(io, path);
        defer mx.allocator.free(bytes);
        const cfg = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer cfg.deinit();
        if (cfg.value != .object) return error.InvalidConfig;
        const kind = cfg.value.object.get("model_type") orelse return error.UnsupportedModel;
        if (kind != .string) return error.UnsupportedModel;
        if (std.mem.eql(u8, kind.string, "nemotron_h")) return .{ .nemotron = try @import("nemotron.zig").Model.init(io, dir, drafts) };
        if (std.mem.eql(u8, kind.string, "qwen4_exp")) return .{ .flash = try @import("flash.zig").Model.init(io, dir, drafts) };
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
    drafter: ?*neural.Drafter = null,
    draft_budget: usize = 0,
    tools: std.json.Value = .null,
    cancellation: @import("cancellation.zig").Cancellation = .{},
    gate: ?*@import("call_gate.zig").Gate = null,
    context: ?*anyopaque = null,
    emit: ?*const fn (?*anyopaque, []const u8) anyerror!void = null,
    fn check(s: Sink) !void {
        try s.cancellation.check();
    }
};
pub const Reply = struct {
    tokens: std.ArrayList(u32) = .empty,
    prompt_tokens: usize = 0,
    content: []u8 = &.{},
    finish_reason: enum { stop, length } = .length,
    pub fn deinit(r: *Reply, a: std.mem.Allocator) void {
        r.tokens.deinit(a);
        a.free(r.content);
    }
};

pub const RequestGeneration = union(std.meta.Tag(Backend)) {
    qwen: Generation(qwen.Model),
    nemotron: Generation(@import("nemotron.zig").Model),
    flash: Generation(@import("flash.zig").Model),
    gemma: Generation(@import("gemma.zig").Model),
    glm: Generation(@import("glm.zig").Model),
    deepseek: Generation(@import("deepseek.zig").Model),

    pub fn init(s: *Session, a: std.mem.Allocator, prompt: []const i32, options: Options, sink: Sink, image: ?*@import("vision.zig").Prompt) !RequestGeneration {
        if (image != null and s.backend != .qwen) return error.UnsupportedModelImages;
        switch (s.backend) {
            inline else => |*m, tag| {
                var request = try Generation(@TypeOf(m.*)).init(m, &s.tokenizer, a, prompt, options, s.draftSink(sink), image);
                errdefer request.deinit();
                if (image == null) try request.setPlan(try s.prefillPlan());
                return @unionInit(RequestGeneration, @tagName(tag), request);
            },
        }
    }

    pub fn step(g: *RequestGeneration, s: *Session) !bool {
        switch (g.*) {
            inline else => |*request, tag| {
                if (@as(std.meta.Tag(Backend), s.backend) != tag) return error.WrongGenerationModel;
                return request.step(&@field(s.backend, @tagName(tag)));
            },
        }
    }

    pub fn takeReply(g: *RequestGeneration) !Reply {
        switch (g.*) {
            inline else => |*request| return request.takeReply(),
        }
    }

    pub fn progress(g: *const RequestGeneration) struct { prefilled: usize, decoded: usize, proposed: usize, accepted: usize, structural_proposed: usize, structural_accepted: usize, neural_proposed: usize, neural_accepted: usize } {
        return switch (g.*) {
            inline else => |*request| .{ .prefilled = request.offset, .decoded = request.reply.tokens.items.len, .proposed = request.proposed, .accepted = request.accepted, .structural_proposed = if (request.proposer) |p| p.structural_tokens else 0, .structural_accepted = if (request.proposer) |p| p.structural_accepted else 0, .neural_proposed = request.neural_proposed, .neural_accepted = request.neural_accepted },
        };
    }

    pub fn memoryLengths(g: *const RequestGeneration) @import("memory_budget.zig").Live {
        switch (g.*) {
            inline else => |*request| return .{ .now = @intCast(request.state.position), .most = request.prompt.len + request.options.max_tokens },
        }
    }

    pub fn isDecoding(g: *const RequestGeneration) bool {
        return switch (g.*) {
            inline else => |request| request.phase == .decode,
        };
    }

    pub fn snapshot(g: *const RequestGeneration) !?Snapshot {
        switch (g.*) {
            inline else => |*request, tag| {
                if (request.image != null or request.reply.tokens.items.len != 0 or (request.phase != .prefill and request.phase != .decode) or !request.chunks.contains(request.offset)) return null;
                return @unionInit(Snapshot, @tagName(tag), try request.state.clone());
            },
        }
    }

    pub fn boundary(g: *const RequestGeneration) @import("prompt_cache.zig").Boundary {
        switch (g.*) {
            inline else => |*request| return .{ .starts = request.chunks.starts },
        }
    }

    pub fn restorePrefix(g: *RequestGeneration, snapshot_value: *const Snapshot) !void {
        switch (g.*) {
            inline else => |*request, tag| {
                if (@as(std.meta.Tag(Backend), snapshot_value.*) != tag) return error.WrongSnapshotModel;
                try request.restorePrefix(&@field(snapshot_value.*, @tagName(tag)));
            },
        }
    }

    pub fn restoreOwnedPrefix(g: *RequestGeneration, snapshot_value: *Snapshot) !void {
        switch (g.*) {
            inline else => |*request, tag| {
                if (@as(std.meta.Tag(Backend), snapshot_value.*) != tag) return error.WrongSnapshotModel;
                try request.restoreOwnedPrefix(&@field(snapshot_value.*, @tagName(tag)));
            },
        }
    }

    pub fn cacheBytes(g: *const RequestGeneration) u64 {
        return switch (g.*) {
            inline else => |*request| request.state.nbytes(),
        };
    }

    pub fn deinit(g: *RequestGeneration) void {
        switch (g.*) {
            inline else => |*request| request.deinit(),
        }
    }
};

pub const Snapshot = union(std.meta.Tag(Backend)) {
    qwen: @import("request_state.zig").State(qwen.Model),
    nemotron: @import("request_state.zig").State(@import("nemotron.zig").Model),
    flash: @import("request_state.zig").State(@import("flash.zig").Model),
    gemma: @import("request_state.zig").State(@import("gemma.zig").Model),
    glm: @import("request_state.zig").State(@import("glm.zig").Model),
    deepseek: @import("request_state.zig").State(@import("deepseek.zig").Model),

    pub fn save(s: *const Snapshot, io: std.Io, path: []const u8, identity: []const u8, tokens: []const i32) !void {
        switch (s.*) {
            inline else => |state| {
                if (state.position <= 0 or state.position != tokens.len or state.rope_delta != 0) return error.InvalidSnapshotState;
                try @import("snapshot_file.zig").save(io, path, identity, tokens, state);
            },
        }
    }

    pub fn load(io: std.Io, path: []const u8, identity: []const u8, tokens: []const i32, tag: std.meta.Tag(Backend)) !Snapshot {
        var reader = try @import("snapshot_file.zig").Reader.open(io, path, identity);
        defer reader.deinit();
        if (!std.mem.eql(i32, tokens, reader.metadata.value.tokens)) return error.IncompatibleSnapshot;
        switch (tag) {
            inline else => |kind| {
                var state = try reader.load(@FieldType(Snapshot, @tagName(kind)));
                errdefer state.deinit();
                if (state.position <= 0 or state.position != tokens.len or state.rope_delta != 0) return error.InvalidSnapshotState;
                return @unionInit(Snapshot, @tagName(kind), state);
            },
        }
    }

    pub fn clone(s: *const Snapshot) !Snapshot {
        switch (s.*) {
            inline else => |*state, tag| return @unionInit(Snapshot, @tagName(tag), try state.clone()),
        }
    }
    pub fn deinit(s: *Snapshot) void {
        switch (s.*) {
            inline else => |*state| state.deinit(),
        }
    }
    pub fn nbytes(s: *const Snapshot) u64 {
        switch (s.*) {
            inline else => |*state| return state.nbytes(),
        }
    }
    pub fn position(s: *const Snapshot) usize {
        switch (s.*) {
            inline else => |*state| return @intCast(state.position),
        }
    }
};

pub const Session = struct {
    backend: Backend,
    tokenizer: tokenizer.Tokenizer,
    io: std.Io,
    directory: []u8,
    chat_template: ?@import("chat.zig").Template = null,
    prefill_plan: ?prefill_plan.Plan = null,
    drafter: ?neural.Drafter = null,
    draft_options: neural.Options = .{},

    pub fn draftSink(s: *Session, sink: Sink) Sink {
        var result = sink;
        result.drafter = if (s.drafter) |*d| d else null;
        result.draft_budget = if (s.draft_options.enabled) s.draft_options.max_draft else 0;
        return result;
    }
    pub fn prefillStep(s: *const Session) usize {
        switch (s.backend) {
            inline else => |m| return Generation(@TypeOf(m)).chunk_size,
        }
    }
    pub fn prefillPlan(s: *Session) !prefill_plan.Plan {
        if (s.prefill_plan) |plan| return plan;
        const step = s.prefillStep();
        var plan = prefill_plan.Plan{ .step = step, .min_chunk = @min(256, step) };
        if (s.chat_template == null) s.chat_template = @import("chat.zig").Template.load(mx.allocator, s.io, s.directory) catch |err| {
            if (err == error.OutOfMemory) return err;
            s.prefill_plan = plan;
            return plan;
        };
        const template = &s.chat_template.?;
        const markers = try template.messageMarkers(template.arena.allocator(), &s.tokenizer, s.backend == .deepseek);
        plan.openers = markers.openers;
        plan.assistant = markers.assistant;
        try plan.validate();
        s.prefill_plan = plan;
        return plan;
    }
    pub fn init(io: std.Io, dir: []const u8) !Session {
        return initWithDraft(io, dir, .{});
    }
    pub fn initWithDraft(io: std.Io, dir: []const u8, options: neural.Options) !Session {
        try options.validate();
        var backend = try Backend.init(io, dir, options.enabled and options.max_draft > 0);
        errdefer backend.deinit();
        if (options.enabled) switch (backend) {
            .gemma => |*m| if (options.directory) |path| try m.loadDraftBits(io, path, options.bits),
            .deepseek => |*m| if (options.directory) |path| try m.loadDraft(io, path),
            inline .nemotron, .flash => |*m| {
                if (options.directory != null) return error.UnsupportedDraftDirectory;
                if (m.mtp) try @import("draft_vocab.zig").install(&m.weights, @TypeOf(m.*).draft_vocabulary, @TypeOf(m.*).vocab, 8);
            },
            .glm => if (options.directory != null) return error.UnsupportedDraftDirectory,
            .qwen => {},
        };
        var drafter: ?neural.Drafter = if (options.enabled and backend == .qwen and options.directory != null) try neural.Drafter.init(io, options.directory.?, &backend.qwen) else null;
        errdefer if (drafter) |*d| d.deinit();
        if (options.calibration) |path| {
            if (drafter) |*d| try d.loadCalibration(io, path) else return error.CalibrationRequiresDFlash2;
        }
        const path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir, mx.allocator);
        errdefer mx.allocator.free(path);
        return .{ .backend = backend, .tokenizer = try tokenizer.loadTokenizer(io, mx.allocator, path), .io = io, .directory = path, .drafter = drafter, .draft_options = options };
    }
    pub fn deinit(s: *Session) void {
        if (s.drafter) |*d| d.deinit();
        if (s.chat_template) |*template| template.deinit();
        s.tokenizer.deinit();
        s.backend.deinit();
        mx.allocator.free(s.directory);
    }
    pub fn renderChat(s: *Session, a: std.mem.Allocator, body: std.json.Value, thinking: bool, effort: ?[]const u8) !@import("chat.zig").Rendered {
        if (s.chat_template == null) s.chat_template = try @import("chat.zig").Template.load(mx.allocator, s.io, s.directory);
        return s.chat_template.?.renderWithDefaults(a, body, thinking, effort);
    }
    pub fn generate(s: *Session, a: std.mem.Allocator, prompt: []const i32, options: Options, sink: Sink) !Reply {
        var generation = try RequestGeneration.init(s, a, prompt, options, sink, null);
        defer generation.deinit();
        while (!try generation.step(s)) {}
        return generation.takeReply();
    }
    pub fn generateImages(s: *Session, a: std.mem.Allocator, prompt: []const i32, options: Options, sink: Sink, images: []const @import("vision.zig").EncodedImage) !Reply {
        if (images.len == 0) return s.generate(a, prompt, options, sink);
        if (s.backend != .qwen) return error.UnsupportedModelImages;
        try sink.check();
        var ids: std.ArrayList(i32) = .empty;
        defer ids.deinit(a);
        try ids.appendSlice(a, prompt);
        var prepared = try @import("vision.zig").Prompt.prepareEncoded(s.io, s.directory, images, &ids, a, &s.backend.qwen.weights);
        defer prepared.deinit();
        return generateModel(&s.backend.qwen, &s.tokenizer, a, ids.items, options, s.draftSink(sink), &prepared);
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

fn generateModel(m: anytype, tok: *tokenizer.Tokenizer, a: std.mem.Allocator, prompt: []const i32, options: Options, sink: Sink, image: ?*@import("vision.zig").Prompt) !Reply {
    var generation = try Generation(@TypeOf(m.*)).init(m, tok, a, prompt, options, sink, image);
    defer generation.deinit();
    while (!try generation.step(m)) {}
    return generation.takeReply();
}

pub fn Generation(comptime M: type) type {
    return struct {
        const Self = @This();
        pub const chunk_size: usize = if (@hasDecl(M, "prefill")) 2048 else 16;
        model: *M,
        a: std.mem.Allocator,
        tokenizer: *tokenizer.Tokenizer,
        prompt: []const i32,
        options: Options,
        sink: Sink,
        image: ?*@import("vision.zig").Prompt,
        state: @import("request_state.zig").State(M),
        chunks: prefill_plan.Chunks,
        reply: Reply,
        budget_arena: std.heap.ArenaAllocator,
        budget: @import("thinking_budget.zig").Budget,
        settings: sampling.Sampling,
        offset: usize = 0,
        next: i32 = 0,
        sent: usize = 0,
        phase: enum { prefill, decode, finished, failed } = .prefill,
        proposer: ?@import("tool_draft.zig").Proposer = null,
        context: std.ArrayList(i32) = .empty,
        next_adjusted: bool = false,
        proposed: usize = 0,
        accepted: usize = 0,
        neural_proposed: usize = 0,
        neural_accepted: usize = 0,

        pub fn init(m: *M, tok: *tokenizer.Tokenizer, a: std.mem.Allocator, prompt: []const i32, options: Options, sink: Sink, image: ?*@import("vision.zig").Prompt) !Self {
            const vocab: i32 = if (M == qwen.Model) 248320 else if (@hasField(M, "vocab")) m.vocab else M.vocab;
            if (prompt.len == 0) return error.EmptyPrompt;
            if (prompt.len > 262144 or options.max_tokens > 262144 - prompt.len) return error.ContextLimitExceeded;
            for (prompt) |id| if (id < 0 or id >= vocab) return error.InvalidToken;
            try options.sampling.validate();
            var arena = std.heap.ArenaAllocator.init(a);
            errdefer arena.deinit();
            const budget = if (options.max_tokens == 0) @import("thinking_budget.zig").Budget{} else try @import("thinking_budget.zig").Budget.init(arena.allocator(), tok, options.thinking_budget);
            var settings = options.sampling;
            settings.seed = options.seed orelse sampling.seedFor(prompt);
            const chunks = try (prefill_plan.Plan{ .step = chunk_size }).chunks(a, prompt);
            errdefer chunks.deinit(a);
            var proposer: ?@import("tool_draft.zig").Proposer = if (options.draft) try @import("tool_draft.zig").Proposer.init(a, tok, sink.tools, prompt.len) else null;
            errdefer if (proposer) |*value| value.deinit();
            var context: std.ArrayList(i32) = .empty;
            errdefer context.deinit(a);
            if (proposer != null) try context.appendSlice(a, prompt);
            return .{ .model = m, .a = a, .tokenizer = tok, .prompt = prompt, .options = options, .sink = sink, .image = image, .state = try @import("request_state.zig").State(M).init(m), .chunks = chunks, .reply = .{ .prompt_tokens = prompt.len }, .budget_arena = arena, .budget = budget, .settings = settings, .phase = if (options.max_tokens == 0) .finished else .prefill, .proposer = proposer, .context = context };
        }

        pub fn deinit(g: *Self) void {
            if (g.proposer) |*proposer| proposer.deinit();
            g.context.deinit(g.a);
            g.state.deinit();
            g.chunks.deinit(g.a);
            g.reply.deinit(g.a);
            g.budget_arena.deinit();
            g.* = undefined;
        }

        pub fn takeReply(g: *Self) !Reply {
            if (g.phase != .finished) return error.IncompleteGeneration;
            const reply = g.reply;
            g.reply = .{};
            return reply;
        }

        pub fn restorePrefix(g: *Self, saved: *const @import("request_state.zig").State(M)) !void {
            _ = try g.validatePrefix(saved);
            var copy = try saved.clone();
            defer copy.deinit();
            try g.restoreOwnedPrefix(&copy);
        }

        fn validatePrefix(g: *const Self, saved: *const @import("request_state.zig").State(M)) !usize {
            if (g.phase != .prefill or g.offset != 0 or g.image != null or saved.rope_delta != 0 or saved.position <= 0) return error.InvalidSnapshotState;
            const offset: usize = @intCast(saved.position);
            if (!g.chunks.contains(offset) or saved.cache.len != g.state.cache.len) return error.IncompatibleSnapshotBoundary;
            return offset;
        }

        /// On success, saved owns the request's former empty state and remains safe to deinitialize.
        pub fn restoreOwnedPrefix(g: *Self, saved: *@import("request_state.zig").State(M)) !void {
            const offset = try g.validatePrefix(saved);
            std.mem.swap(@TypeOf(g.state), &g.state, saved);
            g.offset = offset;
        }

        pub fn setPlan(g: *Self, plan: prefill_plan.Plan) !void {
            if (g.offset != 0 or g.image != null) return error.InvalidSnapshotState;
            if (plan.step > chunk_size) return error.UnsupportedPrefillChunk;
            const chunks = try plan.chunks(g.a, g.prompt);
            g.chunks.deinit(g.a);
            g.chunks = chunks;
        }

        /// One prefill chunk or verified decode block; no model pass survives the call.
        pub fn step(g: *Self, m: *M) !bool {
            if (m != g.model) return error.WrongGenerationModel;
            if (g.phase == .finished) return true;
            if (g.phase == .failed) return error.FailedGeneration;
            errdefer g.phase = .failed;
            try g.sink.check();
            g.state.swap(m);
            defer g.state.swap(m);
            if (g.sink.drafter) |d| g.state.swapDFlash(d);
            defer if (g.sink.drafter) |d| g.state.swapDFlash(d);
            if (g.phase == .prefill) try g.prefill(m) else try g.decode(m);
            return g.phase == .finished;
        }

        fn prefill(g: *Self, m: *M) !void {
            const count = g.chunks.next(g.offset) - g.offset;
            const tokens = g.prompt[g.offset..][0..count];
            var image_scope = mx.Scope{};
            defer image_scope.deinit();
            var pass = if (M == qwen.Model) blk: {
                if (g.image) |p| break :blk try m.prefillImage(tokens, try image_scope.slice(p.embeddings, 1, @intCast(g.offset), @intCast(g.offset + count)), try p.positions.chunk(&image_scope, g.offset, g.offset + count), p.positions.delta);
                break :blk try m.prefill(tokens);
            } else if (@hasDecl(M, "prefill")) try m.prefill(tokens) else try m.forward(tokens);
            defer pass.deinit();
            const vocab: i32 = if (M == qwen.Model) 248320 else if (@hasField(M, "vocab")) m.vocab else M.vocab;
            const logits = try pass.scope.reshape(pass.logits, &.{ -1, vocab });
            const rows = mx.dim(logits, 0);
            const ids = try sampling.rows(&m.kernels, &pass.scope, try pass.scope.slice(logits, 0, rows - 1, rows), &.{@intCast(g.offset + count)}, g.settings);
            defer mx.allocator.free(ids);
            g.next = ids[0];
            if (M == qwen.Model) {
                var kept: [2048]i32 = undefined;
                for (kept[0..count], 0..) |*row, j| row.* = @intCast(j);
                try m.commit(&pass, kept[0..count]);
            } else try m.commit(&pass, count);
            var rows_kept: [2048]i32 = undefined;
            for (rows_kept[0..count], 0..) |*row, j| row.* = @intCast(j);
            if (g.sink.draft_budget > 0) try neural.absorb(m, &g.state, g.sink.drafter, &pass, tokens, rows_kept[0..count]);
            g.offset += count;
            if (g.offset == g.prompt.len) g.phase = .decode;
        }

        fn eos(m: *M, id: i32) bool {
            return if (M == qwen.Model) id == 248044 or id == 248046 else if (@hasDecl(M, "isEos")) m.isEos(id) else M.eos(id);
        }

        fn decode(g: *Self, m: *M) !void {
            if (!g.next_adjusted) g.next = try g.budget.next(g.sink.gate, g.reply.tokens.items.len, g.next, eos(m, g.next));
            g.next_adjusted = false;
            try g.emitToken(m);
            if (g.phase == .finished) return;
            var window: [16]i32 = undefined;
            var parents: [16]i32 = undefined;
            var positions: [16]i32 = undefined;
            window[0] = g.next;
            parents[0] = -1;
            var draft = @import("drafter.zig").Proposal{};
            if (g.proposer) |*proposer| {
                draft = try proposer.propose(g.a, g.context.items, @min(15, g.options.max_tokens - g.reply.tokens.items.len));
            }
            const from_neural = draft.len == 0 and g.options.draft and g.sink.draft_budget > 0 and neural.enabled(m, g.sink.drafter);
            if (from_neural) draft = try neural.propose(m, &g.state, g.sink.drafter, g.next, @min(g.sink.draft_budget, g.options.max_tokens - g.reply.tokens.items.len), g.settings);
            const proposed = draft.len;
            @memcpy(window[1..][0..proposed], draft.tokens[0..proposed]);
            const count = proposed + 1;
            positions[0] = m.position + 1;
            for (0..proposed) |i| {
                parents[i + 1] = draft.parents[i] + 1;
                positions[i + 1] = positions[@intCast(parents[i + 1])] + 1;
            }
            var pass = if (M == qwen.Model) try m.forward(window[0..count], parents[0..count]) else try m.forward(window[0..count]);
            defer pass.deinit();
            const ids = try sampling.rows(&m.kernels, &pass.scope, pass.logits, positions[0..count], g.settings);
            defer mx.allocator.free(ids);
            var keep: usize = 1;
            var kept: [16]i32 = undefined;
            kept[0] = 0;
            var accepted: usize = 0;
            var row: usize = 0;
            while (true) {
                try g.sink.check();
                var has_children = false;
                for (parents[1..count]) |parent| if (parent == row) {
                    has_children = true;
                    break;
                };
                if (!has_children) {
                    g.next = ids[row];
                    break;
                }
                g.next = try g.budget.next(g.sink.gate, g.reply.tokens.items.len, ids[row], eos(m, ids[row]));
                var child: ?usize = null;
                for (row + 1..count) |i| if (parents[i] == row and window[i] == g.next) {
                    child = i;
                    break;
                };
                if (child == null) {
                    g.next_adjusted = true;
                    break;
                }
                accepted += 1;
                try g.emitToken(m);
                if (g.phase == .finished) break;
                row = child.?;
                kept[keep] = @intCast(row);
                keep += 1;
            }
            if (M == qwen.Model) {
                try m.commit(&pass, kept[0..keep]);
            } else try m.commit(&pass, keep);
            if (g.sink.draft_budget > 0) try neural.absorb(m, &g.state, g.sink.drafter, &pass, window[0..count], kept[0..keep]);
            g.proposed += proposed;
            g.accepted += accepted;
            if (from_neural) {
                g.neural_proposed += proposed;
                g.neural_accepted += accepted;
            } else if (g.proposer) |*proposer| proposer.observe(proposed, accepted);
        }

        fn emitToken(g: *Self, m: *M) !void {
            if (eos(m, g.next) and !g.options.ignore_eos) {
                const ending = try g.tokenizer.decode(g.a, &.{@intCast(g.next)}, false);
                defer g.a.free(ending);
                for ([_][]const u8{ "</tool_call>", "<tool_call|>", "</｜DSML｜tool_calls>" }) |close| if (std.mem.eql(u8, ending, close)) {
                    try g.reply.tokens.append(g.a, @intCast(g.next));
                    break;
                };
                g.reply.finish_reason = .stop;
                return g.finish();
            }
            try g.reply.tokens.append(g.a, @intCast(g.next));
            if (g.proposer != null) try g.context.append(g.a, g.next);
            const decoded = try g.tokenizer.decode(g.a, g.reply.tokens.items, false);
            defer g.a.free(decoded);
            const stopped = text.stopAt(decoded, g.options.stops) != null;
            const shown = text.visible(decoded, g.options.stops, !stopped);
            if (shown.len >= g.sent and std.unicode.utf8ValidateSlice(shown) and !std.mem.endsWith(u8, shown, "�")) {
                if (g.sink.emit) |emit| try emit(g.sink.context, shown[g.sent..]);
                g.sent = shown.len;
            }
            if (stopped) g.reply.finish_reason = .stop;
            if (stopped or g.reply.tokens.items.len == g.options.max_tokens) return g.finish();
        }

        fn finish(g: *Self) !void {
            const decoded = try g.tokenizer.decode(g.a, g.reply.tokens.items, false);
            defer g.a.free(decoded);
            g.reply.content = try g.a.dupe(u8, text.visible(decoded, g.options.stops, false));
            if (g.sink.emit) |emit| if (g.reply.content.len > g.sent) try emit(g.sink.context, g.reply.content[g.sent..]);
            g.phase = .finished;
        }
    };
}
