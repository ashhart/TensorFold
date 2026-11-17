//! Message starts (#3617): every message the template renders begins at one, and the prompt's chunk plan is unchanged.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const prompt_mod = @import("prompt.zig");
const chunk_plan = @import("chunk_plan.zig");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const t = std.testing;

const Script = struct {
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768, .prompt_cache = true, .prefill_step = 16, .prefill_min_chunk = 16 };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

// Turn formats for families whose checkpoint templates the repository lacks (Gemma 4's as prompt_tool_turn_test)
const gemma4 =
    \\{% for m in messages %}{% if m.role != 'assistant' or loop.index0 == 0 or messages[loop.index0 - 1].role != 'tool' %}<|turn>{{ m.role }}
    \\{% endif %}{{ m.content }}{% if m.role != 'tool' or loop.last %}<turn|>
    \\{% endif %}{% endfor %}{% if add_generation_prompt %}<|turn>model
    \\{% endif %}
;
const llama =
    \\<|begin_of_text|>{% for m in messages %}<|start_header_id|>{{ m.role if m.role != 'tool' else 'ipython' }}<|end_header_id|>
    \\
    \\{{ m.content }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>
    \\
    \\{% endif %}
;
const nemotron =
    \\{% for m in messages %}{% if m.role == 'tool' %}<|im_start|>user
    \\<tool_response>{{ m.content }}</tool_response><|im_end|>
    \\{% else %}<|im_start|>{{ m.role }}
    \\{{ m.content }}<|im_end|>
    \\{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
    \\<think></think>{% endif %}
;

// A system message, user, assistant and tool turns, every message far closer to the last than any min_chunk
const talk =
    \\[{"role":"system","content":"KSYS"},{"role":"user","content":"KUA"},{"role":"assistant","content":"KAB"},
    \\{"role":"tool","content":"KTC"},{"role":"tool","content":"KTD"},{"role":"assistant","content":"KAE"},
    \\{"role":"user","content":"KUF"},{"role":"assistant","content":"KAG"},{"role":"user","content":"KUH"}]
;
const contents = [_][]const u8{ "KSYS", "KUA", "KAB", "KTC", "KTD", "KAE", "KUF", "KAG", "KUH" };

fn served(a: std.mem.Allocator, f: anytype, script: *Script) !*Server {
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, f.path, null, a, &problem);
    return Server.init(a, t.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
}

fn prepared(srv: *Server, a: std.mem.Allocator) !prompt_mod.Rendered {
    var cx: errors.Cx = .{ .a = a };
    return prompt_mod.prepare(srv, &cx, .{ .messages = (try json.parse(a, talk)).ok, .fields = .null }, false, null);
}

/// Each message's start precedes its own text and follows the one before: the segments hold the contents in order.
fn expectEveryStart(srv: *Server, a: std.mem.Allocator, r: prompt_mod.Rendered) !void {
    const s = r.message_starts;
    const lead: usize = if (s.len == contents.len) 1 else 0; // the first message at token 0 has no start past 0
    try t.expectEqual(contents.len, s.len + 1 - lead);
    for (contents, 0..) |want, k| {
        const from: u32 = if (k + lead == 0) 0 else s[k + lead - 1];
        const to: usize = if (k + lead < s.len) s[k + lead] else r.ids.len;
        const seg = try srv.text.decode(a, r.ids[from..to]);
        try t.expect(std.mem.indexOf(u8, seg, want) != null);
        for (contents, 0..) |other, j| if (j != k) try t.expect(std.mem.indexOf(u8, seg, other) == null);
    }
}

fn family(source: []const u8, config: []const u8, kind: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try @import("thinking_family_fixture.zig").init(t.allocator, t.io, source, config, kind);
    defer f.deinit();
    var script: Script = .{};
    const srv = try served(a, &f, &script);
    defer srv.deinit();
    try expectEveryStart(srv, a, try prepared(srv, a));
}

test "message starts: every message of Qwen3.8's and GLM-5.3's own templates starts at one" {
    try family(@embedFile("fixtures/thinking/qwen38.jinja"), @embedFile("fixtures/thinking/qwen38-tokenizer_config.json"), "qwen3_5");
    try family(@embedFile("fixtures/thinking/glm53.jinja"), @embedFile("fixtures/thinking/glm53-tokenizer_config.json"), "glm5_next");
}

test "message starts: Gemma 4's, Llama's and Nemotron's turn formats, tool turns joined to their reply included" {
    try family(gemma4, "{}", "gemma4");
    try family(llama, "{}", "llama");
    try family(nemotron, "{}", "nemotron_h");
}

test "message starts: the chunk plan of a request is main's, and a raw completion has no starts" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try @import("thinking_family_fixture.zig").init(t.allocator, t.io, @embedFile("fixtures/thinking/qwen38.jinja"), @embedFile("fixtures/thinking/qwen38-tokenizer_config.json"), "qwen3_5");
    defer f.deinit();
    var script: Script = .{};
    const srv = try served(a, &f, &script);
    defer srv.deinit();
    const r = try prepared(srv, a);
    // pinned from main f0a54bdf56: the starts must never move the prompt's cuts
    const chunks = try chunk_plan.withCut(a, try srv.chunks.starts(a, r.ids), 0, r.ids.len, srv.chunks.min_chunk);
    try t.expectEqualSlices(u32, &pinned_chunks, chunks);
    const Gone = struct {
        pub fn check(_: @This()) bool {
            return false;
        }
    };
    var actual_cx: errors.Cx = .{ .a = a };
    const messages = (try json.parse(a, talk)).ok;
    const actual = try @import("chat.zig").prepare(srv, &actual_cx, .{ .messages = messages, .fields = .null }, Gone{});
    defer srv.release(actual.preparing, actual.request);
    // Main's chat.prepare expression, frozen here: the comparison observes the actual prepared Request.chunks.
    const system_len = prompt_mod.systemPrefixLen(srv, &actual_cx, messages, &.{}, r.ids, false, null);
    const reference = try chunk_plan.withCut(a, try srv.chunks.starts(a, r.ids), if (srv.chunks.step > 0) @intCast(@max(system_len, 1) - 1) else 0, r.ids.len, srv.chunks.min_chunk);
    try t.expectEqualSlices(u32, reference, actual.request.chunks);
    try t.expectEqualSlices(u32, r.message_starts, actual.request.message_starts);
    var cx: errors.Cx = .{ .a = a };
    const raw = try prompt_mod.prepare(srv, &cx, .{ .prompt = .{ .text = "KUA KAB" }, .fields = .null }, false, null);
    try t.expectEqual(@as(usize, 0), raw.message_starts.len);
}

const pinned_chunks = [_]u32{ 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304 };
