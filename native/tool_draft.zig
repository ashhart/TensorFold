const std = @import("std");
const Tokenizer = @import("vendor/tokenizer.zig").Tokenizer;
const Proposal = @import("drafter.zig").Proposal;
const open = "<tool_call>";
const end = "\x00end";
const closing = "</function>\n</tool_call>";

pub const Schema = std.StringArrayHashMapUnmanaged([]const []const u8);

pub fn schema(a: std.mem.Allocator, tools: std.json.Value) !Schema {
    var result: Schema = .empty;
    if (tools == .null) return result;
    if (tools != .array) return error.InvalidToolSchema;
    for (tools.array.items) |tool| {
        if (tool != .object) return error.InvalidToolSchema;
        const function = tool.object.get("function") orelse .null;
        const spec = if (function == .object) function else tool;
        const name = spec.object.get("name") orelse continue;
        if (name != .string) return error.InvalidToolSchema;
        if (name.string.len == 0) continue;
        var parameters = spec.object.get("parameters") orelse .null;
        if (parameters == .null or (parameters == .object and parameters.object.count() == 0)) parameters = spec.object.get("input_schema") orelse .null;
        var names: std.ArrayList([]const u8) = .empty;
        if (parameters != .null) {
            if (parameters != .object) return error.InvalidToolSchema;
            const properties = parameters.object.get("properties") orelse .null;
            if (properties != .null) {
                if (properties != .object) return error.InvalidToolSchema;
                const required = parameters.object.get("required") orelse .null;
                if (required != .null) {
                    if (required != .array) return error.InvalidToolSchema;
                    for (required.array.items) |key| if (key == .string and properties.object.contains(key.string)) try names.append(a, try a.dupe(u8, key.string));
                }
                for (properties.object.keys()) |key| if (!contains(names.items, key)) try names.append(a, try a.dupe(u8, key));
            }
        }
        try result.put(a, try a.dupe(u8, name.string), try names.toOwnedSlice(a));
    }
    return result;
}

fn contains(options: []const []const u8, name: []const u8) bool {
    for (options) |option| if (std.mem.eql(u8, option, name)) return true;
    return false;
}

fn unique(partial: []const u8, options: []const []const u8) ?[]const u8 {
    var match: ?[]const u8 = null;
    for (options) |option| if (std.mem.startsWith(u8, option, partial)) {
        if (match != null) return null;
        match = option[partial.len..];
    };
    return match;
}

fn whitespace(codepoint: u21) bool {
    return switch (codepoint) {
        9...13, 28...32, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

fn trim(value: []const u8) []const u8 {
    var iterator = (std.unicode.Utf8View.init(value) catch return value).iterator();
    var first: ?usize = null;
    var last: usize = 0;
    while (iterator.i < value.len) {
        const before = iterator.i;
        if (!whitespace(iterator.nextCodepoint().?)) {
            if (first == null) first = before;
            last = iterator.i;
        }
    }
    return value[first orelse 0 .. last];
}

const Tag = struct { name: []const u8, end_at: usize, closed: bool };
fn tag(text: []const u8, prefix: []const u8, start: usize) ?Tag {
    const at = (std.mem.indexOfPos(u8, text, start, prefix) orelse return null) + prefix.len;
    const end_at = std.mem.indexOfAnyPos(u8, text, at, ">\n") orelse text.len;
    const closed = end_at < text.len and text[end_at] == '>';
    return .{ .name = text[at..end_at], .end_at = end_at + @intFromBool(closed), .closed = closed };
}

fn singleLine(a: std.mem.Allocator, name: []const u8) !bool {
    var lowered: std.ArrayList(u8) = .empty;
    var it = (try std.unicode.Utf8View.init(name)).iterator();
    while (it.nextCodepoint()) |cp| try lowered.append(a, switch (cp) {
        0...127 => std.ascii.toLower(@intCast(cp)),
        0x130, 0x131 => 'i',
        0x17f => 's',
        0x212a => 'k',
        else => 0,
    });
    for ([_][]const u8{ "path", "file", "dir", "name", "pattern", "glob", "url", "query", "id", "mode", "lang", "cwd", "limit", "offset" }) |word| if (std.mem.indexOf(u8, lowered.items, word) != null) return true;
    return false;
}

fn nextPart(a: std.mem.Allocator, remaining: []const []const u8) ![]const u8 {
    return if (remaining.len == 0) closing else try std.fmt.allocPrint(a, "<parameter={s}>\n", .{remaining[0]});
}

pub fn structure(a: std.mem.Allocator, specs: Schema, value: []const u8, open_at_start: bool, end_text: bool) !?[]const u8 {
    if (trim(value).len == 0) return if (open_at_start and specs.count() > 0) open ++ "\n<function=" else null;
    const start = std.mem.lastIndexOf(u8, value, open) orelse return null;
    const call = value[start..];
    if (std.mem.endsWith(u8, std.mem.trimEnd(u8, call, "\n"), "</tool_call>") and std.mem.count(u8, call, "</tool_call>") == 1) return if (end_text) end else null;
    if (std.mem.indexOf(u8, call, "</tool_call>") != null) return null;
    if (std.mem.endsWith(u8, call, "</function>")) return "\n</tool_call>";
    const function = tag(call, "<function=", 0) orelse return if (std.mem.eql(u8, trim(call), open)) "\n<function=" else null;
    if (!function.closed) {
        const rest = unique(function.name, specs.keys()) orelse return null;
        const name = try std.mem.concat(a, u8, &.{ function.name, rest });
        return try std.mem.concat(a, u8, &.{ rest, ">\n", try nextPart(a, specs.get(name).?) });
    }
    const params = specs.get(function.name) orelse return null;
    var used: std.ArrayList([]const u8) = .empty;
    var position: usize = 0;
    while (tag(call, "<parameter=", position)) |parameter| {
        if (parameter.closed and parameter.name.len > 0) try used.append(a, parameter.name);
        position = parameter.end_at;
    }
    var remaining: std.ArrayList([]const u8) = .empty;
    for (params) |parameter| if (!contains(used.items, parameter)) try remaining.append(a, parameter);
    if (trim(call[function.end_at..]).len == 0) return try std.mem.concat(a, u8, &.{ "\n", try nextPart(a, remaining.items) });
    position = 0;
    while (tag(call, "<parameter=", position)) |parameter| {
        position = parameter.end_at;
        // Python's '$' also matches immediately before a final newline.
        if (!parameter.closed and (parameter.end_at == call.len or (parameter.end_at + 1 == call.len and call[parameter.end_at] == '\n'))) {
            const rest = unique(parameter.name, remaining.items) orelse return null;
            return try std.mem.concat(a, u8, &.{ rest, ">\n" });
        }
    }
    if (std.mem.endsWith(u8, call, "</parameter>")) return try std.mem.concat(a, u8, &.{ "\n", try nextPart(a, remaining.items) });
    // The upstream regex selects the first qualifying header, even if later ones exist.
    position = 0;
    while (tag(call, "<parameter=", position)) |parameter| {
        position = parameter.end_at;
        if (!parameter.closed or parameter.name.len == 0 or position >= call.len or call[position] != '\n') continue;
        const body = call[position + 1 ..];
        if (std.mem.endsWith(u8, body, "\n") and std.mem.indexOf(u8, body, "</parameter>") == null and try singleLine(a, parameter.name) and std.mem.count(u8, body, "\n") == 1 and trim(body).len > 0) return try std.mem.concat(a, u8, &.{ "</parameter>\n", try nextPart(a, remaining.items) });
        break;
    }
    return null;
}

pub const Proposer = struct {
    arena: std.heap.ArenaAllocator,
    tokenizer: *const Tokenizer,
    specs: Schema,
    prompt_len: usize,
    open_at_start: bool = true,
    last_structural: bool = false,
    structural_proposals: usize = 0,
    structural_tokens: usize = 0,
    structural_accepted: usize = 0,
    fallback: @import("suffix_lookup.zig").Lookup,
    fallback_enabled: bool = true,
    decoded_upto: usize = 0,
    last_open: ?usize = null,
    seen_text: bool = false,
    output: []const u8 = "",
    output_arena: std.heap.ArenaAllocator,

    pub fn init(a: std.mem.Allocator, tokenizer: *const Tokenizer, tools: std.json.Value, prompt_len: usize) !Proposer {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const specs = try schema(arena.allocator(), tools);
        return .{ .arena = arena, .output_arena = .init(a), .tokenizer = tokenizer, .specs = specs, .prompt_len = prompt_len, .fallback = .{ .allocator = a } };
    }

    pub fn deinit(p: *Proposer) void {
        p.fallback.deinit();
        p.output_arena.deinit();
        p.arena.deinit();
    }

    fn decoded(p: *Proposer, context: []const i32) ![]const u8 {
        const count = context.len - p.prompt_len;
        if (count == p.decoded_upto) return p.output;
        _ = p.output_arena.reset(.retain_capacity);
        const a = p.output_arena.allocator();
        const emitted = context[p.prompt_len..];
        const opener = p.tokenizer.vocab.get(open) orelse p.tokenizer.unk_id;
        if (opener == null or count < p.decoded_upto) {
            p.output = try decode(p.tokenizer, a, emitted);
        } else {
            const added = emitted[p.decoded_upto..];
            for (added, p.decoded_upto..) |token, i| if (token == opener.?) {
                p.last_open = i;
            };
            if (!p.seen_text and added.len > 0 and trim(try decode(p.tokenizer, a, added)).len > 0) p.seen_text = true;
            p.output = if (p.last_open) |at| try decode(p.tokenizer, a, emitted[@min(at, count)..]) else if (p.seen_text) " ." else "";
        }
        p.decoded_upto = count;
        return p.output;
    }

    fn decode(tokenizer: *const Tokenizer, a: std.mem.Allocator, values: []const i32) ![]u8 {
        const ids = try a.alloc(u32, values.len);
        for (ids, values) |*out, token| out.* = @intCast(token);
        return tokenizer.decode(a, ids, false);
    }

    pub fn propose(p: *Proposer, a: std.mem.Allocator, context: []const i32, max_draft: usize) !Proposal {
        return p.proposeMode(a, context, max_draft, false);
    }

    pub fn proposeTree(p: *Proposer, a: std.mem.Allocator, context: []const i32, max_draft: usize) !Proposal {
        return p.proposeMode(a, context, max_draft, true);
    }

    fn proposeMode(p: *Proposer, a: std.mem.Allocator, context: []const i32, max_draft: usize, tree: bool) !Proposal {
        p.last_structural = false;
        p.fallback.last_confident = false;
        if (max_draft == 0) return .{};
        if (context.len < p.prompt_len) return error.InvalidDraftContext;
        var scratch = std.heap.ArenaAllocator.init(a);
        defer scratch.deinit();
        const allocator = scratch.allocator();
        const output = try p.decoded(context);
        const end_id = p.tokenizer.vocab.get("<|im_end|>") orelse p.tokenizer.unk_id;
        const follow = (try structure(allocator, p.specs, output, p.open_at_start, end_id != null)) orelse return if (p.fallback_enabled and !tree) p.fallback.propose(context, max_draft) else .{};
        const ids = if (std.mem.eql(u8, follow, end)) &[_]u32{end_id.?} else try p.tokenizer.encode(allocator, follow);
        var result = Proposal{};
        result.len = @min(@min(ids.len, max_draft), result.tokens.len);
        for (0..result.len) |index| {
            result.tokens[index] = @intCast(ids[index]);
            result.parents[index] = @as(i32, @intCast(index)) - 1;
            result.scores[index] = 0;
            result.probabilities[index] = 1;
        }
        if (result.len > 0) {
            p.last_structural = true;
            p.structural_proposals += 1;
            p.structural_tokens += result.len;
        }
        return result;
    }

    pub fn telemetry(p: *const Proposer) struct { structural_proposals: usize, structural_tokens: usize, structural_accepted: usize, proposals: usize, proposed_tokens: usize, judged_tokens: usize, accepted_tokens: usize, silenced_rounds: usize } {
        return .{ .structural_proposals = p.structural_proposals, .structural_tokens = p.structural_tokens, .structural_accepted = p.structural_accepted, .proposals = p.fallback.proposals, .proposed_tokens = p.fallback.proposed_tokens, .judged_tokens = p.fallback.judged_tokens, .accepted_tokens = p.fallback.accepted_tokens, .silenced_rounds = p.fallback.silenced_rounds };
    }

    pub fn observe(p: *Proposer, proposed: usize, accepted: usize) void {
        if (p.last_structural) p.structural_accepted += accepted else if (p.fallback_enabled) p.fallback.observe(proposed, accepted);
    }

    pub fn confident(p: *const Proposer) bool {
        return p.last_structural or (p.fallback_enabled and p.fallback.last_confident);
    }

    pub fn matchLength(p: *const Proposer) usize {
        return if (p.last_structural) 1 << 30 else if (p.fallback_enabled) p.fallback.last_match else 0;
    }
};
