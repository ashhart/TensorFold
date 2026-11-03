//! GGUF text uses its vocabulary, merge ranks, special-token kinds and template through the native text engines.
const std = @import("std");
const g = @import("core").gguf;
const tokenizer = @import("tokenizer");
const template = @import("template");
const A = std.mem.Allocator;
const Read = struct {
    doc: *const g.Document,
    a: A,
    problem: *[]const u8,
    fn fail(r: Read, key: []const u8, why: []const u8) !void {
        r.problem.* = try std.fmt.allocPrint(r.a, "{s}: {s}", .{ key, why });
        return error.UnsupportedGgufText;
    }
    fn get(r: Read, key: []const u8) !g.Value {
        return r.doc.get(key) orelse {
            try r.fail(key, "missing GGUF tokenizer key");
            unreachable;
        };
    }
    fn string(r: Read, key: []const u8) ![]const u8 {
        const v = try r.get(key);
        if (v != .string) try r.fail(key, "expected string");
        return v.string;
    }
    fn array(r: Read, key: []const u8, kind: g.MetadataKind) !g.Array {
        const v = try r.get(key);
        if (v != .array or v.array.kind != kind) try r.fail(key, "wrong array type");
        return v.array;
    }
    fn id(r: Read, key: []const u8, count: usize) !u32 {
        const v = (try r.get(key)).unsigned() catch {
            try r.fail(key, "expected unsigned token id");
            unreachable;
        };
        if (v >= count) try r.fail(key, "id outside vocabulary");
        return @intCast(v);
    }
};
const pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
pub fn tokenizerJson(a: A, pa: A, doc: *const g.Document, problem: *[]const u8) ![]u8 {
    const r: Read = .{ .doc = doc, .a = pa, .problem = problem };
    if (!std.mem.eql(u8, try r.string("tokenizer.ggml.model"), "gpt2")) try r.fail("tokenizer.ggml.model", "only gpt2 BPE is supported");
    if (!std.mem.eql(u8, try r.string("tokenizer.ggml.pre"), "qwen35")) try r.fail("tokenizer.ggml.pre", "only qwen35 is supported");
    const tokens = try r.array("tokenizer.ggml.tokens", .string);
    const types = try r.array("tokenizer.ggml.token_type", .int32);
    const merges = try r.array("tokenizer.ggml.merges", .string);
    if (tokens.len != types.len or tokens.len == 0) try r.fail("tokenizer.ggml.token_type", "count differs from vocabulary");
    var out: std.Io.Writer.Allocating = .init(a);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"normalizer\":{\"type\":\"NFC\"},\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":");
    try std.json.Stringify.value(pattern, .{}, w);
    try w.writeAll("},\"behavior\":\"Isolated\",\"invert\":false},{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":false}]},\"decoder\":{\"type\":\"ByteLevel\"},\"model\":{\"type\":\"BPE\",\"vocab\":{");
    var it = tokens.iterator();
    var id: usize = 0;
    while (try it.next()) |token| : (id += 1) {
        if (id != 0) try w.writeByte(',');
        try std.json.Stringify.value(token.string, .{}, w);
        try w.print(":{d}", .{id});
    }
    try w.writeAll("},\"merges\":[");
    it = merges.iterator();
    id = 0;
    while (try it.next()) |merge| : (id += 1) {
        if (id != 0) try w.writeByte(',');
        try std.json.Stringify.value(merge.string, .{}, w);
    }
    try w.writeAll("]},\"added_tokens\":[");
    it = tokens.iterator();
    var kinds = types.iterator();
    var first = true;
    id = 0;
    while (try it.next()) |token| : (id += 1) {
        const kind = (try kinds.next()).?.unsigned() catch {
            try r.fail("tokenizer.ggml.token_type", "invalid token kind");
            unreachable;
        };
        if (kind < 1 or kind > 6) try r.fail("tokenizer.ggml.token_type", "unsupported token kind");
        if (kind != 3 and kind != 4) continue;
        if (!first) try w.writeByte(',');
        first = false;
        try w.print("{{\"id\":{d},\"content\":", .{id});
        try std.json.Stringify.value(token.string, .{}, w);
        try w.print(",\"special\":{s},\"normalized\":false,\"single_word\":false,\"lstrip\":false,\"rstrip\":false}}", .{if (kind == 3) "true" else "false"});
    }
    try w.writeAll("]}");
    return out.toOwnedSlice();
}
pub fn load(comptime Text: type, gpa: A, io: std.Io, path: []const u8, override: ?[]const u8, pa: A, problem: *[]const u8) !*Text {
    var issue: g.FileProblem = .{};
    defer issue.deinit(gpa);
    var file = g.File.openDiagnosed(gpa, io, path, .{}, &issue) catch |err| {
        if (issue.name) |name| problem.* = try std.fmt.allocPrint(pa, "{s}: unsupported GGML quantization type {d}", .{ name, issue.kind });
        return err;
    };
    defer file.close(io);
    const doc = &file.document;
    const r: Read = .{ .doc = doc, .a = pa, .problem = problem };
    if (!std.mem.eql(u8, try r.string("general.architecture"), "qwen35")) try r.fail("general.architecture", "only qwen35 GGUF text is supported");
    const source = try r.string("tokenizer.chat_template");
    if (source.len == 0) try r.fail("tokenizer.chat_template", "empty template");
    const t = try gpa.create(Text);
    errdefer gpa.destroy(t);
    t.* = .{ .gpa = gpa, .arena = .init(gpa), .tok = undefined };
    errdefer t.arena.deinit();
    const a = t.arena.allocator();
    const json = try tokenizerJson(a, pa, doc, problem);
    t.tok = try tokenizer.parse(gpa, json);
    errdefer t.tok.deinit();
    const size = t.tok.id_to_token.tokens.items.len;
    t.eos = try a.dupe(u32, &.{try r.id("tokenizer.ggml.eos_token_id", size)});
    const bos = try r.id("tokenizer.ggml.bos_token_id", size);
    const add_bos = try r.get("tokenizer.ggml.add_bos_token");
    if (add_bos != .boolean) try r.fail("tokenizer.ggml.add_bos_token", "expected bool");
    if (add_bos.boolean) t.prefix = try a.dupe(u32, &.{bos});
    if (doc.get("tokenizer.ggml.add_eos_token")) |v| {
        if (v != .boolean) try r.fail("tokenizer.ggml.add_eos_token", "expected bool");
        if (v.boolean) t.suffix = try a.dupe(u32, t.eos);
    }
    var specials: std.json.ObjectMap = .empty;
    try specials.put(a, "bos_token", .{ .string = t.tok.id_to_token.get(bos).? });
    try specials.put(a, "eos_token", .{ .string = t.tok.id_to_token.get(t.eos[0]).? });
    t.specials = .{ .object = specials };
    t.source = if (override) |name| try std.Io.Dir.cwd().readFileAlloc(io, name, a, .limited(16 << 20)) else try a.dupe(u8, source);
    var diag: template.Diag = .{};
    defer if (diag.msg.len > 0) gpa.free(diag.msg);
    t.compiled = template.compile(gpa, t.source, &diag) catch |err| {
        problem.* = try std.fmt.allocPrint(pa, "tokenizer.chat_template: {s} ({s})", .{ diag.msg, @errorName(err) });
        return err;
    };
    errdefer if (t.compiled) |*compiled| compiled.deinit();
    t.tok.memo = try tokenizer.Memo.create(gpa, 64 << 20);
    return t;
}
