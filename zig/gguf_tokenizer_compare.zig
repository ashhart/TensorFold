//! CPU-only parity checks use the actual GGUF and MLX text providers without opening model weights.
const std = @import("std");
pub const gguf = @import("src/core/gguf.zig");
const HfText = @import("src/server/hf_text.zig").HfText;
const metadata = @import("src/families/qwen27/gguf_config.zig");
const json = @import("json");
const A = std.mem.Allocator;
const Counts = struct { strings: usize = 0, encodings: usize = 0, encoding_failures: usize = 0, decodings: usize = 0, decoding_failures: usize = 0, vocabulary: usize = 0, padding: usize = 0, vocabulary_failures: usize = 0, added: usize = 0, added_failures: usize = 0, stop_failures: usize = 0, chat_failures: usize = 0, chat_tokens: usize = 0 };
fn emit(a: A, value: anytype) !void {
    const text = try std.json.Stringify.valueAlloc(a, value, .{});
    defer a.free(text);
    std.debug.print("{s}\n", .{text});
}
fn vocabulary(g: *const HfText, m: *const HfText, doc: *const gguf.Document, counts: *Counts) !void {
    const view = try doc.tokenizer();
    var words = view.tokens.?.iterator();
    var types = view.types.?.iterator();
    var id: u32 = 0;
    while (try words.next()) |word| : (id += 1) {
        const kind = try (try types.next()).?.unsigned();
        const left = g.tok.id_to_token.get(id);
        const right = m.tok.id_to_token.get(id);
        counts.vocabulary += 1;
        if (kind == 5) {
            counts.padding += 1;
            if (left != null or right != null) counts.vocabulary_failures += 1;
        } else if (left == null or right == null or !std.mem.eql(u8, left.?, right.?) or !std.mem.eql(u8, left.?, word.string)) counts.vocabulary_failures += 1;
    }
    if (g.tok.added.len != m.tok.added.len) counts.added_failures += 1;
    for (m.tok.added) |want| {
        counts.added += 1;
        const got = for (g.tok.added) |entry| {
            if (entry.id == want.id) break entry;
        } else {
            counts.added_failures += 1;
            continue;
        };
        if (!std.mem.eql(u8, got.content, want.content) or got.special != want.special or got.lstrip != want.lstrip or got.rstrip != want.rstrip or got.normalized != want.normalized) counts.added_failures += 1;
    }
}
fn strings(a: A, g: *HfText, m: *HfText, cases: std.json.Value, counts: *Counts) !void {
    if (cases != .array) return error.ExpectedCaseArray;
    if (cases.array.items.len == 0) return error.EmptyCaseArray;
    counts.strings = cases.array.items.len;
    for (cases.array.items, 0..) |case, i| {
        const value = if (case == .object) case.object.get("text") orelse return error.ExpectedTextCase else case;
        if (value != .string) return error.ExpectedTextCase;
        const text = value.string;
        for ([_]bool{ false, true }) |plain| {
            const left = if (plain) try g.tok.encodePlain(a, text) else try g.tok.encode(a, text);
            defer a.free(left);
            const right = if (plain) try m.tok.encodePlain(a, text) else try m.tok.encode(a, text);
            defer a.free(right);
            counts.encodings += 1;
            if (!std.mem.eql(u32, left, right)) {
                counts.encoding_failures += 1;
                if (counts.encoding_failures <= 5) std.debug.print("encoding mismatch case={d} plain={}\n", .{ i, plain });
            }
            if (!plain and case == .object) if (case.object.get("ids")) |ids| {
                if (ids != .array or ids.array.items.len != right.len) return error.IndependentExpectedIdMismatch;
                for (ids.array.items, right) |want, got| if (want != .integer or want.integer < 0 or want.integer != got) return error.IndependentExpectedIdMismatch;
            };
            for ([_][]const u32{ left, right }) |tokens| for ([_]bool{ false, true }) |skip| {
                const l = try g.tok.decode(a, tokens, skip);
                defer a.free(l);
                const r = try m.tok.decode(a, tokens, skip);
                defer a.free(r);
                counts.decodings += 1;
                if (!std.mem.eql(u8, l, r)) counts.decoding_failures += 1;
            };
        }
    }
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(a);
    for (m.tok.id_to_token.tokens.items, 0..) |token, id| if (token != null) try ids.append(a, @intCast(id));
    for ([_]bool{ false, true }) |skip| {
        const l = try g.tok.decode(a, ids.items, skip);
        defer a.free(l);
        const r = try m.tok.decode(a, ids.items, skip);
        defer a.free(r);
        counts.decodings += 1;
        if (!std.mem.eql(u8, l, r)) counts.decoding_failures += 1;
    }
}
fn chat(a: A, g: *HfText, m: *HfText, counts: *Counts, problem: *[]const u8) !void {
    const messages = (try json.parse(a, "[{\"role\":\"system\",\"content\":\"Reply briefly.\"},{\"role\":\"user\",\"content\":\"What is 2 + 2? Include the word café.\"}]")).ok;
    const left = try g.text().render(a, messages, .{ .enable_thinking = false }, problem);
    const right = try m.text().render(a, messages, .{ .enable_thinking = false }, problem);
    const l = try left.encode(g.text(), a, problem);
    const r = try right.encode(m.text(), a, problem);
    counts.chat_tokens = r.len;
    if (!std.mem.eql(u8, left.text, right.text) or !std.mem.eql(u32, l, r)) counts.chat_failures += 1;
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 and args.len != 4) return error.ExpectedGgufAndMlxDirectoryThenOptionalCasesJson;
    if (!@import("src/server/gguf_path.zig").isFile(io, args[1]) or @import("src/server/gguf_path.zig").isFile(io, args[2])) return error.ExpectedGgufFileAndMlxDirectory;
    var issue: []const u8 = "";
    const g = try HfText.load(init.gpa, io, args[1], null, a, &issue);
    defer g.deinit();
    const m = try HfText.load(init.gpa, io, args[2], null, a, &issue);
    defer m.deinit();
    var file = try gguf.File.open(a, io, args[1], .{});
    defer file.close(io);
    var problem: metadata.Problem = .{};
    const config = try metadata.parse(&file.document, &problem);
    var counts: Counts = .{};
    if (!std.mem.eql(u32, g.eos, m.eos) or !std.mem.eql(u32, config.eos[0..config.eos_count], m.eos)) counts.stop_failures += 1;
    try emit(a, .{ .record = "stops", .gguf_text = g.eos, .gguf_model = config.eos[0..config.eos_count], .mlx = m.eos });
    try vocabulary(g, m, &file.document, &counts);
    const bytes = if (args.len == 4) try std.Io.Dir.cwd().readFileAlloc(io, args[3], a, .limited(8 << 20)) else @embedFile("testdata/qwen35-text-cases.json");
    const cases = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    try strings(init.gpa, g, m, cases, &counts);
    try chat(a, g, m, &counts, &issue);
    try emit(a, .{ .record = "summary", .same_native_engine_two_assets = true, .counts = counts });
    if (counts.encoding_failures + counts.decoding_failures + counts.vocabulary_failures + counts.added_failures + counts.stop_failures + counts.chat_failures != 0) return error.GgufTextParityFailed;
}
