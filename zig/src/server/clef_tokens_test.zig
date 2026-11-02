//! Clef's encoder against the reference's pinned ids and spans (fixtures/clef_tokens.json) when TF_CLEF_DIR is set.
const std = @import("std");
const json = @import("json");
const tokenizer = @import("tokenizer");
const enc = @import("clef_encode.zig");
const model_text = @import("model_text.zig");
const Allocator = std.mem.Allocator;

const fixture = @embedFile("fixtures/clef_tokens.json");

/// The checkpoint's tokenizer as the encoder asks for it: special-token text as its token, nothing added.
const Real = struct {
    tok: tokenizer.Tokenizer,

    const vtable: model_text.Text.VTable = .{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = source };

    fn text(r: *Real) model_text.Text {
        return .{ .ctx = r, .vtable = &vtable };
    }
    fn encode(ctx: *anyopaque, a: Allocator, s: []const u8, _: bool) model_text.Error![]u32 {
        const r: *Real = @ptrCast(@alignCast(ctx));
        return r.tok.encode(a, s) catch |e| if (e == error.OutOfMemory) error.OutOfMemory else error.Template;
    }
    fn decode(_: *anyopaque, a: Allocator, _: []const u32) Allocator.Error![]u8 {
        return a.alloc(u8, 0);
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, _: u32) Allocator.Error![]u8 {
        return a.alloc(u8, 0);
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 0;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, _: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return error.Template;
    }
    fn source(_: *anyopaque) []const u8 {
        return "";
    }
};

/// tokenizer.json with its first pre-tokenizer Regex replaced by tokenizer_config.json's `pretokenize_regex`.
fn withConfigRegex(a: Allocator, file: []const u8, config: []const u8) ![]const u8 {
    const cfg = try std.json.parseFromSliceLeaky(std.json.Value, a, config, .{});
    const regex = (cfg.object.get("pretokenize_regex") orelse return error.NoRegex).string;
    const pre = std.mem.indexOf(u8, file, "\"pre_tokenizer\"") orelse return error.NoRegex;
    const key = std.mem.indexOfPos(u8, file, pre, "\"Regex\"") orelse return error.NoRegex;
    const open = std.mem.indexOfScalarPos(u8, file, key + 7, '"') orelse return error.NoRegex;
    var end = open + 1;
    while (file[end] != '"') end += if (file[end] == '\\') 2 else 1;
    const quoted = try std.json.Stringify.valueAlloc(a, regex, .{});
    return std.mem.concat(a, u8, &.{ file[0..open], quoted, file[end + 1 ..] });
}

/// How many pinned records the tokenizer reproduces: ids, spans, option ids and types all equal.
fn matching(a: Allocator, real: *Real, cases: []json.Value, report: bool) !usize {
    var same: usize = 0;
    for (cases, 0..) |c, i| {
        const record = c.get("record").?;
        const max: u32 = @intCast(c.get("max_length").?.int64().?);
        const r = try enc.encode(a, real.text(), record.get("state").?, try enc.fields(a, record.get("questions").?), max);
        var ok = true;
        const ids = c.get("ids").?.array;
        if (ids.len != r.ids.len) ok = false else for (ids, r.ids) |w, got| {
            if (w.int64().? != got) ok = false;
        }
        const qs = c.get("questions").?.array;
        if (qs.len != r.questions.len) ok = false else for (qs, r.questions) |w, got| {
            const span = w.get("span").?.array;
            if (span[0].int64().? != got.span[0] or span[1].int64().? != got.span[1]) ok = false;
            if (w.get("type").?.int64().? != @backingInt(got.kind)) ok = false;
            if (!std.mem.eql(u8, w.get("id").?.string, got.id)) ok = false;
            const opts = w.get("options").?.array;
            const names = w.get("option_ids").?.array;
            if (opts.len != got.options.len) ok = false else for (opts, names, got.options, got.option_ids) |o, n, s, name| {
                if (o.array[0].int64().? != s[0] or o.array[1].int64().? != s[1] or !std.mem.eql(u8, n.string, name)) ok = false;
            }
        }
        if (ok) same += 1 else if (report) std.debug.print("Clef tokens: record {d} differs ({d} ids, want {d})\n", .{ i, r.ids.len, ids.len });
    }
    return same;
}

test "Clef: the encoder gives the reference's ids and spans with the checkpoint's tokenizer (TF_CLEF_DIR)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = (try json.parseText(a, fixture)).ok;
    const cases = doc.get("cases").?.array;
    try std.testing.expect(cases.len >= 12);
    const dir_z = std.c.getenv("TF_CLEF_DIR") orelse {
        std.debug.print("Clef tokens: skipped, TF_CLEF_DIR names no clef-flash checkpoint ({d} records pinned)\n", .{cases.len});
        return error.SkipZigTest;
    };
    const dir = std.mem.span(dir_z);
    const read = struct {
        fn f(x: Allocator, d: []const u8, name: []const u8) ![]u8 {
            return std.Io.Dir.cwd().readFileAlloc(std.testing.io, try std.fs.path.join(x, &.{ d, name }), x, .limited(1 << 30));
        }
    }.f;
    const file = try read(a, dir, "tokenizer.json");
    const config = try read(a, dir, "tokenizer_config.json");
    var real: Real = .{ .tok = try tokenizer.parse(std.testing.allocator, try withConfigRegex(a, file, config)) };
    defer real.tok.deinit();
    const same = try matching(a, &real, cases, true);
    var plain: Real = .{ .tok = try tokenizer.parse(std.testing.allocator, file) };
    defer plain.tok.deinit();
    const plain_same = try matching(a, &plain, cases, false);
    std.debug.print("Clef tokens: {d}/{d} records equal with pretokenize_regex, {d}/{d} with tokenizer.json's regex\n", .{ same, cases.len, plain_same, cases.len });
    try std.testing.expectEqual(cases.len, same);
}
