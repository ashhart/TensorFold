//! Reference goldens from TF_DSV41_MODEL (tokenizer.json dir) and TF_DSV41_GOLDEN (gen_*.py output); skipped if unset.
const std = @import("std");
const testing = std.testing;
const tokenizer = @import("tokenizer.zig");

pub fn env(name: []const u8) ?[]const u8 {
    return testing.environ.getPosix(name);
}

pub fn readGolden(a: std.mem.Allocator, name: []const u8) !?[]u8 {
    const dir = env("TF_DSV41_GOLDEN") orelse return null;
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(1 << 30)) catch |e| switch (e) {
        error.FileNotFound => null, // that generator was not run
        else => e,
    };
}

const Reader = struct {
    b: []const u8,
    at: usize = 0,
    fn int(r: *Reader) u32 {
        const v = std.mem.readInt(u32, r.b[r.at..][0..4], .little);
        r.at += 4;
        return v;
    }
    fn bytes(r: *Reader, n: usize) []const u8 {
        defer r.at += n;
        return r.b[r.at..][0..n];
    }
    fn ids(r: *Reader, a: std.mem.Allocator, n: usize) ![]u32 {
        const out = try a.alloc(u32, n);
        for (out) |*x| x.* = r.int();
        return out;
    }
};

pub fn loadTokenizer(a: std.mem.Allocator) !?*tokenizer.Tokenizer {
    const dir = env("TF_DSV41_MODEL") orelse return null;
    return try tokenizer.Tokenizer.load(a, testing.io, dir);
}

test "the release tokenizer encodes the golden corpus to tokenizers' ids exactly" {
    const a = testing.allocator;
    const tok = try loadTokenizer(a) orelse return error.SkipZigTest;
    defer tok.deinit();
    const data = try readGolden(a, "tokenizer.golden") orelse return error.SkipZigTest;
    defer a.free(data);
    var r: Reader = .{ .b = data };
    const enc = try tok.acquire();
    defer tok.release(enc);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    var docs: usize = 0;
    var bad: usize = 0;
    while (r.at < data.len) {
        const text = r.bytes(r.int());
        const want = try r.ids(a, r.int());
        defer a.free(want);
        out.clearRetainingCapacity();
        try enc.encode(a, text, &out);
        docs += 1;
        if (!std.mem.eql(u32, want, out.items)) {
            bad += 1;
            if (bad <= 5) {
                var first: usize = 0;
                while (first < @min(want.len, out.items.len) and want[first] == out.items[first]) first += 1;
                std.debug.print("doc {d}: ids differ at {d} (want {d} got {d}); text {f}\n", .{ docs - 1, first, want.len, out.items.len, std.json.fmt(text[0..@min(text.len, 160)], .{}) });
                std.debug.print("  want {any}\n  got  {any}\n", .{ want[first..@min(want.len, first + 8)], out.items[first..@min(out.items.len, first + 8)] });
            }
        }
    }
    if (bad > 0 or env("TF_DSV41_GOLDEN") != null) std.debug.print("tokenizer golden: {d}/{d} documents equal\n", .{ docs - bad, docs });
    try testing.expectEqual(@as(usize, 0), bad);
}

test "the release tokenizer decodes ids to tokenizers' text exactly, whole and a token at a time" {
    const a = testing.allocator;
    const tok = try loadTokenizer(a) orelse return error.SkipZigTest;
    defer tok.deinit();
    const data = try readGolden(a, "decode.golden") orelse return error.SkipZigTest;
    defer a.free(data);
    var r: Reader = .{ .b = data };
    var cases: usize = 0;
    var bad: usize = 0;
    var streamed: std.ArrayList(u8) = .empty;
    defer streamed.deinit(a);
    while (r.at < data.len) {
        const ids = try r.ids(a, r.int());
        defer a.free(ids);
        const want = r.bytes(r.int());
        const got = try tok.decodeAlloc(a, ids);
        defer a.free(got);
        var d: tokenizer.Detokenizer = .{ .tok = tok };
        streamed.clearRetainingCapacity();
        for (ids) |id| try d.push(a, id, &streamed);
        try d.flush(a, &streamed);
        cases += 1;
        if (!std.mem.eql(u8, want, got) or !std.mem.eql(u8, want, streamed.items)) {
            bad += 1;
            if (bad <= 5) std.debug.print("decode case {d}: want {f}\n got {f}\n", .{ cases - 1, std.json.fmt(want[0..@min(want.len, 120)], .{}), std.json.fmt(got[0..@min(got.len, 120)], .{}) });
        }
    }
    if (bad > 0 or env("TF_DSV41_GOLDEN") != null) std.debug.print("decode golden: {d}/{d} cases equal\n", .{ cases - bad, cases });
    try testing.expectEqual(@as(usize, 0), bad);
}

const json = @import("json");
const template = @import("template.zig");

fn imageText(ctx: ?*anyopaque, _: json.Value) []const u8 {
    const s: *const []const u8 = @ptrCast(@alignCast(ctx.?));
    return s.*;
}

/// Each line of a gen_template.py golden rendered here: equal text, or a refusal where the reference refused.
fn checkTemplates(lines: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var it = std.mem.splitScalar(u8, lines, '\n');
    var cases: usize = 0;
    var bad: usize = 0;
    var jinja: usize = 0;
    var efforts: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();
        const c = (try json.parseText(a, line)).ok;
        cases += 1;
        var image_text: []const u8 = "";
        var opts: template.Options = .{
            .thinking = c.get("thinking").?.bool,
            .budget = template.budgetOf(c.get("budget").?) orelse 0,
            .drop_thinking = c.get("drop_thinking").?.bool,
            .add_generation_prompt = c.get("generation").?.bool,
        };
        if (c.get("effort_field")) |field| {
            // the request's reasoning_effort resolved here, as the server resolves it, must give the app's choice
            const text: []const u8 = switch (field) {
                .string, .int => |t| t,
                else => unreachable,
            };
            const level = template.effortOf(text) orelse return error.TestUnexpectedRefusal;
            opts.thinking = level != null;
            opts.budget = level orelse 75;
            try testing.expectEqual(c.get("thinking").?.bool, opts.thinking);
            efforts += 1;
        }
        if (c.get("tools")) |t| if (t == .array) {
            opts.tools = t.array;
        };
        if (c.get("response_format")) |rf| if (rf != .null) {
            opts.response_format = rf;
        };
        if (c.get("image_text")) |t| if (t == .string) {
            image_text = t.string;
            opts.image = .{ .ctx = @ptrCast(&image_text), .text = imageText };
        };
        var problem: template.Problem = .{};
        const got = template.render(a, c.get("messages").?, opts, &problem);
        const want = c.get("expected").?;
        const ok = if (want == .string) (if (got) |g| std.mem.eql(u8, g, want.string) else |_| false) else (if (got) |_| false else |e| e == error.Refused);
        if (c.get("jinja")) |j| if (j == .string) {
            jinja += 1;
        };
        if (!ok) {
            bad += 1;
            if (bad <= 3) {
                std.debug.print("template case {d} differs ({s})\n", .{ cases - 1, if (want == .string) "want text" else c.get("error").?.string });
                if (got) |g| {
                    if (want == .string) {
                        var k: usize = 0;
                        while (k < @min(g.len, want.string.len) and g[k] == want.string[k]) k += 1;
                        std.debug.print("  at byte {d}:\n  want {f}\n  got  {f}\n", .{ k, std.json.fmt(want.string[k..@min(want.string.len, k + 160)], .{}), std.json.fmt(g[k..@min(g.len, k + 160)], .{}) });
                    }
                } else |_| std.debug.print("  refused: {s}\n", .{problem.message});
            }
        }
    }
    if (bad > 0 or env("TF_DSV41_GOLDEN") != null) std.debug.print("template golden: {d}/{d} equal to encoding.encode ({d} reasoning_effort values resolved as app.options does; {d} also the release chat_template.jinja's text)\n", .{ cases - bad, cases, efforts, jinja });
    try testing.expectEqual(@as(usize, 0), bad);
}

test "the chat encoding renders the committed conversations as the Python engine and the release template do" {
    try checkTemplates(@import("dsv41_serve_fixtures").template);
}

test "the chat encoding renders the full generated set (TF_DSV41_GOLDEN/template.jsonl)" {
    const data = try readGolden(testing.allocator, "template.jsonl") orelse return error.SkipZigTest;
    defer testing.allocator.free(data);
    try checkTemplates(data);
}

const tools = @import("tools.zig");

fn deltaEqual(want: json.Value, got: tools.Delta) bool {
    const w = want.array;
    const kind = w[0].string;
    return switch (got) {
        .reasoning => |t| std.mem.eql(u8, kind, "r") and std.mem.eql(u8, w[1].string, t),
        .content => |t| std.mem.eql(u8, kind, "c") and std.mem.eql(u8, w[1].string, t),
        .call => |c| std.mem.eql(u8, kind, "open") and w[1].int64().? == c.index and std.mem.eql(u8, w[2].string, c.name),
        .arguments => |g| std.mem.eql(u8, kind, "args") and w[1].int64().? == g.index and std.mem.eql(u8, w[2].string, g.text),
    };
}

fn callsEqual(a: std.mem.Allocator, want: json.Value, got: []const tools.Call) !bool {
    if (want.array.len != got.len) return false;
    for (want.array, got) |w, c| {
        if (!std.mem.eql(u8, w.array[0].string, c.name)) return false;
        if (!std.mem.eql(u8, w.array[1].string, try c.arguments(a))) return false;
    }
    return true;
}

/// Each recorded reply parsed whole and fed in the recorded pieces: the same parts, deltas at every feed, and calls.
fn checkDsml(lines: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var it = std.mem.splitScalar(u8, lines, '\n');
    var cases: usize = 0;
    var bad: usize = 0;
    var feeds: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();
        const c = (try json.parseText(a, line)).ok;
        cases += 1;
        const text = c.get("text").?.string;
        const thinking = c.get("thinking").?.bool;
        var offered = try tools.Tools.init(a, if (c.get("tools").? == .array) c.get("tools").?.array else &.{});
        offered.unknown_calls = false; // the reference behaviour: a call to a tool not offered stays in the content
        var why: ?[]const u8 = null;
        const p = try tools.parse(a, text, thinking, offered, .{});
        const wp = c.get("parse").?;
        if (!std.mem.eql(u8, wp.get("reasoning").?.string, p.reasoning)) why = "parse reasoning";
        if (why == null and !std.mem.eql(u8, wp.get("content").?.string, p.content)) why = "parse content";
        if (why == null and !try callsEqual(a, wp.get("calls").?, p.calls)) why = "parse calls";
        var s = tools.Stream.init(a, thinking, offered, .{});
        var out: std.ArrayList(tools.Delta) = .empty;
        const cuts = c.get("cuts").?.array;
        const deltas = c.get("deltas").?.array;
        for (deltas, 0..) |want, k| {
            if (why != null) break;
            out.clearRetainingCapacity();
            if (k < cuts.len) try s.feed(text[0..@intCast(cuts[k].int64().?)], false, &out) else try s.finish(text, &out);
            feeds += 1;
            if (want.array.len != out.items.len) {
                why = "stream delta count";
            } else for (want.array, out.items) |w, g| {
                if (!deltaEqual(w, g)) why = "stream delta";
            }
            if (why != null and bad < 3) std.debug.print("  feed {d}/{d}: want {d} deltas, got {d}\n", .{ k, deltas.len, want.array.len, out.items.len });
        }
        if (why == null and !try callsEqual(a, c.get("stream_calls").?, s.calls.items)) why = "stream calls";
        if (why) |w| {
            bad += 1;
            if (bad <= 3) std.debug.print("dsml case {d}: {s}; thinking={} text {f}\n", .{ cases - 1, w, thinking, std.json.fmt(text[0..@min(text.len, 400)], .{}) });
        }
    }
    if (bad > 0 or env("TF_DSV41_GOLDEN") != null) std.debug.print("dsml golden: {d}/{d} replies equal to dsml.py ({d} stream feeds)\n", .{ cases - bad, cases, feeds });
    try testing.expectEqual(@as(usize, 0), bad);
}

test "DSML replies parse and stream as the Python engine's dsml.py does (committed set)" {
    try checkDsml(@import("dsv41_serve_fixtures").dsml);
}

test "DSML replies parse and stream as dsml.py does (TF_DSV41_GOLDEN/dsml.jsonl)" {
    const data = try readGolden(testing.allocator, "dsml.jsonl") orelse return error.SkipZigTest;
    defer testing.allocator.free(data);
    try checkDsml(data);
}

const sampling = @import("sampling.zig");

/// Each rank's best ``k`` of its vocabulary half (``pick.top``), gathered in rank order, as the engine hands them.
fn gather(a: std.mem.Allocator, row: []const f32, world: usize, k: usize, values: *std.ArrayList(f32), ids: *std.ArrayList(u32)) !void {
    values.clearRetainingCapacity();
    ids.clearRetainingCapacity();
    const Key = struct {
        fn of(v: f32, id: u32) u64 {
            const bits: u32 = @bitCast(v + 0.0);
            const mono: u32 = if (bits & 0x8000_0000 == 0) bits | 0x8000_0000 else ~bits;
            return @as(u64, mono) << 32 | (0xFFFF_FFFF - id);
        }
    };
    var keys: std.ArrayList(u64) = .empty;
    defer keys.deinit(a);
    var lo: usize = 0;
    for (0..world) |w| {
        // numpy's array_split: the first (len % world) halves one longer
        const size = row.len / world + @intFromBool(w < row.len % world);
        keys.clearRetainingCapacity();
        for (lo..lo + size) |i| try keys.append(a, Key.of(row[i], @intCast(i)));
        std.sort.pdq(u64, keys.items, {}, std.sort.desc(u64));
        for (keys.items[0..@min(k, size)]) |kk| {
            const id = 0xFFFF_FFFF - @as(u32, @truncate(kk));
            try values.append(a, row[id]);
            try ids.append(a, id);
        }
        lo += size;
    }
}

fn checkSampling(lines: []const u8) !void {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var picker = try sampling.Picker.init(a, 129280 * 2);
    defer picker.deinit();
    var values: std.ArrayList(f32) = .empty;
    defer values.deinit(a);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(a);
    var it = std.mem.splitScalar(u8, lines, '\n');
    var rows: usize = 0;
    var bad: usize = 0;
    var fallbacks: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        _ = arena.reset(.retain_capacity);
        const x = arena.allocator();
        const c = (try json.parseText(x, line)).ok;
        rows += 1;
        const b64 = c.get("logits").?.string;
        const raw = try x.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(b64));
        try std.base64.standard.Decoder.decode(raw, b64);
        const logits = try x.alloc(f32, raw.len / 4);
        for (logits, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));
        const world: usize = @intCast(c.get("world").?.int64().?);
        const count: usize = @intCast(c.get("count").?.int64().?);
        const sv = c.get("sampling").?;
        const s: ?sampling.Sampling = if (sv == .array) .{
            .seed = @intCast(std.fmt.parseInt(u64, sv.array[0].int, 10) catch unreachable),
            .temperature = sv.array[1].float64().?,
            .top_k = @intCast(sv.array[2].int64().?),
            .top_p = sv.array[3].float64().?,
            .min_p = sv.array[4].float64().?,
        } else null;
        const position: u64 = @intCast(c.get("position").?.int64().?);
        try testing.expectEqual(count, sampling.candidateCount(s, @intCast(logits.len), sampling.nucleus_count));
        var stats: std.ArrayList([2]f64) = .empty;
        for (c.get("stats").?.array) |st| try stats.append(x, .{ st.array[0].float64().?, st.array[1].float64().? });
        try gather(a, logits, world, count, &values, &ids);
        var got = picker.pick(.{ .values = values.items, .ids = ids.items, .stats = stats.items, .count = count }, position, s);
        const full = got == .full_row;
        if (full) {
            fallbacks += 1;
            try gather(a, logits, world, logits.len, &values, &ids);
            got = picker.pick(.{ .values = values.items, .ids = ids.items }, position, s);
        }
        const want: u32 = @intCast(c.get("token").?.int64().?);
        if (got != .token or got.token != want or full != c.get("full").?.bool) {
            bad += 1;
            if (bad <= 5) std.debug.print("sampling row {d}: want {d} (full {}) got {any} (full {}); vocab {d} world {d} count {d} sampling {any}\n", .{ rows - 1, want, c.get("full").?.bool, got, full, logits.len, world, count, s });
        }
    }
    if (bad > 0 or env("TF_DSV41_GOLDEN") != null) std.debug.print("sampling golden: {d}/{d} rows picked as the Python engine picks them ({d} nucleus rows on the whole row)\n", .{ rows - bad, rows, fallbacks });
    try testing.expectEqual(@as(usize, 0), bad);
}

test "rows sample as the Python engine's host rule does (TF_DSV41_GOLDEN/sampling.jsonl)" {
    const data = try readGolden(testing.allocator, "sampling.jsonl") orelse return error.SkipZigTest;
    defer testing.allocator.free(data);
    try checkSampling(data);
}

test "rows sample as the Python engine's host rule does (committed set)" {
    try checkSampling(@import("dsv41_serve_fixtures").sampling);
}
