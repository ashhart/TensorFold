//! Clef: the encoder and head against Cloudflare's own Python (fp32, CPU) on recorded goldens when they are present.
const std = @import("std");
const json = @import("json");
const tokenizer = @import("tokenizer");
const enc = @import("clef_encode.zig");
const head = @import("clef_head.zig");

/// The tokenizer as clef_encode asks: ids of text, special-token text as its token.
const Ids = struct {
    t: *const tokenizer.Tokenizer,
    pub fn ids(s: Ids, a: std.mem.Allocator, text: []const u8) ![]u32 {
        return s.t.encode(a, text);
    }
};

fn env(name: [*:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

test "Clef: render is json.dumps with sorted keys, compact separators and text as itself" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = (try json.parseText(a, "{\"b\":[1,2.5,1e20,-0.0,true,null],\"a\":\"caf\\u00e9 \\\"q\\\"\\n\",\"\\u00e9\":{\"z\":1,\"y\":0.1}}")).ok;
    try std.testing.expectEqualStrings("{\"a\":\"café \\\"q\\\"\\n\",\"b\":[1,2.5,1e+20,-0.0,true,null],\"é\":{\"y\":0.1,\"z\":1}}", try enc.render(a, parsed));
    try std.testing.expectEqualStrings("plain text", try enc.render(a, .{ .string = "plain text" }));
}

test "Clef: records encode and score as Cloudflare's Python does (TF_CLEF_DIR, TF_CLEF_GOLDEN)" {
    const dir = env("TF_CLEF_DIR") orelse return error.SkipZigTest;
    const gold = env("TF_CLEF_GOLDEN") orelse return error.SkipZigTest;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tok = try tokenizer.loadTokenizer(io, a, try std.fs.path.join(a, &.{ dir, "tokenizer.json" }));
    var weights = try head.Files.open(a, io, try std.fs.path.join(a, &.{ dir, "joint_head.safetensors" }));
    defer weights.close(io);
    var h = try head.Head.load(std.heap.page_allocator, &weights, .{});
    defer h.deinit();
    var rows = try head.Files.open(a, io, try std.fs.path.join(a, &.{ dir, "lm_head_rows.safetensors" }));
    defer rows.close(io);
    const table = rows.get("lm_head.weight").?;
    const lex: head.Lexical = .{ .table = table.bytes, .rows = table.n / 5120, .width = 5120 };
    var i: usize = 0;
    while (true) : (i += 1) {
        const rec_path = try std.fmt.allocPrint(a, "{s}/rec{d}.json", .{ gold, i });
        const rec_text = std.Io.Dir.cwd().readFileAlloc(io, rec_path, a, .unlimited) catch break;
        const g_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/gold{d}.json", .{ gold, i }), a, .unlimited);
        const record = (try json.parseText(a, rec_text)).ok;
        const g = try std.json.parseFromSliceLeaky(std.json.Value, a, g_text, .{});
        const r = try enc.encode(a, Ids{ .t = &tok }, record, 16384);
        const want = g.object.get("input_ids").?.array.items;
        try std.testing.expectEqual(want.len, r.ids.len);
        for (want, r.ids) |w, got| try std.testing.expectEqual(@as(u32, @intCast(w.integer)), got);
        const gq = g.object.get("questions").?.array.items;
        try std.testing.expectEqual(gq.len, r.questions.len);
        for (gq, r.questions) |q, mine| {
            const span = q.object.get("span").?.array.items;
            try std.testing.expectEqual(@as(u32, @intCast(span[0].integer)), mine.span[0]);
            try std.testing.expectEqual(@as(u32, @intCast(span[1].integer)), mine.span[1]);
            for (q.object.get("option_ids").?.array.items, mine.option_ids) |o, m| try std.testing.expectEqualStrings(o.string, m);
        }
        const hb = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/hidden{d}.f32", .{ gold, i }), a, .unlimited);
        const hidden: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, hb));
        const t0 = std.Io.Clock.awake.now(io).toMilliseconds();
        const logits = try h.forward(a, hidden, r, lex);
        const ms = std.Io.Clock.awake.now(io).toMilliseconds() - t0;
        var worst: f64 = 0;
        var worst_p: f64 = 0;
        for (g.object.get("logits").?.array.items, logits) |wl, mine| {
            const want_l = wl.array.items;
            var pw: [256]f64 = undefined;
            var pm: [256]f64 = undefined;
            for (want_l, mine, 0..) |w, m, k| {
                const wf: f64 = switch (w) {
                    .float => |f| f,
                    .integer => |n| @floatFromInt(n),
                    else => unreachable,
                };
                worst = @max(worst, @abs(wf - m));
                pw[k] = wf;
                pm[k] = m;
            }
            worst_p = @max(worst_p, maxProbDiff(pw[0..mine.len], pm[0..mine.len]));
        }
        std.debug.print("Clef golden {d}: {d} rows, {d} questions, max |logit diff| {e}, max |prob diff| {e}, head {d} ms\n", .{ i, r.ids.len, r.questions.len, worst, worst_p, ms });
        try std.testing.expect(worst_p < 1e-4);
    }
    try std.testing.expect(i > 0);
}

fn maxProbDiff(x: []const f64, y: []const f64) f64 {
    var mx: f64 = -std.math.inf(f64);
    var my: f64 = -std.math.inf(f64);
    for (x, y) |a, b| {
        mx = @max(mx, a);
        my = @max(my, b);
    }
    var sx: f64 = 0;
    var sy: f64 = 0;
    for (x, y) |a, b| {
        sx += @exp(a - mx);
        sy += @exp(b - my);
    }
    var worst: f64 = 0;
    for (x, y) |a, b| worst = @max(worst, @abs(@exp(a - mx) / sx - @exp(b - my) / sy));
    return worst;
}

/// The tiny fixture's tensors as clef_head's loader reads them: bf16 bits from JSON, by name.
const Tiny = struct {
    names: std.StringHashMap([]const u8),
    pub fn get(t: *const Tiny, name: []const u8) ?head.Files.T {
        const b = t.names.get(name) orelse return null;
        return .{ .dtype = .bf16, .bytes = b, .n = b.len / 2 };
    }
};

fn bf16Bytes(a: std.mem.Allocator, v: std.json.Value) ![]u8 {
    const items = v.array.items;
    const out = try a.alloc(u8, items.len * 2);
    for (items, 0..) |x, i| std.mem.writeInt(u16, out[2 * i ..][0..2], @intCast(x.integer), .little);
    return out;
}

fn num(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        else => unreachable,
    };
}

test "Clef: a tiny random head equals Cloudflare's JointSchemaHead, and 1 and 4 threads give the same bits" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // fixtures/clef_tiny.json: tools/clef_tiny_golden.py ran Cloudflare's joint_schema_model.py on these weights
    const g = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/clef_tiny.json"), .{});
    const c = g.object.get("config").?.object;
    const cfg: head.Config = .{ .hidden = @intCast(c.get("hidden").?.integer), .width = @intCast(c.get("width").?.integer), .routing = @intCast(c.get("routing").?.integer), .layers = @intCast(c.get("layers").?.integer), .heads = @intCast(c.get("heads").?.integer), .ff = @intCast(c.get("ff").?.integer) };
    var tiny: Tiny = .{ .names = .init(a) };
    var it = g.object.get("tensors").?.object.iterator();
    while (it.next()) |e| try tiny.names.put(e.key_ptr.*, try bf16Bytes(a, e.value_ptr.*));
    const table = try bf16Bytes(a, g.object.get("lm_head").?);
    const lex: head.Lexical = .{ .table = table, .rows = table.len / 2 / cfg.hidden, .width = cfg.hidden };
    const hv = g.object.get("hidden").?.array.items;
    const hidden = try a.alloc(f32, hv.len);
    for (hidden, hv) |*h, v| h.* = @floatCast(num(v));
    const iv = g.object.get("ids").?.array.items;
    const ids = try a.alloc(u32, iv.len);
    for (ids, iv) |*d, v| d.* = @intCast(v.integer);
    const qv = g.object.get("questions").?.array.items;
    const qs = try a.alloc(enc.Question, qv.len);
    for (qs, qv) |*q, v| {
        const sp = v.object.get("span").?.array.items;
        const ov = v.object.get("options").?.array.items;
        const opts = try a.alloc([2]u32, ov.len);
        const oids = try a.alloc([]const u8, ov.len);
        for (opts, oids, ov) |*o, *id, x| {
            o.* = .{ @intCast(x.array.items[0].integer), @intCast(x.array.items[1].integer) };
            id.* = "";
        }
        q.* = .{ .id = "", .kind = @fromBackingInt(@intCast(@as(u8, @intCast(v.object.get("kind").?.integer)))), .span = .{ @intCast(sp[0].integer), @intCast(sp[1].integer) }, .options = opts, .option_ids = oids };
    }
    const rec: enc.Record = .{ .ids = ids, .questions = qs, .state_rows = @intCast(ids.len) };
    var got: [2][][]f32 = undefined;
    for ([_]usize{ 1, 4 }, 0..) |threads, k| {
        var c2 = cfg;
        c2.threads = threads;
        var h = try head.Head.load(std.testing.allocator, &tiny, c2);
        defer h.deinit();
        got[k] = try h.forward(a, hidden, rec, lex);
    }
    for (got[0], got[1]) |x, y| for (x, y) |p, q| try std.testing.expectEqual(@as(u32, @bitCast(p)), @as(u32, @bitCast(q)));
    for (g.object.get("logits").?.array.items, got[0]) |want, mine| for (want.array.items, mine) |w, m| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(num(w))), m, 1e-4);
    };
}
