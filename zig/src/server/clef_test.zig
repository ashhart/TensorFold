//! Clef: the encoder and head against Cloudflare's own Python (fp32, CPU) on recorded goldens when they are present.
const std = @import("std");
const json = @import("json");
const tokenizer = @import("tokenizer");
const enc = @import("clef_encode.zig");
const head = @import("clef_head.zig");

/// A safetensors file mapped read-only: `get(name)` as clef_head's loader reads it.
const Tensors = struct {
    map: std.Io.File.MemoryMap,
    file: std.Io.File,
    data: usize,
    header: std.json.Parsed(std.json.Value),

    const T = struct {
        dtype: enum { bf16, other },
        bytes: []const u8,
        n: usize,
        pub fn numel(t: T) usize {
            return t.n;
        }
    };

    fn open(a: std.mem.Allocator, io: std.Io, path: []const u8) !Tensors {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        const len: usize = @intCast(try file.length(io));
        const map = try std.Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        const hl: usize = @intCast(std.mem.readInt(u64, map.memory[0..8], .little));
        const header = try std.json.parseFromSlice(std.json.Value, a, map.memory[8..][0..hl], .{});
        return .{ .map = map, .file = file, .data = 8 + hl, .header = header };
    }
    fn close(t: *Tensors, io: std.Io) void {
        t.header.deinit();
        t.map.destroy(io);
        t.file.close(io);
    }
    pub fn get(t: *const Tensors, name: []const u8) ?T {
        const e = t.header.value.object.get(name) orelse return null;
        const off = e.object.get("data_offsets").?.array.items;
        const lo: usize = @intCast(off[0].integer);
        const hi: usize = @intCast(off[1].integer);
        var n: usize = 1;
        for (e.object.get("shape").?.array.items) |d| n *= @intCast(d.integer);
        const bf = std.mem.eql(u8, e.object.get("dtype").?.string, "BF16");
        return .{ .dtype = if (bf) .bf16 else .other, .bytes = t.map.memory[t.data + lo .. t.data + hi], .n = n };
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
    var weights = try Tensors.open(a, io, try std.fs.path.join(a, &.{ dir, "joint_head.safetensors" }));
    defer weights.close(io);
    var h = try head.Head.load(std.heap.page_allocator, &weights, .{});
    defer h.deinit();
    var rows = try Tensors.open(a, io, try std.fs.path.join(a, &.{ dir, "lm_head_rows.safetensors" }));
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
        const r = try enc.encode(a, &tok, record, 16384);
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
