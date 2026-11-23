//! tf-inkling2-engine: the serving engine's checks on one or two ranks (both run the same loop, no plan words).
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");

const usage = "usage: tf-inkling2-engine graphs|greedy <model dir> <ids.json> <out dir> [context] [new]\n" ++
    "  graphs: decode rows eager in some slots and graphed in others on the same rows; logits must match bit for bit\n" ++
    "  greedy: each text's prompt pass then greedy decode rows, eager (writes greedy.json: name -> tokens)\n" ++
    "  two ranks: TF_INKLING2_RANK, TF_INKLING2_WORLD and NCCL_COMM_ID on both\n";

const Text = struct { name: []const u8, ids: []u32 };

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const graphs = std.mem.eql(u8, args[1], "graphs");
    if (!graphs and !std.mem.eql(u8, args[1], "greedy")) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[3], a, .limited(1 << 26));
    var doc = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer doc.deinit();
    var list: std.ArrayList(Text) = .empty;
    var it = doc.value.object.iterator();
    while (it.next()) |e| {
        const v = e.value_ptr.array.items;
        const ids = try a.alloc(u32, v.len);
        for (v, ids) |x, *id| id.* = @intCast(x.integer);
        try list.append(a, .{ .name = e.key_ptr.*, .ids = ids });
    }
    const context = if (args.len > 5) try std.fmt.parseInt(usize, args[5], 10) else 32768;
    const new = if (args.len > 6) try std.fmt.parseInt(usize, args[6], 10) else 64;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    if (try ctx.capability() != 121) return error.Inkling2RequiresSm121;
    const place = try ink.admit.place();
    var group: ?cuda.tp_group.Group = null;
    defer if (group) |*g| g.close(); // after the engine (its graphs hold the comm's kernels), as the server closes
    const e = try ink.engine.Engine.init(gpa, io, &driver, args[2], place.rank, place.world, .{ .capacity = ink.native_exl3.capacity(context) });
    defer e.deinit();
    if (place.world > 1) {
        group = try cuda.tp_group.Group.open(&driver, place.rank, place.world);
        try group.?.openRoce(gpa, io, &driver, e.stream, e.roceBytes());
        e.exchange = .{ .nccl = .{ .lib = &group.?.lib, .comm = group.?.comm, .group = &group.? } };
    }
    try e.reserve(16);
    try std.Io.Dir.cwd().createDirPath(io, args[4]);
    var out = try std.Io.Dir.cwd().openDir(io, args[4], .{});
    defer out.close(io);
    const r0 = place.rank == 0;
    if (!graphs) {
        e.graphs.settings.on = false;
        var json: std.ArrayList(u8) = .empty;
        try json.appendSlice(a, "{");
        for (list.items, 0..) |t, i| {
            const toks = try greedy(e, a, t.ids, new);
            try json.print(a, "{s}\"{s}\": [", .{ if (i > 0) ", " else "", t.name });
            for (toks, 0..) |x, j| try json.print(a, "{s}{d}", .{ if (j > 0) ", " else "", x });
            try json.appendSlice(a, "]");
            if (r0) std.debug.print("{s}: {d} prompt tokens, {d} greedy tokens\n", .{ t.name, t.ids.len, toks.len });
        }
        try json.appendSlice(a, "}\n");
        if (r0) try out.writeFile(io, .{ .sub_path = "greedy.json", .data = json.items });
        return 0;
    }
    // cases: k streams, `steps` decode rows each; a text of 16,380+ tokens also crosses a global chunk
    const Case = struct { k: usize, steps: usize, at: usize };
    var cases: std.ArrayList(Case) = .empty;
    for ([_]usize{ 1, 2, 3, 4, 8 }) |k| try cases.append(a, .{ .k = k, .steps = 6, .at = 0 });
    for (list.items) |t| if (t.ids.len >= 16390) {
        try cases.append(a, .{ .k = 1, .steps = 8, .at = 16380 });
        try cases.append(a, .{ .k = 2, .steps = 8, .at = 16380 });
        break;
    };
    var pass: usize = 0;
    var total: usize = 0;
    for (cases.items) |cs| {
        const diff = try graphCase(e, a, list.items, cs.k, cs.steps, cs.at);
        total += cs.steps;
        pass += cs.steps - diff;
        if (r0) std.debug.print("case {d} streams from {s}: {d}/{d} steps equal\n", .{ cs.k, if (cs.at > 0) "16380" else "prompt end", cs.steps - diff, cs.steps });
    }
    const st = e.graphs.stats;
    if (r0) std.debug.print("graphs: {d}/{d} steps bit-identical; captured {d}, replayed {d}, eager {d}\n", .{ pass, total, st.captured, st.replayed, st.eager });
    return if (pass == total) 0 else 1;
}

/// A text's prompt pass on the bound slot then `n` greedy tokens, one decode row at a time.
fn greedy(e: *ink.engine.Engine, a: std.mem.Allocator, ids: []const u32, n: usize) ![]u32 {
    try e.bind(0);
    e.begin(null);
    var at: usize = 0;
    var tok: u32 = while (true) : (at += ink.engine.prompt_rows) {
        if (try e.prefillChunk(ids, at)) |t| break t;
    };
    const toks = try a.alloc(u32, n);
    for (toks) |*x| {
        x.* = tok;
        var o: [1]u32 = undefined;
        try e.verifyShared(&.{.{ .slot = 0, .ids = &.{tok}, .out = &o }});
        tok = o[0];
    }
    return toks;
}

/// One case: the same streams in slots 0..k (eager) and k..2k (graphed); returns the steps whose logits differ.
fn graphCase(e: *ink.engine.Engine, a: std.mem.Allocator, texts: []const Text, k: usize, steps: usize, at: usize) !usize {
    var pend: [8]u32 = undefined;
    var src: [8][]const u32 = undefined;
    var long: ?[]const u32 = null;
    for (texts) |t| if (t.ids.len >= 16390) {
        long = t.ids;
    };
    for (0..k) |i| {
        src[i] = if (at > 0) long.?[0..at] else texts[i % texts.len].ids;
        for ([_]usize{ i, k + i }) |slot| {
            try e.bind(@intCast(slot));
            e.begin(null);
            var p: usize = 0;
            pend[i] = while (true) : (p += ink.engine.prompt_rows) {
                if (try e.prefillChunk(src[i], p)) |t| break t;
            };
        }
    }
    const nv = e.w.share.vocab.len();
    var diff: usize = 0;
    for (0..steps) |_| {
        var outs: [2][8][1]u32 = undefined;
        var logits: [2][]u8 = undefined;
        for (0..2) |arm| {
            var parts: [8]ink.engine.Part = undefined;
            for (0..k) |i| parts[i] = .{ .slot = @intCast(arm * k + i), .ids = pend[i..][0..1], .out = &outs[arm][i] };
            e.graphs.settings.on = arm == 1;
            try e.verifyShared(parts[0..k]);
            try e.stream.synchronize();
            logits[arm] = try a.alloc(u8, k * nv * 4);
            try e.bd.logits.download(0, logits[arm]);
        }
        if (!std.mem.eql(u8, logits[0], logits[1])) diff += 1;
        for (0..k) |i| pend[i] = outs[0][i][0];
    }
    return diff;
}
