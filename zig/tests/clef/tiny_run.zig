//! `tf-clef-test tiny DIR`: the device head on the tiny fixture's head, against its FP64 and torch BF16 logits.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");

const js = core.joint_schema;
const tiny = js.tiny;
const bf16 = core.bf16_round;

/// The piece sizes it consumes the 48 rows in: whole, a third, an odd 7, one row at a time.
const pieces = [_]usize{ 48, 16, 7, 1 };

fn num(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len != 1) return error.Usage;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, args[0], .{});
    defer dir.close(io);
    try tiny.write(a, io, dir, tiny.cfg, .none);
    var p: js.Problem = .{};
    var host = js.load.open(gpa, io, args[0], &p) catch |e| {
        std.debug.print("tiny head: {s}\n", .{p.text});
        return e;
    };
    defer host.close(io);
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, tiny.fixture, .{});
    // the fixture's record: question spans and types, options grouped by question
    var qs: std.ArrayList([2]u32) = .empty;
    var types: std.ArrayList(u8) = .empty;
    var os: std.ArrayList([2]u32) = .empty;
    var owner: std.ArrayList(u16) = .empty;
    var owner32: std.ArrayList(u32) = .empty;
    for (doc.object.get("questions").?.array.items, 0..) |q, k| {
        const sp = q.object.get("span").?.array.items;
        try qs.append(a, .{ @intCast(sp[0].integer), @intCast(sp[1].integer) });
        try types.append(a, @intCast(q.object.get("type").?.integer));
        for (q.object.get("options").?.array.items) |o| {
            try os.append(a, .{ @intCast(o.array.items[0].integer), @intCast(o.array.items[1].integer) });
            try owner.append(a, @intCast(k));
            try owner32.append(a, @intCast(k));
        }
    }
    const ids = try tiny.ids(a);
    const hidden = try tiny.hidden(a);
    const table = try tiny.table(a);
    const rows = try a.alloc(u16, hidden.len);
    for (rows, hidden) |*b, v| b.* = tiny.bits(v);
    // the device
    var r = try hip.Runtime.open();
    defer r.close();
    if (try r.deviceCount() != 1) return error.SelectOneHipDevice;
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var l = try hip.Launcher.load(&r, .{}, hip.kernels.images);
    defer l.unload();
    var blas = try hip.blas.Blas.open(stream);
    defer blas.close();
    var d_rows = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(rows));
    defer d_rows.free();
    var d_table = try hip.DeviceBuffer.fromHost(&r, table);
    defer d_table.free();
    var w = try hip.joint_schema.weights.Weights.upload(gpa, &r, &host.file, &host.weights);
    var owned = true; // the head owns the weights once it starts
    errdefer if (owned) w.deinit();
    const bounds: hip.joint_schema.plan.Bounds = .{ .rows = tiny.rows, .questions = 4, .options = 16, .piece = tiny.rows };
    const head = try hip.joint_schema.Head.init(gpa, &r, &l, &blas, w, bounds, d_table.base(), tiny.vocab);
    owned = false;
    defer head.deinit();
    const H: usize = tiny.cfg.hidden_size;
    const req: hip.joint_schema.Request = .{ .question_spans = qs.items, .question_types = types.items, .option_spans = os.items, .option_question = owner.items, .prompt = ids };
    var first: ?[]u16 = null;
    for (pieces) |piece| {
        const got = try a.alloc(u16, os.items.len);
        const st = try hip.joint_schema.state.open(head, req);
        defer hip.joint_schema.state.close(head, st);
        var at: usize = 0;
        while (at < tiny.rows) : (at += piece) {
            const n = @min(piece, tiny.rows - at);
            try hip.joint_schema.consume(head, st, d_rows.base() + at * H * 2, at, n, stream);
        }
        try hip.joint_schema.finish(head, st, got, stream);
        if (first) |f| {
            if (!std.mem.eql(u16, f, got)) {
                std.debug.print("FAIL tiny head: {d}-row pieces give other logits than whole\n", .{piece});
                return error.CutsDiffer;
            }
        } else first = got;
    }
    // against the fixture's FP64 and torch BF16 logits and the host's BF16 emulation
    const want64 = doc.object.get("fp64").?.array.items;
    const want16 = doc.object.get("bf16").?.array.items;
    const ref_rec: js.reference.Record = .{ .ids = ids, .question_spans = qs.items, .question_types = types.items, .option_spans = os.items, .option_question = owner32.items };
    const emul = try js.reference.logits(.{ .a = a, .mode = .bf16 }, &host.weights, hidden, ref_rec, .{ .bytes = table, .rows = tiny.vocab });
    var native: [2]f64 = .{ 0, 0 };
    var torch: [2]f64 = .{ 0, 0 };
    var to_torch: f64 = 0;
    var to_emul: f64 = 0;
    var top: f64 = 0;
    const n: f64 = @floatFromInt(want64.len);
    // a non-finite logit fails here: @max below would keep the finite operand of a NaN
    for (first.?, 0..) |g, i| if (!std.math.isFinite(bf16.value(g))) {
        std.debug.print("FAIL tiny head: logit {d} is not finite\n", .{i});
        return error.NonFiniteLogit;
    };
    for (first.?, want64, want16, emul) |g, w64, w16, e| {
        const got = bf16.value(g);
        const torch16 = bf16.value(@intCast(w16.integer));
        const exact = num(w64);
        native = .{ @max(native[0], @abs(got - exact)), native[1] + @abs(got - exact) / n };
        torch = .{ @max(torch[0], @abs(torch16 - exact)), torch[1] + @abs(torch16 - exact) / n };
        to_torch = @max(to_torch, @abs(got - torch16));
        to_emul = @max(to_emul, @abs(got - e));
        top = @max(top, @abs(exact));
    }
    const step = std.math.ldexp(@as(f64, 1), std.math.frexp(top).exponent - 8);
    std.debug.print("tiny head: {d} logits, pieces {any} bit-equal; against FP64 (worst, mean) native {d:.4} {d:.4}, torch BF16 {d:.4} {d:.4}; " ++
        "native vs torch BF16 {d:.4}, vs host emulation {d:.4} ({d:.2} and {d:.2} steps at the largest logit)\n", .{ want64.len, pieces, native[0], native[1], torch[0], torch[1], to_torch, to_emul, to_torch / step, to_emul / step });
    if (to_torch > 4 * step or to_emul > 4 * step) return error.HeadDiffers;
    std.debug.print("PASS tiny head\n", .{});
}
