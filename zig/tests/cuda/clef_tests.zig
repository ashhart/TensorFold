//! `tf-cuda-test clef tiny DIR | head MODEL CASES OUT [--piece N]`: the CUDA joint schema head, as tf-clef-test on HIP.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const qwen35 = @import("qwen35");
const check = @import("check.zig");

const js = core.joint_schema;
const dev = qwen35.joint_schema;
const tiny = js.tiny;
const bf16 = core.bf16_round;
const npy = core.npy;

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len >= 2 and std.mem.eql(u8, rest[0], "tiny")) return runTiny(gpu, rest[1..]);
    if (rest.len >= 4 and std.mem.eql(u8, rest[0], "head")) return runHead(gpu, rest[1..]);
    std.debug.print("usage: tf-cuda-test clef tiny <scratch dir> | head <model dir> <cases dir> <out dir> [--piece N]\n", .{});
    return error.Usage;
}

/// The piece sizes it consumes the 48 rows in: whole, a third, an odd 7, one row at a time.
const pieces = [_]usize{ 48, 16, 7, 1 };

fn num(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

/// The tiny fixture's head on the device over its packed lm_head, against its FP64 and torch BF16 logits.
fn runTiny(gpu: check.Gpu, args: []const [:0]const u8) !void {
    if (args.len != 1) return error.Usage;
    const gpa = gpu.gpa;
    const io = gpu.io;
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
    const H: usize = tiny.cfg.hidden_size;
    // the lm_head as the engine keeps it: qmmf's BF16 lane layout, rows padded to 128
    const packed_head = try a.alloc(u16, (tiny.vocab + 127) / 128 * 128 * H);
    cuda.qmmf.packBf16(packed_head, @alignCast(std.mem.bytesAsSlice(u16, table)), tiny.vocab, H);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var d_rows = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(rows));
    defer d_rows.free();
    var d_table = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(packed_head));
    defer d_table.free();
    var w = try dev.weights.Weights.upload(gpa, gpu.d, &host.file, &host.weights);
    var owned = true; // the head owns the weights once it starts
    errdefer if (owned) w.deinit();
    const bounds: dev.plan.Bounds = .{ .rows = tiny.rows, .questions = 4, .options = 16, .piece = tiny.rows };
    const head = try dev.Head.init(gpa, gpu.d, w, bounds, d_table.ptr, tiny.vocab);
    owned = false;
    defer head.deinit();
    const req: dev.Request = .{ .question_spans = qs.items, .question_types = types.items, .option_spans = os.items, .option_question = owner.items, .prompt = ids };
    var first: ?[]u16 = null;
    for (pieces) |piece| {
        const got = try a.alloc(u16, os.items.len);
        const st = try dev.state.open(head, req, stream);
        defer dev.state.close(head, st);
        var at: usize = 0;
        while (at < tiny.rows) : (at += piece) {
            const n = @min(piece, tiny.rows - at);
            try dev.consume(head, st, d_rows.ptr + at * H * 2, at, n, stream);
        }
        try dev.finish(head, st, got, stream);
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

/// One case as tools/clef_head_gate.py writes it: the record's ids and BF16 rows (.npy) and its spans.
const Case = struct {
    name: []const u8,
    ids: []const u8,
    rows: []const u8,
    question_spans: []const [2]u32,
    question_types: []const u8,
    option_spans: []const [2]u32,
    option_question: []const u16,
};

fn read(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 31));
}

/// The backbone's lm_head.weight from the checkpoint's shards, packed on the device as the engine packs it.
fn lmHead(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, dir: []const u8) !struct { buf: cuda.DeviceBuffer, rows: usize } {
    const index = try std.json.parseFromSliceLeaky(std.json.Value, a, try read(a, io, try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" })), .{});
    const shard = index.object.get("weight_map").?.object.get("lm_head.weight").?.string;
    var file = try core.safetensors.File.open(a, io, try std.fs.path.join(a, &.{ dir, shard }));
    defer file.close(io);
    const t = file.get("lm_head.weight") orelse return error.MissingLmHead;
    if (t.dtype != .bf16 or t.rank != 2) return error.BadLmHead;
    const n = t.dim(0);
    const k = t.dim(1);
    var raw = try cuda.DeviceBuffer.fromHost(d, t.bytes);
    defer raw.free();
    var out = try cuda.DeviceBuffer.alloc(d, cuda.modelopt.padded(n) * k * 2);
    errdefer out.free();
    var packer = try cuda.modelopt.Packer.load(d);
    defer packer.unload();
    try packer.bf16Lane(s, raw.ptr, out.ptr, n, k);
    try s.synchronize();
    return .{ .buf = out, .rows = n };
}

/// Gate 3's native head on the reference's BF16 rows, each case's logits written to OUT/<case>.json.
fn runHead(gpu: check.Gpu, args: []const [:0]const u8) !void {
    if (args.len != 3 and args.len != 5) return error.Usage;
    const gpa = gpu.gpa;
    const io = gpu.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const model = args[0];
    const cases_dir = args[1];
    const out_dir = args[2];
    var piece_arg: ?usize = null;
    if (args.len == 5) {
        if (!std.mem.eql(u8, args[3], "--piece")) return error.Usage;
        if (!std.mem.eql(u8, args[4], "whole")) piece_arg = try std.fmt.parseInt(usize, args[4], 10);
    }
    const index = try std.json.parseFromSliceLeaky(std.json.Value, a, try read(a, io, try std.fmt.allocPrint(a, "{s}/index.json", .{cases_dir})), .{});
    const names = index.object.get("cases").?.array.items;
    const cases = try a.alloc(Case, names.len);
    var bounds: dev.plan.Bounds = .{ .rows = 1, .questions = 1, .options = 1, .piece = 1 };
    for (names, cases) |n, *c| {
        const text = try read(a, io, try std.fmt.allocPrint(a, "{s}/{s}.json", .{ cases_dir, n.string }));
        c.* = try std.json.parseFromSliceLeaky(Case, a, text, .{ .ignore_unknown_fields = true });
        const ids = try npy.parseHeader(try read(a, io, c.ids));
        bounds.rows = @max(bounds.rows, ids.shape[0]);
        bounds.questions = @max(bounds.questions, c.question_spans.len);
        bounds.options = @max(bounds.options, c.option_spans.len);
    }
    bounds.piece = piece_arg orelse bounds.rows;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var p: js.Problem = .{};
    var host = js.load.open(gpa, io, model, &p) catch |e| {
        std.debug.print("joint head: {s}\n", .{p.text});
        return e;
    };
    defer host.close(io);
    var table = try lmHead(a, io, gpu.d, stream, model);
    defer table.buf.free();
    var w = try dev.weights.Weights.upload(gpa, gpu.d, &host.file, &host.weights);
    var owned = true;
    errdefer if (owned) w.deinit();
    const head = try dev.Head.init(gpa, gpu.d, w, bounds, table.buf.ptr, table.rows);
    owned = false;
    defer head.deinit();
    const H: usize = host.weights.cfg.hidden_size;
    try std.Io.Dir.cwd().createDirPath(io, out_dir);
    for (cases) |c| {
        const ids_arr = try npy.parse(try read(a, io, c.ids));
        const rows_arr = try npy.parse(try read(a, io, c.rows));
        const L = ids_arr.shape[0];
        if (!std.mem.eql(u8, rows_arr.descr, "<u2") or rows_arr.shape[0] != L or rows_arr.shape[1] != H) return error.BadRows;
        const prompt = try a.alloc(u32, L);
        for (prompt, std.mem.bytesAsSlice(i32, ids_arr.data)) |*x, id| x.* = @intCast(id);
        var d_rows = try cuda.DeviceBuffer.fromHost(gpu.d, rows_arr.data);
        defer d_rows.free();
        const st = try dev.state.open(head, .{ .question_spans = c.question_spans, .question_types = c.question_types, .option_spans = c.option_spans, .option_question = c.option_question, .prompt = prompt }, stream);
        defer dev.state.close(head, st);
        var out: std.Io.Writer.Allocating = .init(a);
        try out.writer.writeAll("{\"pieces\":[");
        var at: usize = 0;
        while (at < L) : (at += bounds.piece) {
            const n = @min(bounds.piece, L - at);
            try dev.consume(head, st, d_rows.ptr + at * H * 2, at, n, stream);
            try out.writer.print("{s}[{d},{d}]", .{ if (at == 0) "" else ",", at, n });
        }
        const logits = try a.alloc(u16, c.option_spans.len);
        try dev.finish(head, st, logits, stream);
        try out.writer.writeAll("],\"logits\":[");
        for (logits, 0..) |b, i| try out.writer.print("{s}{d}", .{ if (i == 0) "" else ",", bf16.value(b) });
        try out.writer.writeAll("]}\n");
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/{s}.json", .{ out_dir, c.name }), .data = out.written() });
        std.debug.print("{s}: {d} rows in {d}-row pieces, {d} logits\n", .{ c.name, L, bounds.piece, logits.len });
    }
}
