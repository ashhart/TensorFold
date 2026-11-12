//! `tf-clef-test head MODEL CASES OUT [--piece N|whole]`: gate 3's native head on the reference's BF16 rows.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const npy = @import("npy");

const js = core.joint_schema;

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

fn caseOf(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !Case {
    const text = try read(a, io, try std.fmt.allocPrint(a, "{s}/{s}.json", .{ dir, name }));
    return std.json.parseFromSliceLeaky(Case, a, text, .{ .ignore_unknown_fields = true });
}

/// The backbone's lm_head.weight from the checkpoint's shards, mapped while it goes up.
fn lmHead(a: std.mem.Allocator, io: std.Io, r: *const hip.Runtime, dir: []const u8) !struct { buf: hip.DeviceBuffer, rows: usize } {
    const index = try std.json.parseFromSliceLeaky(std.json.Value, a, try read(a, io, try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" })), .{});
    const shard = index.object.get("weight_map").?.object.get("lm_head.weight").?.string;
    var file = try core.safetensors.File.open(a, io, try std.fs.path.join(a, &.{ dir, shard }));
    defer file.close(io);
    const t = file.get("lm_head.weight") orelse return error.MissingLmHead;
    if (t.dtype != .bf16 or t.rank != 2) return error.BadLmHead;
    return .{ .buf = try hip.DeviceBuffer.fromHost(r, t.bytes), .rows = t.dim(0) };
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len != 3 and args.len != 5) return error.Usage;
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
    var bounds: hip.joint_schema.plan.Bounds = .{ .rows = 1, .questions = 1, .options = 1, .piece = 1 };
    for (names, cases) |n, *c| {
        c.* = try caseOf(a, io, cases_dir, n.string);
        const ids = try npy.parseHeader(try read(a, io, c.ids));
        bounds.rows = @max(bounds.rows, ids.shape[0]);
        bounds.questions = @max(bounds.questions, c.question_spans.len);
        bounds.options = @max(bounds.options, c.option_spans.len);
    }
    bounds.piece = piece_arg orelse bounds.rows;
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
    var p: js.Problem = .{};
    var host = js.load.open(gpa, io, model, &p) catch |e| {
        std.debug.print("joint head: {s}\n", .{p.text});
        return e;
    };
    defer host.close(io);
    var table = try lmHead(a, io, &r, model);
    defer table.buf.free();
    const w = try hip.joint_schema.weights.Weights.upload(gpa, &r, &host.file, &host.weights);
    const head = try hip.joint_schema.Head.init(gpa, &r, &l, &blas, w, bounds, table.buf.base(), table.rows);
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
        var d_rows = try hip.DeviceBuffer.fromHost(&r, rows_arr.data);
        defer d_rows.free();
        const st = try hip.joint_schema.state.open(head, .{ .question_spans = c.question_spans, .question_types = c.question_types, .option_spans = c.option_spans, .option_question = c.option_question, .prompt = prompt });
        defer hip.joint_schema.state.close(head, st);
        var at: usize = 0;
        while (at < L) : (at += bounds.piece) try hip.joint_schema.consume(head, st, d_rows.base() + at * H * 2, at, @min(bounds.piece, L - at), stream);
        const logits = try a.alloc(u16, c.option_spans.len);
        try hip.joint_schema.finish(head, st, logits, stream);
        var out: std.Io.Writer.Allocating = .init(a);
        try out.writer.writeAll("{\"logits\":[");
        for (logits, 0..) |b, i| try out.writer.print("{s}{d}", .{ if (i == 0) "" else ",", core.bf16_round.value(b) });
        try out.writer.writeAll("]}\n");
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/{s}.json", .{ out_dir, c.name }), .data = out.written() });
        std.debug.print("{s}: {d} rows in {d}-row pieces, {d} logits\n", .{ c.name, L, bounds.piece, logits.len });
    }
}
