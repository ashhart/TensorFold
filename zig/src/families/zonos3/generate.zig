//! zonos3-generate: ZONOS3 Mini's backbone on one CUDA GPU against the PyTorch reference's dumps.
const std = @import("std");
const z3 = @import("zonos3");

const cfg = z3.config;
const W = cfg.frame_width;
const D = cfg.dim;

const usage =
    \\usage: zonos3-generate --model LM_DIR --check DUMP_DIR [--max-rows 64] [--max-ctx 4096]
    \\
    \\--check runs DUMP_DIR/input_ids.npy (prompt rows [n, 11]) with DUMP_DIR/input_ids.cond.json through the backbone
    \\and compares every sublayer with the reference's .npy captures (tools/dump_golden.py lm of Zyphra/ZONOS3-Mini).
    \\The last stderr line is "RESULT zonos3 check PASS|FAIL ...".
    \\
;

const Check = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    e: *z3.Engine,
    dir: []const u8,
    n: usize = 0,
    fails: usize = 0,
    missing: usize = 0,
    worst: f64 = 1.0,
    worst_rel: f64 = 0,

    fn probe(ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize) anyerror!void {
        const c: *Check = @ptrCast(@alignCast(ctx));
        var nb: [96]u8 = undefined;
        const path = try std.fs.path.join(c.gpa, &.{ c.dir, try std.fmt.bufPrint(&nb, "{s}.npy", .{name}) });
        defer c.gpa.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(c.io, path, c.gpa, .limited(1 << 30)) catch {
            c.missing += 1;
            std.debug.print("  {s:<20} (no reference tensor)\n", .{name});
            return;
        };
        defer c.gpa.free(bytes);
        const arr = try z3.npy.parse(bytes);
        if (arr.numel() != rows * width) return error.ReferenceShape;
        const ref = try arr.floats(c.gpa);
        defer c.gpa.free(ref);
        const mine = try c.gpa.alloc(f32, rows * width);
        defer c.gpa.free(mine);
        try c.e.download(ptr, std.mem.sliceAsBytes(mine));
        var worst_row: f64 = 1;
        var at: usize = 0;
        var max_d: f64 = 0;
        var max_ref: f64 = 0;
        for (0..rows) |r| {
            var d: f64 = 0;
            var a2: f64 = 0;
            var b2: f64 = 0;
            for (mine[r * width ..][0..width], ref[r * width ..][0..width]) |x, y| {
                d += @as(f64, x) * y;
                a2 += @as(f64, x) * x;
                b2 += @as(f64, y) * y;
                max_d = @max(max_d, @abs(@as(f64, x) - y));
                max_ref = @max(max_ref, @abs(@as(f64, y)));
            }
            const cos: f64 = if (a2 > 0 and b2 > 0) d / @sqrt(a2 * b2) else if (a2 == b2) 1.0 else 0.0;
            if (cos < worst_row) {
                worst_row = cos;
                at = r;
            }
        }
        const rel = if (max_ref > 0) max_d / max_ref else max_d;
        const ok = worst_row >= 0.9999;
        c.n += 1;
        c.fails += @intFromBool(!ok);
        c.worst = @min(c.worst, worst_row);
        c.worst_rel = @max(c.worst_rel, rel);
        std.debug.print("  {s:<20} rows {d:>4} worst-row cos {d:.7} (row {d}) max|d| {e:.3} of {e:.3} {s}\n", .{ name, rows, worst_row, at, max_d, max_ref, if (ok) "ok" else "LOW" });
    }
};

/// The dump's prompt: rows and the rows a speaker or condition vector replaces.
fn loadPrompt(gpa: std.mem.Allocator, io: std.Io, e: *z3.Engine, dir: []const u8, rows: *[][W]i32, replaced: *std.ArrayList(z3.Replace)) !void {
    const ids_path = try std.fs.path.join(gpa, &.{ dir, "input_ids.npy" });
    defer gpa.free(ids_path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, ids_path, gpa, .limited(1 << 28));
    defer gpa.free(bytes);
    const arr = try z3.npy.parse(bytes);
    if (arr.rank != 2 or arr.shape[1] != W) return error.BadPromptRows;
    const flat = try arr.ints(gpa);
    defer gpa.free(flat);
    rows.* = try gpa.alloc([W]i32, arr.shape[0]);
    for (rows.*, 0..) |*r, i| r.* = flat[i * W ..][0..W].*;

    const cond_path = try std.fs.path.join(gpa, &.{ dir, "input_ids.cond.json" });
    defer gpa.free(cond_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, cond_path, gpa, .limited(1 << 22));
    defer gpa.free(text);
    var tree = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer tree.deinit();
    const o = tree.value.object;
    const speaker_row = o.get("speaker_row").?.integer;
    if (speaker_row >= 0) {
        const v = o.get("speaker_embedding").?.array.items;
        if (v.len != cfg.speaker_dim) return error.BadSpeakerEmbedding;
        var vec: [cfg.speaker_dim]f32 = undefined;
        for (&vec, v) |*x, j| x.* = switch (j) {
            .float => |f| @floatCast(f),
            .integer => |n| @floatFromInt(n),
            else => return error.BadSpeakerEmbedding,
        };
        var r: z3.Replace = .{ .row = @intCast(speaker_row), .vec = undefined };
        e.w.speakerRow(&vec, &r.vec);
        try replaced.append(gpa, r);
    }
    const first: usize = @intCast(o.get("condition_row").?.integer);
    for (o.get("condition_buckets").?.array.items, 0..) |b, i| {
        const bucket: usize = @intCast(b.integer);
        if (i >= cfg.conditions or bucket > e.c.buckets[i]) return error.BadConditionBucket;
        var r: z3.Replace = .{ .row = first + i, .vec = undefined };
        @memcpy(&r.vec, e.w.conditions[i][bucket * D ..][0..D]);
        try replaced.append(gpa, r);
    }
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const gpa = std.heap.c_allocator;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(arena);
    var model: []const u8 = "";
    var check: ?[]const u8 = null;
    var opts: z3.Options = .{};
    var i: usize = 1;
    while (i + 1 < argv.len) : (i += 2) {
        const k = argv[i];
        const v = argv[i + 1];
        if (std.mem.eql(u8, k, "--model")) model = v else if (std.mem.eql(u8, k, "--check")) check = v else if (std.mem.eql(u8, k, "--max-rows")) opts.max_rows = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--max-ctx")) opts.max_ctx = try std.fmt.parseInt(usize, v, 10) else {
            std.debug.print("unknown flag {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    const dir = check orelse {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    if (model.len == 0 or i != argv.len) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const e = try z3.Engine.create(gpa, io, model, opts);
    defer e.destroy();
    var rows: [][W]i32 = &.{};
    var replaced: std.ArrayList(z3.Replace) = .empty;
    try loadPrompt(gpa, io, e, dir, &rows, &replaced);
    var c: Check = .{ .gpa = gpa, .io = io, .e = e, .dir = dir };
    std.debug.print("zonos3 check: {d} prompt rows, {d} replaced, {d} MiB of weights\n", .{ rows.len, replaced.items.len, e.w.bytes >> 20 });
    try e.run(rows, replaced.items, .{ .ctx = &c, .f = Check.probe });
    const pass = c.fails == 0 and c.n > 0;
    std.debug.print("RESULT zonos3 check {s} probes={d} low={d} missing={d} worst_row_cos={d:.7} worst_rel_diff={e:.3}\n", .{ if (pass) "PASS" else "FAIL", c.n, c.fails, c.missing, c.worst, c.worst_rel });
    return if (pass) 0 else 1;
}
