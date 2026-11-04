//! LFM3 prompt windows of 64, 128, 256 (the most) rows and a ragged cut: rows, logits and the next step, bit for bit.

const std = @import("std");
const rt = @import("rig.zig");
const lfm3 = @import("lfm3_xpu");
const stop = @import("xpu").stop;
const Model = lfm3.model.Model;
const H = lfm3.config.hidden;
const V = lfm3.config.vocab;

const Rows = struct {
    out: []u16,
    at: usize = 0,
    fn take(ctx: *anyopaque, layer: i32, rows: []const u16) void {
        if (layer != -1) return;
        const r: *Rows = @ptrCast(@alignCast(ctx));
        @memcpy(r.out[r.at..][0..rows.len], rows);
        r.at += rows.len;
    }
};

/// The prompt in windows of `cut` (cycled): each row's final norm, the last logits, then one decode step's logits.
fn run(m: *Model, ids: []const u32, cut: []const usize, prompt: bool, rows: []u16, last: []f32, step: []f32) !u32 {
    var cap: Rows = .{ .out = rows };
    try m.reset();
    var i: usize = 0;
    var c: usize = 0;
    while (i < ids.len) : (c += 1) {
        const n = @min(cut[c % cut.len], ids.len - i);
        const hk: lfm3.model.Hook = .{ .ctx = &cap, .f = Rows.take };
        if (prompt) try m.forwardPrompt(ids[i .. i + n], i + n == ids.len, hk) else try m.forward(ids[i .. i + n], i + n == ids.len, hk);
        i += n;
    }
    try m.fetchLogits(last);
    const tok = try m.argmax();
    try m.forward(&.{tok}, true, null);
    try m.fetchLogits(step);
    return tok;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    stop.install();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("skipped: usage tf-xpu-lfm3_chunks-test MODEL_DIR REF_DIR (its items prompt, repeated to 2,100 tokens)\n", .{});
        return;
    }
    var r = try rt.open();
    defer r.deinit();
    var c = try lfm3.config.Config.read(gpa, io, args[1]);
    defer c.deinit();
    var l = try @import("xpu").loader.Loader.init(gpa, &r, args[1]);
    var m = try Model.load(gpa, &r, &l, &c, 2304, 256);
    const path = try std.fmt.allocPrint(gpa, "{s}/items/meta.json", .{args[2]});
    const meta = try std.json.parseFromSlice(std.json.Value, gpa, try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited), .{});
    const vals = meta.value.object.get("ids").?.array.items;
    const ids = try gpa.alloc(u32, 2100);
    for (ids, 0..) |*o, i| o.* = @intCast(vals[if (i < vals.len) i else 1 + (i - 1) % (vals.len - 1)].integer);
    const cuts = [_]struct { name: []const u8, cut: []const usize, prompt: bool = true }{
        .{ .name = "windows of 128", .cut = &.{128} },
        .{ .name = "windows of 64", .cut = &.{64} },
        .{ .name = "windows of 256", .cut = &.{256} },
        .{ .name = "ragged (1, 7, 100, 3, 250, 64, 1)", .cut = &.{ 1, 7, 100, 3, 250, 64, 1 } },
        .{ .name = "planted control: windows of 128 in pdense's order (must differ)", .cut = &.{128}, .prompt = false },
    };
    var ref_rows: []u16 = &.{};
    var ref_last: []f32 = &.{};
    var ref_step: []f32 = &.{};
    var ref_tok: u32 = 0;
    var bad = false;
    for (cuts, 0..) |ct, k| {
        const rows = try gpa.alloc(u16, ids.len * H);
        const last = try gpa.alloc(f32, V);
        const step = try gpa.alloc(f32, V);
        const tok = try run(&m, ids, ct.cut, ct.prompt, rows, last, step);
        if (k == 0) {
            ref_rows = rows;
            ref_last = last;
            ref_step = step;
            ref_tok = tok;
            std.debug.print("  {s}: the reference ({d} rows, next token {d})\n", .{ ct.name, ids.len, tok });
            continue;
        }
        const same = std.mem.eql(u16, rows, ref_rows) and std.mem.eql(f32, last, ref_last) and std.mem.eql(f32, step, ref_step) and tok == ref_tok;
        if (same != ct.prompt) bad = true;
        std.debug.print("  {s}: every row, the logits and the decode step after {s}\n", .{ ct.name, if (same) "bit for bit equal" else "differ" });
    }
    std.debug.print("{s}\n", .{if (bad) "lfm3_chunks: FAILED" else "lfm3_chunks: ok"});
    if (bad) return error.TestFailed;
}
