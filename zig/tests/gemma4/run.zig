//! Gemma 4 dense greedy decode on the native engine: tokens and top-two margins for gemma4_gate.py check.
const std = @import("std");
const tf = @import("tensorfold");

const g4 = tf.gemma4;

const Top2 = struct { ids: [2]u32, logits: [2]f32, position: usize };

fn top2(row: []const u16, position: usize) Top2 {
    var best: [2]u32 = .{ 0, 0 };
    var val: [2]f32 = .{ -std.math.inf(f32), -std.math.inf(f32) };
    for (row, 0..) |bits, i| {
        const v: f32 = @bitCast(@as(u32, bits) << 16);
        if (v > val[0]) {
            best[1] = best[0];
            val[1] = val[0];
            best[0] = @intCast(i);
            val[0] = v;
        } else if (v > val[1]) {
            best[1] = @intCast(i);
            val[1] = v;
        }
    }
    return .{ .ids = best, .logits = val, .position = position };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const model_dir = args.next() orelse return error.Usage;
    const oracle = args.next() orelse return error.Usage;
    const out = args.next() orelse return error.Usage;
    const count = if (args.next()) |a| try std.fmt.parseInt(usize, a, 10) else 128;
    const width = if (args.next()) |a| try std.fmt.parseInt(usize, a, 10) else 1; // prompt rows a window
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, out);
    const t0 = std.c.mach_absolute_time();
    const m = try g4.Model.load(gpa, io, model_dir);
    defer m.deinit();
    std.debug.print("loaded in {d:.1} s\n", .{@as(f64, @floatFromInt(std.c.mach_absolute_time() - t0)) / 24e6});
    var dir = try cwd.openDir(io, oracle, .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".ids.json")) continue;
        const name = entry.name[0 .. entry.name.len - ".ids.json".len];
        const bytes = try dir.readFileAlloc(io, entry.name, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        const ids = try std.json.parseFromSlice([]u32, gpa, bytes, .{});
        defer ids.deinit();
        const capacity = (ids.value.len + count + 64 + 63) / 64 * 64;
        var cache = try g4.state.Cache.init(m.device, capacity);
        defer cache.deinit();
        var s = try g4.state.Scratch.init(m.device, capacity);
        defer s.deinit();
        const t1 = std.c.mach_absolute_time();
        var ps: ?g4.prompt.Scratch = if (width == g4.state.chunk_rows) try g4.prompt.Scratch.init(m.device, capacity) else null;
        defer if (ps) |*p| p.deinit();
        var at: usize = 0;
        while (at < ids.value.len) : (at += width) {
            const end = @min(at + width, ids.value.len);
            if (ps) |*p| try g4.prompt.run(m, p, &cache, ids.value[at..end]) else try g4.forward.window(m, &s, &cache, ids.value[at..end], at);
        }
        if (ps) |*p| @memcpy(s.logits.slice(u16, g4.config.vocab), p.logits.slice(u16, g4.config.vocab));
        const last = if (ps != null) 0 else (ids.value.len - 1 - (ids.value.len - 1) / width * width) * g4.config.vocab;
        const prefill_s = @as(f64, @floatFromInt(std.c.mach_absolute_time() - t1)) / 24e6;
        var lp: [256]u8 = undefined;
        const row = s.logits.slice(u16, g4.state.prompt_rows * g4.config.vocab)[last..][0..g4.config.vocab];
        const f32s = try gpa.alloc(f32, g4.config.vocab);
        defer gpa.free(f32s);
        for (f32s, row) |*f, b| f.* = @bitCast(@as(u32, b) << 16);
        try cwd.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&lp, "{s}/{s}.last.f32", .{ out, name }), .data = std.mem.sliceAsBytes(f32s) });
        @memcpy(s.logits.slice(u16, g4.config.vocab), row);
        var tokens: std.ArrayList(u32) = .empty;
        defer tokens.deinit(gpa);
        var tops: std.ArrayList(Top2) = .empty;
        defer tops.deinit(gpa);
        var pos = ids.value.len;
        while (tokens.items.len < count) : (pos += 1) {
            const t = top2(s.logits.slice(u16, g4.config.vocab), pos);
            try tops.append(gpa, t);
            try tokens.append(gpa, t.ids[0]);
            if (tokens.items.len == count) break;
            try g4.forward.window(m, &s, &cache, &.{t.ids[0]}, pos);
        }
        const secs = @as(f64, @floatFromInt(std.c.mach_absolute_time() - t1)) / 24e6;
        var path: [256]u8 = undefined;
        var w: std.Io.Writer.Allocating = .init(gpa);
        defer w.deinit();
        try std.json.Stringify.value(tokens.items, .{}, &w.writer);
        try cwd.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&path, "{s}/{s}.tokens.json", .{ out, name }), .data = w.written() });
        w.clearRetainingCapacity();
        try w.writer.writeByte('[');
        for (tops.items, 0..) |t, i| {
            if (i > 0) try w.writer.writeAll(", ");
            try w.writer.print("{{\"ids\": [{d}, {d}], \"logits\": [{d}, {d}], \"position\": {d}, \"dtype\": \"mlx.core.bfloat16\"}}", .{ t.ids[0], t.ids[1], @as(f64, t.logits[0]), @as(f64, t.logits[1]), t.position });
        }
        try w.writer.writeByte(']');
        try cwd.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&path, "{s}/{s}.top2.json", .{ out, name }), .data = w.written() });
        std.debug.print("{s}: {d} prompt tokens in {d:.1} s ({d:.0} tok/s, windows of {d}), {d} decoded, {d:.1} s in all\n", .{ name, ids.value.len, prefill_s, @as(f64, @floatFromInt(ids.value.len)) / prefill_s, width, count, secs });
    }
}
