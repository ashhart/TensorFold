//! Gemma 4 E4B teacher-forced logits for each reference's prompt, as <name>.logits.npy [n, vocab] float32.
const std = @import("std");
const tf = @import("tensorfold");

const g4 = tf.gemma4;

const Ref = struct { name: []const u8, prompt_ids: []u32, generated_ids: []u32 };

fn npyHeader(buf: []u8, rows: usize, cols: usize) ![]const u8 {
    var dict: [128]u8 = undefined;
    const d = try std.fmt.bufPrint(&dict, "{{'descr': '<f4', 'fortran_order': False, 'shape': ({d}, {d}), }}", .{ rows, cols });
    const len = (10 + d.len + 1 + 63) / 64 * 64; // magic, version, header length, dict, newline: a multiple of 64
    @memcpy(buf[0..8], "\x93NUMPY\x01\x00");
    std.mem.writeInt(u16, buf[8..10], @intCast(len - 10), .little);
    @memcpy(buf[10..][0..d.len], d);
    @memset(buf[10 + d.len .. len - 1], ' ');
    buf[len - 1] = '\n';
    return buf[0..len];
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const model_dir = args.next() orelse return error.Usage;
    const ref_dir = args.next() orelse return error.Usage;
    const out = args.next() orelse return error.Usage;
    const width = if (args.next()) |a| try std.fmt.parseInt(usize, a, 10) else 16; // rows a window
    const only = args.next();
    const tile = std.c.getenv("E4B_TILE") != null;
    if (width == 0 or width > g4.state.prompt_rows) return error.Usage;
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, out);
    const t0 = std.c.mach_absolute_time();
    const m = try g4.Model.load(gpa, io, model_dir);
    defer m.deinit();
    std.debug.print("loaded {s} in {d:.1} s\n", .{ m.config.geo.name, @as(f64, @floatFromInt(std.c.mach_absolute_time() - t0)) / 24e6 });
    const vocab = g4.config.vocab;
    var dir = try cwd.openDir(io, ref_dir, .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".json")) continue;
        const bytes = try dir.readFileAlloc(io, entry.name, gpa, .limited(1 << 24));
        defer gpa.free(bytes);
        const parsed = std.json.parseFromSlice(Ref, gpa, bytes, .{ .ignore_unknown_fields = true }) catch continue; // not a prompt
        defer parsed.deinit();
        const r = parsed.value;
        if (only) |o| if (!std.mem.eql(u8, o, r.name)) continue;
        const n = r.generated_ids.len;
        const ids = try std.mem.concat(gpa, u32, &.{ r.prompt_ids, r.generated_ids[0 .. n - 1] });
        defer gpa.free(ids);
        const capacity = (ids.len + 64 + 63) / 64 * 64;
        var cache = try g4.state.Cache.init(m.device, m.config.geo, capacity);
        defer cache.deinit();
        var s = try g4.state.Scratch.init(m.device, m.config.geo, capacity);
        defer s.deinit();
        const rows = try gpa.alloc(f32, n * vocab);
        defer gpa.free(rows);
        const first = r.prompt_ids.len - 1; // the position whose logits predict the first generated token
        const t1 = std.c.mach_absolute_time();
        var at: usize = 0;
        var ps: ?g4.prompt.Scratch = if (tile) try g4.prompt.Scratch.init(m.device, m.config.geo, capacity) else null;
        defer if (ps) |*p| p.deinit();
        if (ps) |*p| { // E4B_TILE=1: the prompt in chunks of 512 on affine_mm's tiles, then decode windows over the rest
            while (at < r.prompt_ids.len) {
                const end = @min(at + g4.state.chunk_rows, r.prompt_ids.len);
                try g4.prompt.run(m, p, &cache, ids[at..end], &.{});
                at = end;
            }
            const got = p.logits.slice(u16, vocab);
            for (rows[0..vocab], got) |*f, b| f.* = @bitCast(@as(u32, b) << 16);
        }
        while (at < ids.len) : (at += width) {
            const end = @min(at + width, ids.len);
            try g4.forward.window(m, &s, &cache, ids[at..end], at);
            const got = s.logits.slice(u16, g4.state.prompt_rows * vocab);
            for (at..end) |p| if (p >= first) {
                for (rows[(p - first) * vocab ..][0..vocab], got[(p - at) * vocab ..][0..vocab]) |*f, b| f.* = @bitCast(@as(u32, b) << 16);
            };
        }
        const secs = @as(f64, @floatFromInt(std.c.mach_absolute_time() - t1)) / 24e6;
        var path: [512]u8 = undefined;
        var head: [128]u8 = undefined;
        const h = try npyHeader(&head, n, vocab);
        const file = try cwd.createFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.logits.npy", .{ out, r.name }), .{});
        defer file.close(io);
        var wb: [1 << 16]u8 = undefined;
        var w = file.writer(io, &wb);
        try w.interface.writeAll(h);
        try w.interface.writeAll(std.mem.sliceAsBytes(rows));
        try w.interface.flush();
        std.debug.print("{s}: {d} tokens in windows of {d}, {d:.2} s ({d:.0} tok/s), {d} positions written\n", .{ r.name, ids.len, width, secs, @as(f64, @floatFromInt(ids.len)) / secs, n });
    }
}
