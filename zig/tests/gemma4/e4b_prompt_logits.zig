//! Gemma 4 E4B prompt logits as the server's prefill makes them: the last row of each prompt and of prefixes, as .npy.
const std = @import("std");
const tf = @import("tensorfold");

const g4 = tf.gemma4;
const lanes = tf.lanes;
const M = g4.backend.Metal;
const vocab = g4.config.vocab;

const Kept = struct {
    metal: *M,
    at: [16]u32 = undefined,
    snap: [16]*anyopaque = undefined,
    n: usize = 0,

    fn hook(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
        const k: *Kept = @ptrCast(@alignCast(ptr));
        const saved = M.snapSave(k.metal, s, at) catch return;
        k.at[k.n] = at;
        k.snap[k.n] = saved;
        k.n += 1;
    }

    fn find(k: *const Kept, at: u32) ?*anyopaque {
        for (k.at[0..k.n], k.snap[0..k.n]) |a, s| if (a == at) return s;
        return null;
    }
};

const Result = struct {
    logits: []u16,
    tokens: [8]u32,
    final: []u16,
};

fn argmax(row: []const u16) u32 {
    var best: u32 = 0;
    var bv: f32 = -std.math.inf(f32);
    for (row, 0..) |b, i| {
        const v: f32 = @bitCast(@as(u32, b) << 16);
        if (v > bv) {
            bv = v;
            best = @intCast(i);
        }
    }
    return best;
}

/// One pass: prompt `ids`, chunk starts `chunks`, marks `marks` (kept into `kept`), resumed from `from` (null: fresh).
fn pass(gpa: std.mem.Allocator, metal: *M, ids: []const u32, chunks: []const u32, marks: []const u32, kept: *Kept, from: ?u32, source: *const Kept) !Result {
    var reuse: lanes.stream.Reuse = .{ .marks = marks, .hook = .{ .ptr = kept, .at = Kept.hook } };
    if (from) |m| {
        reuse.saved = source.find(m) orelse return error.NoState;
        reuse.at = m;
    }
    var s = try lanes.Stream.init(gpa, .{ .id = "seam", .prompt = ids, .max_new = 16, .chunks = chunks, .reuse = reuse });
    defer s.deinit(gpa);
    const b = metal.backend();
    try b.vtable.prefill(b.ptr, &s);
    if (from) |m| if (s.cached != m) return error.NotResumed;
    const cache = metal.caches.get(&s).?;
    const logits = try gpa.dupe(u16, cache.logits.slice(u16, vocab));
    var tokens: [8]u32 = undefined;
    var row = logits;
    var final: []u16 = logits;
    for (&tokens) |*t| {
        t.* = argmax(row);
        try g4.forward.window(metal.model, &metal.scratch, cache, &.{t.*}, cache.len);
        row = metal.scratch.logits.slice(u16, g4.state.prompt_rows * vocab)[0..vocab];
        final = row;
    }
    const kept_final = try gpa.dupe(u16, final);
    b.vtable.release(b.ptr, &s);
    return .{ .logits = logits, .tokens = tokens, .final = kept_final };
}

fn npyHeader(buf: []u8, rows: usize, cols: usize) ![]const u8 {
    var dict: [128]u8 = undefined;
    const d = try std.fmt.bufPrint(&dict, "{{'descr': '<f4', 'fortran_order': False, 'shape': ({d}, {d}), }}", .{ rows, cols });
    const len = (10 + d.len + 1 + 63) / 64 * 64;
    @memcpy(buf[0..8], "\x93NUMPY\x01\x00");
    std.mem.writeInt(u16, buf[8..10], @intCast(len - 10), .little);
    @memcpy(buf[10..][0..d.len], d);
    @memset(buf[10 + d.len .. len - 1], ' ');
    buf[len - 1] = '\n';
    return buf[0..len];
}

fn write(io: std.Io, gpa: std.mem.Allocator, path: []const u8, rows: []const []const u16) !void {
    const flat = try gpa.alloc(f32, rows.len * vocab);
    defer gpa.free(flat);
    for (rows, 0..) |r, i| for (flat[i * vocab ..][0..vocab], r) |*f, b| {
        f.* = @bitCast(@as(u32, b) << 16);
    };
    var head: [128]u8 = undefined;
    const h = try npyHeader(&head, rows.len, vocab);
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var wb: [1 << 16]u8 = undefined;
    var w = file.writer(io, &wb);
    try w.interface.writeAll(h);
    try w.interface.writeAll(std.mem.sliceAsBytes(flat));
    try w.interface.flush();
}

const Ref = struct { name: []const u8, prompt_ids: []u32 };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const model_dir = args.next() orelse return error.Usage;
    const ref_dir = args.next() orelse return error.Usage;
    const out = args.next() orelse return error.Usage;
    const model = try g4.Model.load(gpa, io, model_dir);
    defer model.deinit();
    const metal = try M.init(gpa, io, model, .{ .capacity = 8192, .streams = 2 });
    defer metal.deinit();
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, out);
    var dir = try cwd.openDir(io, ref_dir, .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    const prefixes = [_]u32{ 130, 300, 700, 1500, 3000, 6000 };
    while (try it.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".json")) continue;
        const bytes = try dir.readFileAlloc(io, entry.name, gpa, .limited(1 << 24));
        defer gpa.free(bytes);
        const parsed = std.json.parseFromSlice(Ref, gpa, bytes, .{ .ignore_unknown_fields = true }) catch continue;
        defer parsed.deinit();
        const r = parsed.value;
        var lens: std.ArrayList(u32) = .empty;
        defer lens.deinit(gpa);
        if (std.mem.eql(u8, r.name, "doc4096")) for (prefixes) |p| try lens.append(gpa, p);
        try lens.append(gpa, @intCast(r.prompt_ids.len));
        var rows: std.ArrayList([]u16) = .empty;
        defer {
            for (rows.items) |x| gpa.free(x);
            rows.deinit(gpa);
        }
        for (lens.items) |len| {
            var planned: [16]u32 = undefined;
            var np: usize = 0;
            var p: u32 = 512;
            while (p < len) : (p += 512) {
                planned[np] = p;
                np += 1;
            }
            var none = Kept{ .metal = metal };
            const res = try pass(gpa, metal, r.prompt_ids[0..len], planned[0..np], &.{}, &none, null, &none);
            gpa.free(res.final);
            try rows.append(gpa, res.logits);
        }
        var path: [512]u8 = undefined;
        try write(io, gpa, try std.fmt.bufPrint(&path, "{s}/{s}.prompt.npy", .{ out, r.name }), rows.items);
        std.debug.print("{s}: prompt rows at {any}\n", .{ r.name, lens.items });
    }
}
