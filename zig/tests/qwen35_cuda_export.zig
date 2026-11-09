//! `tf-qwen35-cuda <model> export <ids.json> <at> <out>`: #1594 step 2b's GB10 half, a kept state in hybrid order.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const hs = @import("hybrid_state");
const E = q.engine;

/// `<model> tail <seq.json> <k> <out>`: the bf16 logits after each of seq's last k tokens, fed one at a time.
fn tail(init: std.process.Init, a: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], a, .limited(16 << 20));
    const seq = (try std.json.parseFromSlice([]u32, a, text, .{})).value;
    const k = try std.fmt.parseInt(usize, args[4], 10);
    if (k == 0 or k >= seq.len) return error.MarkOutOfRange;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const e = try E.Engine.init(init.gpa, init.io, &ctx, args[1], false, null);
    defer e.deinit();
    var cache = try E.Cache.init(init.gpa, e, seq.len + 64);
    defer cache.deinit(init.gpa);
    const head = seq.len - k;
    var buf: [E.max_rows]u32 = undefined;
    var done: usize = 0;
    while (done < head) {
        const n = @min(E.max_rows, head - done);
        @memcpy(buf[0..n], seq[done..][0..n]);
        try e.forward(&.{.{ .cache = &cache, .rows = n }}, buf[0..n], .last);
        done += n;
    }
    var out: std.ArrayList(u8) = .empty;
    for (seq[head..]) |t| {
        try e.forward(&.{.{ .cache = &cache, .rows = 1 }}, &.{t}, .logits);
        try out.appendSlice(a, std.mem.sliceAsBytes(e.rowLogits(0)));
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[5], .data = out.items });
    std.debug.print("RESULT tail: {d} rows of bf16 logits after {d} tokens, to {s}\n", .{ k, head, args[5] });
    return 0;
}

/// `<model> export-many <prompts.json> <tail> <dir>`: each prompt to its length less `tail`, in turn, to dir/s<k>.tfhs.
fn many(init: std.process.Init, a: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], a, .limited(64 << 20));
    const prompts = (try std.json.parseFromSlice([][]u32, a, text, .{})).value;
    const tail_n = try std.fmt.parseInt(u32, args[4], 10);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const e = try E.Engine.init(init.gpa, init.io, &ctx, args[1], false, null);
    defer e.deinit();
    var buf: [E.max_rows]u32 = undefined;
    { // a warm-up prompt in its own cache, so the first timed prefill doesn't pay the first launch
        var warm = try E.Cache.init(init.gpa, e, E.max_rows + 64);
        defer warm.deinit(init.gpa);
        const k = @min(E.max_rows, prompts[0].len);
        @memcpy(buf[0..k], prompts[0][0..k]);
        try e.forward(&.{.{ .cache = &warm, .rows = k }}, buf[0..k], .last);
        try e.stream.synchronize();
    }
    const clock = std.Io.Clock.awake;
    var prefill_ns: i96 = 0;
    var save_ns: i96 = 0;
    var bytes: u64 = 0;
    for (prompts, 0..) |ids, i| {
        if (tail_n == 0 or tail_n >= ids.len) return error.MarkOutOfRange;
        const at: u32 = @intCast(ids.len - tail_n);
        var cache = try E.Cache.init(init.gpa, e, ids.len + 64);
        defer cache.deinit(init.gpa);
        const t0 = clock.now(init.io).toNanoseconds();
        var done: usize = 0;
        while (done < at) {
            const k = @min(E.max_rows, at - done);
            @memcpy(buf[0..k], ids[done..][0..k]);
            try e.forward(&.{.{ .cache = &cache, .rows = k }}, buf[0..k], .last);
            done += k;
        }
        try e.stream.synchronize();
        const t1 = clock.now(init.io).toNanoseconds();
        const copy = try q.reuse.saveCache(e, &cache, at, &.{}, false);
        defer q.reuse.dropCopy(e, copy);
        var ranges: std.ArrayList(hs.Range) = .empty;
        try hs.cudaRanges(hs.qwen27, at, 0, &ranges, a);
        const body = try a.alloc(u8, @intCast(hs.qwen27.bytes(at)));
        defer a.free(body);
        try hs.gather(copy.host.bytes[0..copy.len], ranges.items, body);
        const t2 = clock.now(init.io).toNanoseconds();
        prefill_ns += t1 - t0;
        save_ns += t2 - t1;
        bytes += body.len;
        const header = hs.Header.of(hs.qwen27, at, ids);
        const path = try std.fmt.allocPrint(a, "{s}/s{d}.tfhs", .{ args[5], i });
        var f = try std.Io.Dir.cwd().createFile(init.io, path, .{});
        defer f.close(init.io);
        try f.writeStreamingAll(init.io, std.mem.asBytes(&header));
        try f.writeStreamingAll(init.io, body);
    }
    std.debug.print("RESULT export-many: {d} states, {d} bytes; prefill {d:.3} s, save and order {d:.3} s, one prompt after another\n", .{ prompts.len, bytes, @as(f64, @floatFromInt(prefill_ns)) / 1e9, @as(f64, @floatFromInt(save_ns)) / 1e9 });
    return 0;
}

pub fn run(init: std.process.Init, a: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    if (args.len != 6) return 2;
    if (std.mem.eql(u8, args[2], "tail")) return tail(init, a, args);
    if (std.mem.eql(u8, args[2], "export-many")) return many(init, a, args);
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], a, .limited(16 << 20));
    const ids = (try std.json.parseFromSlice([]u32, a, text, .{})).value;
    const at = try std.fmt.parseInt(u32, args[4], 10);
    if (at == 0 or at >= ids.len) return error.MarkOutOfRange;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const e = try E.Engine.init(init.gpa, init.io, &ctx, args[1], false, null);
    defer e.deinit();
    // the copy is this model's state in this engine's order: its sizes must be the hybrid order's, piece for piece
    const l = q.reuse.layoutOf(e, at);
    if (l.total() != hs.qwen27.bytes(at) or e.g.stateBytes() != hs.qwen27.fixed() or e.g.recBytes() != hs.qwen27.rec_bytes or
        e.g.convBytes() != hs.qwen27.conv_bytes or e.g.deltas() != hs.qwen27.linear) return error.HybridGeometryDiffers;
    var cache = try E.Cache.init(init.gpa, e, ids.len + 64);
    defer cache.deinit(init.gpa);
    var buf: [E.max_rows]u32 = undefined;
    { // a warm-up prompt in its own cache, so the timed prefill doesn't pay the first launch
        var warm = try E.Cache.init(init.gpa, e, E.max_rows + 64);
        defer warm.deinit(init.gpa);
        const k = @min(E.max_rows, ids.len);
        @memcpy(buf[0..k], ids[0..k]);
        try e.forward(&.{.{ .cache = &warm, .rows = k }}, buf[0..k], .last);
        try e.stream.synchronize();
    }
    const t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
    var done: usize = 0;
    while (done < at) {
        const k = @min(E.max_rows, at - done);
        @memcpy(buf[0..k], ids[done..][0..k]);
        try e.forward(&.{.{ .cache = &cache, .rows = k }}, buf[0..k], .last);
        done += k;
    }
    try e.stream.synchronize();
    const t1 = std.Io.Clock.awake.now(init.io).toNanoseconds();
    const copy = try q.reuse.saveCache(e, &cache, at, &.{}, false);
    defer q.reuse.dropCopy(e, copy);
    if (copy.delta != 0) return error.RotaryShift; // a text prompt's: an image prompt's state is not this tool's
    var ranges: std.ArrayList(hs.Range) = .empty;
    try hs.cudaRanges(hs.qwen27, at, 0, &ranges, a); // the copy's host bytes, without cuda_disk.zig's words
    const body = try a.alloc(u8, @intCast(hs.qwen27.bytes(at)));
    try hs.gather(copy.host.bytes[0..copy.len], ranges.items, body);
    const t2 = std.Io.Clock.awake.now(init.io).toNanoseconds();
    const header = hs.Header.of(hs.qwen27, at, ids);
    var f = try std.Io.Dir.cwd().createFile(init.io, args[5], .{});
    defer f.close(init.io);
    try f.writeStreamingAll(init.io, std.mem.asBytes(&header));
    try f.writeStreamingAll(init.io, body);
    const ms = struct {
        fn of(ns: i96) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.of;
    std.debug.print("RESULT export: CUDA state after {d} of {d} tokens, {d} bytes, to {s}; prefill {d:.1} ms, save and order {d:.1} ms\n", .{ at, ids.len, body.len, args[5], ms(t1 - t0), ms(t2 - t1) });
    return 0;
}
