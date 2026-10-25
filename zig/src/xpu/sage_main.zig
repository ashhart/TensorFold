//! Development runner of the Sage-A1.5 XPU forward (first form): greedy tokens and timings.
const std = @import("std");
const sage = @import("sage.zig");
const now = @import("qwen27.zig").now;

const alloc = std.heap.c_allocator;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;

fn say(comptime f: []const u8, args: anytype) void {
    var buf: [256]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, f, args) catch return;
    _ = write(1, line.ptr, line.len);
}

/// Sampling on the host from the fp16 logits: presence penalty on the tokens of this reply, temperature, top-k, top-p.
const Sampling = struct {
    temp: f32 = 0,
    top_p: f32 = 1,
    top_k: usize = 0,
    presence: f32 = 0,
    rng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(0),

    /// `s=TEMP,TOP_P,TOP_K,PRESENCE,SEED`
    fn parse(w: []const u8) !Sampling {
        var it = std.mem.tokenizeScalar(u8, w[2..], ',');
        var s: Sampling = .{};
        s.temp = try std.fmt.parseFloat(f32, it.next() orelse return error.BadSampling);
        s.top_p = try std.fmt.parseFloat(f32, it.next() orelse return error.BadSampling);
        s.top_k = try std.fmt.parseInt(usize, it.next() orelse return error.BadSampling, 10);
        s.presence = try std.fmt.parseFloat(f32, it.next() orelse return error.BadSampling);
        s.rng = std.Random.DefaultPrng.init(try std.fmt.parseInt(u64, it.next() orelse "0", 10));
        return s;
    }

    fn pick(s: *Sampling, logits: []const f16, seen: []const i32) i32 {
        const K = 64;
        var top_i: [K]i32 = undefined;
        var top_v: [K]f32 = undefined;
        const k = @min(if (s.top_k == 0) K else s.top_k, K);
        var n: usize = 0;
        for (logits, 0..) |l, i| {
            var v: f32 = @floatCast(l);
            if (s.presence != 0 and std.mem.indexOfScalar(i32, seen, @intCast(i)) != null) v -= s.presence;
            if (n == k and v <= top_v[k - 1]) continue;
            var j = if (n < k) n else k - 1;
            if (n < k) n += 1;
            while (j > 0 and top_v[j - 1] < v) : (j -= 1) {
                top_v[j] = top_v[j - 1];
                top_i[j] = top_i[j - 1];
            }
            top_v[j] = v;
            top_i[j] = @intCast(i);
        }
        if (s.temp <= 0) return top_i[0];
        var p: [K]f32 = undefined;
        var sum: f32 = 0;
        for (0..n) |j| {
            p[j] = @exp((top_v[j] - top_v[0]) / s.temp);
            sum += p[j];
        }
        var keep: usize = 0;
        var acc: f32 = 0;
        while (keep < n) {
            acc += p[keep] / sum;
            keep += 1;
            if (acc >= s.top_p) break;
        }
        const u = s.rng.random().float(f32) * acc * sum;
        var c: f32 = 0;
        for (0..keep) |j| {
            c += p[j];
            if (u < c) return top_i[j];
        }
        return top_i[keep - 1];
    }
};

/// `--loop`: one request a line on stdin, `MAX STOPS [s=TEMP,TOP_P,TOP_K,PRESENCE,SEED] ID ID ...` (STOPS: ids with
/// commas, or `-`). Without `s=` the reply is greedy. The presence penalty counts the tokens of this reply only. For
/// each request the runner starts a fresh sequence, feeds the ids, and writes `t ID` for each token as soon as it exists, then
/// `end stop|length PROMPT_S DECODE_S N`. A stop token ends the reply and is not written. A front end (a server, a
/// bench) drives it through a pipe and owns the tokenizer.
fn loop(r: *sage.Runner) !void {
    var line: std.ArrayList(u8) = .empty;
    var ids: std.ArrayList(i32) = .empty;
    var stops: std.ArrayList(i32) = .empty;
    var out: std.ArrayList(i32) = .empty;
    const logits = try alloc.alloc(f16, r.z.s.vocab);
    defer alloc.free(logits);
    var byte: [1]u8 = undefined;
    say("ready\n", .{});
    while (true) {
        line.clearRetainingCapacity();
        while (true) {
            const n = read(0, &byte, 1);
            if (n <= 0) return;
            if (byte[0] == '\n') break;
            try line.append(alloc, byte[0]);
        }
        var it = std.mem.tokenizeAny(u8, line.items, " ");
        const max = std.fmt.parseInt(usize, it.next() orelse continue, 10) catch {
            say("end error 0 0 0\n", .{});
            continue;
        };
        stops.clearRetainingCapacity();
        const sw = it.next() orelse "-";
        var st = std.mem.tokenizeAny(u8, if (std.mem.eql(u8, sw, "-")) "" else sw, ",");
        while (st.next()) |w| try stops.append(alloc, try std.fmt.parseInt(i32, w, 10));
        ids.clearRetainingCapacity();
        var sampling: ?Sampling = null;
        while (it.next()) |w| {
            if (std.mem.startsWith(u8, w, "s=")) {
                sampling = Sampling.parse(w) catch null;
                continue;
            }
            try ids.append(alloc, try std.fmt.parseInt(i32, w, 10));
        }
        if (ids.items.len == 0 or ids.items.len + max > r.ctx) {
            say("end error 0 0 0\n", .{});
            continue;
        }
        try r.reset();
        out.clearRetainingCapacity();
        const t0 = now();
        try r.prompt(ids.items);
        if (sampling) |*sp| {
            try r.logits(logits);
            r.last = sp.pick(logits, out.items);
        }
        const t1 = now();
        var n: usize = 0;
        var reason: []const u8 = "length";
        while (n < max) {
            if (std.mem.indexOfScalar(i32, stops.items, r.last) != null) {
                reason = "stop";
                break;
            }
            say("t {d}\n", .{r.last});
            try out.append(alloc, r.last);
            n += 1;
            if (n == max) break;
            try r.step(r.last);
            if (sampling) |*sp| {
                try r.logits(logits);
                r.last = sp.pick(logits, out.items);
            }
        }
        say("end {s} {d:.6} {d:.6} {d}\n", .{ reason, t1 - t0, now() - t1, n });
    }
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) {
        std.debug.print("usage: xpu-sage MODEL_DIR IDS_FILE [STEPS]\n       xpu-sage MODEL_DIR --loop\n", .{});
        return;
    }
    const ctx: u32 = if (std.c.getenv("XPU_CTX")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 4096;
    if (std.mem.eql(u8, argv[2], "--loop")) {
        const r = try sage.Runner.init(argv[1], ctx);
        defer r.deinit();
        return loop(r);
    }
    const steps: usize = if (argv.len > 3) try std.fmt.parseInt(usize, argv[3], 10) else 32;
    var ids: std.ArrayList(i32) = .empty;
    const path = try std.fmt.allocPrintSentinel(alloc, "{s}", .{argv[2]}, 0);
    const fd = open(path.ptr, 0);
    if (fd < 0) return error.FileNotFound;
    const size: usize = @intCast(lseek(fd, 0, 2));
    const text: [*]const u8 = @ptrCast(mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed);
    var it = std.mem.tokenizeAny(u8, text[0..size], " \n,[]");
    while (it.next()) |w| try ids.append(alloc, try std.fmt.parseInt(i32, w, 10));
    const t_load = now();
    const r = try sage.Runner.init(argv[1], ctx);
    defer r.deinit();
    std.debug.print("loaded in {d:.1} s\n", .{now() - t_load});
    const t0 = now();
    try r.prompt(ids.items);
    const t1 = now();
    @import("dev.zig").profReport("prompt");
    var out: std.ArrayList(i32) = .empty;
    try out.append(alloc, r.last);
    for (1..steps) |_| {
        try r.step(r.last);
        try out.append(alloc, r.last);
    }
    const t2 = now();
    @import("dev.zig").profReport("decode");
    std.debug.print("prompt: {d} tokens in {d:.3} s ({d:.0} tok/s)\n", .{ ids.items.len, t1 - t0, @as(f64, @floatFromInt(ids.items.len)) / (t1 - t0) });
    std.debug.print("decode: {d} tokens in {d:.3} s ({d:.1} tok/s)\n", .{ steps - 1, t2 - t1, @as(f64, @floatFromInt(steps - 1)) / (t2 - t1) });
    std.debug.print("tokens:", .{});
    for (out.items) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
}
