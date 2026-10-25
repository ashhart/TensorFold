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

/// `--loop`: one request a line on stdin, `REQ MAX STOPS [s=TEMP,TOP_P,TOP_K,PRESENCE,SEED] ID ID ...` (REQ: the front
/// end's request number; STOPS: ids with commas, or `-`). Without `s=` the reply is greedy. The presence penalty counts
/// the tokens of this reply only. Up to XPU_SLOTS requests run at once: each takes a free slot (a fresh sequence) and
/// its prompt, then the running slots decode together, one stepMulti row a slot. The runner writes `t REQ ID` for each
/// token as soon as it exists, then `end REQ stop|length|error PROMPT_S DECODE_S N`. A stop token ends the reply and is
/// not written. A front end (a server, a bench) drives it through a pipe and owns the tokenizer.
const Req = struct {
    req: u64 = 0,
    max: usize = 0,
    stops: std.ArrayList(i32) = .empty,
    ids: std.ArrayList(i32) = .empty,
    sampling: ?Sampling = null,
};

const Slot = struct {
    on: bool = false,
    req: Req = .{},
    out: std.ArrayList(i32) = .empty,
    next: i32 = -1, // the token to check, write and feed
    t0: f64 = 0,
    t1: f64 = 0,
};

/// One request line into `q`, or false at the end of stdin.
fn readReq(q: *std.ArrayList(Req)) !bool {
    var line: std.ArrayList(u8) = .empty;
    defer line.deinit(alloc);
    var byte: [1]u8 = undefined;
    while (true) {
        const n = read(0, &byte, 1);
        if (n <= 0) return false;
        if (byte[0] == '\n') break;
        try line.append(alloc, byte[0]);
    }
    var it = std.mem.tokenizeAny(u8, line.items, " ");
    var r: Req = .{};
    r.req = std.fmt.parseInt(u64, it.next() orelse return true, 10) catch return true;
    r.max = std.fmt.parseInt(usize, it.next() orelse "", 10) catch {
        say("end {d} error 0 0 0\n", .{r.req});
        return true;
    };
    const sw = it.next() orelse "-";
    var st = std.mem.tokenizeAny(u8, if (std.mem.eql(u8, sw, "-")) "" else sw, ",");
    while (st.next()) |w| try r.stops.append(alloc, std.fmt.parseInt(i32, w, 10) catch 0);
    while (it.next()) |w| {
        if (std.mem.startsWith(u8, w, "s=")) {
            r.sampling = Sampling.parse(w) catch null;
            continue;
        }
        try r.ids.append(alloc, std.fmt.parseInt(i32, w, 10) catch 0);
    }
    try q.append(alloc, r);
    return true;
}

fn stdinReady() bool {
    var fds = [1]std.posix.pollfd{.{ .fd = 0, .events = std.posix.POLL.IN, .revents = 0 }};
    return (std.posix.poll(&fds, 0) catch 0) > 0;
}

fn loop(r: *sage.Runner) !void {
    var q: std.ArrayList(Req) = .empty;
    var slots: [sage.MAX_SLOTS]Slot = @splat(.{});
    const logits = try alloc.alloc(f16, r.z.s.vocab);
    defer alloc.free(logits);
    var open_in = true;
    say("ready\n", .{});
    while (true) {
        var busy: usize = 0;
        for (slots[0..r.slots]) |sl| busy += @intFromBool(sl.on);
        if (busy == 0 and q.items.len == 0) {
            if (!open_in or !try readReq(&q)) return;
        }
        while (open_in and stdinReady()) open_in = try readReq(&q);
        // admit: a free slot takes the oldest request and its prompt
        for (slots[0..r.slots], 0..) |*sl, i| {
            if (sl.on or q.items.len == 0) continue;
            const req = q.orderedRemove(0);
            if (req.ids.items.len == 0 or req.ids.items.len + req.max > r.ctx) {
                say("end {d} error 0 0 0\n", .{req.req});
                continue;
            }
            sl.* = .{ .on = true, .req = req, .t0 = now() };
            try r.use(@intCast(i));
            try r.reset();
            try r.prompt(req.ids.items);
            sl.next = r.last;
            if (sl.req.sampling) |*sp| {
                try r.logits(logits);
                sl.next = sp.pick(logits, sl.out.items);
            }
            sl.t1 = now();
        }
        // check, write and count each running slot's token; the ones to feed go to one step
        var feed: [sage.MAX_SLOTS]u32 = undefined;
        var toks: [sage.MAX_SLOTS]i32 = undefined;
        var nf: usize = 0;
        for (slots[0..r.slots], 0..) |*sl, i| {
            if (!sl.on) continue;
            var reason: ?[]const u8 = null;
            if (std.mem.indexOfScalar(i32, sl.req.stops.items, sl.next) != null) {
                reason = "stop";
            } else {
                say("t {d} {d}\n", .{ sl.req.req, sl.next });
                try sl.out.append(alloc, sl.next);
                if (sl.out.items.len == sl.req.max) reason = "length";
            }
            if (reason) |why| {
                say("end {d} {s} {d:.6} {d:.6} {d}\n", .{ sl.req.req, why, sl.t1 - sl.t0, now() - sl.t1, sl.out.items.len });
                sl.req.stops.deinit(alloc);
                sl.req.ids.deinit(alloc);
                sl.out.deinit(alloc);
                sl.* = .{};
                continue;
            }
            feed[nf] = @intCast(i);
            toks[nf] = sl.next;
            nf += 1;
        }
        if (nf == 0) continue;
        if (nf == 1) { // one stream: the single-row forward
            try r.use(feed[0]);
            try r.step(toks[0]);
            toks[0] = r.last;
        } else try r.stepMulti(feed[0..nf], toks[0..nf], toks[0..nf]);
        for (feed[0..nf], 0..) |si, j| {
            const sl = &slots[si];
            sl.next = toks[j];
            if (sl.req.sampling) |*sp| {
                if (nf == 1) try r.logits(logits) else try r.rowLogits(@intCast(j), logits);
                sl.next = sp.pick(logits, sl.out.items);
            }
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) {
        std.debug.print("usage: xpu-sage MODEL_DIR IDS_FILE [STEPS]\n       xpu-sage MODEL_DIR --loop\n", .{});
        return;
    }
    const ctx: u32 = if (std.c.getenv("XPU_CTX")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 4096;
    const slots: u32 = if (std.c.getenv("XPU_SLOTS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 1;
    if (std.mem.eql(u8, argv[2], "--tower") and argv.len > 4) return tower(init.io, argv[1], argv[3], argv[4]);
    if (std.mem.eql(u8, argv[2], "--loop")) {
        const r = try sage.Runner.init(argv[1], ctx, slots);
        defer r.deinit();
        return loop(r);
    }
    const steps: usize = if (argv.len > 3 and !std.mem.eql(u8, argv[3], "--nll")) try std.fmt.parseInt(usize, argv[3], 10) else 32;
    var ids: std.ArrayList(i32) = .empty;
    const path = try std.fmt.allocPrintSentinel(alloc, "{s}", .{argv[2]}, 0);
    const fd = open(path.ptr, 0);
    if (fd < 0) return error.FileNotFound;
    const size: usize = @intCast(lseek(fd, 0, 2));
    const text: [*]const u8 = @ptrCast(mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed);
    var it = std.mem.tokenizeAny(u8, text[0..size], " \n,[]");
    while (it.next()) |w| try ids.append(alloc, try std.fmt.parseInt(i32, w, 10));
    const t_load = now();
    const r = try sage.Runner.init(argv[1], ctx, slots);
    defer r.deinit();
    std.debug.print("loaded in {d:.1} s\n", .{now() - t_load});
    if (argv.len > 3 and std.mem.eql(u8, argv[3], "--nll")) return nll(r, ids.items);
    if (slots > 1) return multiCheck(r, ids.items, steps);
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

/// XPU_SLOTS > 1: slot i takes the prompt less its last 7 i tokens; `steps` stepMulti rounds over all slots, timed,
/// then each slot again alone (Runner.step) on slot 0, and whether its tokens match.
fn multiCheck(r: *sage.Runner, ids: []const i32, steps: usize) !void {
    const n = r.slots;
    var slots: [sage.MAX_SLOTS]u32 = undefined;
    var toks: [sage.MAX_SLOTS]i32 = undefined;
    const got = try alloc.alloc(i32, n * steps);
    defer alloc.free(got);
    for (0..n) |i| {
        slots[i] = @intCast(i);
        try r.use(@intCast(i));
        try r.reset();
        try r.prompt(ids[0 .. ids.len - 7 * i]);
        toks[i] = r.last;
        got[i * steps] = r.last;
    }
    @import("dev.zig").profReport("prompts");
    const t0 = now();
    for (1..steps) |k| {
        try r.stepMulti(slots[0..n], toks[0..n], toks[0..n]);
        for (0..n) |i| got[i * steps + k] = toks[i];
    }
    const dt = now() - t0;
    @import("dev.zig").profReport("decode");
    std.debug.print("multi: {d} slots, {d} steps in {d:.3} s ({d:.1} tok/s a slot, {d:.1} aggregate)\n", .{ n, steps - 1, dt, @as(f64, @floatFromInt(steps - 1)) / dt, @as(f64, @floatFromInt(n * (steps - 1))) / dt });
    try r.use(0);
    for (0..n) |i| {
        try r.reset();
        try r.prompt(ids[0 .. ids.len - 7 * i]);
        var same: usize = 0;
        var first_diff: usize = steps;
        for (0..steps) |k| {
            if (k > 0) try r.step(r.last);
            if (r.last == got[i * steps + k]) same += 1 else if (first_diff == steps) {
                first_diff = k;
            }
        }
        std.debug.print("slot {d}: {d}/{d} tokens as alone, first difference at {d}\n", .{ i, same, steps, first_diff });
    }
}

/// `--nll`: the mean negative log likelihood of each id given the ones before it, one decode step a token, from the
/// fp16 logits: a check that two checkpoints of one model predict text alike.
fn nll(r: *sage.Runner, ids: []const i32) !void {
    const logits = try alloc.alloc(f16, r.z.s.vocab);
    defer alloc.free(logits);
    var sum: f64 = 0;
    var top1: usize = 0;
    try r.prompt(ids[0..1]);
    for (1..ids.len) |i| {
        if (i > 1) try r.step(ids[i - 1]);
        try r.logits(logits);
        var mx: f64 = -std.math.inf(f64);
        var best: usize = 0;
        for (logits, 0..) |l, j| if (@as(f64, l) > mx) {
            mx = l;
            best = j;
        };
        var z: f64 = 0;
        for (logits) |l| z += @exp(@as(f64, l) - mx);
        sum += mx + @log(z) - @as(f64, logits[@intCast(ids[i])]);
        top1 += @intFromBool(best == @as(usize, @intCast(ids[i])));
    }
    const n: f64 = @floatFromInt(ids.len - 1);
    std.debug.print("nll: {d} tokens, mean {d:.4}, top-1 {d:.3}\n", .{ ids.len - 1, sum / n, @as(f64, @floatFromInt(top1)) / n });
}

/// `--tower IMAGE OUT`: the image's patches through the vision tower, as a check against a reference: OUT.grid (gh gw),
/// OUT.pixels (fp32 patches) and OUT.rows (fp32 [tokens, hidden]).
fn tower(io: std.Io, dir: []const u8, image: []const u8, out: []const u8) !void {
    const core = @import("core");
    const vision = @import("vision.zig");
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const a = arena.allocator();
    const readFile = struct {
        fn f(al: std.mem.Allocator, i: std.Io, d: []const u8, name: []const u8) ?[]const u8 {
            const p = std.fs.path.join(al, &.{ d, name }) catch return null;
            return std.Io.Dir.cwd().readFileAlloc(i, p, al, .limited(64 << 20)) catch null;
        }
    }.f;
    const c = (try core.qwen_vl.Config.parse(a, readFile(a, io, dir, "config.json") orelse return error.NoConfig, readFile(a, io, dir, "preprocessor_config.json"), readFile(a, io, dir, "processor_config.json"))) orelse return error.NoVisionConfig;
    const bytes = readFile(a, io, "", image) orelse return error.NoImage;
    const p = try core.qwen_vl.geometry.prepare(io, a, bytes, c, 4096);
    const r = try sage.Runner.init(dir, 4096, 1);
    defer r.deinit();
    const t = try vision.Tower.load(&r.d, &r.ckpt, "vision_tower.", .{ .hidden = c.hidden, .heads = c.heads, .depth = c.depth, .intermediate = c.intermediate, .out_hidden = c.out_hidden }, p.gh * p.gw);
    defer t.deinit();
    const rows = try a.alloc(f32, @as(usize, p.tokens) * c.out_hidden);
    const dst = try r.d.alloc_(rows.len * 4);
    const t0 = now();
    try t.encode(&r.d, p.pixels, p.gh, p.gw, dst);
    const t1 = now();
    try @import("ze").check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, rows.ptr, dst, rows.len * 4, null, 0, null));
    try r.d.sync();
    std.debug.print("tower: {d} x {d} patches, {d} tokens in {d:.3} s\n", .{ p.gh, p.gw, p.tokens, t1 - t0 });
    const grid = [2]u32{ p.gh, p.gw };
    for ([_][]const u8{ "grid", "pixels", "rows" }, [_][]const u8{ std.mem.sliceAsBytes(&grid), std.mem.sliceAsBytes(p.pixels), std.mem.sliceAsBytes(rows) }) |ext, data| {
        const path = try std.fmt.allocPrintSentinel(a, "{s}.{s}", .{ out, ext }, 0);
        const fd = open(path.ptr, 0o1101, @as(c_int, 0o644)); // O_WRONLY | O_CREAT | O_TRUNC
        if (fd < 0) return error.CannotWrite;
        var done: usize = 0;
        while (done < data.len) done += @intCast(@max(1, write(fd, data.ptr + done, data.len - done)));
    }
}
