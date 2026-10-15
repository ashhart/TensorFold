//! Flash Next model checks on HIP: `tf-flashnext-hip-test generate <dir> [group flags] [--text T] [--chat] [--max N]`.
const std = @import("std");
const hip = @import("hip");
const fx = @import("flashnext");
const tokenizer = @import("tokenizer");

/// No per-thread alternate signal stack: its thread-local leaves RCCL's and HIP's threads too little stack to start.
pub const std_options: std.Options = .{ .signal_stack_size = null };

const usage = "usage: tf-flashnext-hip-test generate <model dir> [--tp N --rank R --master H --port P --policy P] [--text T] [--chat] [--max N] [--capacity N] [--repeat N] [--chunk N] [--hash]\n";

const Args = struct {
    dir: []const u8 = "",
    world: usize = 1,
    rank: usize = 0,
    master: []const u8 = "127.0.0.1",
    port: u16 = 29561,
    policy: []const u8 = "",
    text: []const u8 = "The capital of France is",
    chat: bool = false,
    max: usize = 32,
    capacity: usize = 8192,
    repeat: usize = 1,
    chunk: usize = 512,
    /// Print a hash of every step's whole logits row, to compare builds and policies bit for bit.
    hash: bool = false,
};

fn parse(args: []const [:0]const u8) !Args {
    var a: Args = .{ .dir = args[0] };
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const name = args[i];
        if (std.mem.eql(u8, name, "--chat") or std.mem.eql(u8, name, "--hash")) {
            if (name[2] == 'c') a.chat = true else a.hash = true;
            continue;
        }
        i += 1;
        if (i >= args.len) return error.MissingArgument;
        const v = args[i];
        if (std.mem.eql(u8, name, "--tp"))
            a.world = try std.fmt.parseInt(usize, v, 10)
        else if (std.mem.eql(u8, name, "--rank"))
            a.rank = try std.fmt.parseInt(usize, v, 10)
        else if (std.mem.eql(u8, name, "--master"))
            a.master = v
        else if (std.mem.eql(u8, name, "--port"))
            a.port = try std.fmt.parseInt(u16, v, 10)
        else if (std.mem.eql(u8, name, "--policy"))
            a.policy = v
        else if (std.mem.eql(u8, name, "--text"))
            a.text = v
        else if (std.mem.eql(u8, name, "--max"))
            a.max = try std.fmt.parseInt(usize, v, 10)
        else if (std.mem.eql(u8, name, "--capacity"))
            a.capacity = try std.fmt.parseInt(usize, v, 10)
        else if (std.mem.eql(u8, name, "--repeat"))
            a.repeat = try std.fmt.parseInt(usize, v, 10)
        else if (std.mem.eql(u8, name, "--chunk"))
            a.chunk = try std.fmt.parseInt(usize, v, 10)
        else
            return error.UnknownOption;
    }
    return a;
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Timestamp.now(io, .awake).nanoseconds;
}

/// Every rank runs the same prompt and greedy steps (the collectives pair up); rank 0 prints the text and the timings.
fn generate(gpa: std.mem.Allocator, io: std.Io, a: Args) !void {
    var driver = try hip.Runtime.open();
    const device: c_int = @intCast(a.rank);
    const caps = try hip.Device.kernelCaps(&driver, device);
    driver.close();
    var notes: hip.Policy.Notes = .{};
    var policy = try hip.Policy.resolve(a.policy, .current, &notes);
    const group: ?hip.collective.Group = if (a.world > 1) try hip.joinRun(
        gpa,
        io,
        @intCast(a.rank),
        @intCast(a.world),
        a.master,
        a.port,
        caps,
        &policy,
        a.dir,
    ) else null;
    defer if (group) |g| g.close(gpa);
    const t0 = nowNs(io);
    const e = try fx.engine.Engine.open(gpa, io, a.dir, .{ .device = device, .rank = a.rank, .world = a.world, .group = group, .policy = policy, .capacity = a.capacity, .chunk = a.chunk });
    defer e.deinit();
    const loud = a.rank == 0;
    if (loud) std.debug.print("loaded in {d:.1} s, {d:.2} GiB on the device\n", .{ @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e9, @as(f64, @floatFromInt(hip.usage(false).device)) / (1 << 30) });
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const tk_path = try std.fs.path.join(arena.allocator(), &.{ a.dir, "tokenizer.json" });
    var tok = try tokenizer.loadTokenizer(io, arena.allocator(), tk_path);
    const text = if (a.chat) try std.fmt.allocPrint(
        arena.allocator(),
        "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        .{a.text},
    ) else a.text;
    const prompt = try tok.encode(arena.allocator(), text);
    for (0..a.repeat) |_| {
        var c = try e.caches(a.capacity);
        defer c.deinit(gpa);
        const m = e.model_();
        var at: usize = 0;
        var start: usize = 0;
        const p0 = nowNs(io);
        var out: fx.forward.Out = undefined;
        while (at < prompt.len) {
            const n = @min(e.o.chunk, prompt.len - at);
            out = try e.prompt(&c, prompt[at..][0..n]);
            start = at;
            at += n;
        }
        const last = fx.forward.at(out.mixed, (prompt.len - 1 - start) * m.dims.hidden);
        var o = e.ops(&e.arena);
        var next: [1]u32 = undefined;
        var digest = std.hash.Fnv1a_64.init();
        const first = try e.logits(o, last, 1);
        if (a.hash) try mix(e, &digest, first);
        try e.drawer.draw(o, e.stream, first, &.{.{ .sampling = null, .position = prompt.len }}, &next, false);
        const p1 = nowNs(io);
        var reply: std.ArrayList(u32) = .empty;
        try reply.append(arena.allocator(), next[0]);
        for (1..a.max) |_| {
            const step = [1]u32{next[0]};
            const seg = [_]fx.forward.Seg{.{ .c = &c, .tokens = &step, .pos = c.len }};
            const y = (try e.round(&seg, try e.choose(&seg))).logits;
            o = e.ops(&e.arena);
            if (a.hash) try mix(e, &digest, y);
            try e.drawer.draw(o, e.stream, y, &.{.{ .sampling = null, .position = c.len + 1 }}, &next, false);
            try e.keepWindow(&c, &step, 1);
            try reply.append(arena.allocator(), next[0]);
        }
        try e.stream.synchronize();
        const p2 = nowNs(io);
        if (!loud) continue;
        const words = try tok.decode(arena.allocator(), reply.items, false);
        const pre = @as(f64, @floatFromInt(p1 - p0)) / 1e9;
        const dec = @as(f64, @floatFromInt(p2 - p1)) / 1e9;
        std.debug.print(
            "prompt {d} tokens in {d:.3} s ({d:.1} tok/s); {d} tokens in {d:.3} s ({d:.2} tok/s)\n",
            .{
                prompt.len,
                pre,
                @as(f64, @floatFromInt(prompt.len)) / pre,
                a.max - 1,
                dec,
                @as(f64, @floatFromInt(a.max - 1)) / dec,
            },
        );
        if (a.hash) std.debug.print("logits hash {x:0>16}\n", .{digest.final()});
        std.debug.print("ids {any}\n---\n{s}{s}\n---\n", .{ reply.items, text, words });
    }
}

/// One logits row folded into `digest`, read back after the stream's work.
fn mix(e: *fx.engine.Engine, digest: *std.hash.Fnv1a_64, y: hip.ops.Tensor) !void {
    const bytes = try e.gpa.alloc(u8, e.cfg.vocab * y.kind.size());
    defer e.gpa.free(bytes);
    try e.stream.synchronize();
    try hip.runtime.check(e.driver.api.hipMemcpyDtoH(bytes.ptr, @ptrFromInt(y.ptr), bytes.len));
    digest.update(bytes);
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3 or !std.mem.eql(u8, args[1], "generate")) {
        std.debug.print(usage, .{});
        return 2;
    }
    const a = parse(args[2..]) catch {
        std.debug.print(usage, .{});
        return 2;
    };
    generate(init.gpa, init.io, a) catch |err| {
        std.debug.print("FAIL {t}\n", .{err});
        if (@errorReturnTrace()) |t| std.debug.dumpErrorReturnTrace(t);
        return 1;
    };
    return 0;
}
