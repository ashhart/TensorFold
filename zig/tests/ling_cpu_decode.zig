//! Greedy Ling-3.0-tiny CPU decode from a GGUF and prompt token IDs; prints the IDs and tokens a second.
const std = @import("std");
const cpu = @import("ling_cpu");

const usage = "usage: ling-cpu-decode [--step-hashes] MODEL.gguf ID,ID,... COUNT [THREADS]\n";

fn top(values: []const f32) u32 {
    var best: usize = 0;
    for (values, 0..) |v, i| if (v > values[best]) {
        best = i;
    };
    return @intCast(best);
}

fn rate(tokens: usize, ns: i96) f64 {
    return @as(f64, @floatFromInt(tokens)) * 1e9 / @as(f64, @floatFromInt(@max(ns, 1)));
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    var buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(init.io, &buf);
    const w = &out.interface;
    var step_hashes = false;
    var pos: [4][]const u8 = undefined;
    var npos: usize = 0;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--step-hashes")) {
            step_hashes = true;
        } else if (npos < pos.len) {
            pos[npos] = arg;
            npos += 1;
        } else return error.Usage;
    }
    if (npos != 3 and npos != 4) {
        try w.writeAll(usage);
        try w.flush();
        return error.Usage;
    }
    var prompt: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeScalar(u8, pos[1], ',');
    while (it.next()) |id| try prompt.append(a, try std.fmt.parseInt(u32, id, 10));
    const count = try std.fmt.parseInt(u32, pos[2], 10);
    const threads = if (npos == 4) try std.fmt.parseInt(usize, pos[3], 10) else 1;
    if (prompt.items.len == 0 or count == 0 or prompt.items.len + count > 4096) return error.PromptCapacity;

    const opening = std.Io.Clock.awake.now(init.io);
    var file = try cpu.gguf.File.open(init.gpa, init.io, pos[0]);
    defer file.close(init.io);
    const c = try file.config();
    var session = try cpu.Gguf.Session.init(init.gpa, &file, c, @intCast(prompt.items.len + count), threads);
    defer session.deinit();
    const opened = std.Io.Clock.awake.now(init.io);
    try w.print("model opened in {d:.2} s, {d} threads, exact\n", .{ @as(f64, @floatFromInt(opening.durationTo(opened).toNanoseconds())) / 1e9, threads });
    try w.flush();

    const hashes: [][32]u8 = if (step_hashes) try a.alloc([32]u8, count) else &.{};
    const tokens: []u32 = if (step_hashes) try a.alloc(u32, count) else &.{};
    var next_hash: [32]u8 = undefined;
    var next: u32 = undefined;
    var at: usize = 0;
    while (at < prompt.items.len) {
        const end = @min(prompt.items.len, at + 8);
        const is_last = end == prompt.items.len;
        const step = try session.prefillChunk(prompt.items[at..end], is_last);
        if (is_last) {
            next = top(step.logits);
            if (step_hashes) next_hash = cpu.step_hash.logits(step.logits);
        }
        at = end;
    }
    const prefilled = std.Io.Clock.awake.now(init.io);
    const prefill_ns = opened.durationTo(prefilled).toNanoseconds();
    try w.print("prefill {d} tokens in {d:.2} s ({d:.2} tokens/s)\n", .{ prompt.items.len, @as(f64, @floatFromInt(prefill_ns)) / 1e9, rate(prompt.items.len, prefill_ns) });
    try w.writeAll("generated:");
    try w.flush();

    var made: usize = 0;
    var stopped = false;
    while (made < count) {
        try w.print(" {d}", .{next});
        try w.flush();
        if (step_hashes) {
            hashes[made] = next_hash;
            tokens[made] = next;
        }
        made += 1;
        const step = try session.step(&.{next});
        if (std.mem.indexOfScalar(u32, c.eos[0..c.eos_n], next) != null) {
            stopped = true;
            break;
        }
        next = top(step.logits);
        if (step_hashes and made < count) next_hash = cpu.step_hash.logits(step.logits);
    }
    const done = std.Io.Clock.awake.now(init.io);
    const decode_ns = prefilled.durationTo(done).toNanoseconds();
    try w.print("\ndecode {d} tokens in {d:.2} s ({d:.2} tokens/s), finish {s}\n", .{ made, @as(f64, @floatFromInt(decode_ns)) / 1e9, rate(made, decode_ns), if (stopped) "stop" else "length" });
    if (step_hashes) {
        for (hashes[0..made], tokens[0..made], 1..) |digest, token, index| {
            try w.print("step-hash {d} token {d} logits-sha256 {s}\n", .{ index, token, std.fmt.bytesToHex(digest, .lower) });
        }
        try w.print("step-hashes-sha256 {s}\n", .{std.fmt.bytesToHex(cpu.step_hash.sequence(hashes[0..made]), .lower)});
    }
    try w.flush();
}
