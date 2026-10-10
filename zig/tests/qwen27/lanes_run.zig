//! A/B on real weights: the DFlash2 serial loop against the shared lane core, same prompt, same tokens.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const lb = @import("qwen27_host").lanes_backend;
const help = "tf-qwen27-lanes-run --model TARGET --draft DRAFTER --tokens IDS.json [--max-tokens 256 --repeats 2 --copies 0|1 --draft-mode q4|bf16]";

fn ids(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    return (try std.json.parseFromSlice([]u32, a, text, .{})).value;
}

fn ms(io: std.Io, since: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(since.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var target_dir: ?[]const u8 = null;
    var draft_dir: ?[]const u8 = null;
    var tokens_file: ?[]const u8 = null;
    var count: u32 = 256;
    var repeats: usize = 2;
    var copies = false;
    var mode: q.dflash.operators.Mode = .prepared_q4_reference;
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        const arg = args[i];
        const value = args[i + 1];
        if (std.mem.eql(u8, arg, "--model")) target_dir = value else if (std.mem.eql(u8, arg, "--draft")) draft_dir = value else if (std.mem.eql(u8, arg, "--tokens")) tokens_file = value else if (std.mem.eql(u8, arg, "--max-tokens")) count = try std.fmt.parseInt(u32, value, 10) else if (std.mem.eql(u8, arg, "--repeats")) repeats = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, arg, "--copies")) copies = std.mem.eql(u8, value, "1") else if (std.mem.eql(u8, arg, "--draft-mode")) mode = if (std.mem.eql(u8, value, "bf16")) .bf16_reference else .prepared_q4_reference else return error.UnknownOption;
    }
    if (target_dir == null or draft_dir == null or tokens_file == null or count < 2) {
        try std.Io.File.stdout().writeStreamingAll(io, help ++ "\n");
        return error.BadOptions;
    }
    const prompt = try ids(a, io, tokens_file.?);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const target = try q.model.Model.load(init.gpa, io, target_dir.?, 128);
    defer target.deinit();
    var runner = try q.decode_round.Runner.init(init.gpa, target, 1, @intCast(prompt.len + count + 64));
    defer runner.deinit();
    const drafter = try q.dflash.runtime_model.Model.load(init.gpa, io, target, draft_dir.?, mode);
    defer drafter.deinit();
    try drafter.setTreeBlock(16);
    const draft = try lb.Draft.attach(init.gpa, &runner, drafter, false);
    defer draft.deinit();
    const backend = try lb.Metal.init(init.gpa, &runner, draft);
    defer backend.deinit();
    try backend.measure(io);
    for (backend.costs[0..backend.timed]) |c| std.debug.print("window rows={d} ms={d:.2}\n", .{ c.width, c.ms });
    var cfg = try lanes.Config.init(init.gpa, backend.facts(), lb.max_rows, lb.max_rows - 1);
    defer cfg.deinit(init.gpa);
    var clock = lanes.backend.WallClock{ .io = io };
    for (0..repeats) |rep| {
        // A: the serial DFlash2 loop
        try draft.reset();
        const a0 = std.Io.Clock.awake.now(io);
        try draft.generation.prefill(prompt, 128);
        const prefill_ms = ms(io, a0);
        const d0 = std.Io.Clock.awake.now(io);
        const serial = try draft.generation.run(count, true, 15);
        defer init.gpa.free(serial.tokens);
        const serial_ms = ms(io, d0);
        // B: the lane core, prefill timed as A's
        var engine = lanes.Engine.init(init.gpa, &cfg, backend.backend(), clock.clock());
        defer engine.deinit();
        var lookup = try lanes.SuffixLookup.init(init.gpa, .{ .min_match = 4 });
        defer lookup.deinit();
        var s = try lanes.Stream.init(init.gpa, .{ .id = "q27", .prompt = prompt, .max_new = count, .proposer = if (copies) lookup.proposer() else null });
        defer s.deinit(init.gpa);
        const b0 = std.Io.Clock.awake.now(io);
        try engine.addStream(&s);
        while (engine.activeCount() > 0) try engine.step();
        const lanes_ms = ms(io, b0) - prefill_ms;
        const equal = std.mem.eql(u32, serial.tokens, s.emitted());
        std.debug.print("rep={d} serial: {d} tokens {d} rounds {d:.2}/round {d:.1} tok/s | lanes: {d} tokens {d} rounds {d:.2}/round {d:.1} tok/s | equal={}\n", .{
            rep,                                                                                              serial.tokens.len,                                                                         serial.rounds,
            @as(f64, @floatFromInt(serial.tokens.len - 1)) / @as(f64, @floatFromInt(@max(serial.rounds, 1))), @as(f64, @floatFromInt(serial.tokens.len)) * 1e3 / serial_ms,                              s.emitted().len,
            s.rounds,                                                                                         @as(f64, @floatFromInt(s.emitted().len - 1)) / @as(f64, @floatFromInt(@max(s.rounds, 1))), @as(f64, @floatFromInt(s.emitted().len)) * 1e3 / lanes_ms,
            equal,
        });
        std.debug.print("rep={d} lanes rows/round={d:.2} drafted={d} accepted={d}\n", .{ rep, 1 + @as(f64, @floatFromInt(s.drafted)) / @as(f64, @floatFromInt(@max(s.rounds, 1))), s.drafted, s.accepted });
        if (!equal) return error.LaneTokensDiffer;
    }
}
