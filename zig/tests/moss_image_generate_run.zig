//! tf-moss-image-generate's pipeline for any backend `B`: the family's options, schedule and noise, then its picture.
const std = @import("std");
const tf = @import("tensorfold");
const guide = @import("moss_image_guided.zig");

const generate = tf.moss_image.generate;
const config = tf.moss_image.config;
const noise = tf.moss_image.noise;
const schedule = tf.moss_image.schedule;

pub fn run(comptime B: type, init: std.process.Init) !u8 {
    const mi = B.mi;
    const a = init.arena.allocator();
    const io = init.io;
    const argv = try init.minimal.args.toSlice(a);
    if (argv.len < 2 or std.mem.startsWith(u8, argv[1], "--")) {
        std.debug.print("tf-moss-image-generate: TRANSFORMER_DIR is required\n{s}\n", .{generate.usage});
        return 2;
    }
    var refused: ?[]const u8 = null;
    var option: ?generate.Option = null;
    const args = generate.parse(argv[2..], &refused, &option) catch |e| {
        generate.complain(a, e, option, refused);
        return 2;
    };
    const transformer = argv[1];
    const checkpoint = std.fs.path.dirname(transformer) orelse ".";
    const grid = try config.grid(args.width, args.height);
    const rows = grid.rows * grid.columns;
    const count = rows * mi.config.channels;
    const steps: u32 = @intCast(args.steps);

    const path = try std.fs.path.join(a, &.{ checkpoint, "scheduler", "scheduler_config.json" });
    const settings = try schedule.parse(a, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)));
    const sched = try schedule.timesteps(a, settings, steps);
    const start = try a.alloc(f32, count);
    try noise.latents(a, args.seed, grid.rows, grid.columns, start);
    std.debug.print("seed {d}: {d} starting latents at {d}x{d}, {d} steps of the {d} asked for\n", .{ args.seed, count, args.width, args.height, sched.timesteps.len, args.steps });

    var gpu: B.Device = undefined;
    try gpu.init();
    defer gpu.deinit();
    var both: [2][]u16 = undefined;
    if (comptime @hasDecl(B.Device, "encode")) {
        const texts = [2][]const u8{ args.prompt, args.negative };
        std.debug.print("both prompts encoded natively: {d:.3} s\n", .{try gpu.encode(a, io, checkpoint, texts, &both)});
    } else {
        std.debug.print("tf-moss-image-generate: this build has no prompt encoder (the Metal build does)\n", .{});
        return 2;
    }
    const text = both[0].len / mi.config.context;
    const negative_rows = both[1].len / mi.config.context;
    // The transformer is freed when its run ends, before the decoder loads.
    const x = denoise: {
        const loading = B.seconds();
        const model = try gpu.load(std.heap.page_allocator, io, transformer, @max(text, negative_rows), args.width, args.height);
        defer model.deinit();
        std.debug.print("{s}: {d} image rows, prompts of {d} and {d} rows, guidance {d:.2}; loaded in {d:.1} s\n", .{ gpu.name(), rows, text, negative_rows, args.guidance, B.seconds() - loading });
        const refining = B.seconds();
        var branches: [2]mi.Prompt = undefined;
        branches[0] = try model.prompt(both[0]);
        defer branches[0].deinit();
        branches[1] = try model.prompt(both[1]);
        defer branches[1].deinit();
        std.debug.print("both prompts refined: {d:.3} s\n", .{B.seconds() - refining});

        const x = try a.dupe(f32, start);
        const timings = try a.alloc(guide.Timing, sched.timesteps.len);
        const total = try guide.denoise(B, a, model, &branches, x, sched.timesteps, sched.sigmas, args.guidance, generate.negative_skip, timings);
        for (timings, 0..) |t, s| std.debug.print("step {d}/{d}: {d:.3} s ({d:.3} s on the GPU)\n", .{ s + 1, sched.timesteps.len, t.wall, t.gpu });
        std.debug.print("denoise: {d:.2} s for {d} steps ({d} forwards)\n", .{ total, sched.timesteps.len, 2 * sched.timesteps.len });
        break :denoise x;
    };
    if (comptime @hasDecl(B.Device, "decode")) {
        const picture = try gpu.decode(a, io, checkpoint, x, args.width, args.height);
        std.debug.print("decoded {d}x{d}: {d:.3} s ({d:.3} s on the GPU), the decoder loaded in {d:.2} s\n", .{ args.width, args.height, picture.wall, picture.gpu, picture.load_s });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args.out, .data = picture.png });
        std.debug.print("wrote {s}\n", .{args.out});
        return 0;
    } else {
        std.debug.print("tf-moss-image-generate: this build has no image decoder (the Metal build does)\n", .{});
        return 2;
    }
}
