//! Speech arriving mid-stage in tf-h3-image-server: each stage gives way and its output is unchanged.
const std = @import("std");
const mtl = @import("metal");
const h3 = @import("h3");
const prompt_tool = @import("h3_image_prompt.zig");
const server = @import("h3_image_server.zig");
const testing = std.testing;

/// An io whose `nth` exclusive tryLock (a stage's probe) first takes speech for `ms`; `join` after the stage.
const Arrival = struct {
    var base: std.Io = undefined;
    var vtable: std.Io.VTable = undefined;
    var lock: []const u8 = undefined;
    var nth: usize = 0;
    var hold_ms: i64 = 0;
    var speech: ?std.Thread = null;
    var watched: []const f32 = &.{};
    var untouched = false;

    fn at(io: std.Io, path: []const u8, probe: usize, ms: i64) h3.gpu_turn.Turn {
        base = io;
        vtable = io.vtable.*;
        vtable.fileTryLock = tryLock;
        lock = path;
        nth = probe;
        hold_ms = ms;
        speech = null;
        watched = &.{};
        return .{ .io = .{ .userdata = io.userdata, .vtable = &vtable }, .at = path };
    }

    fn tryLock(userdata: ?*anyopaque, file: std.Io.File, kind: std.Io.File.Lock) std.Io.File.LockError!bool {
        if (kind == .exclusive and nth > 0) {
            nth -= 1;
            if (nth == 0) speech = speak() catch null;
        }
        return base.vtable.fileTryLock(userdata, file, kind);
    }

    /// Speech holding `lock` for `hold_ms` from now, released from its own thread.
    fn speak() !std.Thread {
        const held = h3.gpu_turn.speak(base, lock) orelse return error.NoSpeechHold;
        return std.Thread.spawn(.{}, release, .{held});
    }

    fn release(held: std.Io.File) void {
        base.sleep(.fromMilliseconds(hold_ms), .awake) catch {};
        untouched = std.mem.allEqual(f32, watched, 0);
        h3.gpu_turn.done(base, held);
    }

    fn join() !void {
        (speech orelse return error.NoSpeechArrived).join();
    }
};

test "speech arriving during the prompt encoder, denoising or the decoder holds its next pass, output unchanged" {
    const io = testing.io;
    const device = try mtl.Device.init();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const lock = try std.fs.path.join(a, &.{ dir_path, "speech.lock" });
    const hold_ms = 500;
    const least = 0.25; // seconds the next pass must wait: the hold less any delay before its probe, with room to spare

    // the prompt encoder and the token refiner, on the prompt selftest's tiny checkpoint
    var prng = std.Random.DefaultPrng.init(0x5133);
    const tiny = try prompt_tool.Tiny.write(io, a, prng.random(), dir_path);
    const g = prompt_tool.tiny_text;
    var runner = try h3.Runner.init(device);
    defer runner.deinit();
    var text_w = try h3.weights.loadText(testing.allocator, io, device, tiny.text_dir, g);
    defer text_w.deinit(testing.allocator);
    var front_w = try h3.weights.loadRefiner(testing.allocator, io, device, tiny.front_dir, prompt_tool.tiny_front);
    defer front_w.deinit(testing.allocator);
    const T = 70;
    const ids = try a.alloc(u32, T);
    for (ids, 0..) |*id, i| id.* = @intCast(i % g.vocab);
    const pos = try a.alloc([3]i32, T);
    for (pos, 0..) |*p, i| p.* = @splat(@intCast(i));
    const rows = try a.alloc(u16, T * g.hidden);
    const text = try a.alloc(u16, T * prompt_tool.tiny_front.hidden);
    try runner.encode(&text_w, ids, pos, null, rows);
    try runner.refine(&front_w, rows, text);
    const held_rows = try a.alloc(u16, rows.len);
    const held_text = try a.alloc(u16, text.len);
    // the second probe: speech arrives once the first of the two layers (or of the refiner's three passes) is submitted
    runner.give_way = Arrival.at(io, lock, 2, hold_ms);
    try runner.encode(&text_w, ids, pos, null, held_rows);
    try Arrival.join();
    try testing.expect(runner.paused >= least);
    runner.give_way = Arrival.at(io, lock, 2, hold_ms);
    try runner.refine(&front_w, held_rows, held_text);
    try Arrival.join();
    try testing.expect(runner.paused >= least);
    try testing.expectEqualSlices(u16, rows, held_rows);
    try testing.expectEqualSlices(u16, text, held_text);

    // the decoder: 2 x 2 tiles of a 20 x 28 cell latent, one pass each
    const d = try h3.Decoder.init(testing.allocator, device, 2, 100);
    defer d.deinit();
    try d.synthetic(0x5eed);
    const config: h3.vae.Config = .{ .mean = @splat(0), .std = @splat(1), .tile = 256, .overlap = 64, .rope_theta = 100 };
    const latent = try a.alloc(f32, try h3.host.latentLength(20, 28));
    for (latent, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) / 8 - 1;
    const pixels = try a.alloc(f32, 3 * 20 * 28 * h3.vae.cell_pixels * h3.vae.cell_pixels);
    _ = try d.decode(a, &config, latent, 20, 28, 2, 1, pixels);
    const held_pixels = try a.alloc(f32, pixels.len);
    d.give_way = Arrival.at(io, lock, 2, hold_ms); // after the first of four tiles
    const took = try d.decode(a, &config, latent, 20, 28, 2, 1, held_pixels);
    try Arrival.join();
    try testing.expect(took.paused >= least);
    try testing.expectEqualSlices(f32, pixels, held_pixels);

    // denoising: one step of the resident transformer
    if (!device.tensorUnits()) return;
    var environ_map: std.process.Environ.Map = .init(testing.allocator);
    defer environ_map.deinit();
    const init = std.process.Init{
        .minimal = .{ .environ = .empty, .args = .{ .vector = &.{} } },
        .arena = &arena,
        .gpa = testing.allocator,
        .io = io,
        .environ_map = &environ_map,
        .preopens = .empty,
    };
    var f = try server.residentFixture(init, a, tmp.dir, dir_path, device);
    defer {
        for (&f.cases) |*c| c.case.file.deinit();
        f.folder.heads.deinit();
        for (f.folder.ck.shards) |*shard| shard.deinit();
        testing.allocator.free(f.folder.ck.shards);
    }
    f.folder.turn = .{ .io = io, .at = lock };
    const case = &f.cases[0].case;
    var b = try server.Binding.init(testing.allocator, &f.folder, case, f.cases[0].tags, null);
    defer b.release();
    const velocity = try a.alloc(f32, case.noise.len);
    _ = try b.model.forward(case.noise, &.{}, 0, velocity, &.{});
    const held_velocity = try a.alloc(f32, velocity.len);
    // the third probe, after `mark` and the forward's first yield: speech arrives once the first block is submitted
    b.model.give_way = Arrival.at(io, lock, 3, hold_ms);
    const step = try b.model.forward(case.noise, &.{}, 0, held_velocity, &.{});
    try Arrival.join();
    try testing.expect(step.paused >= least and !step.timing_valid);
    try testing.expectEqualSlices(f32, velocity, held_velocity);
    // the probe after the last block's: speech arriving as the last blocks drain holds the output heads
    const out = b.model.video_velocity.slice(f32, b.model.video * b.model.video_in);
    @memset(out, 0);
    b.model.give_way = Arrival.at(io, lock, 3 + b.model.blocks.len, hold_ms);
    Arrival.watched = out;
    const last = try b.model.forward(case.noise, &.{}, 0, held_velocity, &.{});
    try Arrival.join();
    try testing.expect(Arrival.untouched and last.paused >= least and !last.timing_valid);
    try testing.expectEqualSlices(f32, velocity, held_velocity);
}
