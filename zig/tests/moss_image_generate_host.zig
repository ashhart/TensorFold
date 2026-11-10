//! The generate driver on an inert backend: its schedule, seeded latents, both branches and the two refusals.
const std = @import("std");
const tf = @import("tensorfold");
const driver = @import("moss_image_generate_run.zig");
const t = std.testing;

/// One forward's timestep, skip, input and output, so a test can check the step and sigma indexing branch by branch.
const Call = struct { timestep: f32, skip: ?usize, first: f32, digest: u64, out_digest: u64 };
var calls: [2 * 200]Call = undefined;
var seen: usize = 0;

fn digest(values: []const f32) u64 {
    var sum: u64 = 0;
    for (values) |v| sum +%= @as(u64, @as(u32, @bitCast(v)));
    return sum;
}

/// Which branch a prompt belongs to, so a forward can tell the two apart by its own prompt.
const Prompt = struct {
    branch: usize = 0,

    pub fn deinit(_: *Prompt) void {}
};
const Timing = struct { wall: f64 = 0, gpu: f64 = 0, host: f64 = 0 };
const Picture = struct { png: []const u8, load_s: f64 = 0, wall: f64 = 0, gpu: f64 = 0 };

/// The branches answer differently, so the guidance term shows in the latents: the positive one doubles its
/// input and the negative one returns it as it is, so one step moves `x` by
/// `(x + guidance * (2 * x - x)) * (sigmas[s + 1] - sigmas[s])`.
const Model = struct {
    a: std.mem.Allocator,
    branches: usize = 0,

    pub fn deinit(m: *Model) void {
        m.a.destroy(m);
    }
    pub fn prompt(m: *Model, _: []const u16) !Prompt {
        const branch = m.branches;
        m.branches += 1;
        return .{ .branch = branch };
    }
    pub fn forward(_: *Model, p: *Prompt, input: []const f32, timestep: f32, skip: ?usize, out: []f32) !Timing {
        const twice: f32 = if (p.branch == 0) 2 else 1;
        for (out, input) |*o, x| o.* = twice * x;
        if (seen < calls.len) {
            calls[seen] = .{ .timestep = timestep, .skip = skip, .first = input[0], .digest = digest(input), .out_digest = digest(out) };
            seen += 1;
        }
        return .{};
    }
};

/// The geometry and the per-branch type the driver reads off the backend.
const Inert = struct {
    pub const config = struct {
        pub const channels: usize = 128;
        pub const context: usize = 4096;
    };
    pub const Model = @import("moss_image_generate_host.zig").Model;
    pub const Prompt = @import("moss_image_generate_host.zig").Prompt;
};

/// The latents the decoder received, the guided loop's last word on them: the 32x32 fixture's two rows of two.
var decoded: [2 * 2 * Inert.config.channels]f32 = undefined;
var decoded_len: usize = 0;

/// The device all three backends share: it hands out the inert model and needs no checkpoint.
const Loaded = struct {
    pub fn init(_: anytype) !void {}
    pub fn deinit(_: anytype) void {}
    pub fn name(_: anytype) []const u8 {
        return "inert generate fixture";
    }
    pub fn load(_: anytype, a: std.mem.Allocator, _: std.Io, _: []const u8, _: usize, _: usize, _: usize) !*Model {
        const m = try a.create(Model);
        m.* = .{ .a = a };
        return m;
    }
};

/// Two context rows of bf16 words per prompt, from the text's own length so the two rows differ.
fn encodeRows(a: std.mem.Allocator, texts: [2][]const u8, rows: *[2][]u16) !f64 {
    for (texts, rows) |text, *r| {
        r.* = try a.alloc(u16, 2 * Inert.config.context);
        for (r.*, 0..) |*w, i| w.* = @truncate(text.len *% 3 +% i);
    }
    return 0;
}

const png = "\x89PNG\r\n\x1a\n";

const FullDevice = struct {
    pub const init = Loaded.init;
    pub const deinit = Loaded.deinit;
    pub const name = Loaded.name;
    pub const load = Loaded.load;
    pub fn encode(_: *FullDevice, a: std.mem.Allocator, _: std.Io, _: []const u8, texts: [2][]const u8, rows: *[2][]u16) !f64 {
        return encodeRows(a, texts, rows);
    }
    pub fn decode(_: *FullDevice, a: std.mem.Allocator, _: std.Io, _: []const u8, latents: []const f32, width: usize, height: usize) !Picture {
        _ = .{ &a, &width, &height };
        decoded_len = latents.len;
        @memcpy(decoded[0..latents.len], latents);
        return .{ .png = png };
    }
};

/// The two devices missing one native half: each absence is a refusal of its own. `Loaded` is the encoderless one.
const DecoderlessDevice = struct {
    pub const init = Loaded.init;
    pub const deinit = Loaded.deinit;
    pub const name = Loaded.name;
    pub const load = Loaded.load;
    pub fn encode(_: *DecoderlessDevice, a: std.mem.Allocator, _: std.Io, _: []const u8, texts: [2][]const u8, rows: *[2][]u16) !f64 {
        return encodeRows(a, texts, rows);
    }
};

/// The backend the driver runs on, and the two that are missing one native half.
const Backend = struct {
    pub const st = @import("safetensors");
    pub const mi = Inert;
    pub fn seconds() f64 {
        return 0;
    }
    pub const Device = FullDevice;
};
const NoEncoder = struct {
    pub const st = Backend.st;
    pub const mi = Backend.mi;
    pub const seconds = Backend.seconds;
    pub const Device = Loaded;
};
const NoDecoder = struct {
    pub const st = Backend.st;
    pub const mi = Backend.mi;
    pub const seconds = Backend.seconds;
    pub const Device = DecoderlessDevice;
};

/// The released checkpoint's scheduler config, the one file the driver reads (through TRANSFORMER_DIR's parent).
const scheduler_config =
    \\{"_class_name": "MOSSImageScheduler", "_diffusers_version": "0.38.0", "base_image_seq_len": 256, "base_shift": 0.5,
    \\ "invert_sigmas": false, "max_image_seq_len": 4096, "max_shift": 1.15, "num_train_timesteps": 1000, "shift": 5.0,
    \\ "shift_terminal": null, "stochastic_sampling": false, "time_shift_type": "exponential", "training_shift": 1.0,
    \\ "use_beta_sigmas": false, "use_dynamic_shifting": false, "use_exponential_sigmas": false, "use_karras_sigmas": false}
;

/// `<dir>/scheduler/scheduler_config.json` in `tmp`, and `tmp`'s real path for the driver's own joins.
fn fixture(a: std.mem.Allocator, tmp: *t.TmpDir) ![:0]const u8 {
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    try tmp.dir.createDirPath(t.io, "scheduler");
    try tmp.dir.writeFile(t.io, .{ .sub_path = "scheduler/scheduler_config.json", .data = scheduler_config });
    return dir;
}

/// The driver's exit code for `argv` on backend `B`, with the recorded forwards reset first.
fn exec(comptime B: type, arena: *std.heap.ArenaAllocator, argv: []const [*:0]const u8) !u8 {
    const a = arena.allocator();
    var environ: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    const init: std.process.Init = .{
        .minimal = .{ .environ = .empty, .args = .{ .vector = argv } },
        .arena = arena,
        .gpa = a,
        .io = t.io,
        .environ_map = &environ,
        .preopens = .empty,
    };
    seen = 0;
    decoded_len = 0;
    return driver.run(B, init);
}

/// The driver's exit code on `B` for a `side` x `side` picture of `steps` steps at `seed` and `guidance`.
fn picture(comptime B: type, a: std.mem.Allocator, arena: *std.heap.ArenaAllocator, dir: [:0]const u8, seed: [*:0]const u8, steps: [*:0]const u8, guidance: [*:0]const u8) !u8 {
    const transformer = try a.dupeSentinel(u8, try a.print("{s}/transformer", .{dir}), 0);
    const out = try a.dupeSentinel(u8, try a.print("{s}/out.png", .{dir}), 0);
    return exec(B, arena, &.{ "tf-moss-image-generate", transformer, "--prompt", "a fox", "--negative", "blurry", "--steps", steps, "--guidance", guidance, "--seed", seed, "--width", "32", "--height", "32", out });
}

test "the seed reaches the latents: one seed repeats its first step, another does not" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try fixture(a, &tmp);
    try t.expectEqual(@as(u8, 0), try picture(Backend, a, &arena, dir, "7", "3", "3.0"));
    const one = calls[0];
    try t.expectEqual(@as(u8, 0), try picture(Backend, a, &arena, dir, "7", "3", "3.0"));
    try t.expectEqual(one.first, calls[0].first);
    try t.expectEqual(one.digest, calls[0].digest);
    try t.expectEqual(@as(u8, 0), try picture(Backend, a, &arena, dir, "8", "3", "3.0"));
    try t.expect(one.digest != calls[0].digest);
    const written = try tmp.dir.readFileAlloc(t.io, "out.png", a, .limited(64));
    try t.expectEqual(png.len, written.len);
}

test "step s reads timesteps[s], both branches share its input, and only the negative one skips 22" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try fixture(a, &tmp);
    const settings = try tf.moss_image.schedule.parse(a, scheduler_config);
    const sched = try tf.moss_image.schedule.timesteps(a, settings, 3);
    const steps = sched.timesteps.len;
    try t.expectEqual(@as(u8, 0), try picture(Backend, a, &arena, dir, "42", "3", "3.0"));
    try t.expectEqual(2 * steps, seen);
    const start = try a.alloc(f32, 2 * 2 * Inert.config.channels);
    try tf.moss_image.noise.latents(a, 42, 2, 2, start);
    var x = start[0];
    const guidance: f32 = 3.0;
    for (0..steps) |s| {
        const cond = calls[2 * s];
        const uncond = calls[2 * s + 1];
        try t.expectEqual(sched.timesteps[s], cond.timestep);
        try t.expectEqual(sched.timesteps[s], uncond.timestep);
        try t.expect(cond.skip == null);
        try t.expectEqual(@as(?usize, 22), uncond.skip);
        try t.expectEqual(cond.digest, uncond.digest);
        // The two branches do answer differently on that one input, so a weighted sum of them is observable.
        try t.expect(cond.out_digest != uncond.out_digest);
        try t.expect(@abs(cond.first - x) < 1e-6);
        const by = sched.sigmas[s + 1] - sched.sigmas[s];
        const pc = 2 * x;
        const pu = x;
        x = x + (pu + guidance * (pc - pu)) * by;
    }
}

test "the guidance scale weights the two branches in the step it takes" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try fixture(a, &tmp);
    const settings = try tf.moss_image.schedule.parse(a, scheduler_config);
    const sched = try tf.moss_image.schedule.timesteps(a, settings, 1);
    try t.expectEqual(@as(u8, 0), try picture(Backend, a, &arena, dir, "42", "1", "2"));
    try t.expectEqual(@as(usize, 2), seen);
    try t.expectEqual(decoded.len, decoded_len);
    const start = try a.alloc(f32, decoded_len);
    try tf.moss_image.noise.latents(a, 42, 2, 2, start);
    const by = sched.sigmas[1] - sched.sigmas[0];
    try t.expect(by != 0);
    const guidance: f32 = 2;
    for (start, decoded[0..decoded_len]) |x, got| {
        const pc = 2 * x;
        const pu = x;
        try t.expect(@abs(got - (x + (pu + guidance * (pc - pu)) * by)) < 1e-6);
    }
    // A step that averaged the branches instead of weighting them would land elsewhere, so this is a control.
    const first = start[0];
    const weighted = first + (first + guidance * (2 * first - first)) * by;
    const averaged = first + ((2 * first + first) / 2) * by;
    try t.expect(@abs(weighted - averaged) > 1e-6);
}

test "a backend without the prompt encoder refuses with 2 before any forward" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try fixture(a, &tmp);
    try t.expectEqual(@as(u8, 2), try picture(NoEncoder, a, &arena, dir, "42", "3", "3.0"));
    try t.expectEqual(@as(usize, 0), seen);
}

test "a backend without the image decoder still runs the schedule and refuses with 2" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try fixture(a, &tmp);
    const settings = try tf.moss_image.schedule.parse(a, scheduler_config);
    const steps = (try tf.moss_image.schedule.timesteps(a, settings, 3)).timesteps.len;
    try t.expectEqual(@as(u8, 2), try picture(NoDecoder, a, &arena, dir, "42", "3", "3.0"));
    try t.expectEqual(2 * steps, seen);
}
