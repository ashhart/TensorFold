const std = @import("std");
const sam = @import("sam3");
const mtl = @import("metal");
const Prediction = sam.session.Prediction;

const Work = struct {
    a: std.mem.Allocator,
    io: std.Io,
    model: *sam.Model,
    images: [2]sam.Image,
    indices: [2]usize,
    frames: usize,
    seeds: []const sam.session.Seed,
    gate: ?*std.Io.Event = null,
    ready: std.Io.Event = .unset,
    failure: ?anyerror = null,
    outputs: [2]?Prediction = @splat(null),

    fn options(self: *Work) sam.session.Options {
        return .{ .max_objects = @max(17, self.seeds.len), .frame_count = self.frames, .direction = if (self.indices[0] < self.indices[1]) .forward else .reverse };
    }

    fn execute(self: *Work) !void {
        var session = try sam.Session.init(self.a, self.model, self.options());
        defer session.deinit();
        self.outputs[0] = try session.seed(self.a, self.images[0], self.indices[0], self.seeds);
        self.outputs[1] = try session.track(self.a, self.images[1], self.indices[1]);
    }

    fn run(self: *Work) void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        if (self.gate) |gate| {
            self.ready.set(self.io);
            gate.waitUncancelable(self.io);
        }
        self.execute() catch |err| {
            self.failure = err;
        };
    }

    fn deinit(self: *Work) void {
        for (&self.outputs) |*output| if (output.*) |*prediction| prediction.deinit();
    }
};

const Advance = struct {
    a: std.mem.Allocator,
    io: std.Io,
    session: *sam.Session,
    image: sam.Image,
    index: usize,
    gate: *std.Io.Event,
    ready: std.Io.Event = .unset,
    failure: ?anyerror = null,
    output: ?Prediction = null,

    fn run(self: *Advance) void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        self.ready.set(self.io);
        self.gate.waitUncancelable(self.io);
        self.output = self.session.track(self.a, self.image, self.index) catch |err| {
            self.failure = err;
            return;
        };
    }
};

fn pair(comptime T: type, io: std.Io, jobs: *[2]T, gate: *std.Io.Event) !void {
    const first = try std.Thread.spawn(.{}, T.run, .{&jobs[0]});
    const second = std.Thread.spawn(.{}, T.run, .{&jobs[1]}) catch |err| {
        gate.set(io);
        first.join();
        return err;
    };
    jobs[0].ready.waitUncancelable(io);
    jobs[1].ready.waitUncancelable(io);
    gate.set(io);
    first.join();
    second.join();
}

fn equal(phase: []const u8, want: Prediction, got: Prediction) !void {
    if (want.frame_index != got.frame_index or want.objects.len != got.objects.len) return error.SessionPredictionShape;
    for (want.objects, got.objects) |x, y| {
        if (x.id != y.id or !std.meta.eql(x.mask.size, y.mask.size) or x.mask.score != y.mask.score or
            !std.meta.eql(x.mask.box, y.mask.box) or !std.mem.eql(u8, x.mask.pixels, y.mask.pixels))
        {
            std.debug.print("SAM {s}: frame {d}, object {d}, scores {d}/{d}, first differing mask byte {?d}\n", .{ phase, want.frame_index, x.id, x.mask.score, y.mask.score, std.mem.indexOfDiff(u8, x.mask.pixels, y.mask.pixels) });
            return error.SessionPredictionChanged;
        }
    }
}

pub fn check(a: std.mem.Allocator, io: std.Io, model: *sam.Model, images: [2]sam.Image, indices: [2]usize, frames: usize, seeds: []const sam.session.Seed) !void {
    var solo = [2]Work{
        .{ .a = a, .io = io, .model = model, .images = images, .indices = indices, .frames = frames, .seeds = seeds },
        .{ .a = a, .io = io, .model = model, .images = .{ images[1], images[0] }, .indices = .{ indices[1], indices[0] }, .frames = frames, .seeds = seeds },
    };
    defer for (&solo) |*job| job.deinit();
    for (&solo) |*job| try job.execute();
    var shared = try sam.Session.init(a, model, solo[0].options());
    defer shared.deinit();
    var seeded = try shared.seed(a, images[0], indices[0], seeds);
    seeded.deinit();
    var duplicate_gate: std.Io.Event = .unset;
    var duplicates = [2]Advance{
        .{ .a = a, .io = io, .session = &shared, .image = images[1], .index = indices[1], .gate = &duplicate_gate },
        .{ .a = a, .io = io, .session = &shared, .image = images[1], .index = indices[1], .gate = &duplicate_gate },
    };
    defer for (&duplicates) |*job| if (job.output) |*prediction| prediction.deinit();
    try pair(Advance, io, &duplicates, &duplicate_gate);
    var accepted: usize = 0;
    for (duplicates) |job| {
        if (job.failure) |err| {
            if (err != error.NonConsecutiveFrame) return err;
        } else {
            accepted += 1;
        }
    }
    if (accepted != 1) return error.DuplicateFrameAccepted;
    for (duplicates) |job| if (job.output) |prediction| try equal("duplicate frame", solo[0].outputs[1].?, prediction);
    std.debug.print("SAM session: simultaneous duplicate frame admits exactly one call\n", .{});
    try shared.reset();
    seeded = try shared.seed(a, images[0], indices[0], seeds);
    defer seeded.deinit();
    try equal("reset seed", solo[0].outputs[0].?, seeded);
    var reset = try shared.track(a, images[1], indices[1]);
    defer reset.deinit();
    try equal("reset propagation", solo[0].outputs[1].?, reset);
    try shared.reset();
    var initial = try shared.seed(a, images[0], indices[0], seeds);
    defer initial.deinit();
    for (seeds, 0..) |seed, i| {
        if (i != 0 and i != 15 and i + 1 != seeds.len) continue;
        try shared.removeObject(seed.id);
        try shared.prompt(seed.id, seed.prompt);
    }
    var reused = try shared.track(a, images[1], indices[1]);
    defer reused.deinit();
    try equal("ID reuse", solo[0].outputs[1].?, reused);
    std.debug.print("SAM session: reset and reused object ID match fresh masks and scores\n", .{});
    var gate: std.Io.Event = .unset;
    var concurrent = [2]Work{
        .{ .a = a, .io = io, .model = model, .images = images, .indices = indices, .frames = frames, .seeds = seeds, .gate = &gate },
        .{ .a = a, .io = io, .model = model, .images = .{ images[1], images[0] }, .indices = .{ indices[1], indices[0] }, .frames = frames, .seeds = seeds, .gate = &gate },
    };
    defer for (&concurrent) |*job| job.deinit();
    try pair(Work, io, &concurrent, &gate);
    for (&concurrent, &solo) |*got, *want| {
        if (got.failure) |err| return err;
        for (0..2) |step| try equal("concurrent sessions", want.outputs[step].?, got.outputs[step].?);
    }
    std.debug.print("SAM sessions: concurrent forward/reverse runs on one model match solo byte for byte\n", .{});
}
