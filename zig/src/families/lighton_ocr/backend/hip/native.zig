//! LightOnOCR-3-1B native HIP admission, lossless weight preparation, shared lane adapter and owned prompt cache.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const lanes = @import("lanes");
const api = @import("engine_api");
const serve = @import("hip_serve");
const cfg = @import("../../config.zig");
const image = @import("../../image.zig");
const weights = @import("../../weights.zig");
const text = @import("text.zig");
const tower = @import("tower.zig");
const vision = @import("vision.zig");
const backend = @import("backend.zig");
const snapshot = @import("snapshot.zig");
const Allocator = std.mem.Allocator;

pub const model_type = "mistral3";
pub const also_model_types: []const []const u8 = &.{"lighton_ocr"};
pub const formats: []const []const u8 = &.{"bf16"};
pub const default_context: i64 = 8192;
/// The 1B template has no thinking switch and its replies carry no reasoning block: all text is answer content.
pub const reasoning = false;
pub const prefill_step: u32 = backend.prompt_rows;
pub const slide_refusal = "--slide: LightOnOCR does not learn on HIP";
pub const confidence_refusal = "--mtp-confidence: LightOnOCR has no MTP head; it verifies copies from the shared lane core";
pub const Vision = backend.Vision;
pub const Options = serve.Options;

/// One rank only: refused before a group forms (hip_families.refused).
pub fn refusal(o: api.Open) ?[]const u8 {
    return if (o.tp != 1) "LightOnOCR on HIP currently needs --tp 1" else null;
}

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    streams: usize,
    startup: []const u8,
    /// Its image tower under --vision and its prompt cache when the budget and the slots allow one.
    offer: serve.Offer,

    /// Development tools capture this exact native loader and graph, without a second weight or runtime path.
    pub fn capture(l: *const Loaded) !Capture {
        const own: *Owned = @ptrCast(@alignCast(l.ctx));
        return .{ .tower = if (own.tower) |*v| v else return error.VisionDisabled, .text = &own.text, .s = own.s, .c = own.layout.c, .r = &own.r };
    }
};

pub const Capture = struct { tower: *tower.Tower, text: *text.Text, s: hip.Stream, c: cfg.Config, r: *const hip.Runtime };

const Owned = struct {
    gpa: Allocator,
    r: hip.Runtime,
    ctx: hip.Context,
    s: hip.Stream,
    shared: hip.Launcher,
    layout: weights.Layout,
    device_weights: hip.DeviceBuffer,
    text: text.Text,
    tower: ?tower.Tower = null,
    vision: ?Vision = null,
    lanes: backend.Hip,
    cache: ?*api.prompt_cache.Store = null,
    startup: []u8,
};

/// Pure memory plan: raw BF16 offsets double into an FP32 arena, preserving every checkpoint value exactly.
pub const Plan = struct {
    weight_bytes: usize,
    fixed_bytes: usize,
    stream_bytes: usize,
    capacity: usize,

    pub fn init(c: cfg.Config, raw_bytes: usize, window: usize, limits: ?vision.Limits) !Plan {
        if (window == 0 or window > c.context) return error.PastContext;
        if (limits) |v| try v.check(window);
        const capacity = try std.math.add(usize, window, backend.max_window);
        var fixed = try std.math.add(usize, try text.scratchBytes(c, capacity), backend.fixedBytes(c));
        if (limits != null) fixed = try std.math.add(usize, fixed, try tower.scratchBytes(c));
        return .{
            .weight_bytes = try std.math.mul(usize, raw_bytes, 2),
            .fixed_bytes = fixed,
            .stream_bytes = try std.math.add(usize, try text.streamBytes(c, capacity), backend.slotBytes(c, limits)),
            .capacity = capacity,
        };
    }

    pub fn total(p: Plan, streams: usize) !usize {
        return std.math.add(usize, try std.math.add(usize, p.weight_bytes, p.fixed_bytes), try std.math.mul(usize, streams, p.stream_bytes));
    }

    pub fn fit(p: Plan, room: usize, asked: usize, fixed: bool) !usize {
        if (fixed and asked > backend.round_rows) return error.ParallelPastRows;
        const base = try std.math.add(usize, p.weight_bytes, p.fixed_bytes);
        if (room <= base or p.stream_bytes == 0) return error.NoStreamFits;
        const count = @min(@min(@max(asked, 1), backend.round_rows), (room - base) / p.stream_bytes);
        if (count == 0) return error.NoStreamFits;
        if (fixed and count < asked) return error.ParallelPastMemory;
        return count;
    }
};

/// The measured pool and full proposed allocation, retained even when admission refuses before allocation.
const Admission = struct {
    device: c_int,
    pool: hip.admission.Pool,
    held: u64,
    room: usize,
    plan: Plan,
    window: usize,
    streams: usize,
    fixed: bool,

    pub fn format(d: Admission, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("HIP device {d} admission: free={d} total={d} reserve={d} limit={?d} held={d} room={d} bytes; " ++
            "weights={d} scratch={d} per_stream={d} minimum={d} requested={d} bytes; " ++
            "window={d} physical={d} streams={d} fixed={} unified={}", .{
            d.device,            d.pool.free,        d.pool.total,        d.pool.reserve,                               d.pool.limit,                                                  d.held,   d.room,
            d.plan.weight_bytes, d.plan.fixed_bytes, d.plan.stream_bytes, d.plan.total(1) catch std.math.maxInt(usize), d.plan.total(@max(d.streams, 1)) catch std.math.maxInt(usize), d.window, d.plan.capacity,
            d.streams,           d.fixed,            d.pool.unified,
        });
    }
};

fn refuse(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) error{Refused} {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

/// The host supplies its common HIP options; an image offer is present only when --vision was requested.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: serve.Options, problem: *[]const u8) !Loaded {
    if (dev.caps.act != .bf16) return refuse(a, problem, "LightOnOCR keeps BF16 activations and needs a HIP GPU with native BF16 support", .{});
    var admission: ?Admission = null;
    return load(gpa, io, dev, dir, o, &admission) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        if (admission) |d| return refuse(a, problem, "cannot load native HIP LightOnOCR ({s}): {s}; {f}", .{
            @errorName(err), explain(err), d,
        });
        return refuse(a, problem, "cannot load native HIP LightOnOCR ({s}): {s}", .{ @errorName(err), explain(err) });
    };
}

fn load(gpa: Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: anytype, admission: *?Admission) !Loaded {
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.cache = null;
    own.tower = null;
    own.vision = null;
    const c = try cfg.Config.read(gpa, io, dir);
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    own.layout = try weights.Layout.init(gpa, &ck, c);
    errdefer own.layout.deinit();
    const limits: ?vision.Limits = if (@hasField(@TypeOf(o), "vision")) if (o.vision) |v| .{ .image_tokens = @intCast(@min(v.image_tokens, o.window)), .max_images = v.max_images } else null else null;
    if (limits != null) try image.readProcessor(gpa, io, dir, c);
    const plan = try Plan.init(c, own.layout.bytes, o.window, limits);
    own.r = try hip.Runtime.open();
    errdefer own.r.close();
    own.ctx = try hip.Context.init(&own.r, dev.index);
    errdefer own.ctx.deinit();
    _ = try hip.Device.kernelCaps(&own.r, dev.index);
    const pool = try hip.admission.budget(gpa, io, own.ctx);
    const held = hip.usage(false).device;
    const room: usize = @intCast(pool.room(held));
    admission.* = .{
        .device = dev.index,
        .pool = pool,
        .held = held,
        .room = room,
        .plan = plan,
        .window = o.window,
        .streams = o.streams,
        .fixed = o.fixed,
    };
    std.debug.print("[tensorfold] {f}\n", .{admission.*.?});
    const streams = try plan.fit(room, o.streams, o.fixed);
    own.s = try hip.Stream.init(&own.r);
    errdefer own.s.deinit();
    own.shared = try hip.Launcher.load(&own.r, dev.policy.choices(), hip.kernels.images);
    errdefer own.shared.unload();
    own.device_weights = try hip.DeviceBuffer.alloc(&own.r, plan.weight_bytes);
    errdefer own.device_weights.free();
    try uploadWeights(&own.r, own.s, own.device_weights, own.layout.entries.items);
    own.layout.entries.deinit(gpa);
    own.layout.entries = .empty;
    own.text = try text.Text.init(gpa, &own.r, own.s, &own.layout, own.device_weights.base(), plan.capacity, streams, &own.shared);
    errdefer own.text.deinit();
    if (limits) |v| {
        own.tower = try tower.Tower.init(gpa, &own.r, own.s, &own.layout, own.device_weights.base(), &own.shared);
        own.vision = .{ .io = io, .c = c, .limits = v, .tower = &own.tower.? };
    }
    errdefer if (own.tower) |*v| v.deinit();
    own.lanes = try backend.Hip.init(gpa, &own.ctx, &own.text, &own.shared, if (own.vision) |*v| v else null, o.window);
    errdefer own.lanes.deinit();
    const cache_budget = try cacheBytes(o.cache_gib, dev.policy.slots);
    if (cache_budget > 0) {
        const cache = try gpa.create(api.prompt_cache.Store);
        cache.* = api.prompt_cache.Store.init(gpa, own.lanes.snapshots(), .{ .lookahead = 0, .min_prompt = 0, .chunk = backend.prompt_rows }, cache_budget);
        cache.most = dev.policy.slots;
        own.cache = cache;
    }
    errdefer if (own.cache) |cache| {
        cache.deinit();
        gpa.destroy(cache);
    };
    own.startup = try std.fmt.allocPrint(gpa, "HIP {s} device {d}: LightOnOCR BF16 rows, lossless FP32 weights {d:.2} GiB; {d} streams at {d} tokens, {d:.2} GiB each; images {s}; host prompt cache {d:.2} GiB", .{ hip.kernels.arch, dev.index, gib(plan.weight_bytes), streams, o.window, gib(plan.stream_bytes), if (limits != null) "on" else "off", gib(cache_budget) });
    return .{ .backend = own.lanes.backend(), .facts = own.lanes.facts(), .rows = backend.max_window, .ctx = own, .deinit = release, .streams = streams, .startup = own.startup, .offer = .{
        .vision = serve.vision.offer(if (own.vision) |*v| v else null),
        .cache = own.cache,
    } };
}

/// One bounded pinned stage widens every BF16 word losslessly; source mappings are released only after completion.
fn uploadWeights(r: *const hip.Runtime, s: hip.Stream, target: hip.DeviceBuffer, entries: []const weights.Entry) !void {
    var stage = try hip.HostBuffer.alloc(r, 4 << 20);
    defer stage.free();
    errdefer snapshot.drain(s);
    for (entries) |entry| {
        const words = std.mem.bytesAsSlice(u16, entry.bytes);
        var from: usize = 0;
        while (from < words.len) {
            const count = @min(words.len - from, stage.bytes.len / 4);
            const out = stage.slice(u32)[0..count];
            for (out, words[from..][0..count]) |*dst, word| dst.* = @as(u32, word) << 16;
            const offset = try std.math.add(usize, try std.math.mul(usize, entry.offset, 2), from * 4);
            if (offset > target.len or count * 4 > target.len - offset) return error.UnexpectedTensor;
            try hip.runtime.check(r.api.hipMemcpyHtoDAsync(@ptrFromInt(target.base() + offset), stage.bytes.ptr, count * 4, s.handle));
            try s.synchronize();
            from += count;
        }
    }
}

fn release(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    own.ctx.makeCurrent() catch {
        std.log.err("cannot select the LightOnOCR HIP context before resource release", .{});
        std.c._exit(1);
    };
    snapshot.drain(own.s);
    if (own.cache) |cache| {
        cache.deinit();
        own.gpa.destroy(cache);
    }
    own.lanes.deinit();
    if (own.tower) |*v| v.deinit();
    own.text.deinit();
    own.device_weights.free();
    own.layout.deinit();
    own.shared.unload();
    own.s.deinit();
    own.ctx.deinit();
    own.r.close();
    own.gpa.free(own.startup);
    own.gpa.destroy(own);
}

pub fn follow(a: Allocator, _: Allocator, _: std.Io, _: hip.Device, _: []const u8, _: anytype, problem: *[]const u8) !void {
    return refuse(a, problem, "LightOnOCR on HIP currently needs --tp 1", .{});
}

fn cacheBytes(requested: ?f64, slots: u32) !usize {
    const value = requested orelse 2.0;
    if (!std.math.isFinite(value) or value < 0 or value * (1 << 30) >= @as(f64, @floatFromInt(std.math.maxInt(usize))))
        return error.BadCacheBudget;
    if (slots == 0) return 0;
    return @intFromFloat(value * (1 << 30));
}

fn gib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn explain(err: anyerror) []const u8 {
    return switch (err) {
        error.NoStreamFits => "the HIP memory budget cannot fit the model, scratch and one stream",
        error.ParallelPastMemory => "the requested parallel count exceeds the HIP memory budget",
        error.ParallelPastRows => "the requested parallel count exceeds the shared forward's row capacity",
        error.PastContext => "the requested context exceeds this checkpoint's window",
        error.BadImageLimits => "image limits must fit the context and allow at least one image and token",
        error.UnsupportedLightOnConfig, error.UnsupportedLightOnGeometry => "this Mistral3 checkpoint is not the implemented LightOnOCR graph",
        error.BadLightOnProcessor, error.UnsupportedLightOnProcessor => "the image processor does not match the implemented checkpoint processor",
        else => "check the checkpoint, embedded HIP kernels, runtime and requested memory limits",
    };
}

test "LightOnOCR HIP stream admission never silently lowers a fixed parallel count" {
    const plan: Plan = .{ .weight_bytes = 100, .fixed_bytes = 20, .stream_bytes = 40, .capacity = 100 };
    try std.testing.expectEqual(@as(usize, 2), try plan.fit(200, 8, false));
    try std.testing.expectEqual(@as(usize, 200), try plan.total(2));
    try std.testing.expectError(error.ParallelPastMemory, plan.fit(200, 8, true));
    try std.testing.expectError(error.NoStreamFits, plan.fit(159, 1, false));
    try std.testing.expectError(error.ParallelPastRows, plan.fit(10000, backend.round_rows + 1, true));
}

test "LightOnOCR HIP cache limits reject nonfinite sizes" {
    try std.testing.expectEqual(@as(usize, 0), try cacheBytes(0, 8));
    try std.testing.expectEqual(@as(usize, 0), try cacheBytes(2, 0));
    try std.testing.expectError(error.BadCacheBudget, cacheBytes(std.math.nan(f64), 8));
    try std.testing.expectError(error.BadCacheBudget, cacheBytes(-1, 8));
}

test "LightOnOCR HIP memory plan includes widened weights and physical verify slack" {
    const c = try cfg.Config.parse(std.testing.allocator, cfg.sample);
    const plan = try Plan.init(c, 2011295744, 8192, .{ .image_tokens = 4096, .max_images = 4 });
    try std.testing.expectEqual(@as(usize, 4022591488), plan.weight_bytes);
    try std.testing.expectEqual(@as(usize, 8208), plan.capacity);
    try std.testing.expectEqual(@as(usize, 28 * 2 * 8208 * 1024 * 2 + 151936 * 2 + 4096 * 1024 * 2), plan.stream_bytes);
    const room = try plan.total(4);
    try std.testing.expectEqual(@as(usize, 4), try plan.fit(room, 8, false));
    try std.testing.expectEqual(@as(usize, 3), try plan.fit(room - 1, 8, false));
    try std.testing.expectError(error.ParallelPastMemory, plan.fit(room - 1, 4, true));
    try std.testing.expectError(error.PastContext, Plan.init(c, 2011295744, 16385, null));
}

test "LightOnOCR HIP admission explains measured pool and complete requested allocation" {
    const d: Admission = .{
        .device = 0,
        .pool = .{ .free = 180, .total = 1000, .reserve = 20, .limit = 150, .unified = false },
        .held = 10,
        .room = 140,
        .plan = .{ .weight_bytes = 100, .fixed_bytes = 20, .stream_bytes = 40, .capacity = 116 },
        .window = 100,
        .streams = 2,
        .fixed = true,
    };
    const message = try std.fmt.allocPrint(std.testing.allocator, "{f}", .{d});
    defer std.testing.allocator.free(message);
    try std.testing.expectEqualStrings("HIP device 0 admission: free=180 total=1000 reserve=20 limit=150 held=10 " ++
        "room=140 bytes; weights=100 scratch=20 per_stream=40 minimum=160 requested=200 bytes; " ++
        "window=100 physical=116 streams=2 fixed=true unified=false", message);
    try std.testing.expectError(error.NoStreamFits, d.plan.fit(d.room, d.streams, d.fixed));
}

test "LightOnOCR HIP receipt server plan fits an idle 30 GiB card" {
    const c = try cfg.Config.parse(std.testing.allocator, cfg.sample);
    const plan = try Plan.init(c, 2011295744, 4096, .{ .image_tokens = 4096, .max_images = 4 });
    try std.testing.expect(try plan.total(2) < 6 << 30);
    try std.testing.expectEqual(@as(usize, 2), try plan.fit(29 << 30, 2, true));
}

test "LightOnOCR HIP native graph entry compiles without opening a device" {
    const O = Options;
    const Entry = struct {
        fn run(a: Allocator, gpa: Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: O, problem: *[]const u8) anyerror!Loaded {
            return open(a, gpa, io, dev, dir, o, problem);
        }
    };
    const entry: *const fn (Allocator, Allocator, std.Io, hip.Device, []const u8, O, *[]const u8) anyerror!Loaded = Entry.run;
    std.mem.doNotOptimizeAway(entry);
}
