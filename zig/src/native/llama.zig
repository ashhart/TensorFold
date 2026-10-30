//! Llama on Metal served by the lane host: one-row rounds, windows of --drafter's or copied drafts, contexts budgeted.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const fl = tf.llama;
const lanes = tf.lanes;
const Resident = @import("resident.zig").Resident;

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *fl.Model,
    metal: *fl.backend.Metal,
    drafter: ?*fl.drafter.Drafter = null, // --drafter: a smaller Llama proposing each stream's next tokens
    warm: mtl.keepalive.Target = undefined, // the residency set's, or the bare queue's, for the lane host's idle ticker
    resident: ?Resident = null, // weights and scratch kept wired between requests, when they fit the 70% limit
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // kept prompt states, in what the streams' caches leave of the budget
    cache_cap: u64 = 0, // --prompt-cache-gib's bytes (the budget when unset)

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        h.config.deinit(h.gpa);
        fl.residency.drop(h);
        if (h.drafter) |d| d.deinit(); // its slots give their bytes back to the backend's ledger
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = fl.Model.load(gpa, io, o.dir) catch |err| {
        problem.* = switch (err) {
            error.LlamaNeedsTensorUnits => "the native Llama engine needs Metal 4 tensor units (an M5 or later) for its prompt GEMM and attention: this Mac's GPU (M1 to M4) has none",
            error.UnqualifiedLlamaCheckpoint => try std.fmt.allocPrint(a, "the native Llama engine serves only the checkpoints checked against Hugging Face ({s}): this llama checkpoint's geometry is not one of them", .{qualifiedNames(.served)}),
            else => try std.fmt.allocPrint(a, "the native Llama engine cannot load this checkpoint ({s})", .{@errorName(err)}),
        };
        return null;
    };
    errdefer model.deinit();
    const q = try model.config.qualification();
    if (q.role != .served) {
        problem.* = try std.fmt.allocPrint(a, "{s} is checked only as a drafter (--drafter): the native Llama engine serves {s}", .{ q.name, qualifiedNames(.served) });
        model.deinit();
        return null;
    }
    // refusals that return null free what they hold themselves: errdefer runs only for errors
    var drafter_model: ?*fl.Model = null;
    errdefer if (drafter_model) |dm| dm.deinit();
    if (drafterDir(o)) |dir| {
        // a smaller Llama on the model's tokens, its commands on the model's queue (one residency set for both)
        const dm = fl.Model.loadOn(gpa, io, dir, model.queue) catch |err| {
            problem.* = if (err == error.UnsupportedLlamaPrecision)
                try std.fmt.allocPrint(a, "--drafter: {s} is a quantized checkpoint; a Llama drafter is served from its bf16 release as published", .{dir})
            else
                try std.fmt.allocPrint(a, "--drafter: the native Llama engine cannot load {s} ({s}); it takes a smaller Llama checkpoint on the served model's tokenizer", .{ dir, @errorName(err) });
            model.deinit();
            return null;
        };
        drafter_model = dm; // before the refusal's message is built, so a failed allocation leaves it to the errdefer
        if (try drafterRefusal(a, dir, &model.config, &dm.config)) |why| {
            problem.* = why;
            dm.deinit();
            model.deinit();
            return null;
        }
    }
    var ram: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &ram, &len, null, 0) != 0) ram = 0;
    const text: ?[]const u8 = if (std.c.getenv("LLAMA_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    // the decode window's and the prompt chunk's buffers
    var scratch: u64 = fl.state.scratchBytes(&model.config, fl.state.batch_rows) + fl.state.scratchBytes(&model.config, fl.state.prompt_rows);
    var most: usize = @min(model.config.context, fl.state.max_capacity);
    if (drafter_model) |dm| {
        scratch += dm.checkpoint.residentBytes() + fl.drafter.Drafter.fixedBytes(&dm.config);
        most = @min(most, dm.config.context); // the drafter drafts within its own trained window
    }
    const budget = fl.budget.caches(ram, model.device.maxWorkingSet(), text, model.checkpoint.residentBytes() + scratch) catch {
        if (drafter_model) |dm| dm.deinit();
        model.deinit();
        problem.* = "LLAMA_LOAD_LIMIT_GB is not a load limit: give a number of GB above 0, or unset it for 75% of RAM";
        return null;
    };
    const streams: u32 = @max(o.lanes, 1);
    const drafted: ?fl.budget.Drafted = if (drafter_model) |dm| .{ .c = &dm.config, .streams = streams } else null;
    const fits = fl.budget.longestWith(&model.config, drafted, budget, most);
    const window: i64 = o.context orelse @intCast(fits);
    if (window < 1 or window > fits) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: this Mac's Llama caches fit {d} tokens under its load limit (75% of RAM, or LLAMA_LOAD_LIMIT_GB){s}", .{ window, fits, if (drafter_model != null) ", the drafter's caches and trained window counted" else "" });
        if (drafter_model) |dm| dm.deinit();
        model.deinit(); // after the message, so a failed allocation leaves the model to the errdefer
        return null;
    }
    std.log.info("llama: {d:.1} GiB for caches, context up to {d} tokens", .{ @as(f64, @floatFromInt(budget)) / (1 << 30), window });
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.io = io;
    h.model = model;
    h.resident = null;
    h.drafter = null;
    h.metal = try fl.backend.Metal.init(gpa, model, .{ .capacity = @intCast(window), .streams = streams, .budget = budget });
    errdefer h.metal.deinit();
    try h.metal.warm();
    if (drafter_model) |dm| {
        const capacity: usize = @as(usize, @intCast(window)) + fl.state.window_rows;
        drafter_model = null; // the drafter's now: freed with it, even when its warm-up fails
        h.drafter = try fl.drafter.Drafter.start(gpa, dm, .{ .capacity = capacity, .streams = streams, .ledger = &h.metal.ledger });
        h.metal.slots = h.drafter.?.slotsHook(); // its slots grow with their streams' admission, in the same budget
        const dc = &dm.config;
        std.log.info("llama drafter: {s}: {d} layers, bf16, {d:.2} GiB of weights, {d} caches of up to {d} tokens ({d:.2} GiB each), each grown with its stream in the same budget", .{ o.drafter.?, dc.layers, @as(f64, @floatFromInt(dm.checkpoint.residentBytes())) / (1 << 30), streams, capacity, @as(f64, @floatFromInt(fl.state.cacheBytes(dc, capacity))) / (1 << 30) });
    }
    errdefer if (h.drafter) |d| d.deinit();
    h.config = try lanes.Config.init(gpa, h.metal.facts(), fl.state.window_rows, fl.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = fl.state.prompt_rows, .structures = true, .logprobs = true });
    h.cache = null;
    h.cache_cap = if (o.prompt_cache_gib) |g| @intFromFloat(g * (1 << 30)) else budget;
    if (h.cache_cap > 0) {
        const M = fl.backend.Metal;
        // prompt chunks change arithmetic with their rows
        const rules: api.prompt_cache.Rules = .{ .lookahead = 0, .planned = true, .grid = fl.state.prompt_rows, .min_prompt = 0 };
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = h.metal, .vtable = &.{ .bytes = M.snapBytes, .save = M.snapSave, .restore = M.snapRestore, .drop = M.snapDrop } }, rules, @min(h.cache_cap, budget));
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
        h.metal.room = .{ .ctx = h, .fit = fitCache };
    }
    errdefer if (h.cache) |*store| store.deinit();
    try fl.residency.hold(h);
    errdefer fl.residency.drop(h);
    if (h.drafter) |d| h.host.proposers = .{ .ctx = d, .make = fl.drafter.Drafter.make, .free = fl.drafter.Drafter.free };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// The names of the checkpoints qualified for `role`, comma-separated.
fn qualifiedNames(comptime role: fl.config.Role) []const u8 {
    comptime var names: []const u8 = "";
    inline for (fl.config.qualified) |q| if (q.role == role) {
        names = names ++ (if (names.len == 0) "" else ", ") ++ q.name;
    };
    return names;
}

/// --drafter's directory when drafts are on; one width (bf16 as published), so --drafter-bits is unread, as in Gemma 4.
fn drafterDir(o: api.Open) ?[]const u8 {
    return if (o.drafts) o.drafter else null;
}

/// Why a drafter cannot serve: tokens that are not the model's.
fn drafterRefusal(a: std.mem.Allocator, dir: []const u8, target: *const fl.config.Config, drafter: *const fl.config.Config) !?[]const u8 {
    var same = drafter.vocab <= target.vocab and drafter.eos_count == target.eos_count;
    if (same) for (drafter.eos[0..drafter.eos_count]) |e| {
        if (!target.isEos(e)) same = false;
    };
    if (!same) return try std.fmt.allocPrint(a, "--drafter: {s} does not share the model's tokens (vocabulary {d} against {d}, or other end tokens); a drafter is a smaller Llama on the served model's tokenizer", .{ dir, drafter.vocab, target.vocab });
    return null;
}

/// Metal.room's fit: the kept states' budget is what the streams leave, at most the cap; the least valuable go first.
fn fitCache(ctx: *anyopaque, bytes: u64, restoring: ?*anyopaque, prompt: []const u32) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const store = if (h.cache) |*owned| owned else return;
    store.budget = @min(h.cache_cap, bytes);
    const Left = struct {
        s: *const api.prompt_cache.Store,
        pub fn fits(l: @This()) bool {
            return l.s.held <= l.s.budget;
        }
    };
    _ = store.giveBack(restoring, prompt, Left{ .s = store });
}

const CacheFixture = struct {
    live: usize = 0,
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 100 + at;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) !api.prompt_cache.Saved {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        const held = try std.testing.allocator.create(u32);
        held.* = at;
        self.live += 1;
        return held;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) !void {}
    fn drop(ptr: *anyopaque, saved: api.prompt_cache.Saved) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        std.testing.allocator.destroy(@as(*u32, @ptrCast(@alignCast(saved))));
        self.live -= 1;
    }
    fn snapshots(self: *@This()) api.prompt_cache.Snapshots {
        return .{ .ptr = self, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
};

test "llama host: the default options serve the drafter, whatever --drafter-bits holds" {
    const plain: api.Open = .{ .dir = "m", .model_type = "llama", .drafter = "d" };
    try std.testing.expectEqual(@as(u8, 4), plain.drafter_bits); // the shared default, Qwen's DFlash width
    try std.testing.expectEqualStrings("d", drafterDir(plain).?);
    var bf16 = plain;
    bf16.drafter_bits = 0;
    try std.testing.expectEqualStrings("d", drafterDir(bf16).?);
    var off = plain;
    off.drafts = false; // --no-drafts
    try std.testing.expectEqual(@as(?[]const u8, null), drafterDir(off));
}

test "llama host: cache callback updates the owned store rather than a temporary value" {
    var f: CacheFixture = .{};
    var h: Host = undefined;
    h.cache_cap = 500;
    h.cache = api.prompt_cache.Store.init(std.testing.allocator, f.snapshots(), .{ .min_prompt = 0 }, 500);
    defer h.cache.?.deinit();
    fitCache(&h, 200, null, &.{});
    try std.testing.expectEqual(@as(u64, 200), h.cache.?.budget);
    fitCache(&h, 800, null, &.{});
    try std.testing.expectEqual(@as(u64, 500), h.cache.?.budget);
}

test "llama host: cache callback evicts from the real store and preserves the restored entry" {
    var f: CacheFixture = .{};
    var h: Host = undefined;
    h.cache_cap = 1000;
    h.cache = api.prompt_cache.Store.init(std.testing.allocator, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer h.cache.?.deinit();
    const store = &h.cache.?;
    try std.testing.expect(store.keep(&.{ 1, 1, 1, 1, 9 }, 4, null, &.{}, &.{}));
    try std.testing.expect(store.keep(&.{ 2, 2, 2, 2, 2, 2, 9 }, 6, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(u64, 210), store.held);
    const kept = store.entries.items[0].saved;
    fitCache(&h, 110, kept, &.{ 1, 1, 1, 1, 9 });
    try std.testing.expectEqual(@as(u64, 110), store.budget);
    try std.testing.expectEqual(@as(u64, 104), store.held);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expect(store.entries.items[0].saved == kept);
    try std.testing.expectEqual(@as(usize, 1), f.live);
    try std.testing.expectEqual(@as(u64, 1), store.counts.evicted);
    fitCache(&h, 110, kept, &.{ 1, 1, 1, 1, 9 });
    try std.testing.expectEqual(@as(u64, 1), store.counts.evicted);
    fitCache(&h, 0, null, &.{});
    try std.testing.expectEqual(@as(u64, 0), store.held);
    try std.testing.expectEqual(@as(usize, 0), store.entries.items.len);
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(@as(u64, 2), store.counts.evicted);
}

test "llama host: cache callback with no configured cache leaves no state to mutate" {
    var h: Host = undefined;
    h.cache = null;
    fitCache(&h, 0, null, &.{});
    try std.testing.expect(h.cache == null);
}
