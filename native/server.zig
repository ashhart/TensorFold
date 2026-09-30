const std = @import("std");
const mx = @import("mlx.zig");
const inference = @import("session.zig");
const chat = @import("chat.zig");
const reply_text = @import("reply_text.zig");
const tool_calls = @import("tool_calls.zig");
const Request = std.http.Server.Request;
const control = @import("server_control.zig");
const memory_policy = @import("memory_budget.zig");
const memory_runtime = @import("memory_runtime.zig");
const PrefixStore = @import("prompt_cache.zig").Store(inference.Snapshot);
const live_status = @import("server_live.zig");

pub fn run(init: std.process.Init, args: []const []const u8) !void {
    var host: []const u8 = "127.0.0.1";
    var port: u16 = 8080;
    var limit: usize = 0;
    var timeout_ms: i64 = 0;
    var shutdown_grace_ms: i64 = 5000;
    var batch_streams: usize = 4;
    var checkpoint_slots: ?usize = null;
    var prompt_cache_bytes: ?u64 = null;
    var name = std.fs.path.basename(args[2]);
    var defaults = try inference.Options.load(init.gpa, init.io, args[2]);
    defaults.max_tokens = 4096;
    var thinking = true;
    var vision_urls = false;
    var drafts = true;
    var draft_options = @import("neural_draft.zig").Options{ .enabled = true };
    var effort: []const u8 = "medium";
    var overrides = std.json.Value{ .object = .empty };
    defer overrides.object.deinit(init.gpa);
    var i: usize = 3;
    while (i < args.len) {
        if (std.mem.eql(u8, args[i], "--no-drafts")) {
            drafts = false;
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--vision-urls") or std.mem.eql(u8, args[i], "--no-vision-urls")) {
            vision_urls = std.mem.eql(u8, args[i], "--vision-urls");
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--thinking") or std.mem.eql(u8, args[i], "--no-thinking")) {
            thinking = std.mem.eql(u8, args[i], "--thinking");
            i += 1;
            continue;
        }
        if (i + 1 == args.len) return error.MissingArgument;
        const flag = args[i];
        const value = args[i + 1];
        i += 2;
        if (std.mem.eql(u8, flag, "--drafter")) {
            draft_options.directory = value;
            continue;
        }
        if (std.mem.eql(u8, flag, "--drafter-bits")) {
            draft_options.bits = try std.fmt.parseInt(i32, value, 10);
            continue;
        }
        if (std.mem.eql(u8, flag, "--max-draft") or std.mem.eql(u8, flag, "--mtp-drafts")) {
            draft_options.max_draft = try std.fmt.parseInt(usize, value, 10);
            continue;
        }
        if (std.mem.eql(u8, flag, "--draft-calibration")) {
            draft_options.calibration = value;
            continue;
        }
        if (std.mem.eql(u8, flag, "--batch-streams")) {
            batch_streams = try std.fmt.parseInt(usize, value, 10);
            if (batch_streams < 1 or batch_streams > 8) return error.InvalidBatchStreams;
            continue;
        }
        if (std.mem.eql(u8, flag, "--checkpoint-slots")) {
            checkpoint_slots = try std.fmt.parseInt(usize, value, 10);
            if (checkpoint_slots.? == 0) return error.InvalidPromptCacheBudget;
            continue;
        }
        if (std.mem.eql(u8, flag, "--prompt-cache-gib")) {
            const size = try std.fmt.parseFloat(f64, value);
            if (!std.math.isFinite(size) or size < 0 or size >= 17179869184) return error.InvalidPromptCacheBudget;
            prompt_cache_bytes = @intFromFloat(size * memory_policy.gib);
            continue;
        }
        if (std.mem.eql(u8, flag, "--request-timeout-seconds")) timeout_ms = try control.seconds(value) else if (std.mem.eql(u8, flag, "--shutdown-grace-seconds")) shutdown_grace_ms = try control.seconds(value) else if (std.mem.eql(u8, flag, "--host")) host = value else if (std.mem.eql(u8, flag, "--port")) port = try std.fmt.parseInt(u16, value, 10) else if (std.mem.eql(u8, flag, "--served-model-name")) name = value else if (std.mem.eql(u8, flag, "--max-requests")) limit = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, flag, "--reasoning-effort")) {
            if (!std.mem.eql(u8, value, "low") and !std.mem.eql(u8, value, "medium") and !std.mem.eql(u8, value, "xhigh")) return error.InvalidReasoningEffort;
            effort = value;
        } else {
            const fields = .{ .{ "--temperature", "temperature" }, .{ "--top-k", "top_k" }, .{ "--top-p", "top_p" }, .{ "--max-tokens", "max_tokens" }, .{ "--thinking-budget", "thinking_budget" } };
            var found = false;
            inline for (fields) |pair| if (std.mem.eql(u8, flag, pair[0])) {
                try overrides.object.put(init.gpa, pair[1], .{ .string = value });
                found = true;
            };
            if (!found) return error.UnknownArgument;
        }
    }
    defaults = try inference.Options.parseWithDefaults(init.gpa, overrides, defaults);
    draft_options.enabled = drafts and draft_options.max_draft > 0;
    try draft_options.validate();
    var signals = control.Signals.install();
    defer signals.deinit();
    var registry = control.Registry{ .io = init.io, .timeout_ms = timeout_ms, .shutdown_grace_ms = shutdown_grace_ms };
    const monitor = try std.Thread.spawn(.{}, control.Registry.watch, .{&registry});
    defer {
        registry.finished.store(true, .release);
        monitor.join();
    }
    const address = try std.Io.net.IpAddress.parse(host, port);
    var listener = try address.listen(init.io, .{ .kernel_backlog = 128 });
    defer listener.deinit(init.io);
    const path = try std.fmt.allocPrint(init.gpa, "{s}/config.json", .{args[2]});
    defer init.gpa.free(path);
    const config = try @import("weights.zig").readFile(init.io, path);
    defer mx.allocator.free(config);
    const parsed = try std.json.parseFromSlice(std.json.Value, init.gpa, config, .{});
    defer parsed.deinit();
    const model_type = if (parsed.value == .object) parsed.value.object.get("model_type") orelse std.json.Value.null else std.json.Value.null;
    const is_glm = model_type == .string and std.mem.eql(u8, model_type.string, "glm5_next");
    const is_flash = model_type == .string and std.mem.eql(u8, model_type.string, "qwen4_exp");
    if (is_glm) try @import("glm.zig").Model.prepareRuntime();
    var jobs: [8]*Job = undefined;
    var stats = live_status.Stats{ .io = init.io, .allocator = init.gpa };
    defer stats.deinit();
    var display = live_status.Display{ .io = init.io, .stats = &stats };
    var worker = Worker{ .io = init.io, .dir = args[2], .queue = .init(&jobs), .defaults = defaults, .thinking = thinking, .effort = effort, .vision_urls = vision_urls, .control = &registry, .batch_streams = batch_streams, .is_glm = is_glm, .is_flash = is_flash, .memory_limit = init.environ_map.get("TENSORFOLD_MEMORY_LIMIT_GB"), .stats = &stats, .display = &display };
    worker.checkpoint_slots = checkpoint_slots orelse @max(8, 3 * batch_streams);
    worker.prompt_cache_bytes = prompt_cache_bytes;
    worker.drafts = drafts;
    worker.draft_options = draft_options;
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    defer {
        worker.queue.close(init.io);
        thread.join();
    }
    worker.ready.waitUncancelable(init.io);
    if (worker.startup_error) |err| return err;
    if (registry.stopping.load(.acquire)) return;
    display.start(init.environ_map.get("TENSORFOLD_NO_LIVE"), init.environ_map.get("COLUMNS"));
    defer display.stop();
    var clients: std.Io.Group = .init;
    defer clients.await(init.io) catch clients.cancel(init.io);
    errdefer registry.stop();
    display.print("Native inference listening at http://{s}:{d} ({s})\n", .{ host, listener.socket.address.getPort(), name });
    const Event = union(enum) { accepted: anyerror!void, stopped: anyerror!void };
    var events: [2]Event = undefined;
    var select = std.Io.Select(Event).init(init.io, &events);
    defer select.cancelDiscard();
    try select.concurrent(.accepted, acceptRequests, .{ &worker, &listener, &clients, init.gpa, name, limit });
    try select.concurrent(.stopped, control.Registry.waitStopped, .{&registry});
    switch (try select.await()) {
        inline else => |result| try result,
    }
}

fn acceptRequests(worker: *Worker, listener: *std.Io.net.Server, clients: *std.Io.Group, a: std.mem.Allocator, name: []const u8, limit: usize) anyerror!void {
    var handled: usize = 0;
    while (limit == 0 or handled < limit) : (handled += 1) {
        const stream = try listener.accept(worker.io);
        const client = worker.control.acquire(stream.socket.handle) orelse {
            stream.close(worker.io);
            if (worker.control.stopping.load(.acquire)) return;
            continue;
        };
        clients.concurrent(worker.io, connectionTask, .{ worker, a, stream, name, handled, client }) catch |err| {
            worker.control.release(client);
            stream.close(worker.io);
            return err;
        };
    }
}

const Job = struct {
    stats: *live_status.Stats,
    activated: bool = false,
    a: std.mem.Allocator,
    request: *Request,
    model: []const u8,
    sequence: usize,
    created: i64,
    client: *control.Client,
    body: std.json.Value,
    is_chat: bool,
    options: inference.Options,
    done: std.Io.Event = .unset,
    failure: ?anyerror = null,
};
const Worker = struct {
    draft_options: @import("neural_draft.zig").Options = .{},
    drafts: bool = true,
    stats: *live_status.Stats,
    display: *live_status.Display,
    io: std.Io,
    dir: []const u8,
    queue: std.Io.Queue(*Job),
    ready: std.Io.Event = .unset,
    startup_error: ?anyerror = null,
    control: *control.Registry,
    defaults: inference.Options,
    thinking: bool,
    effort: []const u8,
    vision_urls: bool,
    batch_streams: usize,
    is_glm: bool,
    is_flash: bool,
    memory_limit: ?[]const u8,
    memory_stats: MemoryStats = .{},
    checkpoint_slots: usize = 8,
    prompt_cache_bytes: ?u64 = null,
    prefix_stats: PrefixStats = .{},
    fn run(w: *Worker) void {
        w.loop() catch |err| {
            w.startup_error = err;
            w.ready.set(w.io);
        };
    }
    fn loop(w: *Worker) !void {
        try mx.init();
        defer mx.shutdown();
        var memory = try memory_runtime.Runtime.init(w.memory_limit, w.is_glm);
        defer memory.deinit();
        try memory.checkWeightsAndDraft(w.io, w.dir, w.is_flash, if (w.draft_options.enabled) w.draft_options.directory else null);
        var session = try inference.Session.initWithDraft(w.io, w.dir, w.draft_options);
        defer session.deinit();
        const profile = try memory_runtime.measure(&session);
        try memory.wire();
        var admission = memory_policy.Admission{ .budget = try memory.admissionBudget(w.io), .memory = profile };
        var gate = memory_policy.StreamGate{ .budget = admission.budget, .per_token = profile.per_token, .work = profile.round_bytes };
        const prefix_budget = w.prompt_cache_bytes orelse @min(memory.ram / 8, 16 * memory_policy.gib);
        var prefixes: ?PrefixStore = if (prefix_budget == 0) null else try PrefixStore.init(mx.allocator, w.checkpoint_slots, prefix_budget);
        defer if (prefixes) |*store| store.deinit();
        if (prefixes) |*store| store.admit_oversize = true;
        w.prefix_stats.update(if (prefixes) |*store| store else null);
        w.memory_stats.budget = memory.budget;
        w.memory_stats.mlx_budget = memory.share;
        w.memory_stats.admission_budget = admission.budget;
        w.memory_stats.update(0);
        std.debug.print("Native memory admission: {d} bytes available, {d} resident, {d} bytes per token; stream={d}, decode work={d}, prefill work={d}\n", .{ admission.budget, try memory_runtime.activeBytes(), profile.per_token, try profile.streamBytes(64), profile.round_bytes, try profile.prefillBytes(profile.chunk) });
        w.ready.set(w.io);
        var active: [8]?*Pending = @splat(null);
        var live: usize = 0;
        var closed = false;
        var activation_order: u64 = 0;
        while (!closed or live > 0) {
            if (!closed and live < w.batch_streams) {
                var ready: [8]*Job = undefined;
                const count = w.queue.getUncancelable(w.io, ready[0 .. w.batch_streams - live], if (live == 0) 1 else 0) catch blk: {
                    closed = true;
                    break :blk 0;
                };
                for (ready[0..count]) |job| {
                    const pending = Pending.start(w, &session, job) catch |err| blk: {
                        job.failure = err;
                        break :blk null;
                    };
                    if (pending) |p| {
                        for (&active) |*slot| if (slot.* == null) {
                            slot.* = p;
                            live += 1;
                            break;
                        };
                    } else job.done.set(w.io);
                }
            }
            try gateRound(&gate, &active, if (prefixes) |*store| store else null);
            for (&active) |*slot| if (slot.*) |pending| if (pending.growth == .ended) {
                pending.reportError(error.StreamGrowthExceedsMemoryBudget) catch |err| {
                    pending.job.failure = err;
                };
                const job = pending.job;
                pending.deinit();
                slot.* = null;
                live -= 1;
                job.done.set(w.io);
                try mx.check(mx.c.mlx_clear_cache());
            };
            for (&active) |*slot| if (slot.*) |pending| {
                const was_active = pending.generation != null;
                const done = pending.advance(&session, &admission, &active, if (prefixes) |*store| store else null) catch |err| blk: {
                    pending.reportError(err) catch |write_err| {
                        pending.job.failure = write_err;
                    };
                    break :blk true;
                };
                if (!was_active and pending.generation != null) {
                    pending.activation_order = activation_order;
                    activation_order += 1;
                }
                w.prefix_stats.update(if (prefixes) |*store| store else null);
                if (done) {
                    const job = pending.job;
                    pending.deinit();
                    slot.* = null;
                    live -= 1;
                    job.done.set(w.io);
                }
            };
            var waiting: u64 = 0;
            for (active) |slot| if (slot) |pending| {
                if (pending.generation == null or pending.growth == .paused) waiting += 1;
            };
            w.memory_stats.growth_waits.store(gate.waits, .release);
            w.memory_stats.growth_ends.store(gate.ends, .release);
            w.memory_stats.update(waiting);
        }
    }
};

fn gateRound(gate: *memory_policy.StreamGate, active: []const ?*Pending, prefixes: ?*PrefixStore) !void {
    var oldest: [8]*Pending = undefined;
    var count: usize = 0;
    for (active) |slot| if (slot) |pending| {
        pending.growth = .run;
        if (pending.generation) |*generation| if (generation.isDecoding()) {
            oldest[count] = pending;
            count += 1;
        };
    };
    std.mem.sort(*Pending, oldest[0..count], {}, struct {
        fn less(_: void, lhs: *Pending, rhs: *Pending) bool {
            return lhs.activation_order < rhs.activation_order;
        }
    }.less);
    var streams: [8]memory_policy.Live = undefined;
    for (oldest[0..count], streams[0..count]) |pending, *stream| {
        stream.* = pending.generation.?.memoryLengths();
        stream.copy_bytes = pending.prefix_reserve;
    }
    const plan = try gate.plan(memory_runtime.Reclaim{ .prefixes = prefixes }, streams[0..count]);
    for (oldest[plan.run..count]) |pending| pending.growth = .paused;
    if (plan.ended) |index| oldest[index].growth = .ended;
}

pub fn checkGrowth(session: *inference.Session, profile: memory_policy.StreamMemory) !void {
    const tokens = [_]i32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    for ([_]f32{ 0, 0.7 }) |temperature| {
        const options = inference.Options{ .max_tokens = 12, .ignore_eos = true, .sampling = .{ .temperature = temperature, .top_k = 0, .top_p = 1, .metal = true }, .seed = 71 };
        var isolated = try inference.RequestGeneration.init(session, mx.allocator, &tokens, options, .{}, null);
        defer isolated.deinit();
        while (!try isolated.step(session)) {}
        var expected = try isolated.takeReply();
        defer expected.deinit(mx.allocator);

        var older = Pending{ .job = undefined, .activation_order = 2 };
        defer if (older.generation) |*g| g.deinit();
        var newer = Pending{ .job = undefined, .activation_order = 9 };
        defer if (newer.generation) |*g| g.deinit();
        for ([_]*Pending{ &older, &newer }) |pending| {
            pending.generation = try inference.RequestGeneration.init(session, mx.allocator, &tokens, options, .{}, null);
            while (!pending.generation.?.isDecoding()) _ = try pending.generation.?.step(session);
        }
        // A reused worker slot can contain a younger stream than a later slot.
        const active = [_]?*Pending{ &newer, null, &older };
        var gate = memory_policy.StreamGate{ .budget = 0, .per_token = memory_policy.gib, .work = 0, .horizon = 1 };
        const memory = memory_runtime.Reclaim{ .prefixes = null };
        try mx.check(mx.c.mlx_clear_cache());
        gate.budget = try memory.used() + memory_policy.gib;
        const before = newer.generation.?.progress();
        try gateRound(&gate, &active, null);
        try std.testing.expect(older.growth == .run and newer.growth == .paused);
        var registry = control.Registry{ .io = session.io };
        var client = control.Client{ .owner = &registry };
        var job: Job = undefined;
        job.client = &client;
        newer.job = &job;
        var admission = memory_policy.Admission{ .budget = gate.budget, .memory = .{ .short_tokens = 0, .short = 0, .long_tokens = 1, .long = 0, .per_token = 1, .prefill_a = 0, .prefill_b = 0, .round_bytes = 0 } };
        try std.testing.expectError(error.RequestCancelled, newer.advance(session, &admission, &active, null));
        client.deadline = 0;
        try std.testing.expectError(error.RequestTimedOut, newer.advance(session, &admission, &active, null));
        client = .{ .owner = &registry };
        registry.stop();
        try std.testing.expectError(error.ServerStopping, newer.advance(session, &admission, &active, null));
        _ = try older.generation.?.step(session);
        try std.testing.expectEqualDeep(before, newer.generation.?.progress());

        gate.budget = std.math.maxInt(u64);
        try gateRound(&gate, &active, null);
        try std.testing.expect(older.growth == .run and newer.growth == .run);
        try mx.check(mx.c.mlx_clear_cache());
        gate.per_token = 0;
        gate.budget = try memory.used() + memory_policy.gib;
        newer.prefix_reserve = 2 * memory_policy.gib;
        try gateRound(&gate, &active, null);
        try std.testing.expect(older.growth == .run and newer.growth == .paused);
        newer.prefix_reserve = 0;
        try gateRound(&gate, &active, null);
        try std.testing.expect(older.growth == .run and newer.growth == .run);
        for ([_]*Pending{ &older, &newer }) |pending| {
            while (!try pending.generation.?.step(session)) {}
            var actual = try pending.generation.?.takeReply();
            defer actual.deinit(mx.allocator);
            try std.testing.expectEqualSlices(u32, expected.tokens.items, actual.tokens.items);
            try std.testing.expectEqualStrings(expected.content, actual.content);
        }
        try std.testing.expectEqual(@as(u64, 2), gate.waits);
        try std.testing.expectEqual(@as(u64, 0), gate.ends);
    }

    var older = Pending{ .job = undefined, .activation_order = 3 };
    defer if (older.generation) |*g| g.deinit();
    var newer = Pending{ .job = undefined, .activation_order = 4 };
    defer if (newer.generation) |*g| g.deinit();
    for ([_]*Pending{ &older, &newer }) |pending| {
        pending.generation = try inference.RequestGeneration.init(session, mx.allocator, &tokens, .{ .max_tokens = 12, .ignore_eos = true }, .{}, null);
        while (!pending.generation.?.isDecoding()) _ = try pending.generation.?.step(session);
    }
    var gate = memory_policy.StreamGate{ .budget = 0, .per_token = 1, .work = 0 };
    try gateRound(&gate, &.{ &newer, &older }, null);
    try std.testing.expect(older.growth == .run and newer.growth == .ended);
    newer.generation.?.deinit();
    newer.generation = null;
    try mx.check(mx.c.mlx_clear_cache());
    gate.budget = std.math.maxInt(u64);
    try gateRound(&gate, &.{&older}, null);
    while (!try older.generation.?.step(session)) {}
    try std.testing.expectEqual(@as(u64, 1), gate.ends);
    const prompt = try mx.allocator.alloc(i32, @intCast(profile.chunk + 1));
    defer mx.allocator.free(prompt);
    for (prompt, 0..) |*token, i| token.* = @intCast(10 + i % 93);
    var filling = Pending{ .job = undefined, .ids = prompt };
    filling.generation = try inference.RequestGeneration.init(session, mx.allocator, prompt, .{ .max_tokens = 4, .ignore_eos = true }, .{}, null);
    defer filling.generation.?.deinit();
    _ = try filling.generation.?.step(session);
    try std.testing.expect(!filling.generation.?.isDecoding());
    const before = filling.generation.?.progress();
    var admission = memory_policy.Admission{ .budget = 0, .memory = profile };
    try std.testing.expectError(error.RequestExceedsMemoryBudget, filling.guardPrefill(admission, &.{&filling}, null));
    try std.testing.expectEqualDeep(before, filling.generation.?.progress());
    admission.budget = std.math.maxInt(u64);
    try filling.guardPrefill(admission, &.{&filling}, null);
    while (!try filling.generation.?.step(session)) {}
    std.debug.print("PASS: live cache growth gating preserves activation order, paused greedy/seeded output and oldest-stream recovery\n", .{});
    std.debug.print("PASS: prefill chunk memory refusal preserves request state for subsequent completion\n", .{});
}

const PrefixStats = struct {
    enabled: std.atomic.Value(bool) = .init(false),
    bytes: std.atomic.Value(u64) = .init(0),
    entries: std.atomic.Value(usize) = .init(0),
    hits: std.atomic.Value(u64) = .init(0),
    misses: std.atomic.Value(u64) = .init(0),
    evictions: std.atomic.Value(u64) = .init(0),

    fn update(s: *PrefixStats, store: ?*const PrefixStore) void {
        if (store) |p| {
            s.enabled.store(true, .release);
            s.bytes.store(p.nbytes(), .release);
            s.entries.store(p.entries.items.len, .release);
            s.hits.store(p.hits, .release);
            s.misses.store(p.misses, .release);
            s.evictions.store(p.evictions, .release);
        }
    }

    fn snapshot(s: *const PrefixStats) struct { enabled: bool, bytes: u64, entries: usize, hits: u64, misses: u64, evictions: u64 } {
        return .{ .enabled = s.enabled.load(.acquire), .bytes = s.bytes.load(.acquire), .entries = s.entries.load(.acquire), .hits = s.hits.load(.acquire), .misses = s.misses.load(.acquire), .evictions = s.evictions.load(.acquire) };
    }
};

const MemoryStats = struct {
    budget: u64 = 0,
    mlx_budget: u64 = 0,
    admission_budget: u64 = 0,
    active: std.atomic.Value(usize) = .init(0),
    cache: std.atomic.Value(usize) = .init(0),
    peak: std.atomic.Value(usize) = .init(0),
    waiting: std.atomic.Value(u64) = .init(0),
    growth_waits: std.atomic.Value(u64) = .init(0),
    growth_ends: std.atomic.Value(u64) = .init(0),

    fn update(stats: *MemoryStats, waiting: u64) void {
        var value: usize = 0;
        if (mx.c.mlx_get_active_memory(&value) == 0) stats.active.store(value, .release);
        if (mx.c.mlx_get_cache_memory(&value) == 0) stats.cache.store(value, .release);
        if (mx.c.mlx_get_peak_memory(&value) == 0) stats.peak.store(value, .release);
        stats.waiting.store(waiting, .release);
    }

    fn snapshot(stats: *const MemoryStats) struct { budget: u64, mlx_budget: u64, admission_budget: u64, active: usize, cache: usize, peak: usize, waiting_requests: u64, growth_waits: u64, growth_ends: u64, probe_repeats: u32 } {
        return .{ .budget = stats.budget, .mlx_budget = stats.mlx_budget, .admission_budget = stats.admission_budget, .active = stats.active.load(.acquire), .cache = stats.cache.load(.acquire), .peak = stats.peak.load(.acquire), .waiting_requests = stats.waiting.load(.acquire), .growth_waits = stats.growth_waits.load(.acquire), .growth_ends = stats.growth_ends.load(.acquire), .probe_repeats = memory_policy.probe_repeats };
    }
};
fn connectionTask(w: *Worker, a: std.mem.Allocator, stream: std.Io.net.Stream, name: []const u8, sequence: usize, client: *control.Client) void {
    defer stream.close(w.io);
    defer w.control.release(client);
    var input: [65536]u8 = undefined;
    var output: [8192]u8 = undefined;
    var reader = stream.reader(w.io, &input);
    var writer = stream.writer(w.io, &output);
    var http = std.http.Server.init(&reader.interface, &writer.interface);
    var request = http.receiveHead() catch return;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    handle(w, arena.allocator(), &request, name, sequence, client) catch |err| {
        w.display.print("HTTP request failed: {s}\n", .{@errorName(err)});
    };
}

fn json(a: std.mem.Allocator, request: *Request, status: std.http.Status, value: anytype) !void {
    const body = try std.json.Stringify.valueAlloc(a, value, .{});
    try request.respond(body, .{ .status = status, .keep_alive = false, .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
}
fn failure(a: std.mem.Allocator, request: *Request, status: std.http.Status, message: []const u8) !void {
    try json(a, request, status, .{ .@"error" = .{ .message = message, .type = "invalid_request_error" } });
}
fn handle(worker: *Worker, a: std.mem.Allocator, request: *Request, model: []const u8, sequence: usize, client: *control.Client) !void {
    var route = request.head.target;
    if (std.mem.indexOfScalar(u8, route, '?')) |at| route = route[0..at];
    route = std.mem.trimEnd(u8, route, "/");
    if (request.head.method == .GET) {
        if (route.len == 0 or std.mem.eql(u8, route, "/health")) return json(a, request, .ok, .{ .status = "ok", .model = model, .warming = false, .max_batch_size = worker.batch_streams, .memory = worker.memory_stats.snapshot(), .prompt_cache = worker.prefix_stats.snapshot(), .inference = worker.stats.snapshot() });
        if (std.mem.eql(u8, route, "/v1/models") or std.mem.eql(u8, route, "/models")) return json(a, request, .ok, .{ .object = "list", .data = &.{.{ .id = model, .object = "model", .created = std.Io.Clock.real.now(worker.io).toSeconds(), .owned_by = "tensorfold" }} });
        return failure(a, request, .not_found, "Unknown route");
    }
    const is_chat = std.mem.eql(u8, route, "/v1/chat/completions") or std.mem.eql(u8, route, "/chat/completions");
    if (request.head.method != .POST or (!is_chat and !std.mem.eql(u8, route, "/v1/completions") and !std.mem.eql(u8, route, "/completions"))) return failure(a, request, .not_found, "Unknown route");
    var body_buffer: [8192]u8 = undefined;
    const body_reader = try request.readerExpectContinue(&body_buffer);
    const bytes = body_reader.allocRemaining(a, .limited(32 * 1024 * 1024)) catch {
        client.cancellation().check() catch |err| return requestFailure(a, request, err);
        return failure(a, request, .bad_request, "Invalid or oversized request body");
    };
    const body = std.json.parseFromSlice(std.json.Value, a, bytes, .{ .allocate = .alloc_always }) catch return failure(a, request, .bad_request, "Invalid JSON");
    const options = inference.Options.parseWithDefaults(a, body.value, worker.defaults) catch |err| return failure(a, request, .bad_request, @errorName(err));
    if (body.value.object.get("model")) |value| if (value != .string or !std.mem.eql(u8, value.string, model)) return failure(a, request, .not_found, "Unknown model");
    client.cancellation().check() catch |err| return requestFailure(a, request, err);
    var job = Job{ .stats = worker.stats, .a = a, .request = request, .model = model, .sequence = sequence, .created = std.Io.Clock.real.now(worker.io).toSeconds(), .client = client, .body = body.value, .is_chat = is_chat, .options = options };
    worker.stats.enqueue();
    defer worker.stats.finish(job.activated);
    if (try worker.queue.putUncancelable(worker.io, &.{&job}, 0) == 0) return failure(a, request, .service_unavailable, "Inference queue is full");
    job.done.waitUncancelable(worker.io);
    if (job.failure) |err| return err;
}

fn requestFailure(a: std.mem.Allocator, request: *Request, err: anyerror) !void {
    return failure(a, request, switch (err) {
        error.RequestTimedOut => .request_timeout,
        error.ServerStopping => .service_unavailable,
        else => .bad_request,
    }, @errorName(err));
}

const Pending = struct {
    job: *Job,
    id: []const u8 = "",
    thinking: bool = false,
    tools: std.json.Value = .null,
    max_calls: ?usize = null,
    markers: reply_text.Markers = .{},
    gate: ?@import("call_gate.zig").Gate = null,
    prepared_image: ?@import("vision.zig").Prepared = null,
    ids: []const i32 = &.{},
    options: inference.Options = .{},
    image: ?@import("vision.zig").Prompt = null,
    generation: ?inference.RequestGeneration = null,
    response: ?std.http.BodyWriter = null,
    stream: ?Stream = null,
    buffer: [8192]u8 = undefined,
    saved_position: usize = 0,
    prefix_reserve: u64 = 0,
    activation_order: u64 = 0,
    growth: enum { run, paused, ended } = .run,

    fn start(w: *Worker, session: *inference.Session, job: *Job) !?*Pending {
        const p = try job.a.create(Pending);
        p.* = .{ .job = job };
        p.prepare(w, session) catch |err| {
            defer p.deinit();
            try p.reportError(err);
            return null;
        };
        return p;
    }

    fn deinit(p: *Pending) void {
        if (p.generation) |*generation| generation.deinit();
        if (p.image) |*image| image.deinit();
        if (p.prepared_image) |*image| image.deinit();
        p.job.a.destroy(p);
    }

    fn reportError(p: *Pending, err: anyerror) !void {
        if (p.response) |*response| {
            const body = try std.json.Stringify.valueAlloc(p.job.a, .{ .@"error" = .{ .message = @errorName(err) } }, .{});
            try response.writer.print("data: {s}\n\ndata: [DONE]\n\n", .{body});
            try response.end();
        } else try requestFailure(p.job.a, p.job.request, err);
    }

    fn prepare(p: *Pending, w: *Worker, session: *inference.Session) !void {
        const a = p.job.a;
        const body = p.job.body;
        const cancellation = p.job.client.cancellation();
        try cancellation.check();
        var options = p.job.options;
        if (!w.drafts) options.draft = false;
        var ids: std.ArrayList(i32) = .empty;
        var raw_images = body.object.get("images") orelse .null;
        if (p.job.is_chat) {
            p.tools = try chat.activeTools(a, body);
            if (body.object.get("parallel_tool_calls")) |parallel| if (parallel != .null) {
                if (parallel != .bool) return error.InvalidParallelToolCalls;
                if (!parallel.bool) p.max_calls = 1;
            };
            const rendered = try session.renderChat(a, body, w.thinking, w.effort);
            try ids.appendSlice(a, try chat.encode(a, &session.tokenizer, rendered));
            raw_images = rendered.images;
            p.thinking = rendered.thinking;
            if (chat.requiresCall(body)) {
                const form = (try session.chat_template.?.callForm(a, &session.tokenizer)) orelse return error.UnsupportedRequiredToolCalls;
                var names: std.ArrayList([]const u8) = .empty;
                if (p.tools == .array) for (p.tools.array.items) |spec| try names.append(a, try chat.toolName(spec));
                p.gate = try @import("call_gate.zig").Gate.init(a, &session.tokenizer, ids.items, form, names.items, session.backend == .gemma);
            }
        } else switch (body.object.get("prompt") orelse return error.MissingPrompt) {
            .string => |value| {
                for (try session.tokenizer.encode(a, value)) |id| try ids.append(a, @intCast(id));
            },
            .array => |values| for (values.items) |value| {
                if (value != .integer or value.integer < 0 or value.integer > std.math.maxInt(i32)) return error.InvalidPromptToken;
                try ids.append(a, @intCast(value.integer));
            },
            else => return error.InvalidPrompt,
        }
        if (!p.thinking) options.thinking_budget = 0;
        try session.validate(ids.items, options);
        if (raw_images == .array and raw_images.array.items.len > 0 and session.backend != .qwen) return error.UnsupportedModelImages;
        const images = try @import("image_source.zig").loadWithCancellation(a, session.io, raw_images, w.vision_urls, cancellation);
        p.id = try std.fmt.allocPrint(a, "{s}cmpl-{d}-{d}", .{ if (p.job.is_chat) "chat" else "", p.job.created, p.job.sequence });
        p.markers = if (session.backend == .gemma) reply_text.gemma_markers else .{};
        try cancellation.check();
        if (images.len > 0) {
            p.prepared_image = try @import("vision.zig").Prepared.init(session.io, session.directory, images, ids.items);
            p.ids = try a.dupe(i32, p.prepared_image.?.tokens.items);
        } else p.ids = ids.items;
        try session.validate(p.ids, options);
        p.options = options;
    }

    fn activate(p: *Pending, session: *inference.Session) !void {
        const a = p.job.a;
        const cancellation = p.job.client.cancellation();
        const options = p.options;
        try cancellation.check();
        p.job.stats.activate();
        p.job.activated = true;
        if (p.prepared_image) |*prepared| {
            p.image = try prepared.encode(session.io, session.directory, &session.backend.qwen.weights);
            prepared.deinit();
            p.prepared_image = null;
        }
        if (options.stream) {
            p.response = try p.job.request.respondStreaming(&p.buffer, .{ .respond_options = .{ .keep_alive = false, .extra_headers = &.{ .{ .name = "content-type", .value = "text/event-stream" }, .{ .name = "cache-control", .value = "no-cache" } } } });
            p.stream = .{ .a = a, .writer = &p.response.?.writer, .transport = p.job.request.server.out, .id = p.id, .model = p.job.model, .created = p.job.created, .is_chat = p.job.is_chat, .thinking = p.thinking, .markers = p.markers, .tools = p.tools, .max_calls = p.max_calls, .cancellation = cancellation };
            if (p.job.is_chat) try p.stream.?.chatChunk(.{ .role = "assistant", .content = "" }, null);
        }
        p.generation = try inference.RequestGeneration.init(session, mx.allocator, p.ids, options, .{ .tools = p.tools, .context = if (p.stream) |*stream| stream else null, .emit = if (p.stream != null) Stream.emit else null, .cancellation = cancellation, .gate = if (p.gate) |*gate| gate else null }, if (p.image) |*image| image else null);
    }

    fn advance(p: *Pending, session: *inference.Session, admission: *memory_policy.Admission, active: []const ?*Pending, prefixes: ?*PrefixStore) !bool {
        try p.job.client.cancellation().check();
        if (p.growth == .ended) return error.StreamGrowthExceedsMemoryBudget;
        if (p.growth == .paused) return false;
        if (p.generation) |*generation| if (!generation.isDecoding()) try p.guardPrefill(admission.*, active, prefixes);
        if (p.generation == null) {
            var live: [8]memory_policy.Live = undefined;
            var count: usize = 0;
            var copies: u64 = 0;
            var work_prompt = p.ids.len;
            for (active) |slot| if (slot) |other| if (other.generation) |*generation| {
                live[count] = memory_policy.reserveLive(generation.memoryLengths(), other.ids.len, memory_policy.growth_horizon);
                count += 1;
                copies +|= other.prefix_reserve;
                if (!generation.isDecoding()) work_prompt = @max(work_prompt, other.ids.len);
            };
            const workspace = if (p.prepared_image) |*prepared| prepared.workspaceBytes() else 0;
            const longest = p.ids.len + memory_policy.reserveReply(p.options.max_tokens, memory_policy.growth_horizon);
            var projected = (try admission.projected(try memory_runtime.activeBytes(), work_prompt, longest, live[0..count])) +| copies;
            while ((p.options.max_tokens > 0 or workspace > 0) and projected +| workspace > admission.budget) {
                const store = prefixes orelse break;
                // Shared arrays may outlive an eviction; remeasure after every release.
                if ((projected +| workspace) -| store.nbytes() > admission.budget or !store.evictOne(null)) break;
                try mx.check(mx.c.mlx_clear_cache());
                projected = (try admission.projected(try memory_runtime.activeBytes(), work_prompt, longest, live[0..count])) +| copies;
            }
            if ((p.options.max_tokens > 0 or workspace > 0) and projected +| workspace > admission.budget) {
                admission.refused +|= 1;
                if (count == 0) return error.RequestExceedsMemoryBudget;
                return false;
            }
            try mx.check(mx.c.mlx_clear_cache());
            try p.activate(session);
            if (p.image == null and p.options.max_tokens > 0) if (prefixes) |store| {
                var hit = store.match(p.ids, p.generation.?.boundary(), false) catch null;
                if (hit) |*value| {
                    defer value.deinit(mx.allocator);
                    p.generation.?.restorePrefix(&value.cache) catch return p.step(session);
                    p.saved_position = value.count;
                    p.prefix_reserve = value.cache.nbytes();
                }
            };
        }
        const done = try p.step(session);
        if (!done) if (prefixes) |store| p.savePrefix(store, admission.*, active) catch {};
        return done;
    }

    fn guardPrefill(p: *Pending, admission: memory_policy.Admission, active: []const ?*Pending, prefixes: ?*PrefixStore) !void {
        var decoding: [8]memory_policy.Live = undefined;
        var count: usize = 0;
        for (active) |slot| if (slot) |other| if (other.growth == .run) {
            if (other.generation) |*generation| if (generation.isDecoding()) {
                decoding[count] = memory_policy.reserveLive(generation.memoryLengths(), other.ids.len, memory_policy.growth_horizon);
                decoding[count].copy_bytes = other.prefix_reserve;
                count += 1;
            };
        };
        const position = p.generation.?.memoryLengths().now;
        const memory = memory_runtime.Reclaim{ .prefixes = prefixes };
        while (true) {
            const needed = try admission.prefillProjected(try memory.used(), p.ids.len, position, p.prefix_reserve, decoding[0..count]);
            if (needed <= admission.budget) return;
            if (needed -| (try memory.freeable()) > admission.budget or !try memory.reclaim()) return error.RequestExceedsMemoryBudget;
        }
    }

    fn savePrefix(p: *Pending, store: *PrefixStore, admission: memory_policy.Admission, active: []const ?*Pending) !void {
        const generation = &p.generation.?;
        if (generation.memoryLengths().now <= p.saved_position) return;
        var snapshot = (try generation.snapshot()) orelse return;
        var adopted = false;
        defer if (!adopted) snapshot.deinit();
        const size = snapshot.nbytes();
        if (store.budget_bytes) |budget| if (size > budget and !store.admit_oversize) return;
        var live: [8]memory_policy.Live = undefined;
        var count: usize = 0;
        var prompt: usize = 0;
        var copies: u64 = 0;
        for (active) |slot| if (slot) |other| if (other.generation) |*request| {
            live[count] = memory_policy.reserveLive(request.memoryLengths(), other.ids.len, memory_policy.growth_horizon);
            count += 1;
            prompt = @max(prompt, other.ids.len);
            copies +|= other.prefix_reserve;
        };
        // Each live request can copy the retained buffers independently as it advances.
        const reserved = (copies -| p.prefix_reserve) +| @max(p.prefix_reserve, size);
        var projected = try admission.projected(try memory_runtime.activeBytes(), prompt, 0, live[0..count]);
        while (projected +| reserved > admission.budget) {
            if ((projected +| reserved) -| store.nbytes() > admission.budget or !store.evictOne(null)) return;
            try mx.check(mx.c.mlx_clear_cache());
            projected = try admission.projected(try memory_runtime.activeBytes(), prompt, 0, live[0..count]);
        }
        const position = snapshot.position();
        adopted = true;
        try store.insertOwned(p.ids[0..position], snapshot, p.ids, false);
        p.saved_position = position;
        p.prefix_reserve = @max(p.prefix_reserve, size);
    }

    fn step(p: *Pending, session: *inference.Session) !bool {
        const before = p.generation.?.progress();
        const started = live_status.now(session.io);
        const done = try p.generation.?.step(session);
        const ended = live_status.now(session.io);
        const after = p.generation.?.progress();
        p.job.stats.record(after.prefilled - before.prefilled, after.decoded - before.decoded, started, ended);
        p.job.stats.recordDrafts(after.proposed - before.proposed, after.accepted - before.accepted, after.structural_proposed - before.structural_proposed, after.structural_accepted - before.structural_accepted);
        p.job.stats.recordNeural(after.neural_proposed - before.neural_proposed, after.neural_accepted - before.neural_accepted);
        if (!done) return false;
        var reply = try p.generation.?.takeReply();
        defer reply.deinit(mx.allocator);
        const a = p.job.a;
        const request = p.job.request;
        const id = p.id;
        const model = p.job.model;
        const created = p.job.created;
        if (p.response) |*response| {
            const state = &p.stream.?;
            if (p.job.is_chat) {
                try state.chatText(true);
                try state.chatChunk(std.json.Value{ .object = .empty }, if (state.calls_sent > 0) "tool_calls" else @tagName(reply.finish_reason));
            } else try state.chunk("", @tagName(reply.finish_reason));
            try response.writer.writeAll("data: [DONE]\n\n");
            try response.end();
        } else {
            const usage = .{ .prompt_tokens = reply.prompt_tokens, .completion_tokens = reply.tokens.items.len, .total_tokens = reply.prompt_tokens + reply.tokens.items.len };
            if (p.job.is_chat) {
                const parts = if (p.thinking) reply_text.splitThinking(reply.content, true, p.markers) else reply_text.Parts{ .content = reply.content };
                var parsed = try tool_calls.parse(a, parts.content, p.tools, p.max_calls, id);
                if (p.max_calls != null and p.tools == .array and p.tools.array.items.len > 0) parsed.content = try tool_calls.singleContent(a, parsed.content);
                try json(a, request, .ok, .{ .id = id, .object = "chat.completion", .created = created, .model = model, .choices = &.{.{ .index = @as(usize, 0), .message = .{ .role = "assistant", .content = parsed.content, .reasoning_content = parts.reasoning, .tool_calls = parsed.calls }, .finish_reason = if (parsed.calls.len > 0) "tool_calls" else @tagName(reply.finish_reason) }}, .usage = usage });
            } else try json(a, request, .ok, .{ .id = id, .object = "text_completion", .created = created, .model = model, .choices = &.{.{ .index = @as(usize, 0), .text = reply.content, .finish_reason = @tagName(reply.finish_reason), .logprobs = @as(?u8, null) }}, .usage = usage });
        }
        return true;
    }
};

const Stream = struct {
    a: std.mem.Allocator,
    writer: *std.Io.Writer,
    transport: *std.Io.Writer,
    id: []const u8,
    model: []const u8,
    created: i64,
    cancellation: @import("cancellation.zig").Cancellation,
    is_chat: bool = false,
    thinking: bool = false,
    markers: reply_text.Markers = .{},
    accumulated: std.ArrayList(u8) = .empty,
    sent_content: usize = 0,
    sent_reasoning: usize = 0,
    tools: std.json.Value = .null,
    max_calls: ?usize = null,
    calls_sent: usize = 0,
    tool_stream: @import("tool_stream.zig").Streamer = .{},
    fn chunk(s: *Stream, value: []const u8, finish: ?[]const u8) !void {
        const body = try std.json.Stringify.valueAlloc(s.a, .{ .id = s.id, .object = "text_completion", .created = s.created, .model = s.model, .choices = &.{.{ .index = @as(usize, 0), .text = value, .finish_reason = finish, .logprobs = @as(?u8, null) }} }, .{});
        try s.writer.print("data: {s}\n\n", .{body});
        try s.writer.flush();
        try s.transport.flush();
    }
    fn emit(context: ?*anyopaque, value: []const u8) !void {
        if (value.len == 0) return;
        const s: *Stream = @ptrCast(@alignCast(context.?));
        try s.cancellation.check();
        if (!s.is_chat) return s.chunk(value, null);
        try s.accumulated.appendSlice(s.a, value);
        try s.chatText(false);
    }
    fn chatChunk(s: *Stream, delta: anytype, finish: ?[]const u8) !void {
        const body = try std.json.Stringify.valueAlloc(s.a, .{ .id = s.id, .object = "chat.completion.chunk", .created = s.created, .model = s.model, .choices = &.{.{ .index = @as(usize, 0), .delta = delta, .finish_reason = finish }} }, .{});
        try s.writer.print("data: {s}\n\n", .{body});
        try s.writer.flush();
        try s.transport.flush();
    }
    fn chatText(s: *Stream, finished: bool) !void {
        const parts = if (s.thinking) reply_text.splitThinking(s.accumulated.items, finished, s.markers) else reply_text.Parts{ .content = s.accumulated.items };
        if (parts.reasoning.len > s.sent_reasoning) {
            try s.chatChunk(.{ .reasoning_content = parts.reasoning[s.sent_reasoning..] }, null);
            s.sent_reasoning = parts.reasoning.len;
        }
        const has_tools = s.tools == .array and s.tools.array.items.len > 0;
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        const a = scratch.allocator();
        if (has_tools and s.max_calls == null) try s.tool_stream.feed(a, parts.content, s.tools, s, toolDelta);
        var parsed = if (finished) try tool_calls.parse(a, parts.content, s.tools, s.max_calls, s.id) else tool_calls.Result{ .content = if (has_tools) try tool_calls.preview(a, parts.content, s.tools, s.max_calls) else parts.content };
        if (s.max_calls != null and finished and has_tools) parsed.content = try tool_calls.singleContent(a, parsed.content);
        if (parsed.content.len > s.sent_content) {
            try s.chatChunk(.{ .content = parsed.content[s.sent_content..] }, null);
            s.sent_content = parsed.content.len;
        }
        for (parsed.calls, 0..) |call, index| {
            if (index < s.tool_stream.count) continue;
            try s.chatChunk(.{ .tool_calls = &.{.{ .index = index, .id = call.id, .type = "function", .function = .{ .name = call.function.name, .arguments = "" } }} }, null);
            try s.chatChunk(.{ .tool_calls = &.{.{ .index = index, .function = .{ .arguments = call.function.arguments } }} }, null);
            s.calls_sent += 1;
        }
    }
    fn toolDelta(context: ?*anyopaque, delta: @import("tool_stream.zig").Delta) !void {
        const s: *Stream = @ptrCast(@alignCast(context.?));
        if (delta.name) |name| {
            const id = try std.fmt.allocPrint(s.a, "call_{s}_{d}", .{ s.id, delta.index });
            try s.chatChunk(.{ .tool_calls = &.{.{ .index = delta.index, .id = id, .type = "function", .function = .{ .name = name, .arguments = "" } }} }, null);
            s.calls_sent += 1;
        } else try s.chatChunk(.{ .tool_calls = &.{.{ .index = delta.index, .function = .{ .arguments = delta.arguments } }} }, null);
    }
};
