//! Both device factories serve one native forward owner through the existing lane host and prompt-cache store.
const std = @import("std");
const cfg = @import("config.zig");
pub const formats: []const []const u8 = &.{ "bf16", "mlx-q4g32", "mlx-q4g64", "mlx-q4g128", "mlx-q8g32", "mlx-q8g64", "mlx-q8g128", "modelopt-fp8", "modelopt-nvfp4" };
pub fn refusal(o: anytype) ?[]const u8 {
    if (o.tp != 1 or o.rank != 0) return "Qwen4: --tp and --rank need the single-device path (1 and 0)";
    if (o.vision != null) return "Qwen4: --vision needs a published dense image schema";
    if (o.kv8) return "Qwen4: --kv-cache fp8 is unsupported; the exact cache stores bf16";
    if (o.prompt_int8) return "Qwen4: --prompt-precision int8 is unsupported; activations stay bf16";
    if (o.precision) |p| if (p != .full) return "Qwen4: --precision checkpoint is unsupported; use full for bf16 activations";
    if (o.drafter != null or o.mtp_drafts != null or o.mtp_confidence != null) return "Qwen4: --drafter and --mtp-* need a published dense draft-head schema";
    if (o.slide) return "Qwen4: --slide is unsupported";
    if (o.learn != null or o.learn_conversations) return "Qwen4: --learn and --learn-conversations are unsupported";
    if (o.recall or o.record != null) return "Qwen4: --recall and --record are unsupported";
    if (o.speed_up != null or o.policy.len != 0) return "Qwen4: --speed-up and --policy are unsupported";
    if (o.segments) |n| if (n != 1) return "Qwen4: --segments must be 1; shared windows already pack independent streams";
    if (o.keep) |n| if (n < 0) return "Qwen4: --checkpoint-slots must be nonnegative";
    if (o.lanes == 0 or o.lanes > 128) return "Qwen4: --parallel must be 1 through 128";
    if (o.prompt_cache_gib) |g| if (!std.math.isFinite(g) or g < 0 or g >= 0x1p34) return "Qwen4: --prompt-cache-gib must be a finite nonnegative budget";
    return null;
}
pub fn Host(comptime P: type, comptime Api: type, comptime L: type) type {
    const E = P.Engine;
    const D = P.Drawer;
    const A = @import("lanes_adapter.zig").Adapter(E, D, L);
    const Hooks = @import("lane_snapshots.zig").Hooks(A, Api.prompt_cache);
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        p: ?*P = null,
        e: ?*E = null,
        d: D = undefined,
        adapter: A = undefined,
        hooks: Hooks = undefined,
        store: Api.prompt_cache.Store = undefined,
        config: L.Config = undefined,
        clock: L.backend.WallClock = undefined,
        core: L.Engine = undefined,
        host: Api.LaneHost = undefined,
        drawer_ready: bool = false,
        adapter_ready: bool = false,
        store_ready: bool = false,
        config_ready: bool = false,
        core_ready: bool = false,
        started: bool = false,
        next: ?*Self = null,
        var retained: ?*Self = null;
        var mutex: std.atomic.Mutex = .unlocked;
        fn lock() void {
            while (!mutex.tryLock()) std.Thread.yield() catch {};
        }
        fn retain(h: *Self) void {
            lock();
            defer mutex.unlock();
            h.next = retained;
            retained = h;
        }
        /// A failed completion keeps the entire parent chain, and blocks a new load until recovery proves it idle.
        pub fn recover() bool {
            lock();
            defer mutex.unlock();
            var at = retained;
            retained = null;
            while (at) |h| {
                at = h.next;
                h.next = null;
                h.finish() catch {
                    h.next = retained;
                    retained = h;
                };
            }
            return retained == null;
        }
        pub fn open(a: std.mem.Allocator, io: std.Io, o: Api.Open, p: *cfg.Problem) !Api.Opened {
            if (!recover()) return error.Qwen4RetainedOwner;
            const h = try a.create(Self);
            h.* = .{ .a = a };
            errdefer h.finish() catch {
                retain(h);
            };
            const path = try std.fs.path.join(a, &.{ o.dir, "config.json" });
            defer a.free(path);
            const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24));
            defer a.free(text);
            var c = try cfg.parse(a, text, p);
            defer c.deinit();
            const cap = if (o.context) |n| (if (n < 0) return p.reject("--context") else if (n == 0) c.context else std.math.cast(usize, n) orelse return p.reject("--context")) else c.context;
            _ = try @import("native_admission.zig").check(&c, E.Gpu.kind, p);
            h.p = try P.open(a, io, o);
            const platform = h.p.?;
            const room = try platform.room(a, io);
            const cache: u64 = if (o.prompt_cache_gib) |g| @intFromFloat(g * 0x1p30) else @min(@as(u64, 8) << 30, room / 10);
            if (cache > room) return error.Qwen4PathDoesNotFit;
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
            var streams: usize = o.lanes;
            while (streams > 0) : (streams -= 1) {
                const rows = @min(@as(usize, 128), streams * 16);
                const draw = try D.bytes(c.vocab, rows, streams);
                if (draw > room - cache) {
                    if (o.lanes_fixed) return error.Qwen4PathDoesNotFit;
                    continue;
                }
                const options: @import("native_admission.zig").Options = .{ .capacity = cap, .window = 16, .streams = streams, .budget = room - cache - draw, .config_digest = digest };
                h.e = E.open(a, io, o.dir, &platform.storage, options, p, platform.loader()) catch |err| {
                    if (err == error.Qwen4PathDoesNotFit and !o.lanes_fixed) continue;
                    return err;
                };
                break;
            }
            const e = h.e orelse return error.Qwen4PathDoesNotFit;
            h.d = try platform.drawer(a, e.c.vocab, e.plan.rows, e.o.streams);
            h.drawer_ready = true;
            h.adapter = try A.init(a, e, &h.d);
            h.adapter_ready = true;
            h.hooks = .{ .adapter = &h.adapter };
            h.store = Api.prompt_cache.Store.init(a, h.hooks.snapshots(), .{ .lookahead = 0, .chunk = 16, .warm = .on }, cache);
            h.store_ready = true;
            if (o.keep) |n| h.store.most = @intCast(n);
            var facts = h.adapter.facts();
            facts.copy_drafts = o.drafts;
            h.config = try L.Config.init(a, facts, @intCast(e.o.window), @intCast(e.o.window - 1));
            h.config_ready = true;
            h.clock = .{ .io = io };
            h.core = L.Engine.init(a, &h.config, h.adapter.backend(), h.clock.clock());
            h.core_ready = true;
            var state = e.slots[0].state;
            state.pos = 1;
            h.host = Api.LaneHost.init(a, io, &h.core, .{ .name = "qwen4", .lanes = @intCast(e.o.streams), .context_window = @intCast(cap), .structures = true, .prefill_step = 16, .prompt_cache = cache > 0, .warm_turns = cache > 0, .plain_only = !o.drafts, .prompt_cache_capacity = .{ .budget_bytes = cache, .minimum_state_bytes = try A.Snap.charge(&state) }, .policy = "Qwen4 dense, stored weights with bf16 activations" });
            h.host.health = .{ .ctx = h, .read = healthy };
            h.host.explain = .{ .ctx = h, .text = explain };
            h.host.cache = if (cache > 0) &h.store else null;
            try h.host.start();
            h.started = true;
            return .{ .engine = h.host.engine(), .ctx = h, .close = close, .halt = halt };
        }
        fn healthy(p: *anyopaque) bool {
            const h: *Self = @ptrCast(@alignCast(p));
            return h.e.?.healthy() and h.p.?.healthy();
        }
        fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return switch (err) {
                error.Qwen4Context => "the prompt and reply exceed --context",
                error.Qwen4TextOnly => "Qwen4 serves text only",
                error.Qwen4LogprobsUnsupported => "Qwen4 logprobs are not available",
                error.Qwen4EngineFailed => "Qwen4 stopped after a device failure; restart the engine",
                else => null,
            };
        }
        fn halt(p: *anyopaque, why: []const u8) void {
            const h: *Self = @ptrCast(@alignCast(p));
            h.host.halt(why);
        }
        fn close(p: *anyopaque) void {
            const h: *Self = @ptrCast(@alignCast(p));
            h.finish() catch {
                retain(h);
            };
        }
        fn finish(h: *Self) !void {
            if (h.started) {
                h.host.stop();
                h.started = false;
            }
            if (h.p) |p| try p.storage.drain();
            if (h.core_ready) {
                h.core.deinit();
                h.core_ready = false;
            }
            if (h.config_ready) {
                h.config.deinit(h.a);
                h.config_ready = false;
            }
            if (h.store_ready) {
                h.store.deinit();
                h.store_ready = false;
            }
            if (h.p) |p| try p.storage.drain();
            if (h.adapter_ready) {
                h.hooks.reclaimDrained();
                h.adapter.deinitDrained();
                h.adapter_ready = false;
            }
            if (h.drawer_ready) {
                h.d.deinitDrained();
                h.drawer_ready = false;
            }
            if (h.e) |e| {
                try e.close();
                h.e = null;
            }
            if (h.p) |p| {
                try p.close();
                h.p = null;
            }
            const a = h.a;
            a.destroy(h);
        }
    };
}
