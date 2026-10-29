//! The Liquid families behind the native CUDA registry: open, admission, the pool and the prompt cache's hooks.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const kv = @import("core").kv_pool;

/// `F` names the family's `wts`, `kern`, `mdl`, `be`, `snap`, `pages_env` and `check` (beyond the admission).
pub fn Native(comptime F: type) type {
    return struct {
        const wts = F.wts;
        const kern = F.kern;
        const mdl = F.mdl;
        const be = F.be;
        const snap = F.snap;
        const Pages = @import("cuda_snap.zig").Pages;

        pub const prompt_rows: u32 = be.prompt_rows;
        /// A kept state is the caches through its position and nothing past it (no draft head reads a token on).
        pub const snap_lookahead: u32 = 0;
        /// A request's last verify window: its pages hold that many rows past its prompt and reply.
        pub const window_rows: u32 = be.max_window;

        /// The server's reading of the room left after the weights; null if unreadable.
        pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
        /// No `precision` field: the server refuses --precision for this family by name.
        pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 8, lanes_fixed: bool = false, room: ?Room = null };

        pub const Loaded = struct {
            backend: lanes.backend.Backend,
            facts: lanes.Model,
            rows: u32,
            stream_bytes: usize,
            free_streams: u32,
            ctx: *anyopaque,
            deinit: *const fn (*anyopaque) void,
            target: *be.Liquid,
            /// The attention caches as pages of the core pool (null: every stream owns a window, made at open).
            layout: ?kv.Layout = null,
            /// With a layout: the first stream's rows are pages too, a window of them set aside by the plan.
            engine_paged: bool = true,
        };

        const Owned = struct { gpa: std.mem.Allocator, w: wts.Weights, k: kern.Kernels, s: cuda.Stream, m: mdl.Model, lb: be.Liquid };

        /// Streams to allocate in `room`: as asked, else the most that fit; a fixed count past them refuses.
        fn fit(room: u64, asked: u32, fixed: bool, plan: anytype) !usize {
            if (fixed and asked > be.round_rows) return error.ParallelPastRound;
            var n = std.math.clamp(@as(usize, asked), 1, be.round_rows);
            while (try plan.bytes(n) > room) : (n -= 1) {
                if (fixed) return error.ParallelPastMemory;
                if (n == 1) return error.NoStreamFits;
            }
            return n;
        }

        /// The pages switch (TENSORFOLD_LFM*_KV_PAGES) at 0: each stream owns its pages, else it leases the pool's.
        fn pagesOn() bool {
            const v = std.c.getenv(F.pages_env) orelse return true;
            return !std.mem.eql(u8, std.mem.span(v), "0");
        }

        /// The device bytes `n` streams take (paged: the first's states and a round's slot buffers, no pages).
        const Plan = struct {
            own: *const Owned,
            d: *const cuda.Driver,
            context: usize,
            paged: bool,

            fn bytes(p: Plan, n: usize) !u64 {
                const o = p.own;
                return try mdl.Model.openBytes(o.gpa, p.d, o.s, &o.w, &o.k, p.context, n, p.paged) + be.Liquid.openBytes(if (p.paged) be.round_rows else n);
            }
        };

        /// The pool: `streams` more slots beside the first and an arena of `n` pages.
        pub fn attach(p: *anyopaque, streams: u32, n: usize) !void {
            const own: *Owned = @ptrCast(@alignCast(p));
            try own.lb.attach(streams, n);
        }

        /// Whether a request's caches could start now (the lane host's admission, on its engine thread).
        pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
            const own: *const Owned = @ptrCast(@alignCast(p));
            return own.lb.admits(prompt, max_tokens);
        }

        pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
            if (kernels != null) return error.CapturedKernels;
            if (o.lanes_fixed and o.lanes > be.round_rows) return error.ParallelPastRound;
            const own = try gpa.create(Owned);
            errdefer gpa.destroy(own);
            own.* = .{ .gpa = gpa, .w = undefined, .k = undefined, .s = undefined, .m = undefined, .lb = undefined };
            own.k = try kern.Kernels.load(ctx);
            errdefer own.k.deinit();
            own.s = try cuda.Stream.init(ctx.d, false);
            errdefer own.s.deinit();
            own.w = try wts.load(gpa, io, ctx.d, own.s, &own.k, dir, null);
            errdefer own.w.deinit();
            try F.check(&own.w.config);
            const context = @min(mdl.Model.positions(o.context), own.w.config.context);
            const paged = pagesOn();
            const plan: Plan = .{ .own = own, .d = ctx.d, .context = context, .paged = paged };
            const room: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
            const n = if (paged) try fitFirst(room, plan) else fit(room, o.lanes, o.lanes_fixed, plan) catch |e| {
                if (e == error.ParallelPastMemory) {
                    const most = fit(room, o.lanes, false, plan) catch 0;
                    std.log.err("--parallel {d} needs {d:.2} GiB at a {d}-token window and the CUDA memory budget leaves {d:.2} GiB after the weights: {d} stream{s} fit", .{ o.lanes, gib(try plan.bytes(o.lanes)), context, gib(room), most, if (most == 1) "" else "s" });
                }
                return e;
            };
            own.m = try mdl.Model.init(gpa, ctx.d, own.s, &own.w, &own.k, context, n, paged);
            errdefer own.m.deinit();
            own.lb = try be.Liquid.init(gpa, ctx.d, &own.m, &own.k, own.s);
            errdefer own.lb.deinit();
            return .{
                .backend = own.lb.backend(),
                .facts = own.lb.facts(),
                .rows = be.max_window,
                .stream_bytes = mdl.Model.streamBytes(context),
                .free_streams = @intCast(n),
                .ctx = own,
                .deinit = release,
                .target = &own.lb,
                .layout = if (paged) own.m.layout() else null,
            };
        }

        /// Paged: the first stream's states and the scratch must fit; the host plans the pool after.
        fn fitFirst(room: u64, plan: anytype) !usize {
            if (try plan.bytes(1) > room) return error.NoStreamFits;
            return 1;
        }

        fn gib(bytes: u64) f64 {
            return @as(f64, @floatFromInt(bytes)) / (1 << 30);
        }

        fn lfm(ptr: *anyopaque) *be.Liquid {
            return @ptrCast(@alignCast(ptr));
        }

        /// Bytes a state at `at` prompt tokens takes: whole snapshot pages.
        pub fn snap_bytes(_: *anyopaque, at: u32) u64 {
            return snap.bytes(at);
        }

        /// The cache's pages, allocated once before any request; the store counts held and spare pages via the hooks.
        pub fn snap_ready(ptr: *anyopaque, budget: u64) !u64 {
            const x = lfm(ptr);
            if (x.snaps != null) return error.SnapshotCacheMade;
            x.snaps = try Pages.init(x.gpa, x.d, budget);
            return 0;
        }

        pub fn snap_charged(_: *anyopaque, saved: *anyopaque) u64 {
            const k: *const snap.Saved = @ptrCast(@alignCast(saved));
            return k.lease.ids.items.len * @import("cuda_snap.zig").page;
        }

        pub fn snap_spare(ptr: *anyopaque) u64 {
            return if (lfm(ptr).snaps) |*p| p.spare() else 0;
        }

        pub fn snap_reuses(ptr: *anyopaque, at: u32) bool {
            return snap_spare(ptr) >= snap.bytes(at);
        }

        /// The prompt pass stands at `at` (a chunk end or a mark): copy the stream's caches there.
        pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
            return @ptrCast(try lfm(ptr).saveState(owner, at));
        }

        pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
            return lfm(ptr).restoreState(owner, @ptrCast(@alignCast(saved)));
        }

        pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
            lfm(ptr).dropState(@ptrCast(@alignCast(saved)));
        }

        fn release(p: *anyopaque) void {
            const own: *Owned = @ptrCast(@alignCast(p));
            own.lb.deinit();
            own.m.deinit();
            own.w.deinit();
            own.s.deinit();
            own.k.deinit();
            own.gpa.destroy(own);
        }
    };
}
