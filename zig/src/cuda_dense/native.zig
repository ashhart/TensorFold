//! Native(R): a dense recipe as the native CUDA host's family, its decls generated once for every recipe.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const kv = @import("core").kv_pool;
const recipe = @import("recipe.zig");
const kern = @import("kernels.zig");
const mdl = @import("model.zig");
const pages = @import("pages.zig");
const backend = @import("backend.zig");

/// The refusals the engine words itself, for every recipe; a recipe's own come from its `explain`.
pub const engine_errors = [_]anyerror{ error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.DenseTextOnly, error.StructuresUnsupported, error.NonFiniteLogits, error.CapturedKernels, error.ParallelPastRound, error.NoStreamFits, error.UnsupportedDenseHeadDim, error.UnsupportedDenseGroup };

pub fn Native(comptime R: type) type {
    comptime recipe.check(R);
    return struct {
        const Dense = backend.Dense(R);

        pub const model_type: []const u8 = R.model_type;
        pub const formats: []const []const u8 = R.formats;
        pub const default_context: i64 = R.default_context;
        pub const max_segments: u32 = 1;
        pub const prompt_rows: u32 = backend.prompt_rows;
        /// A request's verify reserve: its pages hold that many rows past its prompt and reply.
        pub const window_rows: u32 = backend.max_window;
        /// `explain` words this family's open refusals too.
        pub const explains_open = true;
        /// No drafter: --drafter is refused before the load.
        pub const takes_drafter = false;
        pub const drafter_refusal = std.fmt.comptimePrint("--drafter: {s} on CUDA loads no drafter; serve it without --drafter", .{R.name});
        /// No prompt states: the host plans no prompt cache for this family and refuses --prompt-cache-gib.
        pub const keeps_no_prompts = true;
        /// Whether replies open with a reasoning block (the recipe's word, else the host's default).
        pub const reasoning: bool = if (@hasDecl(R, "reasoning")) R.reasoning else true;

        /// The server's reading of the room left after the weights; null if unreadable.
        pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
        /// No vision, precision, kv8, prompt_int8, loops or rank field: the host refuses those flags by name.
        pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 8, lanes_fixed: bool = false, room: ?Room = null };

        pub const Loaded = struct {
            backend: lanes.backend.Backend,
            facts: lanes.Model,
            rows: u32,
            stream_bytes: usize,
            free_streams: u32 = 1,
            ctx: *anyopaque,
            deinit: *const fn (*anyopaque) void,
            target: *Dense,
            /// Every stream's keys and values are pages of the core pool, the first stream's too.
            layout: ?kv.Layout,
            engine_paged: bool = true,
        };

        const Owned = struct { gpa: std.mem.Allocator, s: cuda.Stream, w: R.Weights, k: kern.Kernels, own: R.Kernels, m: mdl.Model, b: Dense };

        /// The checkpoint, the shared and the recipe's kernels, the scratch; streams' pages come with attach.
        pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
            if (kernels != null) return error.CapturedKernels;
            if (o.lanes_fixed and o.lanes > backend.round_rows) return error.ParallelPastRound;
            const g = try R.geometry(gpa, io, dir);
            _ = try recipe.attentionShape(g); // refused by name before the weights load
            const own = try gpa.create(Owned);
            errdefer gpa.destroy(own);
            own.gpa = gpa;
            own.s = try cuda.Stream.init(ctx.d, false);
            errdefer own.s.deinit();
            own.w = try R.load(gpa, io, ctx.d, own.s, dir);
            errdefer own.w.deinit();
            own.k = try kern.Kernels.load(ctx, g);
            errdefer own.k.deinit();
            own.own = try R.Kernels.load(ctx);
            errdefer own.own.deinit();
            const room: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
            const bytes = try mdl.Model.openBytes(gpa, ctx.d, own.s, g, o.context, &own.k.lane) + Dense.openBytes(g.vocab);
            if (bytes > room) return error.NoStreamFits;
            own.m = try mdl.Model.init(gpa, ctx.d, own.s, g, o.context, &own.k.lane);
            errdefer own.m.deinit();
            own.b = try Dense.init(gpa, ctx.d, &own.m, &own.k, &own.own, &own.w, own.s);
            return .{
                .backend = own.b.backend(),
                .facts = own.b.facts(),
                .rows = backend.max_window,
                .stream_bytes = own.m.streamBytes(),
                .ctx = own,
                .deinit = release,
                .target = &own.b,
                .layout = own.m.layout(),
            };
        }

        /// The checkpoint's weight bytes, counted from its tensors before the load.
        pub fn textWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !?u64 {
            return try R.weightBytes(a, io, dir);
        }

        fn owned(p: *anyopaque) *Owned {
            return @ptrCast(@alignCast(p));
        }

        /// The pool: `streams` more slots beside the first and an arena of `n` pages.
        pub fn attach(p: *anyopaque, streams: u32, n: usize) !void {
            try owned(p).b.attach(streams, n);
        }

        /// Whether a request's pages could be promised now (the lane host's admission, on its engine thread).
        pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
            return owned(p).b.admits(prompt, max_tokens);
        }

        /// The pool the lane host's ledger admits this family's promises by.
        pub fn ledgerPool(p: *anyopaque) ?*kv.Pool {
            const b = &owned(p).b;
            return if (b.pool) |*pool| pool else null;
        }

        /// A request or load this engine refuses, in words; then the recipe's own; null: none.
        pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return words(err) orelse if (@hasDecl(R, "explain")) R.explain(err) else null;
        }

        fn words(err: anyerror) ?[]const u8 {
            const name = R.name;
            return switch (err) {
                error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
                error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
                error.NoDraftHead => name ++ " on CUDA has no draft head: it drafts from the prompt's own text",
                error.DenseTextOnly => name ++ " on CUDA reads text prompts only so far: this request has images",
                error.StructuresUnsupported => "structured output is not built for " ++ name ++ " on CUDA yet",
                error.NonFiniteLogits => "the model's logits went non-finite at this row: the request is stopped rather than answered from them",
                error.CapturedKernels => name ++ " runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
                error.ParallelPastRound => std.fmt.comptimePrint("--parallel: {s} on CUDA holds at most {d} streams (a round's rows): serve --parallel {d} or fewer, or auto", .{ name, backend.round_rows, backend.round_rows }),
                error.NoStreamFits => "the CUDA memory budget fits no " ++ name ++ " stream after the weights: lower --context, free device memory, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
                error.UnsupportedDenseHeadDim => "this " ++ name ++ " checkpoint's attention head dim is not 64 or 128, the two the CUDA dense engine's attention kernels take",
                error.UnsupportedDenseGroup => "this " ++ name ++ " checkpoint's query heads a KV head are not an even count of at most 16, the groups the CUDA dense engine's attention kernels take",
                else => null,
            };
        }

        fn release(p: *anyopaque) void {
            const own = owned(p);
            own.b.deinit();
            own.m.deinit();
            own.own.deinit();
            own.k.deinit();
            own.w.deinit();
            own.s.deinit();
            own.gpa.destroy(own);
        }
    };
}
