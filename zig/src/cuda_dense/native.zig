//! Native(R): a dense recipe as the native CUDA host's family, its decls generated once for every recipe.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const kv = @import("core").kv_pool;
const recipe = @import("recipe.zig");
const kern = @import("kernels.zig");
const mdl = @import("model.zig");
const backend = @import("backend.zig");

/// The refusals the engine words itself, for every recipe; a recipe's own come from its `explain`.
pub const engine_errors = [_]anyerror{ error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.DenseTextOnly, error.StructuresUnsupported, error.NonFiniteLogits, error.CapturedKernels, error.ParallelPastRound, error.NoStreamFits, error.UnsupportedDenseHeadDim, error.UnsupportedDenseGroup, error.DenseImageSpan, error.DenseServedWithoutVision, error.DenseTowerWidth };

/// The host decls for recipe `R`: text-only, or with its tower's vision decls (the host reads their presence).
pub fn Native(comptime R: type) type {
    comptime recipe.check(R);
    return if (recipe.towerOf(R)) |T| WithTower(R, T) else TextOnly(R);
}

/// A text-only recipe: no vision decl or field, so the host refuses --vision by name.
fn TextOnly(comptime R: type) type {
    const C = Core(R);
    return struct {
        pub const model_type = C.model_type;
        pub const formats = C.formats;
        pub const default_context = C.default_context;
        pub const max_segments = C.max_segments;
        pub const prompt_rows = C.prompt_rows;
        /// A request's verify reserve: its pages hold that many rows past its prompt and reply.
        pub const window_rows = C.window_rows;
        /// `explain` words this family's open refusals too.
        pub const explains_open = true;
        /// No drafter: --drafter is refused before the load.
        pub const takes_drafter = false;
        pub const drafter_refusal = C.drafter_refusal;
        /// No prompt states: the host plans no prompt cache for this family and refuses --prompt-cache-gib.
        pub const keeps_no_prompts = true;
        pub const reasoning = C.reasoning;
        pub const Room = C.Room;
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
            target: *C.Dense,
            /// Every stream's keys and values are pages of the core pool, the first stream's too.
            layout: ?kv.Layout,
            engine_paged: bool = true,
        };

        /// The checkpoint, the shared and the recipe's kernels, the scratch; streams' pages come with attach.
        pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
            return C.loaded(Loaded, try C.open(gpa, io, ctx, dir, kernels, o));
        }

        pub const textWeightBytes = C.textWeightBytes;
        pub const attach = C.attach;
        pub const fits = C.fits;
        pub const ledgerPool = C.ledgerPool;
        pub const explain = C.explain;
    };
}

/// A recipe with tower `T`: the text-only decls, plus the options, loaded tower and bytes of --vision.
fn WithTower(comptime R: type, comptime T: type) type {
    const C = Core(R);
    return struct {
        pub const model_type = C.model_type;
        pub const formats = C.formats;
        pub const default_context = C.default_context;
        pub const max_segments = C.max_segments;
        pub const prompt_rows = C.prompt_rows;
        /// A request's verify reserve: its pages hold that many rows past its prompt and reply.
        pub const window_rows = C.window_rows;
        /// `explain` words this family's open refusals too.
        pub const explains_open = true;
        /// No drafter: --drafter is refused before the load.
        pub const takes_drafter = false;
        pub const drafter_refusal = C.drafter_refusal;
        /// No prompt states: the host plans no prompt cache for this family and refuses --prompt-cache-gib.
        pub const keeps_no_prompts = true;
        pub const reasoning = C.reasoning;
        /// A request's image tokens over all its images, unless --vision-image-tokens says otherwise.
        pub const default_image_tokens: u32 = T.default_image_tokens;
        pub const Room = C.Room;
        /// --vision: the tower's limits; no precision, kv8, prompt_int8, loops or rank field.
        pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 8, lanes_fixed: bool = false, room: ?Room = null, vision: ?T.Limits = null };
        pub const Loaded = struct {
            backend: lanes.backend.Backend,
            facts: lanes.Model,
            rows: u32,
            stream_bytes: usize,
            free_streams: u32 = 1,
            ctx: *anyopaque,
            deinit: *const fn (*anyopaque) void,
            target: *C.Dense,
            /// Every stream's keys and values are pages of the core pool, the first stream's too.
            layout: ?kv.Layout,
            engine_paged: bool = true,
            /// --vision: what the server's image requests reach (native/cuda_vision_offer.zig).
            vision: ?*T = null,
        };

        /// The checkpoint, the tower with --vision, the kernels, the scratch; streams' pages come with attach.
        pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
            return C.loaded(Loaded, try C.open(gpa, io, ctx, dir, kernels, o));
        }

        /// The tower's device bytes beside the text weights: its weights, its largest image's scratch, its workspace.
        pub fn visionBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
            return try T.weightBytes(a, io, dir) + try T.openBytes(a, io, dir);
        }

        pub const textWeightBytes = C.textWeightBytes;
        pub const attach = C.attach;
        pub const fits = C.fits;
        pub const ledgerPool = C.ledgerPool;
        pub const explain = C.explain;
    };
}

/// What both variants share: the open path, the page hooks and the refusals' words (not a host decl set).
pub fn Core(comptime R: type) type {
    const Tower = recipe.towerOf(R);
    return struct {
        pub const Dense = backend.Dense(R);

        const model_type: []const u8 = R.model_type;
        const formats: []const []const u8 = R.formats;
        const default_context: i64 = R.default_context;
        const max_segments: u32 = 1;
        const prompt_rows: u32 = backend.prompt_rows;
        const window_rows: u32 = backend.max_window;
        const drafter_refusal = std.fmt.comptimePrint("--drafter: {s} on CUDA loads no drafter; serve it without --drafter", .{R.name});
        /// Whether replies open with a reasoning block (the recipe's word, else the host's default).
        const reasoning: bool = if (@hasDecl(R, "reasoning")) R.reasoning else true;

        /// The server's reading of the room left after the weights; null if unreadable.
        const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };

        pub const Owned = struct { gpa: std.mem.Allocator, s: cuda.Stream, w: R.Weights, v: Dense.Vision, k: kern.Kernels, own: R.Kernels, m: mdl.Model, b: Dense };

        /// The checkpoint, the tower when `o` asks for it, the kernels, the scratch; one lane's image rows in the room.
        fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: anytype) !*Owned {
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
            own.v = try openTower(gpa, io, ctx, own.s, dir, g.hidden, o);
            errdefer closeTower(own.v);
            own.k = try kern.Kernels.load(ctx, g);
            errdefer own.k.deinit();
            own.own = try R.Kernels.load(ctx);
            errdefer own.own.deinit();
            const room: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
            const model = try mdl.Model.openBytes(gpa, ctx.d, own.s, g, o.context, &own.k.lane);
            const bytes = model + Dense.openBytes(g.vocab) + Dense.featureBytes(own.v, g.hidden);
            if (bytes > room) return error.NoStreamFits;
            own.m = try mdl.Model.init(gpa, ctx.d, own.s, g, o.context, &own.k.lane);
            errdefer own.m.deinit();
            own.b = try Dense.init(gpa, ctx.d, &own.m, &own.k, &own.own, &own.w, own.s, own.v);
            return own;
        }

        /// The tower with --vision, its rows `hidden` wide; null without the flag, void for a text-only recipe.
        fn openTower(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, s: cuda.Stream, dir: []const u8, hidden: usize, o: anytype) !Dense.Vision {
            if (Tower) |T| {
                const limits: ?T.Limits = if (@hasField(@TypeOf(o), "vision")) o.vision else null;
                const t = try T.open(gpa, io, ctx, s, dir, limits orelse return null);
                errdefer t.deinit();
                if (t.width != hidden) return error.DenseTowerWidth;
                return t;
            } else return {};
        }

        fn closeTower(v: Dense.Vision) void {
            if (Tower) |_| {
                if (v) |t| t.deinit();
            }
        }

        /// The host's view of an open engine: every stream's bytes and fixed part carry one lane's image rows.
        pub fn loaded(comptime L: type, own: *Owned) L {
            const feat = Dense.featureBytes(own.v, own.m.g.hidden);
            var layout = own.m.layout();
            layout.fixed_bytes += feat;
            var out: L = .{
                .backend = own.b.backend(),
                .facts = own.b.facts(),
                .rows = backend.max_window,
                .stream_bytes = own.m.streamBytes() + feat,
                .ctx = own,
                .deinit = release,
                .target = &own.b,
                .layout = layout,
            };
            if (@hasField(L, "vision")) out.vision = own.v;
            return out;
        }

        /// The checkpoint's weight bytes, counted from its tensors before the load.
        fn textWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !?u64 {
            return try R.weightBytes(a, io, dir);
        }

        fn owned(p: *anyopaque) *Owned {
            return @ptrCast(@alignCast(p));
        }

        /// The pool: `streams` more slots beside the first and an arena of `n` pages; their image rows with a tower.
        fn attach(p: *anyopaque, streams: u32, n: usize) !void {
            try owned(p).b.attach(streams, n);
        }

        /// Whether a request's pages could be promised now (the lane host's admission, on its engine thread).
        fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
            return owned(p).b.admits(prompt, max_tokens);
        }

        /// The pool the lane host's ledger admits this family's promises by.
        fn ledgerPool(p: *anyopaque) ?*kv.Pool {
            const b = &owned(p).b;
            return if (b.pool) |*pool| pool else null;
        }

        /// A request or load this engine refuses, in words; then the recipe's own, then its tower's; null: none.
        fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return words(err) orelse recipeWords(err) orelse towerWords(err);
        }

        fn recipeWords(err: anyerror) ?[]const u8 {
            return if (@hasDecl(R, "explain")) R.explain(err) else null;
        }

        fn towerWords(err: anyerror) ?[]const u8 {
            return if (Tower) |T| T.explain(err) else null;
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
                error.DenseImageSpan => "each image must fill its own run of image tokens in the prompt, in order and within the request's image token budget (--vision-image-tokens)",
                error.DenseServedWithoutVision => name ++ " on CUDA was served without --vision: serve it with --vision to send images",
                error.DenseTowerWidth => "this " ++ name ++ " checkpoint's image tower writes rows of another width than its decoder's hidden size",
                else => null,
            };
        }

        fn release(p: *anyopaque) void {
            const own = owned(p);
            own.b.deinit();
            own.m.deinit();
            own.own.deinit();
            own.k.deinit();
            closeTower(own.v);
            own.w.deinit();
            own.s.deinit();
            own.gpa.destroy(own);
        }
    };
}
