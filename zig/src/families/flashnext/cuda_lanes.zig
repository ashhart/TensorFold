//! Flash Next's lane backend on the native CUDA server: chunked prompts, MTP draft windows, and shared rounds.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const weights = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");
const mtp_ops = @import("cuda_mtp_ops.zig");
const lanes_draft = @import("cuda_lanes_draft.zig");
const snap = @import("cuda_snap.zig");
const ranks = @import("cuda_rank_engine.zig");
const Glue = @import("cuda_glue.zig").Glue;
const living = @import("living");
const slide_mod = @import("cuda_slide.zig");
const train = @import("cuda_train.zig");
const bytes = @import("cuda_lanes_bytes.zig");
const kvl = @import("cuda_kv_lanes.zig");
const be = lanes.backend;
const ring = 1024;
/// Streams a round takes at most (a round with more live streams takes them in turns).
const round_streams = 32;
/// The Triton entry points the pass launches and the head's two more: a captured set replaces our glue with all.
pub const captured_entries = [_][]const u8{ "_embed", "_hc_writeback", "_qmm_hcdown", "_hc_reduce_act", "_qmm_upmix", "_ple_gate", "_ple_conv", "_attn_prep", "_attn_gate", "_pool", "_scores", "_chunks", "_merge", "_router", "_topk_rows" };
pub const captured_head_entries = [_][]const u8{ "_rmsnorm", "_add_streams" };
/// `Tri` launches our glue, or a captured set's. `open` loads the checkpoint and serves greedy or keyed draws.
pub fn Namespace(comptime Tri: type) type {
    return struct {
        pub const model_type = "qwen4_exp";
        pub const tp_draft_snapshots = false;
        pub const formats: []const []const u8 = &.{ "mlx-q4g32", "modelopt-mixed" };
        // the model's window: past 2,048 keys the block indexer picks each row's 512 best blocks of 4 (cuda_batch.zig)
        pub const default_context: i64 = @intCast(prompt.max_tokens);
        // a prompt runs on the batched pass in chunks of this many rows, one a prefill_step: no staggered segments
        pub const max_segments: u32 = 1;
        pub const prompt_rows: u32 = @intCast(prompt.batch.default_chunk);
        /// The prompt cache's copies of a stream's state (cuda_snap.zig), called with `Loaded.target`.
        pub const snap_bytes = snapBytes;
        pub const snap_repeat_min: u32 = 16;
        pub const snap_save = snapSave;
        pub const snap_restore = snapRestore;
        pub const snap_drop = snapDrop;
        pub const kept_disk = @import("cuda_disk.zig").Kept(@This()); // their disk image, as qwen3_5 declares its own
        /// TF_FLASHNEXT_KV_PAGES: the pool's lanes, the pages of a request's rows and a drafted window (kvl).
        pub const window_rows = kvl.window_rows;
        pub fn attach(p: *anyopaque, streams: u32, count: usize) !void {
            const own = of(p);
            own.kv.pool = try kvl.Pool.init(own.gpa, own.engine, own.kv.layout.?, own.context, own.kv.head, streams, count);
            own.kv.paged_only = kvl.pagedOnly() and streams > 0;
        }
        pub fn fits(p: *anyopaque, rows: u32, max_tokens: u32) bool {
            return of(p).kv.fits(rows, max_tokens);
        }
        /// The pass ignores its cuts, so states keep at any token; the MTP head reads one token on (lookahead 1).
        pub const cache_rules = struct {
            pub const lookahead: u32 = 1;
            pub const planned = false;
            pub const min_gap: u32 = 256;
            pub const min_prompt: u32 = 512;
        };
        pub const Options = struct { vision: ?@import("cuda_images.zig").Limits = null, context: usize, drafts: bool, segments: usize = 1, slide: bool = false, tp: usize = 1, rank: usize = 0, master: ?[]const u8 = null, master_port: u16 = 29551 };
        /// Living Weights' learner (--slide): lessons put neurons in each layer's pocket beside its shared expert.
        pub const slide = slide_mod.Slide(Tri);
        /// The first file a lesson would rewrite through a link, or null: a teaching copy may hard-link the others.
        pub const linked = slide_mod.linked;
        /// What a learner over the loaded engine `p` learns into; `dir` must outlive it.
        pub fn learnModel(p: *anyopaque, dir: []const u8) slide.Model {
            const own: *Cuda = @ptrCast(@alignCast(p));
            return .{ .e = own.engine, .tri = own.tri(), .dir = dir, .lay = own.lay };
        }
        /// One stream's prompt-and-reply driver, as native/cuda.zig types it; this family has none.
        pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;
        pub const Loaded = struct {
            vision: ?*@import("cuda_images.zig").Vision = null,
            backend: be.Backend,
            facts: lanes.Model,
            rows: u32,
            prompt_rows: u32,
            /// Device bytes each admitted stream allocates: its caches (and with the head, its cache and window).
            stream_bytes: usize,
            /// Every stream allocates its own sequence; none rides on the engine's.
            free_streams: u32 = 0,
            ctx: *anyopaque,
            deinit: *const fn (*anyopaque) void,
            lone: ?LoneRun = null,
            learns: ?*anyopaque = null,
            learn_bytes: usize = 0,
            /// What the prompt cache's copies are taken from: the backend itself (snap_* look up the owner stream).
            target: *anyopaque,
            /// Whether our own glue runs (no captured set given, or one without this family's entry points).
            own_glue: bool = true,
            follow: ?*const fn (*anyopaque) anyerror!void = null,
            /// TF_FLASHNEXT_KV_PAGES: the streams beside the first take pages of one pool (null: their own caches).
            layout: ?@import("core").kv_pool.Layout = null,
        };

        /// Device bytes of the checkpoint: the n-gram tables stay mapped, the MTP head counts; 0 if unread.
        pub fn deviceWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) u64 {
            return bytes.deviceWeightBytes(gpa, io, dir);
        }

        pub fn placementWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize, rank: usize, drafts: bool) !u64 {
            return bytes.placementWeightBytes(gpa, io, dir, world, rank, drafts);
        }

        /// The tower's weights and one call's scratch, which rank zero holds beside the text weights with --vision.
        pub fn visionBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
            return bytes.visionBytes(a, io, dir);
        }

        pub fn healthy(ptr: *anyopaque) bool {
            if (of(ptr).ranks) |r| return r.healthy();
            return true;
        }

        pub fn rankRoom(ptr: *anyopaque, local: u64) !u64 {
            if (of(ptr).ranks) |r| return r.room(local);
            return local;
        }

        /// A request or load this engine refuses, in words; null: none of its own.
        pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return @import("cuda_lanes_explain.zig").explain(err);
        }

        /// A stream's caches; with the head, its cache, a window's buffers and the drafts it holds for the next round.
        const Lane = struct {
            seq: prompt.Seq,
            draft: ?mtp.Draft = null,
            chain: mtp.Chain = .{},
            /// The prompt's next row while its pass runs a chunk a call; null once the prompt is in.
            filled: ?usize = 0,
            /// Where its caches came from: the pool's lanes, or its own (the one a pool leaves beside it).
            from: kvl.From = .none,
        };

        const Ops = mtp.Ops(Tri);

        pub const Cuda = struct {
            gpa: std.mem.Allocator,
            context: usize,
            engine: *prompt.Engine,
            vision: ?*@import("cuda_images.zig").Vision = null,
            ranks: ?*ranks.Session = null,
            head: ?*mtp.Head = null,
            depth: usize = mtp.default_depth,
            confidence: f64 = mtp.default_confidence,
            /// A captured Triton set (TENSORFOLD_CUDA_KERNELS) for comparison, else null: our own glue.
            set: ?cuda.aot.Set = null,
            glue: Glue,
            stream: cuda.Stream,
            lanes: std.AutoHashMapUnmanaged(*lanes.Stream, Lane) = .empty,
            /// A round's scores, every row [round_streams, vocab] bf16.
            logits: cuda.DeviceBuffer,
            drawn: [ring]u32 = undefined,
            next: u64 = 0,
            /// --slide: the pocket's layout, from config.json at load.
            lay: living.Layout = undefined,
            kv: kvl.State = .{},

            pub fn tri(self: *Cuda) Tri {
                return .{ .set = if (self.set) |*x| x else null, .own = &self.glue, .s = self.stream };
            }

            pub fn ops(self: *Cuda) Ops {
                return .{ .e = self.engine, .tri = self.tri(), .h = self.head.?, .main = if (self.ranks != null) .{
                    .ptr = self,
                    .forward = mainForward,
                    .verify = mainVerify,
                    .rollback = mainRollback,
                } else null };
            }

            fn mainForward(ptr: *anyopaque, q: *prompt.Seq, ids: []const u32, hooks: prompt.batch.Hooks) anyerror!void {
                const self = of(ptr);
                return self.ranks.?.forward(Tri, self.tri(), q, ids, hooks);
            }

            fn mainVerify(ptr: *anyopaque, q: *prompt.Seq, d: *mtp.Draft, ids: []const u32) anyerror!void {
                const self = of(ptr);
                return self.ranks.?.verify(Tri, self.tri(), q, d, ids);
            }

            fn mainRollback(ptr: *anyopaque, q: *prompt.Seq, d: *mtp.Draft, keep: usize) anyerror!void {
                const self = of(ptr);
                return self.ranks.?.keep(Tri, self.tri(), q, d, keep);
            }

            pub fn backend(self: *Cuda) be.Backend {
                const scoring_table: be.Backend.VTable = .{
                    .prefill = prefillFn,
                    .prefill_step = prefillStepFn,
                    .cancel_prefill = cancelPrefillFn,
                    .first = firstFn,
                    .queue = queueFn,
                    .read = readFn,
                    .verify = verifyFn,
                    .keep = keepFn,
                    .draft = draftFn,
                    .release = releaseFn,
                    .tree = treeFn,
                    .probabilities = probabilitiesFn,
                    .score = @import("cuda_score.zig").Hook(Cuda, Tri).call,
                    .first_masked = firstMaskedFn,
                };
                const ranked_table = @import("cuda_score.zig").withoutScore(scoring_table);
                return .{ .ptr = self, .proposal_head = if (self.head) |h| &h.service else null, .vtable = if (self.ranks == null) &scoring_table else &ranked_table };
            }

            /// With the head: chains of up to 16 rows cut by its confidence, copies of up to 32; else one-row rounds.
            pub fn facts(self: *const Cuda) lanes.Model {
                const drafting = self.head != null;
                return .{
                    .exact_width = if (drafting) mtp.max_rows else 2,
                    .first_copy_rows = if (drafting) mtp.max_chain + 1 else 0,
                    .gpu_tokens = false,
                    .mtp = drafting,
                    .speculate = drafting,
                    .speculate_early = false,
                    .drafts = if (drafting) @intCast(self.depth) else 0,
                    .draft_probabilities = drafting,
                    .streams_exact = true,
                    .hidden_rows = true,
                    .batch_rows = if (drafting) mtp.round_rows else round_streams,
                    .max_streams = round_streams,
                    // a shared round's streams draft in one head pass a step (lanes_draft.batched)
                    .draft_streams = drafting and lanes_draft.batchHeads(),
                };
            }

            fn take(self: *Cuda, token: u32) u64 {
                const h = self.next;
                self.drawn[h % ring] = token;
                self.next += 1;
                return h;
            }

            pub fn value(self: *const Cuda, feed: be.Feed) u32 {
                return switch (feed) {
                    .handle => |h| self.drawn[h % ring],
                    .value => |v| v,
                };
            }

            /// The token after seq.logits' row at `position`, within `mask` if any: the draw serial decoding makes.
            fn tokenAt(self: *Cuda, seq: *prompt.Seq, s: *lanes.Stream, position: u64, mask: []const u32) !u32 {
                if (s.sampling == null and mask.len == 0) return prompt.argmax(seq.logits);
                const row = try self.gpa.alloc(u16, prompt.vocabulary);
                defer self.gpa.free(row);
                for (row, 0..) |*v, i| v.* = std.mem.readInt(u16, seq.logits[2 * i ..][0..2], .little);
                if (mask.len > 0) mtp.maskRow(row, mask);
                return mtp.drawRow(self.gpa, row, null, position, s.sampling);
            }

            pub fn lane(self: *Cuda, s: *lanes.Stream) !*Lane {
                return self.lanes.getPtr(s) orelse return error.UnknownStream;
            }

            /// The stream's caches, from the kept state the cache names on a restore; its prompt pass starts there.
            fn open(self: *Cuda, s: *lanes.Stream) !*Lane {
                const ids = s.prompt();
                if (ids.len == 0 or ids.len + s.max_new > self.context) return error.PromptTooLong;
                const gop = try self.lanes.getOrPut(self.gpa, s);
                if (gop.found_existing) dropLane(self, gop.value_ptr);
                const drafting = self.head != null and s.drafts;
                // without the head's drafts, one row a stream a round: only a drafted window can rewind its states
                if (!drafting) {
                    s.drafts = false;
                    s.proposer = null;
                }
                // a window's rows past the reply's last token (its drafts) still need cache slots
                const cap = @min(ids.len + s.max_new + (if (drafting) mtp.max_rows else 0), prompt.max_tokens);
                const from = self.kv.choose();
                gop.value_ptr.* = .{ .seq = (if (self.ranks) |r| r.create(cap) else if (from == .pool) self.kv.pool.?.open(cap) else self.engine.newSeq(cap)) catch |err| {
                    self.lanes.removeByPtr(gop.key_ptr);
                    return err;
                }, .from = from };
                self.kv.opened(from);
                const l = gop.value_ptr;
                l.seq.untaught = s.untaught;
                errdefer {
                    dropLane(self, l);
                    self.lanes.removeByPtr(gop.key_ptr);
                }
                if (drafting) l.draft = if (from == .pool) try self.kv.pool.?.draft(&l.seq, cap) else try mtp.Draft.init(self.gpa, self.engine.driver, self.stream, cap, self.head.?.ids.len, self.engine.chunk);
                if (s.images.len != 0) {
                    const v = self.vision orelse return error.VisionNotEnabled;
                    l.seq.media = try v.state(ids, s.images);
                    if (self.ranks) |r| try r.setMedia(&l.seq);
                }
                if (l.draft) |*d| d.media = if (l.seq.media) |*m| m.view() else null;

                s.cached = 0;
                const saved = s.reuse.saved orelse return l;
                const kept: *snap.Snap = @ptrCast(@alignCast(saved));
                if (kept.at != s.reuse.at or kept.at == 0 or kept.at >= ids.len) {
                    s.reuse_failed = true;
                } else if (self.ranks != null and l.draft != null) {
                    s.reuse_failed = true;
                } else if (if (self.ranks) |r| r.restore(&l.seq, kept) else snap.restore(self.stream, &l.seq, if (l.draft) |*d| d else null, kept)) |_| {
                    l.filled = kept.at;
                    s.cached = @intCast(kept.at);
                } else |err| switch (err) {
                    error.SequenceTooShort, error.NoHeadState => s.reuse_failed = true,
                    else => return err,
                }
                return l;
            }
            /// One prompt chunk, cut at the next cache mark; true once the last row's scores are in seq.logits.
            fn step(self: *Cuda, s: *lanes.Stream, l: *Lane) !bool {
                const ids = s.prompt();
                const at = l.filled orelse return true;
                if (s.isCancelled()) {
                    s.keepInterruptedPrefill(@intCast(at));
                    return error.Cancelled;
                }
                var stop = @min(ids.len, at + self.engine.chunk);
                for (s.reuse.marks) |m| if (m > at and m < stop) {
                    stop = m;
                    break;
                };
                const last = stop == ids.len;
                self.engine.ngram.warmAhead(ids, stop, @min(ids.len, stop + self.engine.chunk));
                if (l.draft) |*d| {
                    try self.ops().prefillChunk(&l.seq, d, ids, at, stop);
                } else if (self.ranks) |r| {
                    try r.forward(Tri, self.tri(), &l.seq, ids[at..stop], .{ .logits = if (last) .last else .none });
                } else try self.engine.forward(Tri, self.tri(), &l.seq, ids[at..stop], .{ .logits = if (last) .last else .none });
                l.filled = if (last) null else stop;
                if (!last and std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(stop)) != null) {
                    if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(stop));
                }
                return last;
            }
        };
        pub fn envUsize(name: [:0]const u8, default: usize) usize {
            const v = std.c.getenv(name) orelse return default;
            return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
        }

        pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, captured: ?[]const u8, o: Options) !Loaded {
            return @import("cuda_lanes_load.zig").open(@This(), gpa, io, ctx, dir, captured, o);
        }

        /// The captured set at `dir` when it holds every entry point (the head's too when drafting), else null.
        pub fn comparison(gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, ctx: *const cuda.Context, dir: []const u8, drafts: bool) !?cuda.aot.Set {
            return @import("cuda_lanes_load.zig").comparison(gpa, io, driver, ctx, dir, drafts, &captured_entries, &captured_head_entries);
        }

        pub fn follow(ptr: *anyopaque) anyerror!void {
            const self = of(ptr);
            return self.ranks.?.follow(Tri, self.tri(), self.logits.ptr);
        }

        pub fn release(p: *anyopaque) void {
            const own: *Cuda = @ptrCast(@alignCast(p));
            cuda.completion.drain(own.engine.driver, own.stream, "flashnext: release did not drain");
            var it = own.lanes.valueIterator();
            while (it.next()) |l| dropLane(own, l);
            own.lanes.deinit(own.gpa);
            own.engine.round_graph.clear();
            own.logits.free();
            if (own.head) |h| h.deinit();
            if (own.vision) |v| v.deinit();
            if (own.ranks) |r| r.deinit();
            if (own.kv.pool) |kp| kp.deinit();
            own.engine.deinit();
            if (own.set) |*x| x.deinit();
            own.glue.deinit();
            own.stream.deinit();
            own.gpa.destroy(own);
        }

        fn dropLane(self: *Cuda, l: *Lane) void {
            cuda.completion.drain(self.engine.driver, self.stream, "flashnext: a failed or closed lane did not drain");
            if (l.draft) |*d| d.deinit();
            if (self.ranks) |r| r.drop(&l.seq) else if (!self.kv.close(l.from, &l.seq)) l.seq.deinit(self.gpa);
        }

        fn of(ptr: *anyopaque) *Cuda {
            return @ptrCast(@alignCast(ptr));
        }

        fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            const self = of(ptr);
            const l = try self.open(s);
            while (!try self.step(s, l)) {}
        }

        /// One chunk of the stream's prompt between the other streams' rounds; the first call opens its caches.
        fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
            const self = of(ptr);
            const l = if (self.lanes.getPtr(s)) |x| (if (x.filled != null) x else try self.open(s)) else try self.open(s);
            return self.step(s, l);
        }

        fn cancelPrefillFn(ptr: *anyopaque, s: *lanes.Stream) !void {
            if (of(ptr).lanes.getPtr(s)) |l| if (l.filled) |at| s.keepInterruptedPrefill(@intCast(at));
        }

        fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
            const self = of(ptr);
            const l = try self.lane(s);
            if (l.filled != null or position != l.seq.fed) return error.PositionMismatch;
            return self.take(try self.tokenAt(&l.seq, s, position, &.{}));
        }

        /// The first token drawn among `mask`'s allowed tokens (structured output), from the prompt's last row.
        fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
            const self = of(ptr);
            const l = try self.lane(s);
            if (l.filled != null or position != l.seq.fed) return error.PositionMismatch;
            return self.take(try self.tokenAt(&l.seq, s, position, mask));
        }

        fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
            _ = ptr;
            _ = s;
            _ = feed;
            _ = position;
            return error.NotPipelined;
        }

        fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
            const self = of(ptr);
            if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
            return self.drawn[handle % ring];
        }

        /// A drafting stream's window in one batched pass; streams without drafts share rounds, each row its own bits.
        fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const self = of(ptr);
            const seqs = try self.gpa.alloc(*prompt.Seq, windows.len);
            defer self.gpa.free(seqs);
            const ids = try self.gpa.alloc(u32, windows.len);
            defer self.gpa.free(ids);
            const at_window = try self.gpa.alloc(usize, windows.len);
            defer self.gpa.free(at_window);
            const row_ids = try self.gpa.alloc([mtp.max_rows]u32, windows.len);
            defer self.gpa.free(row_ids);
            const drafted = try self.gpa.alloc(usize, windows.len);
            defer self.gpa.free(drafted);
            var plain: usize = 0;
            var held: usize = 0;
            for (windows, out, 0..) |w, *o, wi| {
                if (w.held != 0 or w.parents != null) return error.HeldDraftsNotBuilt;
                const l = try self.lane(w.stream);
                const rows = w.rows();
                if (l.draft != null) {
                    if (rows > mtp.max_rows) return error.WindowTooWide;
                    for (w.positions, 0..) |p, r| if (p != l.seq.fed + 1 + r) return error.PositionMismatch;
                    row_ids[wi][0] = w.pending;
                    @memcpy(row_ids[wi][1..rows], w.tokens);
                    drafted[held] = wi;
                    held += 1;
                } else {
                    if (rows != 1) return error.CannotRewind;
                    if (w.positions[0] != l.seq.fed + 1) return error.PositionMismatch;
                    seqs[plain] = &l.seq;
                    ids[plain] = w.pending;
                    at_window[plain] = wi;
                    plain += 1;
                }
                if (w.tokens.len > 0) @memcpy(o.drafts[0..w.tokens.len], w.tokens);
            }
            const cap = @min(@min(envUsize("TF_FLASHNEXT_WINDOW_ROWS", mtp.round_rows), mtp.round_rows), self.engine.chunk);
            try @import("cuda_lanes_verify.zig").windows(self, windows, out, drafted[0..held], row_ids, cap);
            var at: usize = 0;
            while (at < plain) : (at += round_streams) {
                const n = @min(round_streams, plain - at);
                if (self.ranks) |r| try r.round(Tri, self.tri(), seqs[at..][0..n], ids[at..][0..n], self.logits.ptr) else try self.engine.round(Tri, self.tri(), seqs[at..][0..n], ids[at..][0..n], self.logits.ptr);
            }
            for (seqs[0..plain], at_window[0..plain]) |q, wi| out[wi].sampled[0] = try self.tokenAt(q, windows[wi].stream, windows[wi].positions[0], windows[wi].masks);
        }

        /// Each window keeps its first rows (a chain's prefix): the DeltaNet states replayed over them alone.
        fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            const self = of(ptr);
            for (windows, paths) |w, path| {
                if (path.len == 0 or path.len > w.rows()) return error.CannotRewind;
                for (path, 0..) |r, i| if (r != i) return error.CannotRewind;
                const l = try self.lane(w.stream);
                if (l.draft) |*d| try self.ops().rollback(&l.seq, d, path.len) else if (path.len != w.rows()) return error.CannotRewind;
            }
        }

        /// The head absorbs the kept rows and chains the next window's drafts (Python decode.draft).
        fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
            const self = of(ptr);
            if (self.head == null) return error.NoDraftHead;
            defer lanes_draft.depthLog(self, requests);
            // a shared round's chains: their heads in one pass a step (the drafts each would get alone)
            var batched: [mtp_ops.max_lanes]bool = @splat(false);
            if (lanes_draft.batchHeads() and requests.len > 1 and requests.len <= mtp_ops.max_lanes) try lanes_draft.batched(self, requests, &batched);
            for (requests, 0..) |r, ri| {
                if (ri < batched.len and batched[ri]) continue;
                if (r.lanes != null) return error.TreesNotBuilt;
                const l = try self.lane(r.stream);
                const d = if (l.draft) |*x| x else return error.NoDraftHead;
                // the window's kept rows: rollback is idempotent, so a later keep finds it done
                if (r.rows) |path| try self.ops().rollback(&l.seq, d, path.len) else {
                    if (d.open) try self.ops().rollback(&l.seq, d, d.rows);
                }
                if (r.position != l.seq.fed + 1) return error.PositionMismatch;
                // the window's rows past the cache's room cannot run: fewer drafts there
                const room = l.seq.cap - l.seq.fed - 1;
                const count = @min(@min(r.depth, self.depth), room);
                if (r.rows) |path| {
                    for (path, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                    if (r.follow.len < path.len) return error.PositionMismatch;
                    try self.ops().draft(d, r.follow[0..path.len], d.streams.ptr, r.position, count, r.stream.sampling, self.confidence, &l.chain);
                } else {
                    const first = self.value(r.first orelse return error.NoFirstToken);
                    try self.ops().draft(d, &.{first}, d.prompt_last, r.position, count, r.stream.sampling, self.confidence, &l.chain);
                }
            }
        }

        /// The held chain as host tokens: the round loop verifies them as it would copied ones.
        fn treeFn(ptr: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
            const self = of(ptr);
            const l = try self.lane(s);
            if (l.draft == null) return null;
            return .{ .count = @intCast(l.chain.n), .tokens = try gpa.dupe(u32, l.chain.slice()) };
        }

        /// Each held draft's temperature-1 probability under the head.
        fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
            const self = of(ptr);
            const l = try self.lane(s);
            if (l.draft == null or out.len > l.chain.n) return false;
            @memcpy(out, l.chain.probs[0..out.len]);
            return true;
        }

        fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
            const self = of(ptr);
            if (self.lanes.fetchRemove(s)) |kv| {
                var l = kv.value;
                // its rounds drain first, or the process ends: a failed wait never frees what the GPU may still use
                @import("qwen_vision").completion.drain(self.engine.driver, self.stream);
                dropLane(self, &l);
            }
        }

        test {
            _ = @import("cuda_images_test.zig").Cases(Cuda, Cuda.open, dropLane);
            _ = @import("cuda_lanes_restore_test.zig").Cases(Cuda, Cuda.open, snapRestore, dropLane);
            _ = @import("cuda_lanes_admission_test.zig").Cases(Cuda, Cuda.open, dropLane);
            _ = @import("cuda_score.zig").CapabilityCases(Cuda);
        }

        test "a bad TF_FLASHNEXT_CHUNK or TF_FLASHNEXT_LOAD_THREADS is refused at load before the loader takes a stream, a module or a buffer" {
            const env = @import("sys").env;
            // every driver entry point undefined: the refusal must come before the first call
            var driver: cuda.Driver = undefined;
            const ctx: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
            for ([_]struct { [*:0]const u8, [*:0]const u8, anyerror }{ .{ "TF_FLASHNEXT_CHUNK", "0", error.BadChunk }, .{ "TF_FLASHNEXT_CHUNK", "16385", error.BadChunk }, .{ "TF_FLASHNEXT_CHUNK", "4k", error.BadChunk }, .{ "TF_FLASHNEXT_LOAD_THREADS", "0", error.BadLoadThreads }, .{ "TF_FLASHNEXT_LOAD_THREADS", "33", error.BadLoadThreads }, .{ "TF_FLASHNEXT_LOAD_THREADS", "8x", error.BadLoadThreads } }) |bad| {
                _ = env.setenv(bad[0], bad[1], 1);
                defer _ = env.unsetenv(bad[0]);
                try std.testing.expectError(bad[2], open(std.testing.allocator, std.testing.io, &ctx, "no-such-checkpoint", null, .{ .context = 4096, .drafts = true }));
            }
        }

        // -- the prompt cache's copies ------------------------------------------------------------------------

        fn snapBytes(ptr: *anyopaque, at: u32) u64 {
            return snap.bytes(at, of(ptr).head != null);
        }

        /// The owner stream's state, its prompt pass standing at `at` (a drafting stream's with the head's rows).
        fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
            const self = of(ptr);
            const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
            const l = try self.lane(s);
            if (l.seq.fed != at) return error.NotAtMark;
            if (self.ranks) |r| {
                if (l.draft != null) return error.TensorRankDraftSnapshotUnavailable;
                return try r.save(&l.seq);
            }
            return try snap.save(self.gpa, self.engine.driver, self.stream, &l.seq, if (l.draft) |*d| d else null);
        }

        fn snapRestore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
            const self = of(ptr);
            const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
            const l = try self.lane(s);
            if (self.ranks) |r| {
                if (l.draft != null) return error.NoHeadState;
                return r.restore(&l.seq, @ptrCast(@alignCast(saved)));
            }
            try snap.restore(self.stream, &l.seq, if (l.draft) |*d| d else null, @ptrCast(@alignCast(saved)));
        }

        fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
            const self = of(ptr);
            if (self.ranks) |r| return r.dropSaved(@ptrCast(@alignCast(saved)));
            snap.drop(self.gpa, self.stream, @ptrCast(@alignCast(saved)));
        }
    };
}
