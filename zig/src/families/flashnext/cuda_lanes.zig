//! Flash Next's lane backend on the native CUDA server: chunked prompts, MTP draft windows, and shared rounds.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const weights = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");
const snap = @import("cuda_snap.zig");
const ranks = @import("cuda_rank_engine.zig");
const config = @import("cuda_config.zig");
const Glue = @import("cuda_glue.zig").Glue;

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
        pub const snap_save = snapSave;
        pub const snap_restore = snapRestore;
        pub const snap_drop = snapDrop;
        /// The pass ignores its cuts, so states keep at any token; the MTP head reads one token on (lookahead 1).
        pub const cache_rules = struct {
            pub const lookahead: u32 = 1;
            pub const planned = false;
            pub const min_gap: u32 = 256;
            pub const min_prompt: u32 = 512;
        };

        pub const Options = struct { vision: ?@import("cuda_images.zig").Limits = null, context: usize, drafts: bool, segments: usize = 1, slide: bool = false, tp: usize = 1, rank: usize = 0, master: ?[]const u8 = null, master_port: u16 = 29551 };

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
        };

        /// Device bytes of the checkpoint: the n-gram tables stay mapped, the MTP head counts; 0 if unread.
        pub fn deviceWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) u64 {
            const main = weights.deviceBytes(gpa, io, dir) catch return 0;
            return main + mtp.headBytes();
        }

        pub fn placementWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize, rank: usize, drafts: bool) !u64 {
            if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
            const mode = try config.mode(gpa, io, dir);
            if (mode == .modelopt) {
                const main = try @import("cuda_rank_memory.zig").estimate(gpa, io, dir, world, rank);
                // Dense draft projections and vocabulary rows exceed the affine head's estimate.
                return main + if (drafts and rank == 0) mtp.headBytes() + (1 << 30) else @as(u64, 0);
            }
            if (world != 1) return error.UnsupportedQuantization;
            const main = try weights.deviceBytes(gpa, io, dir);
            return main + if (drafts) mtp.headBytes() else @as(u64, 0);
        }

        /// The tower's weights and one call's scratch, which rank zero holds beside the text weights with --vision.
        pub fn visionBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
            const v = @import("qwen_vision");
            const c = ((try v.Settings.read(a, io, dir)) orelse return error.NoVisionTower).tower;
            return v.Tower.weightBytes(c) + v.Tower.scratchBytes(c, v.max_patches);
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
            return switch (err) {
                error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
                error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
                error.UnsupportedCheckpoint => "the checkpoint differs from the supported Flash Next CUDA layout; the log names its config field",
                error.TensorRankDraftSnapshotUnavailable => "tensor-rank MTP snapshots are unavailable: disable the prompt cache for this configuration",
                error.RankDisconnected, error.RankTimeout, error.RemoteRank, error.BadRankFrame => "a tensor-parallel rank failed; restart both ranks from the same checkpoint before retrying",
                error.LearnedPocket => "this folder holds Living Weights pockets, which Flash Next on CUDA does not serve yet",
                else => @import("cuda_images.zig").Vision.explain(err),
            };
        }

        /// A stream's caches; with the head, its cache, a window's buffers and the drafts it holds for the next round.
        const Lane = struct {
            seq: prompt.Seq,
            draft: ?mtp.Draft = null,
            chain: mtp.Chain = .{},
            /// The prompt's next row while its pass runs a chunk a call; null once the prompt is in.
            filled: ?usize = 0,
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

            fn tri(self: *Cuda) Tri {
                return .{ .set = if (self.set) |*x| x else null, .own = &self.glue, .s = self.stream };
            }

            fn ops(self: *Cuda) Ops {
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
                return .{ .ptr = self, .vtable = &.{
                    .prefill = prefillFn,
                    .prefill_step = prefillStepFn,
                    .first = firstFn,
                    .queue = queueFn,
                    .read = readFn,
                    .verify = verifyFn,
                    .keep = keepFn,
                    .draft = draftFn,
                    .release = releaseFn,
                    .tree = treeFn,
                    .probabilities = probabilitiesFn,
                } };
            }

            /// With the head: windows of up to 16 rows cut by its confidence; else shared one-row rounds.
            pub fn facts(self: *const Cuda) lanes.Model {
                const drafting = self.head != null;
                return .{
                    .exact_width = if (drafting) mtp.max_rows else 2,
                    .gpu_tokens = false,
                    .mtp = drafting,
                    .speculate = drafting,
                    .speculate_early = false,
                    .drafts = if (drafting) @intCast(self.depth) else 0,
                    .draft_probabilities = drafting,
                    .streams_exact = true,
                    .hidden_rows = true,
                    .batch_rows = if (drafting) 4 * mtp.max_rows else round_streams,
                    .max_streams = round_streams,
                };
            }

            fn take(self: *Cuda, token: u32) u64 {
                const h = self.next;
                self.drawn[h % ring] = token;
                self.next += 1;
                return h;
            }

            fn value(self: *const Cuda, feed: be.Feed) u32 {
                return switch (feed) {
                    .handle => |h| self.drawn[h % ring],
                    .value => |v| v,
                };
            }

            /// The token after the sequence's last row (seq.logits) at `position`: the draws serial decoding makes.
            fn tokenAt(self: *Cuda, seq: *prompt.Seq, s: *lanes.Stream, position: u64) !u32 {
                if (s.sampling == null) return prompt.argmax(seq.logits);
                const row = try self.gpa.alloc(u16, prompt.vocabulary);
                defer self.gpa.free(row);
                for (row, 0..) |*v, i| v.* = std.mem.readInt(u16, seq.logits[2 * i ..][0..2], .little);
                return mtp.drawRow(self.gpa, row, null, position, s.sampling);
            }

            fn lane(self: *Cuda, s: *lanes.Stream) !*Lane {
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
                gop.value_ptr.* = .{ .seq = (if (self.ranks) |r| r.create(cap) else self.engine.newSeq(cap)) catch |err| {
                    self.lanes.removeByPtr(gop.key_ptr);
                    return err;
                } };
                const l = gop.value_ptr;
                errdefer {
                    dropLane(self, l);
                    self.lanes.removeByPtr(gop.key_ptr);
                }
                if (drafting) l.draft = try mtp.Draft.init(self.gpa, self.engine.driver, self.stream, cap, self.head.?.ids.len, self.engine.chunk);
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
                if (s.isCancelled()) return error.Cancelled;
                var stop = @min(ids.len, at + self.engine.chunk);
                for (s.reuse.marks) |m| if (m > at and m < stop) {
                    stop = m;
                    break;
                };
                const last = stop == ids.len;
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
            var set = try cuda.aot.Set.load(gpa, io, driver, ctx.device, dir);
            const holds = for (captured_entries) |name| {
                if (!set.has(name)) break false;
            } else for (captured_head_entries) |name| {
                if (drafts and !set.has(name)) break false;
            } else true;
            if (holds) return set;
            set.deinit();
            std.log.info("flashnext: the captured set at {s} lacks Flash Next's kernels; its own glue runs", .{dir});
            return null;
        }

        pub fn follow(ptr: *anyopaque) anyerror!void {
            const self = of(ptr);
            return self.ranks.?.follow(Tri, self.tri(), self.logits.ptr);
        }

        pub fn release(p: *anyopaque) void {
            const own: *Cuda = @ptrCast(@alignCast(p));
            var it = own.lanes.valueIterator();
            while (it.next()) |l| dropLane(own, l);
            own.lanes.deinit(own.gpa);
            own.engine.round_graph.clear();
            own.logits.free();
            if (own.head) |h| h.deinit();
            if (own.vision) |v| v.deinit();
            if (own.ranks) |r| r.deinit();
            own.engine.deinit();
            if (own.set) |*x| x.deinit();
            own.glue.deinit();
            own.stream.deinit();
            own.gpa.destroy(own);
        }

        fn dropLane(self: *Cuda, l: *Lane) void {
            if (l.draft) |*d| d.deinit();
            if (self.ranks) |r| r.drop(&l.seq) else l.seq.deinit(self.gpa);
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

        fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
            const self = of(ptr);
            const l = try self.lane(s);
            if (l.filled != null or position != l.seq.fed) return error.PositionMismatch;
            return self.take(try self.tokenAt(&l.seq, s, position));
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
            var plain: usize = 0;
            for (windows, out, 0..) |w, *o, wi| {
                if (w.held != 0 or w.parents != null) return error.HeldDraftsNotBuilt;
                const l = try self.lane(w.stream);
                const rows = w.rows();
                if (l.draft) |*d| {
                    if (rows > mtp.max_rows) return error.WindowTooWide;
                    for (w.positions, 0..) |p, r| if (p != l.seq.fed + 1 + r) return error.PositionMismatch;
                    var row_ids: [mtp.max_rows]u32 = undefined;
                    row_ids[0] = w.pending;
                    @memcpy(row_ids[1..rows], w.tokens);
                    try self.ops().verify(&l.seq, d, row_ids[0..rows]);
                    try self.ops().draws(d, w.stream.sampling, o.sampled[0..rows]);
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
            var at: usize = 0;
            while (at < plain) : (at += round_streams) {
                const n = @min(round_streams, plain - at);
                if (self.ranks) |r| try r.round(Tri, self.tri(), seqs[at..][0..n], ids[at..][0..n], self.logits.ptr) else try self.engine.round(Tri, self.tri(), seqs[at..][0..n], ids[at..][0..n], self.logits.ptr);
            }
            for (seqs[0..plain], at_window[0..plain]) |q, wi| out[wi].sampled[0] = try self.tokenAt(q, windows[wi].stream, windows[wi].positions[0]);
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
            for (requests) |r| {
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
