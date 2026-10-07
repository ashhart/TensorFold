//! Flash Next on the native CUDA server: one stream's prompt, then one token at a time. No draft head.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");

const be = lanes.backend;
const ring = 1024;

/// `Tri` is the captured-kernel launcher. `open` loads the checkpoint and serves greedy or keyed draws.
pub fn Namespace(comptime Tri: type) type {
    return struct {
        pub const model_type = "qwen4_exp";
        pub const formats: []const []const u8 = &.{"mlx-q4g32"};
        pub const default_context: i64 = 262144;
        pub const prefill_step: u32 = 0;

        pub const Options = struct { context: usize, drafts: bool };

        pub const Loaded = struct {
            backend: be.Backend,
            facts: lanes.Model,
            rows: u32,
            ctx: *anyopaque,
            deinit: *const fn (*anyopaque) void,
        };

        const Lane = struct { seq: prompt.Seq };

        const Cuda = struct {
            gpa: std.mem.Allocator,
            context: usize,
            engine: *prompt.Engine,
            set: cuda.aot.Set,
            stream: cuda.Stream,
            lanes: std.AutoHashMapUnmanaged(*lanes.Stream, Lane) = .empty,
            drawn: [ring]u32 = undefined,
            next: u64 = 0,

            fn tri(self: *Cuda) Tri {
                return .{ .set = &self.set, .s = self.stream };
            }

            fn backend(self: *Cuda) be.Backend {
                return .{ .ptr = self, .vtable = &.{
                    .prefill = prefillFn,
                    .first = firstFn,
                    .queue = queueFn,
                    .read = readFn,
                    .verify = verifyFn,
                    .keep = keepFn,
                    .draft = draftFn,
                    .release = releaseFn,
                } };
            }

            fn facts() lanes.Model {
                return .{
                    .exact_width = 1,
                    .gpu_tokens = false,
                    .mtp = false,
                    .speculate = false,
                    .drafts = 0,
                    .batch_rows = 1,
                    .max_streams = 1,
                };
            }

            fn take(self: *Cuda, token: u32) u64 {
                const h = self.next;
                self.drawn[h % ring] = token;
                self.next += 1;
                return h;
            }

            fn tokenAt(self: *Cuda, seq: *prompt.Seq, s: *lanes.Stream, position: u64) !u32 {
                if (s.sampling) |sm| {
                    const n = prompt.vocabulary;
                    const vals = try self.gpa.alloc(f64, n);
                    defer self.gpa.free(vals);
                    const ids = try self.gpa.alloc(u64, n);
                    defer self.gpa.free(ids);
                    for (vals, ids, 0..) |*v, *id, i| {
                        const bits = std.mem.readInt(u16, seq.logits[2 * i ..][0..2], .little);
                        v.* = @floatCast(@as(f32, @bitCast(@as(u32, bits) << 16)));
                        id.* = @intCast(i);
                    }
                    return @intCast(try lanes.sampling.choose(self.gpa, vals, ids, position, sm));
                }
                return prompt.argmax(seq.logits);
            }

            fn lane(self: *Cuda, s: *lanes.Stream) !*Lane {
                return self.lanes.getPtr(s) orelse return error.UnknownStream;
            }
        };

        pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
            _ = o.drafts; // this model is served one token at a time; there is no draft head
            const driver: *cuda.Driver = @constCast(ctx.d);
            const own = try gpa.create(Cuda);
            errdefer gpa.destroy(own);
            own.gpa = gpa;
            own.context = o.context;
            own.stream = try cuda.Stream.init(driver, true);
            errdefer own.stream.deinit();
            own.set = try cuda.aot.Set.load(gpa, io, driver, ctx.device, kernels);
            errdefer own.set.deinit();
            own.engine = try prompt.Engine.init(gpa, io, driver, &own.stream, dir);
            own.lanes = .empty;
            own.next = 0;
            return .{
                .backend = own.backend(),
                .facts = Cuda.facts(),
                .rows = 1,
                .ctx = own,
                .deinit = release,
            };
        }

        fn release(p: *anyopaque) void {
            const own: *Cuda = @ptrCast(@alignCast(p));
            var it = own.lanes.valueIterator();
            while (it.next()) |l| l.seq.deinit(own.gpa);
            own.lanes.deinit(own.gpa);
            own.engine.deinit();
            own.set.deinit();
            own.stream.deinit();
            own.gpa.destroy(own);
        }

        fn of(ptr: *anyopaque) *Cuda {
            return @ptrCast(@alignCast(ptr));
        }

        fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            const self = of(ptr);
            const ids = s.prompt();
            if (ids.len == 0 or ids.len + s.max_new > self.context) return error.PromptTooLong;
            const gop = try self.lanes.getOrPut(self.gpa, s);
            if (gop.found_existing) gop.value_ptr.seq.deinit(self.gpa);
            gop.value_ptr.* = .{ .seq = self.engine.newSeq(ids.len + s.max_new) catch |err| {
                self.lanes.removeByPtr(gop.key_ptr);
                return err;
            } };
            self.engine.predict(Tri, self.tri(), &gop.value_ptr.seq, ids) catch |err| {
                gop.value_ptr.seq.deinit(self.gpa);
                self.lanes.removeByPtr(gop.key_ptr);
                return err;
            };
        }

        fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
            const self = of(ptr);
            const l = try self.lane(s);
            if (position != l.seq.fed) return error.PositionMismatch;
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

        fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const self = of(ptr);
            for (windows, out) |w, *o| {
                if (w.held != 0 or w.parents != null) return error.HeldDraftsNotBuilt;
                const l = try self.lane(w.stream);
                const rows = w.rows();
                var input = w.pending;
                for (0..rows) |r| {
                    if (w.positions[r] != l.seq.fed + 1) return error.PositionMismatch;
                    var one = [_]u32{input};
                    try self.engine.predict(Tri, self.tri(), &l.seq, &one);
                    o.sampled[r] = try self.tokenAt(&l.seq, w.stream, w.positions[r]);
                    if (r + 1 < rows) input = w.tokens[r];
                }
                if (w.tokens.len > 0) @memcpy(o.drafts[0..w.tokens.len], w.tokens);
            }
        }

        fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            _ = ptr;
            for (windows, paths) |w, path| {
                if (path.len != w.rows()) return error.CannotRewind;
                for (path, 0..) |r, i| if (r != i) return error.CannotRewind;
            }
        }

        fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
            _ = ptr;
            _ = requests;
            return error.NoDraftHead;
        }

        fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
            const self = of(ptr);
            if (self.lanes.fetchRemove(s)) |kv| {
                var l = kv.value;
                l.seq.deinit(self.gpa);
            }
        }
    };
}
