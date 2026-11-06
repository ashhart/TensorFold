//! Kept states in device buffers on disk (--learn-conversations): one image, its copies, probe and identity facts.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const disk_identity = @import("disk_identity.zig");
const rows = api.prompt_cache.rows;
const Allocator = std.mem.Allocator;

/// What the prompt's tokens came from, beside the weights the checkpoint's stamps bind.
const prompt_files = [_][]const u8{
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "chat_template.jinja",
    "chat_template.json",
};

/// The words of a state that holds nothing but its position (and its family's Words otherwise).
const Position = extern struct { at: u64, pad: u64 = 0 };

/// The tokenizer's and template's bytes, then the drafter's checkpoint: what a family's image cannot show.
pub fn promptHash(a: Allocator, io: std.Io, dir: []const u8, drafter: ?[]const u8) !u64 {
    var h = std.hash.Wyhash.init(0x6b70);
    for (prompt_files) |name| {
        const path = try std.fs.path.join(a, &.{ dir, name });
        defer a.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28)) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => return e,
        };
        defer a.free(bytes);
        h.update(name);
        h.update(bytes);
    }
    if (drafter) |d| try disk_identity.checkpoint(&h, a, io, d);
    return h.final();
}

/// The disk tier of a family's `View`: its target, state type, buffers, parts, blank and facts (kept_view).
pub fn Kept(comptime V: type) type {
    return struct {
        const Target = V.Target;
        const State = V.State;
        const Words = if (@hasDecl(V, "Words")) V.Words else Position;
        const words = @sizeOf(Words);

        pub const image: rows.Image = .{
            .parts = parts,
            .gather = gather,
            .blank = blank,
            .scatter = scatter,
            .whole = if (@hasDecl(V, "whole")) whole else null,
        };
        /// A family that splits over ranks keeps one rank's states: --tp refuses --learn-conversations.
        pub const one_rank = @hasDecl(V, "one_rank") and V.one_rank;

        fn of(ptr: *anyopaque) *Target {
            return @ptrCast(@alignCast(ptr));
        }

        fn driver(t: *Target) *const cuda.Driver {
            return if (@hasDecl(V, "driver")) V.driver(t) else t.d;
        }

        fn stream(t: *Target) cuda.Stream {
            return if (@hasDecl(V, "stream")) V.stream(t) else t.s;
        }

        fn wordsOf(st: *const State) Words {
            return if (@hasDecl(V, "wordsOf")) V.wordsOf(st) else .{ .at = st.at };
        }

        /// A state read back stands where it was kept: else it restores nowhere.
        fn setWords(st: *State, w: Words) !void {
            if (@hasDecl(V, "setWords")) return V.setWords(st, w);
            if (w.at != st.at) return error.KeptWordsDiffer;
        }

        const Parts = std.ArrayList(rows.image.Part);
        const Ranges = []const rows.image.Range;
        const Saved = api.prompt_cache.Saved;

        fn parts(ptr: *anyopaque, at: u32, out: *Parts, a: Allocator) anyerror!void {
            try out.append(a, .{ .bytes = words });
            try V.parts(of(ptr), at, out, a);
        }

        /// Image ranges between `host` and the state's buffers; `host` is the store's: every copy lands before return.
        fn copy(t: *Target, st: *State, ranges: Ranges, host: []u8, out: bool) !void {
            const d = driver(t);
            const s = stream(t);
            // a later range's failure leaves earlier copies queued on `host`, which its caller then frees or reuses
            errdefer cuda.completion.drain(d, s, "a failed kept-state copy did not drain");
            const bufs = V.buffers(st);
            const first: usize = if (bufs[0]) |b| b.len else 0;
            const held = first + if (bufs[1]) |b| b.len else 0;
            var w = wordsOf(st);
            var o: usize = 0;
            for (ranges) |g| {
                var off: usize = @intCast(g.off);
                var len: usize = @intCast(g.len);
                if (off + len > words + held) return error.BadKeptRange;
                if (off < words) {
                    const n = @min(len, words - off);
                    const bytes = std.mem.asBytes(&w)[off..][0..n];
                    if (out) @memcpy(host[o..][0..n], bytes) else @memcpy(bytes, host[o..][0..n]);
                    o += n;
                    off += n;
                    len -= n;
                }
                while (len > 0) {
                    const in = off - words;
                    const b = (if (in < first) bufs[0] else bufs[1]).?;
                    const at = if (in < first) in else in - first;
                    const n = @min(len, b.len - at);
                    const h = host[o..].ptr;
                    if (out) {
                        try d.check(d.api.cuMemcpyDtoHAsync_v2(h, b.ptr + at, n, s.handle), "cuMemcpyDtoHAsync");
                    } else try d.check(d.api.cuMemcpyHtoDAsync_v2(b.ptr + at, h, n, s.handle), "cuMemcpyHtoDAsync");
                    o += n;
                    off += n;
                    len -= n;
                }
            }
            try s.synchronize();
            if (!out) try setWords(st, w);
        }

        fn gather(ptr: *anyopaque, saved: Saved, ranges: Ranges, out: []u8) anyerror!void {
            try copy(of(ptr), @ptrCast(@alignCast(saved)), ranges, out, true);
        }

        fn scatter(ptr: *anyopaque, saved: Saved, ranges: Ranges, in: []const u8) anyerror!void {
            try copy(of(ptr), @ptrCast(@alignCast(saved)), ranges, @constCast(in), false);
        }

        fn blank(ptr: *anyopaque, at: u32) anyerror!Saved {
            return @ptrCast(try V.blank(of(ptr), at));
        }

        fn whole(ptr: *anyopaque, saved: Saved) bool {
            return V.whole(of(ptr), @ptrCast(@alignCast(saved)));
        }

        /// Whether a probe's state is one to hash (a family's draft context is absent from a probe, which drafts none).
        fn probed(t: *Target, st: *const State) bool {
            if (@hasDecl(V, "probed")) return V.probed(t, st);
            return if (@hasDecl(V, "whole")) V.whole(t, st) else true;
        }

        /// A fixed prompt's state at a mark inside its last chunk, hashed, and the prompt rate it took.
        pub fn probe(a: Allocator, io: std.Io, backend: anytype, ptr: anytype) !struct { hash: u64, rate: f64 } {
            const t: *Target = @ptrCast(@alignCast(ptr)); // Flash Next's is opaque
            var toks: [4096 + 40 + 7]u32 = undefined;
            for (&toks, 0..) |*x, i| x.* = @intCast(1000 + i * 7919 % 50000);
            const count = if (@hasDecl(V, "context")) @min(toks.len, V.context(t) / 2) else toks.len;
            const Hook = struct {
                t: *Target,
                saved: ?*State = null,
                failed: ?anyerror = null,
                fn at(p: *anyopaque, s: *lanes.Stream, mark: u32) void {
                    const h: *@This() = @ptrCast(@alignCast(p));
                    h.saved = V.save(h.t, s, mark) catch |e| {
                        h.failed = e;
                        return;
                    };
                }
            };
            var hook: Hook = .{ .t = t };
            defer if (hook.saved) |saved| V.drop(t, saved);
            const marks = [_]u32{@intCast(count - 7)};
            var s = try lanes.Stream.init(a, .{
                .id = "probe",
                .prompt = toks[0..count],
                .max_new = 1,
                .drafts = V.drafts(t),
                .reuse = .{ .marks = &marks, .hook = .{ .ptr = &hook, .at = Hook.at } },
            });
            defer s.deinit(a);
            const ns = pass: {
                defer backend.vtable.release(backend.ptr, &s); // once, whether the pass, its wait or nothing failed
                const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
                try backend.vtable.prefill(backend.ptr, &s);
                try stream(t).synchronize();
                break :pass std.Io.Clock.awake.now(io).toNanoseconds() - t0;
            };
            const st = hook.saved orelse return hook.failed orelse error.ProbeUnmarked;
            if (!probed(t, st)) return error.ProbeUnmarked;
            const bufs = V.buffers(st);
            const held = (if (bufs[0]) |b| b.len else 0) + (if (bufs[1]) |b| b.len else 0);
            const bytes = try a.alloc(u8, words + held);
            defer a.free(bytes);
            try copy(t, st, &.{.{ .off = 0, .len = bytes.len }}, bytes, true);
            const seconds = @as(f64, @floatFromInt(@max(ns, 1))) / 1e9;
            const rate = @as(f64, @floatFromInt(count)) / seconds;
            return .{ .hash = std.hash.Wyhash.hash(V.probe_seed, bytes), .rate = rate };
        }

        /// What the states depend on past the image and the probe: the family's KV form (keptKey).
        pub fn facts(ptr: anytype) @typeInfo(@TypeOf(V.facts)).@"fn".return_type.? {
            return V.facts(@as(*Target, @ptrCast(@alignCast(ptr))));
        }

        /// The family's view: keptKey reads its policy there when it declares one.
        pub const view = V;

        /// `id` extended with promptHash: what the prompt's tokens came from, which the image cannot show.
        pub fn prompted(a: Allocator, io: std.Io, o: api.Open, id: u64) !u64 {
            const ph = [_]u64{try promptHash(a, io, o.dir, o.drafter)};
            return @import("persistence_policy.zig").extend(id, "kept-prompt-v1", &ph);
        }
    };
}

test {
    _ = @import("cuda_kept_test.zig");
}
