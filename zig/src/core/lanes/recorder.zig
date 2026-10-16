//! Drafter training data from serving: each committed row's token, position and tapped features, files per stream.
const std = @import("std");
const platform = @import("sys");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const be = @import("backend.zig");
const Stream = @import("stream.zig").Stream;
const Held = @import("stream.zig").Held;
const Row = @import("logprob.zig").Row;
const disk = @import("learned_disk.zig");

/// A stream's files, appended row by row: u32 tokens, u32 cache rows, u8 kinds (1: a reply's row), the features.
const exts = [_][]const u8{ ".tok", ".pos", ".kind", ".st" };

/// What the recorder keeps of a stream it writes.
const Open = struct {
    base: []u8, // the files' path without the extension
    files: [exts.len]Io.File,
    rows: u64 = 0,
    from: ?u64 = null, // the cache length the last verify started from: its kept rows are written before the next
    layout: ?be.Features = null, // the first rows' row_bytes and dtype; later rows must match
    gap: ?[]const u8 = null, // why the stream's rows stopped there (static text)
};

pub const Recorder = struct {
    gpa: Allocator,
    io: Io,
    dir: [:0]const u8, // an existing folder, for the free-space check
    space: Io.Dir, // this recorder's own new folder <dir>/<pid>-<k>, held open; each stream writes <n>.* in it
    name: []u8, // that folder's name under `dir`
    taps: []const u32,
    floor: u64 = 25 << 30, // a stream starts no files below this many free bytes
    streams: std.AutoHashMapUnmanaged(*Stream, Open) = .empty,
    made: u64 = 0,
    scratch: std.ArrayList(u8) = .empty,

    /// A folder no other recorder or file has under `dir`: created, never reused, opened without following links.
    pub fn init(gpa: Allocator, io: Io, dir: [:0]const u8, taps: []const u32) !Recorder {
        return initTries(gpa, io, dir, taps, 1 << 16);
    }

    /// `init` giving up after `tries` names that already exist.
    pub fn initTries(gpa: Allocator, io: Io, dir: [:0]const u8, taps: []const u32, tries: u32) !Recorder {
        var root = try Io.Dir.cwd().openDir(io, dir, .{});
        defer root.close(io);
        var k: u32 = 0;
        while (true) : (k += 1) {
            const name = try std.fmt.allocPrint(gpa, "{d}-{d}", .{ platform.c.getpid(), k });
            errdefer gpa.free(name);
            root.createDir(io, name, .default_dir) catch |e| switch (e) {
                error.PathAlreadyExists => {
                    if (k + 1 == tries) return e; // the errdefer frees `name`
                    gpa.free(name);
                    continue;
                },
                else => return e,
            };
            const space = try root.openDir(io, name, .{ .follow_symlinks = false });
            return .{ .gpa = gpa, .io = io, .dir = dir, .space = space, .name = name, .taps = taps };
        }
    }

    /// Streams still open end their files here, without their last verify's rows (the target is gone).
    pub fn deinit(r: *Recorder) void {
        var it = r.streams.iterator();
        while (it.next()) |e| r.close(e.key_ptr.*, e.value_ptr, "the recorder closed first");
        r.streams.deinit(r.gpa);
        r.scratch.deinit(r.gpa);
        r.space.close(r.io);
        r.gpa.free(r.name);
    }

    /// Tells the target the taps before any forward; error.NoFeatures if it gives no rows.
    pub fn attach(r: *const Recorder, target: be.Backend) !void {
        if (target.vtable.features == null) return error.NoFeatures;
        if (target.vtable.prepare_features) |prep| try prep(target.ptr, r.taps);
    }

    /// After the prompt pass: the stream's files, and the rows it computed (from `cached` on).
    pub fn prompt(r: *Recorder, target: be.Backend, s: *Stream) void {
        if (s.yield_check != null) return; // a background pass: the request that reads it computes the rows it needs
        if (r.streams.contains(s)) return;
        if (disk.available(r.dir)) |free| if (free < r.floor) return;
        r.open(s) catch |e| {
            std.log.warn("recorder: no files for a stream under {s} ({s})", .{ r.dir, @errorName(e) });
            return;
        };
        const o = r.streams.getPtr(s).?;
        if (s.prompt_len > s.cached) r.write(target, s, o, s.cached, @intCast(s.prompt_len - s.cached));
    }

    /// Before a verify: each window's stream writes the rows its last verify kept, then counts from its cache length.
    pub fn before(r: *Recorder, target: be.Backend, windows: []const be.Window) void {
        for (windows) |w| if (r.streams.getPtr(w.stream)) |o| {
            r.flush(target, w.stream, o);
            o.from = w.stream.cache_len;
        };
    }

    /// A one-token round holds no rows for the recorder: the stream's files end before it.
    pub fn untracked(r: *Recorder, s: *Stream) void {
        if (r.streams.getPtr(s)) |o| if (o.gap == null) {
            o.gap = "a one-token round";
        };
    }

    /// Before the target releases the stream: its last verify's rows, then its files closed with their summary.
    pub fn finish(r: *Recorder, target: be.Backend, s: *Stream) void {
        const kv = r.streams.fetchRemove(s) orelse return;
        var o = kv.value;
        r.flush(target, s, &o);
        r.close(s, &o, null);
    }

    fn open(r: *Recorder, s: *Stream) !void {
        try r.streams.ensureUnusedCapacity(r.gpa, 1);
        const base = try std.fmt.allocPrint(r.gpa, "{d}", .{r.made});
        errdefer r.gpa.free(base);
        r.made += 1;
        var files: [exts.len]Io.File = undefined;
        var made: usize = 0;
        errdefer for (files[0..made]) |f| f.close(r.io);
        for (exts, &files) |ext, *f| {
            const path = try std.fmt.allocPrint(r.gpa, "{s}{s}", .{ base, ext });
            defer r.gpa.free(path);
            f.* = try r.space.createFile(r.io, path, .{ .exclusive = true });
            made += 1;
        }
        r.streams.putAssumeCapacity(s, .{ .base = base, .files = files });
    }

    fn flush(r: *Recorder, target: be.Backend, s: *Stream, o: *Open) void {
        const from = o.from orelse return;
        o.from = null;
        if (s.cache_len > from) r.write(target, s, o, from, @intCast(s.cache_len - from));
    }

    /// Cache rows [start, start + count): their tokens, rows, kinds and features, or the gap that stops the stream.
    fn write(r: *Recorder, target: be.Backend, s: *Stream, o: *Open, start: u64, count: u32) void {
        if (o.gap != null) return;
        o.gap = r.append(target, s, o, start, count) catch |e| switch (e) {
            error.OutOfMemory => "out of memory",
            else => "a write failed",
        };
        if (o.gap) |why| std.log.warn("recorder: {s}/{s}/{s} stops at row {d}: {s}", .{ r.dir, r.name, o.base, start, why });
    }

    fn append(r: *Recorder, target: be.Backend, s: *Stream, o: *Open, start: u64, count: u32) !?[]const u8 {
        if (start + count > s.context.items.len) return "rows past the stream's tokens";
        const f = target.features(s, r.taps, start, count) catch return "the target gave no rows";
        if (f.space != .host or f.ready != .none) return "rows not ready on the host";
        if (f.rows != count) return "the target gave other rows";
        if (o.layout) |l| if (l.row_bytes != f.row_bytes or l.dtype != f.dtype) return "the rows changed shape";
        o.layout = f;
        const tokens = s.context.items[start..][0..count];
        r.scratch.clearRetainingCapacity();
        try r.scratch.ensureTotalCapacity(r.gpa, count * 5);
        for (0..count) |i| r.scratch.appendSliceAssumeCapacity(std.mem.asBytes(&@as(u32, @intCast(start + i))));
        for (0..count) |i| r.scratch.appendAssumeCapacity(@intFromBool(start + i >= s.prompt_len));
        const rows: [*]const u8 = @ptrFromInt(f.buffer + f.offset);
        const parts = [exts.len][]const u8{ std.mem.sliceAsBytes(tokens), r.scratch.items[0 .. count * 4], r.scratch.items[count * 4 ..], rows[0 .. @as(usize, count) * f.row_bytes] };
        for (o.files, parts) |file, bytes| try file.writeStreamingAll(r.io, bytes);
        o.rows += count;
        return null;
    }

    /// The files closed, then `<base>.json`: rows, taps, layout, the first row, and the logprob rows if asked.
    fn close(r: *Recorder, s: *Stream, o: *Open, why: ?[]const u8) void {
        defer r.gpa.free(o.base);
        for (o.files) |f| f.close(r.io);
        if (why) |w| if (o.gap == null) {
            o.gap = w;
        };
        r.summary(s, o) catch |e| std.log.warn("recorder: no summary for {s}/{s}/{s} ({s})", .{ r.dir, r.name, o.base, @errorName(e) });
    }

    fn summary(r: *Recorder, s: *Stream, o: *const Open) !void {
        var arena: std.heap.ArenaAllocator = .init(r.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const top: u8 = if (s.logprobs) |k| if (s.rows.items.len > 0) k else 0 else 0;
        if (top > 0) try r.topK(a, s, o, top);
        var text: std.Io.Writer.Allocating = .init(a);
        var j: std.json.Stringify = .{ .writer = &text.writer };
        try j.write(.{
            .rows = o.rows,
            .first_row = s.cached,
            .prompt_len = s.prompt_len,
            .taps = r.taps,
            .row_bytes = if (o.layout) |l| l.row_bytes else 0,
            .dtype = if (o.layout) |l| @tagName(l.dtype) else "",
            .top_k = top,
            .gap = o.gap,
        });
        try r.create(try std.fmt.allocPrint(a, "{s}.json", .{o.base}), text.written());
    }

    /// A new file in the recorder's folder with these bytes; one already there is an error, never overwritten.
    fn create(r: *Recorder, path: []const u8, bytes: []const u8) !void {
        const f = try r.space.createFile(r.io, path, .{ .exclusive = true });
        defer f.close(r.io);
        try f.writeStreamingAll(r.io, bytes);
    }

    /// Reply token j's logprob row (from cache row prompt_len - 1 + j): k u32 ids, then k f32 logprobs, zero-padded.
    fn topK(r: *Recorder, a: Allocator, s: *Stream, o: *const Open, k: u8) !void {
        const out = try a.alloc(u8, s.rows.items.len * @as(usize, k) * 8);
        @memset(out, 0);
        for (s.rows.items, 0..) |row, i| {
            const at = out[i * @as(usize, k) * 8 ..];
            const n = @min(row.count, k);
            @memcpy(at[0 .. n * 4], std.mem.sliceAsBytes(row.ids[0..n]));
            @memcpy(at[@as(usize, k) * 4 ..][0 .. n * 4], std.mem.sliceAsBytes(row.logprobs[0..n]));
        }
        try r.create(try std.fmt.allocPrint(a, "{s}.top", .{o.base}), out);
    }
};

/// A target whose prompt passes, verifies, one-token rounds and releases are also shown to the recorder.
pub const Recorded = struct {
    inner: be.Backend,
    recorder: *Recorder,
    table: be.Backend.VTable = undefined,

    /// The target told the taps; the entries it gives forwarded, and only those.
    pub fn init(inner: be.Backend, recorder: *Recorder) !Recorded {
        try recorder.attach(inner);
        return .{ .inner = inner, .recorder = recorder };
    }

    pub fn backend(x: *Recorded) be.Backend {
        const v = x.inner.vtable;
        x.table = .{ .prefill = prefill, .prefill_step = if (v.prefill_step != null) prefillStep else null, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .unspeculate = if (v.unspeculate != null) unspeculate else null, .probabilities = if (v.probabilities != null) probabilities else null, .tree = if (v.tree != null) tree else null, .alternatives = if (v.alternatives != null) alternatives else null, .prepare_features = if (v.prepare_features != null) prepareFeatures else null, .features = features, .first_row = if (v.first_row != null) firstRow else null, .first_masked = if (v.first_masked != null) firstMasked else null, .held = if (v.held != null) held else null, .prompt_rows = v.prompt_rows, .release = release };
        return .{ .ptr = x, .vtable = &x.table };
    }

    fn of(p: *anyopaque) *Recorded {
        return @ptrCast(@alignCast(p));
    }
    fn prefill(p: *anyopaque, s: *Stream) anyerror!void {
        const x = of(p);
        try x.inner.prefill(s);
        x.recorder.prompt(x.inner, s);
    }
    fn prefillStep(p: *anyopaque, s: *Stream) anyerror!bool {
        const x = of(p);
        const done = try x.inner.vtable.prefill_step.?(x.inner.ptr, s);
        if (done) x.recorder.prompt(x.inner, s);
        return done;
    }
    fn first(p: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        return of(p).inner.first(s, position);
    }
    fn queue(p: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = of(p);
        x.recorder.untracked(s);
        return x.inner.queue(s, feed, position);
    }
    fn read(p: *anyopaque, handle: u64) anyerror!u32 {
        return of(p).inner.read(handle);
    }
    fn verify(p: *anyopaque, w: []const be.Window, out: []be.Verified) anyerror!void {
        const x = of(p);
        x.recorder.before(x.inner, w);
        return x.inner.verify(w, out);
    }
    fn keep(p: *anyopaque, w: []const be.Window, paths: []const []const u32) anyerror!void {
        return of(p).inner.keep(w, paths);
    }
    fn draft(p: *anyopaque, r: []const be.DraftRequest) anyerror!void {
        return of(p).inner.draft(r);
    }
    fn unspeculate(p: *anyopaque, s: *Stream) anyerror!void {
        const i = of(p).inner;
        return i.vtable.unspeculate.?(i.ptr, s);
    }
    fn probabilities(p: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const i = of(p).inner;
        return i.vtable.probabilities.?(i.ptr, s, out);
    }
    fn tree(p: *anyopaque, s: *Stream, gpa: Allocator) anyerror!?Held {
        const i = of(p).inner;
        return i.vtable.tree.?(i.ptr, s, gpa);
    }
    fn alternatives(p: *anyopaque, s: *Stream, out: []be.Alternative) anyerror!usize {
        const i = of(p).inner;
        return i.vtable.alternatives.?(i.ptr, s, out);
    }
    fn prepareFeatures(p: *anyopaque, taps: []const u32) anyerror!void {
        const i = of(p).inner;
        return i.vtable.prepare_features.?(i.ptr, taps);
    }
    fn features(p: *anyopaque, s: *Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
        return of(p).inner.features(s, taps, start, count);
    }
    fn firstRow(p: *anyopaque, s: *Stream) anyerror!Row {
        const i = of(p).inner;
        return i.vtable.first_row.?(i.ptr, s);
    }
    fn firstMasked(p: *anyopaque, s: *Stream, position: u64, mask: []const u32) anyerror!u64 {
        const i = of(p).inner;
        return i.vtable.first_masked.?(i.ptr, s, position, mask);
    }
    fn held(p: *anyopaque, s: *Stream, out: []u32) anyerror!void {
        const i = of(p).inner;
        return i.vtable.held.?(i.ptr, s, out);
    }
    fn release(p: *anyopaque, s: *Stream) void {
        const x = of(p);
        x.recorder.finish(x.inner, s);
        x.inner.release(s);
    }
};
