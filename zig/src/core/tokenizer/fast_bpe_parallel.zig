//! Deterministic parallel dispatch on top of fast_bpe for large prompts and multi-segment chats.
const std = @import("std");
const models = @import("model.zig");
const fast_split = @import("fast_split.zig");
const fast_bpe = @import("fast_bpe.zig");
const Allocator = std.mem.Allocator;

const FastPath = fast_bpe.FastPath;
pub const Segment = fast_bpe.Segment;

/// Thread-dispatch thresholds for parallel ByteLevel-BPE encoding.
pub const Config = struct {
    parallel_min_bytes: usize = 64 * 1024,
    parallel_max_threads: usize = 10,
};

pub const default_config: Config = .{};

const max_cut_threads = 30; // fits in a [32]usize cuts array with leading 0 and trailing text.len

const ChunkJob = struct {
    fast: *const FastPath,
    model: *const models.Model,
    vocab: *const std.StringHashMap(u32),
    a: Allocator,
    text: []const u8,
    spans: []const [2]usize = &.{},
    segs: []const Segment = &.{},
    out: std.ArrayList(u32) = .empty,
    err: ?Allocator.Error = null,
};

fn runSliceJob(job: *ChunkJob) void {
    fast_bpe.encodeSlice(job.fast, job.model, job.vocab, job.a, job.text, &job.out) catch |e| {
        job.err = e;
    };
}

fn runSpansJob(job: *ChunkJob) void {
    fast_bpe.encodeSpans(job.fast, job.model, job.vocab, job.a, job.text, job.spans, &job.out) catch |e| {
        job.err = e;
    };
}

fn runSegsJob(job: *ChunkJob) void {
    fast_bpe.encodeSegmentsSerial(job.fast, job.model, job.vocab, job.a, job.text, job.segs, &job.out) catch |e| {
        job.err = e;
    };
}

fn chooseThreads(cfg: Config, len: usize) usize {
    const min_b = @max(cfg.parallel_min_bytes, 1);
    if (len < min_b * 2 or cfg.parallel_max_threads <= 1) return 1;
    const cpus = std.Thread.getCpuCount() catch 1;
    const max_t = @min(@min(cfg.parallel_max_threads, @max(cpus, 1)), max_cut_threads);
    return @min(max_t, len / min_b);
}

fn runJobs(
    worker_a: Allocator,
    scratch_a: Allocator,
    jobs: []ChunkJob,
    comptime worker_fn: fn (*ChunkJob) void,
    out: *std.ArrayList(u32),
) Allocator.Error!void {
    const threads = try scratch_a.alloc(?std.Thread, jobs.len);
    defer scratch_a.free(threads);
    @memset(threads, null);
    defer {
        for (threads) |maybe_t| if (maybe_t) |th| th.join();
        for (jobs) |*j| j.out.deinit(worker_a);
    }
    for (1..jobs.len) |i| {
        threads[i] = std.Thread.spawn(.{}, worker_fn, .{&jobs[i]}) catch null;
        if (threads[i] == null) worker_fn(&jobs[i]);
    }
    worker_fn(&jobs[0]);
    for (threads) |*maybe_t| {
        if (maybe_t.*) |th| {
            th.join();
            maybe_t.* = null;
        }
    }
    for (jobs) |j| if (j.err) |e| return e;
    var total: usize = 0;
    for (jobs) |j| total += j.out.items.len;
    try out.ensureUnusedCapacity(scratch_a, total);
    for (jobs) |j| out.appendSliceAssumeCapacity(j.out.items);
}

fn encodeParallelChunks(
    fast: *const FastPath,
    model: *const models.Model,
    vocab: *const std.StringHashMap(u32),
    worker_a: Allocator,
    scratch_a: Allocator,
    text: []const u8,
    cuts: []const usize,
    out: *std.ArrayList(u32),
) Allocator.Error!void {
    const n = cuts.len - 1;
    const jobs = try scratch_a.alloc(ChunkJob, n);
    defer scratch_a.free(jobs);
    for (0..n) |i| {
        jobs[i] = .{ .fast = fast, .model = model, .vocab = vocab, .a = worker_a, .text = text[cuts[i]..cuts[i + 1]] };
    }
    return runJobs(worker_a, scratch_a, jobs, runSliceJob, out);
}

fn encodeParallelSpans(
    fast: *const FastPath,
    model: *const models.Model,
    vocab: *const std.StringHashMap(u32),
    worker_a: Allocator,
    scratch_a: Allocator,
    text: []const u8,
    n_threads: usize,
    out: *std.ArrayList(u32),
) Allocator.Error!void {
    var spans: std.ArrayList([2]usize) = .empty;
    defer spans.deinit(scratch_a);
    try spans.ensureTotalCapacity(scratch_a, text.len / 4 + 16);
    var pos: usize = 0;
    while (fast_split.next(fast.split, text, &pos)) |sp| try spans.append(scratch_a, sp);
    const n = @min(n_threads, @max(spans.items.len / 256, 1));
    if (n <= 1) return fast_bpe.encodeSpans(fast, model, vocab, scratch_a, text, spans.items, out);

    const jobs = try scratch_a.alloc(ChunkJob, n);
    defer scratch_a.free(jobs);
    const per = spans.items.len / n;
    for (0..n) |i| {
        const lo = i * per;
        const hi = if (i + 1 == n) spans.items.len else (i + 1) * per;
        jobs[i] = .{ .fast = fast, .model = model, .vocab = vocab, .a = worker_a, .text = text, .spans = spans.items[lo..hi] };
    }
    return runJobs(worker_a, scratch_a, jobs, runSpansJob, out);
}

fn encodeParallelSegments(
    fast: *const FastPath,
    model: *const models.Model,
    vocab: *const std.StringHashMap(u32),
    worker_a: Allocator,
    scratch_a: Allocator,
    text: []const u8,
    segs: []const Segment,
    total_bytes: usize,
    n_threads: usize,
    out: *std.ArrayList(u32),
) Allocator.Error!void {
    const jobs = try scratch_a.alloc(ChunkJob, n_threads);
    defer scratch_a.free(jobs);
    const target = @max(total_bytes / n_threads, 1);
    var count: usize = 0;
    var start: usize = 0;
    var acc: usize = 0;
    for (segs, 0..) |seg, idx| {
        acc += if (seg.id != null) 0 else seg.end - seg.start;
        const remaining = segs.len - (idx + 1);
        if (count + 1 < n_threads and acc >= target and remaining >= n_threads - (count + 1)) {
            jobs[count] = .{ .fast = fast, .model = model, .vocab = vocab, .a = worker_a, .text = text, .segs = segs[start .. idx + 1] };
            count += 1;
            start = idx + 1;
            acc = 0;
        }
    }
    jobs[count] = .{ .fast = fast, .model = model, .vocab = vocab, .a = worker_a, .text = text, .segs = segs[start..] };
    count += 1;
    if (count <= 1) return fast_bpe.encodeSegmentsSerial(fast, model, vocab, scratch_a, text, segs, out);
    return runJobs(worker_a, scratch_a, jobs[0..count], runSegsJob, out);
}

/// Encodes already-split added/text segments using default parallel thresholds.
pub fn encodeSegments(
    fast: *const FastPath,
    model: *const models.Model,
    vocab: *const std.StringHashMap(u32),
    worker_a: Allocator,
    scratch_a: Allocator,
    text: []const u8,
    segs: []const Segment,
    out: *std.ArrayList(u32),
) Allocator.Error!void {
    return encodeSegmentsWithConfig(fast, model, vocab, worker_a, scratch_a, text, segs, out, default_config);
}

/// Encodes already-split added/text segments with caller-supplied parallel thresholds.
pub fn encodeSegmentsWithConfig(
    fast: *const FastPath,
    model: *const models.Model,
    vocab: *const std.StringHashMap(u32),
    worker_a: Allocator,
    scratch_a: Allocator,
    text: []const u8,
    segs: []const Segment,
    out: *std.ArrayList(u32),
    cfg: Config,
) Allocator.Error!void {
    var total_bytes: usize = 0;
    var max_seg: usize = 0;
    var text_segs: usize = 0;
    for (segs) |seg| {
        if (seg.id == null and seg.end > seg.start) {
            const len = seg.end - seg.start;
            total_bytes += len;
            max_seg = @max(max_seg, len);
            text_segs += 1;
        }
    }
    const n_threads = chooseThreads(cfg, total_bytes);
    if (n_threads <= 1) return fast_bpe.encodeSegmentsSerial(fast, model, vocab, scratch_a, text, segs, out);
    if (text_segs >= n_threads * 2 and max_seg <= total_bytes / @max(n_threads / 2, 2)) {
        return encodeParallelSegments(fast, model, vocab, worker_a, scratch_a, text, segs, total_bytes, n_threads, out);
    }
    var s: fast_bpe.Scratch = .{};
    defer s.deinit(scratch_a);
    for (segs) |seg| {
        if (seg.id) |id| {
            try out.append(scratch_a, id);
            continue;
        }
        if (seg.end == seg.start) continue;
        const slice = text[seg.start..seg.end];
        if (chooseThreads(cfg, slice.len) > 1) {
            try encodeWithConfig(fast, model, vocab, worker_a, scratch_a, slice, out, cfg);
        } else {
            var pos: usize = 0;
            while (fast_split.next(fast.split, slice, &pos)) |sp| {
                try fast_bpe.encodePiece(fast, model, vocab, &s, scratch_a, text, .{ seg.start + sp[0], seg.start + sp[1] }, out);
            }
        }
    }
}

/// Encodes `text` using the fast ByteLevel BPE path with default parallel thresholds.
pub fn encode(
    fast: *const FastPath,
    model: *const models.Model,
    vocab: *const std.StringHashMap(u32),
    worker_a: Allocator,
    scratch_a: Allocator,
    text: []const u8,
    out: *std.ArrayList(u32),
) Allocator.Error!void {
    return encodeWithConfig(fast, model, vocab, worker_a, scratch_a, text, out, default_config);
}

/// Encodes `text` using the fast ByteLevel BPE path and caller-supplied parallel thresholds.
pub fn encodeWithConfig(
    fast: *const FastPath,
    model: *const models.Model,
    vocab: *const std.StringHashMap(u32),
    worker_a: Allocator,
    scratch_a: Allocator,
    text: []const u8,
    out: *std.ArrayList(u32),
    cfg: Config,
) Allocator.Error!void {
    if (text.len == 0) return;
    const n_threads = chooseThreads(cfg, text.len);
    if (n_threads <= 1) return fast_bpe.encodeSlice(fast, model, vocab, scratch_a, text, out);

    var cuts_buf: [max_cut_threads + 2]usize = undefined;
    var count: usize = 1;
    cuts_buf[0] = 0;
    var prev: usize = 0;
    const step = text.len / n_threads;
    for (1..n_threads) |i| {
        const target = i * step;
        const margin = step / 4;
        const lo = @max(prev + 1, target -| margin);
        const hi = @min(text.len - 1, target + margin);
        if (fast_split.findSafeCut(fast.split, text, target, lo, hi)) |cut| {
            cuts_buf[count] = cut;
            count += 1;
            prev = cut;
        }
    }
    cuts_buf[count] = text.len;
    count += 1;
    if (count >= 3) {
        return encodeParallelChunks(fast, model, vocab, worker_a, scratch_a, text, cuts_buf[0..count], out);
    }
    return encodeParallelSpans(fast, model, vocab, worker_a, scratch_a, text, n_threads, out);
}
