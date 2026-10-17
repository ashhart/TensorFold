//! Nemotron 3.5 Bolt's lane backend over libbolt.so: prompts, shared verify windows, rollback, drafts, keyed draws.
const std = @import("std");
const lanes = @import("lanes");
const reuse = @import("cuda_reuse.zig");
const cuda = @import("cuda");
const radio = @import("radio");
const native = @import("cuda_native.zig");
const lib = @import("bolt_api.zig");
pub const vision = native.vision;

pub const model_type = "nemotron_h_puzzle";
pub const formats: []const []const u8 = &.{"modelopt-mixed"};
pub const default_context: i64 = 262144;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = 2048;
pub const lone_min_budget: u64 = 0;
pub const explains_open = true; // explain() words open refusals too (--vision on a checkpoint without a tower)
pub const warm_quiet: u64 = 0;
/// No head reads a token ahead of a prompt: a state keys at its own row.
pub const snap_lookahead: u32 = 0;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, vision: ?vision.Limits = null, lanes: u32 = 1 };

/// Rows a verify window carries at most: the pending token and seven drafts.
const window: u32 = 8;
/// Prompt rows a cancel waits for at most, and the smaller piece while other streams decode.
const piece_rows: usize = 2048;
const busy_piece_rows: usize = 512;
/// Streams and rows one shared forward carries (libbolt's SEG_MAX and BATCH_MAX).
const max_segs: usize = 16;
const max_batch: u32 = 128;
/// Image spans a prompt piece may cross, and the rows its images take together (a piece cuts two 3,328-token images).
const max_runs = 64;
const image_rows: u32 = 16384;
/// Candidates kept per row for the keyed draw.
const cand_max: usize = 64;

/// Acceptance by depth until a stream has its own (greedy replies on the drafter's own text: 0.86 a level).
const prior = [_]f64{ 0.80, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68 };

const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(clk: c_int, ts: *Timespec) c_int;
fn nowMs() f64 {
    var ts: Timespec = undefined;
    _ = clock_gettime(1, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1000.0 + @as(f64, @floatFromInt(ts.nsec)) / 1e6;
}
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// What the backend keeps for one live stream: its libbolt sequence, the prompt pass in progress, and the drafts held.
const Slot = struct {
    seq: c_int,
    ids: []c_int = &.{}, // the prompt, while its pass runs
    at: usize = 0, // prompt tokens done
    decoding: bool = false, // the prompt pass is over
    held: [window]u32 = @splat(0), // the drafts held for the stream's next window
    held_n: u32 = 0,
};

/// Draws wait here between the forward and the round loop reading them back.
const draw_ring = 4096;

const Engine = struct {
    gpa: std.mem.Allocator,
    api: lib.Api,
    model: *anyopaque,
    drafts: bool,
    seqs: u32,
    slots: std.AutoHashMapUnmanaged(*const lanes.Stream, *Slot) = .empty,
    ring: [draw_ring]u32 = @splat(0),
    drawn: u64 = 0,
    decoding: u32 = 0, // streams past their prompt pass
    stat_on: bool = false, // BOLT_STATS: log the round loop's time in the backend every 100 windows
    stat: struct { rounds: u32 = 0, streams: u32 = 0, rows: u32 = 0, verify: f64 = 0, draw: f64 = 0, draft: f64 = 0, keep: f64 = 0 } = .{},
    target: reuse.Target = undefined,
    vision: ?*vision.Vision = null, // --vision: the RADIO tower and a rows buffer one prompt piece uses
    stream: ?cuda.Stream = null, // the stream the tower runs on
    maskbuf: std.ArrayList(u32) = .empty, // structured output: a verify's masked rows' allowed-token bits, back to back

    /// Candidates the keyed draw needs for this stream's sampling: 1 when greedy.
    fn candCount(s: *const lanes.Stream) c_int {
        const sm = s.sampling orelse return 1;
        if (sm.temperature <= 0) return 1;
        const k: usize = if (sm.top_k == 0 or sm.top_k > cand_max) cand_max else sm.top_k;
        return if (k <= 20) 20 else @intCast(cand_max);
    }

    fn slotOf(e: *Engine, s: *const lanes.Stream) !*Slot {
        return e.slots.get(s) orelse error.BoltNoSequence;
    }

    /// The token drawn from one row of the last forward at `position`, keyed by the stream's seed.
    fn draw(e: *Engine, s: *const lanes.Stream, row: usize, position: u64, k: c_int) !u32 {
        var vals: [cand_max]f32 = undefined;
        var ids: [cand_max]c_int = undefined;
        e.api.candidates(e.model, @intCast(row), k, &vals, &ids);
        const sm = s.sampling orelse return @intCast(ids[0]);
        if (sm.temperature <= 0) return @intCast(ids[0]);
        var n: usize = @intCast(k);
        while (n > 1 and vals[n - 1] < -3.0e38) n -= 1; // a masked row with fewer allowed tokens than candidates
        var v: [cand_max]f64 = undefined;
        var d: [cand_max]u64 = undefined;
        for (0..n) |i| {
            v[i] = vals[i];
            d[i] = @intCast(ids[i]);
        }
        return @intCast(try lanes.sampling.choose(e.gpa, v[0..n], d[0..n], position, sm));
    }

    fn remember(e: *Engine, token: u32) u64 {
        const h = e.drawn;
        e.ring[h % draw_ring] = token;
        e.drawn += 1;
        return h;
    }

    /// Take a sequence for the stream and resume it from a kept prompt state when the cache has one for its prefix.
    fn begin(e: *Engine, s: *lanes.Stream) !*Slot {
        if (s.images.len > 0 and e.vision == null) return error.NoVision;
        const seq = e.api.seq_alloc(e.model);
        if (seq < 0) return error.LaneMemory; // every sequence is in use: the stream waits at the queue's head
        const slot = try e.gpa.create(Slot);
        errdefer e.gpa.destroy(slot);
        slot.* = .{ .seq = seq };
        errdefer e.api.seq_release(e.model, seq);
        try e.slots.put(e.gpa, s, slot);
        errdefer _ = e.slots.remove(s);
        const prompt = s.prompt();
        s.cached = 0;
        if (s.reuse.saved) |saved| {
            if (s.reuse.at < prompt.len and e.api.snap_restore(e.model, seq, saved) == 0) {
                slot.at = s.reuse.at;
                s.cached = @intCast(slot.at);
            } else {
                s.reuse_failed = true; // the restore refused before touching the sequence: it starts from 0
            }
        }
        e.api.prompt_end(e.model, seq, @intCast(prompt.len));
        std.log.info("bolt: prefill {d} tokens ({d} from the cache), sequence {d}", .{ prompt.len, slot.at, slot.seq });
        slot.ids = try e.gpa.alloc(c_int, prompt.len);
        for (slot.ids, prompt) |*d, t| d.* = @intCast(t);
        return slot;
    }

    /// One piece of the prompt pass, cut at the cache's marks, small enough that a cancel lands quickly.
    fn prefillStep(ptr: *anyopaque, s: *lanes.Stream) !bool {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const prompt_len = s.prompt().len;
        const slot = e.slots.get(s) orelse try e.begin(s);
        if (s.isCancelled() or s.yields()) return error.Cancelled;
        if (slot.at >= prompt_len) return true;
        const rows: usize = if (e.decoding > 0) busy_piece_rows else piece_rows;
        var end = @min(prompt_len, slot.at + rows);
        for (s.reuse.marks) |m| {
            if (m > slot.at and m < end) end = m;
        }
        const last = end == prompt_len;
        const cand: c_int = if (last) candCount(s) else 0;
        var runs: [max_runs]lib.RowRun = undefined;
        const nruns = if (s.images.len > 0) try e.imageRuns(s, slot.at, end, &runs) else 0;
        const rc = if (nruns > 0)
            e.api.prefill_rows(e.model, slot.seq, slot.ids.ptr + slot.at, @intCast(end - slot.at), cand, &runs, @intCast(nruns))
        else
            e.api.prefill(e.model, slot.seq, slot.ids.ptr + slot.at, @intCast(end - slot.at), cand);
        if (rc != 0) return error.BoltForward;
        slot.at = end;
        if (s.reuse.hook) |hk| {
            if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(slot.at)) != null) hk.at(hk.ptr, s, @intCast(slot.at));
        }
        if (!last) return false;
        e.gpa.free(slot.ids);
        slot.ids = &.{};
        slot.decoding = true;
        e.decoding += 1;
        return true;
    }

    /// The images rows [lo, hi) cross, encoded into the rows buffer, and the runs libbolt reads (a cut one is redone).
    fn imageRuns(e: *Engine, s: *const lanes.Stream, lo: usize, hi: usize, runs: []lib.RowRun) !usize {
        const v = e.vision orelse return error.NoVision;
        const st = e.stream.?;
        const row_bytes = v.rows.hidden * 2;
        var top: u32 = 0;
        var n: usize = 0;
        for (s.images) |img| {
            const a: usize = img.at;
            const b = a + img.tokens;
            if (b <= lo) continue;
            if (a >= hi) break; // in prompt order
            if (n == runs.len or top + img.tokens > v.rows.limit) return error.ImageRows;
            const base = v.rows.buf.ptr + @as(u64, top) * row_bytes;
            try v.tower.picture(st, img, base);
            const from = @max(lo, a);
            const to = @min(hi, b);
            runs[n] = .{ .pos = @intCast(from - lo), .count = @intCast(to - from), .rows = base + @as(u64, @intCast(from - a)) * row_bytes };
            n += 1;
            top += img.tokens;
        }
        if (n > 0) try st.synchronize(); // the rows are whole before libbolt copies them
        return n;
    }

    fn prefill(ptr: *anyopaque, s: *lanes.Stream) !void {
        while (!try prefillStep(ptr, s)) {}
    }

    fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) !u64 {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const token = try e.draw(s, 0, position, candCount(s));
        return e.remember(token);
    }

    /// The first token again, drawn among the tokens `mask` allows (the prompt pass drew it without the grammar).
    fn firstMasked(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const k = candCount(s);
        const row0 = [1]c_int{0};
        if (mask.len == 0 or e.api.mask_rows(e.model, 1, &row0, mask.ptr, @intCast(mask.len), k) != 0) return error.BoltForward;
        const token = try e.draw(s, 0, position, k);
        return e.remember(token);
    }

    /// The drafts held for the stream's next window, as host tokens in window order (a grammar walks them).
    fn held(ptr: *anyopaque, s: *lanes.Stream, out: []u32) !void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const slot = try e.slotOf(s);
        if (out.len > slot.held_n) return error.HeldDraftsUnreadable;
        @memcpy(out, slot.held[0..out.len]);
    }

    fn queue(ptr: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, position: u64) !u64 {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const slot = try e.slotOf(s);
        const tok: u32 = switch (feed) {
            .handle => |h| e.ring[h % draw_ring],
            .value => |v| v,
        };
        const ids = [1]c_int{@intCast(tok)};
        const k = candCount(s);
        // a one-row window, kept at once: the same attention arithmetic as every verify window, alone or among streams
        if (e.api.forward(e.model, slot.seq, &ids, 1, 1, k) != 0) return error.BoltForward;
        e.api.keep(e.model, slot.seq, 1);
        const next = try e.draw(s, 0, position, k);
        return e.remember(next);
    }

    fn read(ptr: *anyopaque, handle: u64) !u32 {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        return e.ring[handle % draw_ring];
    }

    /// One forward over every window (one per stream), each row drawn at its position with its stream's sampling.
    fn verify(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) !void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        if (windows.len == 0 or windows.len > max_segs) return error.WindowTooWide;
        var seqs: [max_segs]c_int = undefined;
        var counts: [max_segs]c_int = undefined;
        var toks: [max_batch]c_int = undefined;
        var total: usize = 0;
        var k: c_int = 1;
        var masked: [max_batch]c_int = undefined; // the block rows whose draw is restricted by a grammar, and their mask width
        var nmasked: usize = 0;
        var mwords: usize = 0;
        e.maskbuf.clearRetainingCapacity();
        for (windows, 0..) |w, i| {
            const slot = try e.slotOf(w.stream);
            const rows = w.rows();
            if (rows > window or w.parents != null or total + rows > max_batch) return error.WindowTooWide;
            if (w.masks.len != 0) {
                const words = w.masks.len / rows;
                if (words == 0 or words * rows != w.masks.len or (mwords != 0 and words != mwords)) return error.BadMask;
                mwords = words;
                try e.maskbuf.appendSlice(e.gpa, w.masks);
                for (0..rows) |r| {
                    masked[nmasked] = @intCast(total + r);
                    nmasked += 1;
                }
            }
            toks[total] = @intCast(w.pending);
            for (0..w.held) |j| {
                toks[total + 1 + j] = @intCast(slot.held[j]);
                out[i].drafts[j] = slot.held[j];
            }
            for (w.tokens, 0..) |t, j| toks[total + 1 + w.held + j] = @intCast(t);
            seqs[i] = slot.seq;
            counts[i] = @intCast(rows);
            total += rows;
            k = @max(k, candCount(w.stream));
        }
        if (e.seqs > 1) k = cand_max; // one candidate count keeps the captured forward graphs few
        const t0 = nowMs();
        if (e.api.forward_multi(e.model, @intCast(windows.len), &seqs, &counts, &toks, k) != 0) return error.BoltForward;
        if (nmasked > 0 and e.api.mask_rows(e.model, @intCast(nmasked), &masked, e.maskbuf.items.ptr, @intCast(mwords), cand_max) != 0) return error.BoltForward;
        const t1 = nowMs();
        var row: usize = 0;
        for (windows, 0..) |w, i| {
            const kw = candCount(w.stream);
            for (0..w.rows()) |r| out[i].sampled[r] = try e.draw(w.stream, row + r, w.positions[r], kw);
            row += w.rows();
            (try e.slotOf(w.stream)).held_n = 0;
        }
        if (e.stat_on) {
            e.stat.rounds += 1;
            e.stat.streams += @intCast(windows.len);
            e.stat.rows += @intCast(total);
            e.stat.verify += t1 - t0;
            e.stat.draw += nowMs() - t1;
            if (e.stat.rounds == 100) {
                const n: f64 = 100;
                std.log.info("bolt stats: 100 rounds, {d:.1} streams {d:.1} rows a round; forward {d:.1} ms, draws {d:.1} ms, drafts {d:.1} ms, keeps {d:.1} ms a round", .{ @as(f64, @floatFromInt(e.stat.streams)) / n, @as(f64, @floatFromInt(e.stat.rows)) / n, e.stat.verify / n, e.stat.draw / n, e.stat.draft / n, e.stat.keep / n });
                e.stat = .{};
            }
        }
    }

    /// Roll each stream back to its window's kept rows.
    fn keep(ptr: *anyopaque, windows: []const lanes.backend.Window, paths: []const []const u32) !void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const t0 = nowMs();
        for (windows, paths) |w, path| {
            const slot = try e.slotOf(w.stream);
            e.api.keep(e.model, slot.seq, @intCast(path.len));
        }
        e.stat.keep += nowMs() - t0;
    }

    /// Hold each stream's drafts for its next window: LiLiCorr over every stream's kept rows in one pass.
    fn draft(ptr: *anyopaque, requests: []const lanes.backend.DraftRequest) !void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const t0 = nowMs();
        defer e.stat.draft += nowMs() - t0;
        var seqs: [max_segs]c_int = undefined;
        var anchors: [max_segs]c_int = undefined;
        var who: [max_segs]*Slot = undefined;
        var depths: [max_segs]u32 = undefined;
        var n: usize = 0;
        for (requests) |r| {
            const slot = try e.slotOf(r.stream);
            slot.held_n = 0;
            // a shared round drafts before it keeps: cut the stream back to its kept rows first (a repeat is a no-op)
            if (r.rows) |rows| e.api.keep(e.model, slot.seq, @intCast(rows.len));
            if (!e.drafts or r.depth == 0 or n == max_segs) continue;
            const pending: u32 = if (r.follow.len > 0) r.follow[r.follow.len - 1] else switch (r.first orelse continue) {
                .handle => |h| e.ring[h % draw_ring],
                .value => |v| v,
            };
            seqs[n] = slot.seq;
            anchors[n] = @intCast(pending);
            who[n] = slot;
            depths[n] = r.depth;
            n += 1;
        }
        if (n == 0) return;
        var out: [max_segs * window]c_int = @splat(0);
        var got: [max_segs]c_int = @splat(0);
        e.api.propose_multi(e.model, @intCast(n), &seqs, &anchors, &out, &got);
        for (0..n) |i| {
            const depth: usize = @min(@as(usize, depths[i]), window - 1);
            const have: usize = @intCast(@max(0, got[i]));
            const pending: u32 = @intCast(anchors[i]);
            for (0..depth) |j| who[i].held[j] = if (j < have) @intCast(out[i * window + j]) else pending;
            who[i].held_n = @intCast(depth);
        }
    }

    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const kv = e.slots.fetchRemove(s) orelse return;
        const slot = kv.value;
        if (slot.decoding) e.decoding -|= 1;
        e.api.seq_release(e.model, slot.seq);
        if (slot.ids.len > 0) e.gpa.free(slot.ids);
        e.gpa.destroy(slot);
    }

    fn backend(e: *Engine) lanes.backend.Backend {
        const vt = struct {
            var v: lanes.backend.Backend.VTable = .{
                .prefill = Engine.prefill,
                .prefill_step = Engine.prefillStep,
                .first = Engine.first,
                .first_masked = Engine.firstMasked,
                .held = Engine.held,
                .queue = Engine.queue,
                .read = Engine.read,
                .verify = Engine.verify,
                .keep = Engine.keep,
                .draft = Engine.draft,
                .release = Engine.release,
            };
        };
        return .{ .ptr = e, .vtable = &vt.v };
    }
};

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?@import("cuda_native.zig").LoneRun = null,
    target: *reuse.Target,
    learns: ?*@import("cuda_engine.zig").Engine = null,
    learn_bytes: usize = 0,
    vision: ?*vision.Vision = null,
};

fn shutdown(p: *anyopaque) void {
    const e: *Engine = @ptrCast(@alignCast(p));
    e.api.free(e.model);
    e.maskbuf.deinit(e.gpa);
    if (e.vision) |v| v.deinit();
    if (e.stream) |*st| st.deinit();
    e.slots.deinit(e.gpa);
    e.gpa.destroy(e);
}

// -- prompt-cache copies: the Mamba state, the attention rows and the drafter's context after `at` prompt tokens ----

pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    return e.api.snap_bytes(e.model, @intCast(at));
}

/// `owner` is the lane core's stream: its sequence is the state to copy.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.BoltSnapshot));
    const slot = e.slots.get(s) orelse return error.BoltSnapshot;
    return e.api.snap_save(e.model, slot.seq, @intCast(at)) orelse error.BoltSnapshot;
}

pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.BoltSnapshot));
    const slot = e.slots.get(s) orelse return error.BoltSnapshot;
    if (e.api.snap_restore(e.model, slot.seq, saved) != 0) return error.BoltSnapshot;
    slot.held_n = 0;
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    e.api.snap_drop(e.model, saved);
}

/// What the loaded libbolt measured: forward ms by window width, and shared forwards by total rows.
var costs: [window]lanes.config.Cost = undefined;
var shared: [32]lanes.config.Cost = undefined;

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = kernels;
    // --vision: the RADIO tower and its rows buffer, before the language model takes its memory
    var tower: ?*vision.Vision = null;
    var problem: []const u8 = "";
    if (o.vision) |lim| {
        // the rows buffer holds one prompt piece's images; a request's token totals are the server's limit alone
        tower = (try vision.Vision.init(gpa, io, ctx.d, dir, .{ .image_tokens = @min(lim.image_tokens, image_rows), .max_images = lim.max_images }, &problem)) orelse {
            std.log.err("--vision: {s}", .{problem});
            return error.VisionRefused;
        };
        tower.?.limits.image_tokens = lim.image_tokens;
        tower.?.limits.video_tokens = lim.video_tokens;
    }
    errdefer if (tower) |v| v.deinit();
    var stream: ?cuda.Stream = if (tower != null) try cuda.Stream.init(ctx.d, false) else null;
    errdefer if (stream) |*st| st.deinit();
    const handle = lib.openLibrary() orelse {
        std.log.err("bolt: libbolt.so: {s} (zig build bolt -Dnvcc=NVCC installs it beside the server)", .{if (lib.dlerror()) |m| std.mem.span(m) else "not found"});
        return error.BoltMissing;
    };
    const api = try lib.Api.load(handle);
    if (api.abi() != lib.abi_version) {
        std.log.err("bolt: libbolt.so speaks ABI {d}, this server {d}: build it from the same source", .{ api.abi(), lib.abi_version });
        return error.BoltMissing;
    }
    // one libbolt sequence a stream (--parallel), unless BOLT_SEQS says otherwise
    var seqs_buf: [16]u8 = undefined;
    _ = setenv("BOLT_SEQS", std.fmt.bufPrintSentinel(&seqs_buf, "{d}", .{o.lanes}, 0) catch "1", 0);
    // no drafter checkpoint is read (or its memory held) when the server runs --no-drafts
    if (!o.drafts) _ = setenv("BOLT_DRAFT_OFF", "1", 1);
    const path = try gpa.allocSentinel(u8, dir.len, 0);
    defer gpa.free(path);
    @memcpy(path[0..dir.len], dir);
    const model = api.open(path, @intCast(o.context)) orelse return error.BoltLoad;
    const e = try gpa.create(Engine);
    const nseq: u32 = @intCast(@max(1, api.seqs(model)));
    e.* = .{ .gpa = gpa, .api = api, .model = model, .drafts = o.drafts and api.has_draft(model) != 0, .seqs = nseq, .stat_on = std.c.getenv("BOLT_STATS") != null, .vision = tower, .stream = stream };
    for (0..window) |i| costs[i] = .{ .width = @intCast(i + 1), .ms = api.cost(model, @intCast(i + 1)) };
    const propose_ms = api.cost(model, 0);
    const propose_multi_ms = api.draft_multi_cost(model);
    var rows_buf: [32]c_int = undefined;
    var ms_buf: [32]f64 = undefined;
    const n_shared: usize = @intCast(@max(0, api.shared_costs(model, &rows_buf, &ms_buf, 32)));
    for (0..n_shared) |i| shared[i] = .{ .width = @intCast(rows_buf[i]), .ms = ms_buf[i] };
    std.log.info("bolt: windows ms {d:.1} {d:.1} {d:.1} {d:.1} {d:.1} {d:.1} {d:.1} {d:.1}, one proposal {d:.1} ({d} streams {d:.1}), drafts {s}, {d} sequence{s}", .{ costs[0].ms, costs[1].ms, costs[2].ms, costs[3].ms, costs[4].ms, costs[5].ms, costs[6].ms, costs[7].ms, propose_ms, nseq, propose_multi_ms, if (e.drafts) "on" else "off", nseq, if (nseq == 1) "" else "s" });
    for (shared[0..n_shared]) |c| std.log.info("bolt: shared round of {d} rows {d:.1} ms", .{ c.width, c.ms });
    return .{
        .backend = e.backend(),
        .facts = .{
            .exact_width = window,
            .gpu_tokens = false,
            .mtp = e.drafts,
            .speculate = e.drafts,
            .speculate_early = false,
            .draft_prior = &prior,
            .drafts = window - 1,
            .window_costs = &costs,
            .shared_costs = shared[0..n_shared],
            .mtp_step_ms = propose_ms / 3.0,
            .draft_streams = nseq > 1,
            .hidden_rows = nseq > 1,
            .streams_exact = true,
            .batch_rows = @min(max_batch, nseq * window),
            .max_streams = nseq,
        },
        .rows = window,
        .stream_bytes = @intCast(api.seq_bytes(model)),
        .free_streams = nseq, // their state is already resident
        .ctx = e,
        .deinit = shutdown,
        .target = @ptrCast(@alignCast(e)), // the prompt cache's opaque handle: the engine itself
        .vision = tower,
    };
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.BoltLoad => "Bolt could not read this checkpoint",
        error.BoltForward => "Bolt's forward failed",
        error.VisionRefused => "--vision: this checkpoint has no image tower Bolt reads (the RADIO tower of an omni checkpoint); serve it without --vision",
        error.NoVision => "this server reads no images: start it with --vision",
        error.WindowTooWide => "Bolt verifies chains of up to eight rows a stream, sixteen streams at a time",
        error.BadMask => "a grammar mask does not match the rows of its window",
        else => null,
    };
}
