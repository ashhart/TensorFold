//! One TP rank's GLM-5.3-Flash CUDA engine (the Python family's decode.Engine); all ranks draw the same tokens.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const config = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const fw = @import("cuda_forward.zig");
const ranks = @import("cuda_ranks.zig");
const sampler = @import("cuda_sampler.zig");
const heat = @import("heat");
const modelopt = @import("core").modelopt;
const exl3 = @import("cuda_exl3.zig");
const images_mod = @import("cuda_images.zig");
const prof_mod = @import("cuda_prof.zig");

pub const Options = struct {
    capacity: usize, // tokens a sequence holds
    mtp: bool, // load the MTP head (drafts)
    settings: ranks.Settings,
    vision: u32 = 0, // --vision's image tokens (0: none): every rank must hold the rows rank 0 sends
};

/// Rank 0's own word on its request (cancelled or not); the other ranks have none and hear rank 0's.
pub const Cancel = struct { ptr: ?*anyopaque, check: *const fn (?*anyopaque) bool };

/// The most drafts a chain holds (decode.DepthPolicy's top depth).
pub const max_drafts = 3;
/// A greedy pick as it crosses ranks: (value, id) and padding, so the RoCE gather (16-byte packs) can carry it.
const pair_bytes = 16;

/// A prompt pass's chunk rows (decode.prefill's).
pub const prompt_rows = fw.prefill_rows;

/// A shared verify's max rows and streams: under hc_pre's 64-row block change, 8 streams of 4 rows.
pub const shared_rows = 32;
pub const max_shared = 8;

/// One stream's window of a shared verify: its slot, its ids, and its drawn tokens.
pub const Part = struct { slot: u32, ids: []const u32, out: []u32 };

/// A kept prompt state as the prompt cache holds it: the id every rank keeps its own copy under, and its position.
pub const Kept = struct { id: u64, at: u32 };

/// A stream's own sequence: state, sampling rule, MTP drafts; buffers are shared, one stream's round at a time.
const Seq = struct {
    st: fw.State,
    sampling: ?sampler.Sampling = null, // null: greedy
    in_head: usize = 0, // drafted rows the MTP caches hold past the kept ones (st.mtp_drafted)
    rows_at: u64 = 0, // its last verify's final-norm rows (bf16 [rows][hidden]), which the next draft reads
};

/// One rank's copy of a kept prompt state: the state arena's bytes and the host fields beside them.
const Snap = struct { buf: cuda.DeviceBuffer, pos: usize, mtp_len: usize, cur: []u1, bases: [2]usize };

/// Where a prompt pass resumes: a kept state (every rank holds `id`) at `at`, a multiple of the chunk rows.
pub const Resume = struct { id: u64, at: usize };

/// Rank 0's marks in a prompt pass: `call` may keep the state at a chunk start; other ranks copy at that chunk's gate.
pub const Marks = struct { at: []const u32, ctx: *anyopaque, call: *const fn (ctx: *anyopaque, at: u32) void };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    c: config.Config,
    share: shard.Share,
    spans: []shard.Span, // every rank's vocabulary span
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    group: ?*const cuda.tp_group.Group = null, // the main stream's all-gathers (RoCE on two ranks); null: lib and comm
    images: ?*images_mod.Images = null, // --vision: a prompt pass's image rows (rank 0 encodes, the others receive)
    profs: [fw.window_rows]?*prof_mod.Prof = @splat(null), // TF_GLM_PROF=1: verify sections' GPU time, by rows
    prompt_profs: [4]?*prof_mod.Prof = @splat(null), // and prompt chunks': to 64, 256, 1024, 2048 rows
    settings: ranks.Settings,
    m: fw.Modules,
    stream: cuda.Stream,
    pieces: fw.Pieces,
    w: weights.Weights,
    bp: fw.Buffers, // prompt chunks
    bd: fw.Buffers, // verify windows
    bm: fw.Buffers, // the MTP head's rows
    bs: ?fw.Buffers = null, // shared verify windows (with the MTP head, as the lane core shares only then)
    hold: cuda.DeviceBuffer, // the prompt's last row's hidden (decode's e.last_hidden), the first draft's source
    pairs: cuda.DeviceBuffer, // every rank's (value, id) of a greedy draw
    smp: sampler.Sampler,
    seqs: std.ArrayList(?*Seq) = .empty, // every stream's sequence by slot (the lane backend's), slot 0 made at init
    cur: *Seq = undefined, // the bound slot's: every step below runs on it
    slot: u32 = 0,
    capacity: usize,
    mtp: bool,
    heat_gate: heat.Gate, // upstream's bands (off unless set), read before each prompt chunk on the gathered maximum
    shared_buf: cuda.DeviceBuffer, // one f64 every rank all-reduces to its maximum
    snaps: std.AutoHashMapUnmanaged(u64, Snap) = .empty, // kept prompt states by id (the same ids on every rank)
    next_snap: u64 = 1, // rank 0: the next id
    announce: u64 = 0, // rank 0: a state kept at this chunk's mark, for the other ranks to copy at its gate
    dropped: std.ArrayList(u32) = .empty, // rank 0: ids dropped since the other ranks last heard (cuda_follow)

    /// Opens rank `rank` of `world` over `comm`; the ranks agree on settings (cuda_ranks) before the weights load.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, checkpoint: []const u8, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, rank: usize, world: usize, o: Options) !*Engine {
        if (world > ranks.max_world) return error.TooManyRanks; // before anything is allocated
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cfg_path = try std.fs.path.join(a, &.{ checkpoint, "config.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(1 << 24));
        var why: modelopt.Why = .{};
        const admitted = if (exl3.isExl3(gpa, text)) exl3.admit(gpa, text, &why) else modelopt.admit(gpa, text, &why);
        admitted catch |err| { // anything but ModelOpt NVFP4 or EXL3 (4-bit mcg routed experts) is refused before loading
            std.log.err("{s}: {s}", .{ cfg_path, why.text() });
            return err;
        };
        const c = try config.parse(gpa, text);
        var stream = try cuda.Stream.init(d, false);
        errdefer stream.deinit();
        const config_hash = try ranks.hashFile(a, io, cfg_path);
        const program_hash = try ranks.hashFile(a, io, "/proc/self/exe");
        var fp = ranks.Fingerprint.of(
            world,
            rank,
            o.settings,
            o.capacity,
            o.mtp,
            config_hash,
            program_hash,
            checkpoint,
        );
        fp.vision = o.vision;
        try ranks.agree(d, lib, comm, stream, fp, try a.alloc(ranks.Fingerprint, world));
        const share = try shard.Share.of(&c, rank, world);
        const spans = try gpa.alloc(shard.Span, world);
        errdefer gpa.free(spans);
        for (spans, 0..) |*sp, q| sp.* = (try shard.Share.of(&c, q, world)).vocab;
        var m = try fw.Modules.load(d, ctx);
        errdefer m.deinit();
        var pieces = try fw.Pieces.init(d, o.settings.pieces);
        errdefer pieces.deinit();
        if (o.settings.layers > c.layers) return error.BadLayers;
        const layers = try a.alloc(u32, if (o.settings.layers > 0) o.settings.layers else c.layers);
        for (layers, 0..) |*l, i| l.* = @intCast(i);
        e.* = .{ .gpa = gpa, .io = io, .heat_gate = .{ .bands = o.settings.heat, .root = heatRoot() }, .shared_buf = undefined, .d = d, .c = c, .share = share, .spans = spans, .lib = lib, .comm = comm, .settings = o.settings, .m = m, .stream = stream, .pieces = pieces, .w = undefined, .capacity = o.capacity, .bp = undefined, .bd = undefined, .bm = undefined, .hold = undefined, .pairs = undefined, .smp = undefined, .mtp = o.mtp };
        e.w = try weights.load(gpa, io, d, stream, e.m.tools(), checkpoint, c, share, layers, o.mtp);
        if (tiledLinears()) { // our bf16 linears (glm_mm) on their tiled weights; TF_GLM_MM=0 keeps the glue's
            try weights.tileLinears(&e.w, &e.m.mm, stream, d);
            e.m.tiled = true;
        }
        errdefer e.w.deinit();
        try e.bind(0);
        errdefer {
            e.free(0);
            e.seqs.deinit(gpa);
        }
        e.bp = try fw.Buffers.init(d, &e.w, fw.prefill_rows, true, o.capacity);
        errdefer e.bp.deinit();
        e.bd = try fw.Buffers.init(d, &e.w, fw.window_rows, false, o.capacity);
        errdefer e.bd.deinit();
        e.bm = try fw.Buffers.init(d, &e.w, fw.window_rows, false, o.capacity);
        errdefer e.bm.deinit();
        if (o.mtp) e.bs = try fw.Buffers.initShared(d, &e.w, shared_rows, o.capacity);
        errdefer if (e.bs) |*b| b.deinit();
        e.hold = try cuda.DeviceBuffer.alloc(d, c.hidden * 2);
        errdefer e.hold.free();
        e.pairs = try cuda.DeviceBuffer.alloc(d, pair_bytes * world);
        errdefer e.pairs.free();
        try d.check(d.api.cuMemsetD8_v2(e.pairs.ptr, 0, pair_bytes * world), "cuMemsetD8"); // each pair's padding stays zero
        e.smp = try sampler.Sampler.init(d, c.vocab, fw.window_rows);
        if (prof_mod.Prof.wanted()) for (&e.profs) |*p| {
            p.* = try prof_mod.Prof.init(d, io);
        };
        if (prof_mod.Prof.wanted()) for (&e.prompt_profs) |*p| {
            p.* = try prof_mod.Prof.init(d, io);
            p.*.?.every = 4;
        };
        errdefer e.smp.deinit();
        e.shared_buf = try cuda.DeviceBuffer.alloc(d, 8);
        try e.stream.synchronize();
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        var it = e.snaps.valueIterator();
        while (it.next()) |s| e.freeSnap(s);
        e.snaps.deinit(e.gpa);
        e.dropped.deinit(e.gpa);
        e.shared_buf.free();
        e.smp.deinit();
        e.pairs.free();
        e.hold.free();
        if (e.bs) |*b| b.deinit();
        e.bm.deinit();
        e.bd.deinit();
        e.bp.deinit();
        for (0..e.seqs.items.len) |i| e.free(i);
        e.seqs.deinit(e.gpa);
        e.w.deinit();
        e.pieces.deinit();
        e.m.deinit();
        e.stream.deinit();
        e.gpa.free(e.spans);
        const gpa = e.gpa;
        gpa.destroy(e);
    }

    fn exchange(e: *const Engine) fw.Exchange {
        return .{ .nccl = .{ .lib = e.lib, .comm = e.comm, .group = e.group } };
    }

    fn view(e: *Engine, b: *fw.Buffers, out: *[1]fw.Rank) fw.Forward {
        out.* = .{.{ .w = &e.w, .st = &e.cur.st, .b = b }};
        const im: ?fw.ImageRows = if (e.images) |v| .{ .v = v, .slot = e.slot } else null;
        return .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = out, .exchange = e.exchange(), .pieces = if (b == &e.bp) &e.pieces else null, .split = e.settings.split, .kda_wide = e.settings.kda_wide, .images = im };
    }

    /// Binds slot `slot`'s sequence for the steps that follow, making it (a zeroed state) on first use.
    pub fn bind(e: *Engine, slot: u32) !void {
        if (slot >= e.seqs.items.len) try e.seqs.appendNTimes(e.gpa, null, slot + 1 - e.seqs.items.len);
        if (e.seqs.items[slot] == null) {
            const q = try e.gpa.create(Seq);
            errdefer e.gpa.destroy(q);
            q.* = .{ .st = try fw.State.init(e.gpa, e.d, &e.w, e.capacity) };
            e.seqs.items[slot] = q;
        }
        e.cur = e.seqs.items[slot].?;
        e.slot = slot;
    }

    /// Frees slot `slot`'s sequence (slot 0 stays: the first stream's, made at init) and binds slot 0.
    pub fn release(e: *Engine, slot: u32) void {
        if (slot != 0) e.free(slot);
        e.cur = e.seqs.items[0].?;
        e.slot = 0;
    }

    fn free(e: *Engine, slot: usize) void {
        const q = e.seqs.items[slot] orelse return;
        e.stream.synchronize() catch {};
        q.st.deinit();
        e.gpa.destroy(q);
        e.seqs.items[slot] = null;
    }

    /// Bytes one more stream's sequence takes on this rank (its state arena).
    pub fn seqBytes(e: *const Engine) u64 {
        return e.seqs.items[0].?.st.arena.buf.len;
    }

    /// The committed length: the pending token sits at this position, a window's row r draws position pos + 1 + r.
    pub fn pos(e: *const Engine) usize {
        return e.cur.st.pos;
    }

    /// A follower's begin: rank 0's next request; no image rows from an earlier one splice into it.
    pub fn beginFollowing(e: *Engine, sampling: ?sampler.Sampling) !void {
        if (e.images) |v| v.clear();
        return e.begin(sampling);
    }

    /// A fresh sequence with `sampling` (null or temperature 0: greedy).
    pub fn begin(e: *Engine, sampling: ?sampler.Sampling) !void {
        e.cur.sampling = sampler.check(sampling);
        e.cur.in_head = 0;
        try e.cur.st.reset(e.stream);
    }

    /// decode.prefill: `ids` in 2,048-row chunks, then the first token from the last row; `from` restores a kept state.
    pub fn prefill(e: *Engine, ids: []const u32, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !u32 {
        var start = try e.prefillStart(ids.len, from);
        while (true) : (start += fw.prefill_rows) {
            if (try e.prefillChunk(ids[start..@min(ids.len, start + fw.prefill_rows + 1)], start, ids.len, null, cancel, marks)) |first| return first;
        }
    }

    /// Where a prompt pass of `len` ids starts: 0, or `from`'s position with its kept state restored.
    fn prefillStart(e: *Engine, len: usize, from: ?Resume) !usize {
        if (len == 0 or len + fw.window_rows >= e.cur.st.capacity) return error.PromptTooLong;
        const r = from orelse return 0;
        if (r.at == 0 or r.at >= len or r.at % fw.prefill_rows != 0) return error.BadResume;
        try e.restoreState(r.id);
        return r.at;
    }

    /// One chunk of a `len`-id prompt pass: `part` from `start` plus the next id; first token at the end, else null.
    pub fn prefillChunk(e: *Engine, part: []const u32, start: usize, len: usize, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !?u32 {
        if ((from != null or start == 0) and try e.prefillStart(len, from) != start) return error.BadChunk;
        if (start >= len or start % fw.prefill_rows != 0) return error.BadChunk;
        const rows = @min(len - start, fw.prefill_rows);
        if (part.len != @min(len - start, rows + 1)) return error.BadChunk;
        var one: [1]fw.Rank = undefined;
        var f = e.view(&e.bp, &one);
        const as_i32: []const i32 = @ptrCast(part);
        if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
        f.prof = e.prompt_profs[if (rows <= 64) 0 else if (rows <= 256) 1 else if (rows <= 1024) 2 else 3];
        const labels = [_][]const u8{ "prompt chunk to 64 rows", "prompt chunk to 256 rows", "prompt chunk to 1024 rows", "prompt chunk to 2048 rows" };
        if (f.prof) |p| try p.begin(e.stream, labels[if (rows <= 64) 0 else if (rows <= 256) 1 else if (rows <= 1024) 2 else 3]);
        defer if (f.prof) |p| p.end();
        try e.gate(cancel);
        if (f.prof) |p| p.mark(e.stream, .gate);
        try f.run(as_i32[0..rows]);
        const last = start + rows == len;
        var first: ?u32 = null;
        if (last) {
            var t: [1]u32 = undefined;
            try e.draw(e.bp.logits, 1, len, &t);
            first = t[0];
        }
        if (f.prof) |p| p.mark(e.stream, .draw);
        if (e.mtp) {
            if (last) try e.copy(e.hold.ptr, e.bp.fnormed + (rows - 1) * e.c.hidden * 2, e.c.hidden * 2);
            const nxt = as_i32[1..];
            if (nxt.len > 0) { // decode._absorb_rows
                const mp = f.prof;
                f.prof = null; // the head's pass as one section
                try f.mtp(nxt, &.{e.bp.fnormed});
                f.prof = mp;
                try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + nxt.len);
            }
        }
        if (f.prof) |p| p.mark(e.stream, .mtp);
        try f.commit(rows);
        if (f.prof) |p| p.mark(e.stream, .commit);
        return first;
    }

    /// The chunk gate on every rank: a kept state at the mark, the heat wait, rank 0's cancel; the id exchanged first.
    fn gate(e: *Engine, cancel: ?Cancel) !void {
        const kept = try e.shared(@floatFromInt(e.announce));
        e.announce = 0;
        if (kept > 0 and e.share.rank != 0) try e.keepAs(@intFromFloat(kept));
        if (e.heat_gate.bands != null) {
            var w: Wait = .{ .e = e, .cancel = cancel };
            const before = e.heat_gate.waited_s;
            e.heat_gate.waitUntil(Wait.read, Wait.sleep, &w, .{ .ptr = &w, .check = Wait.stop }) catch |err| return w.failed orelse err;
            if (e.heat_gate.waited_s > before) heat.note(e.heat_gate.waited_s - before);
        }
        if (try e.shared(if (cancel) |c| @floatFromInt(@intFromBool(c.check(c.ptr))) else 0) > 0) return error.Cancelled;
    }

    /// Bytes one kept state takes on this rank: the whole state arena (its caches are sized for the window, not `at`).
    pub fn stateBytes(e: *const Engine) u64 {
        return e.cur.st.arena.buf.len;
    }

    /// Rank 0, at a mark in a prompt pass: a copy of the state under a new id, named to the other ranks at the gate.
    pub fn keepState(e: *Engine) !u64 {
        const id = e.next_snap;
        try e.keepAs(id);
        e.next_snap += 1;
        e.announce = id;
        return id;
    }

    fn keepAs(e: *Engine, id: u64) !void {
        if (e.snaps.contains(id)) return error.StateKept;
        var buf = try cuda.DeviceBuffer.alloc(e.d, e.cur.st.arena.buf.len);
        errdefer buf.free();
        try e.copy(buf.ptr, e.cur.st.arena.buf.ptr, buf.len);
        try e.stream.synchronize();
        const cur = try e.gpa.dupe(u1, e.cur.st.cur);
        errdefer e.gpa.free(cur);
        try e.snaps.put(e.gpa, id, .{ .buf = buf, .pos = e.cur.st.pos, .mtp_len = e.cur.st.mtp_len, .cur = cur, .bases = .{ e.cur.st.base, e.cur.st.mtp_base } });
    }

    /// The live state made kept state `id`'s; a rank without it refuses rather than prefill from another position.
    fn restoreState(e: *Engine, id: u64) !void {
        const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
        try e.copy(e.cur.st.arena.buf.ptr, s.buf.ptr, s.buf.len);
        try e.cur.st.restoreHost(e.stream, s.pos, s.mtp_len, s.cur, s.bases);
        e.cur.in_head = 0;
        try e.stream.synchronize();
    }

    pub fn holds(e: *const Engine, id: u64) bool {
        return e.snaps.contains(id);
    }

    /// Frees kept state `id`; rank 0 also notes it for the other ranks (cuda_follow sends it before its next step).
    pub fn dropState(e: *Engine, id: u64) void {
        var kv = e.snaps.fetchRemove(id) orelse return;
        e.freeSnap(&kv.value);
        if (e.share.rank == 0 and e.share.world > 1) e.dropped.append(e.gpa, @intCast(id)) catch
            std.log.warn("kept state {d}: the other ranks keep their copy until they exit (no memory for the note)", .{id});
    }

    fn freeSnap(e: *Engine, s: *Snap) void {
        s.buf.free();
        e.gpa.free(s.cur);
    }

    /// The ranks' maximum of `v` (an all-reduce on the engine's stream).
    fn shared(e: *Engine, v: f64) !f64 {
        var x = v;
        try e.shared_buf.uploadAsync(0, std.mem.asBytes(&x), e.stream.handle);
        try e.lib.check(e.lib.api.ncclAllReduce(e.shared_buf.ptr, e.shared_buf.ptr, 1, .f64, .max, e.comm, e.stream.handle), "ncclAllReduce");
        try e.stream.synchronize();
        try e.shared_buf.download(0, std.mem.asBytes(&x));
        return x;
    }

    /// heat.Gate.waitUntil's hooks over the ranks: gathered max reading, rank 0's stop: all pause together.
    const Wait = struct {
        e: *Engine,
        cancel: ?Cancel,
        failed: ?anyerror = null,

        fn read(ptr: *anyopaque) anyerror!f64 {
            const w: *Wait = @ptrCast(@alignCast(ptr));
            const local: f64 = blk: {
                var dir = std.Io.Dir.cwd().openDir(w.e.io, w.e.heat_gate.root, .{ .iterate = true }) catch break :blk -std.math.inf(f64);
                defer dir.close(w.e.io);
                break :blk heat.hottest(w.e.io, dir) catch -std.math.inf(f64);
            };
            return w.e.shared(local);
        }

        fn sleep(ptr: *anyopaque) void {
            const w: *Wait = @ptrCast(@alignCast(ptr));
            std.Io.sleep(w.e.io, .fromMilliseconds(2000), .awake) catch {};
        }

        fn stop(ptr: *anyopaque) bool {
            const w: *Wait = @ptrCast(@alignCast(ptr));
            const mine: f64 = if (w.cancel) |c| @floatFromInt(@intFromBool(c.check(c.ptr))) else 0;
            const any = w.e.shared(mine) catch |err| {
                w.failed = err;
                return true;
            };
            return any > 0;
        }
    };

    /// A verify window: `ids` (pending token, then drafts) at the committed position; `out[r]` drawn at pos + 1 + r.
    pub fn verify(e: *Engine, ids: []const u32, out: []u32) !void {
        if (ids.len == 0 or ids.len > fw.window_rows) return error.WindowTooWide;
        var one: [1]fw.Rank = undefined;
        var f = e.view(&e.bd, &one);
        const at = e.cur.st.pos;
        f.prof = e.profs[ids.len - 1];
        if (f.prof) |p| try p.begin(e.stream, verifyLabel(ids.len));
        try f.run(@ptrCast(ids));
        e.cur.rows_at = e.bd.fnormed;
        try e.draw(e.bd.logits, ids.len, at + 1, out);
        if (f.prof) |p| p.end();
    }

    fn verifyLabel(rows: usize) []const u8 {
        const l = [_][]const u8{ "verify 1 row", "verify 2 rows", "verify 3 rows", "verify 4 rows", "verify 5 rows", "verify 6 rows", "verify 7 rows", "verify 8 rows" };
        return l[@min(rows, l.len) - 1];
    }

    /// Several streams' windows in one forward: shared row ops, per-part attention and KDA.
    pub fn verifyShared(e: *Engine, parts: []const Part) !void {
        const b = if (e.bs) |*x| x else return error.NoSharedWindows;
        if (parts.len == 0 or parts.len > max_shared) return error.WindowTooWide;
        var segs: [max_shared]fw.Seg = undefined;
        var ids: [shared_rows]u32 = undefined;
        var rows: usize = 0;
        for (parts, 0..) |p, i| {
            if (p.ids.len == 0 or p.ids.len > fw.window_rows or rows + p.ids.len > shared_rows) return error.WindowTooWide;
            for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
            try e.bind(p.slot);
            segs[i] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
            @memcpy(ids[rows..][0..p.ids.len], p.ids);
            rows += p.ids.len;
        }
        var one: [1]fw.Rank = .{.{ .w = &e.w, .st = segs[0].st, .b = b, .segs = segs[0..parts.len] }};
        const f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .pieces = null, .split = e.settings.split, .kda_wide = e.settings.kda_wide };
        try f.run(@ptrCast(ids[0..rows]));
        for (parts, segs[0..parts.len]) |p, g| {
            try e.bind(p.slot);
            e.cur.rows_at = b.fnormed + g.row0 * e.c.hidden * 2;
            try e.draw(b.logits + g.row0 * e.share.vocab.len() * 2, g.rows, g.st.pos + 1, p.out);
        }
    }

    /// forward.commit of the last window's first `n` rows (the KDA replay when fewer than it ran).
    pub fn keep(e: *Engine, n: usize) !void {
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bd, &one);
        try f.commit(n);
    }

    /// decode.draft: the MTP head absorbs the kept rows (committed first) with `follow`, then chains `depth` drafts.
    pub fn draft(e: *Engine, from_prompt: bool, follow: []const u32, depth: usize, out: []u32) !void {
        if (!e.mtp) return error.NoDraftHead;
        if (depth > max_drafts) return error.TooDeep;
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bm, &one);
        const D = e.c.hidden;
        const hidden = if (from_prompt) e.hold.ptr else e.cur.rows_at;
        try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len - e.cur.in_head);
        e.cur.in_head = 0;
        const next: []const i32 = @ptrCast(follow);
        var s0: usize = 0;
        while (s0 < next.len) : (s0 += fw.window_rows) {
            const part = next[s0..@min(next.len, s0 + fw.window_rows)];
            try f.mtp(part, &.{hidden + s0 * D * 2});
            try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + part.len);
        }
        for (0..depth) |j| {
            try e.draw(e.bm.logits, 1, e.cur.st.pos + 1 + j, out[j..][0..1]);
            if (j + 1 < depth) {
                try f.mtp(@ptrCast(out[j..][0..1]), &.{e.bm.fnormed});
                try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + 1);
                e.cur.in_head += 1;
            }
        }
    }

    /// `rows` logits rows of this rank's vocabulary -> the tokens at positions first, first + 1, ...
    fn draw(e: *Engine, logits: u64, rows: usize, first: u64, out: []u32) !void {
        if (e.cur.sampling) |s| {
            try e.smp.draw(e.stream, e.exchange(), e.spans, &.{.{ .rank = e.share.rank, .logits = logits }}, rows, first, sampler.rule(s), out);
            return e.checkGathers(); // the draw read rows the forward's gathers made
        }
        for (0..rows) |r| out[r] = try e.greedy(logits + r * e.share.vocab.len() * 2);
    }

    /// decode.sample_rows' greedy pick: per-rank top-1 (tf_glm_top1), all-gathered, largest value then lowest id.
    fn greedy(e: *Engine, logits: u64) !u32 {
        const world = e.share.world;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(i64, @intCast(e.share.vocab.len())));
        a.add(@as(u32, @intCast(e.share.vocab.lo)));
        a.add(e.pairs.ptr + e.share.rank * pair_bytes);
        try cuda.launch.launch(e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
        const mine = e.pairs.ptr + e.share.rank * pair_bytes;
        if (e.group) |g| try g.allGather(e.stream, mine, e.pairs.ptr, pair_bytes) else try e.lib.check(e.lib.api.ncclAllGather(mine, e.pairs.ptr, pair_bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
        try e.stream.synchronize();
        try e.checkGathers();
        var all: [ranks.max_world][4]u32 = undefined; // Engine.init refuses a larger world
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&all, e.pairs.ptr, world * pair_bytes), "cuMemcpyDtoH");
        var pick = all[0];
        for (all[1..world]) |p| {
            const v: f32 = @bitCast(p[0]);
            const b: f32 = @bitCast(pick[0]);
            if (v > b or (v == b and p[1] < pick[1])) pick = p;
        }
        return pick[1];
    }

    /// A failed RoCE gather's error, after a synchronize whose result read gathered rows (they hold nothing then).
    fn checkGathers(e: *const Engine) !void {
        if (e.group) |g| try g.check();
    }

    /// TF_GLM_MM=0 keeps the bf16 linears on the glue's b16 kernels (the weights stay as the checkpoint stores them).
    fn tiledLinears() bool {
        const v = std.c.getenv("TF_GLM_MM") orelse return true;
        return !std.mem.eql(u8, std.mem.span(v), "0");
    }

    fn heatRoot() []const u8 {
        const p = std.c.getenv("TF_HEAT_ROOT") orelse return heat.default_root;
        const s = std.mem.span(p);
        return if (s.len == 0) heat.default_root else s;
    }

    fn copy(e: *const Engine, dst: u64, src: u64, bytes: usize) !void {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, e.stream.handle), "cuMemcpyDtoDAsync");
    }
};
