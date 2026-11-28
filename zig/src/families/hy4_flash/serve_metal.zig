//! Hy4-flash behind the lane engine on Metal
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const target = @import("metal_model.zig");
const dfly = @import("metal_draft.zig");
const index = @import("metal_index.zig");
const engram = @import("engram.zig");
const be = lanes.backend;

pub const key_width = 576; // BF16 words in a latent KV slot: 512 latent + 64 rope
pub const window: u32 = 512; // the sliding layers' keys
pub const prompt_rows: u32 = 512; // a prompt chunk's rows
pub const max_window: u32 = 6; // the pending row and DFly2's five drafts
pub const draft_stride: u32 = 2560; // a draft row's keys at most: a recalled list (2,048) and its own rows

pub const Options = struct {
    capacity: u32, // slots a stream's caches hold (a multiple of 512)
};

/// One stream's caches.
pub const Cache = struct {
    kv: []mtl.Buffer,
    token_keys: []?mtl.Buffer,
    group_keys: []?mtl.Buffer,
    len: u32 = 0, // slots written and kept
    hist: std.ArrayList(u32) = .empty, // the token at each kept slot
    held: [max_window]u32 = @splat(0),
    nheld: u32 = 0,
    logits: []u16, // the last pass's last row
    anchor: u32 = 0, // the row of the last pass whose final hidden the drafter reads (with no kept rows given)
    selected: bool = false, // the last pass's sparse layers attended recalled lists

    fn deinit(self: *Cache, gpa: std.mem.Allocator) void {
        for (self.kv) |b| b.deinit();
        for (self.token_keys) |b| if (b) |x| x.deinit();
        for (self.group_keys) |b| if (b) |x| x.deinit();
        gpa.free(self.kv);
        gpa.free(self.token_keys);
        gpa.free(self.group_keys);
        gpa.free(self.logits);
        self.hist.deinit(gpa);
    }
};

pub const Metal = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    model: target.Model,
    draft: dfly.Draft,
    frame: target.Frame,
    dframe: dfly.Frame,
    capacity: u32,
    vocab: u32,
    pad: u32,
    mask_token: u32,
    sliding: []bool,
    lanes: []target.LayerCache,
    positions: mtl.Buffer,
    writes: mtl.Buffer,
    slide_keys: mtl.Buffer,
    slide_counts: mtl.Buffer,
    sparse_keys: mtl.Buffer,
    sparse_counts: mtl.Buffer,
    slide_ranges: mtl.Buffer,
    sparse_ranges: mtl.Buffer,
    tiles: mtl.Buffer,
    scores: mtl.Buffer,
    ix_tokens: []?mtl.Buffer,
    ix_counts: []?mtl.Buffer,
    d_ids: mtl.Buffer,
    d_tokens: mtl.Buffer,
    d_positions: mtl.Buffer,
    d_writes: mtl.Buffer,
    d_keys: []mtl.Buffer,
    d_counts: []mtl.Buffer,
    d_caches: []target.LayerCache,
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *Cache) = .empty,
    kept: ?*Cache = null, // the last released stream's caches, for a prompt that shares their prefix
    costs: [max_window]lanes.allocate.Cost = undefined,
    draft_ms: f64 = 0,
    /// The residency set's hook: caches made after the set is built join it.
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, options: Options) !*Metal {
        if (options.capacity < 2048 or options.capacity % 512 != 0 or options.capacity / 4 > 65536) return error.BadHy4Capacity;
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        const device = try mtl.Device.init();
        errdefer device.deinit();
        const queue = try device.queue();
        errdefer queue.deinit();
        out.* = undefined;
        out.gpa = gpa;
        out.device = device;
        out.queue = queue;
        out.capacity = options.capacity;
        out.caches = .empty;
        out.kept = null;
        out.draft_ms = 0;
        out.follow = null;
        out.model = try target.Model.open(gpa, io, device, dir);
        errdefer out.model.deinit();
        const t = out.model.host.config.value.text_config;
        out.vocab = t.vocab_size;
        out.pad = t.engram_pad_id;
        out.mask_token = out.model.host.config.value.dfly_mask_token_id;
        out.draft = try dfly.Draft.open(gpa, device, &out.model.host);
        errdefer out.draft.deinit();
        if (out.draft.width > max_window) return error.Hy4DraftTooWide;
        out.frame = try target.Frame.init(gpa, device, &out.model, prompt_rows, index.keys_a_row);
        errdefer out.frame.deinit();
        out.dframe = try dfly.Frame.init(gpa, device, &out.model, &out.draft, 1, draft_stride);
        errdefer out.dframe.deinit();
        const layers = out.model.blocks.len;
        const opts = mtl.ResourceOptions.shared;
        const C = options.capacity;
        out.positions = try device.buffer(prompt_rows * 3 * 4, opts);
        out.writes = try device.buffer(prompt_rows * 4, opts);
        out.slide_keys = try device.buffer(prompt_rows * window * 4, opts);
        out.slide_counts = try device.buffer(prompt_rows * 4, opts);
        out.sparse_keys = try device.buffer(prompt_rows * index.keys_a_row * 4, opts);
        out.sparse_counts = try device.buffer(prompt_rows * 4, opts);
        out.slide_ranges = try device.buffer(prompt_rows * 8, opts);
        out.sparse_ranges = try device.buffer(prompt_rows * 8, opts);
        out.tiles = try device.buffer(C / 32 * 4, opts);
        for (out.tiles.slice(u32, C / 32), 0..) |*x, i| x.* = @intCast(32 * i);
        out.scores = try device.buffer(@as(usize, prompt_rows) * (C / 4) * 4, mtl.ResourceOptions.private);
        out.sliding = try gpa.alloc(bool, layers);
        out.lanes = try gpa.alloc(target.LayerCache, layers);
        out.ix_tokens = try gpa.alloc(?mtl.Buffer, layers);
        out.ix_counts = try gpa.alloc(?mtl.Buffer, layers);
        for (0..layers) |l| {
            const slide = t.layer_types[l] == .sliding_attention;
            out.sliding[l] = slide;
            out.ix_tokens[l] = if (slide) null else try device.buffer(@as(usize, prompt_rows) * index.keys_a_row * 4, opts);
            out.ix_counts[l] = if (slide) null else try device.buffer(prompt_rows * 4, opts);
            out.lanes[l] = .{
                .rows = .{
                    .positions = .{ .buffer = out.positions },
                    .write_slots = .{ .buffer = out.writes },
                    .retained = .{ .buffer = if (slide) out.slide_keys else out.sparse_keys },
                    .counts = .{ .buffer = if (slide) out.slide_counts else out.sparse_counts },
                    .cache = .{ .buffer = out.positions }, // bound to a stream's caches before each pass
                    .capacity = C,
                    .ranges = .{ .ranges = .{ .buffer = if (slide) out.slide_ranges else out.sparse_ranges }, .tiles = .{ .buffer = out.tiles } },
                },
                .key_stride = if (slide) window else index.keys_a_row,
            };
        }
        const w = out.draft.width;
        out.d_ids = try device.buffer(w * 4, opts);
        out.d_tokens = try device.buffer(16 * 4, opts);
        out.d_positions = try device.buffer(w * 3 * 4, opts);
        out.d_writes = try device.buffer(w * 4, opts);
        const blocks = out.draft.blocks.len;
        out.d_keys = try gpa.alloc(mtl.Buffer, blocks);
        out.d_counts = try gpa.alloc(mtl.Buffer, blocks);
        out.d_caches = try gpa.alloc(target.LayerCache, blocks);
        for (out.d_keys, out.d_counts, out.d_caches) |*k, *c, *dc| {
            k.* = try device.buffer(@as(usize, w) * draft_stride * 4, opts);
            c.* = try device.buffer(w * 4, opts);
            dc.* = .{ .rows = .{
                .positions = .{ .buffer = out.d_positions },
                .write_slots = .{ .buffer = out.d_writes },
                .retained = .{ .buffer = k.* },
                .counts = .{ .buffer = c.* },
                .cache = .{ .buffer = out.positions },
                .capacity = C,
            }, .key_stride = draft_stride };
        }
        return out;
    }

    pub fn deinit(self: *Metal) void {
        var it = self.caches.valueIterator();
        while (it.next()) |c| {
            c.*.deinit(self.gpa);
            self.gpa.destroy(c.*);
        }
        self.caches.deinit(self.gpa);
        if (self.kept) |c| {
            c.deinit(self.gpa);
            self.gpa.destroy(c);
        }
        for (self.d_keys) |b| b.deinit();
        for (self.d_counts) |b| b.deinit();
        for (self.ix_tokens) |b| if (b) |x| x.deinit();
        for (self.ix_counts) |b| if (b) |x| x.deinit();
        for ([_]mtl.Buffer{ self.positions, self.writes, self.slide_keys, self.slide_counts, self.sparse_keys, self.sparse_counts, self.slide_ranges, self.sparse_ranges, self.tiles, self.scores, self.d_ids, self.d_tokens, self.d_positions, self.d_writes }) |b| b.deinit();
        self.gpa.free(self.d_keys);
        self.gpa.free(self.d_counts);
        self.gpa.free(self.d_caches);
        self.gpa.free(self.ix_tokens);
        self.gpa.free(self.ix_counts);
        self.gpa.free(self.lanes);
        self.gpa.free(self.sliding);
        self.dframe.deinit();
        self.frame.deinit();
        self.draft.deinit();
        self.model.deinit();
        self.queue.deinit();
        self.device.deinit();
        self.gpa.destroy(self);
    }

    /// Every Metal buffer the model, drafter, frames, pass state and kept caches hold, for one residency set.
    pub fn residentBuffers(self: *Metal, list: *std.ArrayList(mtl.Buffer)) !void {
        return @import("serve_resident.zig").buffers(self, list);
    }

    /// Bytes one stream's caches take at `capacity` slots.
    pub fn cacheBytes(self: *const Metal, capacity: u32) u64 {
        var sparse: u64 = 0;
        for (self.sliding) |s| sparse += @intFromBool(!s);
        return @as(u64, self.sliding.len) * capacity * key_width * 2 + sparse * (index.Cache.tokenBytes(capacity) + index.Cache.groupBytes(capacity));
    }

    fn newCache(self: *Metal) !*Cache {
        const c = try self.gpa.create(Cache);
        errdefer self.gpa.destroy(c);
        const layers = self.sliding.len;
        const opts = mtl.ResourceOptions.private;
        const kv = try self.gpa.alloc(mtl.Buffer, layers);
        errdefer self.gpa.free(kv);
        const tk = try self.gpa.alloc(?mtl.Buffer, layers);
        errdefer self.gpa.free(tk);
        const gk = try self.gpa.alloc(?mtl.Buffer, layers);
        errdefer self.gpa.free(gk);
        @memset(tk, null);
        @memset(gk, null);
        var made: usize = 0;
        errdefer for (kv[0..made], tk[0..made], gk[0..made]) |b, x, y| {
            b.deinit();
            if (x) |v| v.deinit();
            if (y) |v| v.deinit();
        };
        for (0..layers) |l| {
            kv[l] = try self.device.buffer(@as(usize, self.capacity) * key_width * 2, opts);
            if (!self.sliding[l]) {
                tk[l] = try self.device.buffer(index.Cache.tokenBytes(self.capacity), opts);
                gk[l] = try self.device.buffer(index.Cache.groupBytes(self.capacity), opts);
            }
            made += 1;
        }
        c.* = .{ .kv = kv, .token_keys = tk, .group_keys = gk, .logits = try self.gpa.alloc(u16, self.vocab) };
        // touch every page now: a fresh buffer's first GPU writes otherwise cost seconds inside a request
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = self.queue.commandBuffer();
        const blit = cb.blit();
        for (0..layers) |l| {
            blit.fill(kv[l], 0, @as(usize, self.capacity) * key_width * 2, 0);
            if (tk[l]) |b| blit.fill(b, 0, index.Cache.tokenBytes(self.capacity), 0);
            if (gk[l]) |b| blit.fill(b, 0, index.Cache.groupBytes(self.capacity), 0);
        }
        blit.end();
        cb.commit();
        cb.wait();
        if (self.follow) |f| for (0..layers) |l| {
            f.join(f.ctx, kv[l]);
            if (tk[l]) |b| f.join(f.ctx, b);
            if (gk[l]) |b| f.join(f.ctx, b);
        };
        return c;
    }

    /// Point the layers' caches at `c` for the next pass.
    pub fn bind(self: *Metal, c: *const Cache) void {
        for (self.lanes, 0..) |*lane, l| {
            lane.rows.cache = .{ .buffer = c.kv[l] };
            lane.index = if (self.sliding[l]) null else .{
                .token_keys = .{ .buffer = c.token_keys[l].? },
                .group_keys = .{ .buffer = c.group_keys[l].? },
                .capacity = self.capacity,
                .scores = .{ .buffer = self.scores },
                .score_groups = self.capacity / 4,
                .tokens = .{ .buffer = self.ix_tokens[l].? },
                .counts = .{ .buffer = self.ix_counts[l].? },
            };
        }
        for (self.d_caches, self.draft.blocks) |*dc, block| dc.rows.cache = .{ .buffer = c.kv[block.target_kv] };
    }

    /// Engram windows and ids for `tokens` at slots start.. (the cache's history holds the tokens before start).
    fn stage(self: *Metal, c: *const Cache, start: u32, tokens: []const u32) !void {
        const n = tokens.len;
        var windows: [prompt_rows][engram.order]u32 = undefined;
        var media: [prompt_rows]u32 = @splat(0);
        for (tokens, 0..) |tk, r| {
            const pos = start + @as(u32, @intCast(r));
            windows[r][0] = tk;
            for (1..engram.order) |lag| {
                windows[r][lag] = if (lag <= r) tokens[r - lag] else if (lag <= pos) c.hist.items[pos - lag] else self.pad;
            }
        }
        try self.frame.stage(&self.model, tokens, windows[0..n], media[0..n]);
    }

    /// Positions, write slots and the key lists and ranges of `n` rows at start.., and the indexer's rows.
    fn layout(self: *Metal, start: u32, n: u32) !void {
        const pos3 = self.positions.slice(i32, prompt_rows * 3);
        const wr = self.writes.slice(i32, prompt_rows);
        const last_end = start + n;
        try self.frame.index.stage(start, n);
        const lists = self.frame.index.selects();
        const sparse_stride: u32 = if (lists) index.keys_a_row else ((last_end + window - 1) / window) * window;
        const sk = self.slide_keys.slice(u32, prompt_rows * window);
        const sc = self.slide_counts.slice(u32, prompt_rows);
        const pk = self.sparse_keys.slice(u32, prompt_rows * index.keys_a_row);
        const pc = self.sparse_counts.slice(u32, prompt_rows);
        for (0..n) |r| {
            const pos = start + @as(u32, @intCast(r));
            const end = pos + 1;
            pos3[r * 3 ..][0..3].* = @splat(@intCast(pos));
            wr[r] = @intCast(pos);
            const count = @min(end, window);
            for (0..count) |j| sk[r * window + j] = end - count + @as(u32, @intCast(j));
            sc[r] = count;
            if (!lists) {
                for (0..end) |j| pk[r * sparse_stride + j] = @intCast(j);
                pc[r] = end;
            }
            self.slide_ranges.slice([2]u32, prompt_rows)[r] = .{ end - count, end };
            self.sparse_ranges.slice([2]u32, prompt_rows)[r] = .{ 0, end };
        }
        for (self.lanes, self.sliding) |*lane, slide| {
            lane.key_stride = if (slide) window else sparse_stride;
            lane.rows.live_keys = if (slide) @min(last_end, window) else if (lists) index.keys_a_row else last_end;
        }
    }

    /// One target pass over `tokens` at the cache's end: every row's logits when `all`, else the last row's.
    fn pass(self: *Metal, c: *Cache, tokens: []const u32, all: bool) !void {
        const n: u32 = @intCast(tokens.len);
        if (n == 0 or n > prompt_rows) return error.BadHy4Pass;
        const start = c.len;
        if (start + n > self.capacity) return error.Hy4ContextFull;
        self.bind(c);
        try self.stage(c, start, tokens);
        try self.layout(start, n);
        self.frame.last_logits = !all and n > 1;
        defer self.frame.last_logits = false;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = self.queue.commandBuffer();
        const e = cb.compute(.concurrent);
        _ = self.model.encode(e, &self.frame, self.lanes) catch |err| {
            e.end();
            return err;
        };
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.log.err("hy4: target pass failed: {s}", .{why});
            return error.Hy4GpuFailed;
        }
        c.len += n;
        try c.hist.appendSlice(self.gpa, tokens);
        c.anchor = n - 1;
        c.selected = self.frame.index.selects();
        const rows: usize = if (self.frame.last_logits) 1 else n;
        @memcpy(c.logits, self.frame.logits.buffer.slice(u16, rows * self.vocab)[(rows - 1) * self.vocab ..][0..self.vocab]);
    }

    fn logitsRow(self: *const Metal, r: usize) []const u16 {
        return self.frame.logits.buffer.slice(u16, (r + 1) * self.vocab)[r * self.vocab ..][0..self.vocab];
    }

    pub fn facts(self: *const Metal) lanes.Model {
        return .{
            .exact_width = max_window,
            .mtp = true,
            .speculate = true,
            .speculate_early = false,
            .drafts = self.draft.width - 1,
            .window_costs = &self.costs,
            .mtp_step_ms = self.draft_ms,
            .max_streams = 1,
            .streams_exact = true,
        };
    }

    pub fn backend(self: *Metal) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .first_row = firstRowFn, .held = heldFn } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, s: *lanes.Stream) !*Cache {
        return self.caches.get(s) orelse error.UnknownHy4Stream;
    }

    /// A prompt chunk, the verify widths and a draft once, timing each width for the lane core's depth rule.
    pub fn warm(self: *Metal) !void {
        const c = try self.newCache();
        errdefer {
            c.deinit(self.gpa);
            self.gpa.destroy(c);
        }
        var ids: [prompt_rows]u32 = undefined;
        for (&ids, 0..) |*x, i| x.* = @intCast(1000 + i * 37 % 20000);
        ids[0] = 120000;
        try self.pass(c, ids[0..prompt_rows], false);
        for (1..max_window + 1) |width| {
            var times: [3]f64 = undefined;
            for (&times) |*tm| {
                const t0 = mtl.clock.seconds();
                try self.pass(c, ids[0..width], true);
                tm.* = mtl.clock.seconds() - t0;
                c.len -= @intCast(width);
                c.hist.items.len -= width;
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            self.costs[width - 1] = .{ .width = @intCast(width), .ms = 1000 * times[1] };
        }
        try self.runDraft(c, c.anchor, ids[1], 3); // the first draft compiles its pipelines
        const t0 = mtl.clock.seconds();
        try self.runDraft(c, c.anchor, ids[1], 3);
        self.draft_ms = 1000 * (mtl.clock.seconds() - t0) / 3;
        // every page of the caches touched once, then kept for the first stream
        c.len = 0;
        c.hist.items.len = 0;
        c.nheld = 0;
        self.kept = c;
        std.log.info("hy4: window ms by width {d:.1} {d:.1} {d:.1} {d:.1} {d:.1} {d:.1}, draft {d:.2} ms a token", .{ self.costs[0].ms, self.costs[1].ms, self.costs[2].ms, self.costs[3].ms, self.costs[4].ms, self.costs[5].ms, self.draft_ms });
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = s.prompt();
        if (ids.len == 0 or self.caches.contains(s)) return error.InvalidHy4Prompt;
        if (ids.len + s.max_new + max_window > self.capacity) return error.Hy4ContextFull;
        // the kept caches, when this prompt starts with their tokens (keep at least one prompt token to run)
        var c: *Cache = undefined;
        var reused: u32 = 0;
        if (self.kept) |k| {
            var same: usize = 0;
            const most = @min(k.hist.items.len, ids.len - 1);
            while (same < most and k.hist.items[same] == ids[same]) same += 1;
            // the caches are reused either way (fresh pages cost seconds to first touch): only the shared prefix stays
            if (same < 64) same = 0;
            c = k;
            self.kept = null;
            c.len = @intCast(same);
            c.hist.items.len = same;
            c.nheld = 0;
            reused = @intCast(same);
        } else c = try self.newCache();
        errdefer {
            c.deinit(self.gpa);
            self.gpa.destroy(c);
        }
        try self.caches.put(self.gpa, s, c);
        errdefer _ = self.caches.remove(s);
        s.cached = reused;
        var at: usize = reused;
        while (at < ids.len) {
            if (s.isCancelled()) return error.Cancelled;
            const end = @min(ids.len, at + prompt_rows);
            try self.pass(c, ids[at..end], false);
            lanes.prefill_widths.completed(self.gpa, s, end - at);
            at = end;
        }
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const c = try self.cacheOf(s);
        if (position != c.len) return error.InvalidHy4Position;
        return try lanes.sampling_row.draw(self.gpa, c.logits, s.sampling, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const c = try self.cacheOf(s);
        if (position != c.len) return error.InvalidHy4Position;
        return try lanes.sampling_row.draw(self.gpa, c.logits, s.sampling, position, mask);
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, f: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const c = try self.cacheOf(s);
        if (position != c.len + 1) return error.InvalidHy4Position;
        const token: u32 = switch (f) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        c.nheld = 0;
        try self.pass(c, &.{token}, false);
        return try lanes.sampling_row.draw(self.gpa, c.logits, s.sampling, position, &.{});
    }

    fn readFn(ptr: *anyopaque, handle: u64) !u32 {
        if (handle >= cast(ptr).vocab) return error.InvalidHy4Token;
        return @intCast(handle);
    }

    fn firstRowFn(ptr: *anyopaque, s: *lanes.Stream) !lanes.LogprobRow {
        const self = cast(ptr);
        const l = (try self.cacheOf(s)).logits;
        return row(l, @intCast(argmax(l)), s.logprobs orelse 0);
    }

    /// The window's pending token and drafts in one pass, every row drawn; `keep` rolls the rest back.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        const words = (self.vocab + 31) / 32;
        if (windows.len != out.len) return error.InvalidHy4Windows;
        for (windows, out) |w, o| {
            const rows = w.rows();
            if (rows > max_window or w.parents != null or o.sampled.len != rows or o.drafts.len != rows - 1) return error.UnsupportedHy4Window;
            const c = try self.cacheOf(w.stream);
            if (w.held > c.nheld) return error.Hy4DraftsNotHeld;
            for (w.positions, 0..) |p, r| if (p != c.len + r + 1) return error.InvalidHy4Position;
            var ids: [max_window]u32 = undefined;
            ids[0] = w.pending;
            @memcpy(ids[1..][0..w.held], c.held[0..w.held]);
            @memcpy(ids[1 + w.held ..][0..w.tokens.len], w.tokens);
            c.nheld = 0;
            try self.pass(c, ids[0..rows], true);
            @memcpy(o.drafts, ids[1..rows]);
            for (0..rows) |r| {
                const l = self.logitsRow(r);
                const mask = if (w.masks.len >= (r + 1) * words) w.masks[r * words ..][0..words] else &.{};
                o.sampled[r] = try lanes.sampling_row.draw(self.gpa, l, w.stream.sampling, w.positions[r], mask);
                if (o.rows.len == rows) o.rows[r] = row(l, o.sampled[r], w.stream.logprobs orelse 0);
            }
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        for (windows, paths) |w, path| {
            const c = try self.cacheOf(w.stream);
            for (path, 0..) |r, i| if (r != i) return error.UnsupportedHy4Tree;
            const rows: u32 = @intCast(w.rows());
            if (path.len == 0 or path.len > rows or rows > c.len) return error.UnsupportedHy4Window;
            c.len = c.len - rows + @as(u32, @intCast(path.len));
            c.hist.items.len = c.len;
            c.anchor = @intCast(path.len - 1);
        }
    }

    /// DFly2 from the anchor row's final hidden
    pub fn runDraft(self: *Metal, c: *Cache, anchor_row: u32, bonus: u32, depth: u32) !void {
        return @import("serve_draft.zig").run(self, c, anchor_row, bonus, depth);
    }

    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        const self = cast(ptr);
        for (requests) |r| {
            const c = try self.cacheOf(r.stream);
            c.nheld = 0;
            var anchor_row = c.anchor;
            if (r.rows) |rows| {
                if (rows.len == 0) return error.UnsupportedHy4Window;
                for (rows, 0..) |x, i| if (x != i) return error.UnsupportedHy4Tree;
                anchor_row = rows[rows.len - 1];
            }
            const bonus: u32 = if (r.follow.len > 0) r.follow[r.follow.len - 1] else switch (r.first orelse return error.NoFirstToken) {
                .value => |v| v,
                .handle => |hd| @intCast(hd),
            };
            if (r.position != @as(u64, c.len) + 1) return error.InvalidHy4Position;
            try self.runDraft(c, anchor_row, bonus, r.depth);
        }
    }

    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) !void {
        const c = try cast(ptr).cacheOf(s);
        if (out.len > c.nheld) return error.Hy4DraftsNotHeld;
        @memcpy(out, c.held[0..out.len]);
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = cast(ptr);
        const entry = self.caches.fetchRemove(s) orelse return;
        if (self.kept) |k| {
            k.deinit(self.gpa);
            self.gpa.destroy(k);
        }
        entry.value.nheld = 0;
        self.kept = entry.value;
    }
};

fn argmax(l: []const u16) usize {
    var best: usize = 0;
    for (l, 0..) |w, i| if (@as(f32, @bitCast(@as(u32, w) << 16)) > @as(f32, @bitCast(@as(u32, l[best]) << 16))) {
        best = i;
    };
    return best;
}

const row = lanes.logprob.bf16Row;
