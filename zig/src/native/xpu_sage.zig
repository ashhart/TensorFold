//! Sage-A1.5 (qwen3_5_moe) on the lane host: a stream per runner slot, one batched decode step over the running slots.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const xq = @import("xpu_qwen27");
const lane = @import("xpu_lane.zig");
const sage = xq.sage;
const vl = @import("core").qwen_vl;
const be = lanes.backend;
const Allocator = std.mem.Allocator;

pub const model_type = "qwen3_5_moe";

/// Drawn tokens a handle can still name (a prompt's first draw, then one a stream a round).
const ring = 64;

pub const Slots = struct {
    gpa: Allocator,
    r: *sage.Runner,
    owners: [sage.MAX_SLOTS]?*lanes.Stream = @splat(null),
    half: []f16, // a row of logits as the runner reads it
    row: []f32, // and widened for the host draw
    ids: []i32, // a prompt chunk as the runner takes it
    first_rows: [sage.MAX_SLOTS]?[]f32 = @splat(null), // a structured stream's prompt-row logits, for first_masked
    masked_vals: ?[]f64 = null, // a structured sampled row's allowed candidates
    masked_ids: ?[]u64 = null,
    fed: [sage.MAX_SLOTS]u32 = @splat(1), // rows the last verify fed each slot
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    vision: ?*Vision = null, // --vision
    depth: u32 = 1, // MTP drafts a lone round (XPU_SPEC; 1 keeps 80 % of drafts on chat text)
    held: [sage.MAX_SLOTS]u32 = @splat(0), // drafts a slot holds for its next window (specMulti makes them then)

    fn of(ptr: *anyopaque) *Slots {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn init(gpa: Allocator, r: *sage.Runner) !Slots {
        const v = r.z.s.vocab;
        const half = try gpa.alloc(f16, v);
        errdefer gpa.free(half);
        const row = try gpa.alloc(f32, v);
        errdefer gpa.free(row);
        return .{ .gpa = gpa, .r = r, .half = half, .row = row, .ids = try gpa.alloc(i32, sage.CHUNK) };
    }

    pub fn deinit(b: *Slots) void {
        b.gpa.free(b.half);
        b.gpa.free(b.row);
        b.gpa.free(b.ids);
        for (b.first_rows) |x| if (x) |row| b.gpa.free(row);
        if (b.masked_vals) |x| b.gpa.free(x);
        if (b.masked_ids) |x| b.gpa.free(x);
    }

    pub fn backend(b: *Slots) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .held = heldFn, .release = releaseFn } };
    }

    /// With an MTP head: greedy windows of the pending row and `depth` held drafts, all in one specMulti forward.
    /// Without: one-row rounds, no drafts; the running streams' rows share one forward (stepMulti). The lane core shares a
    /// forward only for a width of 2 or more, so the width is 2 with copy drafts off: every window is one row.
    /// stepMulti's grouped expert kernels add in another order than step, so a near-tie can pick another token.
    pub fn facts(b: *const Slots) lanes.Model {
        if (b.r.mtp != null) return .{ .exact_width = b.depth + 1, .copy_drafts = false, .gpu_tokens = false, .mtp = true, .speculate = true, .speculate_early = false, .drafts = b.depth, .draft_streams = true, .hidden_rows = true, .streams_exact = true, .batch_rows = sage.WINDOW, .max_streams = b.r.slots };
        return .{ .exact_width = 2, .copy_drafts = false, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .hidden_rows = true, .streams_exact = true, .batch_rows = b.r.slots, .max_streams = b.r.slots };
    }

    fn push(b: *Slots, token: u32) u64 {
        b.drawn[b.next % ring] = token;
        b.next += 1;
        return b.next - 1;
    }

    fn slotOf(b: *const Slots, s: *const lanes.Stream) ?u32 {
        for (b.owners[0..b.r.slots], 0..) |o, i| if (o == s) return @intCast(i);
        return null;
    }

    fn freeSlot(b: *const Slots) ?u32 {
        for (b.owners[0..b.r.slots], 0..) |o, i| if (o == null) return @intCast(i);
        return null;
    }

    /// The stream's next token: the runner's greedy one, else drawn on the host from logits row `row` (null: the
    /// current slot's last forward) at `position`.
    fn draw(b: *Slots, s: *const lanes.Stream, greedy: i32, row: ?u32, position: u64) !u32 {
        const sp = s.sampling orelse return @intCast(greedy);
        if (sp.temperature <= 0) return @intCast(greedy);
        return lane.draw(b.gpa, try b.fetchRow(row), sp, position);
    }

    /// Row `row` of the last forward (null: the current slot's last row) as f32 logits in b.row.
    fn fetchRow(b: *Slots, row: ?u32) ![]f32 {
        if (row) |i| try b.r.rowLogits(i, b.half) else try b.r.logits(b.half);
        for (b.row, b.half) |*w, h| w.* = h;
        return b.row;
    }

    /// Structured output: the draw among the tokens `mask` allows, greedy (first argmax) or sampled (keyed).
    fn drawMasked(b: *Slots, s: *const lanes.Stream, row: []const f32, position: u64, mask: []const u32) !u32 {
        if (s.sampling) |sp| if (sp.temperature > 0) {
            const vals = b.masked_vals orelse try b.gpa.alloc(f64, row.len);
            b.masked_vals = vals;
            const ids = b.masked_ids orelse try b.gpa.alloc(u64, row.len);
            b.masked_ids = ids;
            const k: usize = if (sp.top_k == 0) row.len else sp.top_k;
            var n: usize = 0;
            for (row, 0..) |v32, i| {
                if (!lanes.grammar.allows(mask, i)) continue;
                const v: f64 = v32;
                if (!std.math.isFinite(v)) return error.NonfiniteLogits;
                if (n == k and !(v > vals[k - 1])) continue; // equal values keep the lower id
                var j = if (n < k) n else k - 1;
                if (n < k) n += 1;
                if (k < row.len) while (j > 0 and v > vals[j - 1]) : (j -= 1) {
                    vals[j] = vals[j - 1];
                    ids[j] = ids[j - 1];
                };
                vals[j] = v;
                ids[j] = i;
            }
            if (n == 0) return error.NoAllowedToken;
            return @intCast(try lanes.sampling.choose(b.gpa, vals[0..n], ids[0..n], position, sp));
        };
        var best: ?usize = null;
        for (row, 0..) |v, id| if (lanes.grammar.allows(mask, id) and (best == null or v > row[best.?])) {
            best = id;
        };
        return @intCast(best orelse return error.NoAllowedToken);
    }

    /// The prompt into a free slot in chunks (a cancel lands between them); its first token drawn at once.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = of(ptr);
        const r = b.r;
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + 1 > r.ctx) return error.PromptTooLong;
        const slot = b.slotOf(s) orelse (b.freeSlot() orelse return error.NoFreeSlot);
        b.owners[slot] = s;
        errdefer b.owners[slot] = null;
        try r.use(slot);
        try r.reset();
        const v: ?*Vision = if (s.images.len == 0) null else b.vision orelse return error.NoVision;
        const layout = if (v) |x| try x.encode(r, s) else vl.rope.Layout{};
        var at: usize = 0;
        while (at < ids.len) {
            if (s.isCancelled()) return error.Cancelled;
            const n = @min(sage.CHUNK, ids.len - at);
            for (b.ids[0..n], ids[at..][0..n]) |*d, t| d.* = @intCast(t);
            if (v) |x| try r.promptFeed(b.ids[0..n], x.feed(r, &layout, @intCast(at), @intCast(n))) else try r.prompt(b.ids[0..n]);
            at += n;
        }
        r.shift[slot] = layout.shift;
        if (s.grammar != null) {
            const keep = b.first_rows[slot] orelse try b.gpa.alloc(f32, b.row.len);
            b.first_rows[slot] = keep;
            @memcpy(keep, try b.fetchRow(null));
        }
        _ = b.push(try b.draw(s, r.last, null, ids.len));
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const b = of(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        if (position != s.prompt_len) return error.PositionMismatch;
        return b.push(try b.drawMasked(s, b.first_rows[slot] orelse return error.NoLogits, position, mask));
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        return b.next - 1;
    }

    fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const b = of(ptr);
        if (handle >= b.next or b.next - handle > ring) return error.NoSuchToken;
        return b.drawn[handle % ring];
    }

    /// Every stream's pending token in one forward: the single-row step for one stream, else stepMulti. A window of
    /// two rows (a grammar's forced token after the pending one) runs alone, row by row: its second row is fed only
    /// when the first row's draw equals it, so the slot holds exactly the rows a keep can take.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = of(ptr);
        const r = b.r;
        if (windows.len == 0 or windows.len > r.slots or windows.len != out.len) return error.BadWindows;
        var slots: [sage.MAX_SLOTS]u32 = undefined;
        var toks: [sage.MAX_SLOTS]i32 = undefined;
        var at: [sage.MAX_SLOTS]usize = undefined; // the batched windows' indices in `windows`
        var n: usize = 0;
        try b.verifyDrafted(windows, out);
        for (windows, 0..) |w, i| {
            if (w.held > 0) continue;
            if (w.rows() > 2 or w.parents != null) return error.WindowTooWide;
            const slot = b.slotOf(w.stream) orelse return error.UnknownStream;
            const pos = if (slot == r.cur) r.pos else r.spos[slot];
            if (w.positions[0] != pos + 1) return error.PositionMismatch;
            b.fed[slot] = 1;
            if (w.rows() == 2) continue;
            slots[n] = slot;
            toks[n] = @intCast(w.pending);
            at[n] = i;
            n += 1;
        }
        if (n == 1) {
            try r.use(slots[0]);
            try r.step(toks[0]);
            toks[0] = r.last;
        } else if (n > 1) try r.stepMulti(slots[0..n], toks[0..n], toks[0..n]);
        for (at[0..n], 0..) |wi, i| {
            const row: ?u32 = if (n == 1) null else @intCast(i);
            out[wi].sampled[0] = try b.drawRow(windows[wi], 0, toks[i], row);
        }
        for (windows, out) |w, o| {
            if (w.rows() != 2 or w.held > 0) continue;
            const slot = b.slotOf(w.stream).?;
            try r.use(slot);
            try r.step(@intCast(w.pending));
            o.sampled[0] = try b.drawRow(w, 0, r.last, null);
            o.sampled[1] = o.sampled[0]; // not read unless the forced token lands
            if (o.sampled[0] == w.tokens[0]) {
                try r.step(@intCast(w.tokens[0]));
                o.sampled[1] = try b.drawRow(w, 1, r.last, null);
                b.fed[slot] = 2;
            }
            @memcpy(o.drafts, w.tokens);
        }
    }

    /// The windows with held drafts, grouped by depth, each group one specMulti: it drafts, verifies and commits.
    /// A window's rows report the kept tokens, and its drafts are the kept ones then one that differs, so the lane
    /// core keeps exactly the rows specMulti committed.
    fn verifyDrafted(b: *Slots, windows: []const be.Window, out: []be.Verified) !void {
        const r = b.r;
        var depth: u32 = 1;
        while (depth < sage.WINDOW) : (depth += 1) {
            var slots: [sage.MAX_SLOTS]u32 = undefined;
            var at: [sage.MAX_SLOTS]usize = undefined;
            var n: usize = 0;
            for (windows, 0..) |w, i| if (w.held == depth) {
                if (w.parents != null or w.masks.len > 0 or w.tokens.len > 0) return error.WindowTooWide;
                if (w.stream.sampling) |sp| if (sp.temperature > 0) return error.SampledDrafts;
                const slot = b.slotOf(w.stream) orelse return error.UnknownStream;
                if (b.held[slot] < depth) return error.NoHeldDrafts;
                if (w.positions[0] != r.spos[slot] + 1 and !(slot == r.cur and w.positions[0] == r.pos + 1)) return error.PositionMismatch;
                slots[n] = slot;
                at[n] = i;
                n += 1;
            };
            if (n == 0) continue;
            var toks: [sage.MAX_SLOTS][sage.WINDOW]i32 = undefined;
            var kept: [sage.MAX_SLOTS]usize = undefined;
            try r.specMulti(slots[0..n], depth, toks[0..n], kept[0..n]);
            for (slots[0..n], at[0..n], toks[0..n], kept[0..n]) |slot, wi, t, k| {
                const o = out[wi];
                b.held[slot] = 0;
                b.fed[slot] = @intCast(k);
                for (o.sampled, 0..) |*x, j| x.* = @intCast(t[@min(j, k - 1)]);
                for (o.drafts, 0..) |*x, j| x.* = if (j + 1 < k) @intCast(t[j]) else @intCast(t[k - 1] ^ 1);
            }
        }
    }

    /// Row `j` of window `w` drawn: masked when the window has grammar masks, else greedy or sampled.
    fn drawRow(b: *Slots, w: be.Window, j: usize, greedy: i32, row: ?u32) !u32 {
        if (w.masks.len == 0) return b.draw(w.stream, greedy, row, w.positions[j]);
        const words = w.masks.len / w.rows();
        return b.drawMasked(w.stream, try b.fetchRow(row), w.positions[j], w.masks[j * words ..][0..words]);
    }

    /// A window keeps at most the rows its verify fed (one, or two when the forced token landed). A shorter path
    /// ends the stream; a next verify of it would fail its position check.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = of(ptr);
        for (windows, paths) |w, p| {
            const slot = b.slotOf(w.stream) orelse return error.UnknownStream;
            if (p.len == 0 or p.len > b.fed[slot]) return error.KeepMismatch;
        }
    }

    /// The MTP absorbed the kept rows in specMulti (or the prompt in prefill): note the drafts each stream's next
    /// window holds. specMulti makes them with its forward.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = of(ptr);
        if (b.r.mtp == null) return error.NoDraftHead;
        for (requests) |q| {
            const slot = b.slotOf(q.stream) orelse return error.UnknownStream;
            b.held[slot] = @min(q.depth, b.depth);
        }
    }

    /// A lone greedy stream on MTP drafts (Runner.specStep, `depth` a round) until it ends (false) or another stream
    /// arrives (true). The lane core then decodes it beside the others, one row a round, with no drafts.
    fn loneFn(ptr: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const b = of(ptr);
        const r = b.r;
        try prefillFn(ptr, s);
        const slot = b.slotOf(s).?;
        var handed = false;
        defer if (!handed) {
            b.owners[slot] = null;
        };
        var pending = b.drawn[(b.next - 1) % ring];
        _ = try s.commit(b.gpa, &.{pending}, &.{});
        hooks.committed(hooks.ctx);
        var out: [sage.WINDOW]i32 = undefined;
        var landed: [sage.WINDOW]u32 = undefined;
        while (!s.finished) {
            if (hooks.yield(hooks.ctx)) {
                s.cache_len = r.pos;
                s.pending = pending;
                handed = true;
                return true;
            }
            if (s.isCancelled()) return error.Cancelled;
            const n = try r.specStep(b.depth, &out);
            for (landed[0..n], out[0..n]) |*d, t| d.* = @intCast(t);
            s.rounds += 1;
            s.drafted += b.depth;
            s.accepted += n - 1;
            _ = try s.commit(b.gpa, landed[0..n], &.{});
            pending = landed[n - 1];
            hooks.committed(hooks.ctx);
        }
        return false;
    }

    /// A grammar stream's held draft as a host token, for the grammar to walk: the slot's MTP draft. Its window then
    /// runs on the two-row path of verifyFn, which feeds the draft only when the masked draw equals it.
    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const b = of(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        if (out.len != 1) return error.GrammarDraftsDeep; // the two-row path takes one draft
        out[0] = @intCast(@max(b.r.sdraft[slot], 0)); // none yet: any token, the draw decides
        b.held[slot] = 0;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = of(ptr);
        const slot = b.slotOf(s) orelse return;
        b.owners[slot] = null;
        b.held[slot] = 0;
    }
};

/// --vision: the checkpoint's image tower, its processor, and the rows one prompt's images take.
const Vision = struct {
    io: std.Io,
    c: vl.Config,
    tower: *xq.vision.Tower,
    rows: *anyopaque, // fp32 [image_tokens, hidden]: a prompt's image rows, image after image
    image_tokens: u32,
    per_image: u32, // tokens one image takes at most: the tower's scratch
    markers: [3]u32, // vision_start, image_pad, vision_end: text that spells one stays text
    spans: [vl.rope.max_images]struct { at: u32, n: u32, row: u32 } = undefined,
    count: u32 = 0,
    rope: [sage.CHUNK][3]i32 = undefined,
    pieces: [vl.rope.max_images]sage.Rows = undefined,

    /// The tower of the checkpoint in `dir` on the runner's card; null with `why` set when it has none it reads.
    fn open(gpa: Allocator, io: std.Io, a: Allocator, r: *sage.Runner, dir: []const u8, o: api.VisionOpen, why: *[]const u8) !?*Vision {
        const read = struct {
            fn f(al: Allocator, i: std.Io, d: []const u8, name: []const u8) ?[]const u8 {
                const path = std.fs.path.join(al, &.{ d, name }) catch return null;
                return std.Io.Dir.cwd().readFileAlloc(i, path, al, .limited(16 << 20)) catch null;
            }
        }.f;
        const config = read(a, io, dir, "config.json") orelse return error.NoConfig;
        const c = (vl.Config.parse(a, config, read(a, io, dir, "preprocessor_config.json"), read(a, io, dir, "processor_config.json")) catch null) orelse {
            why.* = "--vision: this checkpoint names no image tower the XPU engine reads (config.json vision_config)";
            return null;
        };
        if (c.out_hidden != r.z.s.hidden or c.patch != xq.vision.patch or c.temporal != xq.vision.temporal or c.merge != xq.vision.merge or c.positions != xq.vision.grid_side * xq.vision.grid_side or !textRope(a, config)) {
            why.* = "--vision: the image tower or the text's M-RoPE (interleaved 11, 11, 10) is not the Qwen3.5 one the XPU engine runs";
            return null;
        }
        const per_image: u32 = @intCast(@min(@as(u64, o.image_tokens), c.max_pixels / (@as(u64, c.factor()) * c.factor())));
        const prefix = if (r.ckpt.has("vision_tower.patch_embed.proj.weight")) "vision_tower." else "model.visual.";
        const tower = xq.vision.Tower.load(&r.d, &r.ckpt, prefix, .{ .hidden = c.hidden, .heads = c.heads, .depth = c.depth, .intermediate = c.intermediate, .out_hidden = c.out_hidden }, per_image * c.merge * c.merge) catch |err| {
            why.* = try std.fmt.allocPrint(a, "--vision: the XPU engine reads the tower in bf16 and could not load this one: {s}", .{@errorName(err)});
            return null;
        };
        errdefer tower.deinit();
        const v = try gpa.create(Vision);
        v.* = .{ .io = io, .c = c, .tower = tower, .rows = try r.d.alloc_(@as(usize, o.image_tokens) * c.out_hidden * 4), .image_tokens = o.image_tokens, .per_image = per_image, .markers = .{ c.vision_start, c.image_token, c.vision_end } };
        return v;
    }

    fn deinit(v: *Vision, gpa: Allocator) void {
        v.tower.deinit();
        gpa.destroy(v);
    }

    /// Info.vision's prepare: an image's bytes to the tower's patches (core/qwen_vl geometry), on the HTTP thread.
    fn prepare(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
        const v: *Vision = @ptrCast(@alignCast(ctx));
        const p = try vl.geometry.prepare(v.io, a, bytes, v.c, @min(max_tokens, v.per_image));
        return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
    }

    fn offer(v: *Vision, o: api.VisionOpen) api.Vision {
        // the template writes Qwen's vision_start and vision_end around the image token itself
        return .{ .ctx = v, .prepare = prepare, .image_token = v.c.image_token, .markers = &v.markers, .image_tokens = v.image_tokens, .max_images = @min(o.max_images, vl.rope.max_images) };
    }

    /// `s`'s images through the tower into `rows`, and its prompt's M-RoPE layout.
    fn encode(v: *Vision, r: *sage.Runner, s: *const lanes.Stream) !vl.rope.Layout {
        const layout = try vl.rope.Layout.init(s.prompt(), s.images, .{ .start = v.c.vision_start, .image = v.c.image_token, .end = v.c.vision_end }, v.c.merge);
        const D: usize = v.c.out_hidden;
        var row: u32 = 0;
        v.count = 0;
        for (s.images) |img| {
            if (row + img.tokens > v.image_tokens or img.tokens > v.per_image) return error.ImageTooLarge;
            try v.tower.encode(&r.d, img.pixels, img.gh, img.gw, off(v.rows, row * D * 4));
            v.spans[v.count] = .{ .at = img.at, .n = img.tokens, .row = row };
            v.count += 1;
            row += img.tokens;
        }
        return layout;
    }

    /// Prompt rows [at, at + n): their positions and the image rows among them.
    fn feed(v: *Vision, r: *const sage.Runner, layout: *const vl.rope.Layout, at: u32, n: u32) sage.Feed {
        const D: usize = r.z.s.hidden;
        for (v.rope[0..n], 0..) |*p, i| p.* = layout.position(at + @as(u32, @intCast(i)));
        var k: usize = 0;
        for (v.spans[0..v.count]) |sp| {
            const lo = @max(sp.at, at);
            const hi = @min(sp.at + sp.n, at + n);
            if (lo >= hi) continue;
            v.pieces[k] = .{ .at = lo - at, .n = hi - lo, .src = off(v.rows, (sp.row + lo - sp.at) * D * 4) };
            k += 1;
        }
        return .{ .rope = v.rope[0..n], .rows = v.pieces[0..k] };
    }
};

/// The text's rotary is Qwen3.5's interleaved M-RoPE over sections 11, 11 and 10 (attn_prep's ROPE).
fn textRope(a: Allocator, config_json: []const u8) bool {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, config_json, .{}) catch return false;
    if (doc != .object) return false;
    const tc = doc.object.get("text_config");
    const text = if (tc != null and tc.? == .object) tc.?.object else doc.object;
    const rope = text.get("rope_parameters") orelse return false;
    if (rope != .object) return false;
    const inter = rope.object.get("mrope_interleaved") orelse return false;
    const sec = rope.object.get("mrope_section") orelse return false;
    if (inter != .bool or !inter.bool or sec != .array or sec.array.items.len != 3) return false;
    for (sec.array.items, [_]i64{ 11, 11, 10 }) |x, want| if (x != .integer or x.integer != want) return false;
    return true;
}

const Served = struct {
    gpa: Allocator,
    back: Slots,
    cfg: lanes.Config,
    wall: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    startup: []u8 = &.{},

    fn close(ptr: *anyopaque) void {
        const h: *Served = @ptrCast(@alignCast(ptr));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        const r = h.back.r;
        if (h.back.vision) |v| v.deinit(h.gpa);
        h.back.deinit();
        r.deinit(); // the card's memory, after the engine has stopped
        h.gpa.free(h.startup);
        h.gpa.destroy(h);
    }
};

/// Sage from `o.dir` (MLX affine 4/64, or AutoRound GPTQ 4/128 shown as MLX at load): --parallel slots (1 to 8) of --context tokens each, int8 K/V with XPU_KV_INT8.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const context: u32 = std.math.cast(u32, o.context orelse 32768) orelse 0;
    if (context == 0) {
        problem.* = try std.fmt.allocPrint(a, "the XPU Sage engine needs a context of 1 to 2^32-1 tokens, not {d}", .{o.context.?});
        return null;
    }
    const slots: u32 = @min(@max(o.lanes, 1), sage.MAX_SLOTS);
    // --drafter: the AutoRound checkpoint whose MTP head drafts for a lone greedy stream
    const mtp = if (o.drafts) o.drafter else null;
    const r = sage.Runner.initMtp(o.dir, context, slots, mtp) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Sage engine could not load {s} with {d} streams of {d} tokens: {s}", .{ o.dir, slots, context, @errorName(err) });
        return null;
    };
    errdefer r.deinit();
    const h = try gpa.create(Served);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .back = try Slots.init(gpa, r), .cfg = undefined, .wall = .{ .io = io }, .core = undefined, .host = undefined };
    errdefer h.back.deinit();
    if (o.vision) |vo| h.back.vision = try Vision.open(gpa, io, a, r, o.dir, vo, problem) orelse return null;
    errdefer if (h.back.vision) |v| v.deinit(gpa);
    if (std.c.getenv("XPU_SPEC")) |v| h.back.depth = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
    if (h.back.depth == 0 or h.back.depth + 1 > sage.WINDOW) {
        problem.* = try std.fmt.allocPrint(a, "XPU_SPEC: 1 to {d} MTP drafts a round, not {d}", .{ sage.WINDOW - 1, h.back.depth });
        return null;
    }
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), if (r.mtp != null) h.back.depth + 1 else 2, if (r.mtp != null) h.back.depth else 0);
    errdefer h.cfg.deinit(gpa);
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.wall.clock());
    errdefer h.core.deinit();
    h.startup = try std.fmt.allocPrint(gpa, "Intel GPU engine: Sage (qwen3_5_moe), {d} stream(s) of {d} tokens, {s} K/V, sampling on the host{s}", .{ slots, context, if (r.cap > 0) "int8" else "bf16", if (r.mtp != null) ", MTP drafts for a lone greedy stream" else "" });
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "sage-xpu", .lanes = slots, .context_window = context, .context_fitted = true, .startup = h.startup, .prefill_step = 0, .structures = true });
    h.host.explain = .{ .text = lane.explain };
    if (h.back.vision) |v| h.host.info_.vision = v.offer(o.vision.?);
    if (r.mtp != null) h.host.lone = .{ .ctx = &h.back, .run = Slots.loneFn };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Served.close, .ctx = h };
}

fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}
