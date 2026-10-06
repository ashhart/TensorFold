//! GLM-5.3-Flash on Metal: one greedy reply at a time. The prompt runs in windows of up to 16 rows on the decode
//! kernels (every row its one-row bits), then rounds verify a window of the last token and the MTP head's drafts.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const mtp = @import("mtp.zig");
const prompt_mod = @import("prompt.zig");
const kernels = @import("kernels.zig");
const Ref = wts.Ref;

pub const Reason = enum { stop, length, cancelled };

pub const Result = struct {
    reason: Reason,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    min_rows: u32 = 0,
    prompt_seconds: f64 = 0,
    decode_seconds: f64 = 0,
    generated: u64 = 0,
};

/// What a reply reports while it runs (called on the engine's thread).
pub const Out = struct {
    ctx: *anyopaque,
    prefilled: *const fn (ctx: *anyopaque) void,
    /// Tokens committed, in order; true ends the reply as a stop (a stop string matched).
    tokens: *const fn (ctx: *anyopaque, toks: []const u32) bool,
    cancelled: *const fn (ctx: *anyopaque) bool,
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    event: mtl.SharedEvent,
    ev: u64 = 0,
    c: cfg.Config,
    k: *kernels.Kernels,
    w: *wts.Weights,
    arena: st.Arena,
    s: st.State,
    sc: st.Scratch,
    prompt_ids: Ref,
    pr: prompt_mod.Prompt,
    chunked: bool, // prompt chunks on the tensor units (GLM_PROMPT=0: every prompt row in 16-row decode windows)
    residency: ?mtl.ResidencySet = null,
    load_seconds: f64 = 0,

    /// The checkpoint in `dir` with caches for `cap` tokens; GLM_LAYERS=N: only the first N layers, the MTP layer and the head.
    pub fn load(gpa: std.mem.Allocator, dir: []const u8, cap: u32) !*Engine {
        const e = try gpa.create(Engine); // undefined memory: every field is set below
        errdefer gpa.destroy(e);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const t0 = std.c.mach_absolute_time();
        e.gpa = gpa;
        e.ev = 0;
        e.residency = null;
        e.device = try mtl.Device.init();
        e.queue = try e.device.queue();
        e.event = try e.device.sharedEvent();
        const path = try std.fmt.allocPrintSentinel(gpa, "{s}/config.json", .{dir}, 0);
        defer gpa.free(path);
        const f = try mtl.MappedFile.open(path);
        defer f.deinit();
        e.c = try cfg.parse(gpa, f.bytes[0..f.size]);
        if (std.c.getenv("GLM_LAYERS")) |v| try cfg.subset(&e.c, std.fmt.parseInt(u32, std.mem.span(v), 10) catch return error.BadLayerCount);
        e.k = try kernels.load(gpa, e.device);
        errdefer {
            e.k.deinit();
            gpa.destroy(e.k);
        }
        if (std.c.getenv("GLM_DRY") != null) { // the weights' plan and checks only, then stop
            const plan = try wts.load(gpa, e.device, dir, &e.c, 16, true);
            plan.deinit();
            gpa.destroy(plan);
            return error.DryRun;
        }
        e.w = try wts.load(gpa, e.device, dir, &e.c, 16, false);
        errdefer {
            e.w.deinit();
            gpa.destroy(e.w);
        }
        e.arena = .{ .device = e.device, .gpa = gpa };
        errdefer e.arena.deinit();
        const both = try st.init(&e.arena, &e.c, cap);
        e.s = both.state;
        e.sc = both.scratch;
        e.prompt_ids = try e.arena.buffer(@as(usize, cap) * 4);
        e.pr = try prompt_mod.init(gpa, &e.arena, e.device, &e.c, &e.sc, cap);
        errdefer e.pr.deinit();
        e.chunked = if (std.c.getenv("GLM_PROMPT")) |v| v[0] != '0' else true;
        try e.prepare();
        // opt-in: wiring 181 GB leaves macOS nothing to reclaim if another model shares the Mac (Flash Next runs without)
        if (std.c.getenv("GLM_RESIDENCY") == null) {} else if (e.device.residencySet(e.w.buffers.items.len + e.arena.buffers.items.len)) |set| {
            for (e.w.buffers.items) |b| set.add(b);
            for (e.arena.buffers.items) |b| set.add(b);
            set.commit();
            set.requestResidency();
            e.queue.addResidencySet(set);
            e.residency = set;
        } else |_| {}
        e.load_seconds = @as(f64, @floatFromInt(std.c.mach_absolute_time() - t0)) / 24e6;
        return e;
    }

    /// The KDA decay rates A = exp(A_log) with MLX's Exp, on the GPU.
    fn prepare(e: *Engine) !void {
        const cb = e.queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (0..e.c.run) |li| switch (e.w.layers[li].attn) {
            .kda => |*a| {
                enc.setPipeline(e.k.exp_f32);
                enc.setBuffer(a.a_log.buf, a.a_log.off, 0);
                enc.setBuffer(a.a.buf, a.a.off, 1);
                enc.setValue(e.c.kda_heads, 2);
                enc.dispatchThreads(mtl.Size.of(e.c.kda_heads, 1, 1), mtl.Size.of(64, 1, 1));
            },
            .mla => {},
        };
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("glm: prepare failed: {s}", .{msg});
            return error.GpuFailed;
        }
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpa;
        if (e.residency) |set| {
            e.queue.removeResidencySet(set);
            set.deinit();
        }
        e.pr.deinit();
        e.arena.deinit();
        e.w.deinit();
        gpa.destroy(e.w);
        e.k.deinit();
        gpa.destroy(e.k);
        e.event.deinit();
        e.queue.deinit();
        e.device.deinit();
        gpa.destroy(e);
    }

    pub fn hasMtp(e: *const Engine) bool {
        return e.w.mtp != null;
    }

    fn ctx(e: *Engine) fwd.Ctx {
        return .{ .k = e.k, .c = &e.c, .w = e.w, .s = &e.s, .sc = &e.sc };
    }

    /// A command buffer ordered after the last one this engine committed.
    fn begin(e: *Engine) struct { cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder } {
        const cb = e.queue.commandBuffer();
        if (e.ev > 0) cb.waitFor(e.event, e.ev);
        return .{ .cb = cb, .enc = cb.compute(.serial) };
    }

    /// Commit and wait: no work of this engine is in flight once it returns, whatever the caller does next.
    fn finish(e: *Engine, cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder) !void {
        enc.end();
        e.ev += 1;
        cb.signal(e.event, e.ev);
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            e.event.set(e.ev); // a failed buffer may never signal: the next one must not wait on it
            std.log.err("glm: command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
    }

    /// Wait until every command buffer this engine committed has completed (before the host touches shared state).
    pub fn sync(e: *Engine) void {
        if (e.event.value() >= e.ev) return;
        if (!e.event.wait(e.ev, 120_000)) std.log.err("glm: the GPU did not finish within 120 s", .{});
    }

    /// One bf16 row of `D` values from `src` to `dst`.
    fn copyRow(x: *const fwd.Ctx, enc: mtl.ComputeEncoder, src: Ref, dst: Ref, D: u32) void {
        enc.setPipeline(x.k.copy_u32);
        enc.setBuffer(src.buf, src.off, 0);
        enc.setBuffer(dst.buf, dst.off, 1);
        enc.setValue(D / 2, 2);
        enc.dispatchThreads(mtl.Size.of(D / 2, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn u32s(r: Ref, n: usize) []u32 {
        return @as([*]u32, @ptrCast(@alignCast(r.addr())))[0..n];
    }

    /// The first window of `prompt` from a fresh state with every sublayer's input and output captured, then its
    /// logits: raw bf16 in glm_ref.py's order (embed; per layer attn in/out, mlp in/out; final; logits).
    pub fn capture(e: *Engine, prompt: []const u32, path: []const u8) !void {
        const c = &e.c;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const n: u32 = @intCast(@min(prompt.len, st.max_rows));
        e.sync();
        e.s.reset();
        @memcpy(u32s(e.prompt_ids, n), prompt[0..n]);
        const plane = @as(usize, n) * c.hidden * 2;
        const bytes = plane * (2 + 4 * @as(usize, c.run)) + @as(usize, n) * c.vocab * 2;
        const dump = try e.arena.buffer(bytes);
        var x = e.ctx();
        x.dump = dump;
        const b = e.begin();
        fwd.backbone(&x, b.enc, e.prompt_ids, n, 0);
        fwd.head(&x, b.enc, e.sc.hidden, dump.at(x.dump_at), e.sc.picks, n);
        try e.finish(b.cb, b.enc);
        fwd.flipKda(&x);
        const file = std.c.fopen(try std.fmt.allocPrintSentinel(e.gpa, "{s}", .{path}, 0), "wb") orelse return error.OpenFailed;
        defer _ = std.c.fclose(file);
        if (std.c.fwrite(dump.addr(), 1, bytes, file) != bytes) return error.WriteFailed;
        std.debug.print("captured {d} rows ({d} bytes) in {s}\n", .{ n, bytes, path });
    }

    /// One greedy reply. `depth` drafts a round (0: one token a round, the reference drafted replies must equal).
    pub fn generate(e: *Engine, prompt: []const u32, max_tokens: usize, eos: []const u32, depth: usize, out: Out) !Result {
        const c = &e.c;
        const D = c.hidden;
        const P: u32 = @intCast(prompt.len);
        if (prompt.len == 0) return error.EmptyPrompt;
        const d: u32 = @intCast(if (e.w.mtp == null) 0 else @min(depth, st.max_rows - 1));
        if (prompt.len + max_tokens + d + 1 > e.s.cap) return error.ContextFull;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        e.sync();
        e.s.reset();
        @memcpy(u32s(e.prompt_ids, prompt.len), prompt);
        var x = e.ctx();
        const t_start = std.c.mach_absolute_time();
        // the prompt: windows of up to 16 rows, the MTP head taking each row with the token after it
        var at: u32 = 0;
        var last_n: u32 = 1;
        while (at < P) {
            if (out.cancelled(out.ctx)) return .{ .reason = .cancelled };
            const chunk = e.chunked and P - at > st.max_rows;
            const n = @min(@as(u32, if (chunk) prompt_mod.max_rows else st.max_rows), P - at);
            const last = at + n == P;
            const absorb = if (last) n - 1 else n;
            const b = e.begin();
            if (chunk) {
                var px = x;
                px.sc = &e.pr.streams;
                prompt_mod.backbone(&e.pr, &px, b.enc, e.prompt_ids.at(@as(usize, at) * 4), n, at);
                fwd.flipKda(&x);
                const hidden = e.pr.streams.hidden;
                if (last) { // the last row into the decode scratch: the head's first token, the MTP head's first row
                    copyRow(&x, b.enc, hidden.at(@as(usize, n - 1) * D * 2), e.sc.hidden, D);
                    fwd.head(&x, b.enc, e.sc.hidden, e.sc.logits, e.sc.picks, 1);
                }
                if (d > 0 and absorb > 0) {
                    prompt_mod.mtp(&e.pr, &px, b.enc, hidden, e.prompt_ids.at(@as(usize, at + 1) * 4), absorb, at);
                    e.s.mtp_pos = at + absorb;
                }
                last_n = 1;
            } else {
                fwd.backbone(&x, b.enc, e.prompt_ids.at(@as(usize, at) * 4), n, at);
                fwd.flipKda(&x);
                if (last) fwd.head(&x, b.enc, e.sc.hidden.at(@as(usize, n - 1) * D * 2), e.sc.logits, e.sc.picks, 1);
                if (d > 0 and absorb > 0) {
                    mtp.run(&x, b.enc, e.sc.hidden, e.prompt_ids.at(@as(usize, at + 1) * 4), absorb, at, e.sc.m_picks);
                    e.s.mtp_pos = at + absorb;
                }
                last_n = n;
            }
            try e.finish(b.cb, b.enc);
            at += n;
        }
        e.s.pos = P;
        const t_prompt = std.c.mach_absolute_time();
        out.prefilled(out.ctx);
        var res: Result = .{ .reason = .length, .prompt_seconds = @as(f64, @floatFromInt(t_prompt - t_start)) / 24e6 };
        var tok = u32s(e.sc.picks, 1)[0];
        res.generated = 1;
        if (out.tokens(out.ctx, &.{tok}) or std.mem.indexOfScalar(u32, eos, tok) != null) {
            res.reason = .stop;
            return res;
        }
        if (max_tokens <= 1) return res;
        var emitted: usize = 1;
        var h0: u32 = last_n - 1; // the rows of `hidden` the MTP head takes next, and how many
        var keep: u32 = 1;
        var window: u32 = 0; // the last forward's rows (0: the prompt's)
        res.min_rows = d + 1;
        u32s(e.sc.next, 1)[0] = tok;
        while (true) {
            const b = e.begin();
            if (window > 0) { // the last window's kept rows (the prompt's windows kept all theirs)
                fwd.keepKda(&x, b.enc, window, keep);
                fwd.flipKda(&x);
            }
            const ids = u32s(e.sc.ids, d + 1);
            ids[0] = tok;
            if (d > 0) {
                const next = if (window == 0) e.sc.next else e.sc.picks;
                mtp.run(&x, b.enc, e.sc.hidden.at(@as(usize, h0) * D * 2), next, keep, e.s.mtp_pos, e.sc.ids.at(4));
                const base = e.s.mtp_pos + keep;
                for (1..d) |j| {
                    const h = if (j == 1) e.sc.m_x.at(@as(usize, keep - 1) * D * 2) else e.sc.m_x;
                    mtp.chain(&x, b.enc, h, e.sc.ids.at(j * 4), base + @as(u32, @intCast(j)) - 1, e.sc.ids.at((j + 1) * 4));
                }
                e.s.mtp_pos = base;
            }
            const R = d + 1;
            fwd.backbone(&x, b.enc, e.sc.ids, R, e.s.pos);
            fwd.head(&x, b.enc, e.sc.hidden, e.sc.logits, e.sc.picks, R);
            try e.finish(b.cb, b.enc);
            const picks = u32s(e.sc.picks, R);
            const drafts = u32s(e.sc.ids, R);
            keep = 1;
            while (keep < R and picks[keep - 1] == drafts[keep]) keep += 1;
            res.rounds += 1;
            res.drafted += d;
            res.accepted += keep - 1;
            var take: usize = 0;
            var stop = false;
            while (take < keep and emitted + take < max_tokens) {
                take += 1;
                if (std.mem.indexOfScalar(u32, eos, picks[take - 1]) != null) {
                    stop = true;
                    break;
                }
            }
            if (take > 0 and out.tokens(out.ctx, picks[0..take])) stop = true;
            emitted += take;
            res.generated = emitted;
            e.s.pos += keep;
            tok = picks[keep - 1];
            window = R;
            h0 = 0;
            if (stop) {
                res.reason = .stop;
                break;
            }
            if (emitted >= max_tokens) break;
            if (out.cancelled(out.ctx)) {
                res.reason = .cancelled;
                break;
            }
            if (e.s.pos + R + 1 > e.s.cap) break;
        }
        res.decode_seconds = @as(f64, @floatFromInt(std.c.mach_absolute_time() - t_prompt)) / 24e6;
        return res;
    }
};
