//! Qwen3.8-27B on an Intel GPU as one engine value: the checkpoint (MLX, EXL3 or GGUF) on the device, one sequence.

const std = @import("std");
const xpu = @import("xpu");
const cfgm = @import("xpu_config.zig");
const model = @import("xpu_model.zig");
const qg = @import("xpu_gguf.zig");
const qw = @import("xpu_win.zig");
const al = @import("xpu_attn_long.zig");
const spec = @import("xpu_spec.zig");
const mtp = @import("xpu_mtp.zig");

const ld = xpu.loader;

pub const Options = struct {
    /// Positions of the KV caches (prompt and generated tokens).
    context: usize = 4096,
    progress: bool = false,
    /// KV cache format (`--kv`); q8 and q4 need no more than the long-context kernels.
    kv: al.Mode = .q8,
    /// Rows of the prompt windows when more than 16 (`--prefill`).
    window_rows: u32 = 0,
};

/// window_rows value of --prefill auto: the window is chosen from the memory left (xpu_win.autoRows).
pub const auto_window: u32 = std.math.maxInt(u32);

pub const DrafterKind = enum { copy, mtp, auto };

/// The windowed generation modes: prompt windows (`--prefill`), speculative decoding (`--spec`).
pub const Fast = struct {
    prefill_rows: u32 = 0,
    spec_k: usize = 0,
    drafter: DrafterKind = .copy,
    /// A separate checkpoint holding the MTP head (EXL3 directory), else the target's own.
    mtp_from: ?[]const u8 = null,
    min_n: usize = 3,
    stop_eos: bool = true,
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    ctx: *const xpu.Context,
    stream: xpu.Stream,
    r: xpu.rt.Runtime,
    loader: ld.Loader,
    cfg: cfgm.Config,
    m: model.Model,
    zeros: []u8,
    max_len: usize,
    load_seconds: f64,
    stats: spec.Stats = .{},

    /// `path`: a checkpoint directory (MLX 4-bit or EXL3) or a .gguf file.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const xpu.Context, path: []const u8, o: Options) !*Engine {
        const t0 = std.Io.Clock.awake.now(io);
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.ctx = ctx;
        e.stream = try xpu.Stream.init(ctx);
        errdefer e.stream.deinit();
        e.r = try xpu.rt.Runtime.init(ctx, e.stream);
        errdefer e.r.deinit(); // drains the queue and forgets it, so the exit handler never calls into a closed driver
        const is_gguf = std.mem.endsWith(u8, path, ".gguf");
        e.loader = if (is_gguf) try ld.Loader.initBare(gpa, &e.r) else try ld.Loader.init(gpa, &e.r, path);
        if (is_gguf) {
            const meta = try qg.readHeader(gpa, &e.loader, path);
            e.cfg = try qg.config(gpa, &e.loader, meta);
        } else {
            const text = try ld.readFile(gpa, path, "config.json");
            e.cfg = (try cfgm.parse(gpa, text)).value;
        }
        al.setKvMode(o.kv);
        if (o.window_rows == auto_window) qw.default_rows = qw.auto_rows else if (o.window_rows > 16) qw.default_rows = o.window_rows;
        e.m = try model.Model.load(gpa, &e.r, &e.loader, e.cfg, @intCast(o.context));
        e.m.bf16_logits = true;
        e.zeros = try gpa.alloc(u8, model.state_bytes);
        @memset(e.zeros, 0);
        e.max_len = o.context;
        e.load_seconds = xpu.decode.seconds(io, t0);
        return e;
    }

    /// The process exits right after: device memory and host staging go with it.
    pub fn deinit(e: *Engine) void {
        e.r.deinit();
        e.stream.deinit();
        e.gpa.destroy(e);
    }

    pub fn deviceBytes(e: *const Engine) u64 {
        return e.loader.total;
    }

    pub fn vocab(e: *const Engine) usize {
        return e.cfg.text_config.vocab_size;
    }

    pub fn isEos(e: *const Engine, token: u32) bool {
        return e.cfg.isEos(token);
    }

    pub fn reset(e: *Engine) !void {
        try e.m.reset(e.zeros);
    }

    pub fn feed(e: *Engine, token: u32, logits: bool) !void {
        try e.m.forward(token, logits);
    }

    pub fn sync(e: *Engine) !void {
        try e.r.sync();
    }

    pub fn argmax(e: *Engine) !u32 {
        return e.m.argmax();
    }

    pub fn fetchLogits(e: *Engine, out: []f32) !void {
        try e.m.fetchLogits(out);
    }

    /// Runs `head` as prompt windows of up to `rows` rows (no logits); the queue is drained when it returns.
    pub fn prefillWindows(e: *Engine, head: []const u32, rows: u32) !void {
        var t: usize = 0;
        while (t < head.len) {
            const n = @min(rows, head.len - t);
            try e.m.forwardRows(head[t .. t + n], 0, true);
            t += n;
        }
        try e.r.sync();
    }

    /// Prompt windows and speculative decoding: `count` tokens identical to the plain greedy run; needs a prompt.
    pub fn generateFast(e: *Engine, io: std.Io, prompt: []const u32, count: usize, o: Fast) !xpu.decode.Result {
        if (prompt.len == 0) return error.NoPromptTokens;
        if (count == 0) return error.NothingToGenerate;
        if (prompt.len + count + 20 > e.max_len) return error.ContextTooSmall;
        const gpa = e.gpa;
        const m = &e.m;
        const rows: u32 = if (o.prefill_rows > 0) o.prefill_rows else 16;
        const uses_mtp = o.spec_k > 0 and o.drafter != .copy;
        try e.reset();
        e.stats = .{};
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(gpa);
        var next: u32 = 0;
        const t0 = std.Io.Clock.awake.now(io);
        var cd: spec.CopyDrafter = .{ .min_n = o.min_n };
        var ml: xpu.loader.Loader = undefined;
        var mt: mtp.Mtp = undefined;
        var md: spec.MtpDrafter = undefined;
        var ad: spec.AutoDrafter = undefined;
        var drafter: spec.Drafter = cd.drafter();
        if (uses_mtp) {
            ml = if (o.mtp_from) |d| try xpu.loader.Loader.init(gpa, &e.r, d) else e.loader;
            mt = try mtp.Mtp.load(gpa, m, &ml, @intCast(e.max_len));
            var mst: mtp.Stats = .{};
            var first: [16]u32 = undefined;
            const pf = try mtp.prefill(m, &mt, prompt, o.spec_k, &first, &mst, rows);
            next = pf.first;
            md = spec.MtpDrafter.init(&mt, first[0..pf.nd]);
            md.st = mst;
            ad = .{ .mtp_d = &md };
            drafter = if (o.drafter == .mtp) md.drafter() else ad.drafter();
        } else {
            var t: usize = 0;
            while (t < prompt.len) {
                const n = @min(rows, prompt.len - t);
                try m.forwardRows(prompt[t .. t + n], if (t + n == prompt.len) 1 else 0, true);
                t += n;
            }
            var g: [1]i32 = undefined;
            try m.rowArgmax(&g);
            next = @intCast(g[0]);
        }
        try out.append(gpa, next);
        const prefill_s = xpu.decode.seconds(io, t0);
        const t1 = std.Io.Clock.awake.now(io);
        const eos = e.cfg.eos_token_id;
        if (out.items.len < count and !(o.stop_eos and e.isEos(next))) {
            if (o.spec_k > 0) {
                var sctx: std.ArrayList(u32) = .empty;
                defer sctx.deinit(gpa);
                try sctx.appendSlice(gpa, prompt);
                try sctx.append(gpa, next);
                try spec.run(m, gpa, &sctx, count - 1, o.spec_k, drafter, eos, !o.stop_eos, &e.stats);
                try out.appendSlice(gpa, sctx.items[prompt.len + 1 ..]);
            } else while (out.items.len < count) {
                try m.forward(next, true);
                next = try m.argmax();
                try out.append(gpa, next);
                if (o.stop_eos and e.isEos(next)) break;
            }
        }
        const decode_s = xpu.decode.seconds(io, t1);
        const rounds = e.stats.windows + e.stats.plain;
        return .{ .tokens = try out.toOwnedSlice(gpa), .top5 = &.{}, .prefill_seconds = prefill_s, .decode_seconds = decode_s, .rounds = rounds };
    }
};
