//! `tensorfold run` and `kl` on a Ripple checkpoint: the CUDA forward driven directly, one stream; --eager: no graphs.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rc = @import("ricursive_cuda");
const kl = @import("cuda_kl.zig");

const fwd = rc.model;

/// The checkpoint at `dir` is a Ripple one: its config.json names model_type ricursive.
pub fn wants(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(gpa, &.{ dir, "config.json" }) catch return false;
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22)) catch return false;
    defer gpa.free(text);
    const Head = struct { model_type: []const u8 = "" };
    const head = std.json.parseFromSlice(Head, gpa, text, .{ .ignore_unknown_fields = true }) catch return false;
    defer head.deinit();
    return std.mem.eql(u8, head.value.model_type, rc.native.model_type);
}

pub const Options = struct {
    tokens: []const u32,
    gold: []const u32 = &.{},
    max_tokens: usize,
    context: ?usize,
    stop_eos: bool = true,
    report: ?[]const u8 = null,
    chunk: ?usize = null, // kl: every prompt row's logits, the prompt in chunks of this many rows
    dump: ?[]const u8 = null,
    against: ?[]const u8 = null,
    graphs: bool = true, // decode rounds as CUDA graphs; false: eager launches
};

fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, context: ?usize) !*rc.native.Owned {
    const own = try rc.native.openOwned(gpa, io, ctx, dir, context orelse @intCast(rc.native.default_context), null);
    std.debug.print("Ripple on CUDA: {d}-token context, prompt chunks of {d} rows\n", .{ own.m.context, fwd.prompt_rows });
    return own;
}

/// The prompt from position 0 in chunks of up to 2,048 rows, its last row through the head.
fn prefill(m: *const fwd.Model, st: *fwd.Seq, ids: []const u32) !void {
    st.len = 0;
    var at: usize = 0;
    while (at < ids.len) {
        const end = @min(ids.len, at + fwd.prompt_rows);
        try m.forward(st, ids[at..end], .prompt);
        if (end == ids.len) try m.head(end - at - 1);
        at = end;
    }
}

/// The greedy token of the head's last row.
fn greedy(own: *rc.native.Owned) !u32 {
    try own.m.draw(own.rb.tok.ptr);
    var out: [1]u32 = undefined;
    try own.m.download(own.rb.tok.ptr, std.mem.sliceAsBytes(&out));
    return out[0];
}

fn seconds(io: std.Io, since: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(since.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
}

/// A greedy reply: the prompt's chunks, then one-row decode steps; the output line and --report as Nemotron's run.
pub fn run(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options) !u8 {
    if (o.tokens.len == 0) return error.NoPromptTokens;
    const own = try open(gpa, io, ctx, dir, o.context);
    defer rc.native.closeOwned(own);
    const m = &own.m;
    m.graphs = o.graphs;
    const st = &own.rb.st;
    const c = own.w.config;
    const count = @max(1, @min(o.max_tokens, m.context -| o.tokens.len));
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    const t0 = std.Io.Timestamp.now(io, .awake);
    try prefill(m, st, o.tokens);
    try out.append(gpa, try greedy(own));
    const prefill_seconds = seconds(io, t0);
    const t1 = std.Io.Timestamp.now(io, .awake);
    while (out.items.len < count) {
        const t = out.items[out.items.len - 1];
        if (o.stop_eos and c.isEos(t)) break;
        try m.forward(st, &.{t}, .decode);
        try m.head(0);
        try out.append(gpa, try greedy(own));
    }
    const decode_seconds = seconds(io, t1);
    var digest: [32]u8 = undefined;
    const text = try core.ids_json.write(gpa, out.items);
    defer gpa.free(text);
    std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    const steps: f64 = @floatFromInt(@max(1, out.items.len - 1));
    const ms = decode_seconds * 1e3 / steps;
    std.debug.print("tokens {d} sha {s} prefill {d:.4}s decode {d:.4}s {d:.3} ms/token {d:.2} tok/s prompt {d:.1} tok/s\n", .{ out.items.len, hex[0..12], prefill_seconds, decode_seconds, ms, 1e3 / ms, @as(f64, @floatFromInt(o.tokens.len)) / prefill_seconds });
    if (o.report) |path| {
        const report = .{
            .engine = "zig-cuda",
            .model_type = rc.native.model_type,
            .prompt_tokens = o.tokens,
            .tokens = out.items,
            .token_sha256 = hex,
            .prefill_seconds = prefill_seconds,
            .decode_seconds = decode_seconds,
            .ms_per_token = ms,
            .tokens_per_second = 1e3 / ms,
            .prompt_tokens_per_second = @as(f64, @floatFromInt(o.tokens.len)) / prefill_seconds,
            .drafts = false,
            .graphs = m.captured() > 0, // a decode round replayed a graph
            .max_len = m.context,
        };
        const json = try std.json.Stringify.valueAlloc(gpa, report, .{});
        defer gpa.free(json);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
    }
    return 0;
}

/// kl's rows after a prefill: row 0 the prompt's last, then one-row decode steps.
const Decode = struct {
    m: *const fwd.Model,
    st: *fwd.Seq,

    fn row(p: *anyopaque, _: usize, out: []u8) anyerror!void {
        const x: *Decode = @ptrCast(@alignCast(p));
        try x.m.download(x.m.b.logits, out);
    }

    fn step(p: *anyopaque, _: usize, t: u32) anyerror!void {
        const x: *Decode = @ptrCast(@alignCast(p));
        try x.m.forward(x.st, &.{t}, .decode);
        try x.m.head(0);
    }
};

/// kl's rows of a whole prompt teacher-forced: row i after ids[0..i+1], the prompt forwarded in chunks of `chunk` rows.
const Chunks = struct {
    m: *const fwd.Model,
    st: *fwd.Seq,
    ids: []const u32,
    chunk: usize,
    base: usize = 0, // the current forward's first row
    done: usize = 0, // rows forwarded
    head_at: ?usize = null, // the forward's row in `rows`' first slot

    fn row(p: *anyopaque, i: usize, out: []u8) anyerror!void {
        const x: *Chunks = @ptrCast(@alignCast(p));
        if (i >= x.done) {
            const end = @min(x.ids.len, x.done + x.chunk);
            try x.m.forward(x.st, x.ids[x.done..end], .prompt);
            x.base = x.done;
            x.done = end;
            x.head_at = null;
        }
        const r = i - x.base;
        const v = x.m.c.vocab;
        if (x.head_at == null or r < x.head_at.? or r >= x.head_at.? + fwd.head_rows) {
            const r0 = r - r % fwd.head_rows;
            try x.m.headRows(r0, @min(fwd.head_rows, x.done - x.base - r0));
            x.head_at = r0;
        }
        try x.m.download(x.m.b.rows + (r - x.head_at.?) * v * 2, out);
    }

    /// The prompt's own tokens only: a reference whose tokens differ is another prompt.
    fn step(p: *anyopaque, i: usize, t: u32) anyerror!void {
        const x: *Chunks = @ptrCast(@alignCast(p));
        if (i + 1 >= x.ids.len or x.ids[i + 1] != t) return error.TokenMismatch;
    }
};

/// --chunk ROWS: every prompt row's logits; else the last prompt row and `max_tokens - 1` decode rows (--gold, greedy).
pub fn klRun(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options) !u8 {
    if (kl.refused(o.tokens, o.dump, o.against)) return 2;
    if (o.chunk) |c| if (c == 0 or c > fwd.prompt_rows) {
        std.debug.print("--chunk {d}: prompt rows a forward, 1 to {d}\n", .{ c, fwd.prompt_rows });
        return 2;
    };
    const own = try open(gpa, io, ctx, dir, o.context);
    defer rc.native.closeOwned(own);
    const m = &own.m;
    m.graphs = o.graphs;
    const st = &own.rb.st;
    const vocab = own.w.config.vocab;
    if (o.chunk) |c| {
        st.len = 0;
        var src: Chunks = .{ .m = m, .st = st, .ids = o.tokens, .chunk = c };
        return kl.compare(gpa, io, .{ .ctx = &src, .vocab = vocab, .row = Chunks.row, .step = Chunks.step }, 1, o.tokens[1..], o.tokens.len, o.dump, o.against);
    }
    try prefill(m, st, o.tokens);
    var src: Decode = .{ .m = m, .st = st };
    return kl.compare(gpa, io, .{ .ctx = &src, .vocab = vocab, .row = Decode.row, .step = Decode.step }, o.tokens.len, o.gold, o.max_tokens, o.dump, o.against);
}
