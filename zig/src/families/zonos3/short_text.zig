//! Short texts (zonos3.tts.short_text): a few words generated with a filler sentence after them, cut where it starts.
const std = @import("std");
const cfg = @import("config.zig");
const Allocator = std.mem.Allocator;

const C = cfg.codebooks;

/// registries/short_text.json of the reference's package: what counts as short, the cut, the silence classifier.
pub const Config = struct {
    /// The marker of the pause between the text and its filler (a private-use character of the tokenizer).
    pause_marker: []const u8,
    max_words: usize,
    /// Short: text tokens * max_ratio[1] <= filler tokens * max_ratio[0].
    max_ratio: [2]usize,
    lookback: usize,
    min_voiced: usize,
    max_frames: usize,
    max_retries: usize,
    threshold: i32,
    /// int8 [codebooks][codebook_size]: a frame is silent when its codes' weights sum past `threshold`.
    weights: []const i8,
    /// A filler sentence a request language, with the space that joins it to the pause marker.
    fillers: []const Filler,

    pub const Filler = struct { language: []const u8, text: []const u8 };

    pub fn parse(a: Allocator, text: []const u8) !Config {
        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        if (root != .object) return error.BadShortTextRegistry;
        const o = root.object;
        const silence = try object(o.get("silence"));
        const cut = try object(o.get("cut"));
        if (try int(silence.get("n_codebooks")) != C or try int(silence.get("codebook_size")) != cfg.codebook_size) return error.ShortTextRegistryOfAnotherCodec;
        const b64 = try string(silence.get("weights"));
        const decoder = std.base64.standard.Decoder;
        const size = decoder.calcSizeForSlice(b64) catch return error.BadShortTextRegistry;
        if (size != C * cfg.codebook_size) return error.BadShortTextRegistry;
        const weights = try a.alloc(u8, size);
        decoder.decode(weights, b64) catch return error.BadShortTextRegistry;
        const ratio = o.get("max_ratio") orelse return error.BadShortTextRegistry;
        if (ratio != .array or ratio.array.items.len != 2) return error.BadShortTextRegistry;
        const listed = try object(o.get("fillers"));
        const fillers = try a.alloc(Filler, listed.count());
        for (fillers, listed.keys(), listed.values()) |*f, language, v| f.* = .{ .language = language, .text = try string(v) };
        return .{
            .pause_marker = try string(o.get("pause_marker")),
            .max_words = try int(o.get("max_words")),
            .max_ratio = .{ try int(ratio.array.items[0]), try int(ratio.array.items[1]) },
            .lookback = try int(cut.get("lookback")),
            .min_voiced = try int(cut.get("min_voiced")),
            .max_frames = try int(cut.get("max_frames")),
            .max_retries = try int(cut.get("max_retries")),
            .threshold = std.math.cast(i32, (silence.get("threshold") orelse return error.BadShortTextRegistry).integer) orelse return error.BadShortTextRegistry,
            .weights = @ptrCast(weights),
            .fillers = fillers,
        };
    }

    fn object(v: ?std.json.Value) !std.json.ObjectMap {
        const x = v orelse return error.BadShortTextRegistry;
        return if (x == .object) x.object else error.BadShortTextRegistry;
    }

    fn string(v: ?std.json.Value) ![]const u8 {
        const x = v orelse return error.BadShortTextRegistry;
        return if (x == .string) x.string else error.BadShortTextRegistry;
    }

    fn int(v: ?std.json.Value) !usize {
        const x = v orelse return error.BadShortTextRegistry;
        return if (x == .integer) std.math.cast(usize, x.integer) orelse error.BadShortTextRegistry else error.BadShortTextRegistry;
    }

    pub fn filler(c: *const Config, language: []const u8) ?[]const u8 {
        for (c.fillers) |f| if (std.mem.eql(u8, f.language, language)) return f.text;
        return null;
    }

    /// The codes-only silence classifier: a frame with a code outside the codebook (EOA, pad) is not silent.
    pub fn silent(c: *const Config, codes: [C]i32) bool {
        var sum: i32 = 0;
        for (codes, 0..) |code, k| {
            if (code < 0 or code >= cfg.codebook_size) return false;
            sum += c.weights[k * cfg.codebook_size + @as(usize, @intCast(code))];
        }
        return sum > c.threshold;
    }
};

/// The cut of a short-text take over its generated frames (frame f is codebook k of generated row f + k).
pub const Cut = struct {
    a: Allocator,
    c: *const Config,
    /// The text-span index of the filler's first token.
    filler_token: usize,
    /// The frame where the reading reached the filler.
    enter: ?usize = null,
    /// `cut`: keep generated frames [0, frame); `run_on`: none was silent in time (`frame`: where the reading entered).
    decided: ?struct { kind: enum { cut, run_on }, frame: usize } = null,
    rows: std.ArrayList([C]i32) = .empty,
    /// Whether each complete frame is silent.
    is_silent: std.ArrayList(bool) = .empty,

    pub fn init(a: Allocator, c: *const Config, filler_token: usize) Cut {
        return .{ .a = a, .c = c, .filler_token = filler_token };
    }

    pub fn deinit(cut: *Cut) void {
        cut.rows.deinit(cut.a);
        cut.is_silent.deinit(cut.a);
    }

    /// Keeps generated rows [0, n): a LACI rollback.
    pub fn truncate(cut: *Cut, n: usize) void {
        cut.rows.shrinkRetainingCapacity(@min(n, cut.rows.items.len));
        cut.is_silent.shrinkRetainingCapacity(@min((n + 1) -| C, cut.is_silent.items.len));
        if (cut.enter) |e| if (e >= n) {
            cut.enter = null;
        };
        cut.decided = null;
        cut.decide();
    }

    /// Once a decode step: `frame` is the generated row of the step's readout.
    pub fn observeReading(cut: *Cut, frame: usize, pos: usize) void {
        if (cut.enter == null and pos >= cut.filler_token) {
            cut.enter = frame;
            cut.decide();
        }
    }

    pub fn observeRow(cut: *Cut, row: [C]i32) !void {
        try cut.rows.append(cut.a, row);
        if (cut.rows.items.len < C) return;
        // the frame this row completes
        const f = cut.rows.items.len - C;
        var codes: [C]i32 = undefined;
        for (&codes, 0..) |*code, k| code.* = cut.rows.items[f + k][k];
        try cut.is_silent.append(cut.a, cut.c.silent(codes));
        cut.decide();
    }

    fn decide(cut: *Cut) void {
        const enter = cut.enter orelse return;
        if (cut.decided != null) return;
        var voiced: usize = 0;
        for (cut.is_silent.items, 0..) |quiet, f| {
            voiced += @intFromBool(!quiet);
            if (f + cut.c.lookback < enter) continue;
            if (f + 1 > enter + cut.c.max_frames) {
                cut.decided = .{ .kind = .run_on, .frame = enter };
                return;
            }
            if (quiet and voiced >= cut.c.min_voiced) {
                cut.decided = .{ .kind = .cut, .frame = f + 1 };
                return;
            }
        }
    }
};

test "the cut is the first silent frame from two before the reading reached the filler, after three voiced ones" {
    const t = std.testing;
    // code 7 in codebook 0 marks a silent frame
    var weights: [C * cfg.codebook_size]i8 = @splat(0);
    weights[7] = 5;
    const c: Config = .{ .pause_marker = "", .max_words = 2, .max_ratio = .{ 2, 5 }, .lookback = 2, .min_voiced = 3, .max_frames = 15, .max_retries = 2, .threshold = 3, .weights = &weights, .fillers = &.{} };
    try t.expect(c.silent(.{ 7, 0, 0, 0, 0, 0, 0, 0, 0, 0 }));
    try t.expect(!c.silent(.{ 7, 0, 0, 0, 0, 0, 0, 0, 0, 4096 }));
    var cut = Cut.init(t.allocator, &c, 5);
    defer cut.deinit();
    // frames 0..5 voiced, 6 on silent: a frame is complete nine rows after its first
    for (0..20) |row| {
        var codes: [C]i32 = @splat(1);
        if (row >= 6) codes[0] = 7;
        try cut.observeRow(codes);
        if (row == 8) cut.observeReading(8, 5);
    }
    try t.expectEqual(@as(?usize, 8), cut.enter);
    try t.expectEqual(@as(usize, 7), cut.decided.?.frame);
    try t.expect(cut.decided.?.kind == .cut);
    cut.truncate(7);
    try t.expect(cut.decided == null and cut.enter == null);
}
