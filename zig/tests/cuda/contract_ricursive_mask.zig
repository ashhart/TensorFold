//! Ripple's grammar mask under the contract: sample.cu's tf_grammar_mask as the backend launches it, and its control.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rc = @import("ricursive_cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig"); // the family-neutral Checked wrapper

const c = core.contract.spec;
const g = core.contract.gates;
const be = rc.backend;

const vocab = 100352;
const threads = 256; // the backend's block: a thread takes every 256th word of a row
const masked: u16 = 0xFF80; // bf16 -inf

/// A round's most rows, a full verify window, a first token's row and two short windows, at the vocabulary.
const served = [_]c.Shape{ .{ .rows = be.round_rows, .n = vocab, .k = 1 }, .{ .rows = 16, .n = vocab, .k = 1 }, .{ .rows = 1, .n = vocab, .k = 1 }, .{ .rows = 3, .n = vocab, .k = 1 }, .{ .rows = 7, .n = vocab, .k = 1 } };

pub fn maskSpec(control: bool) c.Spec {
    const name = if (control) "control: cuda ricursive tf_grammar_mask reading a word's bits from the top (token t at bit 31 - t % 32)" else "cuda ricursive tf_grammar_mask (a row's disallowed tokens to -inf, in place)";
    const rule = "an allowed token keeps its bits, NaN patterns included; a disallowed token is -inf";
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .in_place = true, .nan = .{ .stated = rule }, .tiles = .{ .{}, .{ .size = 32 * threads, .step = 1, .min = 1 }, .{} }, .served = &served };
}

/// Rows of bf16 logits masked in place by each row's allowed-token words (lanes/grammar.zig's bit order).
pub const Mask = struct {
    f: cuda.Function,
    /// control: each word's bits read from the top
    ctl: bool = false,
    shape: c.Shape = undefined,
    logits: []u16 = &.{},
    bits: []u32 = &.{},
    dev: u64 = 0, // the bits
    out: u64 = 0, // the rows

    pub fn spec(self: *const Mask) c.Spec {
        return maskSpec(self.ctl);
    }

    pub fn control(self: *const Mask) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *Mask, a: std.mem.Allocator) void {
        a.free(self.logits);
        a.free(self.bits);
        self.logits = &.{};
        self.bits = &.{};
    }

    /// Logits in [-30, 30); words all set, all clear or random; non-finite: NaN and -Inf allowed, +Inf disallowed.
    pub fn prepare(self: *Mask, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        const nw = be.Ripple.maskWords(n);
        self.logits = try ref.bfs(a, rng, shape.rows * n, -30, 30);
        self.bits = try a.alloc(u32, shape.rows * nw);
        for (self.bits) |*w| w.* = switch (rng.uintLessThan(u8, 4)) {
            0 => 0xFFFF_FFFF,
            1 => 0,
            else => rng.int(u32),
        };
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, [_]bool{ true, false, true }, 0..) |x, allowed, r| if (r < shape.rows) {
            const t = (7 * r + 1) % n;
            self.logits[r * n + t] = c.toBf16(x);
            const bit = @as(u32, 1) << @intCast(t % 32);
            const w = &self.bits[r * nw + t / 32];
            w.* = if (allowed) w.* | bit else w.* & ~bit;
        };
        self.dev = try ref.input(rig, self.bits);
        self.out = try rig.output(shape.rows * n * 2); // in place: restored before each launch
        return true;
    }

    pub fn run(self: *Mask, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.out, std.mem.sliceAsBytes(self.logits));
        var a: cuda.Args = .{};
        a.add(self.out);
        a.add(@as(u32, @intCast(self.shape.n)));
        a.add(self.dev);
        a.add(@as(u32, @intCast(be.Ripple.maskWords(self.shape.n))));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = threads } }, rig.s, &a);
    }

    pub fn count(self: *const Mask) usize {
        return self.shape.rows * self.shape.n * 2;
    }

    pub fn check(self: *const Mask, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.n;
        const nw = be.Ripple.maskWords(n);
        for (0..self.shape.rows) |r| for (0..n) |t| {
            const allowed = (self.bits[r * nw + t / 32] >> @intCast(t % 32)) & 1 == 1;
            const want: u16 = if (allowed) self.logits[r * n + t] else masked;
            const have = std.mem.readInt(u16, got[(r * n + t) * 2 ..][0..2], .little);
            if (have != want) return try std.fmt.bufPrint(msg, "row {d} token {d}: {x} where {x} is due", .{ r, t, have, want });
        };
        return null;
    }
};

/// The registrations, held where the gates point at them (the caller's heap set).
pub const Set = struct {
    sample: cuda.Module = undefined,
    masks: [2]ref.Checked(Mask) = undefined,

    /// The served launch and its control (from `ctl`, contract_controls' image), appended to `list`.
    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, ctl: cuda.Module, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        set.sample = try cuda.Module.load(gpu.d, cuda.kernels.sample);
        errdefer set.sample.unload();
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        set.masks = .{
            .{ .rig = rig, .k = .{ .f = try set.sample.function("tf_grammar_mask") } },
            .{ .rig = rig, .k = .{ .f = try ctl.function("tf_ctl_rc_grammar_mask"), .ctl = true } },
        };
        for (&set.masks) |*x| try list.append(a, x.kernel());
    }

    pub fn deinit(set: *Set) void {
        for (&set.masks) |*x| x.deinit();
        set.sample.unload();
    }
};
