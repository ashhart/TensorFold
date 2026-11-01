//! idol_decoder.cu's kernels and controls under the contract (idol-decoder-contract), idol.cu's attention at anchors.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const idol = @import("idol_cuda");
const dk = @import("idol_decoder_cuda");
const check = @import("check.zig");
const case = @import("contract_idol_case.zig");
const steps = @import("contract_idol_dec_steps.zig");
const dattn = @import("contract_idol_dec_attn.zig");
const attn = @import("contract_idol_attn.zig");
const r = @import("contract_idol_ref.zig");

const g = core.contract.gates;
const c = core.contract.spec;
const Rig = @import("contract_rig.zig").Rig;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

/// Self attention over 32k's and 262k's anchors, one head, and 16 heads at 1,025 (1M's 32,768: too slow in float64).
const self_served = [_]c.Shape{ .{ .rows = 1, .n = 1024, .k = 1024 }, .{ .rows = 1, .n = 8192, .k = 8192 }, .{ .rows = r.heads, .n = 1025, .k = 1025 } };

fn Step(comptime Op: type) type {
    return case.CaseOf(dk.Kernels, Op);
}

/// One control: the kernel field it replaces, the control's function, what it breaks, the gate that must go red.
const Control = struct { field: []const u8, symbol: [:0]const u8, what: []const u8, gate: g.Gate };

const planted = [_]Control{
    .{ .field = "embed", .symbol = "tf_ctl_idol_dec_embed_pe16", .what = "embed adds fp16(pe)", .gate = .shapes },
    .{ .field = "embed", .symbol = "tf_ctl_idol_dec_embed_sum16", .what = "embed sums in_proj in fp16", .gate = .sums },
    .{ .field = "half", .symbol = "tf_ctl_idol_dec_half_early", .what = "half rounds y to fp16 before the bias", .gate = .shapes },
    .{ .field = "half", .symbol = "tf_ctl_idol_dec_half_writes", .what = "half writes its input", .gate = .repeat },
    .{ .field = "kv", .symbol = "tf_ctl_idol_dec_kv_early", .what = "to_kv rounds ctx w to fp16 before the bias", .gate = .shapes },
    .{ .field = "kv", .symbol = "tf_ctl_idol_dec_kv_sum16", .what = "to_kv sums in fp16", .gate = .sums },
    .{ .field = "heads", .symbol = "tf_ctl_idol_dec_heads_fmax", .what = "heads clamp the norm by fmaxf, dropping a NaN", .gate = .nonfinite },
    .{ .field = "heads", .symbol = "tf_ctl_idol_dec_heads_div16", .what = "heads add fp16(x / n)", .gate = .shapes },
    .{ .field = "heads", .symbol = "tf_ctl_idol_dec_heads_sq16", .what = "heads sum the squares in fp16", .gate = .sums },
    .{ .field = "attention", .symbol = "tf_ctl_idol_dec_attention_unmasked", .what = "attention leaves keys past Lk unmasked", .gate = .shapes },
    .{ .field = "attention", .symbol = "tf_ctl_idol_dec_attention_pv16", .what = "attention sums P V in fp16", .gate = .sums },
    .{ .field = "few", .symbol = "tf_ctl_idol_dec_few_p16", .what = "the short form rounds P to fp16", .gate = .shapes },
    .{ .field = "few", .symbol = "tf_ctl_idol_dec_few_pv16", .what = "the short form sums P V in fp16", .gate = .sums },
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: dk.Kernels,
    own: idol.Kernels,
    ctl: cuda.Module,
    /// The kernels with one function swapped for a control's, one copy a control.
    swapped: [planted.len]dk.Kernels = undefined,
    names: [planted.len][]u8 = @splat(&.{}),
    embed: [1 + 2]Step(steps.Embed) = undefined,
    half: [1 + 2]Step(steps.Half) = undefined,
    kv: [1 + 2]Step(steps.Kv) = undefined,
    heads: [1 + 3]Step(steps.Heads) = undefined,
    attention: [2 + 4]dattn.Attention = undefined,
    self_attn: attn.Attn = undefined,
    made: struct { embed: usize = 0, half: usize = 0, kv: usize = 0, heads: usize = 0, attention: usize = 0 } = .{},

    /// The set, its kernels appended to `list` (each kernel's controls after it).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try dk.Kernels.load(gpu.d);
        errdefer k.unload();
        var own = try idol.Kernels.load(gpu.d);
        errdefer own.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k, .own = own, .ctl = ctl };
        errdefer for (set.names) |n| gpu.gpa.free(n);
        for (&set.swapped, &set.names, planted) |*w, *n, x| {
            w.* = k;
            inline for (.{ "embed", "half", "kv", "heads", "attention", "few" }) |f| {
                if (std.mem.eql(u8, x.field, f)) @field(w.*, f) = try ctl.function(x.symbol);
            }
            n.* = try std.fmt.allocPrint(gpu.gpa, "control: idol decoder {s}", .{x.what});
        }
        const rig: Rig = .{ .gpu = gpu, .s = s };
        set.add(rig, null);
        for (0..planted.len) |i| set.add(rig, i);
        set.self_attn = .{ .rig = rig, .k = &set.own, .name = "cuda idol decoder self attention: tf_idol_attention over anchors", .served = &self_served };
        for (set.embed[0..set.made.embed]) |*x| try list.append(a, x.kernel());
        for (set.half[0..set.made.half]) |*x| try list.append(a, x.kernel());
        for (set.kv[0..set.made.kv]) |*x| try list.append(a, x.kernel());
        for (set.heads[0..set.made.heads]) |*x| try list.append(a, x.kernel());
        for (set.attention[0..set.made.attention]) |*x| try list.append(a, x.kernel());
        try list.append(a, set.self_attn.kernel());
        return set;
    }

    /// The kernels' tests on the set's kernels, or with control `ctl` the one test of the kernel it breaks.
    fn add(set: *Set, rig: Rig, ctl: ?usize) void {
        const x: ?Control = if (ctl) |i| planted[i] else null;
        const w: *const dk.Kernels = if (ctl) |i| &set.swapped[i] else &set.k;
        const name: ?[]const u8 = if (ctl) |i| set.names[i] else null;
        const gate: ?g.Gate = if (x) |y| y.gate else null;
        const field = if (x) |y| y.field else "";
        const all = x == null;
        if (all or std.mem.eql(u8, field, "embed")) {
            set.embed[set.made.embed] = case.makeOf(dk.Kernels, steps.Embed, rig, w.*, name orelse "cuda idol decoder tf_idol_dec_embed, in_proj and the point embedding", 0, gate);
            set.made.embed += 1;
        }
        if (all or std.mem.eql(u8, field, "half")) {
            set.half[set.made.half] = case.makeOf(dk.Kernels, steps.Half, rig, w.*, name orelse "cuda idol decoder tf_idol_dec_half, fp16(y + b)", 0, gate);
            set.made.half += 1;
        }
        if (all or std.mem.eql(u8, field, "kv")) {
            set.kv[set.made.kv] = case.makeOf(dk.Kernels, steps.Kv, rig, w.*, name orelse "cuda idol decoder tf_idol_dec_kv, to_kv of the fp16 latent", 0, gate);
            set.made.kv += 1;
        }
        if (all or std.mem.eql(u8, field, "heads")) {
            set.heads[set.made.heads] = case.makeOf(dk.Kernels, steps.Heads, rig, w.*, name orelse "cuda idol decoder tf_idol_dec_heads, MultiHeadRMSNorm in place", 0, gate);
            set.made.heads += 1;
        }
        const forms = [_]struct { []const u8, []const u8 }{
            .{ "attention", "cuda idol decoder tf_idol_dec_attention, rows<64, 8>, P to fp16" },
            .{ "few", "cuda idol decoder tf_idol_dec_few, fp32 for 8 queries or fewer" },
        };
        for (forms, 0..) |f, which| if (all or std.mem.eql(u8, field, f[0])) {
            set.attention[set.made.attention] = .{ .rig = rig, .k = w, .which = which, .name = name orelse f[1], .control = gate };
            set.made.attention += 1;
        };
    }

    pub fn deinit(set: *Set) void {
        for (set.embed[0..set.made.embed]) |*x| x.deinit();
        for (set.half[0..set.made.half]) |*x| x.deinit();
        for (set.kv[0..set.made.kv]) |*x| x.deinit();
        for (set.heads[0..set.made.heads]) |*x| x.deinit();
        for (set.attention[0..set.made.attention]) |*x| x.deinit();
        set.self_attn.deinit();
        for (set.names) |n| set.gpa.free(n);
        set.ctl.unload();
        set.own.unload();
        set.k.unload();
        set.gpa.destroy(set);
    }
};

/// The decoders' kernels' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x1d0d, &t);
    std.debug.print("SUMMARY idol decoder: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "idol-decoder-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
