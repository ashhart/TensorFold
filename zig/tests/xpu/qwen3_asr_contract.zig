//! tf-xpu-qwen3_asr_contract-test [--device N] [--only TEXT]: gates 1 to 4, each control red, gate 5 live.
const std = @import("std");
const core = @import("core");
const kit = @import("qwen3_asr_contract_kit.zig");
const tower = @import("qwen3_asr_contract_tower.zig");
const row_k = @import("qwen3_asr_contract_rows.zig");
const lm = @import("qwen3_asr_contract_lm.zig");

const g = core.contract.gates;
const limits = core.contract.xpu_limits;
const records = core.contract.qwen3_asr_xpu_records;
const Mutation = kit.Mutation;

fn ms(io: std.Io) i96 {
    return @divTrunc(std.Io.Clock.awake.now(io).toNanoseconds(), std.time.ns_per_ms);
}

/// Every test on the heap (the gates hold pointers to them): each kernel, then its control.
const Suite = struct {
    a: std.mem.Allocator,
    frees: std.ArrayList(Free) = .empty,

    const Free = struct { ctx: *anyopaque, f: *const fn (std.mem.Allocator, *anyopaque) void };

    fn add(s: *Suite, comptime T: type, out: *std.ArrayList(g.Kernel), only: ?[]const u8, value: T) !void {
        const x = try s.a.create(T);
        x.* = value;
        try s.keep(T, x, out, only);
    }

    fn keep(s: *Suite, comptime T: type, x: *T, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        try s.frees.append(s.a, .{ .ctx = x, .f = struct {
            fn f(a: std.mem.Allocator, p: *anyopaque) void {
                const t: *T = @ptrCast(@alignCast(p));
                t.deinit();
                a.destroy(t);
            }
        }.f });
        const k = x.kernel();
        if (only == null or std.mem.indexOf(u8, k.spec.name, only.?) != null) try out.append(s.a, k);
    }

    fn deinit(s: *Suite) void {
        for (s.frees.items) |e| e.f(s.a, e.ctx);
        s.frees.deinit(s.a);
    }
};

fn collect(s: *Suite, dev: *kit.Dev, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
    const a = s.a;
    for ([_]struct { tower.Epi, Mutation }{ .{ .plain, .gemm_bf16_sums }, .{ .bias_res, .gemm_no_residual }, .{ .bias_gelu, .none } }) |p| {
        try s.add(tower.Gemm, out, only, try tower.Gemm.init(a, dev, .none, p[0]));
        if (p[1] != .none) try s.add(tower.Gemm, out, only, try tower.Gemm.init(a, dev, p[1], p[0]));
    }
    for ([_]Mutation{ .none, .gemv_steps_reversed }) |m| try s.add(tower.Gemm, out, only, try tower.Gemm.initGemv(a, dev, m));
    for ([_]Mutation{ .none, .conv_taps_transposed }) |m| try s.add(tower.Conv, out, only, try tower.Conv.init(a, dev, m));
    for ([_]Mutation{ .none, .flatten_f_major }) |m| try s.add(tower.Flatten, out, only, try tower.Flatten.init(a, dev, m));
    for ([_]Mutation{ .none, .attn_causal }) |m| try s.add(tower.AttnFull, out, only, try tower.AttnFull.init(a, dev, m));
    for ([_]struct { row_k.Which, Mutation }{ .{ .layernorm, .layernorm_no_bias }, .{ .gelu, .gelu_is_silu }, .{ .rms, .rms_no_weight } }) |p| {
        for ([_]Mutation{ .none, p[1] }) |m| try s.add(row_k.Row, out, only, try row_k.Row.init(a, dev, m, p[0]));
    }
    for ([_]Mutation{ .none, .swiglu_swapped }) |m| try s.add(row_k.Swiglu, out, only, try row_k.Swiglu.init(a, dev, m));
    for ([_]Mutation{ .none, .rope_turns_back }) |m| try s.add(lm.QkRope, out, only, try lm.QkRope.init(a, dev, m, .rows));
    const cache = try a.create(lm.QkCache);
    cache.init(a, dev, .none) catch |e| {
        a.destroy(cache);
        return e;
    };
    try s.keep(lm.QkCache, cache, out, only);
    for ([_]Mutation{ .none, .attn_reads_ahead }) |m| try s.add(lm.Attn, out, only, try lm.Attn.init(a, dev, m));
}

/// Gate 5 on this card: its properties against the Gen12LP row, each loaded kernel against its record, the control red.
fn live(a: std.mem.Allocator, dev: *kit.Dev) !usize {
    var n: usize = 0;
    const t = limits.gen12lp;
    const p = dev.ctx.gpu.props;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    const device = limits.Device{ .deviceId = p.deviceId, .numEUsPerSubslice = p.numEUsPerSubslice, .numThreadsPerEU = p.numThreadsPerEU, .physicalEUSimdWidth = p.physicalEUSimdWidth, .numSlices = p.numSlices, .numSubslicesPerSlice = p.numSubslicesPerSlice };
    const row = try limits.liveRow(device, t, &out, a);
    n += row;
    std.debug.print("{s} limits live {s}: PCI id 0x{x}, {d} EUs a subslice, {d} threads an EU, SIMD {d}, {d} slices x {d} subslices{s}", .{
        if (row == 0) "PASS" else "FAIL", t.name, device.deviceId, device.numEUsPerSubslice, device.numThreadsPerEU, device.physicalEUSimdWidth, device.numSlices, device.numSubslicesPerSlice, if (out.items.len > 0) ": " else "\n",
    });
    if (out.items.len > 0) std.debug.print("{s}", .{out.items});
    for (records.records) |rec| {
        out.clearRetainingCapacity();
        const img = std.meta.stringToEnum(kit.Image, rec.launch.image) orelse return error.UnknownImage;
        const sym = try a.allocSentinel(u8, rec.launch.symbol.len, 0);
        defer a.free(sym);
        @memcpy(sym, rec.launch.symbol);
        var k = try (try dev.module(img, .none)).kernelAt(sym);
        defer k.deinit();
        var props: KernelProps = .{};
        try dev.ctx.d.check(dev.ctx.d.api.zeKernelGetProperties(k.handle, &props), "zeKernelGetProperties");
        const u = rec.usage;
        const loaded = limits.Loaded{
            .threads = if (props.requiredGroupSize[0] == 0) u.threads else props.requiredGroupSize,
            .sub_group = if (props.requiredSubgroupSize == 0) u.sub_group else props.requiredSubgroupSize,
            .max_sub_group = if (props.maxSubgroupSize == 0) u.sub_group else props.maxSubgroupSize,
            .slm = props.localMemSize,
            .spill = props.spillMemSize,
            .args = props.numKernelArgs,
        };
        const problems = try limits.liveKernel(rec.launch, u, loaded, rec.args, t, &out, a);
        const fits = try limits.fits(rec.launch, t, &out, a);
        n += problems + fits;
        const items = @as(u64, loaded.threads[0]) * loaded.threads[1] * loaded.threads[2];
        std.debug.print("{s} limits live {s}: {d} work-items (required {d}), sub-group {d} (required {d}, the kernel's most {d}), {d} SLM bytes, " ++
            "{d} spill bytes, {d} private bytes, {d} arguments{s}", .{
            if (problems + fits != 0) "FAIL" else "PASS", rec.launch.name,            items,                 props.requiredGroupSize[0] * @max(props.requiredGroupSize[1], 1) * @max(props.requiredGroupSize[2], 1),
            loaded.sub_group,                             props.requiredSubgroupSize, props.maxSubgroupSize, loaded.slm,
            loaded.spill,                                 props.privateMemSize,       loaded.args,           if (out.items.len > 0) ": " else "\n",
        });
        if (out.items.len > 0) std.debug.print("{s}", .{out.items});
    }
    out.clearRetainingCapacity();
    const red = try limits.fits(records.control.launch, t, &out, a);
    if (red == 0) n += 1;
    std.debug.print("{s} limits live {s}: declared only{s}", .{ if (red != 0) "RED " else "MISSED", records.control.launch.name, if (out.items.len > 0) ": " else "\n" });
    if (out.items.len > 0) std.debug.print("{s}", .{out.items});
    return n;
}

const KernelProps = @import("xpu").ze.KernelProperties;

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const usage = "usage: tf-xpu-qwen3_asr_contract-test [--device N] [--only TEXT]\n";
    var ordinal: ?u32 = null;
    var only: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) {
            std.debug.print(usage, .{});
            return 2;
        }
        if (std.mem.eql(u8, args[i], "--device")) {
            ordinal = std.fmt.parseInt(u32, args[i + 1], 10) catch {
                std.debug.print(usage, .{});
                return 2;
            };
        } else if (std.mem.eql(u8, args[i], "--only")) only = args[i + 1] else {
            std.debug.print(usage, .{});
            return 2;
        }
    }
    var dev: kit.Dev = undefined;
    dev.init(ordinal) catch |e| {
        std.debug.print("FAIL qwen3_asr_contract: no card opened ({t})\n", .{e});
        return 1;
    };
    defer dev.deinit();
    var suite: Suite = .{ .a = a };
    defer suite.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    try collect(&suite, &dev, &kernels, only);
    var t: g.Tally = .{};
    const t0 = ms(init.io);
    try g.runAll(a, kernels.items, 0x2105, &t);
    const t1 = ms(init.io);
    const lim = if (only == null) try live(a, &dev) else 0;
    const t2 = ms(init.io);
    const tally = .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim, t1 - t0, t2 - t1 };
    std.debug.print("SUMMARY {s} 0x{x} {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: " ++
        "{d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms\n", .{ limits.gen12lp.name, dev.ctx.gpu.props.deviceId } ++ tally);
    return @intCast(@min(t.problems() + lim, 255));
}
