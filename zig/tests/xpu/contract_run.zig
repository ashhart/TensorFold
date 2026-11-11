//! tf-xpu-contract-run: the XPU kernels through gates 1 to 4 on the card, each control red, then gate 5 live.
const std = @import("std");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const gate5 = @import("contract_limits.zig");
const live5 = @import("contract_live.zig");
const norm_k = @import("contract_norm.zig");
const swiglu_k = @import("contract_swiglu.zig");
const argmax_k = @import("contract_argmax.zig");
const topk_k = @import("contract_topk.zig");
const embed_k = @import("contract_embed.zig");
const controls_k = @import("contract_controls.zig");

const g = core.gates;

fn ms(io: std.Io) i96 {
    return @divTrunc(std.Io.Clock.awake.now(io).toNanoseconds(), std.time.ns_per_ms);
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var ordinal: ?u32 = null;
    if (args.len == 3 and std.mem.eql(u8, args[1], "--device")) {
        ordinal = std.fmt.parseInt(u32, args[2], 10) catch {
            std.debug.print("usage: tf-xpu-contract-run [--device N]\n", .{});
            return 2;
        };
    } else if (args.len != 1) {
        std.debug.print("usage: tf-xpu-contract-run [--device N]\n", .{});
        return 2;
    }
    var dev: rig_.Dev = undefined;
    dev.init(ordinal) catch |e| {
        std.debug.print("FAIL xpu-contract-run: no card opened ({t})\n", .{e});
        return 1;
    };
    defer dev.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    var norms: [std.enums.values(norm_k.Which).len]norm_k.Norm = undefined;
    for (&norms, std.enums.values(norm_k.Which)) |*x, w| x.* = try norm_k.Norm.init(a, &dev, w);
    defer for (&norms) |*x| x.deinit();
    for (&norms) |*x| try kernels.append(a, x.kernel());
    var swiglu: [std.enums.values(swiglu_k.Part).len]swiglu_k.Swiglu = undefined;
    for (&swiglu, std.enums.values(swiglu_k.Part)) |*x, p| x.* = try swiglu_k.Swiglu.init(a, &dev, p);
    defer for (&swiglu) |*x| x.deinit();
    for (&swiglu) |*x| try kernels.append(a, x.kernel());
    var argmax = try argmax_k.Argmax.init(a, &dev);
    defer argmax.deinit();
    try kernels.append(a, argmax.kernel());
    var topk: [std.enums.values(topk_k.Which).len]topk_k.Topk = undefined;
    for (&topk, std.enums.values(topk_k.Which)) |*x, w| x.* = try topk_k.Topk.init(a, &dev, w);
    defer for (&topk) |*x| x.deinit();
    for (&topk) |*x| try kernels.append(a, x.kernel());
    var embed = try embed_k.Embed.init(a, &dev);
    defer embed.deinit();
    try kernels.append(a, embed.kernel());
    var controls: [std.enums.values(controls_k.Which).len]controls_k.Control = undefined;
    for (&controls, std.enums.values(controls_k.Which)) |*x, w| x.* = try controls_k.Control.init(a, &dev, w);
    defer for (&controls) |*x| x.deinit();
    for (&controls) |*x| try kernels.append(a, x.kernel());
    var t: g.Tally = .{};
    const t0 = ms(init.io);
    try g.runAll(a, kernels.items, 0x6b63, &t);
    const t1 = ms(init.io);
    const lim = try gate5.table(a) + try live5.live(a, &dev);
    const t2 = ms(init.io);
    const tally = .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim, t1 - t0, t2 - t1 };
    std.debug.print("SUMMARY bmg-g31 0x{x} {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: " ++
        "{d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms\n", .{dev.ctx.gpu.props.deviceId} ++ tally);
    return @intCast(t.problems() + lim);
}
