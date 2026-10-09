//! Greedy Nemotron-H generation through the multi-row window path (one row a step): numerics of nem_rows.cl.

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("nemotron_xpu").cfg;
const model = @import("nemotron_xpu").model;
const nw = @import("nemotron_xpu").win;
const stop = @import("xpu").stop;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const Top = struct { id: u32, v: f32 };

/// The five largest logits, best first.
fn top5(logits: []const f32) [5]Top {
    var best: [5]Top = @splat(.{ .id = 0, .v = -std.math.inf(f32) });
    for (logits, 0..) |v, i| {
        if (v <= best[4].v) continue;
        var j: usize = 4;
        while (j > 0 and v > best[j - 1].v) : (j -= 1) best[j] = best[j - 1];
        best[j] = .{ .id = @intCast(i), .v = v };
    }
    return best;
}

fn parseIds(gpa: std.mem.Allocator, s: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeScalar(u8, s, ',');
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseInt(u32, std.mem.trim(u8, t, " \r\n"), 10));
    return out.toOwnedSlice(gpa);
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !u8 {
    stop.install();
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: xpu-nem_gen-test CHECKPOINT_DIR ID,ID,... N_NEW [--logits] [--trace] [--ctx N] [--ignore-eos] [--force ID,ID,...] [--bf16-logits]\n", .{});
        return 2;
    }
    const dir = args[1];
    const prompt = try parseIds(gpa, if (args[2][0] == 0x40) try ld.readFile(gpa, "/", args[2][2..]) else args[2]); // @/abs/file: ids from a file
    const n_new = try std.fmt.parseInt(u32, args[3], 10);
    var show_logits = false;
    var trace = false;
    var ignore_eos = false;
    var ctx: u32 = 4096;
    var bf16_logits = false;
    var forced: ?[]u32 = null;
    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--logits")) show_logits = true else if (std.mem.eql(u8, args[i], "--trace")) trace = true else if (std.mem.eql(u8, args[i], "--ignore-eos")) ignore_eos = true else if (std.mem.eql(u8, args[i], "--bf16-logits")) bf16_logits = true else if (std.mem.eql(u8, args[i], "--force")) {
            i += 1;
            forced = try parseIds(gpa, args[i]);
        } else if (std.mem.eql(u8, args[i], "--ctx")) {
            i += 1;
            ctx = try std.fmt.parseInt(u32, args[i], 10);
        } else return error.BadArgument;
    }
    if (forced != null) ignore_eos = true;
    // forcing: one step line per forced token, so the last forced token is never fed
    if (prompt.len == 0) return error.EmptyPrompt; // before the device is opened
    if (forced) |f| if (f.len == 0) return error.EmptyForced;
    const total = if (forced) |f| prompt.len + f.len - 1 else prompt.len + n_new;
    if (total > ctx) return error.ContextTooSmall;

    const cfg_bytes = try ld.readFile(gpa, dir, "config.json");
    const cfg = try cfgm.parse(gpa, cfg_bytes);
    var r = try rt.open();
    defer r.deinit();
    var props: @import("xpu").abi.DeviceProperties = undefined;
    props.stype = @import("xpu").abi.structure_type_device_properties;
    props.next = null;
    if (r.drv.api.zeDeviceGetProperties(r.device, &props) == 0) {
        std.debug.print("device {s}, max single allocation {d:.2} GB\n", .{ std.mem.sliceTo(&props.name, 0), @as(f64, @floatFromInt(props.max_mem_alloc_size)) / 1e9 });
    }
    var l = try ld.Loader.init(gpa, &r, dir);
    const t_load = nowNs();
    var m = try model.Model.load(gpa, &r, &l, cfg.value, ctx);
    const pf_rows: u32 = if (std.c.getenv("NEM_PREFILL_ROWS")) |v| (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0) else 0; // prompt in windows of this many rows (above 16: the prefill GEMMs)
    if (pf_rows > 16) nw.default_rows = pf_rows;
    var w = try nw.Win.init(gpa, &m);
    std.debug.print("loaded in {d:.1} s, device memory {d:.2} GB ({d} tensors in headers)\n", .{ @as(f64, @floatFromInt(nowNs() - t_load)) / 1e9, @as(f64, @floatFromInt(l.total)) / 1e9, l.map.count() });

    const logits = try gpa.alloc(f32, cfg.value.vocab_size);
    var generated: std.ArrayList(u32) = .empty;
    var step_ns: std.ArrayList(u64) = .empty;
    var next: u32 = 0;
    m.bf16_logits = bf16_logits;
    var t: usize = 0;
    if (pf_rows > 0 and prompt.len > 1) {
        const t_pf = nowNs();
        while (t + 1 < prompt.len) {
            try nw.stopCheck(&r);
            const rows = @min(pf_rows, prompt.len - 1 - t);
            try w.forward(&m, prompt[t .. t + rows], true, 0);
            try step_ns.append(gpa, 0);
            t += rows;
        }
        try r.sync();
        const pf_s = @as(f64, @floatFromInt(nowNs() - t_pf)) / 1e9;
        std.debug.print("prefill {d} tokens in {d:.1} s ({d:.1} tokens/s)\n", .{ t, pf_s, @as(f64, @floatFromInt(t)) / pf_s });
        for (1..t) |_| try step_ns.append(gpa, 0); // keep one entry a prompt token for the statistics below
        while (step_ns.items.len > prompt.len - 1) _ = step_ns.pop();
    }
    while (t < total) : (t += 1) {
        try nw.stopCheck(&r);
        const tok: u32 = if (t < prompt.len) prompt[t] else if (forced) |f| f[t - prompt.len] else next;
        const last_prompt = t + 1 == prompt.len;
        const need = t + 1 >= prompt.len;
        m.trace = trace and last_prompt;
        const t0 = nowNs();
        try w.forward(&m, &[_]u32{tok}, true, if (need) 1 else 0);
        if (need) {
            var one: [1]i32 = undefined;
            try w.rowArgmax(&m, &one);
            next = @intCast(one[0]);
        } else try r.sync();
        try step_ns.append(gpa, nowNs() - t0);
        if (need) {
            try generated.append(gpa, next);
            if (show_logits) {
                try w.fetchRowLogits(&m, logits);
                const tp = top5(logits);
                std.debug.print("step {d} pos {d} top5:", .{ generated.items.len - 1, t });
                for (tp) |e| std.debug.print(" {d}:{d:.4}", .{ e.id, e.v });
                std.debug.print("\n", .{});
                if (forced) |f| { // log-probability of the forced token and of our top-1, for the reference comparison
                    const mx: f64 = tp[0].v;
                    var se: f64 = 0;
                    for (logits) |v| se += @exp(@as(f64, v) - mx);
                    const lse = mx + @log(se);
                    std.debug.print("step {d} lpf {d} {d:.5} {d:.5}\n", .{ generated.items.len - 1, f[generated.items.len - 1], @as(f64, logits[f[generated.items.len - 1]]) - lse, @as(f64, tp[0].v) - lse });
                }
            }
            if (!ignore_eos) for (cfg.value.eos_token_id) |e| if (next == e) {
                t = total;
                break;
            };
        }
    }
    std.debug.print("generated ids:", .{});
    for (generated.items) |g| std.debug.print(" {d}", .{g});
    std.debug.print("\n", .{});
    var sum_prompt: u64 = 0;
    var sum_gen: u64 = 0;
    for (step_ns.items, 0..) |ns, k| {
        if (k < prompt.len - 1) sum_prompt += ns else sum_gen += ns;
    }
    const ng = step_ns.items.len -| (prompt.len - 1);
    std.debug.print("prompt tokens {d} ({d:.1} ms/token), generated {d} ({d:.1} ms/token = {d:.2} tokens/s), first token {d:.1} ms\n", .{
        prompt.len,
        @as(f64, @floatFromInt(sum_prompt)) / 1e6 / @as(f64, @floatFromInt(@max(prompt.len - 1, 1))),
        ng,
        @as(f64, @floatFromInt(sum_gen)) / 1e6 / @as(f64, @floatFromInt(@max(ng, 1))),
        @as(f64, @floatFromInt(ng)) / (@as(f64, @floatFromInt(sum_gen)) / 1e9),
        @as(f64, @floatFromInt(step_ns.items[0])) / 1e6,
    });
    std.debug.print("device memory {d:.2} GB\n", .{@as(f64, @floatFromInt(l.total)) / 1e9});
    return 0;
}
