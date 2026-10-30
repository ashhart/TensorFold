//! Flash Next replay internals split along ownership and dispatch boundaries.
const fz = @import("replay.zig");
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const roles_gen = @import("roles_gen.zig");
const verdict_kernels = @import("verdict.zig");
const variants = fz.variants;
const D = fz.D;
const WIDE = fz.WIDE;
const MAXR = fz.MAXR;
const Buf = fz.Buf;
const Variant = fz.Variant;
const glue_source = fz.glue_source;
const xnew_source = @import("replay_expert_source.zig").text;
const hc_up_source = @import("replay_hc_source.zig").text;
const Run = fz.Run;
const Hc = fz.Hc;
const buffer = Run.buffer;
const class = Run.class;
const callAs = Run.callAs;

pub fn compileChecked(r: *Run) !void {
    @import("rotary_cache.zig").start(r);
    const lanes = !r.device.tensorUnits();
    for (&roles_gen.entries) |*e| {
        const gop = try r.roles.getOrPut(r.arena, e.site);
        if (gop.found_existing) return error.DuplicateSite;
        const v = r.variants.get(e.function) orelse blk: {
            var name = e.file;
            if (lanes) if (std.mem.indexOf(u8, e.file, "lane_qmm_bytes_grouped") != null) {
                const at = std.mem.indexOf(u8, e.file, ".metal") orelse return error.NoSource;
                name = try std.fmt.allocPrint(r.arena, "{s}-lanes{s}", .{ e.file[0..at], e.file[at..] });
            };
            const src: []const u8 = for (ks.flashnext_gen.sources) |s| {
                if (std.mem.eql(u8, s.name, name)) break s.text;
            } else return error.NoSource;
            const compiled = @import("rotary_cache.zig").forRun(r, src);
            const lib = try r.ownResource(try mtl.Library.fromSource(r.device, compiled, mtl.CompileOptions.mlx()));
            const nv = try r.arena.create(Variant);
            nv.* = .{ .inputs = try r.arena.dupe([]const u8, e.inputs), .outputs = try r.arena.dupe([]const u8, e.outputs), .meta = try r.arena.dupe([]const u8, e.meta), .pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, e.function, false)), .file = name, .name = e.function, .text = compiled };
            try r.variants.put(r.arena, nv.name, nv);
            break :blk nv;
        };
        gop.value_ptr.* = .{ .v = v, .grid = mtl.Size.of(e.grid[0], e.grid[1], e.grid[2]), .tg = mtl.Size.of(e.tg[0], e.tg[1], e.tg[2]) };
    }
    try r.compileGlue();
    if (r.xnew) { // the checked-in gate/up kernel's header (simd_topk, bsilu) with the full-width expert kernels after it
        var text: ?[]const u8 = null;
        for (&roles_gen.entries) |*e| if (std.mem.indexOf(u8, e.file, "qa_expert_gateup") != null) {
            text = (r.variants.get(e.function) orelse return error.NoGateup).text;
        };
        try r.compileXnew(text orelse return error.NoGateup);
    }
    if (r.hc_up) r.compileHc() catch |err| { // the checked-in row kernels then
        std.log.warn("fz_hc_up off: {s}", .{@errorName(err)});
        r.hc_up = false;
    };
}

pub fn compile(r: *Run, dir: []const u8) !void {
    @import("rotary_cache.zig").start(r);
    const path = try std.fmt.allocPrintSentinel(r.arena, "{s}/plan.json", .{dir}, 0);
    const f = try r.mapFile(path);
    const plan = try std.json.parseFromSliceLeaky(std.json.Value, r.arena, f.bytes[0..f.size], .{});
    var vit = plan.object.get("variants").?.object.iterator();
    while (vit.next()) |kv| {
        const o = kv.value_ptr.object;
        const src_path = try std.fmt.allocPrintSentinel(r.arena, "{s}/{s}", .{ dir, o.get("file").?.string }, 0);
        const src = try r.mapFile(src_path);
        const compiled = @import("rotary_cache.zig").forRun(r, src.bytes[0..src.size]);
        const lib = try r.ownResource(try mtl.Library.fromSource(r.device, compiled, mtl.CompileOptions.mlx()));
        const v = try r.arena.create(Variant);
        v.* = .{ .inputs = try strings(r.arena, o.get("inputs").?), .outputs = try strings(r.arena, o.get("outputs").?), .meta = try strings(r.arena, o.get("meta").?), .pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, kv.key_ptr.*, false)), .file = try std.fmt.allocPrint(r.arena, "{s}/{s}", .{ dir, o.get("file").?.string }), .name = kv.key_ptr.*, .text = compiled };
        try r.variants.put(r.arena, kv.key_ptr.*, v);
    }
    var rit = plan.object.get("roles").?.object.iterator();
    while (rit.next()) |kv| {
        const o = kv.value_ptr.object;
        try r.roles.put(r.arena, kv.key_ptr.*, .{ .v = r.variants.get(o.get("function").?.string).?, .grid = size3(o.get("grid").?), .tg = size3(o.get("threadgroup").?) });
    }
    try r.compileGlue();
    if (r.xnew) { // the recorded gate/up kernel's header (simd_topk, bsilu) with the full-width expert kernels after it
        var hit: ?[]const u8 = null;
        var it = plan.object.get("variants").?.object.iterator();
        while (it.next()) |kv| if (std.mem.indexOf(u8, kv.key_ptr.*, "qa_expert_gateup") != null) {
            hit = kv.value_ptr.object.get("file").?.string;
        };
        const fp = try std.fmt.allocPrintSentinel(r.arena, "{s}/{s}", .{ dir, hit orelse return error.NoGateup }, 0);
        const ff = try mtl.MappedFile.open(fp);
        try r.compileXnew(ff.bytes[0..ff.size]);
    }
    if (r.hc_up) r.compileHc() catch |err| { // the recorded kernels then
        std.log.warn("fz_hc_up off: {s}", .{@errorName(err)});
        r.hc_up = false;
    };
}

pub fn compileGlue(r: *Run) !void {
    const glue = try r.ownResource(try mtl.Library.fromSource(r.device, glue_source, mtl.CompileOptions.mlx()));
    r.kv_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, glue, "fz_kv_write", false));
    r.argmax_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, glue, "fz_argmax", false));
    r.add_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, glue, "fz_bcast_add", false));
    r.argids_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, glue, "fz_argmax_ids", false));
    r.argpart_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, glue, "fz_argmax_part", false));
    r.argjoin_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, glue, "fz_argmax_join", false));
    r.arg_parts = .{ .b = try r.buffer(MAXR * 64 * 2 * 4) };
    r.pleids_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, glue, "fz_ple_ids", false));
    r.copy_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, glue, "fz_copy_kept", false));
    const verdict = try r.ownResource(try mtl.Library.fromSource(r.device, verdict_kernels.source, mtl.CompileOptions.mlx()));
    r.accept_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, verdict, "fz_accept", false));
    r.lookup_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, verdict, "fz_lookup", false));
    r.force_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, verdict, "fz_force", false));
    r.span_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, verdict, "fz_span", false));
    r.positions_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, verdict, "fz_positions", false));
    const draws = try r.ownResource(try mtl.Library.fromSource(r.device, ks.nemotron_sample, mtl.CompileOptions.mlx()));
    r.draws = try @import("sample.zig").Draws.init(r.device, draws, MAXR);
    for ([_]mtl.Pipeline{ r.draws.full, r.draws.full_ids, r.draws.slices, r.draws.race, r.draws.race_ids }) |p| _ = try r.ownResource(p);
    for ([_]mtl.Buffer{ r.draws.ck, r.draws.ci }) |b| _ = try r.ownResource(b);
    r.take_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, glue, "fz_mtp_take", false));
    r.repack_w = try r.ownResource(try mtl.Pipeline.init(r.device, glue, "fz_repack_w", false));
    r.repack_s = try r.ownResource(try mtl.Pipeline.init(r.device, glue, "fz_repack_s", false));
    r.touch_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, glue, "fz_touch", false));
    const slib = try r.ownResource(try mtl.Library.fromSource(r.device, @import("rotary_cache.zig").forRun(r, ks.flashnext_select), mtl.CompileOptions.mlx()));
    r.sel_meta_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, slib, "fz_sel_meta", false));
    r.pool_abs_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, slib, "fz_idx_pool_abs", false));
    const plib = try r.ownResource(try mtl.Library.fromSource(r.device, @import("rotary_cache.zig").forRun(r, ks.flashnext_pool_pos), mtl.CompileOptions.mlx()));
    r.pool_pos_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, plib, "fz_idx_pool_pos", false));
    r.sink = .{ .b = try r.buffer(64) };
}

pub fn compileXnew(r: *Run, text: []const u8) !void {
    const cut = std.mem.indexOf(u8, text, "[[kernel]]") orelse return error.NoKernel;
    const define: []const u8 = if (r.xpack) "#define FZ_PACKED 1\n" else "#define FZ_PACKED 0\n";
    const expert_source = if (r.window.ssd_experts != null) try @import("fsr_xnew_source.zig").patch(r.arena, xnew_source) else xnew_source;
    const full = try std.mem.concat(r.arena, u8, &.{ define, text[0..cut], expert_source });
    r.xnew_header = try std.mem.concat(r.arena, u8, &.{ define, text[0..cut] });
    const lib = try r.ownResource(try mtl.Library.fromSource(r.device, full, mtl.CompileOptions.mlx()));
    r.xgu_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_xgu", false));
    r.xdown_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_xdown", false));
    r.dense8_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_dense8", false));
    r.dense16_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_dense16", false));
    r.route_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_route", false));
    r.router_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_router", false));
    r.xfused_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_xfused", false));
    r.xgu_sx_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_xgu_sx", false));
    r.xdown_sx_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_xdown_sx", false));
    r.xpart = .{ .b = try r.buffer(MAXR * 11 * 4 * D * 4) };
    r.xdone = .{ .b = try r.buffer(MAXR * 11 * 4) };
    r.ggu_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_ggu", false));
    r.gdown_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_gdown", false));
    r.ul = .{ .b = try r.buffer((1 + 2 * 320 + 320 * 32) * 4) };
}

pub fn compileHc(r: *Run) !void {
    const a = r.arena;
    var header: ?[]const u8 = null;
    var it = r.variants.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        if (std.mem.indexOf(u8, name, "qa_hc_down_row") != null) {
            const text = try variantText(a, kv.value_ptr.*);
            header = text[0 .. std.mem.indexOf(u8, text, "[[max_total_threads_per_threadgroup") orelse return error.NoKernel];
        }
        if (std.mem.indexOf(u8, name, "qa_hc_up_") == null) continue;
        const text = try variantText(a, kv.value_ptr.*);
        for ([_][]const u8{ "int S = 4;", "int D = 2560;", "int BITS = 6;", "int GS = 32;", "int LOW = 320;", "int KS = 10;" }) |c|
            if (std.mem.indexOf(u8, text, c) == null) return error.HcConstants;
    }
    const src = try std.mem.concat(a, u8, &.{ header orelse return error.NoHcHeader, hc_up_source });
    const lib = try r.ownResource(try mtl.Library.fromSource(r.device, src, mtl.CompileOptions.mlx()));
    r.hc_up_pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_hc_up", false));
}

pub fn variantText(a: std.mem.Allocator, v: *const Variant) ![]const u8 {
    if (v.text.len != 0) return v.text;
    const f = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(a, "{s}", .{v.file}, 0));
    return f.bytes[0..f.size];
}

pub fn hcProject(r: *Run, down: []const u8, up: []const u8, as_rows: usize, hn: Buf, ssp: Buf, hc: Hc, eps: Buf, rows: Buf, part: Buf, mixed: Buf, inj: Buf) !void {
    var cd: [2]variants.Const = undefined;
    var cu: [2]variants.Const = undefined;
    const kd = variants.width(&cd, "BITS", "GS", if (hc.w) |w| w[0] else null);
    const ku = variants.width(&cu, "BITS", "GS", if (hc.w) |w| w[1] else null);
    try variants.callAs(r, down, as_rows, &.{ hn, ssp, hc.scale, hc.dw, hc.ds, hc.db, eps, rows }, &.{part}, kd);
    const ins = [_]Buf{ hn, ssp, hc.scale, part, hc.uw, hc.us, hc.ub, eps, rows };
    if (!r.hc_up or hc.w != null or as_rows < 3 or as_rows > 8) return variants.callAs(r, up, as_rows, &ins, &.{ mixed, inj }, ku);
    if (r.skip & class(up) != 0 or r.hcskip & 4 != 0) return;
    const nd: u32 = @intCast(hc.dw.b.length() / (WIDE * 6 / 8));
    r.enc.setPipeline(r.hc_up_pipe);
    for (ins ++ [_]Buf{ mixed, inj }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
    r.enc.setBytes(std.mem.asBytes(&nd), 11);
    r.enc.dispatchThreads(mtl.Size.of(D / 16 * 640, (as_rows + 3) / 4, 1), mtl.Size.of(640, 1, 1));
    if (!r.serial) r.enc.barrier();
}

pub fn strings(a: std.mem.Allocator, v: std.json.Value) ![][]const u8 {
    const out = try a.alloc([]const u8, v.array.items.len);
    for (v.array.items, 0..) |s, i| out[i] = s.string;
    return out;
}

pub fn size3(v: std.json.Value) mtl.Size {
    const a = v.array.items;
    return mtl.Size.of(@intCast(a[0].integer), @intCast(a[1].integer), @intCast(a[2].integer));
}

pub fn repack(r: *Run, ex: []Buf) !void {
    const cb = r.queue.commandBuffer();
    const enc = cb.compute(.concurrent);
    for (0..6) |j| {
        const parts: u32 = if (j < 4) 16 else 4;
        const d = [2]u32{ parts, 5 };
        const words = ex[j * 3].b.length() / 4;
        const rows = words / (parts * 30);
        enc.setPipeline(r.repack_w);
        enc.setBuffer(ex[j * 3].b, ex[j * 3].off, 0);
        enc.setBytes(std.mem.asBytes(&d), 1);
        enc.dispatchThreads(mtl.Size.of(128 * rows, 1, 1), mtl.Size.of(128, 1, 1));
        for (1..3) |k| {
            enc.setPipeline(r.repack_s);
            enc.setBuffer(ex[j * 3 + k].b, ex[j * 3 + k].off, 0);
            enc.setBytes(std.mem.asBytes(&d), 1);
            enc.dispatchThreads(mtl.Size.of(128 * rows, 1, 1), mtl.Size.of(128, 1, 1));
        }
    }
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("repack failed: {s}", .{msg});
        return error.GpuFailed;
    }
}
