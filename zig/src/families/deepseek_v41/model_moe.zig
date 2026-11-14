//! The model's routed-expert block.
const builtin = @import("builtin");
const std = @import("std");
const mtl = @import("metal");
const checkpoint = @import("core").checkpoint_metal;
const exl3 = @import("core").exl3_projection;
const config = @import("config.zig");
const weights = @import("weights.zig");
const work = @import("workspace.zig");
const attention = @import("attention.zig");
const kv = @import("kv.zig");
const ops = @import("ops.zig");
const mhc = @import("mhc.zig");
const csa2 = @import("csa2.zig");
const csa2_index = @import("csa2_index.zig");
const rope = @import("rope.zig");
const route = @import("route.zig");
const fusion = @import("model_moe_fuse.zig");
const engram = @import("engram.zig");
const engram_stream = @import("engram_rows_safetensors.zig");
const gpu_counters = @import("profiler.zig");
const expert_groups = @import("core").expert_groups;
const scan_probe = @import("scan_probe.zig");
const model = @import("model.zig");
const Model = model.Model;
const CondProf = Model.CondProf;
const View = model.View;
const tensor = model.tensor;
const asF32 = model.asF32;
const convert = model.convert;
const splitStage = model.splitStage;
const completed = model.completed;

pub fn moe(m: *Model, enc: *mtl.ComputeEncoder, e: ?*mtl.CommandBuffer, w: *const weights.Moe, rows: u32, layer_index: u32, topk: u32, prof: ?*CondProf) !void {
    var encoder = enc.*;
    // A block's expert count is what its descriptor table was bound with: 384 for the backbone.
    const experts: u32 = @intCast(w.descriptors[0].length() / @sizeOf(exl3.ExpertDescriptor));
    const ws = &m.workspace;
    const p = &m.pipelines;
    // A block's own router geometry: the backbone's 384 experts and six picks.
    const picks: u32 = if (topk == 0) m.cfg.topk else topk;
    const dspark_block: bool = experts != m.cfg.experts;
    // The drafter's logits live in their own buffer: its gate has 128 columns.
    const logits_dest: View = if (dspark_block) blk: {
        if (m.dspark_logits == null) m.dspark_logits = try m.device.buffer(256 * @as(usize, experts) * 4, mtl.ResourceOptions.shared);
        break :blk View{ .buffer = m.dspark_logits.?, .offset = 0 };
    } else ws.view(.router_logits);
    // TF_DSV41_ROUTE_TENSOR: the tiled path.
    const route_tensor = std.c.getenv("TF_DSV41_ROUTE_TENSOR") != null;
    // The bf16 twin is built a layer at load (weights.bind), not cached once for the model: the gate is per layer.
    const gemm_min_rows: u32 = if (std.c.getenv("TF_DSV41_ROUTE_GEMM_MINROWS")) |v|
        (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 64)
    else
        64;
    // The gate GEMM is the promoted default for prompt rows. TF_DSV41_ROUTE_GEMM=0 restores the shipped tree.
    const gemm_raw = std.c.getenv("TF_DSV41_ROUTE_GEMM");
    const gemm_on = if (gemm_raw) |r| !std.mem.eql(u8, std.mem.span(r), "0") else true;
    const gemm_env = gemm_on and rows >= gemm_min_rows;
    // The f16 entry is the one that ships for these packs: their router gate is f16.
    const gate_f16_gemm = gemm_env and p.ops.gate_f16_f32 != null and w.gate.dtype == .f16;
    const gate_bf16_gemm = !gate_f16_gemm and gemm_env and p.ops.gate_f32 != null and w.gate.dtype == .bf16;
    const route_gemm = gate_f16_gemm or gate_bf16_gemm;
    if (std.c.getenv("TF_DSV41_ROUTE_DEBUG") != null)
        std.log.info("route gemm: flag {} f16 {} bf16f32 {} bf16out {} dtype {s} rows {d} -> {s}", .{
            std.c.getenv("TF_DSV41_ROUTE_GEMM") != null, p.ops.gate_f16_f32 != null, p.ops.gate_f32 != null,
            p.ops.gate != null,                          @tagName(w.gate.dtype),     rows,
            if (route_gemm) "gemm" else "dense-col",
        });
    // TF_DSV41_ROUTE_GEMM_SCRATCH: the GEMM runs but writes a scratch.
    const gate_scratch = route_gemm and std.c.getenv("TF_DSV41_ROUTE_GEMM_SCRATCH") != null;
    // TF_DSV41_ROUTE_GEMM_PRIVATE: the GEMM owns its buffer; the workspace logits stay untouched.
    const gate_private = route_gemm and std.c.getenv("TF_DSV41_ROUTE_GEMM_PRIVATE") != null;
    var gate_pdest: ?mtl.Buffer = null;
    if (route_gemm) {
        const dest = if (gate_scratch or gate_private) blk: {
            const b = try m.device.buffer(@as(usize, rows) * experts * 4, mtl.ResourceOptions.shared);
            gate_pdest = b;
            break :blk View{ .buffer = b, .offset = 0 };
        } else logits_dest;
        if (gate_f16_gemm) {
            if (m.gate_x == null) m.gate_x = try m.device.buffer(@as(usize, m.workspace.plan.rows) * 5120 * 2, mtl.ResourceOptions.shared);
            const xf = View{ .buffer = m.gate_x.?, .offset = 0 };
            ops.encodeBf16ToF16(encoder, &p.ops, ws.view(.normalized), xf, rows * 5120);
            encoder.barrier();
            ops.encodeGateGemmF16F32(encoder, &p.ops, xf, tensor(w.gate), dest, .{ .rows = rows, .k = 5120 });
        } else if (gate_bf16_gemm) {
            ops.encodeGateGemmF32(encoder, &p.ops, ws.view(.normalized), tensor(w.gate), dest, .{ .rows = rows, .k = 5120 });
        }
        encoder.barrier();
    }
    // Default on for prompt rows; TF_DSV41_GATE_TREE=0 forces the shipped tf-dense-col back.
    const gate_tree = rows >= 64 and !Model.gateTreeOff();
    // TF_DSV41_GATE_WIDE: the wide gate, same order and logits, with eight rows by four columns a threadgroup.
    const gate_wide = gate_tree and Model.gateWideOn();
    // The wide gate -- eight rows and four columns a threadgroup with the tiles staged in threadgroup memory.
    if (!route_gemm or gate_scratch) if (gate_wide and (w.gate.dtype == .bf16 or w.gate.dtype == .f16)) {
        mhc.encodeGateTree8(encoder, &p.mhc, ws.view(.normalized), tensor(w.gate), logits_dest, .{ .rows = rows, .k = 5120, .n = experts, .x_stride = 5120, .bf16_w = @intFromBool(w.gate.dtype == .bf16) });
    } else if (!route_gemm or gate_scratch) if (gate_tree and (w.gate.dtype == .bf16 or w.gate.dtype == .f16)) {
        mhc.encodeGateTree4(encoder, &p.mhc, ws.view(.normalized), tensor(w.gate), logits_dest, .{ .rows = rows, .k = 5120, .n = experts, .x_stride = 5120, .bf16_w = @intFromBool(w.gate.dtype == .bf16) });
    } else if (!route_gemm or gate_scratch) if (!route_tensor and (w.gate.dtype == .bf16 or w.gate.dtype == .f16)) {
        mhc.encodeColDense(encoder, &p.mhc, ws.view(.normalized), tensor(w.gate), logits_dest, .{ .rows = rows, .k = 5120, .n = experts, .x_stride = 5120, .bf16_w = @intFromBool(w.gate.dtype == .bf16) });
    } else ops.encodeDense(encoder, &p.ops, ws.view(.normalized), tensor(w.gate), logits_dest, .{ .rows = rows, .k = 5120, .n = experts, .x_stride = 5120, .w_stride = 5120, .y_stride = experts, .w_dtype = if (w.gate.dtype == .bf16) .bf16 else .f16, .y_dtype = .f32 });
    encoder.barrier();
    // TF_DSV41_GATE_DUMP=<dir>: the gate stage's own input and the logits it produced.
    if (std.c.getenv("TF_DSV41_GATE_DUMP")) |dump_dir| {
        const want: u32 = if (std.c.getenv("TF_DSV41_GATE_DUMP_ROWS")) |v|
            (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 512)
        else
            512;
        const n = @min(rows, want);
        const x_elems: u32 = n * 5120;
        const l_elems: u32 = n * 384;
        if (m.gate_dump_x == null) {
            m.gate_dump_x = try m.device.buffer(@as(usize, want) * 5120 * 4, mtl.ResourceOptions.shared);
            m.gate_dump_l = try m.device.buffer(@as(usize, want) * 384 * 4, mtl.ResourceOptions.shared);
        }
        const stg_x = m.gate_dump_x.?;
        const stg_l = m.gate_dump_l.?;
        ops.encodeDump(encoder, &p.ops, ws.view(.normalized), .{ .buffer = stg_x, .offset = 0 }, x_elems, true);
        ops.encodeDump(encoder, &p.ops, ws.view(.router_logits), .{ .buffer = stg_l, .offset = 0 }, l_elems, false);
        try splitStage(m, enc, e, "gatedump", 0);
        encoder = enc.*;
        const x_path = try std.fmt.allocPrint(m.gpa, "{s}/L{d:0>2}-r{d}-x.bin", .{ dump_dir, layer_index, n });
        defer m.gpa.free(x_path);
        {
            var f = try std.Io.Dir.cwd().createFile(m.io, x_path, .{});
            defer f.close(m.io);
            try f.writeStreamingAll(m.io, stg_x.contents()[0 .. @as(usize, x_elems) * 4]);
        }
        const l_path = try std.fmt.allocPrint(m.gpa, "{s}/L{d:0>2}-r{d}-logits.bin", .{ dump_dir, layer_index, n });
        defer m.gpa.free(l_path);
        {
            var f = try std.Io.Dir.cwd().createFile(m.io, l_path, .{});
            defer f.close(m.io);
            try f.writeStreamingAll(m.io, stg_l.contents()[0 .. @as(usize, l_elems) * 4]);
        }
    }
    const skip_experts = std.c.getenv("TF_DSV41_SKIP_EXPERTS") != null;
    // TF_DSV41_GROUP_MIN_ROWS lowers the threshold below a prompt chunk's eight rows so windows can use it.
    const group_min: u32 = if (std.c.getenv("TF_DSV41_GROUP_MIN_ROWS")) |v|
        (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 8)
    else
        8;
    const grouped: ?expert_groups.Scratch = if (m.grouped_prefill and rows >= group_min and rows >= 2) try ws.grouping(rows, picks, experts) else null;
    // The fused gate+up call: one indexed dispatch over a table of 2E descriptors.
    const fuse = w.fused != null and rows < 8 and std.c.getenv("TF_DSV41_NO_GATE_UP_FUSION") == null;
    const fused_grouped: ?expert_groups.Scratch = if (fuse and grouped != null) try ws.groupingFused(rows, picks, experts) else null;
    // The merged MoE up dispatch needs the fused table's two shared descriptors and an ungrouped call.
    const moe_up_merged = fuse and fused_grouped == null and
        w.fused.?.length() / @sizeOf(exl3.ExpertDescriptor) == 2 * experts + 2 and
        std.c.getenv("TF_DSV41_NO_FUSE_MOE_UP") == null;
    // THE SHARED EXPERT'S SLOTS OUT OF THE FUSED INDEXED CALL: the routed-only call plus two dense shared calls.
    const shared_split = moe_up_merged and rows >= 2 and std.c.getenv("TF_DSV41_SHARED_DENSE") != null;
    // The routed SwiGLU writes the down projection's rotated input in the same pass.
    const swiglu_rot = std.c.getenv("TF_DSV41_NO_SWIGLU_ROT") == null and std.c.getenv("TF_DSV41_SKIP_SWIGLU") == null;
    // The shared SwiGLU writes the down projection's ROTATED input in the same pass.
    const shared_rot = moe_up_merged and w.shared_down != null and w.shared_down_ids != null and
        std.c.getenv("TF_DSV41_NO_FUSE_SHARED_ROT") == null;
    const mlp_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_MLP") != null) 2 else 1;
    // THE FUSED CALL'S UP HALF IS OFFSET BY rows * picks: six for the backbone, three for the drafter.
    const fuse_up_picks: u32 = if (std.c.getenv("TF_DSV41_FUSE_UP_SIX") != null) 6 else picks;
    const skip_shared = std.c.getenv("TF_DSV41_SKIP_SHARED") != null;
    // The gate+up call stops at its f32 sums and one pass does its rotate_out, the SwiGLUs and the down rotate_in.
    const fused_pass = try fusion.plan(ws, w, .{ .rows = rows, .picks = picks, .merged = moe_up_merged, .shared_split = shared_split, .swiglu_rot = swiglu_rot, .shared_rot = shared_rot, .skip_experts = skip_experts, .skip_shared = skip_shared, .mlp_passes = mlp_passes, .fuse_up_picks = fuse_up_picks });
    var logits_view = logits_dest;
    const logit_dtype: u32 = 0;
    if (gate_private) {
        logits_view = .{ .buffer = gate_pdest.?, .offset = 0 };
    }
    // The MoE's parts -- the gate, the route, the gate/up, the SwiGLU, the down and the shared expert.
    if (prof) |pf| {
        if (pf.blk_on) {
            pf.split("blk", "moe/gate", pf.block, enc);
            encoder = enc.*;
        }
    }
    route.encode(encoder, &p.route, logits_view, tensor(w.bias), ws.view(.route_ids), ws.view(.route_weights), .{ .ids = ws.view(.fused_ids), .sources = ws.view(.fused_sources) }, .{ .logit_dtype = logit_dtype, .rows = rows, .bias_dtype = switch (w.bias.dtype) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        else => .f16,
    }, .fused = if (fuse) experts else 0, .logit_stride = experts, .output_stride = picks, .experts = experts, .topk = picks, .moe_up = @intFromBool(moe_up_merged and !shared_split) });
    try splitStage(m, enc, e, "route", 0);
    encoder = enc.*;
    if (prof) |pf| {
        if (pf.blk_on) {
            pf.split("blk", "moe/route", pf.block, enc);
            encoder = enc.*;
        }
    }

    encoder.barrier();
    // TF_DSV41_ROUTE_DEBUG3: the GPU-side read-back (the buffers are not host-readable).
    if (std.c.getenv("TF_DSV41_ROUTE_DEBUG3") != null and !model.gate_dump_done) {
        model.gate_dump_done = true;
        const stg_gate = try m.device.buffer(40 * 4, mtl.ResourceOptions.shared);
        const stg_logits = try m.device.buffer(1200 * 4, mtl.ResourceOptions.shared);
        if (w.gate_bf16) |g| ops.encodeDump(encoder, &p.ops, .{ .buffer = g.buffer, .offset = g.offset }, .{ .buffer = stg_gate, .offset = 0 }, 40, true);
        // The last row's logits: the reply's first token comes from them, and row 0 alone cannot rule coverage out.
        const last_off: usize = @as(usize, rows - 1) * 384 * 4;
        ops.encodeDump(encoder, &p.ops, .{ .buffer = ws.view(.router_logits).buffer, .offset = ws.view(.router_logits).offset + last_off }, .{ .buffer = stg_logits, .offset = 0 }, 384, false);
        try splitStage(m, enc, e, "dump", 0);
        encoder = enc.*;
        const g40: [*]const f32 = @ptrCast(@alignCast(stg_gate.contents()));
        const l40: [*]const f32 = @ptrCast(@alignCast(stg_logits.contents()));
        std.log.info("dump gate[0..3] {d:.4} {d:.4} {d:.4} {d:.4}", .{ g40[0], g40[1], g40[2], g40[3] });
        var best: [3]f32 = .{ -1e30, -1e30, -1e30 };
        var best_id: [3]u32 = .{ 0, 0, 0 };
        for (0..384) |col| {
            const v = l40[col];
            if (v > best[0]) {
                best[2] = best[1];
                best_id[2] = best_id[1];
                best[1] = best[0];
                best_id[1] = best_id[0];
                best[0] = v;
                best_id[0] = @intCast(col);
            } else if (v > best[1]) {
                best[2] = best[1];
                best_id[2] = best_id[1];
                best[1] = v;
                best_id[1] = @intCast(col);
            } else if (v > best[2]) {
                best[2] = v;
                best_id[2] = @intCast(col);
            }
        }
        std.log.info("dump row0 top3: {d}@{d:.4} {d}@{d:.4} {d}@{d:.4}", .{ best_id[0], best[0], best_id[1], best[1], best_id[2], best[2] });
    }
    // The gate's first eight logits of row 0, read both ways.
    if (std.c.getenv("TF_DSV41_ROUTE_DEBUG2") != null and !model.gate_dump_done) {
        model.gate_dump_done = true;
        try splitStage(m, enc, e, "gatedump", 0);
        encoder = enc.*;
        const v = ws.view(.router_logits);
        const base = v.buffer.contents() + v.offset;
        const p32: [*]const f32 = @ptrCast(@alignCast(base));
        const p16: [*]const u16 = @ptrCast(@alignCast(base));
        var a32: [8]f32 = undefined;
        var a16: [8]f32 = undefined;
        for (0..8) |i| {
            a32[i] = p32[i];
            a16[i] = @bitCast(@as(u32, p16[i]) << 16);
        }
        std.log.info("gate logits row0 as f32: {d:.4} {d:.4} {d:.4} {d:.4}", .{ a32[0], a32[1], a32[2], a32[3] });
        std.log.info("gate logits row0 as bf16: {d:.4} {d:.4} {d:.4} {d:.4}", .{ a16[0], a16[1], a16[2], a16[3] });
    }
    // The last row's six router weights and six picks: the only router output never compared.
    if (std.c.getenv("TF_DSV41_ROUTE_DEBUG4") != null and !model.gate_dump_done2) {
        model.gate_dump_done2 = true;
        const stg_wt = try m.device.buffer(6 * 4, mtl.ResourceOptions.shared);
        const stg_id = try m.device.buffer(6 * 4, mtl.ResourceOptions.shared);
        const rw = ws.view(.route_weights);
        const ri = ws.view(.route_ids);
        const off_w: usize = @as(usize, rows - 1) * 6 * 4;
        ops.encodeDump(encoder, &p.ops, .{ .buffer = rw.buffer, .offset = rw.offset + off_w }, .{ .buffer = stg_wt, .offset = 0 }, 6, false);
        ops.encodeDump(encoder, &p.ops, .{ .buffer = ri.buffer, .offset = ri.offset + off_w }, .{ .buffer = stg_id, .offset = 0 }, 6, false);
        try splitStage(m, enc, e, "picks", 0);
        encoder = enc.*;
        const w6: [*]const f32 = @ptrCast(@alignCast(stg_wt.contents()));
        const id6: [*]const f32 = @ptrCast(@alignCast(stg_id.contents()));
        std.log.info("picks ids {d} {d} {d} {d} {d} {d}", .{ @as(u32, @bitCast(id6[0])), @as(u32, @bitCast(id6[1])), @as(u32, @bitCast(id6[2])), @as(u32, @bitCast(id6[3])), @as(u32, @bitCast(id6[4])), @as(u32, @bitCast(id6[5])) });
        std.log.info("picks weights {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} {d:.4}", .{ w6[0], w6[1], w6[2], w6[3], w6[4], w6[5] });
    }
    // (inert unless TF_DSV41_ROUTE_IDS=1): the last rows' route ids, read back from the GPU.
    const want_ids = std.c.getenv("TF_DSV41_ROUTE_IDS") != null and
        ((rows == 2 and !model.route_ids_done_pair) or (rows > 8 and !model.route_ids_done_prompt));
    if (want_ids) {
        if (rows == 2) model.route_ids_done_pair = true else model.route_ids_done_prompt = true;
        const nrows: usize = @min(rows, 2);
        const n: u32 = @intCast(nrows * picks);
        const stg_ids = try m.device.buffer(@as(usize, n) * 4, mtl.ResourceOptions.shared);
        const ri = ws.view(.route_ids);
        const off_i: usize = (@as(usize, rows) - nrows) * picks * 4;
        ops.encodeDump(encoder, &p.ops, .{ .buffer = ri.buffer, .offset = ri.offset + off_i }, .{ .buffer = stg_ids, .offset = 0 }, n, false);
        try splitStage(m, enc, e, "ids", 0);
        encoder = enc.*;
        const ip: [*]const u32 = @ptrCast(@alignCast(stg_ids.contents()));
        var distinct: u32 = 0;
        var buf: [256]u32 = undefined;
        for (0..n) |i| {
            const iu: usize = @intCast(i);
            buf[iu] = ip[i];
            var seen = false;
            for (0..iu) |j| {
                if (buf[j] == ip[i]) seen = true;
            }
            if (!seen) distinct += 1;
        }
        std.log.info("route ids: rows {d} slots {d} distinct {d}", .{ nrows, n, distinct });
        var line: [2048]u8 = undefined;
        var at: usize = 0;
        for (0..n) |i| {
            const s = std.fmt.bufPrint(line[at..], "{d} ", .{ip[i]}) catch break;
            at += s.len;
        }
        std.log.info("route ids: {s}", .{line[0..at]});
    }
    var routed_rows = exl3.Rows{ .count = @as(usize, rows) * picks, .x_stride = 5120, .y_stride = 2304, .input_rows = rows, .output_rows = @as(usize, rows) * picks, .tile_rows = Model.moeTileRows(), .routes = .{ .ids = convert(ws.view(.route_ids)), .repeat = picks } };
    if (grouped) |g| {
        try expert_groups.encode(encoder, &p.groups, convert(ws.view(.route_ids)), g, picks, 0);
        routed_rows.count = g.plan.padded;
        routed_rows.input_ids = g.view(.token_ids);
        routed_rows.output_ids = g.view(.slot_ids);
        routed_rows.routes = .{ .ids = g.view(.expert_ids) };
        routed_rows.grouped = true;
        // (inert unless TF_DSV41_ROUTE_HIST=1): the grouped plan's own geometry, read back from the GPU.
        if (std.c.getenv("TF_DSV41_ROUTE_HIST") != null and !model.plan_hist_done) {
            model.plan_hist_done = true;
            const stg_cnt = try m.device.buffer(384 * 4, mtl.ResourceOptions.shared);
            const stg_off = try m.device.buffer(384 * 4, mtl.ResourceOptions.shared);
            ops.encodeDump(encoder, &p.ops, .{ .buffer = g.view(.counts).buffer, .offset = g.view(.counts).offset }, .{ .buffer = stg_cnt, .offset = 0 }, 384, false);
            ops.encodeDump(encoder, &p.ops, .{ .buffer = g.view(.offsets).buffer, .offset = g.view(.offsets).offset }, .{ .buffer = stg_off, .offset = 0 }, 384, false);
            try splitStage(m, enc, e, "plan", 0);
            encoder = enc.*;
            const cp: [*]const u32 = @ptrCast(@alignCast(stg_cnt.contents()));
            const op: [*]const u32 = @ptrCast(@alignCast(stg_off.contents()));
            var live: u64 = 0;
            var tiles: u64 = 0;
            var used: u64 = 0;
            var nz: u32 = 0;
            var mx: u32 = 0;
            var hist: [10]u32 = [_]u32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            for (0..384) |ei| {
                const c = cp[ei];
                live += @as(u64, c);
                if (c > 0) {
                    nz += 1;
                    if (c > mx) mx = c;
                    const t = (c + 63) / 64;
                    tiles += @as(u64, t);
                    used += @as(u64, t) * 64;
                    hist[@min(t, 9)] += 1;
                }
            }
            std.log.info("plan: rows {d} live {d} experts_used {d} max_count {d} tiles_used {d} used_rows {d} plan_padded {d}", .{ rows * picks, live, nz, mx, tiles, used, g.plan.padded });
            std.log.info("plan tiles a expert: 1:{d} 2:{d} 3:{d} 4:{d} 5:{d} 6:{d} 7:{d} 8:{d} 9+:{d}", .{ hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7], hist[8], hist[9] });
            var line: [4096]u8 = undefined;
            var at: usize = 0;
            for (0..384) |ei| {
                const s = std.fmt.bufPrint(line[at..], "{d},", .{cp[ei]}) catch break;
                at += s.len;
            }
            std.log.info("plan counts: {s}", .{line[0..at]});
            var line2: [4096]u8 = undefined;
            var at2: usize = 0;
            for (0..384) |ei| {
                const s = std.fmt.bufPrint(line2[at2..], "{d},", .{op[ei]}) catch break;
                at2 += s.len;
            }
            std.log.info("plan offsets: {s}", .{line2[0..at2]});
        }
    }
    // The fused table is two blocks of six picks a token, so its sort reads a slot's block for the token.
    if (fused_grouped) |g| try expert_groups.encode(encoder, &p.groups, convert(ws.view(.fused_ids)), g, picks, picks * rows);
    if (fuse) {
        var both = routed_rows;
        if (fused_grouped) |g| {
            both.count = g.plan.padded;
            both.input_ids = g.view(.token_ids);
            both.output_ids = g.view(.slot_ids);
            both.routes = .{ .ids = g.view(.expert_ids) };
            both.grouped = true;
        } else {
            const r = @as(usize, rows);
            const routed_count = r * 2 * picks;
            const merge_up = moe_up_merged;
            both.count = routed_count + if (merge_up and !shared_split) 2 * r else 0;
            both.input_rows = r;
            both.input_ids = convert(ws.view(.fused_sources));
            both.routes = .{ .ids = convert(ws.view(.fused_ids)), .repeat = 1 };
            both.output_rows = both.count;
        }
        if (fused_grouped != null) both.output_rows = @as(usize, rows) * 2 * picks;
        const fused_passes: u32 = Model.expertPasses();
        const fused_experts: u32 = 2 * experts + 2 * @as(u32, @intFromBool(moe_up_merged));
        both.no_epilogue = fused_pass.gateup;
        for (0..fused_passes) |_| if (!skip_experts) try exl3.encodeIndexed(&p.projection, encoder, .{ .descriptors = .{ .buffer = w.fused.? }, .count = fused_experts, .k = 5120, .n = 2304 }, convert(ws.view(.normalized)), convert(ws.view(.gate)), ws.projection(), both);
        try splitStage(m, enc, e, "gateup", 0);
        encoder = enc.*;
        if (prof) |pf| {
            if (pf.blk_on) {
                pf.split("blk", "moe/gateup", pf.block, enc);
                encoder = enc.*;
            }
        }

        encoder.barrier();
    } else for (0..2) |i| {
        if (!skip_experts) try exl3.encodeIndexed(&p.projection, encoder, .{ .descriptors = .{ .buffer = w.descriptors[i] }, .count = experts, .k = 5120, .n = 2304 }, convert(ws.view(.normalized)), convert(ws.view(if (i == 0) .gate else .up)), ws.projection(), routed_rows);
        encoder.barrier();
    }
    const up_view = if (fuse) View{ .buffer = ws.view(.gate).buffer, .offset = ws.view(.gate).offset + @as(usize, rows) * fuse_up_picks * 2304 * 2 } else ws.view(.up);
    if (fused_pass.gateup) {
        const gate_up_rows = @as(usize, rows) * (2 * picks + 2);
        const sums = exl3.sumsView(ws.projection(), gate_up_rows, 5120);
        exl3.encodeGateUpSwiGLU(&p.projection, encoder, sums, convert(ws.view(.fused_ids)), convert(ws.view(.route_weights)), convert(ws.view(.route_ids)), .{ .buffer = w.fused.? }, .{ .buffer = if (fused_pass.shared_down) w.down_shared.? else w.descriptors[2] }, .{ .buffer = if (fused_pass.shared_down) w.down_shared.? else w.shared_down.?, .offset = if (fused_pass.shared_down) @as(usize, experts) * @sizeOf(exl3.ExpertDescriptor) else 0 }, ws.projection(), .{ .rows = @as(usize, rows) * picks, .shared_rows = rows, .k = 2304, .experts = experts, .fused_experts = 2 * experts + 2, .shared_dst = fused_pass.shared_row, .shared_id = if (fused_pass.shared_down) experts else null });
    } else if (swiglu_rot)
        exl3.encodeSwiGLUPreRotate(&p.projection, encoder, convert(ws.view(.gate)), convert(up_view), convert(ws.view(.route_weights)), convert(ws.view(.route_ids)), if (grouped) |g| g.view(.destination) else null, convert(.{ .buffer = w.descriptors[2] }), ws.projection(), .{ .rows = @as(usize, rows) * picks, .k = 2304, .input_stride = 2304, .output_stride = 2304, .experts = experts })
    else for (0..mlp_passes) |_| if (std.c.getenv("TF_DSV41_SKIP_SWIGLU") == null) ops.encodeSwiGLU(encoder, &p.ops, ws.view(.gate), up_view, ws.view(.route_weights), ws.view(.activation), .{ .rows = rows * picks, .width = 2304, .input_stride = 2304, .output_stride = 2304, .weighted = true });
    try splitStage(m, enc, e, "swiglu", 0);
    encoder = enc.*;
    if (prof) |pf| {
        if (pf.blk_on) {
            pf.split("blk", "moe/swiglu", pf.block, enc);
            encoder = enc.*;
        }
    }

    encoder.barrier();
    routed_rows.x_stride = 2304;
    routed_rows.y_stride = 5120;
    routed_rows.routes.?.repeat = 1;
    routed_rows.input_rows = @as(usize, rows) * picks; // The block's own pick count, not the backbone's six
    if (grouped) |g| routed_rows.input_ids = g.view(.slot_ids);
    const down_passes: u32 = Model.expertPasses();
    routed_rows.prerotated = swiglu_rot;
    // The shared expert's down rows follow the routed rows: one more table entry, ids written by the fused pass.
    if (fused_pass.shared_down) routed_rows.count += rows;
    routed_rows.no_epilogue = fused_pass.down_combine;
    const down_input: exl3.View = if (swiglu_rot) ws.projection() else convert(ws.view(.activation));
    for (0..down_passes) |_| if (!skip_experts) try exl3.encodeIndexed(&p.projection, encoder, .{ .descriptors = .{ .buffer = if (fused_pass.shared_down) w.down_shared.? else w.descriptors[2] }, .count = experts + @intFromBool(fused_pass.shared_down), .k = 2304, .n = 5120 }, down_input, convert(ws.view(.routed)), ws.projection(), routed_rows);
    try splitStage(m, enc, e, "down", 0);
    encoder = enc.*;
    if (prof) |pf| {
        if (pf.blk_on) {
            pf.split("blk", "moe/down", pf.block, enc);
            encoder = enc.*;
        }
    }

    encoder.barrier();
    if (fused_pass.shared_down) {
        // The shared expert's down ran inside the routed down call.
    } else if (!skip_shared and fused_pass.gateup) {
        // The fused pass wrote the shared expert's rotated input past the routed rows' scratch.
        const shared_scratch = fusion.sharedScratch(ws, fused_pass);
        try exl3.encode(&p.projection, encoder, w.shared[2], shared_scratch, convert(ws.view(.shared_down)), shared_scratch, .{ .count = rows, .x_stride = 2304, .y_stride = 5120, .prerotated = true });
    } else if (!skip_shared) {
        // The shared expert's gate and up in one indexed dispatch.
        const shared_fuse = rows <= 8 and w.shared_fused != null and std.c.getenv("TF_DSV41_NO_FUSE_SHARED") == null;
        for (0..mlp_passes) |_| {
            if (moe_up_merged) {
                // The shared gate/up already landed: [12r, 13r) is its gate, [13r, 14r) its up, after routed [0, 12r).
                const gv = ws.view(.gate);
                const sg = View{ .buffer = gv.buffer, .offset = gv.offset + @as(usize, rows) * 2 * picks * 2304 * 2 };
                const su = View{ .buffer = gv.buffer, .offset = sg.offset + @as(usize, rows) * 2304 * 2 };
                if (shared_split) {
                    if (!model.shared_split_logged) {
                        model.shared_split_logged = true;
                        std.log.info("shared split: dense multi-row shared gate/up, rows {d}", .{rows});
                    }
                    try m.project(encoder, w.shared[0], ws.view(.normalized), sg, rows, 5120, 2304, .bf16);
                    try m.project(encoder, w.shared[1], ws.view(.normalized), su, rows, 5120, 2304, .bf16);
                    encoder.barrier();
                }
                if (shared_rot) {
                    exl3.encodeSwiGLUPreRotate(&p.projection, encoder, convert(sg), convert(su), convert(ws.view(.route_weights)), convert(View{ .buffer = w.shared_down_ids.?, .offset = 0 }), null, convert(View{ .buffer = w.shared_down.?, .offset = 0 }), ws.projection(), .{ .rows = rows, .k = 2304, .input_stride = 2304, .output_stride = 2304, .weighted = false, .experts = 1 });
                } else ops.encodeSwiGLU(encoder, &p.ops, sg, su, ws.view(.route_weights), ws.view(.shared_activation), .{ .rows = rows, .width = 2304, .input_stride = 2304, .output_stride = 2304 });
            } else if (shared_fuse) {
                try exl3.encodeIndexed(&p.projection, encoder, .{ .descriptors = .{ .buffer = w.shared_fused.?, .offset = 0 }, .count = 2, .k = 5120, .n = 2304 }, convert(ws.view(.normalized)), convert(ws.view(.up)), ws.projection(), .{
                    .count = @as(usize, rows) * 2,
                    .x_stride = 5120,
                    .y_stride = 2304,
                    .output = .bf16,
                    .input_ids = convert(ws.sharedMapInputs(rows)),
                    .input_rows = rows,
                    .output_rows = @as(usize, rows) * 2,
                    .routes = .{ .ids = convert(ws.sharedMapRoutes(rows)) },
                });
                const fused = ws.view(.up);
                const up_v = View{ .buffer = fused.buffer, .offset = fused.offset + @as(usize, rows) * 2304 * 2 };
                ops.encodeSwiGLU(encoder, &p.ops, fused, up_v, ws.view(.route_weights), ws.view(.shared_activation), .{ .rows = rows, .width = 2304, .input_stride = 2304, .output_stride = 2304 });
            } else {
                try m.project(encoder, w.shared[0], ws.view(.normalized), ws.view(.shared_gate), rows, 5120, 2304, .bf16);
                try m.project(encoder, w.shared[1], ws.view(.normalized), ws.view(.shared_up), rows, 5120, 2304, .bf16);
                ops.encodeSwiGLU(encoder, &p.ops, ws.view(.shared_gate), ws.view(.shared_up), ws.view(.route_weights), ws.view(.shared_activation), .{ .rows = rows, .width = 2304, .input_stride = 2304, .output_stride = 2304 });
            }
            encoder.barrier();
            if (shared_rot) {
                try exl3.encode(&p.projection, encoder, w.shared[2], ws.projection(), convert(ws.view(.shared_down)), ws.projection(), .{ .count = rows, .x_stride = 2304, .y_stride = 5120, .prerotated = true });
            } else try m.project(encoder, w.shared[2], ws.view(.shared_activation), ws.view(.shared_down), rows, 2304, 5120, .bf16);
        }
    }
    try splitStage(m, enc, e, "shared", 0);
    encoder = enc.*;
    if (prof) |pf| {
        if (pf.blk_on) {
            pf.split("blk", "moe/shared", pf.block, enc);
            encoder = enc.*;
        }
    }

    const routed_view = ws.view(.routed);
    const shared_down_view = if (fused_pass.shared_down) View{ .buffer = routed_view.buffer, .offset = routed_view.offset + @as(usize, rows) * picks * 5120 * 2 } else ws.view(.shared_down);
    if (fused_pass.down_combine) {
        const sums = exl3.sumsView(ws.projection(), routed_rows.count, 2304);
        exl3.encodeDownCombine(&p.projection, encoder, sums, convert(ws.view(.route_ids)), .{ .buffer = w.down_shared.? }, convert(ws.view(.branch)), .{ .tokens = rows, .picks = picks, .width = 5120, .experts = experts, .y_stride = 5120, .shared_row = @as(usize, rows) * picks });
    } else if (std.c.getenv("TF_DSV41_SKIP_COMBINE") == null) ops.encodeCombine(encoder, &p.ops, ws.view(.routed), shared_down_view, ws.view(.route_ids), ws.view(.branch), .{ .rows = rows, .width = 5120, .routed_row_stride = picks * 5120, .slot_stride = 5120, .shared_stride = 5120, .y_stride = 5120, .id_stride = picks });
    // A second identical pass leaves the value unchanged and isolates the kernel in place.
    if (std.c.getenv("TF_DSV41_DOUBLE_COMBINE") != null) ops.encodeCombine(encoder, &p.ops, ws.view(.routed), shared_down_view, ws.view(.route_ids), ws.view(.branch), .{ .rows = rows, .width = 5120, .routed_row_stride = picks * 5120, .slot_stride = 5120, .shared_stride = 5120, .y_stride = 5120, .id_stride = picks });
    encoder.barrier();
    if (prof) |pf| {
        if (pf.blk_on) {
            pf.split("blk", "moe/combine", pf.block, enc);
            encoder = enc.*;
        }
    }
}
