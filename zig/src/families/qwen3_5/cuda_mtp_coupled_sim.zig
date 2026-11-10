//! Host CUDA driver simulator for Qwen MTP coupled-draw tests: memory, copies, causal layer state, and sample.cu draws.
const std = @import("std");
const cuda = @import("cuda");
const abi = cuda.abi;
const lanes = @import("lanes");
const geometry = @import("cuda_geometry.zig");
const mtp = @import("cuda_mtp.zig");
const batched = @import("cuda_mtp_batch.zig");

pub const vocab = 64;
pub const hidden = 64;
pub const q_heads = 2;
pub const kv_heads = 1;
pub const aw = q_heads * geometry.head_dim;
pub const kw = kv_heads * geometry.head_dim;
pub const init_state: u64 = 0x517C_C1B7_2722_0A95;

pub const fn_sums = 1;
pub const fn_gemv = 2;
pub const fn_group = 3;
pub const fn_prefill = 4;
pub const fn_verify = 5;
pub const fn_embed = 10;
pub const fn_norm = 11;
pub const fn_argmax = 12;
pub const fn_draw_rows = 13;
pub const fn_logprob_rows = 14;
pub const fn_norm_rope = 15;
pub const fn_attn = 16;
pub const fn_attn_tree = 17;
pub const fn_attn_rows = 18;
pub const fn_gather_taps = 19;
pub const fn_gather_rows = 20;
pub const fn_ring_put = 21;
pub const fn_noop = 22;

pub const tag_fc_e: u64 = 0xA001;
pub const tag_fc_h: u64 = 0xA002;
pub const tag_q: u64 = 0xA003;
pub const tag_k: u64 = 0xA004;
pub const tag_v: u64 = 0xA005;
pub const tag_o: u64 = 0xA006;
pub const tag_head: u64 = 0xA007;

pub const w_final_norm: u64 = 0xB001;
pub const w_norm_e: u64 = 0xB002;
pub const w_norm_h: u64 = 0xB003;
pub const w_input_norm: u64 = 0xB004;
pub const w_post_norm: u64 = 0xB005;
pub const w_q_norm: u64 = 0xB006;
pub const w_k_norm: u64 = 0xB007;
pub const w_norm: u64 = 0xB008;

pub fn mix(x_: u64) u64 {
    var x = x_ +% 0x9E37_79B9_7F4A_7C15;
    x = (x ^ (x >> 30)) *% 0xBF58_476D_1CE4_E5B9;
    x = (x ^ (x >> 27)) *% 0x94D0_49BB_1331_11EB;
    return x ^ (x >> 31);
}

pub fn stepState(prev: u64, tok: u32, slot_1based: u64) u64 {
    return mix(prev ^ (@as(u64, tok) *% 0xD6E8_FEB8_6659_FD93) ^ (slot_1based *% 0xA076_1D64_78BD_642F));
}

pub fn bf16Bits(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast(((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF);
}

pub fn bf16ToF32(w: u16) f32 {
    return @bitCast(@as(u32, w) << 16);
}

/// Smooth 64-token BF16 logit profile keyed by `state` so greedy argmax rarely matches sampled Gumbel draws.
pub fn writeLogits(state: u64, out: []u16) void {
    for (out, 0..) |*dst, i| {
        const h = mix(state ^ (@as(u64, i) *% 0xBF58_476D_1CE4_E5B9));
        const val = @as(f32, @floatFromInt(h & 0x3FF)) * (1.0 / 256.0) - 2.0;
        dst.* = bf16Bits(val);
    }
}

pub fn samplingFromRule(rl: cuda.sample.Rule) lanes.Sampling {
    const min_p: f64 = if (rl.min_log <= -std.math.inf(f32)) 0.0 else @exp(@as(f64, rl.min_log));
    const s: lanes.Sampling = .{
        .seed = rl.seed,
        .temperature = 1.0 / @as(f64, rl.inv_t),
        .top_p = @as(f64, rl.top_p),
        .min_p = min_p,
        .top_k = rl.top_k,
    };
    const check = cuda.sample.rule(s);
    std.debug.assert(check.seed == rl.seed and check.inv_t == rl.inv_t and check.top_p == rl.top_p);
    std.debug.assert(check.min_log == rl.min_log and check.top_k == rl.top_k);
    return s;
}

pub fn drawState(gpa: std.mem.Allocator, state: u64, sampling: ?lanes.Sampling, position: u64) !u32 {
    var words: [vocab]u16 = undefined;
    writeLogits(state, &words);
    var vals: [vocab]f32 = undefined;
    for (words, &vals) |w, *v| v.* = bf16ToF32(w);
    if (cuda.sample.ruleFor(sampling)) |rl| {
        return lanes.gpu_full.draw(gpa, &vals, samplingFromRule(rl), @truncate(position), null);
    }
    return lanes.gpu_rule.argmax(&vals);
}

pub const Sim = struct {
    pub var head_bias: u64 = 0;
    pub var rules_addr: u64 = 0;
    pub var rules_end: u64 = 0;
    pub var argmax_launches: usize = 0;
    pub var draw_rows_launches: usize = 0;
    pub var draw_rows_total: usize = 0;
    pub var logprob_launches: usize = 0;
    pub var rule_uploads: usize = 0;
    pub var ring_fault: bool = false;
    pub var next_ev: std.atomic.Value(usize) = .init(1);
    pub var live_allocs: std.atomic.Value(usize) = .init(0);

    pub fn resetCounters() void {
        argmax_launches = 0;
        draw_rows_launches = 0;
        draw_rows_total = 0;
        logprob_launches = 0;
        rule_uploads = 0;
        ring_fault = false;
    }

    fn rawAlloc(len: usize) ?usize {
        const total = len + 256 + @sizeOf(usize);
        const raw = std.c.malloc(total) orelse return null;
        const base = @intFromPtr(raw);
        const aligned = std.mem.alignForward(usize, base + @sizeOf(usize), 256);
        @as(*usize, @ptrFromInt(aligned - @sizeOf(usize))).* = base;
        _ = live_allocs.fetchAdd(1, .monotonic);
        return aligned;
    }

    fn rawFree(addr: usize) void {
        if (addr == 0) return;
        const base = @as(*const usize, @ptrFromInt(addr - @sizeOf(usize))).*;
        std.c.free(@ptrFromInt(base));
        _ = live_allocs.fetchSub(1, .monotonic);
    }

    pub fn memAlloc(out: *abi.DevicePtr, len: usize) callconv(.c) abi.Result {
        const p = rawAlloc(@max(len, 16)) orelse return 2;
        @memset(@as([*]u8, @ptrFromInt(p))[0..@max(len, 16)], 0);
        out.* = @intCast(p);
        return 0;
    }

    pub fn memFree(p: abi.DevicePtr) callconv(.c) abi.Result {
        rawFree(@intCast(p));
        return 0;
    }

    pub fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        const p = rawAlloc(@max(len, 16)) orelse return 2;
        @memset(@as([*]u8, @ptrFromInt(p))[0..@max(len, 16)], 0);
        out.* = @ptrFromInt(p);
        return 0;
    }

    pub fn hostFree(p: ?*anyopaque) callconv(.c) abi.Result {
        if (p) |ptr| rawFree(@intFromPtr(ptr));
        return 0;
    }

    pub fn htod(dst: abi.DevicePtr, src: ?*const anyopaque, len: usize) callconv(.c) abi.Result {
        if (len > 0) @memcpy(@as([*]u8, @ptrFromInt(dst))[0..len], @as([*]const u8, @ptrCast(src.?))[0..len]);
        return 0;
    }

    pub fn htodAsync(dst: abi.DevicePtr, src: ?*const anyopaque, len: usize, _: abi.Stream) callconv(.c) abi.Result {
        if (rules_addr != 0 and dst >= rules_addr and dst < rules_end) rule_uploads += 1;
        return htod(dst, src, len);
    }

    pub fn dtohAsync(dst: ?*anyopaque, src: abi.DevicePtr, len: usize, _: abi.Stream) callconv(.c) abi.Result {
        if (len > 0) @memcpy(@as([*]u8, @ptrCast(dst.?))[0..len], @as([*]const u8, @ptrFromInt(src))[0..len]);
        return 0;
    }

    pub fn dtod(dst: abi.DevicePtr, src: abi.DevicePtr, len: usize) callconv(.c) abi.Result {
        if (len > 0 and dst != src) {
            const d_slice = @as([*]u8, @ptrFromInt(dst))[0..len];
            const s_slice = @as([*]const u8, @ptrFromInt(src))[0..len];
            std.mem.copyForwards(u8, d_slice, s_slice);
        }
        return 0;
    }

    pub fn dtodAsync(dst: abi.DevicePtr, src: abi.DevicePtr, len: usize, _: abi.Stream) callconv(.c) abi.Result {
        return dtod(dst, src, len);
    }

    pub fn memset8(dst: abi.DevicePtr, val: u8, len: usize) callconv(.c) abi.Result {
        if (len > 0) @memset(@as([*]u8, @ptrFromInt(dst))[0..len], val);
        return 0;
    }

    pub fn memset8Async(dst: abi.DevicePtr, val: u8, len: usize, _: abi.Stream) callconv(.c) abi.Result {
        return memset8(dst, val, len);
    }

    pub fn syncStream(_: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }

    pub fn syncCtx() callconv(.c) abi.Result {
        return 0;
    }

    pub fn evCreate(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        out.* = @ptrFromInt(next_ev.fetchAdd(1, .monotonic));
        return 0;
    }

    pub fn evRecord(_: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }

    pub fn evSync(_: abi.Event) callconv(.c) abi.Result {
        return 0;
    }

    pub fn evDestroy(_: abi.Event) callconv(.c) abi.Result {
        return 0;
    }

    pub fn cell(addr: u64, stride_bytes: usize, row: usize) *[3]u64 {
        return @ptrFromInt(addr + row * stride_bytes);
    }

    fn runProj(tag: u64, x: u64, out: u64, rows: usize) void {
        for (0..rows) |m| switch (tag) {
            tag_fc_e, tag_fc_h => cell(out, hidden * 2, m)[0] = cell(x, hidden * 2, m)[0],
            tag_q => cell(out, 2 * aw * 2, m).* = cell(x, hidden * 2, m).*,
            tag_k, tag_v => cell(out, kw * 2, m).* = cell(x, hidden * 2, m).*,
            tag_o => cell(out, hidden * 2, m)[0] = cell(x, aw * 2, m)[0],
            tag_head => {
                const st = cell(x, hidden * 2, m)[0] ^ head_bias;
                writeLogits(st, @as([*]u16, @ptrFromInt(out + m * vocab * 2))[0..vocab]);
            },
            else => unreachable,
        };
    }

    fn arg(comptime T: type, params: ?[*]?*anyopaque, idx: usize) T {
        return @as(*const T, @ptrCast(@alignCast(params.?[idx].?))).*;
    }

    fn checkPrev(prev: [3]u64, cur: [3]u64) void {
        if (prev[0] != 0 and (prev[0] + 1 != cur[0] or prev[1] != cur[2])) ring_fault = true;
    }

    fn dispatch(f: abi.Function, gx: c_uint, params: ?[*]?*anyopaque) abi.Result {
        switch (@intFromPtr(f)) {
            fn_sums, fn_noop => return 0,
            fn_gemv => {
                const p = arg(cuda.qlinear.GemvPart, params, 2);
                runProj(p.w, arg(u64, params, 0), p.out, @intCast(arg(c_int, params, 3)));
                return 0;
            },
            fn_group, fn_verify => {
                const x = arg(u64, params, 0);
                const parts = arg(cuda.qlinear.Parts, params, 2);
                const rows: usize = @intCast(arg(c_int, params, 3));
                for (parts.p[0..@intCast(parts.count)]) |part| runProj(part.w, x, part.out, rows);
                return 0;
            },
            fn_prefill => {
                runProj(arg(u64, params, 1), arg(u64, params, 0), arg(u64, params, 4), @intCast(arg(c_int, params, 5)));
                return 0;
            },
            fn_embed => {
                const id_slice = @as([*]const u32, @ptrFromInt(arg(u64, params, 0)))[0..gx];
                const out = arg(u64, params, 4);
                for (id_slice, 0..) |tok, m| cell(out, hidden * 2, m).* = .{ tok, 0, 0 };
                return 0;
            },
            fn_norm => {
                const h = arg(u64, params, 0);
                const r = arg(u64, params, 1);
                const w = arg(u64, params, 2);
                const out = arg(u64, params, 3);
                for (0..gx) |m| switch (w) {
                    w_final_norm, w_norm_e, w_norm_h, w_norm => cell(out, hidden * 2, m).* = cell(h, hidden * 2, m).*,
                    w_input_norm => {
                        const v = [3]u64{ cell(r, hidden * 2, m)[0], cell(h, hidden * 2, m)[0], 0 };
                        cell(h, hidden * 2, m).* = v;
                        cell(out, hidden * 2, m).* = v;
                    },
                    w_post_norm => {
                        const v = [3]u64{ cell(r, hidden * 2, m)[0], 0, 0 };
                        cell(h, hidden * 2, m).* = v;
                        cell(out, hidden * 2, m).* = v;
                    },
                    else => unreachable,
                };
                return 0;
            },
            fn_norm_rope => {
                const src = arg(u64, params, 0);
                const w = arg(u64, params, 1);
                const out = arg(u64, params, 2);
                const heads: usize = @intCast(arg(c_int, params, 3));
                const stride: usize = @intCast(arg(c_int, params, 4));
                const pos0 = arg(c_longlong, params, 6);
                const step = arg(c_int, params, 7);
                const rows = gx / @as(c_uint, @intCast(heads));
                for (0..rows) |m| {
                    const c = cell(src, stride * 2, m).*;
                    if (w == w_k_norm) {
                        const slot: u64 = @intCast(pos0 + @as(c_longlong, @intCast(m)) * step);
                        cell(out, heads * geometry.head_dim * 2, m).* = .{ slot + 1, stepState(c[0], @truncate(c[1]), slot + 2), c[0] };
                    } else cell(out, heads * geometry.head_dim * 2, m).* = c;
                }
                return 0;
            },
            fn_ring_put => {
                const kb = arg(u64, params, 0);
                const vb = arg(u64, params, 1);
                const rk = arg(u64, params, 2);
                const rv = arg(u64, params, 3);
                const row_bytes = @as(usize, @intCast(arg(c_int, params, 4))) * 2;
                const pos0 = arg(c_longlong, params, 5);
                const cap = arg(c_int, params, 6);
                for (0..gx) |i| {
                    const slot: usize = @intCast(@mod(pos0 + @as(c_longlong, @intCast(i)), cap));
                    _ = dtod(rk + slot * row_bytes, kb + i * row_bytes, row_bytes);
                    _ = dtod(rv + slot * row_bytes, vb + i * row_bytes, row_bytes);
                }
                return 0;
            },
            fn_attn => {
                const kb = arg(u64, params, 3);
                const rk = arg(u64, params, 5);
                const cap = arg(c_int, params, 9);
                const begin = arg(c_longlong, params, 10);
                const pos0 = arg(c_longlong, params, 11);
                const att = arg(u64, params, 12);
                for (0..gx) |m| {
                    const cur = cell(kb, kw * 2, m).*;
                    if (m > 0) checkPrev(cell(kb, kw * 2, m - 1).*, cur) else if (pos0 > begin) checkPrev(cell(rk, kw * 2, @intCast(@mod(pos0 - 1, cap))).*, cur);
                    cell(att, aw * 2, m).* = .{ cur[1], 0, 0 };
                }
                return 0;
            },
            fn_attn_rows => {
                const kb = arg(u64, params, 3);
                const tab = @as([*]const batched.RingRow, @ptrFromInt(arg(u64, params, 5)))[0..gx];
                const att = arg(u64, params, 8);
                for (0..gx) |m| {
                    const cur = cell(kb, kw * 2, m).*;
                    const tr = tab[m];
                    if (m > @as(usize, @intCast(tr.seg))) checkPrev(cell(kb, kw * 2, m - 1).*, cur) else if (tr.end > tr.begin) checkPrev(cell(tr.rk, kw * 2, @intCast(@mod(tr.end - 1, tr.w))).*, cur);
                    cell(att, aw * 2, m).* = .{ cur[1], 0, 0 };
                }
                return 0;
            },
            fn_attn_tree => {
                const kb = arg(u64, params, 3);
                const lk = arg(u64, params, 5);
                const anc_ptr = arg(u64, params, 7);
                const anc_stride: usize = @intCast(arg(c_int, params, 8));
                const rk = arg(u64, params, 9);
                const cap = arg(c_int, params, 13);
                const begin = arg(c_longlong, params, 14);
                const end = arg(c_longlong, params, 15);
                const att = arg(u64, params, 16);
                for (0..gx) |m| {
                    const cur = cell(kb, kw * 2, m).*;
                    const anc = @as([*]const i32, @ptrFromInt(anc_ptr + m * anc_stride * 4));
                    const n_anc: usize = @intCast(anc[0]);
                    if (n_anc > 0) checkPrev(cell(lk, kw * 2, @intCast(anc[n_anc])).*, cur) else if (end > begin) checkPrev(cell(rk, kw * 2, @intCast(@mod(end - 1, cap))).*, cur);
                    cell(att, aw * 2, m).* = .{ cur[1], 0, 0 };
                }
                return 0;
            },
            fn_gather_taps => {
                const ring = arg(u64, params, 0);
                const cap = arg(c_int, params, 1);
                const start = arg(c_longlong, params, 2);
                const width: usize = @intCast(arg(c_int, params, 3));
                const out = arg(u64, params, 4);
                for (0..gx) |i| _ = dtod(out + i * width * 2, ring + @as(usize, @intCast(@mod(start + @as(c_longlong, @intCast(i)), cap))) * width * 2, width * 2);
                return 0;
            },
            fn_gather_rows => {
                const ring = arg(u64, params, 0);
                const cap = arg(c_int, params, 1);
                const start = arg(c_longlong, params, 2);
                const tr = arg(mtp.TapRows, params, 3);
                const width: usize = @intCast(arg(c_int, params, 4));
                const out = arg(u64, params, 5);
                for (0..gx) |i| _ = dtod(out + i * width * 2, ring + @as(usize, @intCast(@mod(start + tr.rows[i], cap))) * width * 2, width * 2);
                return 0;
            },
            fn_argmax => {
                argmax_launches += 1;
                const logits_ptr = arg(u64, params, 0);
                const v: usize = @intCast(arg(c_int, params, 1));
                const out = @as([*]u32, @ptrFromInt(arg(u64, params, 2)))[0..gx];
                for (out, 0..) |*dst, r| {
                    const words = @as([*]const u16, @ptrFromInt(logits_ptr + r * v * 2))[0..v];
                    var vals: [vocab]f32 = undefined;
                    for (words, vals[0..v]) |w, *fv| fv.* = bf16ToF32(w);
                    dst.* = lanes.gpu_rule.argmax(vals[0..v]);
                }
                return 0;
            },
            fn_draw_rows => {
                draw_rows_launches += 1;
                draw_rows_total += gx;
                const logits_ptr = arg(u64, params, 0);
                const v: usize = @intCast(arg(u32, params, 1));
                const rules = @as([*]const cuda.sample.Rule, @ptrFromInt(arg(u64, params, 2)))[0..gx];
                const meta = @as([*]const c_int, @ptrFromInt(arg(u64, params, 3)))[0..gx];
                const out = @as([*]u32, @ptrFromInt(arg(u64, params, 4)))[0..gx];
                for (out, rules, meta, 0..) |*dst, rl, m_val, r| {
                    const words = @as([*]const u16, @ptrFromInt(logits_ptr + r * v * 2))[0..v];
                    var vals: [vocab]f32 = undefined;
                    for (words, vals[0..v]) |w, *fv| fv.* = bf16ToF32(w);
                    const pos_u32: u32 = @as(u32, @bitCast(m_val)) +% 1;
                    dst.* = lanes.gpu_full.draw(std.heap.page_allocator, vals[0..v], samplingFromRule(rl), pos_u32, null) catch return 1;
                }
                return 0;
            },
            fn_logprob_rows => {
                logprob_launches += 1;
                const logits_ptr = arg(u64, params, 0);
                const v: usize = @intCast(arg(u32, params, 1));
                const k: usize = @intCast(arg(u32, params, 3));
                const out_ptr = arg(u64, params, 5);
                for (0..gx) |r| {
                    const words = @as([*]const u16, @ptrFromInt(logits_ptr + r * v * 2))[0..v];
                    var cands: [vocab]lanes.gpu_rule.Cand = undefined;
                    for (words, 0..) |w, i| cands[i] = .{ .v = bf16ToF32(w), .i = @intCast(i) };
                    std.mem.sort(lanes.gpu_rule.Cand, cands[0..v], {}, lanes.gpu_rule.better);
                    const dst = @as([*]u32, @ptrFromInt(out_ptr + r * mtp.lp_words * 4))[0..mtp.lp_words];
                    @memset(dst, 0xFFFF_FFFF);
                    dst[0] = 0;
                    dst[1] = 0;
                    for (0..@min(k, v)) |j| {
                        dst[2 + j] = cands[j].i;
                        dst[2 + 20 + j] = @bitCast(cands[j].v);
                    }
                }
                return 0;
            },
            else => return 1,
        }
    }

    pub fn launch(f: abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        return dispatch(f, gx, params);
    }

    pub fn launchEx(c: *const abi.LaunchConfig, f: abi.Function, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        return dispatch(f, c.grid_x, params);
    }
};
