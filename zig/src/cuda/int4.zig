//! GPTQ int4 routed experts and lm_head from AutoGPTQ v1 qweight int32 [K/8, N], fp16 scales and qzeros all 7.
const std = @import("std");
const module_ = @import("module.zig");
const driver = @import("driver.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const grouped = @import("grouped.zig");

/// The group AutoGPTQ stores and this kernel reads.
pub const checkpoint_group: usize = 128;
/// Pairs an item of a decode plan holds.
pub const plan_tile: usize = 16;
/// Pairs an item of a prompt plan holds, for the paired and the prompt kernels alike.
pub const prompt_tile: usize = 64;

/// One layer's routed experts: up = [E][gate, up] and down = [E] packed matrices with their scales.
pub const Experts = struct {
    up: u64 = 0,
    up_s: u64 = 0,
    down: u64 = 0,
    down_s: u64 = 0,
    /// the intermediate width (gate/up outputs, down inputs)
    width: u32 = 0,
    dims: u32 = 0,
};

/// One packed matrix [n outputs, k inputs] (the head).
pub const Mat = struct { w: u64 = 0, s: u64 = 0, n: u32 = 0, k: u32 = 0 };

/// Words a packed [n, k] matrix holds (8 codes a word).
pub fn wordsOf(n: usize, k: usize) usize {
    return n * k / 8;
}

/// fp16 scales a packed [n, k] matrix holds: one a group of 128 inputs and output column.
pub fn scalesOf(n: usize, k: usize) usize {
    return n * (k / checkpoint_group);
}

/// A GPTQ word's eight codes (input j at nibble j) reordered for the kernel: nibble i = 2i and i + 4 = 2i + 1.
pub fn shuffle(w: u32) u32 {
    var out: u32 = 0;
    inline for (0..4) |i| {
        out |= ((w >> (8 * i)) & 0xF) << (4 * i);
        out |= ((w >> (8 * i + 4)) & 0xF) << (4 * (i + 4));
    }
    return out;
}

/// One checkpoint matrix: qweight int32 [k_full / 8, n_full] and scales fp16 [k_full / 128, n_full], little-endian.
pub const Source = struct { qweight: []const u8, scales: []const u8, n_full: usize, k_full: usize };

/// Columns [n0, n0 + n) and inputs [k0, k0 + k) of a Source, in groups of `gs` inputs.
pub const Slice = struct { n0: usize, n: usize, k0: usize, k: usize, gs: usize };

/// A slice the kernel takes: groups of 128 inside the source's groups of 128, whole n8 tiles, in bounds.
pub fn checkSlice(src: Source, sl: Slice) !void {
    if (sl.gs != checkpoint_group) return error.BadSlice;
    if (sl.n == 0 or sl.k == 0 or sl.n % 8 != 0 or sl.k % sl.gs != 0 or sl.k0 % sl.gs != 0) return error.BadSlice;
    if (sl.n0 + sl.n > src.n_full or sl.k0 + sl.k > src.k_full or src.k_full % checkpoint_group != 0) return error.BadSlice;
    if (src.qweight.len != src.k_full / 8 * src.n_full * 4 or src.scales.len != src.k_full / checkpoint_group * src.n_full * 2) return error.BadLength;
}

/// Columns [n0, n0 + n) and inputs [k0, k0 + k) of `src` in the kernel's word order: [n/8][k/gs][32][gs/32].
pub fn packWords(src: Source, sl: Slice, out: []u32) !void {
    try checkSlice(src, sl);
    if (out.len != wordsOf(sl.n, sl.k)) return error.BadLength;
    const kg = sl.k / sl.gs;
    const per = sl.gs / 32;
    var at: usize = 0;
    for (0..sl.n / 8) |nt| for (0..kg) |g| for (0..32) |lane| {
        const col = sl.n0 + nt * 8 + lane / 4;
        const t = lane % 4;
        for (0..per) |b| {
            const kp = (sl.k0 + g * sl.gs + b * 32) / 8 + t;
            out[at] = shuffle(std.mem.readInt(u32, src.qweight[(kp * src.n_full + col) * 4 ..][0..4], .little));
            at += 1;
        }
    };
}

/// The scales of the same slice: fp16 bits [n/8][k/gs][8].
pub fn packScales(src: Source, sl: Slice, out: []u16) !void {
    try checkSlice(src, sl);
    const kg = sl.k / sl.gs;
    if (out.len != scalesOf(sl.n, sl.k)) return error.BadLength;
    var at: usize = 0;
    for (0..sl.n / 8) |nt| for (0..kg) |g| for (0..8) |c| {
        const col = sl.n0 + nt * 8 + c;
        const row = (sl.k0 + g * sl.gs) / checkpoint_group;
        out[at] = std.mem.readInt(u16, src.scales[(row * src.n_full + col) * 2 ..][0..2], .little);
        at += 1;
    };
}

/// AutoGPTQ v1 symmetric zero points: every nibble of qzeros [k_full / 128, n_full / 8] is 7 (8 minus one).
pub fn zerosAreSymmetric(qzeros: []const u8) bool {
    var i: usize = 0;
    while (i + 8 <= qzeros.len) : (i += 8) if (std.mem.readInt(u64, qzeros[i..][0..8], .little) != 0x7777777777777777) return false;
    while (i < qzeros.len) : (i += 1) if (qzeros[i] != 0x77) return false;
    return true;
}

/// Code q of input k, column n of `src` (0..15; the weight is (q - 8) * scale).
pub fn code(src: Source, n: usize, k: usize) u4 {
    const w = std.mem.readInt(u32, src.qweight[((k / 8) * src.n_full + n) * 4 ..][0..4], .little);
    return @intCast((w >> @intCast(4 * (k % 8))) & 0xF);
}

/// The fp16 scale of input k's group, column n, widened to f32.
pub fn scale(src: Source, n: usize, k: usize) f32 {
    const h: f16 = @bitCast(std.mem.readInt(u16, src.scales[((k / checkpoint_group) * src.n_full + n) * 2 ..][0..2], .little));
    return @floatCast(h);
}

/// The dequantized weight of input k, column n: (q - 8) * scale, exact in fp32.
pub fn weight(src: Source, n: usize, k: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, code(src, n, k)) - 8)) * scale(src, n, k);
}

/// The same weight in binary64, the reference the device test compares against: both factors are exact.
pub fn weight64(src: Source, n: usize, k: usize) f64 {
    const s: f64 = @floatCast(scale(src, n, k));
    return @as(f64, @floatFromInt(@as(i32, code(src, n, k)) - 8)) * s;
}

// ---- the kernels -------------------------------------------------------------------------------------------------

pub const warps = 4;
/// The three epilogues int4_kernel is built for: gate/up SwiGLU, down to fp32, down to bf16.
const n_kinds = 3;
pub const nts = [_]usize{ 1, 2, 4 };
/// Row tiles of 16 a pass: 1 (decode), 2 (prompt items of up to 64 pairs).
pub const mts = [_]usize{ 1, 2 };

fn symbol(comptime gs: usize, comptime nt: usize, comptime mt: usize, comptime mats: usize, comptime epi: usize) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN10tf_fn_int411int4_kernelILi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi4EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_iPvifi", .{ gs, nt, mt, mats, epi });
}

const kinds = [n_kinds]struct { mats: usize, epi: usize }{
    .{ .mats = 2, .epi = 2 }, .{ .mats = 1, .epi = 0 }, .{ .mats = 1, .epi = 3 },
};

/// Every kept int4_kernel instantiation: [kind][n8 tiles a warp][row tiles a pass], groups of 128 inputs.
pub const symbols = blk: {
    var out: [n_kinds][nts.len][mts.len][:0]const u8 = undefined;
    for (kinds, 0..) |k, i| for (nts, 0..) |nt, j| for (mts, 0..) |mt, q| {
        out[i][j][q] = symbol(checkpoint_group, nt, mt, k.mats, k.epi);
    };
    break :blk out;
};

const prompt_threads = 256;
/// Stages the prompt kernel keeps in flight by output type: fp32 out, bf16 out.
const prompt_stages = [2]usize{ 2, 2 };

fn promptSymbol(comptime gs: usize, comptime epi: usize, comptime st: usize) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN10tf_fn_int418int4_prompt_kernelILi{d}ELi1ELi{d}ELi2ELi4ELi2ELi4ELi{d}EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_Pvifi", .{ gs, epi, st });
}

/// The routed down of a prompt call: a block an (item of up to 64 pairs, 128 columns), 8 warps, by output type.
pub const prompt_symbols = [2][:0]const u8{ promptSymbol(checkpoint_group, 0, 2), promptSymbol(checkpoint_group, 3, 2) };

/// The routed gate/up of a prompt call: {row warps, column warps, row tiles a warp, n8 tiles a warp, stages}.
pub const PromptGu = struct { wr: usize, wc: usize, rt: usize, wn: usize, stages: usize };
pub const prompt_gu: PromptGu = .{ .wr = 2, .wc = 4, .rt = 2, .wn = 4, .stages = 3 };

pub const prompt_gu_symbol = std.fmt.comptimePrint("_ZN10tf_fn_int418int4_prompt_kernelILi128ELi2ELi2ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_Pvifi", .{ prompt_gu.wr, prompt_gu.wc, prompt_gu.rt, prompt_gu.wn, prompt_gu.stages });

/// int4_prompt_kernel's dynamic shared memory: stages x (rows of a group, each matrix's words and scales).
pub fn promptSmemOf(mats: usize, br: usize, bt: usize, stages: usize) u32 {
    const gs = checkpoint_group;
    return @intCast(stages * (br * gs * 2 + mats * bt * 32 * (gs / 32) * 4 + mats * bt * 16));
}

/// The routed down's shared memory: `prompt_tile` rows, 16 n8 tiles, one matrix.
pub fn promptSmem(stages: usize) u32 {
    return promptSmemOf(1, prompt_tile, 16, stages);
}

pub fn promptGuRows(p: PromptGu) usize {
    return p.wr * p.rt * 16;
}
pub fn promptGuTiles(p: PromptGu) usize {
    return p.wc * p.wn;
}
pub fn promptGuSmem(p: PromptGu) u32 {
    return promptSmemOf(2, promptGuRows(p), promptGuTiles(p), p.stages);
}

/// sm_121's (GB10's) opt-in shared memory a block.
pub const sm121_smem_optin = 101376;
comptime {
    std.debug.assert(promptGuSmem(prompt_gu) <= sm121_smem_optin);
}

/// Whether a block of prompt_gu fits a GPU: its dynamic plus `static` shared memory within the opt-in limit.
pub fn promptGuFits(static: usize, optin: usize) bool {
    return promptGuSmem(prompt_gu) + static <= optin;
}

/// Whether gateUpPrompt takes a call on `ex`: whole column blocks, K groups of 128, aligned rows, words and stores.
pub fn gateUpPromptFits(ex: Experts, x: u64, x_stride: usize, out: u64) bool {
    const cols = 8 * promptGuTiles(prompt_gu);
    return ex.width % cols == 0 and ex.dims % checkpoint_group == 0 and x_stride % 8 == 0 and x % 16 == 0 and ex.up % 16 == 0 and ex.up_s % 16 == 0 and out % 8 == 0;
}

/// The active context's device and the two limits `Kernels.load` sizes from.
pub const Limits = struct {
    device: c_int,
    /// multiprocessor count: resident blocks are an occupancy times this
    sms: usize,
    /// max_shared_memory_per_block_optin: the prompt gate/up is admitted only when its block fits this
    optin: usize,
};

/// The limits of the device the current context runs on, through `cuCtxGetDevice`; ordinal 0's are another GPU's.
pub fn activeLimits(d: *const driver.Driver) !Limits {
    var device: c_int = 0;
    try d.check(d.api.cuCtxGetDevice(&device), "cuCtxGetDevice");
    var sms: c_int = 0;
    var optin: c_int = 0;
    try d.check(d.api.cuDeviceGetAttribute(&sms, .multiprocessor_count, device), "cuDeviceGetAttribute");
    try d.check(d.api.cuDeviceGetAttribute(&optin, .max_shared_memory_per_block_optin, device), "cuDeviceGetAttribute");
    return .{ .device = device, .sms = @intCast(sms), .optin = @intCast(optin) };
}

/// Resident blocks of a kernel on a device: its occupancy an SM (at least one), times the device's SM count.
pub fn residentBlocks(occupancy: u32, sms: usize) usize {
    return @max(1, occupancy) * sms;
}

pub const Kernels = struct {
    module: module_.Module,
    fns: [n_kinds][nts.len][mts.len]module_.Function,
    /// resident blocks of each function on this GPU (occupancy x SMs)
    blocks: [n_kinds][nts.len][mts.len]usize,
    /// the prompt down by output type (fp32, bf16) and its resident blocks
    pdown: [2]module_.Function,
    pdown_blocks: [2]usize,
    /// the prompt gate/up, resident blocks and whether the GPU's shared memory takes a block of it
    pgu_fn: module_.Function,
    pgu_blocks: usize,
    pgu_ready: bool,

    pub fn load(d: *const driver.Driver) !Kernels {
        if (!kernels.available) return error.BuiltWithoutKernels;
        return loadModule(d, try module_.Module.load(d, kernels.fn_int4));
    }

    /// `load` over a module it owns, split so a host test drives every sizing decision through a fake driver.
    pub fn loadModule(d: *const driver.Driver, module: module_.Module) !Kernels {
        var k: Kernels = undefined;
        k.module = module;
        errdefer k.module.unload(); // ownership passes first: every later error unloads it once
        const limits = try activeLimits(d);
        const sms = limits.sms;
        for (symbols, 0..) |row, i| for (row, 0..) |cols, j| for (cols, 0..) |s, q| {
            k.fns[i][j][q] = try k.module.function(s);
            k.blocks[i][j][q] = residentBlocks(try k.fns[i][j][q].occupancy(warps * 32, 0), sms);
        };
        for (0..2) |o| {
            const sm = promptSmem(prompt_stages[o]);
            k.pdown[o] = try k.module.function(prompt_symbols[o]);
            try k.pdown[o].allowDynamicShared(sm);
            k.pdown_blocks[o] = residentBlocks(try k.pdown[o].occupancy(prompt_threads, sm), sms);
        }
        const sm_gu = promptGuSmem(prompt_gu);
        k.pgu_fn = try k.module.function(prompt_gu_symbol);
        const static: usize = @intCast(try k.pgu_fn.attribute(.shared_size_bytes));
        k.pgu_ready = promptGuFits(static, limits.optin);
        k.pgu_blocks = 0;
        if (k.pgu_ready) {
            try k.pgu_fn.allowDynamicShared(sm_gu);
            k.pgu_blocks = residentBlocks(try k.pgu_fn.occupancy(@intCast(prompt_gu.wr * prompt_gu.wc * 32), sm_gu), sms);
        }
        std.log.info("int4 kernels: prompt gate/up {s}, prompt down {d} blocks", .{ if (k.pgu_ready) "a block a 64-row item" else "not taken", k.pdown_blocks[0] });
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.module.unload();
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

/// The widest n8-tile count whose units still fill the GPU twice over (the same bits at any NT).
fn pickNt(k: *const Kernels, kind: usize, max_items: usize, n: usize, q: usize) usize {
    var j: usize = nts.len;
    while (j > 1) {
        j -= 1;
        const units = max_items * (n / (8 * nts[j]));
        if (n % (8 * nts[j]) == 0 and units >= 2 * k.blocks[kind][j][q] * warps) return j;
    }
    return 0;
}

fn launch(k: *const Kernels, s: stream_.Stream, kind: usize, max_items: usize, x: u64, x_stride: usize, slots: usize, w: u64, sc: u64, kk: usize, n: usize, p: ?grouped.Plan, rows: usize, out: u64, out_stride: usize, skip: c_int, mt: usize) !void {
    // uint4 loads of the input rows, uint4 weight words, half2 scales, float2/bf16x2 stores
    if (n % 8 != 0 or kk % checkpoint_group != 0 or x_stride % 8 != 0 or x % 16 != 0 or w % 16 != 0 or sc % 4 != 0 or out % 8 != 0) return error.Invalid;
    const q = std.mem.indexOfScalar(usize, &mts, mt) orelse return error.Invalid;
    const j = pickNt(k, kind, max_items, n, q);
    if (n % (8 * nts[j]) != 0) return error.Invalid;
    const units = max_items * (n / (8 * nts[j]));
    const grid = @min((units + warps - 1) / warps, k.blocks[kind][j][q]);
    if (grid < 1) return;
    var a: launch_.Args = .{};
    a.add(x);
    a.add(int(x_stride));
    a.add(int(slots));
    a.add(w);
    a.add(sc);
    a.add(int(kk));
    a.add(int(n));
    if (p) |pl| {
        a.add(pl.items);
        a.add(pl.counts);
        a.add(pl.members);
    } else {
        a.add(@as(u64, 0));
        a.add(@as(u64, 0));
        a.add(@as(u64, 0));
    }
    a.add(int(rows));
    a.add(out);
    a.add(int(out_stride));
    a.add(@as(f32, 0));
    a.add(skip);
    try launch_.launch(k.fns[kind][j][q], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = warps * 32, .y = 1, .z = 1 } }, s, &a);
}

/// The routed gate/up with SwiGLU: x [R, D] token rows -> out [R * slots, width] bf16, `skip` the expert to leave.
pub fn gateUp(k: *const Kernels, s: stream_.Stream, x: u64, x_stride: usize, ex: Experts, p: grouped.Plan, plan_slots: usize, plan_experts: usize, out: u64, rows: usize, skip: c_int, tile: usize) !void {
    const items = grouped.maxItems(rows * plan_slots, plan_experts, tile);
    try launch(k, s, 0, items, x, x_stride, plan_slots, ex.up, ex.up_s, ex.dims, ex.width, p, 0, out, ex.width, skip, if (tile > plan_tile) 2 else 1);
}

/// The routed experts' down: act [R * slots, width] pair rows -> out [R * slots, D], fp32 (`f32_out`) or bf16.
pub fn down(k: *const Kernels, s: stream_.Stream, act: u64, act_stride: usize, ex: Experts, p: grouped.Plan, plan_slots: usize, plan_experts: usize, out: u64, f32_out: bool, rows: usize, skip: c_int, tile: usize) !void {
    const items = grouped.maxItems(rows * plan_slots, plan_experts, tile);
    const kind: usize = if (f32_out) 1 else 2;
    try launch(k, s, kind, items, act, act_stride, 0, ex.down, ex.down_s, ex.width, ex.dims, p, 0, out, ex.dims, skip, if (tile > plan_tile) 2 else 1);
}

/// The routed down of a prompt call on int4_prompt_kernel: `down`'s outputs, from a plan of `prompt_tile` items.
pub fn downPrompt(k: *const Kernels, s: stream_.Stream, act: u64, act_stride: usize, ex: Experts, p: grouped.Plan, plan_slots: usize, plan_experts: usize, out: u64, f32_out: bool, rows: usize, skip: c_int) !void {
    const n: usize = ex.dims;
    const kk: usize = ex.width;
    if (n % 128 != 0 or kk % checkpoint_group != 0 or act_stride % 8 != 0 or act % 16 != 0 or ex.down % 16 != 0 or ex.down_s % 16 != 0 or out % 8 != 0) return error.Invalid;
    const g = @intFromBool(!f32_out);
    const items = grouped.maxItems(rows * plan_slots, plan_experts, prompt_tile);
    const units = items * (n / 128);
    const grid = @min(units, k.pdown_blocks[g]);
    if (grid < 1) return;
    var a: launch_.Args = .{};
    a.add(act);
    a.add(int(act_stride));
    a.add(@as(c_int, 0));
    a.add(ex.down);
    a.add(ex.down_s);
    a.add(int(kk));
    a.add(int(n));
    a.add(p.items);
    a.add(p.counts);
    a.add(p.members);
    a.add(out);
    a.add(int(n));
    a.add(@as(f32, 0));
    a.add(skip);
    try launch_.launch(k.pdown[g], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = prompt_threads, .y = 1, .z = 1 }, .shared = promptSmem(prompt_stages[g]) }, s, &a);
}

/// The routed gate/up of a prompt call on int4_prompt_kernel (prompt_gu): `gateUp`'s outputs, byte for byte.
pub fn gateUpPrompt(k: *const Kernels, s: stream_.Stream, x: u64, x_stride: usize, ex: Experts, p: grouped.Plan, plan_slots: usize, plan_experts: usize, out: u64, rows: usize, skip: c_int) !void {
    const sh = prompt_gu;
    const n: usize = ex.width;
    if (!k.pgu_ready or !gateUpPromptFits(ex, x, x_stride, out)) return error.Invalid;
    const items = grouped.maxItems(rows * plan_slots, plan_experts, prompt_tile);
    const grid = @min(items * (n / (8 * promptGuTiles(sh))), k.pgu_blocks);
    if (grid < 1) return;
    var a: launch_.Args = .{};
    a.add(x);
    a.add(int(x_stride));
    a.add(int(plan_slots));
    a.add(ex.up);
    a.add(ex.up_s);
    a.add(int(ex.dims));
    a.add(int(n));
    a.add(p.items);
    a.add(p.counts);
    a.add(p.members);
    a.add(out);
    a.add(int(n));
    a.add(@as(f32, 0));
    a.add(skip);
    try launch_.launch(k.pgu_fn, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = @intCast(sh.wr * sh.wc * 32), .y = 1, .z = 1 }, .shared = promptGuSmem(sh) }, s, &a);
}

/// A dense matmul (the head): x [rows, k] -> out [rows, n] bf16 (or fp32), rows `x_stride` apart.
pub fn dense(k: *const Kernels, s: stream_.Stream, x: u64, x_stride: usize, m: Mat, out: u64, f32_out: bool, rows: usize) !void {
    if (rows == 0) return;
    const kind: usize = if (f32_out) 1 else 2;
    try launch(k, s, kind, (rows + 15) / 16, x, if (rows == 1) m.k else x_stride, 0, m.w, m.s, m.k, m.n, null, rows, out, m.n, -1, 1);
}

test {
    _ = @import("int4_test.zig");
}
