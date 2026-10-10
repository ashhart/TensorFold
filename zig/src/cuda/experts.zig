//! Grouped experts behind one call: a layer view per format on grouped.zig's plan, decode rows and prompt rows apart.
const std = @import("std");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const grouped = @import("grouped.zig");
const driver = @import("driver.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const Plan = grouped.Plan;

pub const Format = enum { affine4, fp8g, nvfp4 };

/// One layer's experts on the device as each format's make() lays them out; a family stacks its shared ones last.
pub const Layer = struct {
    format: Format,
    up: u64, // affine4 [E][NI/32][D/64][288] int32; fp8g [E][NI/32][D/32][2][32][2][4]; nvfp4 [E][NI/32][D/32][2][144]
    down: u64, // affine4 [E][D/32][NI/64][288]; fp8g [E][D/32][NI/32][1][32][2][4]; nvfp4 [E][D/32][NI/32][1][144]
    up_scale: u64 = 0, // fp8g [E][2][NI/128][D/128] fp32; nvfp4 [E][2] global scales; affine4's are in the blocks
    down_scale: u64 = 0, // fp8g [E][1][D/128][NI/128]; nvfp4 [E][1]
    width: usize, // NI
    dims: usize, // D
    experts: usize,
    limit: f32 = 0.0, // fp8g, nvfp4: the SwiGLU clamp, 0 for none
    relu2: bool = false, // nvfp4: up is one projection under relu^2 (Nemotron), not SwiGLU's gate and up
};

/// A call's epilogue: up with its format's activation (affine4 relu^2, else SwiGLU), down to fp32 or to bf16 sums.
pub const Epi = enum { up, down_f32, down_bf16 };

/// The rows in: bf16 `x` rows `stride` apart; `up` reads token rows (`slots` pairs a row), down reads pair rows.
pub const Rows = struct { x: u64, stride: usize, slots: usize = 0 };

/// The kernels of every format a family loaded; a layer whose format is not loaded is refused.
pub const Grouped = struct {
    affine4: ?*const Affine4Experts = null,
    fp8g: ?*const Fp8Experts = null,
    nvfp4: ?*const Nvfp4Experts = null,

    /// Decode rows: `items` bounds the plan's items (grouped.maxItems at tile 16); `skip` an expert left out, or -1.
    pub fn decode(g: Grouped, s: Stream, epi: Epi, in: Rows, l: Layer, p: Plan, out: u64, items: usize, skip: i32) !void {
        switch (l.format) {
            .affine4 => {
                const a = g.affine4 orelse return error.FormatNotLoaded;
                if (epi == .down_bf16) return error.EpilogueNotBuilt;
                if (skip >= 0 or l.limit != 0) return error.OptionNotBuilt;
                const up = epi == .up;
                const nb = (if (up) l.width else l.dims) / 32;
                const w = Affine4Experts.view(l, up);
                return a.run(s, up, in.x, in.stride, if (up) in.slots else 0, w.ptr, w.kg, nb, p, out, w.n, items * nb);
            },
            .fp8g => return (g.fp8g orelse return error.FormatNotLoaded).run(s, epi, in, l, p, false, out, items, skip),
            .nvfp4 => return (g.nvfp4 orelse return error.FormatNotLoaded).run(s, epi, in, l, p, out, items, skip),
        }
    }

    /// Prompt rows: `items` bounds the plan's items at the format's prompt tile (prompt_tile).
    pub fn prompt(g: Grouped, s: Stream, epi: Epi, in: Rows, l: Layer, p: Plan, out: u64, items: usize, skip: i32) !void {
        switch (l.format) {
            .affine4 => {
                const a = g.affine4 orelse return error.FormatNotLoaded;
                if (epi == .down_f32) return error.EpilogueNotBuilt;
                if (skip >= 0 or l.limit != 0) return error.OptionNotBuilt;
                const up = epi == .up;
                const nb = (if (up) l.width else l.dims) / 32;
                const w = Affine4Experts.view(l, up);
                return a.runPrompt(s, up, in.x, in.stride, if (up) in.slots else 0, w.ptr, w.kg, nb, p, out, w.n, items);
            },
            .fp8g => return (g.fp8g orelse return error.FormatNotLoaded).run(s, epi, in, l, p, true, out, items, skip),
            // the decode form over the prompt plan, bf16 sums down, until a staged NVFP4 prompt kernel lands (#548)
            .nvfp4 => return (g.nvfp4 orelse return error.FormatNotLoaded).run(s, epi, in, l, p, out, items, skip),
        }
    }
};

/// Pairs a prompt item holds, per format (the Python engines' prompt tiles).
pub fn promptTile(f: Format) usize {
    return switch (f) {
        .affine4, .fp8g, .nvfp4 => 64,
    };
}

fn int(x: usize) c_int {
    return @intCast(x);
}

/// MLX affine-4 experts (experts.cu, experts_prefill.cu, experts_pack.cu): relu^2 up, down to fp32 or bf16 sums.
pub const Affine4Experts = struct {
    up_fn: Function,
    down_fn: Function,
    pre_up_fn: Function,
    pre_down_fn: Function,
    pack_fn: Function,
    resident: [2]usize, // blocks the decode up and down kernels fill: per SM times SMs

    pub const prompt_smem: u32 = 38400; // Pre<64, 1, 2, 2, 4>: three stages of 800 uint4

    /// The instances' names (cuobjdump -symbols of the experts, experts_prefill and experts_pack fatbins).
    pub const symbols = struct {
        pub const up = "_ZN10tf_experts13expert_kernelILi64ELi1ELi1ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
        pub const down = "_ZN10tf_experts13expert_kernelILi64ELi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
        pub const pre_up = "_ZN18tf_experts_prefill14prefill_kernelILi64ELi1ELi1ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
        pub const pre_down = "_ZN18tf_experts_prefill14prefill_kernelILi64ELi1ELi3ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
        pub const pack = "_ZN15tf_experts_pack11pack_kernelILi2EEEvPKjPKtS4_Pjiiii";
    };

    /// From loaded experts, experts_prefill and experts_pack modules.
    pub fn resolve(decode_mod: Module, prompt_mod: Module, pack_mod: Module, sms: usize) !Affine4Experts {
        var a: Affine4Experts = undefined;
        a.up_fn = try decode_mod.function(symbols.up);
        a.down_fn = try decode_mod.function(symbols.down);
        a.pre_up_fn = try prompt_mod.function(symbols.pre_up);
        a.pre_down_fn = try prompt_mod.function(symbols.pre_down);
        a.pack_fn = try pack_mod.function(symbols.pack);
        try a.pre_up_fn.allowDynamicShared(prompt_smem);
        try a.pre_down_fn.allowDynamicShared(prompt_smem);
        a.resident = .{ @max(1, try a.up_fn.occupancy(128, 0)) * sms, @max(1, try a.down_fn.occupancy(128, 0)) * sms };
        return a;
    }

    /// The table a call reads: up takes D inputs to NI outputs, down NI to D; K in groups of 64.
    fn view(l: Layer, up: bool) struct { ptr: u64, kg: usize, n: usize } {
        return if (up) .{ .ptr = l.up, .kg = l.dims / 64, .n = l.width } else .{ .ptr = l.down, .kg = l.width / 64, .n = l.dims };
    }

    fn args(x: u64, x_stride: usize, slots: usize, w: u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize) launch_.Args {
        var a: launch_.Args = .{};
        a.add(x);
        a.add(int(x_stride));
        a.add(int(slots));
        a.add(w);
        a.add(int(kg));
        a.add(int(nb));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(int(n));
        a.add(@as(f32, 0.0));
        return a;
    }

    /// experts.run (decode form): `up` takes token rows (relu^2, bf16 out), else pair rows (fp32 out).
    pub fn run(e: *const Affine4Experts, s: Stream, up: bool, x: u64, x_stride: usize, slots: usize, w: u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize, max_units: usize) !void {
        const grid = @min((max_units + 3) / 4, e.resident[if (up) 0 else 1]);
        if (grid < 1) return;
        var a = args(x, x_stride, slots, w, kg, nb, p, out, n);
        const cfg: launch_.Config = .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128 }, .shared = 0 };
        try launch_.launch(if (up) e.up_fn else e.down_fn, cfg, s, &a);
    }

    /// experts.prefill: 64 pairs x 128 columns a CTA; `up` relu^2, else bf16 sums (epilogue 3).
    pub fn runPrompt(e: *const Affine4Experts, s: Stream, up: bool, x: u64, x_stride: usize, slots: usize, w: u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize, max_items: usize) !void {
        const grid = max_items * ((nb + 3) / 4);
        if (grid < 1) return;
        var a = args(x, x_stride, slots, w, kg, nb, p, out, n);
        const cfg: launch_.Config = .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 256 }, .shared = prompt_smem };
        try launch_.launch(if (up) e.pre_up_fn else e.pre_down_fn, cfg, s, &a);
    }

    /// experts.make: e experts' MLX words (n, k/8), scales and biases (n, k/64) on the device into `out`'s blocks.
    pub fn pack(e: *const Affine4Experts, s: Stream, words: u64, scales: u64, biases: u64, out: u64, count: usize, n: usize, k: usize) !void {
        const kg = k / 64;
        const nb = n / 32;
        var a: launch_.Args = .{};
        for ([_]u64{ words, scales, biases, out }) |v| a.add(v);
        for ([_]usize{ n, k / 8, kg, nb }) |v| a.add(int(v));
        try launch_.launch(e.pack_fn, .{ .grid = .{ .x = @intCast(kg), .y = @intCast(nb), .z = @intCast(count) }, .block = .{ .x = 288 } }, s, &a);
    }

    /// Bytes of e experts' blocks for n outputs from k inputs ([E][n/32][k/64][288] int32).
    pub fn bytes(count: usize, n: usize, k: usize) usize {
        return count * (n / 32) * (k / 64) * 288 * 4;
    }
};

/// Block-FP8 experts (fp8/experts.cu): the Python packing, kernels and grid choice, decode and prompt forms.
pub const Fp8Experts = struct {
    mod: Module,
    fns: [2][3]Function, // decode, prompt; by Epi
    resident: [2][3]usize, // blocks the grid may hold: per SM times SMs

    pub const cols = 32; // output columns a warp
    pub const block = 128; // the checkpoint's scale block, both ways

    pub const symbols = struct {
        pub const gate_up = "_ZN14tf_fp8_experts17fp8_expert_kernelILi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
        pub const down_f32 = "_ZN14tf_fp8_experts17fp8_expert_kernelILi1ELi0ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
        pub const down_bf16 = "_ZN14tf_fp8_experts17fp8_expert_kernelILi1ELi3ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
        pub const p_gate_up = "_ZN14tf_fp8_experts24fp8_expert_prompt_kernelILi2ELi2ELi2ELi2EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
        pub const p_down_f32 = "_ZN14tf_fp8_experts24fp8_expert_prompt_kernelILi1ELi0ELi4ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
        pub const p_down_bf16 = "_ZN14tf_fp8_experts24fp8_expert_prompt_kernelILi1ELi3ELi4ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
    };

    pub fn load(d: *const Driver, sms: usize) !Fp8Experts {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.fp8_experts);
        errdefer mod.unload();
        var e: Fp8Experts = .{ .mod = mod, .fns = undefined, .resident = undefined };
        const names = [2][3][:0]const u8{ .{ symbols.gate_up, symbols.down_f32, symbols.down_bf16 }, .{ symbols.p_gate_up, symbols.p_down_f32, symbols.p_down_bf16 } };
        for (names, 0..) |set, i| for (set, 0..) |name, j| {
            e.fns[i][j] = try mod.function(name);
            e.resident[i][j] = @max(1, try e.fns[i][j].occupancy(128, 0)) * sms;
        };
        return e;
    }

    pub fn unload(e: *Fp8Experts) void {
        e.mod.unload();
    }

    /// One call of fp8/experts.py's _run on a plan of at most `items` items.
    pub fn run(e: *const Fp8Experts, s: Stream, epi: Epi, in: Rows, l: Layer, p: Plan, prompt: bool, out: u64, items: usize, skip: i32) !void {
        const gate = epi == .up;
        const kg = (if (gate) l.dims else l.width) / 32;
        const nb = (if (gate) l.width else l.dims) / cols;
        const n = if (gate) l.width else l.dims;
        const pi: usize = @intFromBool(prompt);
        const ei: usize = @backingInt(epi);
        const grid: usize = if (prompt) blk: {
            const cw: usize = if (gate) 2 else 4;
            break :blk @min(items * (nb / cw), e.resident[pi][ei]);
        } else @min((items * nb + 3) / 4, e.resident[pi][ei]);
        if (grid < 1) return;
        var a: launch_.Args = .{};
        a.add(in.x);
        a.add(@as(i32, @intCast(in.stride)));
        a.add(@as(i32, @intCast(if (gate) in.slots else 0)));
        a.add(if (gate) l.up else l.down);
        a.add(if (gate) l.up_scale else l.down_scale);
        a.add(@as(i32, @intCast(kg)));
        a.add(@as(i32, @intCast(nb)));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(n)));
        a.add(if (gate) l.limit else @as(f32, 0.0));
        a.add(skip);
        try launch_.launch(e.fns[pi][ei], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }

    /// e4m3 [n, k] of one expert as int32 [n/32][k/32][32][2][4]: lane (gq, t)'s half h, tile j (fp8/experts.py pack).
    pub fn packOne(out: []u32, w: []const u8, n: usize, k: usize) void {
        std.debug.assert(n % block == 0 and k % block == 0 and out.len == n * k / 4 and w.len == n * k);
        const words = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, w));
        var i: usize = 0;
        for (0..n / cols) |cb| for (0..k / 32) |g| for (0..8) |gq| for (0..4) |t| for (0..2) |h| for (0..4) |j| {
            const row = cb * cols + j * 8 + gq;
            out[i] = words[row * (k / 4) + g * 8 + h * 4 + t];
            i += 1;
        };
    }

    /// Gate and up interleaved per (column block, k32 group) as make() stacks them: [n/32][k/32][2][32][2][4].
    pub fn packGateUp(gpa: std.mem.Allocator, out: []u32, gate: []const u8, up: []const u8, n: usize, k: usize) !void {
        const half = n * k / 4;
        std.debug.assert(out.len == 2 * half);
        const g = try gpa.alloc(u32, half);
        defer gpa.free(g);
        const u = try gpa.alloc(u32, half);
        defer gpa.free(u);
        packOne(g, gate, n, k);
        packOne(u, up, n, k);
        const unit = 32 * 2 * 4; // a lane block of one k32 group
        for (0..half / unit) |b| {
            @memcpy(out[(2 * b) * unit ..][0..unit], g[b * unit ..][0..unit]);
            @memcpy(out[(2 * b + 1) * unit ..][0..unit], u[b * unit ..][0..unit]);
        }
    }
};

/// NVFP4 experts (nvfp4/experts.cu): the Python packing and kernel, 16 pairs at a time, items of any size.
pub const Nvfp4Experts = struct {
    mod: Module,
    fns: [4]Function, // by Epi, then relu^2 up
    resident: [4]usize, // blocks the grid may hold: per SM times SMs

    pub const cols = 32; // output columns a block
    pub const words = 144; // int32 a (32 columns, 32 inputs) block: 128 code words, then 16 of e4m3 scales

    pub const symbols = struct {
        pub const gate_up = "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
        pub const down_f32 = "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi1ELi0ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
        pub const down_bf16 = "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi1ELi3ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
        pub const up_relu2 = "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi1ELi1ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
    };

    pub fn load(d: *const Driver, sms: usize) !Nvfp4Experts {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.nvfp4_experts);
        errdefer mod.unload();
        var e: Nvfp4Experts = .{ .mod = mod, .fns = undefined, .resident = undefined };
        for ([_][:0]const u8{ symbols.gate_up, symbols.down_f32, symbols.down_bf16, symbols.up_relu2 }, 0..) |name, i| {
            e.fns[i] = try mod.function(name);
            e.resident[i] = @max(1, try e.fns[i].occupancy(128, 0)) * sms;
        }
        return e;
    }

    pub fn unload(e: *Nvfp4Experts) void {
        e.mod.unload();
    }

    /// One call of nvfp4/experts.py's _run on a plan of at most `items` items.
    pub fn run(e: *const Nvfp4Experts, s: Stream, epi: Epi, in: Rows, l: Layer, p: Plan, out: u64, items: usize, skip: i32) !void {
        const gate = epi == .up;
        const kg = (if (gate) l.dims else l.width) / 32;
        const nb = (if (gate) l.width else l.dims) / cols;
        const ei: usize = if (gate and l.relu2) 3 else @backingInt(epi);
        const grid = @min((items * nb + 3) / 4, e.resident[ei]);
        if (grid < 1) return;
        var a: launch_.Args = .{};
        a.add(in.x);
        a.add(@as(i32, @intCast(in.stride)));
        a.add(@as(i32, @intCast(if (gate) in.slots else 0)));
        a.add(if (gate) l.up else l.down);
        a.add(if (gate) l.up_scale else l.down_scale);
        a.add(@as(i32, @intCast(kg)));
        a.add(@as(i32, @intCast(nb)));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(if (gate) l.width else l.dims)));
        a.add(if (gate) l.limit else @as(f32, 0.0));
        a.add(skip);
        try launch_.launch(e.fns[ei], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }

    /// Bytes of `count` experts' blocks for n outputs from k inputs: [E][n/32][k/32][parts][144] int32.
    pub fn bytes(count: usize, n: usize, k: usize, parts: usize) usize {
        return count * (n / cols) * (k / 32) * parts * words * 4;
    }

    /// Where input 16h + 4t + q of a lane's word sits: nibble 2h + q/2 + 4(q%2) (fp4pair's field order).
    fn slot(h: usize, q: usize) u5 {
        return @intCast(4 * (2 * h + q / 2 + 4 * (q % 2)));
    }

    /// One expert's words [n, k/2] (low nibble first) and e4m3 scales [n, k/16] as blocks [n/32][k/32][144] (_pack).
    pub fn packOne(out: []u32, codes: []const u8, scales: []const u8, n: usize, k: usize) void {
        const kg = k / 32;
        std.debug.assert(n % cols == 0 and k % 32 == 0 and out.len == n / cols * kg * words);
        std.debug.assert(codes.len == n * k / 2 and scales.len == n * k / 16);
        for (0..n / cols) |cb| for (0..kg) |g| {
            const block = out[(cb * kg + g) * words ..][0..words];
            for (0..8) |gq| for (0..4) |t| for (0..4) |j| {
                const row = cb * cols + j * 8 + gq;
                var w: u32 = 0;
                for (0..2) |h| for (0..4) |q| {
                    const input = g * 32 + h * 16 + t * 4 + q;
                    const byte = codes[row * (k / 2) + input / 2];
                    const nib: u32 = if (input & 1 == 0) byte & 0xF else byte >> 4;
                    w |= nib << slot(h, q);
                };
                block[(gq * 4 + t) * 4 + j] = w;
            };
            const sc = std.mem.sliceAsBytes(block[128..]);
            for (0..4) |t| for (0..2) |h| for (0..4) |j| for (0..2) |c| {
                const row = cb * cols + j * 8 + t * 2 + c;
                sc[((t * 2 + h) * 4 + j) * 2 + c] = scales[row * (k / 16) + g * 2 + h];
            };
        };
    }

    /// Gate and up interleaved per (column block, k32 group) as make() stacks them: [n/32][k/32][2][144].
    pub fn packGateUp(gpa: std.mem.Allocator, out: []u32, gate: [2][]const u8, up: [2][]const u8, n: usize, k: usize) !void {
        const half = n / cols * (k / 32) * words;
        std.debug.assert(out.len == 2 * half);
        const g = try gpa.alloc(u32, half);
        defer gpa.free(g);
        const u = try gpa.alloc(u32, half);
        defer gpa.free(u);
        packOne(g, gate[0], gate[1], n, k);
        packOne(u, up[0], up[1], n, k);
        for (0..half / words) |b| {
            @memcpy(out[(2 * b) * words ..][0..words], g[b * words ..][0..words]);
            @memcpy(out[(2 * b + 1) * words ..][0..words], u[b * words ..][0..words]);
        }
    }
};

test "a packed nvfp4 block puts each nibble and scale where nvfp4/experts.py's _pack does" {
    const gpa = std.testing.allocator;
    const n = 64;
    const k = 64;
    const codes = try gpa.alloc(u8, n * k / 2);
    defer gpa.free(codes);
    for (codes, 0..) |*c, i| c.* = @truncate(i * 29 + 3);
    var scales: [n * k / 16]u8 = undefined;
    for (&scales, 0..) |*c, i| c.* = @truncate(i * 7 + 1);
    var out: [n / 32 * (k / 32) * Nvfp4Experts.words]u32 = undefined;
    Nvfp4Experts.packOne(&out, codes, &scales, n, k);
    const words = Nvfp4Experts.words;
    // block (cb 1, g 1), lane gq 3, t 2, tile j 1: row 32 + 8 + 3 = 43; input 32 + 16 + 8 + 3 (h 1, q 3) -> slot 7
    const w = out[(1 * 2 + 1) * words + (3 * 4 + 2) * 4 + 1];
    try std.testing.expectEqual(@as(u32, codes[43 * 32 + 59 / 2] >> 4), (w >> Nvfp4Experts.slot(1, 3)) & 0xF);
    // its scale byte (t 2, h 1, j 1, c 0): row 32 + 8 + 4 = 44, k16 group 3
    const sc = std.mem.sliceAsBytes(out[(1 * 2 + 1) * words + 128 ..][0..16]);
    try std.testing.expectEqual(scales[44 * 4 + 3], sc[((2 * 2 + 1) * 4 + 1) * 2]);
}

test "nvfp4 rows of either kind are refused when the format is not loaded" {
    const g: Grouped = .{};
    const l: Layer = .{ .format = .nvfp4, .up = 0, .down = 0, .width = 64, .dims = 64, .experts = 2, .relu2 = true };
    const s: Stream = undefined; // never reached: the refusal comes first
    const p: Plan = .{ .members = 0, .items = 0, .counts = 0, .rank = 0, .hist = 0 };
    try std.testing.expectError(error.FormatNotLoaded, g.prompt(s, .up, .{ .x = 0, .stride = 64 }, l, p, 0, 1, -1));
    try std.testing.expectError(error.FormatNotLoaded, g.decode(s, .down_bf16, .{ .x = 0, .stride = 64 }, l, p, 0, 1, -1));
    try std.testing.expectEqual(@as(usize, 2 * 2 * 3 * 144 * 4), Nvfp4Experts.bytes(2, 64, 96, 1));
}

test "a packed fp8 expert puts each word where fp8/experts.py's pack does" {
    const gpa = std.testing.allocator;
    const n = 128;
    const k = 128;
    const w = try gpa.alloc(u8, n * k);
    defer gpa.free(w);
    for (w, 0..) |*b, i| b.* = @truncate(i * 31 + 7);
    const out = try gpa.alloc(u32, n * k / 4);
    defer gpa.free(out);
    Fp8Experts.packOne(out, w, n, k);
    // (cb 1, g 2, gq 5, t 3, h 1, j 2): row 32 + 16 + 5 = 53, word 2 * 8 + 4 + 3 = 23
    const idx = (((((1 * 4 + 2) * 8 + 5) * 4 + 3) * 2 + 1) * 4) + 2;
    const words = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, w));
    try std.testing.expectEqual(words[53 * (k / 4) + 23], out[idx]);
}

test "a format that is not loaded, or an option it was not built with, is refused before any launch" {
    const g: Grouped = .{};
    const l: Layer = .{ .format = .fp8g, .up = 0, .down = 0, .width = 64, .dims = 64, .experts = 2 };
    const s: Stream = undefined; // never reached: the refusal comes first
    const p: Plan = .{ .members = 0, .items = 0, .counts = 0, .rank = 0, .hist = 0 };
    try std.testing.expectError(error.FormatNotLoaded, g.decode(s, .up, .{ .x = 0, .stride = 64 }, l, p, 0, 1, -1));
    const a4: Affine4Experts = undefined;
    const h: Grouped = .{ .affine4 = &a4 };
    var la = l;
    la.format = .affine4;
    try std.testing.expectError(error.EpilogueNotBuilt, h.decode(s, .down_bf16, .{ .x = 0, .stride = 64 }, la, p, 0, 1, -1));
    try std.testing.expectError(error.EpilogueNotBuilt, h.prompt(s, .down_f32, .{ .x = 0, .stride = 64 }, la, p, 0, 1, -1));
    try std.testing.expectError(error.OptionNotBuilt, h.decode(s, .up, .{ .x = 0, .stride = 64 }, la, p, 0, 1, 3));
}
