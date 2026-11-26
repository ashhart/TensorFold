// The GDN contract's device layer: the fixture's buffers, the uploads and the launches.
const std = @import("std");
const vk = @import("vulkan");
const contract = @import("contract");
const fx = @import("stage4_fixture.zig");
const refs = @import("contract_gdn_ref.zig");
const host = @import("contract_gdn_host.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

const Which = refs.Which;
const heads = refs.heads;
const dk = refs.dk;
const linear_dim = refs.linear_dim;
const channels = refs.channels;
const taps = refs.taps;
const state_elems = refs.state_elems;
const chunk_fixed = refs.chunk_fixed;
const timeout_ns = refs.timeout_ns;
const spec = refs.spec;
const walk = refs.walk;
const decodeGates = refs.decodeGates;
const f32Of = refs.f32Of;
const bf = refs.bf;
const recurYDepth = refs.recurYDepth;
const recurStateDepth = refs.recurStateDepth;
const decodeYDepth = refs.decodeYDepth;
const decodeStateDepth = refs.decodeStateDepth;
const snapDepth = refs.snapDepth;
const Rig = refs.Rig;
const poison = refs.poison;
const putF32 = refs.putF32;
const PrefillInput = refs.PrefillInput;
const DecodeInput = refs.DecodeInput;

pub const Gdn = struct {
    a: std.mem.Allocator,
    dev: *const vk.Device,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    m: *vk.Gdn,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    tokens: usize = 0,
    chunks: usize = 1,
    max_state: f64 = 0,

    // host inputs (the uploaded bits, kept for the references)
    conv_out_bits: []u16 = &.{},
    a_bits: []u16 = &.{},
    b_bits: []u16 = &.{},
    dt_bits: []u16 = &.{},
    a_log_bits: []u16 = &.{},
    state_bits: []u16 = &.{}, // conv state (bf16) or recurrence state (f32 pairs)
    state_init_bits: []u16 = &.{}, // the decode loops' uploaded initial state, never written
    state_swap_bits: []u16 = &.{}, // poison for the decode loops' scratch carry buffers
    qkv_bits: []u16 = &.{}, // conv_prefill projection rows; conv_decode/decode x rows
    weight_bits: []u16 = &.{},
    soa_bits: []u16 = &.{},
    g_pairs: []u16 = &.{},
    beta_bits: []u16 = &.{},
    h0_bits: []u16 = &.{},
    walk_st: []f64 = &.{}, // the walker's state scratch

    // host outputs and the contract bookkeeping
    out: []u16 = &.{},
    out_words: usize = 0, // out's data width in u16 words (f32 outputs take two a word)
    want: []f64 = &.{},
    mag: []f64 = &.{},
    input_snapshot: []u8 = &.{},
    input_now: []u8 = &.{},
    bufs: [16]vk.Buffer = undefined,
    made: usize = 0,
    /// The decode loops' carry buffer holding the final state after run().
    final_state_idx: usize = 0,
    rig: Rig = .{},

    pub fn init(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.Gdn, which: Which) Gdn {
        return .{ .a = a, .dev = dev, .cmds = cmds, .staging = staging, .m = m, .which = which, .arena = .init(a) };
    }

    pub fn deinit(f: *Gdn) void {
        f.reset();
        f.arena.deinit();
    }

    pub fn reset(f: *Gdn) void {
        while (f.made > 0) {
            f.made -= 1;
            f.bufs[f.made].deinit();
        }
        _ = f.arena.reset(.retain_capacity);
        f.conv_out_bits = &.{};
        f.a_bits = &.{};
        f.b_bits = &.{};
        f.dt_bits = &.{};
        f.a_log_bits = &.{};
        f.state_bits = &.{};
        f.state_init_bits = &.{};
        f.state_swap_bits = &.{};
        f.qkv_bits = &.{};
        f.weight_bits = &.{};
        f.soa_bits = &.{};
        f.g_pairs = &.{};
        f.beta_bits = &.{};
        f.h0_bits = &.{};
        f.walk_st = &.{};
        f.out = &.{};
        f.want = &.{};
        f.mag = &.{};
        f.input_snapshot = &.{};
        f.input_now = &.{};
        f.rig = .{};
        f.shape = .{ .rows = 0, .n = 0, .k = 0 };
        f.tokens = 0;
        f.chunks = 1;
        f.max_state = 0;
    }

    const vt = g.hooks(Gdn);
    pub fn kernel(f: *Gdn) g.Kernel {
        return .{ .spec = spec(f.which), .ctx = f, .vt = &vt };
    }

    pub fn allocU16(f: *Gdn, n: usize, fill: ?u16) ![]u16 {
        const v = try f.arena.allocator().alloc(u16, n);
        if (fill) |p| @memset(v, p) else for (v) |*x| x.* = 0;
        return v;
    }

    pub fn allocF64(f: *Gdn, n: usize) ![]f64 {
        return f.arena.allocator().alloc(f64, n);
    }

    // `devBuf` uploads input `bytes` or poisons output `poison_bytes`, registers cleanup, and returns its index.
    fn devBuf(f: *Gdn, bytes: []const u8) !usize {
        const b = try vk.Buffer.init(f.dev, .device, bytes.len);
        f.bufs[f.made] = b;
        f.made += 1;
        try vk.orExit(vk.upload(f.cmds, f.bufs[f.made - 1], 0, f.staging, bytes, timeout_ns));
        return f.made - 1;
    }

    fn view(f: *Gdn, idx: usize, kind: vk.tensor.Kind, elements: u64) !vk.tensor.View {
        return vk.tensor.View.init(&f.bufs[idx], kind, 0, elements);
    }

    pub fn prepare(f: *Gdn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const out_len = (try f.prepareHost(shape, fill, seed)) orelse return false;
        const a = f.arena.allocator();

        // Fixed per-`which` buffers upload input bits and poisoned output regions or scratch.
        var input_slices: [8][]u16 = undefined;
        const inputs = f.inputList(&input_slices);
        var input_total: usize = 0;
        for (inputs) |s| input_total += std.mem.sliceAsBytes(s).len;
        try f.uploadBuffers();
        f.input_snapshot = try a.alloc(u8, input_total);
        f.input_now = try a.alloc(u8, input_total);
        var pos: usize = 0;
        for (inputs) |s| {
            const bytes = std.mem.sliceAsBytes(s);
            @memcpy(f.input_snapshot[pos..][0..bytes.len], bytes);
            pos += bytes.len;
        }
        f.rig = .{ .output = std.mem.sliceAsBytes(f.out[0..out_len]), .inputs = f.input_snapshot, .guards_ok = true };
        return true;
    }

    // The host fills and the upload order live in contract_gdn_host.zig; these keep the fixture's own call sites.
    pub fn prepareHost(f: *Gdn, shape: c.Shape, fill: g.Fill, seed: u64) !?usize {
        return host.prepareHost(f, shape, fill, seed);
    }

    pub fn inputList(f: *Gdn, buf: *[8][]u16) []const []u16 {
        return host.inputList(f, buf);
    }

    pub fn uploadWords(f: *Gdn, buf: *[16][]u16) !usize {
        return host.uploadWords(f, buf);
    }

    // `uploadWords`/`run` share indices; `out` poison catches dead stores, and other outputs use fresh scratch.
    fn uploadBuffers(f: *Gdn) !void {
        var words: [16][]u16 = undefined;
        const n = try f.uploadWords(&words);
        for (words[0..n]) |w| _ = try f.devBuf(std.mem.sliceAsBytes(w));
    }

    pub fn run(f: *Gdn) !void {
        const T = f.tokens;
        switch (f.which) {
            .split_soa, .split_g => {
                const conv_out = try f.view(0, .bf16, @intCast(T * channels));
                const a_v = try f.view(1, .bf16, @intCast(T * heads));
                const b_v = try f.view(2, .bf16, @intCast(T * heads));
                const dt_v = try f.view(3, .bf16, heads);
                const a_log_v = try f.view(4, .f32, heads);
                const soa_v = try f.view(5, .bf16, @intCast(3 * T * linear_dim));
                const g_v = try f.view(6, .f32, @intCast(T * heads));
                const beta_v = try f.view(7, .bf16, @intCast(T * heads));
                try vk.orExit(f.m.split(f.cmds, soa_v, g_v, beta_v, conv_out, a_v, b_v, a_log_v, dt_v, T, timeout_ns));
            },
            .conv_prefill => {
                const state_v = try f.view(0, .bf16, @intCast(3 * channels));
                const qkv_v = try f.view(1, .bf16, @intCast(T * channels));
                const weight_v = try f.view(2, .bf16, @intCast(taps * channels));
                const a_v = try f.view(3, .bf16, @intCast(T * heads));
                const b_v = try f.view(4, .bf16, @intCast(T * heads));
                const dt_v = try f.view(5, .bf16, heads);
                const a_log_v = try f.view(6, .f32, heads);
                const conv_input_v = try f.view(7, .bf16, @intCast((T + 3) * channels));
                const conv_out_v = try f.view(8, .bf16, @intCast(T * channels));
                const soa_v = try f.view(9, .bf16, @intCast(3 * T * linear_dim));
                const g_v = try f.view(10, .f32, @intCast(T * heads));
                const beta_v = try f.view(11, .bf16, @intCast(T * heads));
                const state_out_v = try f.view(12, .bf16, @intCast(3 * channels));
                try vk.orExit(f.m.convPrefill(f.cmds, soa_v, g_v, beta_v, conv_out_v, conv_input_v, state_v, qkv_v, weight_v, a_v, b_v, a_log_v, dt_v, state_out_v, T, timeout_ns));
            },
            .conv_decode => {
                const x_v = try f.view(3, .bf16, @intCast(T * channels));
                const weight_v = try f.view(4, .bf16, @intCast(taps * channels));
                const a_v = try f.view(5, .bf16, @intCast(T * heads));
                const b_v = try f.view(6, .bf16, @intCast(T * heads));
                const dt_v = try f.view(7, .bf16, heads);
                const a_log_v = try f.view(8, .f32, heads);
                const act_v = try f.view(9, .bf16, @intCast(T * channels));
                const soa_v = try f.view(10, .bf16, @intCast(3 * T * linear_dim));
                const g_v = try f.view(11, .f32, @intCast(T * heads));
                const beta_v = try f.view(12, .bf16, @intCast(T * heads));
                var in_idx: usize = 0; // state_init; carry buffers are 1 and 2
                for (0..T) |t| {
                    const out_idx: usize = if (in_idx == 1) 2 else 1;
                    const st_in = try f.view(in_idx, .bf16, @intCast(3 * channels));
                    const st_out = try f.view(out_idx, .bf16, @intCast(3 * channels));
                    try vk.orExit(f.m.convDecodeStep(f.cmds, try soa_v.slice(t * 3 * linear_dim, 3 * linear_dim), try g_v.slice(t * heads, heads), try beta_v.slice(t * heads, heads), try act_v.slice(t * channels, channels), st_in, try x_v.slice(t * channels, channels), weight_v, try a_v.slice(t * heads, heads), try b_v.slice(t * heads, heads), a_log_v, dt_v, st_out, timeout_ns));
                    in_idx = out_idx;
                }
                f.final_state_idx = in_idx; // the initial buffer at 0 was never written
            },
            .prefill_y, .prefill_state => {
                const soa_v = try f.view(0, .bf16, @intCast(3 * T * linear_dim));
                const g_v = try f.view(1, .f32, @intCast(T * heads));
                const beta_v = try f.view(2, .bf16, @intCast(T * heads));
                const h0_v = try f.view(3, .f32, state_elems);
                const y_v = try f.view(4, .bf16, @intCast(T * linear_dim));
                const hf_v = try f.view(5, .f32, state_elems);
                const snap_v: ?vk.tensor.View = if (f.chunks > 1) try f.view(6, .f32, state_elems * (f.chunks - 1)) else null;
                try vk.orExit(f.m.prefillRecur(f.cmds, soa_v, g_v, beta_v, h0_v, y_v, hf_v, snap_v, T, chunk_fixed, timeout_ns));
            },
            .decode_y, .decode_state => {
                const a_v = try f.view(1, .bf16, @intCast(T * heads));
                const b_v = try f.view(2, .bf16, @intCast(T * heads));
                const a_log_v = try f.view(3, .u16, 2 * heads);
                const dt_v = try f.view(4, .bf16, heads);
                const y_v = try f.view(8, .bf16, @intCast(T * linear_dim));
                var in_idx: usize = 5; // state_init
                for (0..T) |t| {
                    const st_in = try f.view(in_idx, .f32, state_elems);
                    const out_idx: usize = if (in_idx == 5) 6 else if (in_idx == 6) 7 else 6;
                    const st_out = try f.view(out_idx, .f32, state_elems);
                    try vk.orExit(f.m.decodeRecur(f.cmds, try f.soaView(t), try a_v.slice(t * heads, heads), try b_v.slice(t * heads, heads), a_log_v, dt_v, st_in, st_out, try y_v.slice(t * linear_dim, linear_dim), timeout_ns));
                    in_idx = out_idx;
                }
                f.final_state_idx = in_idx;
            },
        }
        try f.fetchOutputs();
    }

    fn soaView(f: *Gdn, t: usize) !vk.tensor.View {
        const soa_v = try f.view(0, .bf16, @intCast(3 * f.tokens * linear_dim));
        return soa_v.slice(t * 3 * linear_dim, 3 * linear_dim);
    }

    // Fetch outputs, reread inputs, verify guard words; downloads use orExit so a fence Timeout stays a FAIL verdict.
    fn fetchOutputs(f: *Gdn) !void {
        const T = f.tokens;
        switch (f.which) {
            .split_soa => {
                try vk.orExit(vk.download(f.cmds, f.bufs[5], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. 3 * T * linear_dim]), timeout_ns));
                try vk.orExit(vk.download(f.cmds, f.bufs[7], 0, f.staging, std.mem.sliceAsBytes(f.out[3 * T * linear_dim .. 3 * T * linear_dim + T * heads]), timeout_ns));
            },
            .split_g => {
                try vk.orExit(vk.download(f.cmds, f.bufs[6], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. 2 * T * heads]), timeout_ns));
            },
            .conv_prefill => {
                try vk.orExit(vk.download(f.cmds, f.bufs[8], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. T * channels]), timeout_ns));
                try vk.orExit(vk.download(f.cmds, f.bufs[12], 0, f.staging, std.mem.sliceAsBytes(f.out[T * channels .. T * channels + 3 * channels]), timeout_ns));
            },
            .conv_decode => {
                try vk.orExit(vk.download(f.cmds, f.bufs[9], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. T * channels]), timeout_ns));
                try vk.orExit(vk.download(f.cmds, f.bufs[f.final_state_idx], 0, f.staging, std.mem.sliceAsBytes(f.out[T * channels .. T * channels + 3 * channels]), timeout_ns));
            },
            .prefill_y => {
                try vk.orExit(vk.download(f.cmds, f.bufs[4], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. T * linear_dim]), timeout_ns));
            },
            .prefill_state => {
                try vk.orExit(vk.download(f.cmds, f.bufs[5], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. 2 * state_elems]), timeout_ns));
                if (f.chunks > 1) try vk.orExit(vk.download(f.cmds, f.bufs[6], 0, f.staging, std.mem.sliceAsBytes(f.out[2 * state_elems .. 2 * state_elems * f.chunks]), timeout_ns));
            },
            .decode_y => {
                try vk.orExit(vk.download(f.cmds, f.bufs[8], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. T * linear_dim]), timeout_ns));
            },
            .decode_state => {
                try vk.orExit(vk.download(f.cmds, f.bufs[f.final_state_idx], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. 2 * state_elems]), timeout_ns));
            },
        }
        // inputs, post-run, in the snapshot's order
        var slices: [8][]u16 = undefined;
        const inputs = f.inputList(&slices);
        var idxs: [8]usize = undefined;
        const dev_idx = host.inputBufIndices(f, &idxs);
        var pos: usize = 0;
        for (inputs, dev_idx) |s, di| {
            try vk.orExit(vk.download(f.cmds, f.bufs[di], 0, f.staging, std.mem.sliceAsBytes(s), timeout_ns));
            const bytes = std.mem.sliceAsBytes(s);
            @memcpy(f.input_now[pos..][0..bytes.len], bytes);
            pos += bytes.len;
        }
        f.rig.inputs = f.input_now;
        f.rig.guards_ok = true;
        for (f.out[f.out_words..]) |w| if (w != poison) {
            f.rig.guards_ok = false;
            break;
        };
    }

    // Gate 4 replays each dot in `accum`/`order`, sums other ops in f64, and writes `out` in `reference()` layout.
    pub fn simulate(f: *Gdn, accum: c.Accum, order: g.Order, out: []f64) !void {
        const T = f.tokens;
        switch (f.which) {
            .split_soa => {
                var terms: [dk]f64 = undefined;
                for (0..T) |t| {
                    for (0..3 * heads) |head| {
                        const hv: usize = head % heads;
                        var vals: [dk]f64 = undefined;
                        var vmax: f64 = 0;
                        for (0..dk) |i| {
                            const cv = bf(f.conv_out_bits[t * channels + head * dk + i]);
                            vals[i] = cv / (1.0 + @exp(-cv));
                            vmax = @max(vmax, @abs(vals[i]));
                        }
                        for (0..dk) |i| terms[i] = vals[i] * vals[i];
                        const ss = rows.sumIn(accum, order, &terms);
                        const inv: f64 = 1.0 / @sqrt(ss / 128.0 + 1e-6 / 128.0);
                        const scale_f: f64 = if (head < heads) 1.0 / 128.0 else @sqrt(1.0 / 128.0);
                        const dst = (head / heads) * (T * linear_dim) + (t * heads + hv) * dk;
                        for (0..dk) |i| out[dst + i] = if (head < 2 * heads) scale_f * (vals[i] * inv) else vals[i];
                    }
                    for (0..heads) |hv| {
                        const ref = fx.gateRefOracle(f.a_bits[t * heads + hv], f.b_bits[t * heads + hv], f32Of(f.a_log_bits, hv), f.dt_bits[hv]);
                        out[3 * T * linear_dim + t * heads + hv] = ref.beta;
                    }
                }
            },
            .split_g => return error.Unimplemented, // accum none; gate 4 is N/A
            .conv_prefill => {
                const ctx: PrefillInput = .{ .state_bits = f.state_bits, .qkv_bits = f.qkv_bits };
                var terms: [taps]f64 = undefined;
                for (0..channels) |ch| {
                    for (0..T) |t| {
                        for (0..taps) |k| terms[k] = bf(ctx.inWord(t + k, ch)) * bf(f.weight_bits[ch * taps + k]);
                        out[t * channels + ch] = rows.sumIn(accum, order, &terms);
                    }
                    for (0..3) |j| out[T * channels + j * channels + ch] = bf(ctx.inWord(T + j, ch));
                }
            },
            .conv_decode => {
                // the chain replays from the UPLOADED initial state
                var terms: [taps]f64 = undefined;
                const chain_a = try f.allocU16(3 * channels, 0);
                const chain_b = try f.allocU16(3 * channels, 0);
                @memcpy(chain_a, f.state_init_bits);
                var chain: []u16 = chain_a;
                for (0..T) |t| {
                    const ctx: DecodeInput = .{ .state_bits = chain, .x_bits = f.qkv_bits, .t = t };
                    for (0..channels) |ch| {
                        for (0..taps) |k| terms[k] = bf(ctx.inWord(t + k, ch)) * bf(f.weight_bits[ch * taps + k]);
                        out[t * channels + ch] = rows.sumIn(accum, order, &terms);
                    }
                    const nxt: []u16 = if (chain.ptr == chain_a.ptr) chain_b else chain_a;
                    for (0..2) |r| @memcpy(nxt[r * channels ..][0..channels], chain[(r + 1) * channels ..][0..channels]);
                    @memcpy(nxt[2 * channels ..][0..channels], f.qkv_bits[t * channels ..][0..channels]);
                    chain = nxt;
                }
                for (0..3 * channels) |i| out[T * channels + i] = bf(chain[i]);
            },
            .prefill_y => walk(T, f.soa_bits, f.g_pairs, f.beta_bits, f.h0_bits, f.walk_st, out, null, &f.max_state, null, null, accum, order),
            .prefill_state => {
                const snap_chunk: ?usize = if (f.chunks > 1) @intCast(chunk_fixed) else null;
                walk(T, f.soa_bits, f.g_pairs, f.beta_bits, f.h0_bits, f.walk_st, null, out[0..state_elems], &f.max_state, snap_chunk, if (f.chunks > 1) out[state_elems..] else null, accum, order);
            },
            .decode_y, .decode_state => {
                const g_t = try f.allocU16(2 * heads, 0);
                const beta_t = try f.allocU16(heads, 0);
                const y_t: ?[]f64 = if (f.which == .decode_y) out else null;
                const st_scratch = try f.allocF64(state_elems);
                const st_out = try f.allocF64(state_elems);
                const bits = try f.allocU16(2 * state_elems, 0);
                @memcpy(bits, f.state_init_bits);
                for (0..T) |t| {
                    decodeGates(f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, t, g_t, beta_t);
                    walk(1, f.soa_bits[t * 3 * linear_dim ..][0 .. 3 * linear_dim], g_t, beta_t, bits, st_scratch, if (y_t) |y| y[t * linear_dim ..][0..linear_dim] else null, st_out, &f.max_state, null, null, accum, order);
                    if (f.which == .decode_state) @memcpy(out[0..state_elems], st_out);
                    for (0..state_elems) |i| putF32(bits, i, @floatCast(st_out[i]));
                }
            },
        }
    }

    // Bounds use stage 4 depth/ops/mids and magnitude scales: `max_state`, row `vmax^2`, `|acc|`, and `|want|`.
    pub fn bounds(f: *Gdn, want: []const f64, mag: []const f64, allow: []f64) !void {
        switch (f.which) {
            .split_soa => for (want, mag, allow) |w, m, *l| {
                l.* = fx.budget(40, 5, 3, w, m);
            },
            .split_g => for (want, mag, allow) |w, m, *l| {
                l.* = fx.budget(1, 4, 2, w, m);
            },
            .conv_prefill, .conv_decode => for (want, mag, allow) |w, m, *l| {
                l.* = fx.budget(taps, 0, 0, w, m);
            },
            .prefill_y => for (want, mag, allow) |w, m, *l| {
                l.* = fx.budget(@floatFromInt(recurYDepth(f.shape)), 4, 0, w, m);
            },
            .prefill_state => for (want, mag, 0..) |w, m, i| {
                allow[i] = fx.budget(if (i < state_elems) @floatFromInt(recurStateDepth(f.shape)) else snapDepth(), 2, 0, w, m);
            },
            .decode_y => for (want, mag, allow) |w, m, *l| {
                l.* = fx.budget(@floatFromInt(decodeYDepth(f.shape)), 4, 0, w, m);
            },
            .decode_state => for (want, mag, allow) |w, m, *l| {
                l.* = fx.budget(@floatFromInt(decodeStateDepth(f.shape)), 2, 0, w, m);
            },
        }
    }

    pub fn count(f: *Gdn) usize {
        return f.want.len;
    }

    pub fn reference(f: *Gdn, want: []f64, mag: []f64) !void {
        @memcpy(want, f.want);
        @memcpy(mag, f.mag);
    }
};
