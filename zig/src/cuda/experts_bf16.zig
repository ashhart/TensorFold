//! bf16 experts (experts_bf16.cu): a checkpoint's bf16 weights in mma fragment blocks, relu^2 up, fp32 or bf16 down.
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const grouped = @import("grouped.zig");
const driver = @import("driver.zig");
const experts = @import("experts.zig");

const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const Plan = grouped.Plan;
const Epi = experts.Epi;
const Rows = experts.Rows;
const Layer = experts.Layer;

pub const Bf16Experts = struct {
    mod: Module,
    fns: [2][3]Function, // decode, prompt; by Epi
    resident: [3]usize, // decode blocks the grid may hold: per SM times SMs
    pack_fn: Function,

    pub const cols = 32; // output columns a block
    pub const warps = 4;
    /// Bytes of a (32 columns, 32 inputs) block.
    pub const block = 32 * 32 * 2;

    pub const symbols = struct {
        pub const decode = [3][:0]const u8{ "tf_bf16_experts_up", "tf_bf16_experts_down_f32", "tf_bf16_experts_down_bf16" };
        pub const prompt = [3][:0]const u8{ "tf_bf16_experts_prompt_up", "tf_bf16_experts_prompt_down_f32", "tf_bf16_experts_prompt_down_bf16" };
        pub const pack = "tf_bf16_experts_pack";
    };

    pub fn load(d: *const driver.Driver, sms: usize) !Bf16Experts {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.experts_bf16);
        errdefer mod.unload();
        var e: Bf16Experts = .{ .mod = mod, .fns = undefined, .resident = undefined, .pack_fn = try mod.function(symbols.pack) };
        for (symbols.decode, symbols.prompt, 0..) |dn, pn, i| {
            e.fns[0][i] = try mod.function(dn);
            e.fns[1][i] = try mod.function(pn);
            e.resident[i] = @max(1, try e.fns[0][i].occupancy(warps * 32, 0)) * sms;
        }
        return e;
    }

    pub fn unload(e: *Bf16Experts) void {
        e.mod.unload();
    }

    /// One call on a plan of at most `items` items: 16 pairs an item in decode, 64 in a prompt.
    pub fn run(e: *const Bf16Experts, s: Stream, prompt: bool, epi: Epi, in: Rows, l: Layer, p: Plan, out: u64, items: usize, skip: i32) !void {
        if (!l.relu2 or l.swiglu or l.limit != 0) return error.OptionNotBuilt;
        const up = epi == .up;
        const kg = (if (up) l.dims else l.width) / 32;
        const nb = (if (up) l.width else l.dims) / cols;
        const ei = @backingInt(epi);
        const units = items * nb;
        const grid = if (prompt) (units + warps - 1) / warps else @min((units + warps - 1) / warps, e.resident[ei]);
        if (grid < 1) return;
        var a: launch_.Args = .{};
        a.add(in.x);
        a.add(@as(i32, @intCast(in.stride)));
        a.add(@as(i32, @intCast(if (up) in.slots else 0)));
        a.add(if (up) l.up else l.down);
        a.add(@as(i32, @intCast(kg)));
        a.add(@as(i32, @intCast(nb)));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(if (up) l.width else l.dims)));
        a.add(skip);
        try launch_.launch(e.fns[@intFromBool(prompt)][ei], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = warps * 32, .y = 1, .z = 1 } }, s, &a);
    }

    /// Bytes of `count` experts' blocks for n outputs from k inputs.
    pub fn bytes(count: usize, n: usize, k: usize) usize {
        return count * (n / cols) * (k / 32) * block;
    }

    /// A source matrix: experts `stride` apart, rows of `ld`, its first output and input, [k][n] (x @ W) or [n][k].
    pub const Source = struct { stride: usize = 0, ld: usize, n0: usize = 0, k0: usize = 0, in_out: bool };

    /// `count` bf16 matrices of n outputs from k inputs at `src` into their blocks at `dst` (bytes(count, n, k)).
    pub fn pack(e: *const Bf16Experts, s: Stream, src: u64, from: Source, dst: u64, count: usize, n: usize, k: usize) !void {
        if (n % cols != 0 or k % 32 != 0) return error.UnexpectedShape;
        const per = bytes(1, n, k) / 16;
        var a: launch_.Args = .{};
        a.add(src);
        a.add(@as(i64, @intCast(from.stride)));
        for ([_]usize{ from.ld, from.n0, from.k0, @intFromBool(from.in_out), n, k }) |v| a.add(@as(i32, @intCast(v)));
        a.add(dst);
        try launch_.launch(e.pack_fn, .{ .grid = .{ .x = @intCast((per + 255) / 256), .y = @intCast(count), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }
};
