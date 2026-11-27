//! MXFP4 experts (experts_mxfp4.cu): stored E2M1 codes and E8M0 scales decoded to bf16, experts_bf16's chain.
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

pub const Mxfp4Experts = struct {
    mod: Module,
    fns: [2][3]Function, // decode, prompt; up (SiTU), down_f32, down_bf16
    resident: [3]usize, // decode blocks the grid may hold: per SM times SMs

    pub const cols = 32; // output columns a block
    pub const warps = 4;
    /// E8M0 codes whose E2M1 multiples are all normal and finite in bf16: 0.5 * 2^(c - 127) to 6 * 2^(c - 127).
    pub const scale_min = 2;
    pub const scale_max = 252;

    pub const symbols = struct {
        pub const decode = [3][:0]const u8{ "tf_mxfp4_experts_up_situ", "tf_mxfp4_experts_down_f32", "tf_mxfp4_experts_down_bf16" };
        pub const prompt = [3][:0]const u8{ "tf_mxfp4_experts_prompt_up_situ", "tf_mxfp4_experts_prompt_down_f32", "tf_mxfp4_experts_prompt_down_bf16" };
    };

    pub fn load(d: *const driver.Driver, sms: usize) !Mxfp4Experts {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.experts_mxfp4);
        errdefer mod.unload();
        var e: Mxfp4Experts = .{ .mod = mod, .fns = undefined, .resident = undefined };
        for (symbols.decode, symbols.prompt, 0..) |dn, pn, i| {
            e.fns[0][i] = try mod.function(dn);
            e.fns[1][i] = try mod.function(pn);
            e.resident[i] = @max(1, try e.fns[0][i].occupancy(warps * 32, 0)) * sms;
        }
        return e;
    }

    pub fn unload(e: *Mxfp4Experts) void {
        e.mod.unload();
    }

    /// An up (SiTU on a stored gate and up, both betas set) or a down: 16 pairs an item in decode, 64 in a prompt.
    pub fn run(e: *const Mxfp4Experts, s: Stream, prompt: bool, epi: Epi, in: Rows, l: Layer, p: Plan, out: u64, items: usize, skip: i32) !void {
        const up = epi == .up;
        if (up and !(l.situ[0] > 0 and l.situ[1] > 0)) return error.EpilogueNotBuilt;
        const k = if (up) l.dims else l.width; // inputs, in whole 32-value scale groups
        const n = if (up) l.width else l.dims;
        if (k % 32 != 0 or n % cols != 0) return error.UnexpectedShape;
        if ((if (up) l.up_scale else l.down_scale) == 0) return error.MissingScales;
        const ei = @backingInt(epi);
        const nb = n / cols;
        const units = items * nb;
        const grid = if (prompt) (units + warps - 1) / warps else @min((units + warps - 1) / warps, e.resident[ei]);
        if (grid < 1) return;
        var a: launch_.Args = .{};
        a.add(in.x);
        a.add(@as(i32, @intCast(in.stride)));
        a.add(@as(i32, @intCast(if (up) in.slots else 0))); // an up reads token rows, a down its pair's row
        a.add(if (up) l.up else l.down);
        a.add(if (up) l.up_scale else l.down_scale);
        a.add(@as(i32, @intCast(k / 32)));
        a.add(@as(i32, @intCast(nb)));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(n)));
        a.add(skip);
        if (up) for (l.situ) |v| a.add(v);
        try launch_.launch(e.fns[@intFromBool(prompt)][ei], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = warps * 32, .y = 1, .z = 1 } }, s, &a);
    }

    /// Bytes of `count` matrices' codes, then of their scales, for n outputs from k inputs ([n][k/2] and [n][k/32]).
    pub fn codeBytes(count: usize, n: usize, k: usize) usize {
        return count * n * (k / 2);
    }

    pub fn scaleBytes(count: usize, n: usize, k: usize) usize {
        return count * n * (k / 32);
    }

    /// A checkpoint's scales, refused unless every code is in [scale_min, scale_max] (255 is E8M0's NaN).
    pub fn checkScales(scales: []const u8) !void {
        for (scales) |c| if (c < scale_min or c > scale_max) return error.Mxfp4ScaleRange;
    }
};
