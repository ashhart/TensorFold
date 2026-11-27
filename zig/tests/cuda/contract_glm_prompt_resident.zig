//! GLM TP=2 routed prompt R3: production runPrompt dispatch against the original prompt entries, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const fx = cuda.experts.Nvfp4Experts;

pub const routed_experts = 288;
pub const plan_experts = routed_experts + 1;
pub const shared_expert = routed_experts;
pub const slots = 9;
pub const tail_bytes = 64;
const materialized_experts = 4; // only ids 0..3 are picked; the plan still spans all 288 routed ids and shared id 288
const prompt_tile = cuda.experts.promptTile(.nvfp4);

pub const Form = enum { up, down, up_nonfinite, down_corrupt };

/// Host-inspectable registration: the dispatched candidate and its baseline, including the planted corruption.
pub const Entry = struct {
    form: Form,
    candidate: [:0]const u8,
    baseline: [:0]const u8,
    control: bool = false,
};

pub const inventory = [_]Entry{
    .{ .form = .up, .candidate = fx.symbols.resident_prompt[0], .baseline = fx.symbols.prompt[1] },
    .{ .form = .down, .candidate = fx.symbols.resident_prompt[1], .baseline = fx.symbols.prompt[2] },
    .{ .form = .up_nonfinite, .candidate = fx.symbols.resident_prompt[0], .baseline = fx.symbols.prompt[1] },
    .{ .form = .down_corrupt, .candidate = fx.symbols.resident_prompt[1], .baseline = fx.symbols.prompt[2], .control = true },
};

pub const up_served = [_]c.Shape{
    .{ .rows = 1, .n = 1024, .k = 4096 },
    .{ .rows = 2, .n = 1024, .k = 4096 },
    .{ .rows = 17, .n = 1024, .k = 4096 },
    .{ .rows = 64, .n = 1024, .k = 4096 },
};
pub const down_served = [_]c.Shape{
    .{ .rows = 1, .n = 4096, .k = 1024 },
    .{ .rows = 2, .n = 4096, .k = 1024 },
    .{ .rows = 17, .n = 4096, .k = 1024 },
    .{ .rows = 64, .n = 4096, .k = 1024 },
};

fn isDown(form: Form) bool {
    return form == .down or form == .down_corrupt;
}

fn noSums(_: c.Shape) usize {
    return 0;
}

pub fn spec(entry: Entry) c.Spec {
    const name = switch (entry.form) {
        .up => "cuda GLM TP=2 NVFP4 prompt R3 SwiGLU tf_nvfp4_prompt3_swiglu",
        .down => "cuda GLM TP=2 NVFP4 prompt R3 down tf_nvfp4_prompt3_down",
        .up_nonfinite => "cuda GLM TP=2 NVFP4 prompt R3 SwiGLU nonfinite",
        .down_corrupt => "control: GLM TP=2 NVFP4 prompt R3 down output byte changed",
    };
    return .{
        .name = name,
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .replaces = entry.baseline,
        .nan = .{ .stated = "Every R3 output byte equals prompt_r3=false, including NaN/Inf payloads, shared slots and the unwritten tail, all starting at 0xff" },
        .served = if (isDown(entry.form)) &down_served else &up_served,
        .sums = .{ .why = "Replacement-only byte equality through Nvfp4Experts.runPrompt" },
    };
}

pub const Prompt = struct {
    rig: rig_.Rig,
    ex: *const fx,
    router: cuda.grouped.Router,
    entry: Entry,
    shape: c.Shape = undefined,
    layer: cuda.experts.Layer = undefined,
    plan: cuda.grouped.Plan = undefined,
    x: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    msg: [192]u8 = undefined,

    pub fn kernel(self: *Prompt) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.control) .shapes else null };
    }

    const vt = g.hooks(Prompt);

    pub fn deinit(self: *Prompt) void {
        self.rig.deinit();
    }

    fn pairs(self: *const Prompt) usize {
        return self.shape.rows * slots;
    }

    fn payloadBytes(self: *const Prompt) usize {
        return self.pairs() * self.shape.n * 2;
    }

    fn outBytes(self: *const Prompt) usize {
        return self.payloadBytes() + tail_bytes;
    }

    pub fn count(self: *Prompt) usize {
        return self.outBytes();
    }

    pub fn prepare(self: *Prompt, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const up = !isDown(self.entry.form);
        const rows_ok = shape.rows >= 1 and shape.rows <= 64;
        const dims_ok = if (up) shape.n == 1024 and shape.k == 4096 else shape.n == 4096 and shape.k == 1024;
        if (!rows_ok or !dims_ok) return false;
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nonfinite = self.entry.form == .up_nonfinite or fill == .nonfinite;

        const x_rows = if (up) shape.rows else self.pairs();
        const x = try a.alloc(u16, x_rows * shape.k);
        defer a.free(x);
        for (x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (nonfinite) for (0..x_rows) |row| for ([_]u16{ 0x7fc1, 0xffc2, 0x7f80, 0xff80 }, 0..) |bits, i| {
            x[row * shape.k + 31 + i * 32] = bits;
        };

        // Loader layout directly: random finite E2M1 codes and finite e4m3 scales in every [2][144]-word part.
        const parts: usize = if (up) 2 else 1;
        const weights = try a.alloc(u8, fx.bytes(materialized_experts, shape.n, shape.k, parts));
        defer a.free(weights);
        var off: usize = 0;
        while (off < weights.len) : (off += fx.words * 4) {
            rng.bytes(weights[off..][0..512]);
            @memset(weights[off + 512 ..][0..64], 0x20);
        }
        if (nonfinite) weights[512] = 0x7f; // a NaN block scale in expert 0's first part

        const globals = try a.alloc(f32, materialized_experts * parts);
        defer a.free(globals);
        for (globals) |*v| v.* = 0.005 + rng.float(f32) * 0.03;
        if (nonfinite) {
            globals[0] = std.math.nan(f32);
            if (globals.len > 1) globals[1] = std.math.inf(f32);
        }

        const picks = try a.alloc(i32, self.pairs());
        defer a.free(picks);
        for (0..shape.rows) |row| for (0..slots) |slot| {
            const at = row * slots + slot;
            picks[at] = if (slot == slots - 1)
                shared_expert
            else if ((row + slot) % 3 == 0)
                @intCast(1 + (row * 7 + slot) % (materialized_experts - 1))
            else
                0; // a hot expert gives full 64-pair items plus ragged remainders; other experts stay partial
        };

        self.x = try self.rig.input(std.mem.sliceAsBytes(x));
        const wd = try self.rig.input(weights);
        const sd = try self.rig.input(std.mem.sliceAsBytes(globals));
        self.layer = .{
            .format = .nvfp4,
            .up = if (up) wd else 0,
            .down = if (up) 0 else wd,
            .up_scale = if (up) sd else 0,
            .down_scale = if (up) 0 else sd,
            .width = 1024,
            .dims = 4096,
            .experts = routed_experts,
            .limit = 10,
        };

        const pair_count = self.pairs();
        const items = cuda.grouped.maxItems(pair_count, plan_experts, prompt_tile);
        const ids = try self.rig.input(std.mem.sliceAsBytes(picks));
        self.plan = .{
            .members = try self.rig.scratch(pair_count * 4),
            .items = try self.rig.scratch(items * 12),
            .counts = try self.rig.scratch(8),
            .rank = try self.rig.scratch(pair_count * 4),
            .hist = try self.rig.scratch(((pair_count + 1023) / 1024) * plan_experts * 4),
        };
        try self.router.route(self.rig.s, ids, pair_count, plan_experts, prompt_tile, self.plan);
        self.out = try self.rig.output(self.outBytes());
        self.base = try self.rig.scratch(self.outBytes());
        try self.rig.s.synchronize();
        return true;
    }

    fn poison(self: *Prompt, ptr: u64) !void {
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = ptr, .len = self.outBytes() };
        try b.fill8(0xff, self.rig.s.handle);
    }

    fn launch(self: *Prompt, r3: bool, target: u64) !void {
        const up = !isDown(self.entry.form);
        const input: cuda.experts.Rows = .{ .x = self.x, .stride = self.shape.k, .slots = if (up) slots else 0 };
        const epi: cuda.experts.Epi = if (up) .up else .down_bf16;
        const items = cuda.grouped.maxItems(self.pairs(), plan_experts, prompt_tile);
        var ex = self.ex.*;
        ex.prompt_r3 = r3;
        try self.poison(target);
        try ex.runPrompt(self.rig.s, epi, input, self.layer, self.plan, target, items, shared_expert);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Prompt) !void {
        try self.launch(true, self.out); // the candidate is never the production default path
        if (self.entry.control) {
            const bytes = try self.rig.outputBytes();
            const changed = [_]u8{bytes[0] ^ 1};
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out, .len = self.outBytes() };
            try b.upload(0, &changed);
        }
    }

    pub fn replaced(self: *Prompt) ![]const u8 {
        try self.launch(false, self.base);
        return self.rig.bytes(self.base, self.outBytes());
    }

    fn untouched(self: *const Prompt, got: []const u8) ?usize {
        const row_bytes = self.shape.n * 2;
        for (0..self.shape.rows) |row| {
            const at = (row * slots + slots - 1) * row_bytes;
            for (got[at..][0..row_bytes], 0..) |v, i| if (v != 0xff) return at + i;
        }
        for (got[self.payloadBytes()..], 0..) |v, i| if (v != 0xff) return self.payloadBytes() + i;
        return null;
    }

    pub fn check(self: *Prompt, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from {s} with prompt_r3=false", .{ at, got.len, self.entry.baseline });
        if (self.untouched(got)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of a skipped shared slot or the unwritten tail changed", .{at});
        return null;
    }

    pub fn reference(_: *Prompt, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
};

pub const Suite = struct {
    ex: fx,
    plan_mod: cuda.Module,
    router: cuda.grouped.Router,
    cells: [inventory.len]Prompt,

    pub fn create(gpu: @import("check.zig").Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Suite {
        const a = gpu.gpa;
        const self = try a.create(Suite);
        errdefer a.destroy(self);
        const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
        self.ex = try fx.loadPromptResidency(gpu.d, sms, true);
        errdefer self.ex.unload();
        self.plan_mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
        errdefer self.plan_mod.unload();
        self.router = try cuda.grouped.Router.resolve(self.plan_mod);
        for (&self.cells, inventory) |*cell, entry| {
            cell.* = .{ .rig = .{ .gpu = gpu, .s = s }, .ex = &self.ex, .router = self.router, .entry = entry };
            try kernels.append(a, cell.kernel());
        }
        return self;
    }

    pub fn destroy(self: *Suite) void {
        const a = self.cells[0].rig.gpu.gpa;
        for (&self.cells) |*cell| cell.deinit();
        self.plan_mod.unload();
        self.ex.unload();
        a.destroy(self);
    }
};
