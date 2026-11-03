//! BF16 prompt tiles meet both the independent FP64 projection bound and the replaced lane's bytes.
const cuda = @import("cuda");
const contract = @import("core").contract;
const c = contract.spec;
const g = contract.gates;
const Product = @import("contract_projection_f32.zig").Product;

const served = [_]c.Shape{
    .{ .rows = 1024, .n = 48, .k = 5120 },
    .{ .rows = 1025, .n = 96, .k = 2048 },
    .{ .rows = 129, .n = 128, .k = 4096 },
    .{ .rows = 129, .n = 640, .k = 1024 },
    .{ .rows = 129, .n = 3072, .k = 512 },
    .{ .rows = 129, .n = 70, .k = 128 },
};

fn depth(shape: c.Shape) usize {
    return shape.k / 16 + 24;
}

pub const Tile = struct {
    pub const callbacks = g.hooks(Tile);
    product: Product,
    rig: *@import("contract_rig.zig").Rig = undefined,
    prompt: *const cuda.nvfp4.Prompt,
    replacement: bool = true,
    control: bool = false,
    base: u64 = 0,

    pub fn deinit(t: *Tile) void {
        t.product.deinit();
    }

    pub fn kernel(t: *Tile) g.Kernel {
        t.rig = &t.product.rig;
        return .{ .ctx = t, .vt = &callbacks, .control = if (t.control) .shapes else null, .spec = .{
            .name = if (t.control) (if (t.product.out_kind == .f32) "control: BF16 prompt fp32 scaled" else "control: BF16 prompt scaled") else if (t.replacement) (if (t.product.out_kind == .f32) "cuda BF16 prompt fp32 lane replacement" else "cuda BF16 prompt lane replacement") else if (t.product.out_kind == .f32) "cuda BF16 prompt fp32 FP64" else "cuda BF16 prompt FP64",
            .accum = .f32,
            .depth = depth,
            .out = if (t.product.out_kind == .f32) .f32 else .bf16,
            .replaces = if (t.replacement) "qmmf BF16 lane with shape-fixed K slices" else null,
            .tiles = .{ .{ .size = 128 }, .{ .size = 128, .min = 1 }, .{ .size = 128, .step = 128, .min = 128 } },
            .served = &served,
        } };
    }

    pub fn prepare(t: *Tile, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try t.product.prepare(shape, fill, seed)) return false;
        t.base = try t.product.rig.scratch(t.count() * t.width());
        return true;
    }

    fn width(t: *const Tile) usize {
        return if (t.product.out_kind == .f32) 4 else 2;
    }

    pub fn run(t: *Tile) !void {
        const p = &t.product;
        const sh = p.d.?.shape;
        var w = p.weight;
        if (t.control) w.scale *= 1.5;
        try t.prompt.matmulOut(p.rig.s, p.x, sh.k, sh.rows, w, p.out, p.out_kind);
        try p.rig.s.synchronize();
    }

    pub fn replaced(t: *Tile) ![]const u8 {
        const p = &t.product;
        const sh = p.d.?.shape;
        try p.lane.matmulOut(p.rig.s, p.x, sh.k, sh.rows, p.weight, t.base, p.out_kind, p.part);
        return p.rig.bytes(t.base, t.count() * t.width());
    }

    pub fn count(t: *Tile) usize {
        return t.product.count();
    }

    pub fn reference(t: *Tile, want: []f64, mag: []f64) !void {
        return t.product.reference(want, mag);
    }

    pub fn simulate(t: *Tile, acc: c.Accum, order: g.Order, out: []f64) !void {
        return t.product.simulate(acc, order, out);
    }
};
