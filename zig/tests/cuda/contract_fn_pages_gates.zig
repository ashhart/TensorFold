//! Gates 1–3 compare every paged cache launch with its contiguous kernel; sum stress is a separate harness.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const page = @import("contract_fn_pages.zig");
const gates = core.contract.gates;
const c = core.contract.spec;

fn depth(_: c.Shape) usize {
    return 0;
}

const State = struct {
    gpu: check.Gpu,
    stream: cuda.Stream,
    glue: *const flash.Glue,
    select: cuda.Function,
    case: page.Case,
    gate: gates.Gate = .shapes,
    inputs_arena: std.heap.ArenaAllocator,
    outputs_arena: std.heap.ArenaAllocator,
    fixture: ?page.Fixture = null,
    output_bytes: []const u8 = &.{},
    flat_bytes: []const u8 = &.{},
    checked: bool = false,

    fn kernel(s: *State, served: []const c.Shape) gates.Kernel {
        return .{ .ctx = s, .vt = &vt, .spec = .{
            .name = "cuda Flash Next paged prep, media, pool, tiled scores and attention address replacements",
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .replaces = "the same contiguous cache launches",
            .nan = .{ .stated = "all outputs and logical caches match the contiguous kernels, including NaN and both infinities" },
            .served = served,
            .tiles = .{ .{ .size = 256, .step = 1, .min = 1 }, .{}, .{} },
        } };
    }
    fn of(p: *anyopaque) *State {
        return @ptrCast(@alignCast(p));
    }
    fn prepare(p: *anyopaque, shape: c.Shape, fill: gates.Fill, seed: u64) !bool {
        const s = of(p);
        s.inputs_arena.deinit();
        s.inputs_arena = .init(s.gpu.gpa);
        var chosen = s.case;
        if (s.gate == .shapes) chosen.rows = shape.rows;
        s.fixture = try page.Fixture.make(s.inputs_arena.allocator(), chosen, seed);
        if (fill == .nonfinite) {
            const f = &s.fixture.?;
            f.proj[0][0] = c.toBf16(std.math.nan(f32));
            f.proj[0][1] = c.toBf16(std.math.inf(f32));
            f.proj[0][2] = c.toBf16(-std.math.inf(f32));
            if (chosen.lens[0] > 0) f.caches[0][1][0] = c.toBf16(std.math.nan(f32));
        }
        s.checked = false;
        return true;
    }
    fn bytes(a: std.mem.Allocator, r: page.Run) ![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        for (r.out) |o| inline for (.{ "q", "iq", "ao", "sc", "ids", "nk", "spr" }) |field| try out.appendSlice(a, @field(o, field));
        for (r.caches) |stream| for (stream) |cache| try out.appendSlice(a, std.mem.sliceAsBytes(cache));
        return out.items;
    }
    fn runOne(p: *anyopaque) !void {
        const s = of(p);
        s.outputs_arena.deinit();
        s.outputs_arena = .init(s.gpu.gpa);
        const a = s.outputs_arena.allocator();
        const f = s.fixture.?;
        const flat = try page.launch(s.gpu, s.stream, s.glue, s.select, f, null, null, null, a);
        const pg = try page.launch(s.gpu, s.stream, s.glue, s.select, f, .{ f.ids[0], f.ids[1] }, null, null, a);
        s.flat_bytes = try bytes(a, flat);
        s.output_bytes = try bytes(a, pg);
        s.checked = true;
    }
    fn output(p: *anyopaque) ![]const u8 {
        return of(p).output_bytes;
    }
    fn replaced(p: *anyopaque) ![]const u8 {
        return of(p).flat_bytes;
    }
    fn inputs(p: *anyopaque) ![]const u8 {
        const s = of(p);
        const f = s.fixture.?;
        var out: std.ArrayList(u8) = .empty;
        const a = s.outputs_arena.allocator();
        for (f.proj) |x| try out.appendSlice(a, std.mem.sliceAsBytes(x));
        for (f.caches) |stream| for (stream) |x| try out.appendSlice(a, std.mem.sliceAsBytes(x));
        inline for (.{ "qw", "kw", "iw", "ikw", "inv" }) |field| try out.appendSlice(a, std.mem.sliceAsBytes(@field(f, field)));
        for (f.ids) |x| try out.appendSlice(a, std.mem.sliceAsBytes(x));
        return out.items;
    }
    fn intact(p: *anyopaque) !bool {
        // launch checks every actual device input, every guard and overlaps before returning a result.
        return of(p).checked;
    }
    fn overlap(_: *anyopaque) bool {
        return false;
    }
    fn count(p: *anyopaque) usize {
        return of(p).output_bytes.len;
    }
    fn reference(_: *anyopaque, _: []f64, _: []f64) !void {
        return error.ReplacementUsesContiguousBytes;
    }
    fn own(p: *anyopaque, _: gates.Gate) !?[]const u8 {
        const s = of(p);
        return if (s.checked and std.mem.eql(u8, s.output_bytes, s.flat_bytes)) null else "paged bytes differ from the contiguous kernels";
    }
    const vt: gates.VTable = .{ .prepare = prepare, .run = runOne, .output = output, .inputs = inputs, .intact = intact, .overlap = overlap, .count = count, .reference = reference, .check = own, .replaced = replaced };
};

pub fn run(gpu: check.Gpu, stream: cuda.Stream, glue: *const flash.Glue, select: cuda.Function) !void {
    const edge_cases = [_]page.Case{
        .{ .lens = .{ 0, 3 }, .rows = 4096, .ptok = 48, .slot = 0, .slots = 1 },
        .{ .lens = .{ 262143, 262141 }, .rows = 1, .ptok = 512, .slot = 0, .slots = 1 },
    };
    for (page.cases ++ edge_cases, 0..) |case, ci| {
        var s: State = .{ .gpu = gpu, .stream = stream, .glue = glue, .select = select, .case = case, .inputs_arena = .init(gpu.gpa), .outputs_arena = .init(gpu.gpa) };
        defer s.inputs_arena.deinit();
        defer s.outputs_arena.deinit();
        const served = [_]c.Shape{.{ .rows = case.rows, .n = case.lens[0], .k = case.ptok }};
        const k = s.kernel(&served);
        for ([_]gates.Gate{ .nonfinite, .repeat, .shapes }) |gate| {
            s.gate = gate;
            try check.expect(try gates.one(gpu.gpa, gate, k, 0xf100 + ci) == .pass, "Flash Next paged gate {t} failed for case {d}", .{ gate, ci });
        }
    }
}
