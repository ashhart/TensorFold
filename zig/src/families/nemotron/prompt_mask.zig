//! Exceptional prompt pipelines and GPU selection; the original prefill pipelines retain their source and options.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const source = sources.nemotron_prompt_masked;

pub const Pipelines = struct {
    masked: [2][4]mtl.Pipeline,
    scan: [2]mtl.Pipeline,
    pick: mtl.Pipeline,

    pub fn load(a: std.mem.Allocator, device: mtl.Device, full: ?bool) !Pipelines {
        var p: Pipelines = undefined;
        var made: std.ArrayList(mtl.Pipeline) = .empty;
        defer made.deinit(a);
        errdefer for (made.items) |x| x.deinit();
        inline for ([_]bool{ false, true }, 0..) |is_paged, j| {
            const raw = if (is_paged) source.paged_source else source.contiguous;
            const text = if (full) |v| try frags.sourceMode(device, a, raw, v) else try frags.source(device, a, raw);
            defer a.free(text);
            const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
            defer lib.deinit();
            inline for (0..4) |i| p.masked[j][i] = try build(lib, device, source.name(is_paged, i), a, &made);
            p.scan[j] = try build(lib, device, if (is_paged) "nemotron_prompt_scan_paged" else "nemotron_prompt_scan", a, &made);
            if (!is_paged) p.pick = try build(lib, device, "nemotron_prompt_pick", a, &made);
        }
        return p;
    }

    fn build(lib: mtl.Library, device: mtl.Device, name: []const u8, a: std.mem.Allocator, made: *std.ArrayList(mtl.Pipeline)) !mtl.Pipeline {
        const p = try mtl.Pipeline.init(device, lib, name, false);
        errdefer p.deinit();
        try made.append(a, p);
        return p;
    }

    pub fn deinit(p: Pipelines) void {
        for (p.masked) |layout| for (layout) |x| x.deinit();
        for (p.scan) |x| x.deinit();
        p.pick.deinit();
    }
};
