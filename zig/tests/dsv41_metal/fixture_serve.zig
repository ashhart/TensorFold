//! The tiny model served on Metal: prompt split, Engram before it, a DSpark drafter; its serve.Metal over the rig.
const std = @import("std");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fixture = @import("fixture.zig");
const ff = @import("fixture_fwd.zig");

pub const limit = 1280;

/// The prompt model (ced's split at layer 4, Engram at 1 and 3) with fixture.dsparkConfig's drafter.
pub fn config() host.config.Config {
    var c = fixture.promptConfig();
    const d = fixture.dsparkConfig();
    c.mtp_layers = d.mtp_layers;
    c.dspark_block = d.dspark_block;
    c.dspark_noise_token = d.dspark_noise_token;
    c.dspark_markov_rank = d.dspark_markov_rank;
    c.dspark_experts = d.dspark_experts;
    c.dspark_topk = d.dspark_topk;
    return c;
}

/// The rig (weights, Engram rows and hashes) and the drafter's weights the served model borrows.
const Held = struct {
    a: std.mem.Allocator,
    f: *ff.Fwd,
    w: pkg.dspark.Weights,

    fn close(ctx: *anyopaque) void {
        const h: *Held = @ptrCast(@alignCast(ctx));
        h.w.deinit();
        h.f.deinit();
        h.a.destroy(h);
    }
};

pub const Opts = struct { slots: u32 = 4, drafts: bool = true, cache_bytes: u64 = 0, chunk: u32 = 64 };

/// The tiny model from `seed` served on this GPU; the returned model owns the rig (closed with it).
pub fn open(a: std.mem.Allocator, io: std.Io, seed: u64, o: Opts) !*pkg.serve.Metal {
    const h = try a.create(Held);
    errdefer a.destroy(h);
    h.a = a;
    h.f = try ff.Fwd.initWith(a, io, config(), limit, seed);
    errdefer h.f.deinit();
    const r = h.f.r;
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const img = try fixture.image(a, &r.c, seed);
        defer a.free(img);
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = img });
        const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(dir);
        var pack = try host.pack.Pack.open(a, io, dir);
        defer pack.deinit();
        h.w = try pkg.dspark.loadWeights(a, io, r.device, &pack, &r.c);
    }
    errdefer h.w.deinit();
    const env: pkg.target.Env = .{ .device = r.device, .queue = r.queue, .c = &r.c, .p = &r.p, .tables = &r.tables, .s = &r.s, .model = h.f.model(), .lookup = h.f.lookup() };
    const parts: pkg.serve.Parts = .{ .env = env, .top = &r.top.?, .dspark = &h.w, .owner = .{ .ctx = h, .close = Held.close } };
    return pkg.serve.Metal.open(a, parts, .{ .slots = o.slots, .limit = limit, .drafts = o.drafts, .cache_bytes = o.cache_bytes, .prompt = .{ .chunk = o.chunk } });
}

/// Closes a model `open` made (the served engine closes its own through serve_lanes).
pub fn close(m: *pkg.serve.Metal) void {
    const md = m.model();
    md.close(md.ptr);
}
