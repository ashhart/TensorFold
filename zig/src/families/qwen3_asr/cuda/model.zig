//! Qwen3-ASR on CUDA: the shared tower and Qwen3 schedules over this directory's bindings, one allocation store.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("../config.zig");
const mel = @import("../mel.zig");
const tower = @import("../tower.zig");
const lm = @import("../lm.zig");
const store = @import("store.zig");
const ctower = @import("tower.zig");
const clm = @import("lm.zig");

pub const Dump = ctower.Dump;
pub const Splice = lm.Splice;

pub const Options = struct {
    group: usize = 16, // chunks a convolution pass
    audio_rows: usize, // the longest clip's tower rows
    pass_rows: usize, // the most ids a language model pass takes
    positions: usize, // the most positions the caches hold
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    c: cfg.Config,
    st: store.Store,
    weight_bytes: u64,
    tw: tower.Weights,
    lw: lm.Weights,
    tb: tower.Buffers,
    lb: lm.Buffers,
    tr: ctower.Tower,
    lr: clm.Lm,

    /// The checkpoint at `dir` on the device (every tensor used), with scratch for `o`'s clips and passes.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, dir: []const u8, c: cfg.Config, o: Options) !*Model {
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.c = c;
        m.st = .{ .gpa = gpa, .d = d };
        errdefer m.st.deinit();
        {
            var ck = try core.Checkpoint.openModel(gpa, io, dir);
            defer ck.close();
            const src: store.Source = .{ .ck = &ck };
            m.tw = try tower.Weights.load(gpa, c.audio, src, &m.st, o.group);
            errdefer m.tw.deinit(gpa);
            m.lw = try lm.Weights.load(gpa, c.text, src, &m.st);
            errdefer m.lw.deinit(gpa);
            if (ck.unused() != 0) return error.UnexpectedTensors;
        }
        errdefer m.tw.deinit(gpa);
        errdefer m.lw.deinit(gpa);
        m.weight_bytes = m.st.bytes;
        m.tb = try tower.Buffers.alloc(c.audio, o.group, o.audio_rows, &m.st);
        m.lb = try lm.Buffers.alloc(c.text, o.pass_rows, &m.st);
        m.tr = try ctower.Tower.init(gpa, d, s, c.audio, o.group, o.audio_rows, &m.st);
        errdefer m.tr.deinit();
        m.lr = try clm.Lm.init(gpa, d, s, c.text, o.pass_rows, o.positions, &m.st);
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.lr.deinit(m.gpa);
        m.tr.deinit();
        m.lw.deinit(m.gpa);
        m.tw.deinit(m.gpa);
        m.st.deinit();
        m.gpa.destroy(m);
    }

    /// The tower over `feats` into `out` [rows, audio.out] bf16, each stage to `sink` when given; the rows returned.
    pub fn encode(m: *Model, feats: mel.Features, out: u64, sink: ?Dump) !usize {
        m.tr.sink = sink;
        defer m.tr.sink = null;
        return tower.encode(&m.tr, m.gpa, m.c.audio, &m.tw, m.tb, feats, out);
    }

    /// One language model pass over `ids` at [pos, pos + ids.len), `audio` spliced in: the greedy next id.
    pub fn forward(m: *Model, ids: []const u32, pos: usize, audio: ?Splice) !u32 {
        return m.forwardDump(ids, pos, audio, null);
    }

    /// forward with its stages (lm_embed, lm_layer{i}, lm_norm) to `sink`.
    pub fn forwardDump(m: *Model, ids: []const u32, pos: usize, audio: ?Splice, sink: ?Dump) !u32 {
        m.lr.sink = sink;
        defer m.lr.sink = null;
        return lm.forward(&m.lr, m.c.text, &m.lw, m.lb, ids, pos, audio);
    }
};
