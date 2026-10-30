//! A reply on Metal: text slots and speech frames in the model's interleaved schedule, the codec streamed between rows.
const std = @import("std");
const cfg = @import("../config.zig");
const common = @import("../common.zig");
const Engine = @import("engine.zig").Engine;

const Sampling = common.Sampling;
const Sink = common.Sink;
const Streaming = common.Streaming;
const Forced = common.Forced;
const Stats = common.Stats;
const OverlapAdd = common.OverlapAdd;
const nowNs = common.nowNs;
const msSince = common.msSince;
const Mode = common.Mode;
const hist_cap = common.hist_cap;

/// Generates a reply after `prefill` into `sink` as it is made, the audio through the codec per `stream`.
pub fn reply(e: *Engine, smp: Sampling, stream: Streaming, sink: Sink, forced: ?*Forced, st_: *Stats, t_req: i128) !void {
    var prng = std.Random.DefaultPrng.init(smp.seed);
    const rng = prng.random();
    const io = e.io;
    var ola = OverlapAdd.init(e.gpa, &e.window);
    defer ola.deinit();
    if (stream.audio) try e.codecReset();
    var history: std.ArrayList([cfg.codebooks]i32) = .empty;
    defer history.deinit(e.gpa);
    var pending: std.ArrayList([cfg.codebooks]i32) = .empty;
    defer pending.deinit(e.gpa);
    var decoded: usize = 0;
    var mode: Mode = .interleaved;
    var speaking = false;
    var text_done = false;
    var slots_left: usize = cfg.interleaved_n_text;
    var fi: usize = 0;
    var si: usize = 1;
    var step: usize = 0;
    // the device path samples frames on the GPU (one copy back a frame); teacher forcing and top-k use the host's
    const device = forced == null and smp.audio_top_k == 0;
    var hist_n: usize = 0;
    while (step < smp.max_new) : (step += 1) {
        const interleaving = mode == .interleaved;
        const last_step = step + 1 == smp.max_new;
        var finished = false;
        if (mode == .audio or (interleaving and speaking)) {
            const t0 = nowNs(io);
            var codes: [cfg.codebooks]i32 = undefined;
            if (device) {
                var order: [cfg.codebooks]i32 = undefined;
                var perm: [cfg.codebooks]usize = undefined;
                for (&perm, 0..) |*x, i| x.* = i;
                if (!smp.identity_order) rng.shuffle(usize, &perm);
                for (&order, perm) |*o, x| o.* = @intCast(x);
                const g0 = e.g.gpu_s;
                try e.frameDevice(smp, &order, step, @min(hist_n, hist_cap), hist_n % hist_cap);
                try e.phase(.frame, g0);
                try e.d2h(std.mem.sliceAsBytes(&codes), e.dframe + cfg.codebooks * 4);
            } else {
                const lo = history.items.len -| smp.audio_repetition_window;
                codes = try e.sampleFrame(smp, rng, history.items[lo..]);
            }
            st_.depth_ms += msSince(io, t0);
            if (forced) |fc| {
                var mine_eos = false;
                for (codes) |c| mine_eos = mine_eos or c == cfg.audio_eos_code;
                if (si < fc.seq.len and fc.seq[si] == cfg.audio_out_token_id and fi < fc.frames.len) {
                    const ref = fc.frames[fi];
                    var all = true;
                    for (codes, ref) |c, r| {
                        fc.code_hits += @intFromBool(c == r);
                        all = all and c == r;
                    }
                    fc.code_total += cfg.codebooks;
                    fc.frame_hits += @intFromBool(all);
                    codes = ref;
                    fi += 1;
                    si += 1;
                } else {
                    fc.frame_hits += @intFromBool(mine_eos);
                    codes = @splat(cfg.audio_eos_code);
                }
                fc.frame_total += 1;
            }
            var eos = false;
            for (codes) |c| eos = eos or c == cfg.audio_eos_code;
            if (eos) {
                codes = @splat(cfg.audio_eos_code);
                mode = .text;
            } else {
                // the device's frame already joined the window (om_frame_end)
                if (device) hist_n += 1 else try history.append(e.gpa, codes);
                st_.frames += 1;
                if (st_.frames == 1) st_.first_frame_ms = msSince(io, t_req);
                if (sink.frame) |f| try f(sink.ctx, &codes);
                try pending.append(e.gpa, codes);
            }
            if (!device) e.g.set(e.dframe + cfg.codebooks * 4, std.mem.sliceAsBytes(&codes));
            e.k.embedCodes(e.cb_embed, e.dframe + cfg.codebooks * 4, e.bb_x, 1, cfg.hidden, cfg.codebooks, cfg.codebook_size);
        } else {
            var tok = try e.textPick();
            if (forced) |fc| {
                if (si >= fc.seq.len) break;
                const ref = fc.seq[si];
                si += 1;
                fc.text_hits += @intFromBool(tok == ref);
                fc.text_total += 1;
                tok = ref;
            }
            st_.text_tokens += 1;
            if (st_.text_tokens == 1) st_.first_text_ms = msSince(io, t_req);
            if (sink.text) |f| try f(sink.ctx, tok);
            if (tok == cfg.text_end_token_id) {
                if (interleaving) {
                    text_done = true;
                    speaking = true;
                } else finished = true;
            } else if (tok == cfg.eos_token_id) {
                finished = true;
            } else if (interleaving) {} else if (tok == cfg.audio_start_token_id) {
                mode = .audio;
            } else if (tok == cfg.mixed_start_token_id) {
                mode = .interleaved;
                slots_left = cfg.interleaved_n_text;
                speaking = false;
                text_done = false;
            }
            if (forced != null) {
                const id: i32 = @intCast(tok);
                e.g.set(e.pick, std.mem.asBytes(&id));
            }
            e.k.embed(e.embed, e.pick, e.bb_x, 1, cfg.hidden);
        }
        if (interleaving and !text_done) {
            slots_left -= 1;
            if (slots_left == 0) {
                speaking = !speaking;
                slots_left = if (speaking) cfg.interleaved_n_audio else cfg.interleaved_n_text;
            }
        }
        if (forced) |fc| if (si >= fc.seq.len and mode != .audio and !(mode == .interleaved and speaking)) {
            finished = true;
        };
        // the codec before the next backbone row: the first frames as soon as they exist, then whole chunks
        var frames_out: []const f32 = &.{};
        if (stream.audio) {
            const want: usize = if (decoded == 0) stream.first else stream.chunk;
            if (pending.items.len >= want or (pending.items.len > 0 and (finished or mode == .text))) {
                const t2 = nowNs(io);
                const g0 = e.g.gpu_s;
                frames_out = try e.codecChunk(pending.items, null);
                try e.phase(.codec, g0);
                decoded += pending.items.len;
                pending.clearRetainingCapacity();
                st_.codec_ms += msSince(io, t2);
            }
        }
        const go_on = !(finished or last_step);
        const t1 = nowNs(io);
        if (go_on) try e.stepRow();
        // the overlap-add on the host while the row runs
        if (frames_out.len > 0) {
            const pcm = try ola.add(frames_out, false);
            if (pcm.len > 0) {
                if (st_.samples == 0) st_.first_pcm_ms = msSince(io, t_req);
                st_.samples += pcm.len;
                if (sink.pcm) |f| try f(sink.ctx, pcm);
            }
        }
        if (!go_on) break;
        if (sink.hidden) |f| try f(sink.ctx);
        st_.step_ms += msSince(io, t1);
    }
    if (stream.audio) try flushCodec(e, &ola, &pending, &decoded, sink, st_, t_req, true);
    st_.total_ms = msSince(io, t_req);
    if (e.prof) {
        for (0..4) |i| std.log.info("lfm3-omni profile: {s} {d} x {d:.3} ms GPU", .{ @tagName(@as(Engine.Phase, @fromBackingInt(@intCast(i)))), e.prof_n[i], if (e.prof_n[i] > 0) e.prof_s[i] * 1e3 / @as(f64, @floatFromInt(e.prof_n[i])) else 0 });
        std.log.info("lfm3-omni profile: {d} dispatches", .{e.g.dispatches});
        e.prof_s = @splat(0);
        e.prof_n = @splat(0);
        e.g.dispatches = 0;
    }
}

fn flushCodec(e: *Engine, ola: *OverlapAdd, pending: *std.ArrayList([cfg.codebooks]i32), decoded: *usize, sink: Sink, st_: *Stats, t_req: i128, final: bool) !void {
    const t0 = nowNs(e.io);
    var at: usize = 0;
    while (at < pending.items.len or (final and at == 0)) {
        const n = @min(e.o.codec_chunk, pending.items.len - at);
        const frames = try e.codecChunk(pending.items[at .. at + n], null);
        const last = final and at + n >= pending.items.len;
        const pcm = try ola.add(frames, last);
        if (pcm.len > 0) {
            if (st_.samples == 0) st_.first_pcm_ms = msSince(e.io, t_req);
            st_.samples += pcm.len;
            if (sink.pcm) |f| try f(sink.ctx, pcm);
        }
        decoded.* += n;
        at += n;
        if (n == 0) break;
    }
    pending.clearRetainingCapacity();
    st_.codec_ms += msSince(e.io, t0);
}
