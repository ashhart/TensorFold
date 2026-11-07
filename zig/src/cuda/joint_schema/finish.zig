//! A readout's end on the device: the span means, steps 3 to 7 of the head, and the BF16 logits to the host.
const std = @import("std");
const Stream = @import("cuda").Stream;
const DeviceBuffer = @import("cuda").DeviceBuffer;
const Head = @import("head.zig").Head;
const State = @import("state.zig").State;
const attn = @import("attention.zig");

/// One logit per option (BF16 bits, option order) once every prompt row is consumed.
pub fn finish(h: *Head, st: *State, logits: []u16, s: Stream) !void {
    if (st.consumed != st.rows) return error.ReadoutUnfinished;
    if (logits.len != st.options) return error.InvalidReadoutLogits;
    const c = h.cfg;
    const H: usize = c.hidden_size;
    const W: usize = c.width;
    const Q = st.questions;
    const T = st.options;
    const L = h.layout;
    const w = &h.w.w;
    const f = h.fns;
    const qv = h.at(L.qv);
    const cv = h.at(L.cv);
    const ev = h.at(L.ev);
    const lex = h.at(L.lex);
    const routed = h.at(L.routed);
    const tmp = h.at(L.tmp);
    const opt_spans = st.spans(h) + Q * 8;
    // step 2: the span means, and each option's mean of the lm_head rows of its tokens (read from the packed head)
    try f.spanMeans(s, st.sums(h), st.spans(h), Q, H, qv);
    try f.spanMeans(s, st.sums(h) + Q * H * 4, opt_spans, T, H, cv);
    const zero: DeviceBuffer = .{ .d = h.d, .ptr = lex, .len = T * H * 4, .borrowed = true };
    try zero.fill8(0, s.handle);
    try f.lexSums(s, h.table, st.ids(h), H, opt_spans, T, lex);
    try f.spanMeans(s, lex, opt_spans, T, H, ev);
    // step 3: (W_ctx c + W_lex e) + W_oq q, each product and sum rounded
    try h.linear(s, cv, T, w.option_context, null, routed, W, H);
    try h.linear(s, ev, T, w.option_lexical, null, tmp, W, H);
    try h.add(s, routed, tmp, routed, T * W);
    const oq = h.at(L.fields);
    try h.linear(s, qv, Q, w.option_question, null, oq, W, H);
    for (0..Q) |k| try h.broadcast(s, oq + k * W * 2, tmp + st.first[k] * W * 2, st.first[k + 1] - st.first[k], W);
    try h.add(s, routed, tmp, routed, T * W);
    // step 4: the options route over the memory
    for (w.evidence) |e| try attn.evidence(h, s, e, routed, T, st.memory(h), st.rows);
    // step 5: each field from its projection, its options' weighted summary, the last row and its type
    const fields = h.at(L.fields);
    const summary = h.at(L.summary);
    try h.linear(s, qv, Q, w.question, null, fields, W, H);
    try f.optionSummary(s, routed, fields, st.firsts(h), Q, W, h.at(L.weights), summary);
    try h.norm(s, summary, w.option_summary_norm, tmp, Q, W);
    try h.add(s, fields, tmp, fields, Q * W);
    const gp = h.at(L.gp);
    try h.linear(s, st.g(h), 1, w.global, null, gp, W, H);
    try h.broadcast(s, gp, tmp, Q, W);
    try h.add(s, fields, tmp, fields, Q * W);
    for (st.types, 0..) |t, k| try h.broadcast(s, w.type_embedding + @as(usize, t) * W * 2, tmp + k * W * 2, 1, W);
    try h.add(s, fields, tmp, fields, Q * W);
    // step 6: the decoder layers over the fields and the unnormed memory, then the field norm
    for (w.layers) |d| try attn.decoder(h, s, d, fields, Q, st.memory(h), st.rows);
    try h.norm(s, fields, w.field_norm, summary, Q, W);
    // step 7: the prior and cosine, the scorer over [f, o, f o, |f - o|], then the gate
    const opts = h.at(L.opts);
    const feat = h.at(L.feat);
    const hid = h.at(L.hid);
    const res = h.at(L.res);
    const pc = h.at(L.pc);
    const out = h.at(L.logits);
    try h.norm(s, routed, w.option_norm, opts, T, W);
    try f.logits(s, .{ .fields = summary, .opts = opts, .qv = qv, .g = st.g(h), .lex = ev, .owner = st.owner(h), .width = W, .hidden = H, .prior_scale = h.w.s.prior_scale, .prior = pc, .cosine = pc + T * 4, .feat = feat }, T);
    try h.linear(s, feat, T, w.scorer_up.weight, w.scorer_up.bias, hid, W, 4 * W);
    try f.gelu(s, hid, T * W);
    try h.linear(s, hid, T, w.scorer_down.weight, w.scorer_down.bias, res, 1, W);
    try f.combine(s, pc, pc + T * 4, res, T, h.w.s.joint_scale, h.w.s.gate, out);
    try s.synchronize();
    const host: DeviceBuffer = .{ .d = h.d, .ptr = out, .len = T * 2, .borrowed = true };
    try host.download(0, std.mem.sliceAsBytes(logits));
}
