// The H3 family's rank-to-rank exchange over RoCE (libibverbs): RDMA writes into every peer's region, one QP a rail.
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define H3X_MAX_RANKS 8
#define H3X_MAX_RAILS 2
#define H3X_PORT 1
#define H3X_SEND_DEPTH 256
#define H3X_RECV_DEPTH 64
#define H3X_CHUNK (512ull << 20)

// 72 bytes: what a peer needs to reach this rank's queue pairs for it
typedef struct {
    uint64_t addr;
    struct {
        uint32_t rkey, qpn, psn, mtu;
        uint8_t gid[16];
    } rail[H3X_MAX_RAILS];
} h3x_info;

typedef struct {
    struct ibv_context* ctx;
    struct ibv_pd* pd;
    struct ibv_cq* cq;
    struct ibv_mr* mr;
    struct ibv_qp* qp[H3X_MAX_RANKS];
    union ibv_gid gid;
    uint32_t mtu;
    uint32_t peer_rkey[H3X_MAX_RANKS];
    uint32_t psn[H3X_MAX_RANKS];
    // the newest sequence number each peer's writes have carried on this rail
    uint32_t got[H3X_MAX_RANKS];
} h3x_rail;

typedef struct {
    h3x_rail rail[H3X_MAX_RAILS];
    int rails;
    uint8_t* region;
    uint64_t len;
    int world, rank, gid_index;
    uint64_t peer_addr[H3X_MAX_RANKS];
    int connected[H3X_MAX_RANKS];
    // this rank's writes not yet acknowledged
    int pending;
    uint64_t bytes_out;
    char err[256];
} h3x;

static void fail(h3x* x, const char* what) { snprintf(x->err, sizeof x->err, "%s: %s", what, strerror(errno)); }

const char* h3x_error(h3x* x) { return x->err; }
uint64_t h3x_bytes(h3x* x) { return x->bytes_out; }
int h3x_rails(h3x* x) { return x->rails; }

void h3x_destroy(h3x* x) {
    if (!x) return;
    for (int r = 0; r < H3X_MAX_RAILS; ++r) {
        h3x_rail* a = &x->rail[r];
        for (int i = 0; i < H3X_MAX_RANKS; ++i)
            if (a->qp[i]) ibv_destroy_qp(a->qp[i]);
        if (a->mr) ibv_dereg_mr(a->mr);
        if (a->cq) ibv_destroy_cq(a->cq);
        if (a->pd) ibv_dealloc_pd(a->pd);
        if (a->ctx) ibv_close_device(a->ctx);
    }
    free(x);
}

static int open_rail(h3x* x, h3x_rail* a, const char* device) {
    int n = 0;
    struct ibv_device** list = ibv_get_device_list(&n);
    struct ibv_device* found = NULL;
    for (int i = 0; list && i < n; ++i)
        if (!strcmp(ibv_get_device_name(list[i]), device)) found = list[i];
    if (found) a->ctx = ibv_open_device(found);
    if (list) ibv_free_device_list(list);
    if (!found) { snprintf(x->err, sizeof x->err, "no RDMA device %s", device); return -1; }
    if (!a->ctx) { fail(x, "ibv_open_device"); return -1; }
    struct ibv_port_attr port;
    if (ibv_query_port(a->ctx, H3X_PORT, &port)) { fail(x, "ibv_query_port"); return -1; }
    if (port.state != IBV_PORT_ACTIVE) { snprintf(x->err, sizeof x->err, "%s: the port is not active", device); return -1; }
    a->mtu = port.active_mtu;
    if (ibv_query_gid(a->ctx, H3X_PORT, x->gid_index, &a->gid)) { fail(x, "ibv_query_gid"); return -1; }
    a->pd = ibv_alloc_pd(a->ctx);
    if (!a->pd) { fail(x, "ibv_alloc_pd"); return -1; }
    a->cq = ibv_create_cq(a->ctx, 1024, NULL, NULL, 0);
    if (!a->cq) { fail(x, "ibv_create_cq"); return -1; }
    a->mr = ibv_reg_mr(a->pd, x->region, x->len, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!a->mr) { fail(x, "ibv_reg_mr"); return -1; }
    for (int p = 0; p < x->world; ++p) {
        if (p == x->rank) continue;
        struct ibv_qp_init_attr q = {0};
        q.send_cq = q.recv_cq = a->cq;
        q.qp_type = IBV_QPT_RC;
        q.cap.max_send_wr = H3X_SEND_DEPTH;
        q.cap.max_recv_wr = H3X_RECV_DEPTH;
        q.cap.max_send_sge = q.cap.max_recv_sge = 1;
        a->qp[p] = ibv_create_qp(a->pd, &q);
        if (!a->qp[p]) { fail(x, "ibv_create_qp"); return -1; }
        a->psn[p] = (uint32_t)(0x1000 + 97 * x->rank + 13 * p) & 0xffffff;
        struct ibv_qp_attr m = {0};
        m.qp_state = IBV_QPS_INIT;
        m.port_num = H3X_PORT;
        m.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        if (ibv_modify_qp(a->qp[p], &m, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) { fail(x, "qp to INIT"); return -1; }
    }
    return 0;
}

// `devices`: RDMA device names separated by commas, one a rail (at most two).
h3x* h3x_create(const char* devices, int gid_index, void* region, uint64_t len, int world, int rank, char* err, uint64_t err_len) {
    h3x* x = calloc(1, sizeof *x);
    if (!x) return NULL;
    x->region = region;
    x->len = len;
    x->world = world;
    x->rank = rank;
    x->gid_index = gid_index;
    char names[256];
    snprintf(names, sizeof names, "%s", devices);
    char* save = NULL;
    for (char* name = strtok_r(names, ", ", &save); name; name = strtok_r(NULL, ", ", &save)) {
        if (x->rails == H3X_MAX_RAILS) { snprintf(x->err, sizeof x->err, "at most %d RDMA devices", H3X_MAX_RAILS); goto bad; }
        if (open_rail(x, &x->rail[x->rails], name)) goto bad;
        x->rails += 1;
    }
    if (x->rails == 0) { snprintf(x->err, sizeof x->err, "no RDMA device named"); goto bad; }
    return x;
bad:
    snprintf(err, err_len, "%s", x->err);
    h3x_destroy(x);
    return NULL;
}

// What `peer` needs to reach this rank's queue pairs for it.
int h3x_info_for(h3x* x, int peer, h3x_info* out) {
    if (peer < 0 || peer >= x->world || peer == x->rank) return -1;
    memset(out, 0, sizeof *out);
    out->addr = (uint64_t)(uintptr_t)x->region;
    for (int r = 0; r < x->rails; ++r) {
        h3x_rail* a = &x->rail[r];
        out->rail[r].rkey = a->mr->rkey;
        out->rail[r].qpn = a->qp[peer]->qp_num;
        out->rail[r].psn = a->psn[peer];
        out->rail[r].mtu = a->mtu;
        memcpy(out->rail[r].gid, a->gid.raw, 16);
    }
    return 0;
}

static int post_recv(h3x_rail* a, int peer) {
    struct ibv_recv_wr wr = {0}, *bad = NULL;
    wr.wr_id = (uint64_t)peer;
    return ibv_post_recv(a->qp[peer], &wr, &bad);
}

// `in`: the record `peer` made for this rank.
int h3x_connect(h3x* x, int peer, const h3x_info* in) {
    if (peer < 0 || peer >= x->world || peer == x->rank) return -1;
    for (int r = 0; r < x->rails; ++r) {
        h3x_rail* a = &x->rail[r];
        if (in->rail[r].qpn == 0) { snprintf(x->err, sizeof x->err, "rank %d has no rail %d", peer, r); return -1; }
        struct ibv_qp_attr q = {0};
        q.qp_state = IBV_QPS_RTR;
        q.path_mtu = in->rail[r].mtu < a->mtu ? in->rail[r].mtu : a->mtu;
        q.dest_qp_num = in->rail[r].qpn;
        q.rq_psn = in->rail[r].psn;
        q.max_dest_rd_atomic = 1;
        q.min_rnr_timer = 12;
        q.ah_attr.is_global = 1;
        q.ah_attr.port_num = H3X_PORT;
        q.ah_attr.grh.sgid_index = x->gid_index;
        q.ah_attr.grh.hop_limit = 64;
        memcpy(q.ah_attr.grh.dgid.raw, in->rail[r].gid, 16);
        if (ibv_modify_qp(a->qp[peer], &q, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
            fail(x, "qp to RTR");
            return -1;
        }
        for (int i = 0; i < H3X_RECV_DEPTH; ++i)
            if (post_recv(a, peer)) { fail(x, "ibv_post_recv"); return -1; }
        memset(&q, 0, sizeof q);
        q.qp_state = IBV_QPS_RTS;
        q.sq_psn = a->psn[peer];
        q.timeout = 14;
        q.retry_cnt = 7;
        q.rnr_retry = 7;
        q.max_rd_atomic = 1;
        if (ibv_modify_qp(a->qp[peer], &q, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC)) {
            fail(x, "qp to RTS");
            return -1;
        }
        a->peer_rkey[peer] = in->rail[r].rkey;
    }
    x->peer_addr[peer] = in->addr;
    x->connected[peer] = 1;
    return 0;
}

// [at, at + left) of this rank's region into the same place at `peer` over rail `a`; the last write carries `seq`.
static int post_rail(h3x* x, h3x_rail* a, int peer, uint64_t at, uint64_t left, uint32_t seq) {
    struct ibv_send_wr wr[8];
    struct ibv_sge sge[8];
    int k = 0;
    while (left > 0) {
        const uint64_t m = left > H3X_CHUNK ? H3X_CHUNK : left;
        const int last = m == left;
        if (k == 8) { snprintf(x->err, sizeof x->err, "a span over %llu MB", (unsigned long long)(8 * H3X_CHUNK >> 20)); return -1; }
        sge[k].addr = (uint64_t)(uintptr_t)(x->region + at);
        sge[k].length = (uint32_t)m;
        sge[k].lkey = a->mr->lkey;
        memset(&wr[k], 0, sizeof wr[k]);
        wr[k].wr_id = (uint64_t)peer;
        wr[k].sg_list = &sge[k];
        wr[k].num_sge = 1;
        wr[k].opcode = last ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE;
        wr[k].send_flags = last ? IBV_SEND_SIGNALED : 0;
        wr[k].imm_data = htonl(seq);
        wr[k].wr.rdma.remote_addr = x->peer_addr[peer] + at;
        wr[k].wr.rdma.rkey = a->peer_rkey[peer];
        if (k > 0) wr[k - 1].next = &wr[k];
        x->bytes_out += m;
        at += m;
        left -= m;
        k += 1;
    }
    struct ibv_send_wr* bad = NULL;
    if (ibv_post_send(a->qp[peer], wr, &bad)) { fail(x, "ibv_post_send"); return -1; }
    x->pending += 1;
    return 0;
}

// `n` spans of this region into the same offsets at `peer`, one part a rail; each rail's last write carries `seq`.
int h3x_post(h3x* x, int peer, const uint64_t* off, const uint64_t* len, int n, uint32_t seq) {
    if (!x->connected[peer] || n != 1) { snprintf(x->err, sizeof x->err, "one span a post, to a connected rank"); return -1; }
    if (off[0] + len[0] > x->len || len[0] == 0) { snprintf(x->err, sizeof x->err, "a span past the region"); return -1; }
    const uint64_t part = (len[0] / x->rails + 4095) & ~4095ull;
    uint64_t at = off[0], left = len[0];
    for (int r = 0; r < x->rails; ++r) {
        // every rail carries the sequence number, an empty part too (one byte, the span's last)
        uint64_t m = r == x->rails - 1 ? left : (part < left ? part : left);
        if (m == 0) { at -= 1; left += 1; m = 1; }
        if (post_rail(x, &x->rail[r], peer, at, m, seq)) return -1;
        at += m;
        left -= m;
    }
    return 0;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

// Waits for this rank's writes and every peer's `seq` (or later) on every rail; -1 on a failure or after `seconds`.
int h3x_wait(h3x* x, uint32_t seq, double seconds) {
    const double until = now() + seconds;
    struct ibv_wc wc[32];
    unsigned spins = 0;
    for (;;) {
        int waiting = x->pending, polled = 0;
        for (int r = 0; r < x->rails; ++r)
            for (int p = 0; p < x->world; ++p)
                if (x->connected[p] && (int32_t)(x->rail[r].got[p] - seq) < 0) waiting += 1;
        if (!waiting) return 0;
        for (int r = 0; r < x->rails; ++r) {
            h3x_rail* a = &x->rail[r];
            const int n = ibv_poll_cq(a->cq, 32, wc);
            if (n < 0) { fail(x, "ibv_poll_cq"); return -1; }
            polled += n;
            for (int i = 0; i < n; ++i) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    snprintf(x->err, sizeof x->err, "completion failed with rank %d on rail %d: %s", (int)wc[i].wr_id, r, ibv_wc_status_str(wc[i].status));
                    return -1;
                }
                if (wc[i].opcode & IBV_WC_RECV) {
                    const int p = (int)wc[i].wr_id;
                    a->got[p] = ntohl(wc[i].imm_data);
                    if (post_recv(a, p)) { fail(x, "ibv_post_recv"); return -1; }
                } else {
                    x->pending -= 1;
                }
            }
        }
        if (polled == 0 && (++spins & 0xfff) == 0 && now() > until) {
            snprintf(x->err, sizeof x->err, "timed out at sequence %u (pending writes %d)", seq, x->pending);
            return -1;
        }
    }
}
