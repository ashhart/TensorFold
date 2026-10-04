// Host side of TensorFold's small all-gather over RoCE between two machines (no GPUDirect needed).
//
// The protocol (a pinned registered ctrl/flags/send/recv region, a proxy thread spinning on ctrl.seq, the payload and
// then a seq flag RDMA-written on one RC QP) follows b12x's "RoCEnante" proxy, b12x/comm/roce/_roce_proxy.c
// (https://github.com/local-inference-lab/b12x, commit 8a99d639410e; Apache License 2.0, Luke Alonso and the b12x
// contributors), as carried in MiaAI-Lab's GLM-5.3-Flash recipe patch 0006 (Apache License 2.0). Rewritten for
// TensorFold: one RC QP between two ranks, no HCA striping, its own region layout, ABI and connection exchange.
//
// Each rank owns one pinned host region, registered with the NIC, laid out as (byte offsets, see tf_rdma_layout):
//   ctrl   {u32 seq, u32 nbytes[2], u32 error}   the GPU rings seq after staging nbytes[seq & 1] bytes
//   flags  [2] x 64 bytes                        the peer writes a slot's seq here after the slot's payload
//   send   [2] x slot_bytes                      staged by the local GPU
//   recv   [2] x slot_bytes                      written by the peer's NIC
// A proxy thread spins on ctrl.seq; for every new sequence number it posts two RDMA writes on one reliable QP: the
// payload into the peer's recv[slot], then the 4-byte seq into the peer's flags[slot]. Writes on one RC QP are placed
// in order, so a flag never becomes visible before its payload. The GPU polls its own flags[slot].
#define _GNU_SOURCE
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TF_RDMA_ABI 1
#define CTRL_BYTES 64
#define FLAG_BYTES 64
#define SLOTS 2

typedef struct {
    // connection details exchanged between the ranks (all as uint64 for a simple int64 exchange)
    uint64_t qpn, psn, rkey, addr, mtu, gid_hi, gid_lo;
} tf_rdma_info;

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    int gid_index;
    union ibv_gid gid;
    enum ibv_mtu mtu;
    uint32_t psn;
    char *region;
    uint64_t region_bytes, slot_bytes;
    tf_rdma_info peer;
    pthread_t thread;
    volatile int running, failed;
    uint32_t last_seq, inflight;
    uint64_t posted, completed;
    char error[256];
} tf_rdma;

int tf_rdma_abi(void) { return TF_RDMA_ABI; }

// offsets {ctrl, flags, send, recv, total} for a slot size (slot_bytes a multiple of 4096)
void tf_rdma_layout(uint64_t slot_bytes, uint64_t *out) {
    out[0] = 0;
    out[1] = CTRL_BYTES;
    out[2] = 4096;
    out[3] = 4096 + SLOTS * slot_bytes;
    out[4] = 4096 + 2 * SLOTS * slot_bytes;
}

static void fail(tf_rdma *r, const char *what, int err) {
    snprintf(r->error, sizeof r->error, "%s: %s", what, strerror(err ? err : errno));
    r->failed = 1;
}

const char *tf_rdma_error(tf_rdma *r) { return r ? r->error : "no context"; }
int tf_rdma_failed(tf_rdma *r) { return r ? r->failed : 1; }
uint64_t tf_rdma_counter(tf_rdma *r, int which) { return which == 0 ? r->posted : which == 1 ? r->completed : r->last_seq; }

void tf_rdma_destroy(tf_rdma *r);

// open device ``name``, register the pinned region, create an RC QP in INIT; NULL (with ``err``) on failure
tf_rdma *tf_rdma_create(const char *name, int gid_index, void *region, uint64_t region_bytes, uint64_t slot_bytes,
                        char *err, uint64_t err_len) {
    tf_rdma *r = calloc(1, sizeof *r);
    if (!r) return NULL;
    r->gid_index = gid_index;
    r->region = region;
    r->region_bytes = region_bytes;
    r->slot_bytes = slot_bytes;
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    for (int i = 0; list && i < n; ++i)
        if (!strcmp(ibv_get_device_name(list[i]), name)) r->ctx = ibv_open_device(list[i]);
    if (list) ibv_free_device_list(list);
    if (!r->ctx) { snprintf(err, err_len, "RDMA device %s not found or not openable", name); free(r); return NULL; }
    struct ibv_port_attr port;
    if (ibv_query_port(r->ctx, 1, &port) || ibv_query_gid(r->ctx, 1, gid_index, &r->gid)) {
        snprintf(err, err_len, "%s: cannot query port 1 / GID %d", name, gid_index);
        tf_rdma_destroy(r);
        return NULL;
    }
    r->mtu = port.active_mtu;
    r->pd = ibv_alloc_pd(r->ctx);
    r->mr = r->pd ? ibv_reg_mr(r->pd, region, region_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE) : NULL;
    r->cq = r->mr ? ibv_create_cq(r->ctx, 256, NULL, NULL, 0) : NULL;
    if (r->cq) {
        struct ibv_qp_init_attr qa;
        memset(&qa, 0, sizeof qa);
        qa.send_cq = qa.recv_cq = r->cq;
        qa.qp_type = IBV_QPT_RC;
        qa.cap.max_send_wr = 128;
        qa.cap.max_recv_wr = 1;
        qa.cap.max_send_sge = qa.cap.max_recv_sge = 1;
        qa.cap.max_inline_data = 16;
        r->qp = ibv_create_qp(r->pd, &qa);
    }
    if (!r->qp) {
        snprintf(err, err_len, "%s: protection domain / memory registration / CQ / QP failed: %s", name, strerror(errno));
        tf_rdma_destroy(r);
        return NULL;
    }
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        snprintf(err, err_len, "%s: QP to INIT failed: %s", name, strerror(errno));
        tf_rdma_destroy(r);
        return NULL;
    }
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    r->psn = (uint32_t)(ts.tv_nsec ^ ts.tv_sec) & 0xffffff;
    return r;
}

void tf_rdma_local(tf_rdma *r, uint64_t *out) {
    uint64_t hi, lo;
    memcpy(&hi, r->gid.raw, 8);
    memcpy(&lo, r->gid.raw + 8, 8);
    tf_rdma_info i = {r->qp->qp_num, r->psn, r->mr->rkey, (uint64_t)(uintptr_t)r->region, (uint64_t)r->mtu, hi, lo};
    memcpy(out, &i, sizeof i);
}

// QP to RTR and RTS against the peer's details (RoCE v2: a global route by GID)
int tf_rdma_connect(tf_rdma *r, const uint64_t *peer) {
    memcpy(&r->peer, peer, sizeof r->peer);
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = r->mtu < (enum ibv_mtu)r->peer.mtu ? r->mtu : (enum ibv_mtu)r->peer.mtu;
    a.dest_qp_num = (uint32_t)r->peer.qpn;
    a.rq_psn = (uint32_t)r->peer.psn;
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer = 12;
    a.ah_attr.is_global = 1;
    memcpy(a.ah_attr.grh.dgid.raw, &r->peer.gid_hi, 8);
    memcpy(a.ah_attr.grh.dgid.raw + 8, &r->peer.gid_lo, 8);
    a.ah_attr.grh.sgid_index = (uint8_t)r->gid_index;
    a.ah_attr.grh.hop_limit = 64;
    a.ah_attr.port_num = 1;
    if (ibv_modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                     IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        fail(r, "QP to RTR", errno);
        return -1;
    }
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_RTS;
    a.timeout = 14;
    a.retry_cnt = 7;
    a.rnr_retry = 7;
    a.sq_psn = r->psn;
    a.max_rd_atomic = 1;
    if (ibv_modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                     IBV_QP_MAX_QP_RD_ATOMIC)) {
        fail(r, "QP to RTS", errno);
        return -1;
    }
    return 0;
}

static int reap(tf_rdma *r) {
    struct ibv_wc wc[16];
    int n = ibv_poll_cq(r->cq, 16, wc);
    if (n < 0) { fail(r, "poll CQ", errno); return -1; }
    for (int i = 0; i < n; ++i) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            snprintf(r->error, sizeof r->error, "RDMA write failed: %s (vendor error 0x%x, seq %u)",
                     ibv_wc_status_str(wc[i].status), wc[i].vendor_err, (unsigned)wc[i].wr_id);
            r->failed = 1;
            return -1;
        }
        r->completed++;
        r->inflight--;
    }
    return 0;
}

// payload of sequence ``seq`` (slot seq & 1) to the peer's recv slot, then its flag
static int post(tf_rdma *r, uint32_t seq) {
    volatile uint32_t *ctrl = (volatile uint32_t *)r->region;
    const uint32_t slot = seq & 1u, nbytes = ctrl[1 + slot];
    uint64_t send_off[5];
    tf_rdma_layout(r->slot_bytes, send_off);
    while (r->inflight + 1 > 64)                        // keep the send queue (128 WRs: 2 an op) from filling
        if (reap(r)) return -1;
    struct ibv_sge data = {(uint64_t)(uintptr_t)(r->region + send_off[2] + slot * r->slot_bytes), nbytes, r->mr->lkey};
    uint32_t value = seq;
    struct ibv_sge flag = {(uint64_t)(uintptr_t)&value, 4, r->mr->lkey};
    struct ibv_send_wr wd, wf, *bad = NULL;
    memset(&wd, 0, sizeof wd);
    memset(&wf, 0, sizeof wf);
    wd.wr_id = seq;
    wd.sg_list = &data;
    wd.num_sge = 1;
    wd.opcode = IBV_WR_RDMA_WRITE;
    wd.wr.rdma.remote_addr = r->peer.addr + send_off[3] + slot * r->slot_bytes;
    wd.wr.rdma.rkey = (uint32_t)r->peer.rkey;
    wd.next = &wf;
    wf.wr_id = seq;
    wf.sg_list = &flag;
    wf.num_sge = 1;
    wf.opcode = IBV_WR_RDMA_WRITE;
    wf.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;   // inline: ``value`` is copied at post time
    wf.wr.rdma.remote_addr = r->peer.addr + send_off[1] + slot * FLAG_BYTES;
    wf.wr.rdma.rkey = (uint32_t)r->peer.rkey;
    int rc = ibv_post_send(r->qp, nbytes ? &wd : &wf, &bad);
    if (rc) { fail(r, "post RDMA write", rc); return -1; }
    r->inflight++;
    r->posted++;
    return 0;
}

static void *loop(void *arg) {
    tf_rdma *r = arg;
    volatile uint32_t *ctrl = (volatile uint32_t *)r->region;
    uint64_t idle = 0;
    while (r->running && !r->failed) {
        const uint32_t seq = __atomic_load_n((uint32_t *)&ctrl[0], __ATOMIC_ACQUIRE);
        if (seq != r->last_seq) {
            while (r->last_seq != seq && !r->failed) {       // in order, including any the GPU rang meanwhile
                r->last_seq++;
                if (post(r, r->last_seq)) break;
            }
            idle = 0;
        } else if (++idle > 2000000) {                        // idle a while: poll gently
            struct timespec ts = {0, 20000};
            nanosleep(&ts, NULL);
        }
        if (r->inflight && reap(r)) break;
    }
    if (r->failed) __atomic_store_n((uint32_t *)&ctrl[3], 1u, __ATOMIC_RELEASE);
    return NULL;
}

int tf_rdma_start(tf_rdma *r) {
    r->running = 1;
    int rc = pthread_create(&r->thread, NULL, loop, r);
    if (rc) { r->running = 0; fail(r, "start proxy thread", rc); return -1; }
    return 0;
}

void tf_rdma_destroy(tf_rdma *r) {
    if (!r) return;
    if (r->running) {
        r->running = 0;
        pthread_join(r->thread, NULL);
    }
    if (r->qp) ibv_destroy_qp(r->qp);
    if (r->cq) ibv_destroy_cq(r->cq);
    if (r->mr) ibv_dereg_mr(r->mr);
    if (r->pd) ibv_dealloc_pd(r->pd);
    if (r->ctx) ibv_close_device(r->ctx);
    free(r);
}
