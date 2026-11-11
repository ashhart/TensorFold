[[max_total_threads_per_threadgroup(256)]]
kernel void im_sam_deconv_baseline(const device float *x, const device float *w, const device float *b, device float *y,
                                  constant SamConv &p, uint i [[thread_position_in_grid]]) {
    if (i>=p.oh*p.ow*p.n) return;
    uint o=i%p.n, yy=i/p.n/p.ow, xx=i/p.n%p.ow;
    float sum=b[o];
    for(uint c=0;c<p.c;c++) sum+=x[((yy/2)*p.w+xx/2)*p.c+c]*w[((c*p.n+o)*2+yy%2)*2+xx%2];
    y[i]=sum;
}
