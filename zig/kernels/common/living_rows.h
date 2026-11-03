// Native additive-learner row pullbacks; the wrappers supply device address spaces and scalar intrinsics.
#ifndef TF_LIVING_ROWS
#define TF_LIVING_ROWS
struct LwParams { unsigned u[16]; float f[4]; };
#ifndef LW_ACC
#define LW_ACC(p,target,value) ((target)=((p).u[14]==4 ? lw_round((target)+(value)) : (target)+(value)))
#endif
LW_INLINE float lw_sig(float x) { const float e=LW_EXP(-LW_ABS(x)); return (x<0?e:1.0f)/(1.0f+e); }
LW_INLINE float lw_silu(float x) { return x*lw_sig(x); }
LW_INLINE float lw_dsilu(float x) { const float s=lw_sig(x); return s*(1.0f+x*(1.0f-s)); }
LW_INLINE float lw_read(LW_DEV const unsigned char* p, unsigned i) { return LW_BITS_FLOAT(unsigned(((LW_DEV const unsigned short*)p)[i])<<16); }
LW_INLINE float lw_round(float x) {
    const unsigned bits=LW_FLOAT_BITS(x);
    const unsigned r=(bits&0x7fffffffU)>0x7f800000U ? ((bits>>16)|0x40U) : ((bits+0x7fffU+((bits>>16)&1U))>>16);
    return LW_BITS_FLOAT(r<<16);
}
LW_INLINE void lw_store(LwParams meta,LW_DEV unsigned char* p,unsigned i,float x) {
    if(meta.u[14]==1 && LW_ISNAN(x))x=0;
    if(meta.u[14]==2)x+=float(meta.u[13]);
    ((LW_DEV unsigned short*)p)[i]=(unsigned short)(LW_FLOAT_BITS(lw_round(x))>>16);
}
LW_INLINE float lw_float(LW_DEV const unsigned char* p,unsigned i) { return ((LW_DEV const float*)p)[i]; }
LW_INLINE void lw_put(LwParams meta,LW_DEV unsigned char* p,unsigned i,float v,bool add) {
    LW_DEV float* out=(LW_DEV float*)p;
    if(add)out[i]+=v;else out[i]=v;
    if(meta.u[14]==1 && LW_ISNAN(out[i]))out[i]=0;
    if(meta.u[14]==2)out[i]+=float(meta.u[13]);
    if(meta.u[14]==4 && add)out[i]=lw_round(out[i]);
}
LW_INLINE void lw_rows(unsigned op,unsigned id,LwParams p,
    LW_DEV const unsigned char* a,LW_DEV const unsigned char* b,LW_DEV const unsigned char* c,
    LW_DEV const unsigned char* d,LW_DEV const unsigned char* e,LW_DEV const unsigned char* f,
    LW_DEV const unsigned char* g,LW_DEV const unsigned char* h,
    LW_DEV unsigned char* x,LW_DEV unsigned char* y,LW_DEV unsigned char* z) {
    if(id>=p.u[15])return;
    if(op==0) {
        const unsigned n=p.u[1],k=p.u[2],row=id/k,col=id%k;
        float sum=0;
        for(unsigned o=0;o<n;o++) {
            const unsigned at=o*k+col;
            float w;
            if(p.u[3]==0)w=lw_read(b,at);
            else {
                const unsigned word=((LW_DEV const unsigned*)b)[at/8];
                const unsigned code=(word>>((at%8)*4))&15U;
                w=LW_FMA(float(code),lw_read(c,at/p.u[4]),lw_read(d,at/p.u[4]));
            }
            LW_ACC(p,sum,lw_float(a,row*(p.u[6]?p.u[6]:n)+p.u[7]+o)*w);
        }
        lw_put(p,x,id,sum,p.u[5]!=0);
    } else if(op==1) {
        const unsigned width=p.u[1],base=(id/width)*width,weights=id%(width*p.u[2]);
        float sq=0,dot=0;
        for(unsigned i=0;i<width;i++) {
            const float v=lw_read(a,base+i);
            LW_ACC(p,sq,v*v);
            LW_ACC(p,dot,v*lw_float(b,(weights/width)*width+i)*lw_float(c,base+i));
        }
        const float inv=1.0f/LW_SQRT(sq/float(width)+p.f[0]);
        const float out=inv*lw_float(b,weights)*lw_float(c,id)-inv*inv*inv*dot*lw_read(a,id)/float(width);
        lw_put(p,x,id,out,p.u[3]!=0);
    } else if(op==2) {
        const float v=lw_round(lw_read(a,id)/float(p.u[2]));
        const float s=lw_sig(v);
        const float derivative=p.u[3]?2.0f*s*(1.0f-s):lw_dsilu(v);
        lw_put(p,x,id,lw_float(b,id)*derivative/float(p.u[2]),false);
    } else if(op==3) {
        const unsigned dim=p.u[1],streams=p.u[2],row=id/(dim*streams),col=id%dim;
        const float up=lw_read(a,id),s=lw_sig(up),grad=lw_float(c,row*dim+col)/float(streams);
        lw_put(p,x,id,grad*lw_round(s),false);
        lw_put(p,y,id,grad*lw_read(b,id)*s*(1.0f-s),false);
    } else if(op==4) {
        const unsigned dim=p.u[1],streams=p.u[2];
        if(id<p.u[0]*dim) {
            const unsigned row=id/dim,col=id%dim;
            float sum=0;
            for(unsigned s=0;s<streams;s++)LW_ACC(p,sum,lw_float(a,(row*streams+s)*dim+col)*lw_read(c,row*streams+s));
            lw_put(p,x,id,sum,false);
        }
        if(id<p.u[0]*streams) {
            const unsigned row=id/streams,stream=id%streams;
            float sum=0;
            for(unsigned i=0;i<dim;i++)LW_ACC(p,sum,lw_float(a,(row*streams+stream)*dim+i)*lw_read(b,row*dim+i));
            lw_put(p,y,id,sum,false);
        }
    } else if(op==5) {
        const float gate=lw_read(a,id),up=lw_read(b,id),grad=lw_float(c,id);
        lw_put(p,x,id,grad*up*lw_dsilu(gate),false);
        lw_put(p,y,id,grad*lw_round(lw_silu(gate)),false);
    } else if(op==6) {
        const unsigned width=p.u[1],row=id;
        const unsigned target=((LW_DEV const unsigned*)b)[row];
        const float weight=lw_float(c,row);
        if(weight==0) {
            for(unsigned i=0;i<width;i++)lw_put(p,x,row*width+i,0,false);
            lw_put(p,y,2*row,0,false);lw_put(p,y,2*row+1,1,false);return;
        }
        float maximum=-LW_INFINITY;
        unsigned pick=0;
        for(unsigned i=0;i<width;i++) { const float v=lw_read(a,row*width+i);if(v>maximum){maximum=v;pick=i;} }
        float total=0;
        for(unsigned i=0;i<width;i++)LW_ACC(p,total,LW_EXP(lw_read(a,row*width+i)-maximum));
        for(unsigned i=0;i<width;i++) {
            const float prob=LW_EXP(lw_read(a,row*width+i)-maximum)/total;
            lw_put(p,x,row*width+i,weight*(prob-(i==target?1.0f:0.0f)),false);
        }
        lw_put(p,y,2*row,maximum+LW_LOG(total)-lw_read(a,row*width+target),false);
        lw_put(p,y,2*row+1,pick==target?1.0f:0.0f,false);
    } else if(op==7) {
        lw_store(p,x,id,lw_read(a,id)+lw_float(b,id));
    } else if(op==8) {
        const unsigned width=p.u[1],dim=p.u[2],row=id/width,col=id%width;
        const unsigned at=row*p.u[3]+(col/dim)*2*dim+dim+col%dim;
        const float gate=lw_read(b,at),s=lw_sig(gate),grad=lw_float(c,id);
        lw_put(p,x,id,grad*lw_round(s),false);
        lw_put(p,y,at,grad*lw_read(a,id)*s*(1.0f-s),false);
    } else if(op==9) {
        const unsigned heads=p.u[1],dim=p.u[2],rot=p.u[3],stride=p.u[4],offset=p.u[5];
        const unsigned head=id/dim,row=head/heads,col=id%dim,base=row*stride+offset+(head%heads)*p.u[6];
        const unsigned rot_half=rot/2;
        float sq=0,dot=0;
        for(unsigned i=0;i<dim;i++) {
            float grad=lw_float(c,head*dim+i);
            if(i<rot) {
                const unsigned mate=i<rot_half?i+rot_half:i-rot_half;
                const float angle=float(row+p.u[7])*lw_float(d,i%rot_half);
                grad=LW_COS(angle)*grad+(i<rot_half?1.0f:-1.0f)*LW_SIN(angle)*lw_float(c,head*dim+mate);
            }
            const float v=lw_read(a,base+i);
            LW_ACC(p,sq,v*v);LW_ACC(p,dot,v*lw_float(b,i)*grad);
        }
        float grad=lw_float(c,id);
        if(col<rot) {
            const unsigned mate=col<rot_half?col+rot_half:col-rot_half;
            const float angle=float(row+p.u[7])*lw_float(d,col%rot_half);
            grad=LW_COS(angle)*grad+(col<rot_half?1.0f:-1.0f)*LW_SIN(angle)*lw_float(c,head*dim+mate);
        }
        const float inv=1.0f/LW_SQRT(sq/float(dim)+p.f[0]);
        lw_put(p,x,base+col,inv*lw_float(b,col)*grad-inv*inv*inv*dot*lw_read(a,base+col)/float(dim),false);
    } else if(op==10 || op==13) {
        const unsigned dim=p.u[1],neurons=p.u[2],cap=p.u[4],row=id/neurons,neuron=id%neurons;
        float gate=0,up=0,back=0;
        for(unsigned i=0;i<dim;i++) {
            const float input=lw_read(a,row*dim+i);
            LW_ACC(p,gate,lw_float(b,neuron*dim+i)*input);
            LW_ACC(p,up,lw_float(b,cap*dim+neuron*dim+i)*input);
            if(op==13)LW_ACC(p,back,lw_float(b,2*cap*dim+neuron*dim+i)*lw_float(c,row*dim+i));
        }
        const float scale=neuron>=p.u[3]?p.f[0]:1.0f;
        if(op==10)lw_put(p,x,id,scale*lw_silu(gate)*up,false);
        else {
            lw_put(p,x,2*id,scale*back*lw_dsilu(gate)*up,false);
            lw_put(p,x,2*id+1,scale*back*lw_silu(gate),false);
        }
    } else if(op==11 || op==12) {
        const unsigned dim=p.u[1],neurons=p.u[2],cap=p.u[4],row=id/dim,col=id%dim;
        float sum=0;
        for(unsigned j=0;j<neurons;j++) {
            if(op==11)LW_ACC(p,sum,lw_float(a,row*neurons+j)*lw_float(b,2*cap*dim+j*dim+col));
            else LW_ACC(p,sum,lw_float(a,2*(row*neurons+j))*lw_float(b,j*dim+col)+
                lw_float(a,2*(row*neurons+j)+1)*lw_float(b,cap*dim+j*dim+col));
        }
        if(op==11)lw_store(p,x,id,lw_read(c,id)+sum);
        else lw_put(p,x,id,sum,true);
    }
}
#endif
