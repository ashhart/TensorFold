// Native Gemma pullbacks preserve each explicit BF16 coefficient and use FP32 sums between stored nodes.
#ifndef TF_LIVING_GEMMA
#define TF_LIVING_GEMMA
LW_INLINE void lw_gemma(unsigned op,unsigned id,LwParams p,
    LW_DEV const unsigned char* a,LW_DEV const unsigned char* b,LW_DEV const unsigned char* c,
    LW_DEV const unsigned char* d,LW_DEV const unsigned char* e,LW_DEV const unsigned char* f,
    LW_DEV const unsigned char* g,LW_DEV const unsigned char* h,
    LW_DEV unsigned char* x,LW_DEV unsigned char* y,LW_DEV unsigned char* z) {
    if(id>=p.u[15])return;
    const unsigned width=p.u[1],row=id/width,col=id%width;
    if(op==0){
        const unsigned stride=p.u[2]?p.u[2]:width;
        const unsigned at=p.u[3]==1?(col/p.u[4])*p.u[0]*p.u[4]+row*p.u[4]+col%p.u[4]:
            row*stride+(p.u[3]==2?(col/16)*32+col%16:col);
        lw_store(p,x,id,lw_read(a,at));
    }else if(op==1)lw_put(p,x,id,0,false);
    else if(op==2)lw_put(p,x,id,p.f[0]*lw_float(a,id),p.u[2]!=0);
    else if(op==3){
        const unsigned base=(id/width)*width;
        float sq=0,dot=0;
        for(unsigned i=0;i<width;i++){
            const float v=lw_read(a,base+i),w=p.u[3]?lw_read(b,i):1.0f;
            LW_ACC(p,sq,v*v);LW_ACC(p,dot,v*w*lw_float(c,base+i));
        }
        const float inv=1.0f/LW_SQRT(sq/float(width)+p.f[0]),w=p.u[3]?lw_read(b,col):1.0f;
        lw_put(p,x,id,inv*w*lw_float(c,id)-inv*inv*inv*dot*lw_read(a,id)/float(width),false);
    }else if(op==4){
        const float v=lw_read(a,id),up=lw_read(b,id),dy=lw_float(c,id);
        const float c0=lw_round(0.7978845608028654f),c1=lw_round(0.044715f);
        const float cube=lw_round(v*v*v),scaled=lw_round(c1*cube),inner=lw_round(v+scaled);
        const float theta=lw_round(c0*inner),th=LW_TANH(theta),stored=lw_round(th),plus=lw_round(1.0f+stored);
        const float gl=lw_round((0.5f*v)*plus),dt=1.0f-th*th;
        const float derivative=0.5f*plus+(dt==0?0.0f:(0.5f*v)*dt*c0*(1.0f+3.0f*c1*v*v));
        lw_put(p,x,id,dy*up*derivative,false);lw_put(p,y,id,dy*gl,false);
    }else if(op==5){
        const float theta=lw_round(lw_read(a,id)/p.f[0]),th=LW_TANH(theta);
        lw_put(p,x,id,lw_float(b,id)*(1.0f-th*th),false);
    }else if(op==6)lw_put(p,x,id,lw_float(a,id)*lw_read(b,0),p.u[2]!=0);
    else if(op==7){
        const unsigned dim=p.u[2],rot_half=dim/2,head_col=col%dim;
        const unsigned pair=head_col%rot_half,base=id-head_col;
        const float angle=float(row+p.u[3])*lw_float(b,pair),co=LW_COS(angle),si=LW_SIN(angle);
        const float left=lw_float(a,base+pair),right=lw_float(a,base+rot_half+pair);
        lw_put(p,x,id,head_col<rot_half?left*co+right*si:right*co-left*si,false);
    }
}
#endif
