// Native delta-rule input adjoints use the same recorded state rows that serving wrote, with no atomic sums.
#ifndef TF_LIVING_GDN
#define TF_LIVING_GDN
LW_INLINE unsigned lw_gdn_channels(LwParams p) { return (2*p.u[1]+p.u[2])*128; }
LW_INLINE float lw_gdn_conv(LW_DEV const unsigned char* raw,LW_DEV const unsigned char* weight,
    LwParams p,unsigned row,unsigned channel) {
    float v=0;
    for(unsigned tap=0;tap<4;tap++) {
        const int at=int(row)+int(tap)-3;
        if(at>=0)LW_ACC(p,v,lw_read(raw,unsigned(at)*p.u[3]+channel)*lw_float(weight,channel*4+tap));
    }
    return p.u[4]?lw_round(v):v;
}
LW_INLINE float lw_gdn_decay(LW_DEV const unsigned char* raw,LW_DEV const unsigned char* alog,
    LW_DEV const unsigned char* dt,LwParams p,unsigned row,unsigned head) {
    const unsigned ch=lw_gdn_channels(p),nv=p.u[2];
    const float a=lw_read(raw,row*p.u[3]+ch+nv*129+head)+lw_float(dt,head);
    const float sp=a>20?a:LW_LOG(1.0f+LW_EXP(a));
    return LW_EXP(-LW_EXP(lw_float(alog,head))*sp);
}
LW_INLINE float lw_gdn_y(LW_DEV const unsigned char* qk,LW_DEV const unsigned char* state,
    LwParams p,unsigned row,unsigned head,unsigned value) {
    const unsigned kh=head/(p.u[2]/p.u[1]);
    float v=0;
    for(unsigned i=0;i<128;i++)LW_ACC(p,v,lw_float(state,((row*p.u[2]+head)*128+value)*128+i)*
        lw_float(qk,(row*2*p.u[1]+kh)*128+i));
    return lw_round(v);
}
LW_INLINE void lw_gdn(unsigned op,unsigned id,LwParams p,
    LW_DEV const unsigned char* a,LW_DEV const unsigned char* b,LW_DEV const unsigned char* c,
    LW_DEV const unsigned char* d,LW_DEV const unsigned char* e,LW_DEV const unsigned char* f,
    LW_DEV const unsigned char* g,LW_DEV const unsigned char* h,
    LW_DEV unsigned char* x,LW_DEV unsigned char* y,LW_DEV unsigned char* z) {
    if(id>=p.u[15])return;
    const unsigned nk=p.u[1],nv=p.u[2],ch=lw_gdn_channels(p),pw=p.u[3];
    if(op==0) {
        const unsigned row=id/ch,col=id%ch;
        lw_store(p,x,id,lw_silu(lw_gdn_conv(a,b,p,row,col)));
    } else if(op==1) {
        const unsigned block=id/128,col=id%128,row=block/(2*nk),head=block%(2*nk);
        float sq=0;
        for(unsigned i=0;i<128;i++){const float v=lw_read(a,row*ch+head*128+i);LW_ACC(p,sq,v*v);}
        const float inv=1.0f/LW_SQRT(sq+1e-6f);
        lw_put(p,x,id,lw_read(a,row*ch+head*128+col)*inv*(head<nk?0.08838834764831845f:1.0f),false);
    } else if(op==2) {
        const unsigned block=id/128,col=id%128,row=block/nv,head=block%nv;
        const float gate_raw=lw_read(a,row*pw+ch+head*128+col);
        const float gate=p.u[5]?lw_sig(gate_raw):lw_silu(gate_raw);
        const float gate_derivative=p.u[5]?lw_sig(gate_raw)*(1.0f-lw_sig(gate_raw)):lw_dsilu(gate_raw);
        float sq=0,dot=0;
        for(unsigned i=0;i<128;i++) {
            const float v=lw_gdn_y(b,c,p,row,head,i);
            const float raw=lw_read(a,row*pw+ch+head*128+i);
            const float gain=p.u[5]?lw_sig(raw):lw_silu(raw);
            LW_ACC(p,sq,v*v);LW_ACC(p,dot,v*lw_float(d,i)*gain*lw_float(e,(row*nv+head)*128+i));
        }
        const float v=lw_gdn_y(b,c,p,row,head,col),inv=1.0f/LW_SQRT(sq/128.0f+p.f[0]);
        const float grad=lw_float(e,id),weight=lw_float(d,col);
        const float dy=inv*weight*gate*grad-inv*inv*inv*dot*v/128.0f;
        lw_put(p,x,id,dy,false);
        float stored=lw_round(v*inv)*weight;
        if(p.u[6])stored=lw_round(stored);
        lw_put(p,y,row*pw+ch+head*128+col,grad*stored*gate_derivative,false);
    } else if(op==3) {
        const unsigned head=id,kh=head/(nv/nk),work=head*128*128;
        for(unsigned i=0;i<128*128;i++)lw_put(p,y,work+i,0,false);
        for(int row=int(p.u[0])-1;row>=0;row--) {
            const unsigned r=unsigned(row),part=(r*nv+head)*386;
            for(unsigned i=0;i<386;i++)lw_put(p,x,part+i,0,false);
            const float decay=lw_gdn_decay(e,f,g,p,r,head);
            const float beta=lw_round(lw_sig(lw_read(e,r*pw+ch+nv*128+head)));
            for(unsigned value=0;value<128;value++) {
                float pred=0,dc=0;
                const float gy=lw_float(d,(r*nv+head)*128+value);
                for(unsigned i=0;i<128;i++) {
                    const unsigned at=work+value*128+i;
                    const float q=lw_float(b,(r*2*nk+kh)*128+i),key=lw_float(b,(r*2*nk+nk+kh)*128+i);
                    const float prev=r?lw_float(c,((r-1)*nv*128+head*128+value)*128+i):0.0f;
                    LW_ACC(p,pred,decay*prev*key);
                    lw_put(p,y,at,gy*q,true);
                    LW_ACC(p,dc,lw_float(y,at)*key);
                    lw_put(p,x,part+i,gy*lw_float(c,((r*nv+head)*128+value)*128+i),true);
                }
                const float delta=lw_read(a,r*ch+2*nk*128+head*128+value)-pred;
                const float correction=beta*delta;
                lw_put(p,x,part+256+value,beta*dc,false);
                lw_put(p,x,part+385,delta*dc,true);
                for(unsigned i=0;i<128;i++) {
                    const unsigned at=work+value*128+i;
                    const float key=lw_float(b,(r*2*nk+nk+kh)*128+i),gs=lw_float(y,at);
                    const float prev=r?lw_float(c,((r-1)*nv*128+head*128+value)*128+i):0.0f;
                    lw_put(p,x,part+128+i,gs*correction-dc*beta*decay*prev,true);
                    const float before=gs-dc*beta*key;
                    lw_put(p,x,part+384,before*prev,true);
                    lw_put(p,y,at,decay*before,false);
                }
            }
        }
    } else if(op==4) {
        const unsigned row=id/ch,col=id%ch;
        if(col>=2*nk*128) {
            const unsigned v=col-2*nk*128,head=v/128;
            lw_put(p,x,id,lw_float(c,(row*nv+head)*386+256+v%128),false);
        } else {
            const bool key=col>=nk*128;
            const unsigned head=(col%(nk*128))/128,component=col%128,group=nv/nk;
            float sq=0,dot=0,own=0;
            for(unsigned i=0;i<128;i++) {
                const float v=lw_read(a,row*ch+(key?nk*128:0)+head*128+i);
                float grad=0;
                for(unsigned j=0;j<group;j++)LW_ACC(p,grad,lw_float(c,(row*nv+head*group+j)*386+(key?128:0)+i));
                LW_ACC(p,sq,v*v);LW_ACC(p,dot,v*grad);if(i==component)own=grad;
            }
            const float v=lw_read(a,id),inv=1.0f/LW_SQRT(sq+1e-6f),scale=key?1.0f:0.08838834764831845f;
            lw_put(p,x,id,scale*(inv*own-inv*inv*inv*dot*v),false);
        }
    } else if(op==5) {
        const unsigned row=id/nv,head=id%nv,part=id*386;
        const float beta=lw_sig(lw_read(b,row*pw+ch+nv*128+head));
        const float raw=lw_read(b,row*pw+ch+nv*129+head)+lw_float(d,head);
        const float decay=lw_gdn_decay(b,c,d,p,row,head);
        const float da=lw_float(a,part+384)*decay*(-LW_EXP(lw_float(c,head)))*(raw>20?1.0f:lw_sig(raw));
        lw_put(p,x,row*pw+ch+nv*128+head,lw_float(a,part+385)*beta*(1.0f-beta),false);
        lw_put(p,x,row*pw+ch+nv*129+head,da,false);
    } else if(op==6) {
        const unsigned row=id/ch,col=id%ch;
        float sum=0;
        for(unsigned future=0;future<4 && row+future<p.u[0];future++) {
            const unsigned at=row+future;
            LW_ACC(p,sum,lw_float(c,at*ch+col)*lw_dsilu(lw_gdn_conv(a,b,p,at,col))*lw_float(b,col*4+3-future));
        }
        lw_put(p,x,row*pw+col,sum,false);
    }
}
#endif
