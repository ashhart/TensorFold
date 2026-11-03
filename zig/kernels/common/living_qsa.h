// A fixed selection tape owns every shared-head use; CUDA differentiates its BF16 exponential operand offsets.
#ifndef TF_LIVING_QSA
#define TF_LIVING_QSA
LW_INLINE unsigned lw_qsa_count(LW_DEV const unsigned char* counts,LW_DEV const unsigned char* sparse,unsigned row) {
    return ((LW_DEV const int*)sparse)[row]?unsigned(((LW_DEV const int*)counts)[row]):row+1;
}
LW_INLINE unsigned lw_qsa_key(LW_DEV const unsigned char* ids,LW_DEV const unsigned char* sparse,LwParams p,unsigned row,unsigned j) {
    return ((LW_DEV const int*)sparse)[row]?unsigned(((LW_DEV const int*)ids)[row*p.u[4]+j]):j;
}
LW_INLINE void lw_qsa(unsigned op,unsigned id,LwParams p,
    LW_DEV const unsigned char* a,LW_DEV const unsigned char* b,LW_DEV const unsigned char* c,
    LW_DEV const unsigned char* d,LW_DEV const unsigned char* e,LW_DEV const unsigned char* f,
    LW_DEV const unsigned char* g,LW_DEV const unsigned char* h,
    LW_DEV unsigned char* x,LW_DEV unsigned char* y,LW_DEV unsigned char* z) {
    if(id>=p.u[15])return;
    const unsigned rows=p.u[0],heads=p.u[1],kv=p.u[2],dim=p.u[3];
    if(op==0) {
        const unsigned row=id/heads,head=id%heads,kh=head/(heads/kv),base=id*rows;
        for(unsigned key=0;key<rows;key++){lw_put(p,x,base+key,0,false);lw_put(p,y,base+key,0,false);}
        const unsigned n=lw_qsa_count(f,g,row);
        for(unsigned j=0;j<n;j++) {
            const unsigned key=lw_qsa_key(e,g,p,row,j);
            float score=0,dp=0;
            for(unsigned i=0;i<dim;i++) {
                LW_ACC(p,score,lw_read(a,(row*heads+head)*dim+i)*lw_read(b,(key*kv+kh)*dim+i));
                LW_ACC(p,dp,lw_float(d,(row*heads+head)*dim+i)*lw_read(c,(key*kv+kh)*dim+i));
            }
            lw_put(p,x,base+key,score*p.f[0],false);lw_put(p,y,base+key,dp,false);
        }
    } else if(op==1) {
        const unsigned row=id/heads,base=id*rows,n=lw_qsa_count(d,e,row);
        float maximum=-LW_INFINITY,total=0;
        for(unsigned j=0;j<n;j++) {const unsigned key=lw_qsa_key(c,e,p,row,j);const float s=lw_float(a,base+key);if(s>maximum)maximum=s;}
        for(unsigned j=0;j<n;j++) {const unsigned key=lw_qsa_key(c,e,p,row,j);LW_ACC(p,total,LW_EXP(lw_float(a,base+key)-maximum));}
        for(unsigned key=0;key<rows;key++){lw_put(p,x,base+key,0,false);lw_put(p,y,base+key,0,false);}
        for(unsigned start=0;start<n;start+=32) {
            float tilemax=-LW_INFINITY;
            for(unsigned j=(start/512)*512;j<n && j<start+32;j++) {
                const unsigned key=lw_qsa_key(c,e,p,row,j);const float s=lw_float(a,base+key);if(s>tilemax)tilemax=s;
            }
            for(unsigned j=start;j<n && j<start+32;j++) {
                const unsigned key=lw_qsa_key(c,e,p,row,j);const float score=lw_float(a,base+key);
                const float exponent=LW_EXP(score-tilemax),offset=p.u[5]?lw_round(exponent)-exponent:0.0f;
                const float weight=(LW_EXP(score-maximum)+offset*LW_EXP(tilemax-maximum))/total;
                lw_put(p,x,base+key,weight,false);
            }
        }
        float mean=0;
        for(unsigned j=0;j<n;j++) {const unsigned key=lw_qsa_key(c,e,p,row,j);LW_ACC(p,mean,lw_float(x,base+key)*lw_float(b,base+key));}
        for(unsigned j=0;j<n;j++) {
            const unsigned key=lw_qsa_key(c,e,p,row,j);
            lw_put(p,y,base+key,LW_EXP(lw_float(a,base+key)-maximum)/total*(lw_float(b,base+key)-mean),false);
        }
        if(p.u[5])for(unsigned start=0;start<n;start+=32) {
            float tilemax=-LW_INFINITY,extra=0;unsigned winner=0;
            for(unsigned j=(start/512)*512;j<n && j<start+32;j++) {
                const unsigned key=lw_qsa_key(c,e,p,row,j);const float s=lw_float(a,base+key);if(s>tilemax){tilemax=s;winner=key;}
            }
            for(unsigned j=start;j<n && j<start+32;j++) {
                const unsigned key=lw_qsa_key(c,e,p,row,j);const float v=LW_EXP(lw_float(a,base+key)-tilemax);
                LW_ACC(p,extra,(lw_round(v)-v)*lw_float(b,base+key));
            }
            lw_put(p,y,base+winner,extra*LW_EXP(tilemax-maximum)/total,true);
        }
    } else if(op==2) {
        const unsigned col=id%dim,head=(id/dim)%heads,row=id/(heads*dim),kh=head/(heads/kv),base=(row*heads+head)*rows;
        const unsigned n=lw_qsa_count(d,e,row);
        float sum=0;
        for(unsigned j=0;j<n;j++) {
            const unsigned key=lw_qsa_key(c,e,p,row,j);
            LW_ACC(p,sum,p.f[0]*lw_float(a,base+key)*lw_read(b,(key*kv+kh)*dim+col));
        }
        lw_put(p,x,id,sum,false);
    } else if(op==3) {
        const unsigned col=id%dim,kh=(id/dim)%kv,key=id/(kv*dim),group=heads/kv;
        float dk=0,dv=0;
        for(unsigned row=key;row<rows;row++) {
            const unsigned n=lw_qsa_count(f,g,row);bool found=false;
            for(unsigned j=0;j<n;j++)if(lw_qsa_key(e,g,p,row,j)==key){found=true;break;}
            if(!found)continue;
            for(unsigned head=kh*group;head<(kh+1)*group;head++) {
                const unsigned at=(row*heads+head)*rows+key;
                LW_ACC(p,dk,p.f[0]*lw_float(a,at)*lw_read(c,(row*heads+head)*dim+col));
                LW_ACC(p,dv,lw_float(b,at)*lw_float(d,(row*heads+head)*dim+col));
            }
        }
        lw_put(p,x,id,dk,false);lw_put(p,y,id,dv,false);
    }
}
#endif
