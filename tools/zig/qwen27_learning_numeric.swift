import Foundation
import Metal
let root = URL(fileURLWithPath: CommandLine.arguments[1])
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let options = MTLCompileOptions()
options.languageVersion = .version4_1
options.mathMode = .safe
options.mathFloatingPointFunctions = .fast
var pipes = [String:MTLComputePipelineState]()
for stem in ["gdn_train", "attn_train", "learn_rows", "learn_linear", "pocket"] {
    let folder = root.appendingPathComponent("zig/kernels/metal/" + (stem.hasPrefix("learn_") ? "core" : "qwen27"))
    var source = stem.hasSuffix("train")
      ? try String(contentsOf:folder.appendingPathComponent(stem+"_impl.h"),encoding:.utf8) + "\n" + String(contentsOf:folder.appendingPathComponent(stem+"_entries.h"),encoding:.utf8)
      : try String(contentsOf:folder.appendingPathComponent(stem+".metal"),encoding:.utf8)
    if CommandLine.arguments.contains("--sigmoid-control") && stem == "gdn_train" {
        source = source.replacingOccurrences(of:"const float dn = d * z * sg * w;",with:"const float dn = d * sg * w;")
    }
    if CommandLine.arguments.contains("--rounded-gate-control") && stem == "learn_linear" {
        source = source.replacingOccurrences(of:"dpq[at] = dgated[i] * context * sigmoid * (1.0f - sigmoid);",
            with:"dpq[at] = dgated[i] * float(bfloat(context * sigmoid)) * (1.0f - sigmoid);")
    }
    let library = try device.makeLibrary(source:source,options:options)
    for name in library.functionNames { pipes[name] = try device.makeComputePipelineState(function:library.makeFunction(name:name)!) }
}
func data<T>(_ x:[T]) -> Data { x.withUnsafeBytes { Data($0) } }
func buffer(_ d:Data) -> MTLBuffer {
    var all = d; all.append(Data(repeating:0xa5,count:64))
    return all.withUnsafeBytes { device.makeBuffer(bytes:$0.baseAddress!,length:all.count,options:.storageModeShared)! }
}
func fbuf(_ x:[Float]) -> MTLBuffer { buffer(data(x)) }
func zeros(_ n:Int) -> MTLBuffer { fbuf([Float](repeating:0,count:n)) }
func bf(_ f:Float) -> UInt16 {
    let u=f.bitPattern
    return UInt16(truncatingIfNeeded:(u &+ 0x7fff &+ ((u>>16)&1))>>16)
}
func fv(_ b:UInt16) -> Float { Float(bitPattern:UInt32(b)<<16) }
func bbuf(_ x:[Float]) -> MTLBuffer { buffer(data(x.map(bf))) }
func floats(_ b:MTLBuffer,_ n:Int) -> [Float] { Array(UnsafeBufferPointer(start:b.contents().assumingMemoryBound(to:Float.self),count:n)) }
func shape(_ u:[UInt32],_ f:[Float]=[0,0,0,0]) -> Data { data(u)+data(f) }
func run(_ name:String,_ b:[MTLBuffer],_ p:Data,_ groups:MTLSize,_ threads:Int=256) throws {
    let cb=queue.makeCommandBuffer()!, e=cb.makeComputeCommandEncoder()!, pipeline=pipes[name]!
    precondition(threads<=pipeline.maxTotalThreadsPerThreadgroup)
    e.setComputePipelineState(pipeline)
    for (i,x) in b.enumerated() { e.setBuffer(x,offset:0,index:i) }
    p.withUnsafeBytes { e.setBytes($0.baseAddress!,length:p.count,index:b.count) }
    e.dispatchThreadgroups(groups,threadsPerThreadgroup:MTLSize(width:threads,height:1,depth:1))
    e.endEncoding();cb.commit();cb.waitUntilCompleted()
    if let err=cb.error { throw err }
}
func linear(_ name:String,_ b:[MTLBuffer],_ p:Data,_ n:Int) throws { try run(name,b,p,MTLSize(width:(n+255)/256,height:1,depth:1)) }
var results=[[String:Any]]()
func check(_ name:String,_ out:MTLBuffer,_ expected:[Double],_ tolerance:Double,_ inputs:[MTLBuffer],_ invoke:() throws -> Void) throws {
    let before=inputs.map { Data(bytes:$0.contents(),count:$0.length) }
    var first:Data?=nil, worst=0.0
    for _ in 0..<5 {
        try invoke()
        let values=floats(out,expected.count)
        for (x,y) in zip(values,expected) {
            if !x.isFinite || abs(Double(x)-y)>tolerance { fatalError("\(name): \(x) expected \(y), bound \(tolerance)") }
            worst=max(worst,abs(Double(x)-y))
        }
        let bytes=Data(bytes:out.contents(),count:out.length)
        if let first { precondition(bytes==first,"repeat differs: \(name)") } else { first=bytes }
        precondition(bytes.suffix(64)==Data(repeating:0xa5,count:64),"guard changed")
        for (input,original) in zip(inputs,before) { precondition(Data(bytes:input.contents(),count:input.length)==original,"input changed") }
    }
    results.append(["name":name,"outputs":expected.count,"repeats":5,"worst_abs":worst,"bound":tolerance,"guards":true,"inputs_unchanged":true])
}
// Full-fp32 affine input gradient. Every product and sum here is a dyadic exact float32 value.
do {
    let R=3,N=17,K=64
    let grad=(0..<R*N).map { Float(($0*7)%19-9)/1024 + 0.999267578125 }
    let words=(0..<N*K/8).map { UInt32(truncatingIfNeeded:$0 &* 1550719 &+ 191) }
    let scales=[Float](repeating:1/64,count:N), biases=[Float](repeating:-1/8,count:N)
    var expected=[Double](repeating:0,count:R*K)
    for r in 0..<R { for k in 0..<K { for n in 0..<N {
        let code=(words[(n*K+k)/8]>>UInt32(4*(k%8)))&15
        expected[r*K+k] += Double(grad[r*N+n])*(Double(scales[n])*Double(code)+Double(biases[n]))
    } } }
    let inputs=[fbuf(grad),buffer(data(words)),bbuf(scales),bbuf(biases)], out=zeros(R*K)
    try check("affine4 input gradient exact",out,expected,0,inputs) {
        try linear("tf_learn_q4_back",inputs+[out],shape([UInt32(R),UInt32(N),UInt32(K),0]),((R+3)/4)*K)
    }
}
// RMS input derivative against the independent binary64 expression, including a partial vector tail.
do {
    let R=3,D=65,eps=Float(1e-6)
    let x=(0..<R*D).map { Float(($0*17)%31-15)/16 }, w=(0..<D).map { Float(($0%7)+1)/8 }
    let dy=(0..<R*D).map { Float(($0*13)%23-11)/32 }
    let inputs=[bbuf(x),bbuf(w),fbuf(dy)], out=zeros(R*D)
    var expected=[Double](repeating:0,count:R*D)
    for r in 0..<R {
        var square=0.0,dot=0.0
        for d in 0..<D { let v=Double(x[r*D+d]); square+=v*v; dot+=v*Double(w[d])*Double(dy[r*D+d]) }
        let inv=1/sqrt(square/Double(D)+Double(eps))
        for d in 0..<D { expected[r*D+d]=inv*Double(w[d])*Double(dy[r*D+d])-inv*inv*inv*Double(x[r*D+d])*dot/Double(D) }
    }
    try check("RMS input gradient",out,expected,2e-6,inputs) {
        // RMS adds to the residual gradient. Reset through the real zero kernel for each repetition.
        try linear("tf_learn_zero",[out],shape([UInt32(R*D),0,0,0]),R*D)
        try run("tf_learn_rms_back",inputs+[out],shape([UInt32(R),UInt32(D),0,0],[eps,0,0,0]),MTLSize(width:R,height:1,depth:1))
    }
}
// SwiGLU's native gate/up layout and full SiLU derivative, central differences of a smooth BF16-node STE.
do {
    let R=3,D=65
    let x=(0..<R*2*D).map { Float(($0*17)%41-20)/16 }, dy=(0..<R*D).map { Float(($0*7)%19-9)/32 }
    var expected=[Double](repeating:0,count:R*2*D), worstFD=0.0
    func loss(_ gate:Double,_ up:Double,_ g:Double)->Double { gate/(1+exp(-gate))*up*g }
    for r in 0..<R { for d in 0..<D {
        let at=r*2*D+d, gate=Double(x[at]), up=Double(x[at+D]), g=Double(dy[r*D+d]), sigmoid=1/(1+exp(-gate))
        expected[at]=g*up*sigmoid*(1+gate*(1-sigmoid)); expected[at+D]=g*gate*sigmoid
        let h=1e-5
        worstFD=max(worstFD,abs((loss(gate+h,up,g)-loss(gate-h,up,g))/(2*h)-expected[at]))
        worstFD=max(worstFD,abs((loss(gate,up+h,g)-loss(gate,up-h,g))/(2*h)-expected[at+D]))
    } }
    precondition(worstFD<1e-9)
    let inputs=[bbuf(x),fbuf(dy)], out=zeros(R*2*D)
    try check("SwiGLU input gradient",out,expected,2e-6,inputs) {
        try linear("tf_learn_swiglu_back",inputs+[out],shape([UInt32(R),UInt32(D),0,0]),R*D)
    }
    results.append(["name":"SwiGLU binary64 central difference","worst_abs":worstFD,"bound":1e-9])
}
// The Metal GDN's output gate is SiLU. z=0 has zero y-gradient but a nonzero gate gradient.
do {
    let R=3,NK=1,NV=2,D=128,conv=2*NK*D+NV*D,proj=conv+NV*D+2*NV
    let eps=Float(1e-6)
    let y=(0..<R*NV*D).map { Float(($0*13)%29-14)/16 }
    let w=(0..<D).map { Float(($0%7)+1)/8 }, dy=(0..<R*NV*D).map { Float(($0*7)%17-8)/32 }
    var p=[Float](repeating:0,count:R*proj)
    for r in 0..<R { for h in 0..<NV { for d in 0..<D { p[r*proj+conv+h*D+d] = Float((r*NV*D+h*D+d)%23-11)/8 } } }
    p[conv]=0
    var expectedY=[Double](repeating:0,count:R*NV*D), expectedP=[Double](repeating:0,count:R*proj), worstFD=0.0
    for r in 0..<R { for h in 0..<NV {
        let row=(r*NV+h)*D
        var ss=0.0,dot=0.0
        for d in 0..<D { let v=Double(y[row+d]);ss+=v*v }
        let inv=1/sqrt(ss/Double(D)+Double(eps))
        var dn=[Double](repeating:0,count:D)
        for d in 0..<D {
            let z=Double(p[r*proj+conv+h*D+d]), sg=1/(1+exp(-z)), delta=Double(dy[row+d])
            let normalized=Double(fv(bf(Float(Double(y[row+d])*inv*Double(w[d])))))
            expectedP[r*proj+conv+h*D+d]=delta*normalized*sg*(1+z*(1-sg))
            dn[d]=delta*z*sg*Double(w[d]);dot+=Double(y[row+d])*dn[d]
            let step=1e-5
            func objective(_ zz:Double)->Double { delta*normalized*zz/(1+exp(-zz)) }
            let fd=(objective(z+step)-objective(z-step))/(2*step)
            worstFD=max(worstFD,abs(fd-expectedP[r*proj+conv+h*D+d]))
        }
        for d in 0..<D { expectedY[row+d]=inv*dn[d]-inv*inv*inv*Double(y[row+d])*dot/Double(D) }
    } }
    precondition(worstFD<1e-9)
    let inputs=[bbuf(y),bbuf(p),bbuf(w),fbuf(dy)], outY=zeros(R*NV*D),outP=zeros(R*proj)
    let params=data([Int32(R),Int32(NK),Int32(NV),Int32(conv),Int32(proj)])+data([eps])
    let invoke = { try run("tf_gdn_norm_back",inputs+[outY,outP],params,MTLSize(width:R,height:NV,depth:1),128) }
    try check("GDN SiLU gated RMS input gradient",outY,expectedY,4e-6,inputs,invoke)
    try check("GDN SiLU output-gate gradient",outP,expectedP,4e-6,inputs,invoke)
    results.append(["name":"GDN SiLU binary64 central difference","worst_abs":worstFD,"bound":1e-9])
}
// Native Q/K RMS normalizers, including both BF16 scale policies and every grouped head contribution.
do {
    let R=3,NK=1,NV=2,D=128,conv=2*NK*D+NV*D,proj=conv+NV*D+2*NV
    let act=(0..<R*conv).map { Float(($0*13)%31-15)/16 }
    let qp=(0..<R*NV*4*D).map { Float(($0*7)%23-11)/1024 }
    let kp=(0..<R*NV*4*D).map { Float(($0*11)%29-14)/1024 }
    var expected=[Double](repeating:0,count:R*conv)
    for r in 0..<R { for which in 0..<2 {
        let parts=which==0 ? qp : kp
        var ss=0.0,dot=0.0,delta=[Double](repeating:0,count:D)
        for d in 0..<D {
            for h in 0..<NV { for part in 0..<4 { delta[d]+=Double(parts[((r*NV+h)*4+part)*D+d]) } }
            let x=Double(act[r*conv+which*D+d]);ss+=x*x;dot+=x*delta[d]
        }
        let inv=1/sqrt(ss/Double(D)+1e-6)
        let scale=Double(fv(bf(which==0 ? Float(1)/Float(D) : Float(1)/sqrt(Float(D)))))
        for d in 0..<D {
            let x=Double(act[r*conv+which*D+d])
            expected[r*conv+which*D+d]=scale*inv*(delta[d]-inv*inv*x*dot/Double(D))
        }
    } }
    let inputs=[bbuf(act),fbuf(qp),fbuf(kp)], out=zeros(R*conv)
    let params=data([Int32(R),Int32(NK),Int32(NV),Int32(conv),Int32(proj)])
    try check("GDN native Q/K RMS gradient",out,expected,3e-7,inputs) {
        try run("tf_gdn_qk_back",inputs+[out],params,MTLSize(width:R,height:NK,depth:1),128)
    }
}
// The delta-rule recurrence's reverse pass, independently differentiated in binary64.
do {
    let R=5,NK=1,NV=2,D=128,C=2*D+NV*D,P=C+NV*D+2*NV,parts=4
    let act=(0..<R*C).map { Float(($0*7)%23-11)/32 }
    let q=(0..<R*D).map { Float(($0*13)%17-8)/256 }
    let k=(0..<R*D).map { Float(($0*11)%19-9)/128 }
    let decay=(0..<R*NV).map { Float(($0%3)+5)/8 }
    let beta=(0..<R*NV).map { Float(($0%3)+1)/8 }
    let dy=(0..<R*NV*D).map { Float(($0*17)%31-15)/64 }
    let stateSize=NV*D*D
    var states=[[Double]](), state=[Double](repeating:0,count:stateSize)
    var checkpoints=[Float](repeating:0,count:2*stateSize)
    var err=[Double](repeating:0,count:R*NV*D)
    for r in 0..<R {
        states.append(state)
        if r%4==0 { for i in 0..<stateSize { checkpoints[(r/4)*stateSize+i]=Float(state[i]) } }
        for h in 0..<NV { for v in 0..<D {
            let base=(h*D+v)*D
            var kv=0.0
            for d in 0..<D { kv+=state[base+d]*Double(decay[r*NV+h])*Double(k[r*D+d]) }
            let error=Double(act[r*C+2*D+h*D+v])-kv
            err[(r*NV+h)*D+v]=error
            for d in 0..<D {
                let sd=state[base+d]*Double(decay[r*NV+h])
                state[base+d]=sd+Double(k[r*D+d])*error*Double(beta[r*NV+h])
            }
        } }
    }
    var ds=[Double](repeating:0,count:stateSize)
    var dq=[Double](repeating:0,count:R*NV*D), dk=dq
    var dv=[Double](repeating:0,count:R*C), dg=[Double](repeating:0,count:R*NV), db=dg
    for r in stride(from:R-1,through:0,by:-1) {
        for h in 0..<NV { for v in 0..<D {
            let base=(h*D+v)*D, delta=err[(r*NV+h)*D+v]*Double(beta[r*NV+h])
            let dout=Double(dy[(r*NV+h)*D+v])
            var dd=0.0
            for d in 0..<D {
                let sd=states[r][base+d]*Double(decay[r*NV+h])
                let after=sd+Double(k[r*D+d])*delta
                dq[(r*NV+h)*D+d]+=dout*after
                ds[base+d]+=dout*Double(q[r*D+d])
                dd+=ds[base+d]*Double(k[r*D+d])
            }
            dv[r*C+2*D+h*D+v]=Double(beta[r*NV+h])*dd
            db[r*NV+h]+=dd*err[(r*NV+h)*D+v]
            let dkv = -Double(beta[r*NV+h])*dd
            for d in 0..<D {
                let sd=states[r][base+d]*Double(decay[r*NV+h])
                dk[(r*NV+h)*D+d]+=delta*ds[base+d]+dkv*sd
                let dsp=ds[base+d]+dkv*Double(k[r*D+d])
                dg[r*NV+h]+=dsp*states[r][base+d]
                ds[base+d]=Double(decay[r*NV+h])*dsp
            }
        } }
    }
    let inputs=[bbuf(act),fbuf(q),fbuf(k),fbuf(decay),fbuf(beta),fbuf(checkpoints),fbuf(dy)]
    let work=zeros(NV*parts*4*32*D), dqp=zeros(R*NV*parts*D), dkp=zeros(R*NV*parts*D)
    let da=zeros(R*C), dgp=zeros(R*NV*parts), dbp=zeros(R*NV*parts)
    let params=data([Int32(R),Int32(NK),Int32(NV),Int32(C),Int32(P),0])
    let before=inputs.map { Data(bytes:$0.contents(),count:$0.length) }
    var first:[Data]?=nil,worst=0.0
    for _ in 0..<5 {
        try run("tf_gdn_scan_back",inputs+[work,dqp,dkp,da,dgp,dbp],params,MTLSize(width:NV*parts,height:1,depth:1))
        let oq=floats(dqp,R*NV*parts*D),ok=floats(dkp,R*NV*parts*D),og=floats(dgp,R*NV*parts),ob=floats(dbp,R*NV*parts)
        var got=[Double](), want=[Double]()
        for r in 0..<R { for h in 0..<NV {
            var gg=0.0,bb=0.0
            for part in 0..<parts { gg+=Double(og[(r*NV+h)*parts+part]);bb+=Double(ob[(r*NV+h)*parts+part]) }
            got.append(contentsOf:[gg,bb]);want.append(contentsOf:[dg[r*NV+h],db[r*NV+h]])
            for d in 0..<D {
                var qq=0.0,kk=0.0
                for part in 0..<parts { qq+=Double(oq[((r*NV+h)*parts+part)*D+d]);kk+=Double(ok[((r*NV+h)*parts+part)*D+d]) }
                got.append(contentsOf:[qq,kk]);want.append(contentsOf:[dq[(r*NV+h)*D+d],dk[(r*NV+h)*D+d]])
            }
        } }
        got.append(contentsOf:floats(da,R*C).map(Double.init));want.append(contentsOf:dv)
        for (x,y) in zip(got,want) { precondition(x.isFinite && abs(x-y)<2e-6,"GDN scan gradient differs: \(x) vs \(y)");worst=max(worst,abs(x-y)) }
        let all=[work,dqp,dkp,da,dgp,dbp].map { Data(bytes:$0.contents(),count:$0.length) }
        if let first { precondition(all==first,"scan repeat differs") } else { first=all }
        for bytes in all { precondition(bytes.suffix(64)==Data(repeating:0xa5,count:64)) }
        for (input,original) in zip(inputs,before) { precondition(Data(bytes:input.contents(),count:input.length)==original) }
    }
    results.append(["name":"GDN recurrence gradients, two heads, checkpoint edge row 4","repeats":5,"worst_abs":worst,"bound":2e-6,"guards":true,"inputs_unchanged":true])
}
// Causal grouped-query softmax gradients at a 3-row seam, independent binary64 probabilities and reverse products.
do {
    let R=3,H=2,KH=1,D=256,Q=H*D,K=KH*D,W=Q+2*K
    let qkv=(0..<R*W).map { Float(($0*13)%31-15)/64 }
    let dout=(0..<R*Q).map { Float(($0*7)%23-11)/128 }
    let scale=Float(1)/sqrt(Float(D))
    var probs=[Double](repeating:0,count:H*R*R), ds=probs
    var dq=[Double](repeating:0,count:R*Q),dk=[Double](repeating:0,count:R*K),dv=dk
    for h in 0..<H { for i in 0..<R {
        var score=[Double](repeating:0,count:i+1),dp=score
        for j in 0...i { for d in 0..<D {
            score[j]+=Double(qkv[i*W+h*D+d])*Double(qkv[j*W+Q+d])*Double(scale)
            dp[j]+=Double(dout[i*Q+h*D+d])*Double(qkv[j*W+Q+K+d])
        } }
        let top=score.max()!, exponent=score.map { exp($0-top) }, total=exponent.reduce(0,+)
        let probability=exponent.map { $0/total }
        var mix=0.0
        for j in 0...i { mix+=probability[j]*dp[j] }
        for j in 0...i {
            let at=(h*R+i)*R+j,p=probability[j],delta=p*(dp[j]-mix)
            probs[at]=p;ds[at]=delta
            for d in 0..<D {
                dq[i*Q+h*D+d]+=delta*Double(qkv[j*W+Q+d])*Double(scale)
                dk[j*K+d]+=delta*Double(qkv[i*W+h*D+d])*Double(scale)
                dv[j*K+d]+=p*Double(dout[i*Q+h*D+d])
            }
        }
    } }
    let input=bbuf(qkv),grad=fbuf(dout),prob=zeros(H*R*R),scores=zeros(H*R*R),outQ=zeros(R*Q),outK=zeros(R*K),outV=zeros(R*K),dummy=zeros(1)
    let params=data([Int32(R),Int32(H),Int32(KH),Int32(D),Int32(W)])+data([scale])+data([Int32(0)])
    let invoke={ try run("tf_attn_q",[input,grad,prob,scores,outQ,dummy,dummy],params,MTLSize(width:H,height:R,depth:1)) }
    try check("attention query gradient",outQ,dq,2e-6,[input,grad],invoke)
    try check("attention probabilities",prob,probs,2e-6,[input,grad],invoke)
    try check("attention score gradient",scores,ds,2e-6,[input,grad],invoke)
    let kvInvoke={
        try invoke()
        try linear("tf_learn_attn_kv_back",[input,grad,prob,scores,outK,outV],shape([UInt32(R),UInt32(H),UInt32(KH),UInt32(D)],[scale,0,0,0]),R*K)
    }
    try check("attention grouped key gradient",outK,dk,2e-6,[input,grad],kvInvoke)
    try check("attention grouped value gradient",outV,dv,2e-6,[input,grad],kvInvoke)
}
// Output-gate derivatives use the pre-gate probability product, not a rounded BF16 gated output.
do {
    let R=3,H=2,KH=1,D=256,Q=H*D,K=KH*D,W=Q+2*K
    let qkv=(0..<R*W).map { Float(($0*13)%31-15)/16 }
    let dgated=(0..<R*Q).map { Float(($0*7)%23-11)/16 }
    var pq=[Float](repeating:0,count:R*2*Q),prob=[Float](repeating:0,count:H*R*R)
    var expected=[Double](repeating:0,count:R*2*Q),worstFD=0.0
    for h in 0..<H { for row in 0..<R {
        // Exact dyadic probabilities, row sums one; the core softmax gradient is tested separately.
        for j in 0...row { prob[(h*R+row)*R+j] = j == row ? (row == 0 ? 1 : 0.5) : (row == 1 ? 0.5 : 0.25) }
        for d in 0..<D {
            let at=row*2*Q+h*2*D+D+d
            pq[at]=Float((row+h+d)%17-8)/8
            var context=0.0
            for j in 0...row { context+=Double(prob[(h*R+row)*R+j])*Double(qkv[j*W+Q+K+d]) }
            let z=Double(pq[at]),sg=1/(1+exp(-z)),g=Double(dgated[row*Q+h*D+d])
            expected[at]=g*context*sg*(1-sg)
            let step=1e-5
            let fd=g*context*(1/(1+exp(-(z+step)))-1/(1+exp(-(z-step))))/(2*step)
            worstFD=max(worstFD,abs(fd-expected[at]))
        }
    } }
    precondition(worstFD<1e-9)
    let inputs=[fbuf(dgated),bbuf(pq),bbuf(qkv),fbuf(prob)],out=zeros(R*2*Q)
    try check("attention pre-gate context derivative",out,expected,2e-6,inputs) {
        try linear("tf_learn_attn_gate_grad",inputs+[out],shape([UInt32(R),UInt32(H),UInt32(KH),UInt32(D)]),R*Q)
    }
    results.append(["name":"attention gate binary64 central difference","worst_abs":worstFD,"bound":1e-9])
}
// The no-open-neuron sentinel must stay an integer: Float(UInt32.max) rounds up to 2^32.
do {
    let R=2,D=65,CAP=64,N=3
    let x=(0..<R*D).map { Float(($0*7)%19-9)/16 }
    let grad=(0..<R*D).map { Float(($0*11)%23-11)/32 }
    var weights=[Float](repeating:0,count:3*CAP*D)
    for part in 0..<3 { for j in 0..<N { for d in 0..<D {
        weights[(part*CAP+j)*D+d]=Float((part*7+j*13+d*17)%23-11)/128
    } } }
    let inputs=[bbuf(x),fbuf(weights),fbuf(grad)],out=zeros(R*D)
    for (from,scale) in [(UInt32.max,Float(0)),(UInt32(1),Float(0)),(UInt32(0),Float(0.5)),(UInt32(0),Float(1))] {
        var expected=[Double](repeating:0,count:R*D)
        for r in 0..<R { for j in 0..<N {
            var gate=0.0,up=0.0,error=0.0
            for d in 0..<D {
                gate+=Double(x[r*D+d])*Double(weights[j*D+d])
                up+=Double(x[r*D+d])*Double(weights[(CAP+j)*D+d])
                error+=Double(grad[r*D+d])*Double(weights[(2*CAP+j)*D+d])
            }
            let sigmoid=1/(1+exp(-gate)),sc=UInt32(j)>=from ? Double(scale) : 1.0
            let cg=sc*error*up*sigmoid*(1+gate*(1-sigmoid)),cu=sc*error*gate*sigmoid
            for d in 0..<D { expected[r*D+d]+=cg*Double(weights[j*D+d])+cu*Double(weights[(CAP+j)*D+d]) }
        } }
        try check("pocket input gradient from=\(from) scale=\(scale)",out,expected,2e-6,inputs) {
            try linear("tf_learn_zero",[out],shape([UInt32(R*D),0,0,0]),R*D)
            try run("tf_learn_pocket_back",inputs+[out],shape([from,UInt32(D),UInt32(N),UInt32(CAP)],[scale,0,0,0]),MTLSize(width:R,height:1,depth:1))
        }
    }
}
let encoded=try JSONSerialization.data(withJSONObject:["pipelines":pipes.keys.sorted(),"tests":results],options:[.prettyPrinted,.sortedKeys])
FileHandle.standardOutput.write(encoded);FileHandle.standardOutput.write(Data([10]))
