import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { EngineService } from './service'
import { builtinEngineProvider } from './provider'
import { modelPickerChoices } from '../../../src/lib/model-picker'
import { makeHarness,seedModel } from './test-fixtures/harness'
const h=await makeHarness(),model='TensorFold/Qwen3.8-27B-MLX-4bit'
await seedModel(h.hub,model)
const repo=path.join(h.hub,'models--TensorFold--Qwen3.8-27B-MLX-4bit')
const header=Buffer.from(JSON.stringify({weight:{dtype:'F32',shape:[2],data_offsets:[0,8]}})),prefix=Buffer.alloc(8);prefix.writeBigUInt64LE(BigInt(header.length))
await fs.writeFile(path.join(repo,'blobs','weights'),Buffer.concat([prefix,header,Buffer.alloc(8,1)]))
await fs.writeFile(path.join(repo,'snapshots','a'.repeat(40),'tokenizer.json'),'{}')
const profile={contextTokens:8192,maxOutputTokens:1024,temperature:0.2,topP:1,topK:0,minP:0,speculativeDecoding:'off' as const}
const service=new EngineService({binaryPath:'/synthetic',binary:h.binary,appHome:h.root,childEnv:h.env,totalMemoryBytes:32*1024**3,pollMs:15,readyTimeoutMs:3000,modelSettings:()=>profile,emit:()=>{}})
try{
 const status=await service.load(model)
 assert.equal(status.state,'ready');assert.equal(status.drafter,null)
 const snapshot=await service.snapshot(),provider={...builtinEngineProvider(),models:[model]}
 const installed=snapshot.installed.find(x=>x.repo===model)!
 const readiness=await service.readiness(model)
 const choice=modelPickerChoices([provider],snapshot).find(x=>x.model===model)!
 const library=await service.scanLibrary({includeHuggingFace:true})
 const alias=library.models.find(x=>x.repo===model)!
 assert.equal(installed.drafter,null,'Off removes the companion requirement from installed metadata')
 const failures=[]
 if(!readiness.ready)failures.push('Off profile must pass readiness without companion')
 if(choice.disabledReason)failures.push('Off profile must remain selectable without companion')
 if(!alias?.loadable)failures.push('Off profile must remain loadable in library without companion')
 assert.deepEqual(failures,[])
}finally{await service.unload().catch(()=>{});service.shutdown();await h.cleanup()}

console.log('Off profiles remain ready, selectable and library-loadable without a companion drafter')
