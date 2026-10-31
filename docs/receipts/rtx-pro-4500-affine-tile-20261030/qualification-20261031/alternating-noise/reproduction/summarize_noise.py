from pathlib import Path
from datetime import datetime,timezone
import itertools,json,math,statistics
r=Path(__file__).resolve().parent;o=r.parent.parent/'current/maintainer-noise-results';state=json.loads((o/'state.json').read_text());assert state['complete']and len(state['boots'])==18
rows=[];reference=None
for triplet in range(6):
 boots=sorted([b for b in state['boots']if b['triplet']==triplet],key=lambda b:b['name']);assert len(boots)==3 and [b['override']for b in boots]==['0','1','0'];samples=[]
 for boot in boots:
  assert boot['complete']and boot['override_startup_confirmed']and boot['token_parity'];data=json.loads((o/(boot['name']+'.json')).read_text());tokens={x['prompt']:[e['token_sha']for e in x['evidence']]for x in data}
  if reference is None:reference=tokens
  assert tokens==reference; samples.append({x['prompt']:x['decode_tps_median']for x in data})
 cells={}
 for name in samples[0]:
  before,candidate,after=[x[name]for x in samples];log_gain=math.log(candidate)-(math.log(before)+math.log(after))/2;cells[name]={'stock_before':before,'candidate':candidate,'stock_after':after,'bracket_reference':math.sqrt(before*after),'log_gain':log_gain,'gain_pct':100*math.expm1(log_gain),'baseline_drift_pct':100*(after/before-1)}
 log_gain=statistics.mean(x['log_gain']for x in cells.values());rows.append({'triplet':triplet,'cells':cells,'log_gain':log_gain,'greedy_geomean_gain_pct':100*math.expm1(log_gain),'boots':[b['name']for b in boots]})
values=[x['log_gain']for x in rows];mean=statistics.mean(values);se=statistics.stdev(values)/math.sqrt(6);critical=2.570581835636305;positive=sum(x>0 for x in values);k=max(positive,6-positive);p=min(1,2*sum(math.comb(6,j)for j in range(k,7))/64)
summary={'complete':True,'computed_utc':datetime.now(timezone.utc).isoformat(),'primary_unit':'One whole independent ABA triplet, n6; each boot2prompts x7repeats. Requests/tokens are not independent samples.','geometric_mean_gain_pct':100*math.expm1(mean),'log_t_95pct_interval_pct':[100*math.expm1(mean-critical*se),100*math.expm1(mean+critical*se)],'interval_assumptions':'Student-t interval on6 independent triplet log ratios (df5); approximate normality, no universal hardware/temperature/workload claim. Shared machine/time dependence may limit generalization.','observed_triplet_range_pct':[min(x['greedy_geomean_gain_pct']for x in rows),max(x['greedy_geomean_gain_pct']for x in rows)],'positive_triplets':positive,'two_sided_exact_sign_test_p':p,'all_tokens_match':True,'triplets':rows,'interruptions':state.get('interruptions',[]),'separate_protocol':'Classic greedy only. Community/long/feature receipts remain distinct; no pooling with old overnight brackets or reserve0 results.'};(r/'noise-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print({k:v for k,v in summary.items()if k not in ['triplets','interruptions']})
