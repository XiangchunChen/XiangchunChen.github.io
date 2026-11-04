"""Open screen final-test exactly once after complete proposal collection."""
import hashlib, itertools, json, math, pathlib, random, statistics, time
from core import ROOT, accuracy, apply_patch, evaluate, load, model_for
from screen import retrieve, card_views
from runner import selected, parse, editor_prompt

def quantile(values, p):
    values=sorted(values); q=(len(values)-1)*p; lo=math.floor(q); hi=math.ceil(q)
    return values[lo] + (values[hi]-values[lo])*(q-lo)
def statistics_for(diffs):
    n=len(diffs); observed=sum(diffs)/n
    samples=[sum(d*s for d,s in zip(diffs, signs))/n for signs in itertools.product((-1,1),repeat=n)]
    p=sum(x>=observed-1e-12 for x in samples)/len(samples)
    rng=random.Random(20261104)
    boot=[sum(rng.choice(diffs) for _ in range(n))/n for _ in range(20000)]
    return {'mean':observed,'p_one_sided_sign_flip':p,'ci95':[quantile(boot,.025),quantile(boot,.975)],'ci99':[quantile(boot,.005),quantile(boot,.995)],'ci97_5_simultaneous_bonferroni':[quantile(boot,.0125),quantile(boot,.9875)],'per_target':diffs}

def main():
    proposal_path=ROOT/'results/screen_proposals.jsonl'; rows=[json.loads(x) for x in proposal_path.read_text(encoding='utf-8').splitlines()]
    targets=selected('screen'); expected={(t['id'],a,r) for t in targets for a in ('none','random','relevant') for r in range(2)}
    actual=[(r['target'],r['arm'],r['repeat']) for r in rows]
    if len(rows)!=len(expected) or set(actual)!=expected or len(set(actual))!=len(actual): raise ValueError('screen proposals incomplete or duplicated')
    lock=json.loads((ROOT/'results/screen_lock.json').read_text(encoding='utf-8'))
    for name,sha in lock['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha: raise ValueError('frozen input changed: '+name)
    archive_raw=(ROOT/'results/archive_frozen.json').read_bytes()
    if hashlib.sha256(archive_raw).hexdigest()!=lock['archive_sha256']: raise ValueError('archive changed')
    datahash=json.loads((ROOT/'data/file_hashes.json').read_text(encoding='utf-8'))
    for target in targets:
        for part in ('public','sealed'):
            path=ROOT/'data'/part/(target['id']+'.pt')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=datahash[str(path.relative_to(ROOT))]: raise ValueError('target file changed')
    target_hashes={t['id']:t['base_hash'] for t in targets}
    if any(r['base_hash']!=target_hashes[r['target']] or r['archive_hash']!=lock['archive_sha256'] for r in rows): raise ValueError('proposal identity mismatch')
    target_map={t['id']:t for t in targets}
    for row in rows:
        if row['patch'] is None:
            if row['scores'] is not None: raise ValueError('invalid patch with scores')
        elif evaluate(target_map[row['target']],row['patch']) != row['scores']:
            raise ValueError('accept/guard replay mismatch')
    journal=[json.loads(x) for x in (ROOT/'logs/screen_journal.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(journal)!=len(expected) or {(r['target'],r['arm'],r['repeat']) for r in journal}!=expected: raise ValueError('journal mismatch')
    journal_by={(r['target'],r['arm'],r['repeat']):r for r in journal}
    rawlog=ROOT/'logs/raw_runs.jsonl'; calls=[json.loads(x) for x in rawlog.read_text(encoding='utf-8').splitlines()]
    archive_cards=json.loads(archive_raw)
    for row in rows:
        target=target_map[row['target']]
        compatible=[c for c in archive_cards if c['architecture']==target['architecture']]
        seed=int(hashlib.sha256((target['id']+':'+str(row['repeat'])).encode()).hexdigest()[:16],16)
        chosen=[] if row['arm']=='none' else (random.Random(seed).sample(compatible,2) if row['arm']=='random' else retrieve(target,archive_cards))
        if [c['source_id'] for c in chosen]!=row['source_ids']: raise ValueError('retrieval replay mismatch')
        key=(row['target'],row['arm'],row['repeat'])
        if journal_by[key]['source_ids']!=row['source_ids']: raise ValueError('journal source mismatch')
        events=[x for x in calls if (x.get('target'),x.get('arm'),x.get('repeat'))==key]
        if not 1<=len(events)<=2: raise ValueError('missing or excess model calls')
        if events[0]['prompt'][0]['content'] != editor_prompt(target,card_views(chosen) if chosen else []): raise ValueError('actual prompt mismatch')
        valid=[x for x in events if x['status']=='VALID']
        if row['patch'] is None:
            if valid: raise ValueError('valid response omitted')
        else:
            if len(valid)!=1 or valid[0]['request_model']!='kiro/claude-haiku-4.5': raise ValueError('model call identity mismatch')
            response=json.loads(valid[0]['raw_response'])
            if response['model']!='claude-haiku-4.5' or parse(response['choices'][0]['message']['content']) != row['patch']: raise ValueError('response patch mismatch')
    with (ROOT/'results/screen_opening.json').open('x',encoding='utf-8') as f: json.dump({'proposal_sha256':hashlib.sha256(proposal_path.read_bytes()).hexdigest(),'raw_log_sha256':hashlib.sha256(rawlog.read_bytes()).hexdigest(),'journal_sha256':hashlib.sha256((ROOT/'logs/screen_journal.jsonl').read_bytes()).hexdigest(),'rule':'single batch opening; no retry'},f)
    output=[]; mechanisms=[]
    cards=json.loads(archive_raw)
    for target in targets:
        path=ROOT/'data/sealed'/f"{target['id']}.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=datahash[str(path.relative_to(ROOT))]: raise ValueError('sealed data hash mismatch')
        sealed=load(path)
        if sealed['base_hash']!=target['base_hash']: raise ValueError('identity mismatch')
        base=model_for(target['state']); base_acc=accuracy(base,sealed['final'])
        for row in [x for x in rows if x['target']==target['id']]:
            apply=bool(row['patch'] is not None and row['scores']['accepted'])
            result=accuracy(model_for(apply_patch(target['state'],row['patch'])),sealed['final']) if apply else base_acc
            output.append({'target':target['id'],'family':target['family'],'arm':row['arm'],'repeat':row['repeat'],'valid':row['patch'] is not None,'accepted':apply,'base':base_acc,'final':result,'gain':result-base_acc})
        positives=[c for c in cards if c['verified_positive'] and c['architecture']==target['architecture']]
        if len(positives)>=2:
            source=retrieve(target,positives)[0]
            timer=time.perf_counter()
            program={'base_hash':target['base_hash'],'operations':source['patch_program']['operations']}
            program_state=apply_patch(target['state'],program)
            delta=source['realized_raw_delta']
            state={k:v+v.new_tensor(delta[k]) for k,v in target['state'].items()}
            def gate(candidate):
                model=model_for(candidate)
                ac0=accuracy(base,target['accept']); ac1=accuracy(model,target['accept'])
                gd0=accuracy(base,target['guard']); gd1=accuracy(model,target['guard'])
                accepted=ac1>=ac0 and gd1>=gd0-.10
                return {'accept_base':ac0,'accept_edited':ac1,'guard_base':gd0,'guard_edited':gd1,'accepted':accepted,
                    'final_if_applied':accuracy(model,sealed['final']),'final_after_gate':accuracy(model,sealed['final']) if accepted else base_acc}
            raw_result=gate(state); program_result=gate(program_state)
            relevant=[x['final'] for x in output if x['target']==target['id'] and x['arm']=='relevant']
            mechanisms.append({'target':target['id'],'source_id':source['source_id'],'raw_delta':raw_result,'program':program_result,
                'regenerated_final_after_gate':relevant,'source_note':'mechanism source is highest-ranked positive card; it may differ from the two cards shown to the editor, so comparisons are not strictly source-matched','base_final':base_acc,'mechanism_eval_elapsed_s':time.perf_counter()-timer})
    by={}
    for target in targets:
        by[target['id']]={a:statistics.mean(r['gain'] for r in output if r['target']==target['id'] and r['arm']==a) for a in ('none','random','relevant')}
    contrasts={name:statistics_for([d['relevant']-d[name] for d in by.values()]) for name in ('none','random')}
    arm_quality={}
    for arm in ('none','random','relevant'):
        arm_rows=[r for r in rows if r['arm']==arm]
        accepted=[r for r in arm_rows if r['scores'] and r['scores']['accepted']]
        arm_quality[arm]={'proposals':len(arm_rows),'invalid':sum(r['patch'] is None for r in arm_rows),'accepted':len(accepted),'mean_guard_change_intention_to_treat':statistics.mean((r['scores']['guard_edited']-r['scores']['guard_base']) if r in accepted else 0 for r in arm_rows)}
    summary={'target_count':len(targets),'contrasts':contrasts,'p_e5':max(v['p_one_sided_sign_flip'] for v in contrasts.values()),'target_gains':by,'wins':{a:sum(d['relevant']>d[a] for d in by.values()) for a in ('none','random')},'joint_wins':sum(d['relevant']>d['none'] and d['relevant']>d['random'] for d in by.values()),'practical_threshold_5pp_both':all(v['mean']>=0.05 for v in contrasts.values()),'arm_quality':arm_quality,'method':'exploratory screen; paired target sign-flip and percentile bootstrap'}
    for name,obj in [('screen_final.json',output),('screen_mechanisms.json',mechanisms),('screen_summary.json',summary)]:
        (ROOT/'results'/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
