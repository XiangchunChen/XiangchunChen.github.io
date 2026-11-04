"""Fixed retrieval and proposal phase. Final-test labels are never read here."""
import hashlib, json, math, os, random, re
from collections import Counter
from pathlib import Path
from core import CFG, ROOT, evaluate
from runner import selected, propose, record, now

def words(s): return re.findall(r'[a-z0-9.\-]+',s.lower())
def cosine_tfidf(query, documents):
    vectors=[Counter(words(t)) for t in [query]+documents]; n=len(documents)
    df=Counter(token for row in vectors[1:] for token in row)
    weights={w:math.log((1+n)/(1+df[w]))+1 for row in vectors for w in row}
    def vec(c): return {w:v*weights[w] for w,v in c.items()}
    q=vec(vectors[0]); qn=math.sqrt(sum(v*v for v in q.values()))
    out=[]
    for row in vectors[1:]:
        d=vec(row); dn=math.sqrt(sum(v*v for v in d.values()))
        out.append(sum(v*d.get(w,0) for w,v in q.items())/(qn*dn) if qn and dn else 0)
    return out

def retrieve(target, cards):
    compatible=[c for c in cards if c['architecture']==target['architecture']]
    if len(compatible)<2: raise ValueError('fewer than two compatible cards')
    sims=cosine_tfidf(target['description'],[c['description'] for c in compatible])
    target_rms={k:float(v.square().mean().sqrt()) for k,v in target['state'].items()}
    scores=[]
    for c,sim in zip(compatible,sims):
        dist=sum(abs(target_rms[k]-c['weights_rms'][k])/(target_rms[k]+c['weights_rms'][k]+0.01) for k in target_rms)/len(target_rms)
        scores.append((0.8*sim+0.2*(1-dist),c['source_id'],c))
    return [x[2] for x in sorted(scores,key=lambda a:(-a[0],a[1]))[:2]]

def card_views(cards):
    """Use the same per-field budget and whitespace-pad without cutting JSON."""
    caps={'source_id':32,'description':160,'architecture':64,'patch_program':1200,'source_validation':500,'failure':160}
    rows=[]
    for card in cards:
        row={}
        for key, cap in caps.items():
            value=json.dumps(card.get(key),ensure_ascii=False,separators=(',',':'))
            if len(value)>cap: raise ValueError('experience field exceeds fixed budget: '+key)
            row[key]=value.ljust(cap)
        rows.append(row)
    return rows

def main():
    if not (ROOT/'results/screen_summary.json').exists(): raise ValueError('screen analysis must finish before confirm')
    raw=(ROOT/'results/archive_frozen.json').read_bytes(); manifest=json.loads((ROOT/'results/archive_manifest.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=manifest['sha256']: raise ValueError('archive hash mismatch')
    cards=json.loads(raw)
    for card in cards: card_views([card])
    if manifest['positive']<4: raise ValueError('insufficient verified positive archive records')
    path=ROOT/'results/confirm_proposals.jsonl'
    if path.exists() or (ROOT/'logs/confirm_journal.jsonl').exists(): raise ValueError('confirm already started; interrupted stages are preserved and cannot be silently restarted')
    prereg=json.loads((ROOT/'config/pre_confirm_commit.json').read_text(encoding='utf-8'))
    prior=json.loads((ROOT/'config/pre_screen_commit.json').read_text(encoding='utf-8'))
    common=['config/frozen_config.json','data/target_manifest.json','data/file_hashes.json','experiment/core.py','experiment/runner.py','PREREGISTRATION.md']
    if any(prereg['sha256'][name]!=prior['sha256'][name] for name in common) or prereg['archive_sha256']!=prior['archive_sha256']: raise ValueError('screen and confirm settings differ')
    required={'config/frozen_config.json','data/target_manifest.json','data/file_hashes.json','experiment/core.py','experiment/runner.py','experiment/confirm.py','experiment/open_confirm.py','PREREGISTRATION.md'}
    if set(prereg['sha256']) != required: raise ValueError('incomplete precommit file list')
    target_rows=json.loads((ROOT/'data/target_manifest.json').read_text(encoding='utf-8'))
    expected_files={str(Path('data')/part/(row['id']+'.pt')) for row in target_rows for part in ('public','sealed')}
    if set(json.loads((ROOT/'data/file_hashes.json').read_text(encoding='utf-8'))) != expected_files: raise ValueError('incomplete data hash list')
    for name, expected in prereg['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected: raise ValueError('precommitted input changed: '+name)
    for name, expected in json.loads((ROOT/'data/file_hashes.json').read_text(encoding='utf-8')).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected: raise ValueError('data changed: '+name)
    if prereg['archive_sha256'] != manifest['sha256']: raise ValueError('precommitted archive changed')
    lock={'created_at':now(),'archive_sha256':manifest['sha256'],'sha256':prereg['sha256'],'interruption_policy':'partial stage invalid for complete paired analysis; no automatic restart or additional proposal'}
    with (ROOT/'results/confirm_lock.json').open('x',encoding='utf-8') as f:
        json.dump(lock,f,indent=2); f.flush(); os.fsync(f.fileno())
    for target in selected('confirm'):
        for repeat in range(CFG['repeats']):
            compatible=[c for c in cards if c['architecture']==target['architecture']]
            seed=int(hashlib.sha256((target['id']+':'+str(repeat)).encode()).hexdigest()[:16],16)
            samples=random.Random(seed).sample(compatible,2)
            relevant=retrieve(target,cards)
            for arm,chosen in [('none',[]),('random',samples),('relevant',relevant)]:
                for name, expected in lock['sha256'].items():
                    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected: raise ValueError('frozen input changed: '+name)
                pending={'status':'REQUEST_STARTED','target':target['id'],'arm':arm,'repeat':repeat,'source_ids':[c['source_id'] for c in chosen],'at':now()}
                with (ROOT/'logs/confirm_journal.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(pending)+'\n'); f.flush(); os.fsync(f.fileno())
                patch=propose(target,card_views(chosen) if chosen else [],arm,repeat)
                row={'target':target['id'],'family':target['family'],'base_hash':target['base_hash'],'repeat':repeat,'arm':arm,
                    'source_ids':[c['source_id'] for c in chosen], 'archive_hash':manifest['sha256'], 'patch':patch,'scores':None,'at':now()}
                if patch is not None: row['scores']=evaluate(target,patch)
                with path.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
if __name__=='__main__': main()
