"""Create archive proposals; all model traffic and invalid proposals are logged."""
import argparse, datetime, hashlib, json, math, pathlib, re, time, urllib.error, urllib.request
from collections import Counter
from core import CFG, ROOT, apply_patch, digest_state, evaluate, load

LOG = ROOT/'logs/raw_runs.jsonl'
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def record(row):
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f: f.write(json.dumps(row, ensure_ascii=False)+'\n')
def public_target(t):
    return {'id':t['id'], 'description':t['description'], 'architecture':t['architecture'], 'base_hash':t['base_hash'],
        'weights':{k:v.tolist() for k,v in t['state'].items()},
        'support':{'x':t['support']['x'].tolist(),'y':t['support']['y'].tolist()}}
def editor_prompt(target, cards):
    context={'target':public_target(target), 'experience':cards}
    return ('You are editing a fixed 2-16-2 ReLU classifier. Predict class by argmax. '
      'Produce exactly one JSON object with keys base_hash and operations. Each operation is an object '
      'with tensor, index (zero-based list), coefficient. Allowed tensor names and shapes are in weights. '
      'Each scalar is increased by coefficient times the original tensor RMS, floor 0.01. '
      'Allowed coefficients: -2,-1,-0.5,0.5,1,2. At most 8 distinct scalar indices; empty list is valid. '
      'Use only the given support labels; do not seek other data. Repair accuracy while preserving behavior. '
      'Return only JSON; no markdown. Context:\n'+json.dumps(context,separators=(',',':')))

def parse(text):
    text=text.strip()
    if text.startswith('```'): text=re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.I)
    return json.loads(text)

def propose(target, cards, arm, repeat):
    prompt=editor_prompt(target,cards); messages=[{'role':'user','content':prompt}]
    for correction in range(2):
        text = None
        body={'model':CFG['editor']['request_model'],'messages':messages,'max_tokens':CFG['editor']['max_tokens'],'temperature':CFG['editor']['temperature']}
        request=urllib.request.Request(CFG['editor']['endpoint'],json.dumps(body).encode(),{'Content-Type':'application/json'},method='POST')
        started=now(); tick=time.monotonic(); event={'target':target['id'],'arm':arm,'repeat':repeat,'correction':correction,'started_at':started,'request_model':body['model'],'prompt':messages,'card_chars':[len(json.dumps(c)) for c in cards]}
        try:
            with urllib.request.urlopen(request,timeout=180) as response: raw=response.read().decode('utf-8')
            reply=json.loads(raw); event['raw_response']=raw; event['elapsed_s']=time.monotonic()-tick
            if reply.get('model') != CFG['editor']['response_model']: raise ValueError('response model identity mismatch')
            text=reply['choices'][0]['message']['content']; patch=parse(text); apply_patch(target['state'],patch)
            event['status']='VALID'; record(event); return patch
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            event.update(status='INVALID',error=str(exc),elapsed_s=time.monotonic()-tick); record(event)
            if text is None or correction: return None
            messages += [{'role':'assistant','content':text},{'role':'user','content':'Validator rejected your JSON: '+str(exc)+'. Return a corrected JSON object only.'}]
        except (urllib.error.URLError,TimeoutError,OSError) as exc:
            event.update(status='INFRASTRUCTURE_ERROR',error=str(exc),elapsed_s=time.monotonic()-tick)
            if isinstance(exc, urllib.error.HTTPError): event['error_body']=exc.read().decode('utf-8', errors='replace')
            record(event); return None
    return None

def selected(split):
    rows=json.loads((ROOT/'data/target_manifest.json').read_text(encoding='utf-8'))
    return [load(ROOT/'data/public'/f"{r['id']}.pt") for r in rows if r['id'].startswith(split+'-') and r['eligible']]

def archive():
    working=ROOT/'results/archive_working.json'
    if (ROOT/'results/archive_frozen.json').exists(): raise RuntimeError('archive already frozen')
    cards=json.loads(working.read_text(encoding='utf-8')) if working.exists() else []
    for i,target in enumerate(selected('archive')[:24]):
        if i < len(cards):
            if cards[i]['base_hash'] != target['base_hash']: raise RuntimeError('working archive target mismatch')
            continue
        if i>=12 and sum(c['verified_positive'] for c in cards)>=4: break
        patch=propose(target,[], 'archive',0)
        card={'source_id':target['id'],'base_hash':target['base_hash'],'architecture':target['architecture'],'description':target['description'],'family':target['family'],'weights_rms':{k:float(v.square().mean().sqrt()) for k,v in target['state'].items()},'patch_program':patch,'source_validation':None,'verified_positive':False,'failure':None}
        if patch is not None:
            scores=evaluate(target,patch); card['source_validation']=scores
            card['verified_positive']=bool(scores['accepted'] and scores['accept_edited']>scores['accept_base'] and patch['operations'])
            card['failure']=None if card['verified_positive'] else 'no positive accepted gain'
            changed=apply_patch(target['state'],patch)
            card['realized_raw_delta']={k:(changed[k]-v).tolist() for k,v in target['state'].items()}
        else: card['failure']='invalid proposal or infrastructure error'; card['realized_raw_delta']=None
        cards.append(card)
        working.write_text(json.dumps(cards,indent=2),encoding='utf-8')
        if len(cards)>=5 and all(c['failure']=='invalid proposal or infrastructure error' for c in cards[-5:]):
            print('STOP: five consecutive invalid or infrastructure failures; inspect logs before resuming')
            return
    data=json.dumps(cards,sort_keys=True,separators=(',',':')).encode()
    (ROOT/'results/archive_frozen.json').write_bytes(data)
    digest=hashlib.sha256(data).hexdigest()
    (ROOT/'results/archive_manifest.json').write_text(json.dumps({'sha256':digest,'frozen_at':now(),'count':len(cards),'positive':sum(c['verified_positive'] for c in cards)},indent=2),encoding='utf-8')
    print(json.dumps({'count':len(cards),'positive':sum(c['verified_positive'] for c in cards),'sha256':digest}))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['archive']); args=p.parse_args()
    if args.stage=='archive': archive()
