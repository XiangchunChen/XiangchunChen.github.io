"""CPU-only tiny target generator and restricted sparse patch evaluator."""
import argparse, hashlib, json, math, pathlib, random
import numpy as np
import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config/frozen_config.json').read_text(encoding='utf-8-sig'))
torch.set_num_threads(1)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 2)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def digest_state(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def labels(x, family, p):
    if family == 'linear': return (x[:, 0] + p * x[:, 1] > 0).long()
    if family == 'radial': return (x.square().sum(1) > p).long()
    return ((x[:, 0] > p) ^ (x[:, 1] > 0)).long()

def data(generator, count, family, p, region):
    x = torch.rand((count, 2), generator=generator) * 2 - 1
    if region == 'train':
        x[:, 0] = x[:, 0] * 0.7 + 0.3
        x[:count//4, 0] = x[:count//4, 0] * (0.3/0.7) - (0.7 + 0.3*0.3/0.7)
    elif region == 'guard': x[:, 0] = x[:, 0] * 0.3 - 0.7
    elif region in ('support', 'accept', 'final'): x[:, 0] = x[:, 0] * 0.7 + 0.3
    return {'x': x, 'y': labels(x, family, p)}

def accuracy(model, batch):
    with torch.no_grad():
        return (model(batch['x']).argmax(1) == batch['y']).float().mean().item()

def generate(split, index):
    seed = CFG['seed_offsets'][split] + index
    random.seed(seed); torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 100000)
    family = ['linear', 'radial', 'xor'][index % 3]
    p = {'linear': [-0.8, -0.4, 0.4, 0.8], 'radial': [0.3, 0.45, 0.6, 0.75], 'xor': [-0.35, -0.15, 0.15, 0.35]}[family][(index // 3 + seed // 1000) % 4]
    if split == 'confirm': p += {'linear': 0.07, 'radial': 0.04, 'xor': 0.04}[family]
    batches = {name: data(g, count, family, p, name) for name, count in CFG['samples'].items()}
    model = Net(); opt = torch.optim.SGD(model.parameters(), lr=0.12)
    # Train only the initial checkpoint, never the patched target.
    for _ in range(12):
        opt.zero_grad(); loss = nn.functional.cross_entropy(model(batches['train']['x']), batches['train']['y']); loss.backward(); opt.step()
    base = accuracy(model, batches['accept'])
    name = f'{split}-{index:02d}'
    state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    public = {'id':name, 'seed':seed, 'family':family, 'parameter':p, 'description':f'{family} two-dimensional classification parameter {p}', 'architecture':'2-16-2 ReLU float32', 'base_hash':digest_state(state), 'state':state, 'support':batches['support'], 'accept':batches['accept'], 'guard':batches['guard'], 'baseline_accept':base, 'eligible':CFG['eligibility'][0] <= base <= CFG['eligibility'][1]}
    (ROOT/'data/public').mkdir(parents=True, exist_ok=True)
    (ROOT/'data/sealed').mkdir(parents=True, exist_ok=True)
    torch.save(public, ROOT/'data/public'/f'{name}.pt')
    torch.save({'id':name, 'base_hash':public['base_hash'], 'final':batches['final']}, ROOT/'data/sealed'/f'{name}.pt')
    return {'id':name, 'family':family, 'parameter':p, 'base_hash':public['base_hash'], 'baseline_accept':base, 'eligible':public['eligible']}

def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)

def apply_patch(state, patch):
    if not isinstance(patch, dict) or set(patch) != {'base_hash','operations'}: raise ValueError('patch must contain base_hash and operations')
    if patch['base_hash'] != digest_state(state): raise ValueError('base hash mismatch')
    ops = patch['operations']
    if not isinstance(ops, list) or len(ops)>8: raise ValueError('at most eight operations')
    result = {k:v.clone() for k,v in state.items()}; seen = set()
    for op in ops:
        if not isinstance(op, dict) or set(op) != {'tensor','index','coefficient'}: raise ValueError('bad operation fields')
        key, index, coeff = op['tensor'], op['index'], op['coefficient']
        if key not in state or not isinstance(index, list) or len(index)!=state[key].ndim or any(type(i) is not int or i<0 or i>=size for i,size in zip(index,state[key].shape)): raise ValueError('tensor or index invalid')
        if isinstance(coeff,bool) or not isinstance(coeff,(float,int)) or not math.isfinite(coeff) or coeff not in CFG['patch']['coefficients']: raise ValueError('coefficient invalid')
        idx = tuple(index)
        if (key,idx) in seen: raise ValueError('duplicate index')
        seen.add((key,idx))
        rms = max(float(state[key].square().mean().sqrt()), CFG['patch']['rms_floor'])
        result[key][idx] += coeff * rms
        if not torch.isfinite(result[key][idx]): raise ValueError('nonfinite result')
    return result

def model_for(state):
    model = Net(); model.load_state_dict(state); model.eval(); return model

def evaluate(target, patch):
    base = model_for(target['state']); edited = model_for(apply_patch(target['state'],patch))
    result = {}
    for split in ('accept','guard'):
        result[split+'_base'] = accuracy(base,target[split]); result[split+'_edited'] = accuracy(edited,target[split])
    result['accepted'] = (result['accept_edited'] >= result['accept_base'] and result['guard_edited'] >= result['guard_base'] - .10)
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command', choices=['generate']); args=parser.parse_args()
    manifest=[]
    for split,n in CFG['splits'].items():
        eligible = 0
        for index in range(100):
            row = generate(split,index); manifest.append(row)
            eligible += int(row['eligible'])
            if eligible == n: break
        if eligible < n: raise RuntimeError(f'Not enough eligible {split} targets')
    (ROOT/'data/target_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({s:sum(x['eligible'] for x in manifest if x['id'].startswith(s)) for s in CFG['splits']}))
if __name__=='__main__': main()
