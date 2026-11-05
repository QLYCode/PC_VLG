import argparse,json,os
from pathlib import Path
import torch
from pc_vlg.model import build_coarse
from pc_vlg.metrics import evaluate


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True)
    p.add_argument('--split',choices=['val','test'],default='test'); p.add_argument('--gpu',default='3'); p.add_argument('--out',required=True)
    a=p.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    state=torch.load(a.checkpoint,map_location='cpu')
    classes=state.get('classes') or state['config']['config']['classes']
    model=build_coarse(classes).cuda(); model.load_state_dict(state['coarse'],strict=True)
    rows=json.loads(Path(a.manifest).read_text())[a.split]
    result=evaluate(model,rows,classes,torch.device('cuda'))
    result.update(checkpoint=a.checkpoint,split=a.split)
    Path(a.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)

if __name__=='__main__': main()
