"""PubMed BERT semantic adaptation, then frozen concept-bank export (Eq. 4-5)."""
import argparse,json,hashlib,random
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModel
from pc_vlg.losses import paired_contrastive


def pooled(model,tokens):
    h=model(**tokens).last_hidden_state
    mask=tokens['attention_mask'].unsqueeze(-1)
    return (h*mask).sum(1)/mask.sum(1).clamp_min(1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--knowledge',required=True)
    p.add_argument('--pubmed-bert',required=True,help='PubMed-pretrained BERT model directory or HF identifier')
    p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--gpu',default='0')
    p.add_argument('--literal-eq5',action='store_true')
    p.add_argument('--local-only',action='store_true')
    a=p.parse_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    torch.manual_seed(1337); random.seed(1337)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw=Path(a.knowledge).read_bytes(); knowledge=json.loads(raw)
    concepts=knowledge['concepts']; triples=knowledge['triples']
    tokenizer=AutoTokenizer.from_pretrained(a.pubmed_bert,local_files_only=a.local_only)
    model=AutoModel.from_pretrained(a.pubmed_bert,local_files_only=a.local_only).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    paired={c['name']:[(c['name'],c['definition'])] for c in concepts}
    for head,rel,tail in triples:
        if head not in paired or tail not in paired: raise ValueError('Triplet contains unknown concept')
        paired[head].append((head+' '+rel,tail))
        paired[tail].append((head,rel+' '+tail))
    for step in range(a.steps):
        pairs=[random.choice(paired[c['name']]) for c in concepts]
        def encode(texts):
            tok=tokenizer(texts,padding=True,truncation=True,max_length=128,return_tensors='pt')
            return pooled(model,{k:v.to(device) for k,v in tok.items()})
        loss=paired_contrastive(encode([x[0] for x in pairs]),encode([x[1] for x in pairs]),literal=a.literal_eq5)
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite text loss')
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
        if step%50==0: print('text step {} loss {:.6f}'.format(step,float(loss)),flush=True)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    model.eval()
    with torch.no_grad(): embeddings=encode([c['name'] for c in concepts]).cpu()
    relation_types=['unspecified']+sorted({r for _,r,_ in triples})
    rid=torch.zeros(len(concepts),len(concepts),dtype=torch.long)
    names=[c['name'] for c in concepts]
    for head,rel,tail in triples: rid[names.index(head),names.index(tail)]=relation_types.index(rel)
    metadata=dict(model=a.pubmed_bert,knowledge_sha256=hashlib.sha256(raw).hexdigest(),steps=a.steps,
                  pooling='attention-mask mean',tau=0.07,literal_eq5=a.literal_eq5,concepts=names,relations=relation_types,
                  knowledge_source=knowledge.get('source','unspecified'))
    torch.save(dict(embeddings=embeddings,relation_ids=rid,metadata=metadata),out/'text_bank.pt')
    model.save_pretrained(out/'bert'); tokenizer.save_pretrained(out/'bert')
    (out/'text_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(out/'text_bank.pt')

if __name__=='__main__': main()
