"""Frozen MedSAM encoders and four aligned VAP/PC-CMG stages."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from .graph import PositionCorrelatedGraph
from .prompts import ConsistencyAwarePromptGenerator


class FrozenMedSAM(nn.Module):
    def __init__(self,checkpoint):
        super().__init__()
        if not Path(checkpoint).is_file(): raise FileNotFoundError('MedSAM checkpoint required: '+str(checkpoint))
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'networks'/'MedSAM'))
        from segment_anything.modeling import ImageEncoderViT,PromptEncoder
        # Instantiate directly: inherited build_sam.py hardcodes GPU 2.
        self.image_encoder=ImageEncoderViT(depth=12,embed_dim=768,img_size=1024,mlp_ratio=4,
            norm_layer=lambda d:nn.LayerNorm(d,eps=1e-6),num_heads=12,patch_size=16,qkv_bias=True,
            use_rel_pos=True,global_attn_indexes=[2,5,8,11],window_size=14,out_chans=256)
        self.prompt_encoder=PromptEncoder(embed_dim=256,image_embedding_size=(64,64),input_image_size=(1024,1024),mask_in_chans=16)
        state=torch.load(checkpoint,map_location='cpu')
        state=state.get('state_dict',state)
        for name,module in [('image_encoder',self.image_encoder),('prompt_encoder',self.prompt_encoder)]:
            selected={k[len(name)+1:]:v for k,v in state.items() if k.startswith(name+'.')}
            module.load_state_dict(selected,strict=True)
        self.register_buffer('pixel_mean',torch.tensor([123.675,116.28,103.53]).view(1,3,1,1))
        self.register_buffer('pixel_std',torch.tensor([58.395,57.12,57.375]).view(1,3,1,1))
        self.requires_grad_(False); self.eval()

    def train(self,mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self,image,boxes,valid):
        b,c,h,w=image.shape
        rgb=image.expand(-1,3,-1,-1) if c==1 else image
        rgb=F.interpolate(rgb.float(),size=(1024,1024),mode='bilinear',align_corners=False)
        image_features=self.image_encoder((rgb*255-self.pixel_mean)/self.pixel_std)
        scale=boxes.new_tensor([1024/w,1024/h,1024/w,1024/h])
        sparse,dense=self.prompt_encoder(points=None,boxes=(boxes*scale).flatten(0,1),masks=None)
        n=boxes.shape[1]
        sparse=sparse.view(b,n,2,256)*valid[:,:,None,None]
        # Prompt spatial PE is fused into the visual branch; dense no-mask encoding is class independent.
        dense=dense.view(b,n,256,64,64).mean(1)
        spatial=self.prompt_encoder.get_dense_pe()
        return image_features+dense+spatial,sparse.flatten(1,2)


class WindowDualAttention(nn.Module):
    def __init__(self,dim=256,heads=8,window=8):
        super().__init__()
        self.window=window
        self.local=nn.TransformerEncoderLayer(dim,heads,dim*4,dropout=0,batch_first=True,norm_first=True)
        self.cross=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.norm=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))

    def forward(self,x,memory):
        b,d,h,w=x.shape; win=self.window
        if h%win or w%win: raise ValueError('VAP sizes must be divisible by attention window')
        windows=x.permute(0,2,3,1).reshape(b,h//win,win,w//win,win,d).permute(0,1,3,2,4,5).reshape(-1,win*win,d)
        windows=self.local(windows)
        tokens=windows.reshape(b,h//win,w//win,win,win,d).permute(0,1,3,2,4,5).reshape(b,h*w,d)
        tokens=tokens+self.cross(self.norm(tokens),memory,memory,need_weights=False)[0]
        tokens=tokens+self.ffn(tokens)
        return tokens.transpose(1,2).reshape(b,d,h,w)


class VLGRNet(nn.Module):
    def __init__(self,prior,text_embeddings,relation_ids,classes,dim=256,heads=8,relations=16,sizes=(16,32,64,128)):
        super().__init__()
        if len(sizes)!=4: raise ValueError('paper requires four graph stages')
        self.prior=prior.eval(); self.prior.requires_grad_(False); self.sizes=sizes
        self.register_buffer('text_embeddings',text_embeddings.float())
        self.register_buffer('relation_ids',relation_ids.long())
        self.text_projection=nn.Linear(text_embeddings.shape[-1],dim)
        self.query_blocks=nn.ModuleList([nn.TransformerDecoderLayer(dim,heads,dim*4,dropout=0,batch_first=True,norm_first=True) for _ in range(6)])
        self.query_stops=(1,2,4,6)
        self.visual=nn.ModuleList([WindowDualAttention(dim,heads) for _ in sizes])
        self.graphs=nn.ModuleList([PositionCorrelatedGraph(dim,relations) for _ in sizes])
        self.head=nn.Conv2d(dim,classes,1)
        self.prompts=ConsistencyAwarePromptGenerator()

    def train(self,mode=True):
        super().train(mode); self.prior.eval()
        return self

    def forward(self,image,coarse,midline,labels=None,labeled=None,return_graph=False,coordinate_transform=None):
        boxes,valid=self.prompts(coarse,labels,labeled)
        encoded,prompt=self.prior(image,boxes,valid)
        # Six query blocks read a compact image memory, avoiding global pixel self-attention.
        memory=F.adaptive_avg_pool2d(encoded,(16,16)).flatten(2).transpose(1,2)
        query=self.text_projection(self.text_embeddings)[None].expand(image.shape[0],-1,-1)
        visual=encoded
        previous=0; traces=[]
        for g,(size,stop) in enumerate(zip(self.sizes,self.query_stops)):
            for j in range(previous,stop): query=self.query_blocks[j](query,memory)
            previous=stop
            visual=F.interpolate(visual.float(),size=(size,size),mode='bilinear',align_corners=False)
            visual=self.visual[g](visual,torch.cat((prompt,query),1))
            visual,trace=self.graphs[g](visual,query,self.relation_ids,midline,coordinate_transform)
            if return_graph: traces.append(trace)
        logits=F.interpolate(self.head(visual).float(),size=image.shape[-2:],mode='bilinear',align_corners=False)
        return (logits,traces) if return_graph else logits


def build_coarse(classes):
    from networks.umambabot_2d import create_umamba_bot_2d
    return create_umamba_bot_2d(1,classes,deep_supervision=False)
