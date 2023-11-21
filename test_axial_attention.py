import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import einsum
import numpy as np

def print_peak_mem(prefix=None):
    memory_stats = torch.cuda.memory_stats(device=0)
    if 'requested_bytes.all.peak' in memory_stats:
        peak_mem = memory_stats['requested_bytes.all.peak']
        current_mem = memory_stats['requested_bytes.all.current']
        print(f'[{prefix}]Memory stats, peak:{peak_mem / (1<<20)}, current:{current_mem / (1<<20)}')
    torch.cuda.reset_peak_memory_stats()

def export_model(model, onnx_name, inputs, input_names=['pair','bias'], output_names=['pair']):
    torch.onnx.export(model, inputs, onnx_name, verbose=True)

    import onnx

    onnx_model = onnx.load(onnx_name)
    onnx.checker.check_model(onnx_model)
    # Optionally, print a human readable representation of the graph
    # print(onnx.helper.printable_graph(onnx_model.graph))
    # Load the ONNX model
    model = onnx.load(onnx_name)
    # Apply shape inference on the model
    inferred_model = onnx.shape_inference.infer_shapes(model)
    # Check the inferred model
    onnx.checker.check_model(inferred_model)
    # Save model with shapes
    onnx.save(inferred_model, onnx_name)

def init_lecun_normal(module, scale=1.0):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape, scale=1.0):
        stddev = np.sqrt(scale/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    module.weight = torch.nn.Parameter( (sample_truncated_normal(module.weight.shape)) )
    return module

class BiasedAxialAttention(nn.Module):
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True):
        super(BiasedAxialAttention, self).__init__()
        #
        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_bias = nn.LayerNorm(d_bias)

        print(f'BiasedAxialAttention:(d_pair:{d_pair}, d_bias:{d_bias}, n_head:{n_head}, d_hidden:{d_hidden}, p_drop:{p_drop}, is_row:{is_row})')

        self.to_q = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False) 
        self.to_g = nn.Linear(d_pair, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_pair)
        
        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        self.dim_out = d_pair

        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair, bias, sym=None):
        O, L = pair.shape[:2] # after subunit mask is applied
        print(f'BiasedAxialAttention.forward:pair:{pair.shape}:{pair.dtype}, bias:{bias.shape}:{bias.dtype}, sym:{sym}')

        if O==1 or sym is None or sym.shape[0]!=O: # asymm mode
            # pair: (B, L, L, d_pair)
            if self.is_row:
                pair = pair.permute(0,2,1,3)
                bias = bias.permute(0,2,1,3)

            pair = self.norm_pair(pair)
            bias = self.norm_bias(bias)

            query = self.to_q(pair).reshape(O, L, L, self.h, self.dim)
            key = self.to_k(pair).reshape(O, L, L, self.h, self.dim)
            value = self.to_v(pair).reshape(O, L, L, self.h, self.dim)
            bias = self.to_b(bias) # (B, L, L, h)
            gate = torch.sigmoid(self.to_g(pair)) # (B, L, L, h*dim) 
        
            # import pdb; pdb.set_trace()

            query = query * self.scaling
            key = key / L # normalize for tied attention
            attn = einsum('bnihk,bnjhk->bijh', query, key) # tied attention
            attn = attn + bias
            attn = F.softmax(attn, dim=-2) # (B, L, L, h)
        
            out = einsum('bijh,bnjhd->bnihd', attn, value).reshape(O, L, L, -1)
            out = gate * out
        
            out = self.to_out(out)
            if self.is_row:
                out = out.permute(0,2,1,3)

        else:

            # symmetric version
            if self.is_row:
                pair = pair[sym[0,:]].permute(0,2,1,3)
                bias = bias[sym[0,:]].permute(0,2,1,3)

            pair = self.norm_pair(pair)
            bias = self.norm_bias(bias)

            query = self.to_q(pair).reshape(O, L, L, self.h, self.dim)
            key = self.to_k(pair).reshape(O, L, L, self.h, self.dim)
            value = self.to_v(pair).reshape(O, L, L, self.h, self.dim)
            bias = self.to_b(bias) # (B, L, L, h)
            gate = torch.sigmoid(self.to_g(pair)) # (B, L, L, h*dim) 

            query = query * self.scaling
            key = key / (O*L) # normalize for tied attention

            attn=torch.zeros((O,L,L,self.h), device=pair.device)
            for i in range(O):
                attn[i] = torch.einsum('bnihk,bnjhk->ijh', query[sym[:,i]], key[sym[:,0]]) # tied attention
            #attn = einsum('bnihk,bnjhk->bijh', query, key) # tied attention
        
            attn = attn + bias # apply bias

            # softmax over dims 0 & 2
            attn = F.softmax(
                attn.transpose(1,2).reshape(O*L,L,self.h), dim=0
            ).reshape(O,L,L,self.h).transpose(1,2)

            out=torch.zeros((O,L,L,self.h,self.dim), device=pair.device)
            for i in range(O):
                out[i] = torch.einsum('bijh,bnjhd->nihd', attn[sym[:,i]], value) # tied attention
            #out = einsum('bijh,bnjhd->bnihd', attn, value).reshape(O, L, L, -1)

            out = gate * out.reshape(O,L,L,-1)
            out = self.to_out(out)

            if self.is_row:
                out = out[sym[0,:]].permute(0,2,1,3)

        return out
    
class Model(nn.Module):
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop):
        super(Model, self).__init__()
        self.row_attn = BiasedAxialAttention(d_pair, d_bias, n_head, d_hidden, p_drop, is_row=True).to(device='cuda').to(dtype=torch.float16)
        self.col_attn = BiasedAxialAttention(d_pair, d_bias, n_head, d_hidden, p_drop, is_row=False).to(device='cuda').to(dtype=torch.float16)

    def forward(self, pair, bias):
        pair = pair + self.row_attn(pair, bias)
        # pair = pair + self.col_attn(pair, bias)
        return pair

if __name__ == '__main__':
    
    d_pair=128
    d_bias=128
    n_head=4
    d_hidden=32
    p_drop=0.0
    
    
    print_peak_mem('initial')
    model = Model(d_pair, d_bias, n_head, d_hidden, p_drop)
    model.eval()

    L = 2000
    pair = torch.rand([1, L, L, n_head*d_hidden], dtype=torch.float16, device='cuda')
    bias = torch.rand([1, L, L, n_head*d_hidden], dtype=torch.float16, device='cuda')
    sym = None

    # import pdb; pdb.set_trace()
    is_export_model = True
    if is_export_model == True:
        onnx_name="axial_attention.onnx"
        export_model(model, onnx_name, (pair, bias))

    print_peak_mem('start')
    with torch.cuda.amp.autocast(True):
        y = model(pair, bias)
    print_peak_mem('end')

    
    pass