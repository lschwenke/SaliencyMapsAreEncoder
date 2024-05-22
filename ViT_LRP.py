""" Vision Transformer (ViT) in PyTorch
Hacked together by / Copyright 2020 Ross Wightman
Adapted From https://github.com/hila-chefer/Transformer-Explainability
"""
import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
from modules.layers_ours import *

from helpers import load_pretrained
from weight_init import trunc_normal_
from layer_helpers import to_2tuple

import torch.nn as nn


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

ACT2FN = {
    "relu": ReLU,
#    "tanh": Tanh,
    "gelu": GELU,
}

default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = SIGMOID()#GELU()
        self.fc2 = Linear(hidden_features, out_features)
        #self.act2 = SIGMOID()#GELU
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.act2(x)
        #x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, vocab_size, max_position_embeddings, dmodel, pad_token_id, type_vocab_size, layer_norm_eps, dropout):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, dmodel, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, dmodel)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, dmodel)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(dmodel, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        self.add1 = Add()
        self.add2 = Add()

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.add1([token_type_embeddings, position_embeddings])
        embeddings = self.add2([embeddings, inputs_embeds])
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def relprop(self, cam, **kwargs):
        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.LayerNorm.relprop(cam, **kwargs)

        # [inputs_embeds, position_embeddings, token_type_embeddings]
        (cam) = self.add2.relprop(cam, **kwargs)

        return cam

class AttentionEncoder(nn.Module):
    def __init__(self, num_hidden_layers, inputsize, dmodel, dff, num_heads, activision="relu", attn_drop=0., hiddenDrop = 0.):
        super().__init__()
        self.layer = nn.ModuleList([AttentionLayer(inputsize, dmodel, dff, num_heads, activision=activision, attn_drop=attn_drop, hiddenDrop = hiddenDrop) if i == 0 else AttentionLayer(dmodel, dmodel, dff, num_heads, activision=activision, attn_drop=attn_drop, hiddenDrop = hiddenDrop) for i in range(num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            gradient_checkpointing = False
            if gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions=output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states, output_attentions=output_attentions
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, output_attentions=output_attentions
                )
            
            if output_attentions:
                hidden_states = layer_outputs[0]
                all_attentions = all_attentions + (layer_outputs[1],)
            else:
                hidden_states = layer_outputs

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return [hidden_states, all_hidden_states, all_attentions]
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    def relprop(self, cam, **kwargs):
        # assuming output_hidden_states is False
        for layer_module in reversed(self.layer):
            cam = layer_module.relprop(cam, **kwargs)
        return cam


class AttentionLayer(nn.Module):
    def __init__(self, inputSize, dmodel, dff, num_heads, activision="relu", attn_drop=0., hiddenDrop = 0.):
        super().__init__()
        #self.attention = Attention(dmodel, num_heads=num_heads, attn_drop=attn_drop) 
        self.attention = AttentionIn(inputSize, dmodel, num_heads=num_heads, attn_drop=attn_drop)
        self.intermediate = AttentionIntermediate(dmodel, dff, activision) # dense into act -> add bei out'?????? (wäre das nicht)
        self.output = AttentionOutput(dmodel, dff, hiddenDrop) # dense, act, dropout, norm, add
        self.clone = Clone()
        self.clone2 = Clone()

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
    ):
        ai1, ai2 = self.clone(hidden_states, 2)

        self_attention_outputs = self.attention(
            ai1, ai2, output_attentions=output_attentions)#,
        #    attention_mask,
        #    head_mask,
        #    output_attentions=output_attentions,
        #)
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        ao1, ao2 = self.clone2(attention_output, 2)
        #intermediate_output = self.intermediate(ao1)
        #layer_output = self.output(intermediate_output, ao2)
        layer_output = self.output(ao1, ao2)

        if output_attentions:
            outputs = (layer_output,) + outputs
            return outputs
        else:
            return layer_output

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.output.relprop(cam, **kwargs)
        # print("output", cam1.sum(), cam2.sum(), cam1.sum() + cam2.sum())
        #cam1 = self.intermediate.relprop(cam1, **kwargs)
        # print("intermediate", cam1.sum())
        cam = self.clone2.relprop((cam1, cam2), **kwargs)
        # print("clone", cam.sum())
        (cam1, cam2) = self.attention.relprop(cam, **kwargs)
        cam = self.clone.relprop((cam1, cam2), **kwargs)

        # print("attention", cam.sum())
        return cam

class AttentionIn(nn.Module):
    def __init__(self, inputLenght, dmodel, num_heads=8, attn_drop=0., layer_norm_eps=1e-6):
        super().__init__()
        #if dmodel % num_heads != 0:
        #    raise ValueError(
        #        "The hidden size (%d) is not a multiple of the number of attention "
        #        "heads (%d)" % (dmodel, num_heads)
        #    )

        self.num_attention_heads = num_heads
        #besser?
        #self.attention_head_size = dmodel #int(dmodel / num_heads)
        self.attention_head_size = int(dmodel / num_heads)
        self.all_head_size = num_heads * self.attention_head_size

        #besser???
        #self.query = Linear(inputLenght, num_heads * dmodel)
        #self.key = Linear(inputLenght, num_heads * dmodel)
        #self.value = Linear(inputLenght, num_heads * dmodel)
        #self.proj = Linear(num_heads * dmodel, dmodel)
        self.query = Linear(inputLenght, dmodel)
        self.key = Linear(inputLenght, dmodel)
        self.value = Linear(inputLenght, dmodel)
        self.proj = Linear(dmodel, dmodel)

        self.mha = nn.MultiheadAttention(inputLenght, num_heads, batch_first=True)

        self.dmodel = dmodel

        self.dropout = Dropout(attn_drop)

        self.matmul1 = MatMul()
        self.matmul2 = MatMul()
        self.softmax = Softmax(dim=-1)
        self.add = Add()
        self.add2 = Add()
        self.LayerNorm = LayerNorm(dmodel, eps=layer_norm_eps)
        self.mul = Mul()
        self.head_mask = None
        self.attention_mask = None
        self.clone = Clone()

        self.attn_cam = None
        self.attn = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        #new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.dmodel)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_relprop(self, x):
        return x.permute(0, 2, 1, 3).flatten(2)

    def forward(
            self,
            hidden_states,
            ai2,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        h1, h2, h3 = self.clone(hidden_states, 3)
        #attn_output, attention_probs = self.mha(h1, h2, h3, need_weights=True)
        #"""
        self.head_mask = head_mask
        self.attention_mask = attention_mask

        #h1, h2, h3 = self.clone(hidden_states, 3)
        #print(self.query)
        mixed_query_layer = self.query(h1)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(h2)
            mixed_value_layer = self.value(h3)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        #query_layer = query_layer / math.sqrt(self.attention_head_size) #TODO i fixed this and at another position???
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = self.matmul1([query_layer, key_layer.transpose(-1, -2)])
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = self.add([attention_scores, attention_mask])

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        self.save_attn(attention_probs)
        if attention_probs.requires_grad:
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = self.matmul2([attention_probs, value_layer])

        #wud? wie sieht concatinate normalerweise aus? + wo linear layer?
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = self.proj(context_layer) #TODO into relprop 
        
        add = self.add2([context_layer, ai2])
        #"""
        #add = self.add2([attn_output, ai2])
        context_layer = self.LayerNorm(add)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def relprop(self, cam, **kwargs):
        cam = self.LayerNorm.relprop(cam, **kwargs)
        (cam1, camI) = self.add2.relprop(cam, **kwargs)
        cam1 = self.proj.relprop(cam1, **kwargs)
        

        # Assume output_attentions == False
        cam = self.transpose_for_scores(cam1)

        # [attention_probs, value_layer]
        (cam1, cam2) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam2 /= 2
        if self.head_mask is not None:
            # [attention_probs, head_mask]
            (cam1, _)= self.mul.relprop(cam1, **kwargs)


        self.save_attn_cam(cam1)

        cam1 = self.dropout.relprop(cam1, **kwargs)

        cam1 = self.softmax.relprop(cam1, **kwargs)

        #TODO ggf cam1 (attention) als einzelne matricen betrachten?
        if self.attention_mask is not None:
            # [attention_scores, attention_mask]
            (cam1, _) = self.add.relprop(cam1, **kwargs)

        # [query_layer, key_layer.transpose(-1, -2)]
        (cam1_1, cam1_2) = self.matmul1.relprop(cam1, **kwargs)
        cam1_1 /= 2
        cam1_2 /= 2

        # query
        cam1_1 = self.transpose_for_scores_relprop(cam1_1)
        cam1_1 = self.query.relprop(cam1_1, **kwargs)

        # key
        cam1_2 = self.transpose_for_scores_relprop(cam1_2.transpose(-1, -2))
        cam1_2 = self.key.relprop(cam1_2, **kwargs)

        # value
        cam2 = self.transpose_for_scores_relprop(cam2)
        cam2 = self.value.relprop(cam2, **kwargs)

        cam = self.clone.relprop((cam1_1, cam1_2, cam2), **kwargs)

        return cam, camI

class AttentionOutput(nn.Module):
    def __init__(self, dmodel, dff, hidden_dropout_prob, layer_norm_eps=1e-6): 
        super().__init__()
        self.dense = Linear(dmodel, dff)
        self.acti = ReLU()#GELU()
        self.dense2 = Linear(dff, dmodel)
        self.LayerNorm = LayerNorm(dmodel, eps=layer_norm_eps)
        self.dropout = Dropout(hidden_dropout_prob)
        self.add = Add()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.acti(hidden_states) #???
        hidden_states = self.dense2(hidden_states) #???
        hidden_states = self.dropout(hidden_states)
        add = self.add([hidden_states, input_tensor])
        hidden_states = self.LayerNorm(add)
        return hidden_states

    def relprop(self, cam, **kwargs):
        # print("in", cam.sum())
        cam = self.LayerNorm.relprop(cam, **kwargs)
        #print(cam.sum())
        # [hidden_states, input_tensor]
        (cam1, cam2)= self.add.relprop(cam, **kwargs)

        
        # print("add", cam1.sum(), cam2.sum(), cam1.sum() + cam2.sum())
        cam1 = self.dropout.relprop(cam1, **kwargs)
        #print(cam1.sum())
        cam1 = self.dense2.relprop(cam1, **kwargs)
        cam1 = self.dense.relprop(cam1, **kwargs)
        # print("dense", cam1.sum())

        # print("out", cam1.sum() + cam2.sum(), cam1.sum(), cam2.sum())
        return (cam1, cam2)

class LRPTransformer(nn.Module):
    #TODO ggf activision nach bzw in attentionOut einbauen?
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, num_classes=6, embed_dim=4, depth=2,
                 num_heads=12, dff=6., drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        #self.patch_embed = PatchEmbed(
        #        img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #num_patches = self.patch_embed.num_patches

        #self.pos_embed = nn.Parameter(torch.zeros(1Mlp, num_patches + 1, embed_dim))
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        #self.blocks = nn.ModuleList([
        #    Block(
        #        dim=embed_dim, num_heads=num_heads, dff=dff, qkv_bias=qkv_bias,
        #        drop=drop_rate, attn_drop=attn_drop_rate)
        #    for i in range(depth)])

        self.norm = LayerNorm(embed_dim)
        if mlp_head:
            # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
            #self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
            self.head = Mlp(embed_dim, int(dff), num_classes)
        else:
            # with a single Linear layer as head, the param count within rounding of paper
            self.head = Linear(embed_dim, num_classes)

        # FIXME not quite sure what the proper weight init is supposed to be,
        # normal / trunc normal w/ std == .02 similar to other Bert like transformers
        #TODO Remove or not?
        #trunc_normal_(self.pos_embed, std=.02)  # embeddings same as weights?
        trunc_normal_(self.cls_token, std=.02)
        #self.apply(self._init_weights)
        self.init_weights()

        self.pool = IndexSelect()
        self.add = Add()

        self.inp_grad = None

    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.05)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        B = x.shape[0]
        #x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #cls_tokens = self.cls_token.expand(B, -1) maybe?
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        #if self.train and x.requires_grad:
        x.register_hook(self.save_inp_grad)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x

    def relprop(self, cam=None,method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam
        
        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam
            
        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.add = Add()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = self.add([x, self.pe[:, : x.size(1)]])
        #x = x + self.pe[:, : x.size(1)]
        return x #self.dropout(x)

    def relprop(self, cam, **kwargs):
        (cam, _) = self.add.relprop(cam, **kwargs)
        return cam

class TSModel(nn.Module):
    def __init__(self, num_hidden_layers, inDim, dmodel, dfff, num_heads, num_classes, dropout, att_dropout, vocab_size = -1, max_position_embeddings= 5000, pad_token_id=-1, type_vocab_size=-1, layer_norm_eps=1e-6, doEmbedding = False, embeddingDim = 1, maskValue = -2, doClsTocken = False): #TODO config fixen!!!!
        super().__init__()

        
        
        self.doEmbedding = doEmbedding
        if self.doEmbedding:
            self.embeddings = BertEmbeddings(vocab_size, max_position_embeddings, embeddingDim, pad_token_id, type_vocab_size, layer_norm_eps, dropout) #yes no? Optional?
            inputsize = embeddingDim
        else:
            inputsize = 1 #embeddingDim?
            #self.embeddings = nn.Embedding(max_position_embeddings, inputsize)
            self.pos_encoder = PositionalEncoding(d_model=dmodel, dropout=0., max_len=5000) #(dmodel, 0)
            self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        self.maskValue= maskValue
        self.dmodel = dmodel
        dff = int(dfff * dmodel)
        self.dff = dff
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        #self.encoder = TransformerEncoder(d_model=dmodel, d_ff=dff, n_heads=num_heads, n_layers=num_hidden_layers, dropout=dropout) 
        self.encoder = AttentionEncoder(num_hidden_layers, dmodel, dmodel, dff, num_heads, activision="relu", attn_drop=att_dropout, hiddenDrop = dropout)
        #encoder_layer = nn.TransformerEncoderLayer(d_model=dmodel, nhead=num_heads, dim_feedforward=dff, dropout=0.1, batch_first =True) #
        #self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)

        #TODO made for 4d?
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dmodel))
        #self.pos_embed = nn.Parameter(torch.zeros(1, 1, inputsize))
        self.flatten = torch.nn.Flatten()
        #self.flatten = torch.flatten() #TODO eigenes flatten mit relprop definieren!

        self.norm = LayerNorm(dmodel)
        self.mlp_head = False
        if self.mlp_head:
            # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
            #self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
            self.head = Mlp((inDim+1) * dmodel, int(dff), num_classes)
            #self.head = Mlp(dmodel, int(dff), num_classes)
        else:
            # with a single Linear layer as head, the param count within rounding of paper
            # 60 because input lenght is 60!!! Fix with variable?
            self.doClsTocken = doClsTocken
            if self.doClsTocken:
                self.head = Linear((inDim+1) * dmodel, num_classes)
            else:
                #self.head = Linear((30) * dmodel, num_classes)
                self.head = Linear((inDim) * dmodel, num_classes)
            #self.head = Linear(dmodel, num_classes)
            self.acth = SIGMOID()

        #self.pool = IndexSelect()

        self.add = Add()
        self.add1 = Add()

        #TODO Remove or not?
        #trunc_normal_(self.pos_embed, std=.02)  # embeddings same as weights?
        trunc_normal_(self.cls_token, std=.02)

        self.inp_grad = None

    def save_inp_grad(self,grad):
            self.inp_grad = grad

    def get_inp_grad(self):
            return self.inp_grad

        #self.apply(self._init_weights)
        #self.init_weights()
       

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value



    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    """def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.05)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)"""

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            singleOutput = True,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            if len(input_shape) == 2:
                input_ids = input_ids.unsqueeze(2)
                input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        #if attention_mask is None:
        #    attention_mask = torch.ones(input_shape, device=device)
        #if token_type_ids is None:
        #    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        #extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        #if self.config.is_decoder and encoder_hidden_states is not None:
        #    encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        #    encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        #    if encoder_attention_mask is None:
        #        encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        #    encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        #else:
        #    encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        #head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        
        #x = self.patch_embed(x)

        #if self.maskValue is not None:
        #    input_ids[input_ids==self.maskValue] += float('-inf')

        if self.doEmbedding:
            embedding_output = self.embeddings(
                input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
            )
        else:
            #bert pos encoding?
            #position_ids = self.position_ids[:, :input_ids.size()[1]]
            #print(position_ids.shape)

            #position_embeddings = self.embeddings(position_ids)
            embedding_output = self.pos_encoder(input_ids)
            
            #print(position_embeddings.shape)
            #embedding_output = self.add1([input_ids, position_ids])
            #print(embedding_output.shape)

            #embedding_output = self.add([input_ids, self.pos_embed.double()])
            #embedding_output = input_ids

        #encoder_batch_size, encoder_sequence_length, _ = embedding_output.size()
        #encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        ##encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

        if(self.doClsTocken):
            B = embedding_output.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            
            #cls_tokens = self.cls_token.expand(B, -1) maybe?

            embedding_output = torch.cat((cls_tokens, embedding_output), dim=1)

        if self.train and embedding_output.requires_grad:
            embedding_output.register_hook(self.save_inp_grad)

        encoder_outputs = self.encoder(
            embedding_output, output_attentions=output_attentions) #, None,
            #ttention_mask=None,
            #head_mask=head_mask,
            #encoder_hidden_states=encoder_hidden_states,
            #encoder_attention_mask=None,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            #return_dict=False,
        #)
        #sequence_output = encoder_outputs 
        sequence_output = encoder_outputs[0]
        if(output_attentions):
            attentionVs = encoder_outputs[2]
        #pooled_output = self.pooler(sequence_output)
        
        x = sequence_output
        #x = self.norm(sequence_output)
        #x = torch.flatten(x, start_dim=1)
        x = self.flatten(x)
        #x = x.squeeze(1)
        x = self.head(x)
        if(not self.mlp_head):
            x = self.acth(x)
        #TODO test both + fix relpro for flatten?
        #x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        #x = x.squeeze(1)
        #x = self.head(x)

        if singleOutput:
            if(self.num_classes == 1):
                return x.squeeze()#[0]
            else:
                return x
        if output_attentions:
            return x, attentionVs
        if not return_dict:
            return (x, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=x,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        cam = self.head.relprop(cam, **kwargs)
        #cam = cam.unsqueeze(1) #TODO unflatten???
        cam = cam.reshape((cam.shape[0], -1, self.dmodel))
        #cam = self.pool.relprop(cam, **kwargs)
        #cam = self.norm.relprop(cam, **kwargs)
        #cam = self.pooler.relprop(cam, **kwargs)
        # print("111111111111",cam.sum())
        cam = self.encoder.relprop(cam, **kwargs)
        # print("222222222222222", cam.sum())
        # print("conservation: ", cam.sum())

        #method = 'full'
        if method == "full":
            #if self.doClsTocken:
            #    cam = cam[:, 0, 1:]
            #else:
            #    cam = cam#[:, 0, 1:]
            #print(cam.shape)
            if self.doClsTocken:
                cam = cam[:, 0:1, :]
            else:
                cam = cam
            #print(cam.shape)
            cam = self.pos_encoder.relprop(cam, **kwargs)
            #print(cam.shape)
            #cam = cam[:, 1:]
            #cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            #cam = cam.sum(dim=1)


            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.encoder.layer:
                attn_heads = blk.attention.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            if self.doClsTocken:
                cam = cam[:, 0, 1:]
            else:
                cam = cam#[:, 0, 1:]
            return cam
        
        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.encoder.layer:
                grad = blk.attention.get_attn_gradients()
                cam = blk.attention.get_attn_cam()
                #cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
                #grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=1)
                #cams.append(cam.unsqueeze(0))
                cams.append(cam)
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            if self.doClsTocken:
                cam = rollout[:, 0, 1:]
            else:
                cam = rollout#[:, 0, 1:]
            return cam
            
        elif method == "last_layer":
            cam = self.encoder.layer[-1].attention.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.encoder.layer[-1].attention.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.encoder.layer[-1].attention.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.encoder.layer[1].attention.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.encoder.layer[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam



        return cam

class AttentionIntermediate(nn.Module):
    def __init__(self, dmodel, dff, activision):
        super().__init__()
        self.dense = Linear(dmodel, dff)
        if isinstance(activision, str):
            self.intermediate_act_fn = ACT2FN[activision]()
        else:
            self.intermediate_act_fn = activision

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    def relprop(self, cam, **kwargs):
        cam = self.intermediate_act_fn.relprop(cam, **kwargs)  # FIXME only ReLU
        #print(cam.sum())
        cam = self.dense.relprop(cam, **kwargs)
        #print(cam.sum())
        return cam

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict