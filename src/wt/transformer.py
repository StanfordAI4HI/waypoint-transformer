"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
from typing import Dict, List, Tuple, Type, Union

from stable_baselines3.common.torch_layers import MlpExtractor

GOAL_NET_OUT = []

import torch
import os
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np

from wt import goal_net

DEPRECATED = True

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.0
    resid_pdrop = float(os.environ.get('DROPOUT', 0.15))
    attn_pdrop = float(os.environ.get('DROPOUT', 0.15))

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_actions = config.model_actions

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        # self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # modified decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Sequential(nn.Linear(config.n_embd, config.vocab_size), nn.Tanh())

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        self.obs_emb = nn.Sequential(nn.Linear(config.feature_dim, config.n_embd), nn.Tanh())
        if self.model_actions:
            self.act_emb = nn.Sequential(nn.Linear(config.act_dim, config.n_embd), nn.Tanh())
        # self.goal_emb = nn.Sequential(nn.Linear(2, config.n_embd))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        # no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, obs):
        # obs = obs.view(-1, self.block_size, self.config.feature_dim)
        obs, actions = obs
        if os.environ.get('debug'):
            print(obs[0, -1, :2])
        B, T, D = obs.shape
        assert D == self.config.feature_dim
        if actions is not None:
            actions = actions.view(B, T, -1)
        if self.model_actions:
            token_embeddings = torch.zeros((B, 2 * T, self.config.n_embd), device = obs.device)
            token_embeddings[:, 0::2] = self.obs_emb(obs)
            token_embeddings[:, 1::2] = self.act_emb(actions) if actions is not None else 0
            assert token_embeddings.shape == (B, 2 * T, self.config.n_embd)
        else:
            token_embeddings = self.obs_emb(obs) # (batch * block_size, n_embd)
            assert token_embeddings.shape == (B, T, self.config.n_embd)

        position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if self.model_actions:
            assert logits.shape == (B, 2 * T, self.config.vocab_size)
            act = logits[:, 1::2]
            logits = logits[:, 0::2]
        else:
            act = None
            assert logits.shape == (B, T, self.config.vocab_size)

        logits = logits.view(B * T, self.config.vocab_size)

        return (logits, act) if not DEPRECATED else logits

class TransformerExtractor(MlpExtractor):
    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
        config = {},
        goal_network_ckpt = None
    ):
        super(TransformerExtractor, self).__init__(config.get('vocab_size', 512), net_arch, activation_fn, device)
        self.config = config

        # preprocessing data points w/ goal network
        self.goal_network = None
        self.cfg = None
        self.goal_network_ckpt = goal_network_ckpt
        # assert goal_network_ckpt is not None
        if goal_network_ckpt is not None:
            # TODO make this prettier
            if 'antmaze' in goal_network_ckpt:
                self.cfg = dict(obs_dim = 29, goal_dim = 2, hidden_dim = 1024, max_T = None)
            elif 'kitchen' in goal_network_ckpt:
                self.cfg = dict(obs_dim = 60, goal_dim = 60, hidden_dim = 1024, max_T = 30,)
            elif 'hopper' in goal_network_ckpt:
                self.cfg = dict(obs_dim = 11, goal_dim = 2, hidden_dim = 1024, max_T = 30, reward = True)
            elif 'cheetah' in goal_network_ckpt:
                self.cfg = dict(obs_dim = 17, goal_dim = 2, hidden_dim = 1024, max_T = 30, reward = True)
            elif 'walker' in goal_network_ckpt:
                self.cfg = dict(obs_dim = 17, goal_dim = 2, hidden_dim = 1024, max_T = 30, reward = True)
            else:
                assert 'manual' in goal_network_ckpt
                self.cfg = dict(obs_dim = 29, goal_dim = 2)
                self.goal_network = goal_net.ManualGoalNetwork(**self.cfg)

        # if config is not specified, reverts to MLP only
        if config is not None:
            model_actions = bool(os.environ.get('MODEL_ACTIONS', False))
            config = GPTConfig(vocab_size = config.get('vocab_size', 512), 
                               block_size = 20 * (1 + model_actions), 
                               n_layer = int(os.environ.get('NUM_LAYERS', 2)),
                               n_head = config.get('n_head', 16), 
                               n_embd = config.get('n_embd', 128),
                               feature_dim = feature_dim + (self.cfg['goal_dim'] if self.goal_network_ckpt else 0), 
                               model_actions = model_actions,
                               act_dim = 8)
            transformer_base = GPT(config).float()
        else:
            transformer_base = lambda x: x
        mlp_head = self.shared_net
        if DEPRECATED:
            self.shared_net = nn.Sequential(transformer_base, mlp_head)
        else:
            self.transformer_base, self.mlp_head = transformer_base, mlp_head

    def forward(self, features):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        features = self.goal_extract(features)
        if not DEPRECATED:
            base_features, self.action_features = self.transformer_base(features)
            shared_latent = self.mlp_head(base_features)
        else:
            shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def goal_extract(self, features):
        if self.goal_network is None and self.goal_network_ckpt is not None:
            self.goal_network = goal_net.KForwardGoalNetwork.load_from_checkpoint(
                self.goal_network_ckpt, **self.cfg
            ).to(features[0].device)
            self.goal_network.eval()
            for p in self.goal_network.parameters():
                p.requires_grad = False
        elif self.goal_network is None:
            return features
        obs, actions = features
        with torch.no_grad():
            new_goals = self.goal_network(obs).detach()
            # print(new_goals[0, -1], obs[0, -1, :2])
            if os.environ.get('TRACK_GOALS'):
                global GOAL_NET_OUT
                GOAL_NET_OUT.extend(new_goals[:, -1].numpy().reshape((-1, 2)).tolist())
        return torch.cat([obs, new_goals], dim = -1), actions
