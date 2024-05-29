import random
import torch
from torch import nn
import numpy as np

class MaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self, num_tokens, mask_ratio):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        mask = list(range(int(self.num_tokens)))
        random.shuffle(mask)
        mask_len = int(self.num_tokens * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        # self.masked_tokens = np.array(self.masked_tokens)
        # self.unmasked_tokens = np.array(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return self.unmasked_tokens, self.masked_tokens


class STMaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self, num_temporal_tokens, num_spatial_tokens, mask_ratio):
        super().__init__()
        self.num_temp_tokens = num_temporal_tokens
        self.num_spa_tokens = num_spatial_tokens
        self.num_tokens = num_temporal_tokens * num_spatial_tokens
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        mask = list(range(int(self.num_tokens)))
        random.shuffle(mask)
        mask_len = int(self.num_tokens * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        indices = np.unravel_index(self.masked_tokens,(self.num_temp_tokens, self.num_spa_tokens))
        indices = np.array(indices).T
        indices = torch.from_numpy(indices)
        return indices, self.unmasked_tokens, self.masked_tokens

    def forward(self):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        indices = np.unravel_index(self.masked_tokens,(self.num_temp_tokens, self.num_spa_tokens)).T
        return indices

class STMaskGeneratorv2(nn.Module):
    """Mask generator."""

    def __init__(self, num_temporal_tokens, num_spatial_tokens, mask_ratio, block_length):
        super().__init__()
        self.num_temp_tokens = num_temporal_tokens
        self.num_spa_tokens = num_spatial_tokens
        self.num_tokens = num_temporal_tokens * num_spatial_tokens
        self.mask_ratio = mask_ratio
        self.block_length = block_length
        self.sort = True

    def uniform_rand(self):
        mask = list(range(int(self.num_tokens)))
        random.shuffle(mask)
        mask_len = int(self.num_tokens * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        
        num_blocks = int(self.num_temp_tokens * self.mask_ratio / self.block_length)
        masked_indices = []
        indice_list =[]

        # 随机生成掩码块的起始位置
        for i in range(self.num_spa_tokens):
            block_starts = np.random.choice(self.num_temp_tokens - self.block_length + 1, num_blocks, replace=False) + i * self.num_temp_tokens
            indice_list.append(block_starts)
            for j in range(1, self.block_length):
                indice_list.append(block_starts+j)
            # print(block_starts)
        # print(indice_list)
        indices = np.concatenate(indice_list, axis = 0)
        # print(indices.shape)
        indices = np.unravel_index(indices,(self.num_temp_tokens, self.num_spa_tokens))
        indices = np.array(indices).T
        indices = torch.from_numpy(indices)
        return indices, indice_list, None
    
    def forward(self):
        indices , self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return indices
