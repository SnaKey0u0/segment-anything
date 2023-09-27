import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Type
import math
from segment_anything.modeling.common import MLPBlock


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class Adapter(nn.Module):
    def __init__(self, input_dim, mid_dim):
        super().__init__()
        self.model = MLP(
            input_dim=input_dim, hidden_dim=mid_dim, output_dim=input_dim, num_layers=2
        )

    def forward(self, features):
        out = features + self.model(features)
        return out


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int, #2
        embedding_dim: int, #256
        num_heads: int,#8
        mlp_dim: int,#2048
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.
        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth # 2
        self.embedding_dim = embedding_dim # 256
        self.num_heads = num_heads # 8
        self.mlp_dim = mlp_dim # 2048
        self.layers = nn.ModuleList() # append 2個

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),# 第0個=true (?)
                )
            )

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_coord,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.
        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        # torch.Size([1, 256, 32, 32, 32]), torch.Size([1, 1, 1, 30, 3])
        print("===init===")
        print("image_embedding init",image_embedding.size())
        print("point_coord init",point_coord.size())
        print("")

        # grid_sample and squeeze
        point_embedding = F.grid_sample(image_embedding, point_coord, align_corners=False)
        print("point_embedding after grid sample", point_embedding.size())

        point_pe = F.grid_sample(image_pe, point_coord, align_corners=False)
        print("point_pe after grid sample", point_pe.size())

        print('''
        之所以維度由[1,256,32,32,32]變成[1,256,1,1,30], 是因為point_coord [1,1,1,30,3]中包含了30個xyz的座標(已正規化到-1~1之間)
        定位了在image_embedding中的30個位置(維度中為32的D*H*W), 並對原始在對應image_embedding空間上的特徵進行插值(僅限這30個點)
        因此結果會是[1,256,1,1,30], 最後一個維度代表其中某一個通道在這30個點中的特徵值
        ''')

        print('''
        接下來squeeze去除1維度
        ''')

        # squeeze
        point_embedding = point_embedding.squeeze(2).squeeze(2)
        print("point_embedding after squeeze", point_embedding.size())
        point_pe= point_pe.squeeze(2).squeeze(2)
        print("point_pe after squeeze", point_pe.size())

        # permute
        print('''
        permute後, 現在我們有包含了點座標資訊的point_embedding特徵以及包含了點座標資訊的point_pe(一個固定的位置編碼矩陣)
        ''')
        point_embedding = point_embedding.permute(0, 2, 1)
        print("point_embedding after permute", point_embedding.size())
        point_pe = point_pe.permute(0, 2, 1)
        print("point_pe after permute", point_pe.size())
        original_shape = image_embedding.shape


        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        print('''
        把沒有經過給定點插植特徵的原始資料也做flatten & permute
        ''')
        print("image_embedding after flatten & permute", image_embedding.size())
        print("image_pe after flatten & permute", image_pe.size())


        print('''
        image_embedding	[1, 32768, 256]
        image_pe    [1, 32768, 256]
        point_embedding	[1, 30, 256]
        point_pe    [1, 30, 256]
        全都丟進transformer block
        ''')
        # Apply transformer blocks and final layernorm
        for i, layer in enumerate(self.layers): # 2個block
            print(f'''
                ======
                call transformer layer {i+1}
                ======
            ''')
            image_embedding, point_embedding = layer(
                image_embedding,
                point_embedding,
                image_pe,
                point_pe,
            )
        print("transformer回傳", image_embedding.size())
        return image_embedding


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int, #256
        num_heads: int, # 8
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.
        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """

        '''
        在TwoWayAttentionBlock類別中, 首先對輸入進行自我注意力操作(self_attn), 然後進行規範化(norm1)。
        接著, 將稀疏輸入(token)與密集輸入(image)進行交叉注意力操作(cross_attn_token_to_image), 並再次進行規範化(norm2)。
        然後, 在稀疏輸入上應用mlp塊(mlp), 並再次進行規範化(norm3)。最後, 將密集輸入與稀疏輸入進行交叉注意力操作(cross_attn_image_to_token), 
        並再次進行規範化(norm4)。
        整個過程中, global_query作為可訓練參數參與其中。
        '''
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        # [1,10,256]的可訓練參數(隨機產生的正態分布*0.1)
        # 可以視為一種全局上下文或者說是先驗知識, 用於引導注意力機制的焦點。
        self.global_query = nn.parameter.Parameter(data=0.1 * torch.randn(1, 10, embedding_dim))

    def forward(self, img_embed, point_embed, img_pe, point_pe) -> Tuple[Tensor, Tensor]:
        # img_pe, point_pe沒用==???
        # image_embedding	[1, 32768, 256]
        # image_pe    [1, 32768, 256] (沒用)
        # point_embedding	[1, 30, 256]
        # point_pe    [1, 30, 256] (沒用)

        print("global_query",self.global_query.size())
        q = torch.cat([self.global_query, point_embed], dim=1) # [1,10,256]+[1,30,256]=[1,40,256]
        self_out = self.self_attn(q=q, k=q, v=q) # 一開始都一樣[1,40,256], 包含了全局(自己學?)以及給定點的256個特徵，return [1, 40, 256]
        self_out = self.norm1(self_out)

        # Cross attention block, tokens attending to image embedding
        queries = q + self_out # 相加，保留原始Q的資訊，同時也加入自注意力模塊學習到的資訊。
        queries = self.norm2(queries)
        point_embed = queries[:, 10:, :] #分離global query和點提示，[1, 30, 256]
        queries = queries[:, :10, :] # [1, 10, 256]

        # MLP block
        mlp_out = self.mlp(queries) # [1, 10, 256]
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        print("cross_attn_image_to_token", img_embed.size(), queries.size())
        attn_out = self.cross_attn_image_to_token(q=img_embed, k=queries, v=queries)
        print("attn_out",attn_out.size())
        keys = img_embed + attn_out
        keys = self.norm4(keys)

        return keys, point_embed


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    '''
    Attention類別是一種注意力層, 允許在投影到query、key和value後降低嵌入的大小。
    它包含了一些屬性, 例如q_proj、k_proj、v_proj和out_proj。在前向傳播過程中, 
    首先對輸入的q、k、v進行投影, 然後將其分開成多個head, 進行注意力計算, 最後將多個head的結果重新組合並通過輸出投影。
    '''

    def __init__(
        self,
        embedding_dim: int, #256
        num_heads: int, #8
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        # 初始化
        self.embedding_dim = embedding_dim # 256
        self.internal_dim = embedding_dim // downsample_rate # 256/1
        self.num_heads = num_heads # 8 
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        # 全連接層 256 to 256
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)

        # 定義輸出層, 轉換回原本維度(但這裡都是256維度)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        # 將輸入分開成多個head, 每個head有自己的QKV
        '''
        首先, 獲取輸入張量的形狀(b, n, c), 其中b是批次大小, n是序列長度, c是特徵數量。
        然後, 將輸入張量重塑為(b, n, num_heads, c // num_heads), 這樣每個"頭"都有自己的查詢、鍵和值。
        最後, 它將第二個和第三個維度進行轉置, 得到形狀為(b, num_heads, n, c // num_heads)的輸出張量。
        '''
        b, n, c = x.shape # [1,40,256]
        '''
        在多頭注意力機制中，輸入的特徵數量會被平均分配到每個"head"中
        '''
        x = x.reshape(b, n, num_heads, c // num_heads) # [1, 40, 8, 256/8=32]
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head # [1, 8, 40, 32]

    def _recombine_heads(self, x: Tensor) -> Tensor:
        # 將多個head的結果重新組合
        b, n_heads, n_tokens, c_per_head = x.shape # [1, 8, 40, 32]
        x = x.transpose(1, 2) # [1, 40, 8, 32]
        '''
        [1, 40, 8, 32] to [1, 40, 8*32=256]
        '''
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        # QKV都是[1,40,256]
        q = self.q_proj(q) # 全連接層 256 to 256
        k = self.k_proj(k) # 全連接層 256 to 256
        v = self.v_proj(v) # 全連接層 256 to 256
        print("QKV after 全連結層256 to 256", q.size())

        # Separate into heads
        q = self._separate_heads(q, self.num_heads) # [1,40,256] to [1, 8, 40, 32]
        k = self._separate_heads(k, self.num_heads) # [1,40,256] to [1, 8, 40, 32]
        v = self._separate_heads(v, self.num_heads) # [1,40,256] to [1, 8, 40, 32]
        print("QKV after _separate_heads", q.size())

        # Attention
        _, _, _, c_per_head = q.shape # 32
        '''[1,8,40,32] @ [1,8,32,40]，最後兩個維度矩陣乘法=>[1,8,40,40]
        這個tensor的每一個元素都表示了一個q和一個k之間的相似度'''
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head) # 這種縮放操作可以防止當特徵數值很大時，點積的結果過大導致的梯度爆炸問題。
        attn = torch.softmax(attn, dim=-1) # 在最後一個維度做，這裡是40，總和會是1，對應於一個Q對所有K的相似度
        print("QK做完矩陣運算", attn.size()) # [1, 8, 40, 40]

        # Get output
        out = attn @ v # [1, 8, 40, 40] @ [1, 8, 40, 32] = [1, 8, 40, 32]
        out = self._recombine_heads(out) # [1,40,256]
        out = self.out_proj(out) # 全連接層 256 to 256
        print("out", out.size())
        print()
        return out # [1,40,256]


class MLPBlock(nn.Module):
    '''
    MLPBlock類別是一種多層感知器塊, 包含了兩個線性層和一個激活函數。
    在前向傳播過程中, 首先將輸入通過第一個線性層和激活函數, 然後再通過第二個線性層。
    '''
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class PromptEncoder(nn.Module):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        num_pos_feats: int = 128,
        mask_prompt = False
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
             torch.randn((3, num_pos_feats)),
        )
        self.mask_prompt = mask_prompt
        if mask_prompt:
            self.default_prompt = nn.parameter.Parameter(torch.randn(1, 256, 32, 32, 32))
            self.mask_encoder = nn.Sequential(
            nn.Conv3d(1, 256 // 4, kernel_size=3, stride=3),
            LayerNorm3d(256 // 4),
            nn.GELU(),
            nn.Conv3d(256 // 4, 256, kernel_size=3, padding = 1, stride=1),
            LayerNorm3d(256),
            nn.GELU(),
            nn.Conv3d(256, 256, kernel_size=1),
            )

    def forward(
        self,
        image_embeddings: torch.Tensor, # ?
        point_coord, #  [1,30,3]
        img_size = [512, 512, 32], # [128,128,128]
        feat_size = [32, 32, 32]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.
        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        image_pe = self.get_img_pe(feat_size, device=image_embeddings.device).detach()
        print("image_pe", image_pe.size())
        '''
        if self.mask_prompt:
            if masks == None:
                image_embeddings += self.default_prompt
            else:
                image_embeddings += self.mask_encoder(masks)
        '''

        # 從0~128 轉換到-1~1
        point_coord[:, :, 0] = (point_coord[:, :, 0]+0.5) * 2 / img_size[2] - 1
        point_coord[:, :, 1] = (point_coord[:, :, 1]+0.5) * 2 / img_size[1] - 1
        point_coord[:, :, 2] = (point_coord[:, :, 2]+0.5) * 2 / img_size[0] - 1
        point_coord = point_coord.reshape(1,1,1,-1,3) # 1,1,1,30,3
        print("送進transformer的三個參數image_embeddings, image_pe, point_coord", image_embeddings.size(), image_pe.size(), point_coord.size())
        features = self.transformer(image_embeddings, image_pe, point_coord)
        features = features.transpose(1,2).reshape([1, -1] + feat_size)

        return features

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        print("coords",coords.size()) # [32,32,32,3]
        coords = 2 * coords - 1 # 正規化到-1~1
        coords = coords @ self.positional_encoding_gaussian_matrix # 跟高斯矩陣[3,128]乘法，變成[32,32,32,128]
        coords = 2 * np.pi * coords * 3 / 2 # 乘一些酷東西正規化
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1) # cat完後[32,32,32,256]

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

    def get_img_pe(self, size: Tuple[int, int], device) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w, d = size
        grid = torch.ones((h, w, d), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5 # 移到點中間
        x_embed = grid.cumsum(dim=1) - 0.5
        z_embed = grid.cumsum(dim=2) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        z_embed = z_embed / d
        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1)) # stack起來是[32,32,32,3]
        print("pe", pe.size()) # [32,32,32,256]
        return pe.permute(3, 0, 1, 2).unsqueeze(0)  # C x D X H x W，return [1, 256, 32, 32, 32]