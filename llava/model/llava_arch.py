"""
Hawkeye 多模态大模型核心网络架构定义文件 (llava_arch.py)
====================================================================================================
论文出处: 
  Hawkeye: Discovering and Grounding Implicit Anomalous Sentiment in Recon-videos 
  via Scene-enhanced Video Large Language Model (ACM MM 2024)

本文件核心功能:
  1. 图结构场景建模 (Section 3.1):
     - 动作敏感图 ASG (Section 3.1.1): pose_feat 骨骼关键点线性投影网络 (85维 -> 4096维)
     - 物体关系敏感图 ORG (Section 3.1.2): GTNLayer & GTN 基于 PyG 的 MaskGTN 场景图卷积网络
  2. 平衡异构混合专家网络 B-H MoE (Section 3.2):
     - MOE: 包含 2 个 TransformerBlock 投影专家 (PE) 与 MLP 软门控路由器 R(h)
  3. 多模态元模型基类 (Section 3.0 & Backbone):
     - LlavaMetaModel: 管理视觉塔、姿态塔、场景塔、MoE网络及其后处理投影器
     - LlavaMetaForCausalLM: 负责在 prepare_inputs_labels_for_multimodal 中将全局视频 Token
       与 MoE 场景细粒度 Token 拼接，替换文本提示词中的占位符，送入大语言模型 (Vicuna-7B) 进行推理
====================================================================================================
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
# 引入 PyTorch Geometric (PyG) 核心图神经网络库，用于实现 Section 3.1.2 中的 MaskGTN
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree

# 多模态编码器与投影器构建工厂函数
from .multimodal_encoder.builder import build_image_tower, build_video_tower
from .multimodal_projector.builder import build_vision_projector
from dataclasses import dataclass

# 支持 FairScale 模型并行技术 (可选分布式训练优化)
try:
    import fairscale.nn.model_parallel.initialize as fs_init
    from fairscale.nn.model_parallel.layers import (
        ParallelEmbedding,
        RowParallelLinear,
        ColumnParallelLinear,
    )
except ImportError:
    pass

from typing import Optional, Tuple
import torch.nn.functional as F

# FlashAttention 高性能注意力计算，若环境未编译 flash_attn 则优雅回退到 PyTorch 原生 scaled_dot_product_attention
try:
    from flash_attn import flash_attn_func
except ImportError:
    def flash_attn_func(q, k, v, dropout_p=0.0, causal=False):
        """FlashAttention 回退实现: 利用 PyTorch 2.0+ 原生高效注意力算子"""
        # q, k, v 维度: [bsz, seqlen, n_heads, head_dim] -> 转置为 [bsz, n_heads, seqlen, head_dim]
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
        return out.transpose(1, 2)

import copy

# 导入关键标记常量:
# IGNORE_INDEX: 交叉熵计算中忽略的标签 ID (-100)，用于遮蔽无需计算损失的视觉 Token
# X_TOKEN_INDEX: 多模态占位符在词表中的 Token ID (例如 <video>、<image>)
from llava.constants import IGNORE_INDEX, X_TOKEN_INDEX, DEFAULT_X_PATCH_TOKEN, DEFAULT_X_START_TOKEN, \
    DEFAULT_X_END_TOKEN


# =========================================================================================================
# Hawkeye 论文方法映射：第 3.1 节 图结构场景建模模块 (Graph-structured Scene Modeling Module)
# ---------------------------------------------------------------------------------------------------------
# CLASSES: RelTR (Relation Transformer) 预训练所采用的目标实体词典 (共 151 个物体类别，来源于 Visual Genome)
# REL_CLASSES: 场景中主体与客体之间的交互谓词/动作关系 (共 51 个谓词类别，对应有向边的边属性 edge_attr)
# =========================================================================================================
CLASSES = ['N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
           'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
           'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
           'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
           'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
           'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
           'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
           'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
           'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
           'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
           'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
           'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
           'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
           'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
               'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
               'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
               'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
               'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
               'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']



# =========================================================================================================
# Hawkeye 论文方法映射：第 3.1.1 节 动作敏感图 (Action-Sensitive Graph, ASG)
# ---------------------------------------------------------------------------------------------------------
# 原理：
# 1. 采用人体姿态估计网络 HigherHRNet 抽取视频中最多 5 个人物的 17 个关键骨骼点坐标 (X_a = {x_a^1, ..., x_a^n})。
# 2. 特征维度：5 人 × 17 关节点 = 85 维。
# 3. 通过线性投影层 self.pose_projector 将 85 维的人体骨骼姿态坐标投影对齐至大模型隐藏空间 (4096 维)。
# =========================================================================================================
class pose_feat(nn.Module):
    """
    动作敏感图 (ASG) 姿态特征编码器：
    输入: [Batch, 85] (5个人 × 17个关节坐标)
    输出: [Batch, 4096] 对齐至与 Vicuna-7B 语言空间同维度的姿态表征向量
    """
    def __init__(self):
        super(pose_feat, self).__init__()
        # 85 维姿态坐标映射到 LLM 隐藏层维度 4096
        self.pose_projector = nn.Linear(85, 4096)
        self.pose_projector.requires_grad_(True)

    def forward(self, pose_feat):
        # 展平输入: [Batch, 5, 17] -> [Batch, 85]
        pose_feat = pose_feat.view(pose_feat.size(0), -1)
        # 线性投射: [Batch, 85] -> [Batch, 4096]
        pose_feat = self.pose_projector(pose_feat)
        return pose_feat


def build_pose_tower():
    """构建姿态塔 (Pose Tower)"""
    return pose_feat()


def build_pose_projector():
    """构建姿态特征后处理投影层 (4096 -> 4096)"""
    return nn.Linear(4096, 4096)



# =========================================================================================================
# Hawkeye 论文方法映射：第 3.1.2 节 物体关系敏感图 (Object-Relation Sensitive Graph, ORG)
# ---------------------------------------------------------------------------------------------------------
# 原理与对应公式：
# 1. 场景关系提取：使用预训练 RelTR 提取场景中的主体、客体及相互作用谓词三元组 (subject, predicate, object)。
# 2. 图神经网络 MaskGTN：基于 PyTorch Geometric (PyG) 搭建图卷积层，融合节点特征与有向边交互属性。
#    对应论文公式 (4)：H^{(l+1)} = \sigma( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} )
# 3. 全局池化：经多层图卷积后，通过 global_mean_pool 将图拓扑压缩并投射至 4096 维作为场景 Token X_s。
# =========================================================================================================
class GTNLayer(MessagePassing):
    """
    MaskGTN 图卷积层 (基于 PyG MessagePassing 消息传递机制):
    - in_channels: 节点特征输入维度 (初始为 151 维物体类别概率分布)
    - out_channels: 节点特征输出维度 (隐藏层通道 4096 维)
    - edge_attr_dim: 交互谓词边属性维度 (51 维谓词概率分布)
    """
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super(GTNLayer, self).__init__(aggr='add')  # 使用求和聚合 (Sum Aggregation)
        self.linear = nn.Linear(in_channels, out_channels)
        self.edge_attr_linear = nn.Linear(edge_attr_dim, in_channels)
        self.edge_attr_dim = edge_attr_dim

    def forward(self, x, edge_index, edge_attr=None):
        # 1. 添加自环 (Self-loops) 并构建对应的边属性
        edge_index, edge_attr = self.add_self_loops_with_edge_attr(edge_index, edge_attr, x.size(0), self.edge_attr_dim)
        # 2. 沿拓扑边执行消息传递与聚合 (Message Passing)
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # 3. 经过线性变换更新节点状态: H^{(l+1)} = H^{(l)} W^{(l)}
        x = self.linear(x)
        return x

    def message(self, x_j, edge_index, edge_attr):
        """
        消息函数: 将邻居节点特征 x_j 与变换后的边属性 edge_attr 进行多模态残差融合
        """
        if edge_attr is not None:
            edge_attr_transformed = self.edge_attr_linear(edge_attr.to(dtype=x_j.dtype))
            return x_j + edge_attr_transformed
        else:
            return x_j

    @staticmethod
    def add_self_loops_with_edge_attr(edge_index, edge_attr, num_nodes, edge_attr_dim):
        """为图拓扑添加自环 (Self-loops)，保证节点自身特征在卷积迭代中得以保留"""
        self_loops = torch.eye(num_nodes, dtype=torch.long)
        self_loops = self_loops.nonzero(as_tuple=False).t().contiguous().cuda()

        # 自环边属性初始化为全 0 向量
        self_loop_attr = torch.zeros((num_nodes, edge_attr_dim))

        # 合并真实拓扑边与自环边
        edge_index = torch.cat([edge_index.cuda(), self_loops.cuda()], dim=1)
        edge_attr = torch.cat([edge_attr.cuda(), self_loop_attr.cuda()], dim=0) if edge_attr is not None else None

        return edge_index, edge_attr


class GTN(nn.Module):
    """
    ORG 场景图网络主体 (Graph Transformer Network):
    输入: scene_feat 353 维 RelTR 检测特征
          - 前 51 维: 关系谓词概率分布 probas (边属性)
          - 中间 151 维: 主体类别概率分布 probas_sub (主节点特征)
          - 后 151 维: 客体类别概率分布 probas_obj (客节点特征)
    输出: [1, 4096] 聚合了环境物体交互拓扑的场景特征向量 X_s
    """
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, edge_attr_dim):
        super(GTN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.num_layers = num_layers

        # 级联多层 GTNLayer 图卷积层
        channels = in_channels
        for _ in range(num_layers):
            self.conv_layers.append(GTNLayer(channels, hidden_channels, edge_attr_dim))
            channels = hidden_channels

        # 最终投影层: hidden_channels (4096) -> out_channels (4096)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.linear.requires_grad_(True)

        for p in self.conv_layers.parameters():
            p.requires_grad = True
        self.linear.requires_grad_(True)

    def forward(self, scene_feat):
        nodes = []
        edges = []
        # 1. 拆分 RelTR 输出的三元组概率向量: 51维谓词 + 151维主体 + 151维客体
        probas, probas_sub, probas_obj = scene_feat[:, :51], scene_feat[:, 51:202], scene_feat[:, 202:]
        node_features = []
        edge_features = probas

        # 2. 动态解析实体节点与交互边 (构建非冗余图拓扑 G_i = (R_i, E_i))
        for i in range(probas.shape[0]):
            sub = CLASSES[probas_sub[i].argmax()]
            obj = CLASSES[probas_obj[i].argmax()]
            if sub not in nodes:
                nodes.append(sub)
                node_features.append(probas_sub[i])
            if obj not in nodes:
                nodes.append(obj)
                node_features.append(probas_obj[i])
            edges.append((sub, obj))
        
        # 构造 PyG 边索引张量 edge_index [2, Num_Edges]
        edge_index = torch.tensor([[nodes.index(src), nodes.index(dst)] for src, dst in edges],
                                  dtype=torch.long).t().contiguous()
        node_features = torch.stack(node_features, dim=0)
        # 封装为 PyG 标准 Data 图对象
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

        # 3. 多层图卷积前向聚合更新
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        for layer in self.conv_layers:
            x = layer(x, edge_index, edge_attr)

        # 4. 全图均值池化 (Global Mean Pooling): 将变长节点集合聚合为一个全局图表征
        x = gnn.global_mean_pool(x, torch.arange(0, x.size(0), dtype=torch.long, device=x.device))

        # 5. 线性映射至 4096 维输出
        x = self.linear(x)
        return x


def build_scene_tower():
    """构建场景图塔 (Scene Tower) - 2层GTN图卷积网络"""
    num_layers = 2
    in_channels = 151       # 151 维物体类别
    hidden_channels = 4096  # 4096 维隐藏层
    out_channels = 4096     # 4096 维输出
    edge_attr_dim = 51      # 51 维关系谓词边属性
    return GTN(num_layers, in_channels, hidden_channels, out_channels, edge_attr_dim)


def build_scene_projector():
    """构建场景特征后处理投影层 (4096 -> 4096)"""
    return nn.Linear(4096, 4096)



# =========================================================================================================
# Transformer 基础组件库 (供 Section 3.2 中 B-H MoE 的 Projection Experts 使用)
# ---------------------------------------------------------------------------------------------------------
# 包含 RMSNorm、RoPE 旋转位置编码、多头注意力 Attention、SwiGLU 门控前馈网络 FeedForward 等
# =========================================================================================================

class RMSNorm(torch.nn.Module):
    """
    均方根层归一化 (Root Mean Square Layer Normalization):
    公式: y = (x / RMS(x)) * gamma, 其中 RMS(x) = sqrt(mean(x^2) + eps)
    相比标准 LayerNorm 去掉了减均值 (mean-centering) 操作，节省约 7% 显存带宽与计算开销。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习通道缩放因子 gamma
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算均方根并归一化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@dataclass
class ModelArgs:
    """MoE 专家 TransformerBlock 网络结构超参数配置"""
    dim: int = 4096          # 隐藏层特征维度 (与 Vicuna-7B 保持一致)
    n_layers: int = 8        # 网络层数
    n_heads: int = 8         # 注意力头数 (在 MoE 专家内会被深拷贝并覆盖为 16)
    vocab_size: int = -1     # 词表大小
    multiple_of: int = 256   # 保证 SwiGLU 隐藏层维度为 256 的倍数以最大化 GPU 算力利用率
    norm_eps: float = 1e-5   # 归一化微小偏置项
    max_batch_size: int = 8  # 推理最大 BatchSize
    max_seq_len: int = 256   # 最大序列长度


default_linear_init = nn.init.xavier_uniform_


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转位置编码 (RoPE, Rotary Position Embedding) 的复数旋转因子矩阵:
    公式: freqs_cis[m, i] = exp(i * m * theta^(-2(i-1)/dim))
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 构建复数形式 cos + i*sin
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """调整 RoPE 频率矩阵形状以支持广播运算"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 Query 和 Key 向量应用旋转位置编码 (RoPE):
    通过复数乘法实现二维平面的向量逆时针旋转，从而注入相对位置语义。
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention):
    支持 RoPE 旋转位置编码、KV Cache 增量推理加速以及 FlashAttention 算子
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        # 线性映射层: W_q, W_k, W_v, W_o
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim)

        self.flash = True
        self.k_cache, self.v_cache = None, None

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],
                prompt=None):
        bsz, seqlen, _ = x.shape
        # 1. 线性投射生成 Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 拆分为多头形式: [B, SeqLen, NumHeads, HeadDim]
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # 2. 注入 RoPE 旋转位置编码
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 3. 管理 KV Cache (用于自回归推理加速)
        if self.k_cache is None or self.v_cache is None:
            keys, values = xk, xv
        else:
            self.k_cache = self.k_cache.to(xk)
            self.v_cache = self.v_cache.to(xv)
            self.k_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xk
            self.v_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xv
            keys = self.k_cache[:bsz, :start_pos + seqlen]
            values = self.v_cache[:bsz, :start_pos + seqlen]

        # 4. 执行注意力打分与加权聚合 (优先调用 FlashAttention)
        output = flash_attn_func(
            xq, keys, values, dropout_p=0.0, causal=mask is not None)
        output = output.contiguous().view(bsz, seqlen, -1)

        # 5. 经过输出矩阵 W_o 投射
        return self.wo(output)

    def allocate_kv_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        """显式分配 KV Cache 显存空间"""
        kv_cache_shape = (max_batch_size, max_seq_len, self.n_local_heads, self.head_dim)
        if self.k_cache is None or self.k_cache.size() != kv_cache_shape:
            self.k_cache = torch.empty(kv_cache_shape)
        if self.v_cache is None or self.v_cache.size() != kv_cache_shape:
            self.v_cache = torch.empty(kv_cache_shape)

    def destroy_kv_cache(self) -> None:
        """释放 KV Cache 显存"""
        self.k_cache, self.v_cache = None, None


class FeedForward(nn.Module):
    """
    SwiGLU 门控前馈神经网络 (Swish Gated Linear Unit):
    公式: FFN(x) = W_2 * (SiLU(W_1 * x) \odot (W_3 * x))
    相比标准 ReLU/GELU FFN，SwiGLU 拥有更丰富的信息门控通路，表征能力显著提升。
    """
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
    ):
        super().__init__()
        # 计算 SwiGLU 中间隐藏维度
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # 门控分支
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # 下投影层
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # 升维线性分支

    def _silu_gating(self, x, y):
        return F.silu(x) * y

    def forward(self, x):
        return self.w2(self._silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    """
    标准 Pre-LayerNorm Transformer 解码层:
    组成: RMSNorm -> Attention -> 残差连接 -> RMSNorm -> SwiGLU FFN -> 残差连接
    在 Hawkeye 中作为 B-H MoE 专家网络 (Projection Expert) 的重采样器 (Resampler)。
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _forward_ffn(self, h):
        return h + self.feed_forward(self.ffn_norm(h))

    def _forward_attention(self, x, start_pos, freqs_cis, mask, prompt):
        return x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, prompt)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],
                prompt=None):
        # 1. 注意力子层 (含前置归一化与残差连接)
        h = self._forward_attention(x, start_pos, freqs_cis, mask, prompt)
        # 2. 前馈子层 (含前置归一化与残差连接)
        out = self._forward_ffn(h)
        return out


class Mlp(nn.Module):
    """
    多层感知机 (MLP):
    用于 B-H MoE 中的模态路由器 (Modality Router R)，将融合输入映射为专家门控打分
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x



# =========================================================================================================
# Hawkeye 论文方法映射：第 3.2 节 平衡异构混合专家网络 (Balanced Heterogeneous MoE, B-H MoE)
# ---------------------------------------------------------------------------------------------------------
# 原理与对应公式：
# 1. 设计动机：人体骨骼动作特征 (pose_feat) 与物体交互拓扑图特征 (scene_feat) 属于异构多模态信息，
#    在联合表征时若简单求和易引发模态主导偏向问题。
# 2. 投影专家 (Projection Experts, PE): 搭建 N=2 个独立的重采样专家网络 E_i (基于 TransformerBlock)。
# 3. 模态路由器 (Modality Router R): 由 MLP 实现自适应软门控权重计算。
#    对应论文公式 (5)：y = LayerNorm( \sum_{i=1}^N R(h)_i E_i(h) )
#    依据输入画面的复杂度，软性权衡动作特征与拓扑图特征对最终情绪异常判断的贡献占比。
# =========================================================================================================
class MOE(nn.Module):
    """
    平衡异构混合专家网络 (B-H MoE):
    输入: pose_feat [B, 1, 4096], scene_feat [B, 1, 4096]
    输出: 动态加权融合后的场景+动作综合表征向量 [B, 4096]
    """
    def __init__(self, params):
        super(MOE, self).__init__()
        self.resample_layers = nn.ModuleDict()
        self.num_experts = 2          # 论文设定：N=2 个投影专家 (Expert 0 & Expert 1)
        self.num_resample_layers = 1  # 每个专家使用 1 层 TransformerBlock 重采样器
        
        # 1. 搭建 N=2 个异构投影专家网络 (Resample Layers)
        for expert in range(self.num_experts):
            expert = str(expert)
            self.resample_layers[expert] = nn.ModuleList()
            resampler_params = copy.deepcopy(params)
            resampler_params.n_heads = 16
            for layer_id in range(self.num_resample_layers):
                self.resample_layers[expert].append(
                    TransformerBlock(layer_id, resampler_params))

        self.resample_tokens = nn.ParameterDict()
        self.routers = nn.ModuleDict()
        self.clip_proj1 = nn.ModuleDict()
        self.clip_proj2 = nn.ModuleDict()
        self.start_tag = nn.ParameterDict()
        self.end_tag = nn.ParameterDict()

        for modal in ['pose']:
            # 2. 模态门控路由器 R: 由 MLP 实现，输入 4096，输出 2 个专家的门控打分
            self.routers[modal] = Mlp(
                4096, 4096 * 4, self.num_experts)

            # 可学习查询 Token (Resample Tokens，长度 30)
            self.resample_tokens[modal] = nn.Parameter(
                torch.empty([1, 30, resampler_params.dim]))
            nn.init.normal_(self.resample_tokens[modal], std=0.02)

            self.clip_proj1[modal] = nn.Sequential(
                nn.Linear(4096, resampler_params.dim),
                nn.LayerNorm(resampler_params.dim))

            # 融合后的投影与归一化层: 对应公式 (5) 外层的 LayerNorm
            self.clip_proj2[modal] = nn.Sequential(
                nn.Linear(resampler_params.dim, params.dim),
                nn.LayerNorm(params.dim))

            self.start_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))
            self.end_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, pose_feat, scene_feat):
        # 1. 拼接动作特征与场景图特征: [B, 2, 4096]
        image_feats = torch.cat((pose_feat, scene_feat), dim=1)
        
        # 2. 对应论文公式 (5): 路由器计算各个专家的路由权重 R(h)
        routing_weights = self.routers['pose'](image_feats).sigmoid()
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        image_feats_experts = []

        # 3. 将输入送入各个专家网络 E_i(h) 并乘以门控权重 R(h)_i
        for expert_id in range(self.num_experts):
            image_feats_expert = image_feats
            for layer in self.resample_layers[str(expert_id)]:
                image_feats_expert = layer(image_feats_expert, 0, None, None)
            image_feats_expert = image_feats_expert[:, :self.resample_tokens['pose'].size(1)]
            routing_weight = routing_weights[:, :self.resample_tokens['pose'].size(
                1), expert_id]
            # [B, L, D] * [B, L, 1] 专家加权组合
            image_feats_expert = image_feats_expert * routing_weight[:, :, None]
            image_feats_experts.append(image_feats_expert)
            
        # 4. 对专家输出求和并应用归一化投影层: y = LayerNorm(\sum R(h)_i E_i(h))
        image_feats = sum(image_feats_experts)
        image_feats = self.clip_proj2['pose'](image_feats)

        # 输出展平为标准 4096 维表征
        return image_feats.reshape(-1, 4096)


def build_moe():
    """构建平衡异构混合专家网络 (B-H MoE) 实例"""
    moe = MOE(ModelArgs())
    for param in moe.parameters():
        param.requires_grad = True
    return moe


def build_moe_projector():
    """构建 MoE 融合特征后处理投影层 (4096 -> 4096)"""
    return nn.Linear(4096, 4096)



# =========================================================================================================
# Hawkeye 多模态元模型基类 (LlavaMetaModel)
# ---------------------------------------------------------------------------------------------------------
# 功能职责:
# 1. 负责管理模型的各个子塔 (Towers):
#    - image_tower: 静态图像编码器 (LanguageBind)
#    - video_tower: 全局视频编码器 (LanguageBind Video, 抽取整体视觉序列)
#    - pose_tower: 人体骨骼姿态塔 (HigherHRNet, 对应 §3.1.1 ASG)
#    - scene_tower: 场景图神经网络塔 (RelTR + MaskGTN, 对应 §3.1.2 ORG)
#    - moe: 平衡异构混合专家网络 (对应 §3.2 B-H MoE)
# 2. 提供统一的模块初始化函数 (initialize_*_modules)，支持预训练权重挂载与 FSDP/ZeRO 显存并行
# =========================================================================================================
class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        # 动态检测配置中的塔结构并依次实例化
        if hasattr(config, "mm_image_tower"):
            self.image_tower = build_image_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        if hasattr(config, "mm_video_tower"):
            # 构建 LanguageBind 视频视觉塔
            self.video_tower = build_video_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        if hasattr(config, "mm_pose_tower"):
            # 构建动作敏感图 (ASG) 姿态塔: §3.1.1
            self.pose_tower = build_pose_tower()
            self.pose_projector = build_pose_projector()
        if hasattr(config, "mm_scene_tower"):
            # 构建物体关系敏感图 (ORG) 场景塔: §3.1.2
            self.scene_tower = build_scene_tower()
            self.scene_projector = build_scene_projector()
        if hasattr(config, "mm_moe"):
            # 构建平衡异构混合专家网络 (B-H MoE): §3.2
            self.moe = build_moe()
            self.moe_projector = build_moe_projector()

    # --- 统一的组件 Getter 接口 (自动解包分布式 FSDP 列表包装) ---
    def get_moe(self):
        moe = getattr(self, 'moe', None)
        if type(moe) is list:
            moe = moe[0]
        return moe

    def get_image_tower(self):
        image_tower = getattr(self, 'image_tower', None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower

    def get_pose_tower(self):
        pose_tower = getattr(self, 'pose_tower', None)
        if type(pose_tower) is list:
            pose_tower = pose_tower[0]
        return pose_tower

    def get_scene_tower(self):
        scene_tower = getattr(self, 'scene_tower', None)
        if type(scene_tower) is list:
            scene_tower = scene_tower[0]
        return scene_tower

    def initialize_image_modules(self, model_args, fsdp=None):
        """初始化图像模块与多模态投影层"""
        image_tower = model_args.image_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_image_tower = image_tower
        image_tower = build_image_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.image_tower = [image_tower]
        else:
            self.image_tower = image_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = image_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        self.mm_projector = build_vision_projector(self.config)

        # 若存在预训练投影层权重则定向载入
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_video_modules(self, model_args, fsdp=None):
        """初始化视频模块与多模态投影层 (加载 LanguageBind 视频编码器)"""
        video_tower = model_args.video_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_video_tower = video_tower
        video_tower = build_video_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.video_tower = [video_tower]
        else:
            self.video_tower = video_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = video_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_pose_modules(self, model_args, fsdp=None):
        """初始化动作敏感图模块 (ASG / HigherHRNet 姿态塔: §3.1.1)"""
        pose_tower = model_args.pose_tower
        self.config.mm_pose_tower = pose_tower

        pose_tower = build_pose_tower()
        if fsdp is not None and len(fsdp) > 0:
            self.pose_tower = [pose_tower]
        else:
            self.pose_tower = pose_tower

        self.pose_projector = build_pose_projector()

    def initialize_scene_modules(self, model_args, fsdp=None):
        """初始化物体关系敏感图模块 (ORG / MaskGTN 场景拓扑塔: §3.1.2)"""
        scene_tower = model_args.scene_tower
        self.config.mm_scene_tower = scene_tower

        scene_tower = build_scene_tower()
        if fsdp is not None and len(fsdp) > 0:
            self.scene_tower = [scene_tower]
        else:
            self.scene_tower = scene_tower

        self.scene_projector = build_scene_projector()

    def initialize_moe_modules(self, model_args, fsdp=None):
        """初始化平衡异构混合专家网络模块 (B-H MoE: §3.2)"""
        moe = model_args.moe
        self.config.mm_moe = moe
        moe = build_moe()
        if fsdp is not None and len(fsdp) > 0:
            self.moe = [moe]
        else:
            self.moe = moe

        self.moe_projector = build_moe_projector()



class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_image_tower(self):
        return self.get_model().get_image_tower()

    def get_video_tower(self):
        return self.get_model().get_video_tower()

    def get_pose_tower(self):
        return self.get_model().get_pose_tower()

    def get_scene_tower(self):
        return self.get_model().get_scene_tower()

    def get_moe(self):
        return self.get_model().get_moe()

    def get_all_tower(self, keys):
        tower = {key: getattr(self, f'get_{key}_tower') for key in keys}
        return tower

    def encode_images(self, images):
        """图像特征编码 (LanguageBind Image Tower + mm_projector)"""
        image_features = self.get_model().get_image_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_videos(self, videos):
        """视频特征编码 (LanguageBind Video Tower + mm_projector)"""
        video_features = self.get_model().get_video_tower()(videos)
        video_features = self.get_model().mm_projector(video_features)
        return video_features

    def encode_poses(self, poses):
        """动作特征编码 (HigherHRNet pose_tower + pose_projector) - 对应 §3.1.1"""
        pose_features = self.get_model().get_pose_tower()(poses)
        pose_features = self.get_model().pose_projector(pose_features)
        return pose_features

    def encode_scenes(self, scenes):
        """场景图特征编码 (RelTR + GTN scene_tower + scene_projector) - 对应 §3.1.2"""
        scene_features = self.get_model().get_scene_tower()(scenes)
        scene_features = self.get_model().scene_projector(scene_features)
        return scene_features

    def moe_route(self, pose_feat, scene_feat):
        """B-H MoE 专家路由融合动作与场景特征 - 对应 §3.2"""
        moe_featers = self.get_model().get_moe()(pose_feat.unsqueeze(0), scene_feat.unsqueeze(0))
        moe_featers = self.get_model().moe_projector(moe_featers)
        return moe_featers

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, X_modalities
    ):
        '''
        多模态序列拼接与大语言模型输入对齐函数:
        X_modalities 包含:
          - Xs: 原始视频帧张量
          - poses: 人体姿态骨骼点序列 (HigherHRNet 提取)
          - scenes: 场景交互关系三元组 (RelTR 提取)
          - keys: 模态标识列表 (例如 ['video'])
        '''
        Xs, poses, scenes, keys = X_modalities

        all_tower = self.get_all_tower(set(keys)) if len(keys) > 0 else None
        if all_tower is None or X_modalities[0][0] is None or input_ids.shape[1] == 1:
            if past_key_values is not None and all_tower is not None and Xs is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
        try:
            # 1. 抽取全局视频视觉特征 Token: [B, T_tokens, 4096]
            X_features_video = [getattr(self, 'encode_videos')(X.unsqueeze(0)).flatten(0, 1) for X in
                                Xs]  # expand to get batchsize
                
        except Exception as e:
            X_features_video = [getattr(self, 'encode_images')(X.unsqueeze(0)).flatten(0, 1) for X in Xs]

        X_features = []

        # =================================================================================================
        # Hawkeye 论文核心融合逻辑：
        # 将原始视频视觉特征与经 MoE 平衡路由后的场景细粒度 Token 进行跨维度拼接
        # =================================================================================================
        for i in range(len(X_features_video)):

            if poses[i] != None:
                # 动作敏感特征编码: §3.1.1
                X_features_pose = getattr(self, 'encode_poses')(poses[i])
                # 物体拓扑场景图编码: §3.1.2
                X_features_scene = getattr(self, 'encode_scenes')(scenes[i])
                # 异构 MoE 专家门控融合: §3.2
                X_moe_feat = getattr(self, 'moe_route')(X_features_pose, X_features_scene)

                # 将视频特征序列与 MoE 场景增强向量拼接，送入大语言模型 (Vicuna-7B)
                X_features.append(torch.cat((X_features_video[i], X_moe_feat), dim=0))
            else:
                X_features.append(X_features_video[i])


        # ---------------------------------------------------------------------------------------------
        # 2. 遍历 Batch 中的每一个样本，将多模态特征切入文本序列 (替换 <video> 占位符)
        # ---------------------------------------------------------------------------------------------
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_X_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 判断当前样本是否包含多模态占位符 (DeepSpeed Zero-3 兼容性分支)
            if (torch.any(torch.stack([cur_input_ids == X_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)).sum() == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_X_features = X_features[cur_X_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_X_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_X_idx += 1
                continue

            # 定位占位符索引 (例如 <video> 的位置)
            X_token_indices = torch.where(
                torch.any(torch.stack([cur_input_ids == X_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)
            )[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            # 循环切分并插入当前样本的多模态表征
            while X_token_indices.numel() > 0:
                cur_X_features = X_features[cur_X_idx]
                X_token_start = X_token_indices[0]

                # 分支 A: 启用了 Start/End 特殊标记包装 (如 <video_start> <video_patch>... <video_end>)
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end', False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:X_token_start - 1]).detach())
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[X_token_start - 1:X_token_start]))
                    cur_new_input_embeds.append(cur_X_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[X_token_start + 1:X_token_start + 2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:X_token_start])
                        # 视觉特征 Token 在训练计算交叉熵时被屏蔽 (填充 IGNORE_INDEX = -100)
                        cur_new_labels.append(torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                                         dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[X_token_start:X_token_start + 1])
                        cur_labels = cur_labels[X_token_start + 2:]
                # 分支 B: 标准直接占位替换 (Hawkeye 默认路径)
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:X_token_start]))
                    # 插入包含视频全局特征 + MoE 细粒度场景特征的多模态张量
                    cur_new_input_embeds.append(cur_X_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:X_token_start])
                        # 关键：多模态特征位置标签填入 IGNORE_INDEX，模型不计算视觉本身的重构损失，只监督文本回答
                        cur_new_labels.append(torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                                         dtype=labels.dtype))
                        cur_labels = cur_labels[X_token_start + 1:]

                cur_X_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end', False):
                    cur_input_ids = cur_input_ids[X_token_start + 2:]
                else:
                    cur_input_ids = cur_input_ids[X_token_start + 1:]
                X_token_indices = torch.where(
                    torch.any(torch.stack([cur_input_ids == X_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0]

            # 拼装占位符后剩余的文本 Token
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # ---------------------------------------------------------------------------------------------
        # 3. 动态 Padding 对齐: 处理同一 Batch 中不同序列长度不一致的问题
        # ---------------------------------------------------------------------------------------------
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            # Embeddings 补齐至最大长度 max_len
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            # Labels 补齐 IGNORE_INDEX
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # Attention Mask 左右对齐补齐
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_X_tokenizer(self, model_args, tokenizer):
        """
        初始化多模态 Tokenizer 与词嵌入权重 (Embedding Layer 扩容):
        1. 向 Tokenizer 注册特殊标记 (如 <video>, <image>, <video_start>, <video_end>)
        2. 扩展模型 Embedding 矩阵尺寸以容纳新 Token
        3. 对新增 Token 的权重使用已有词嵌入的均值进行初始化，防止训练初期产生剧烈梯度抖动
        4. 根据微调配置 (tune_mm_mlp_adapter) 冻结或解冻输入输出 Embedding 层的参数梯度
        """
        if model_args.mm_use_x_patch_token:
            for x in model_args.X:
                tokenizer.add_tokens([DEFAULT_X_PATCH_TOKEN[x.upper()]], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_x_start_end:
            num_new_tokens = 0
            for x in model_args.X:
                num_new_tokens += tokenizer.add_tokens(
                    [DEFAULT_X_START_TOKEN[x.upper()], DEFAULT_X_END_TOKEN[x.upper()]], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            # 对新增标记执行均值初始化 (Mean Initialization)
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # 参数微调梯度控制策略
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_x_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False