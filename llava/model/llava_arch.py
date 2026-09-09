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

# 导入 Python 抽象基类工具，用于定义 LlavaMetaForCausalLM 统一接口规范
from abc import ABC, abstractmethod

# 导入 PyTorch 深度学习核心计算库
import torch
# 导入 PyTorch 神经网络核心层与参数容器 (Module, Linear, Parameter 等)
import torch.nn as nn
# 引入 PyTorch Geometric (PyG) 图神经网络算子库 (例如全局均值池化 global_mean_pool)
import torch_geometric.nn as gnn
# 导入 PyG 核心消息传递基类，用于实现论文 Section 3.1.2 中的 MaskGTN 图卷积层
from torch_geometric.nn import MessagePassing
# 导入 PyG 标准图结构数据容器 Data (封装节点特征 x, 边索引 edge_index, 边属性 edge_attr)
from torch_geometric.data import Data
# 导入 PyG 图拓扑变换算子: add_self_loops 添加自环边，degree 计算节点出入度
from torch_geometric.utils import add_self_loops, degree

# 从多模态编码器模块导入静态图像塔与视频视觉塔的构建工厂函数
from .multimodal_encoder.builder import build_image_tower, build_video_tower
# 从多模态投影器模块导入视觉投影器 (MLP/Linear Projector) 的构建工厂函数
from .multimodal_projector.builder import build_vision_projector
# 导入数据类装饰器 dataclass，用于精简声明 Transformer 专家的配置结构体
from dataclasses import dataclass

# 支持 FairScale 分布式模型并行技术 (用于大参数量模型的张量并行与显存优化)
try:
    # 尝试导入 FairScale 模型并行环境初始化算子
    import fairscale.nn.model_parallel.initialize as fs_init
    # 尝试导入 FairScale 常见模型并行算子: 词表并行嵌入、行并行线性层、列并行线性层
    from fairscale.nn.model_parallel.layers import (
        ParallelEmbedding,
        RowParallelLinear,
        ColumnParallelLinear,
    )
except ImportError:
    # 若未安装 FairScale 依赖库，则跳过，在单机单卡或常规 DDP/DeepSpeed 环境下正常运行
    pass

# 导入类型提示注解 Optional (可选类型) 与 Tuple (元组类型)
from typing import Optional, Tuple
# 导入 PyTorch 常用函数式神经网络算子库 (激活函数 silu, 缩放点积注意力等)
import torch.nn.functional as F

# FlashAttention 高性能注意力计算算子
try:
    # 优先导入经过 CUDA 优化的 FlashAttention-2 核心函数
    from flash_attn import flash_attn_func
except ImportError:
    # 若环境未编译安装 flash_attn，则优雅回退到 PyTorch 2.0+ 原生的 scaled_dot_product_attention
    def flash_attn_func(q, k, v, dropout_p=0.0, causal=False):
        """FlashAttention 回退实现: 利用 PyTorch 2.0+ 原生高效注意力算子"""
        # 将 Query 张量从 [bsz, seqlen, n_heads, head_dim] 转置为标准注意力格式 [bsz, n_heads, seqlen, head_dim]
        q_t = q.transpose(1, 2)
        # 将 Key 张量转置为 [bsz, n_heads, seqlen, head_dim]
        k_t = k.transpose(1, 2)
        # 将 Value 张量转置为 [bsz, n_heads, seqlen, head_dim]
        v_t = v.transpose(1, 2)
        # 调用 PyTorch 原生内核级融合的缩放点积注意力计算，causal=True 表示应用因果下三角注意力掩码
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
        # 将计算结果维度还原为 [bsz, seqlen, n_heads, head_dim] 并返回
        return out.transpose(1, 2)

# 导入 Python 内存深浅拷贝模块 copy，用于克隆超参数结构体
import copy

# 导入 Hawkeye 多模态系统关键标记常量:
# IGNORE_INDEX: 交叉熵计算中忽略的标签 ID (-100)，用于遮蔽视觉 Token，避免大语言模型在多模态特征上反传文本损失
# X_TOKEN_INDEX: 多模态占位符在词表中的 Token ID (例如 <video> 对应 32000)
# DEFAULT_X_*_TOKEN: 各模态的 Patch 占位符及起始/结束特殊 Token (例如 <video_start>, <video_end>)
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
        # 调用父类 nn.Module 的构造函数完成基础模块初始化
        super(pose_feat, self).__init__()
        # 定义线性投影层: 将 85 维人体骨骼姿态坐标投影到与语言模型对齐的 4096 维特征空间
        self.pose_projector = nn.Linear(85, 4096)
        # 显式开启该线性层权重的梯度更新，确保微调训练时能自适应优化
        self.pose_projector.requires_grad_(True)

    def forward(self, pose_feat):
        # 展平骨骼输入特征: 从 [Batch, 5, 17] 展平为 [Batch, 85]
        pose_feat = pose_feat.view(pose_feat.size(0), -1)
        # 通过线性层将姿态特征投影映射至 4096 维隐藏空间: [Batch, 85] -> [Batch, 4096]
        pose_feat = self.pose_projector(pose_feat)
        # 返回对齐后的人体动作敏感特征表征
        return pose_feat


def build_pose_tower():
    """构建姿态塔 (Pose Tower) 实例"""
    # 实例化并返回基于 HigherHRNet 关节点的姿态特征线性编码器
    return pose_feat()


def build_pose_projector():
    """构建姿态特征后处理投影层 (4096 -> 4096)"""
    # 实例化并返回用于姿态特征微调转换的线性投影层 (保持 4096 维不变)
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
        # 调用 PyG 消息传递基类构造函数，聚合策略指定为累加求和 ('add')
        super(GTNLayer, self).__init__(aggr='add')
        # 节点状态更新线性变换层: 将聚合后的邻居信息映射至输出通道 out_channels (4096)
        self.linear = nn.Linear(in_channels, out_channels)
        # 边属性投影层: 将 51 维的关系谓词特征映射为与节点特征相同的维度 in_channels
        self.edge_attr_linear = nn.Linear(edge_attr_dim, in_channels)
        # 记录有向边属性的特征维度 (51)
        self.edge_attr_dim = edge_attr_dim

    def forward(self, x, edge_index, edge_attr=None):
        # 1. 为图拓扑添加自环 (Self-loops) 并补充全 0 自环边属性，保证节点自身历史状态在卷积迭代中得以保留
        edge_index, edge_attr = self.add_self_loops_with_edge_attr(edge_index, edge_attr, x.size(0), self.edge_attr_dim)
        # 2. 调用 PyG propagate 核心消息传递，沿边索引 edge_index 执行邻居节点与边属性的汇聚
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # 3. 线性变换更新节点状态: 对应公式 (4) 中的权重矩阵乘法 H^{(l+1)} = H^{(l)} * W^{(l)}
        x = self.linear(x)
        # 返回更新后的节点特征矩阵 [Num_Nodes, out_channels]
        return x

    def message(self, x_j, edge_index, edge_attr):
        """
        消息传递核心计算函数: 决定从源节点 j 流向目标节点 i 的信息内容
        - x_j: 源节点特征 [Num_Edges, in_channels]
        - edge_attr: 有向边的谓词特征 [Num_Edges, edge_attr_dim]
        """
        # 判断当前图拓扑中是否存在边属性特征
        if edge_attr is not None:
            # 将边属性转为与节点特征相同的数据类型 (如 float32/fp16)，并通过线性投影层变换
            edge_attr_transformed = self.edge_attr_linear(edge_attr.to(dtype=x_j.dtype))
            # 将边属性残差相加到源节点特征上，实现物体类别与交互动作的深层语义绑定
            return x_j + edge_attr_transformed
        else:
            # 若无边属性，则仅传递源节点特征本身
            return x_j

    @staticmethod
    def add_self_loops_with_edge_attr(edge_index, edge_attr, num_nodes, edge_attr_dim):
        """为图拓扑添加自环 (Self-loops)，保证节点自身特征在卷积迭代中得以保留"""
        # 生成大小为 [num_nodes, num_nodes] 的单位对角矩阵
        self_loops = torch.eye(num_nodes, dtype=torch.long)
        # 提取非零元素坐标作为自环边的源节点与目标节点索引: [2, num_nodes]
        self_loops = self_loops.nonzero(as_tuple=False).t().contiguous().cuda()

        # 为每个自环边创建全零属性向量: [num_nodes, edge_attr_dim]
        self_loop_attr = torch.zeros((num_nodes, edge_attr_dim))

        # 将原始拓扑边索引与自环边索引沿列拼接: [2, Num_Edges + num_nodes]
        edge_index = torch.cat([edge_index.cuda(), self_loops.cuda()], dim=1)
        # 将原始边属性与自环边全零属性沿行拼接: [Num_Edges + num_nodes, edge_attr_dim]
        edge_attr = torch.cat([edge_attr.cuda(), self_loop_attr.cuda()], dim=0) if edge_attr is not None else None

        # 返回扩增自环后的边索引与边属性元组
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
        # 调用父类 nn.Module 基础构造函数
        super(GTN, self).__init__()
        # 创建用于存放图卷积层的 ModuleList 容器
        self.conv_layers = nn.ModuleList()
        # 记录图卷积网络层数 (论文默认设定为 2 层)
        self.num_layers = num_layers

        # 逐层级联构建多层 GTNLayer 图卷积层
        channels = in_channels  # 初始输入特征维度为 151 (物体类别数)
        for _ in range(num_layers):
            # 向网络添加一层 GTNLayer 图卷积层
            self.conv_layers.append(GTNLayer(channels, hidden_channels, edge_attr_dim))
            # 后续各层的输入维度切换为中间隐藏通道 hidden_channels (4096)
            channels = hidden_channels

        # 最终全连接映射层: 将池化后的图表征映射至输出特征空间 (4096 -> 4096)
        self.linear = nn.Linear(hidden_channels, out_channels)
        # 开启该输出线性投影层的梯度更新
        self.linear.requires_grad_(True)

        # 遍历开启所有图卷积层参数的可微训练梯度
        for p in self.conv_layers.parameters():
            p.requires_grad = True
        # 确保输出线性层梯度始终保持开启
        self.linear.requires_grad_(True)

    def forward(self, scene_feat):
        # 初始化节点实体类别名列表 (用于图拓扑去重)
        nodes = []
        # 初始化有向边关系列表 (记录主体与客体之间的连线)
        edges = []
        # 1. 拆分 RelTR 输出的三元组概率向量: 51维谓词 + 151维主体 + 151维客体
        probas = scene_feat[:, :51]          # 提取前 51 维关系谓词概率分布 (作为边属性 edge_attr)
        probas_sub = scene_feat[:, 51:202]   # 提取中间 151 维主体类别概率分布 (主节点特征)
        probas_obj = scene_feat[:, 202:]     # 提取最后 151 维客体类别概率分布 (客节点特征)
        # 初始化节点特征向量列表
        node_features = []
        # 将谓词概率分布赋值给有向边的特征
        edge_features = probas

        # 2. 动态解析实体节点与交互边 (构建非冗余拓扑图 G_i = (R_i, E_i))
        for i in range(probas.shape[0]):
            sub = CLASSES[probas_sub[i].argmax()]  # 获取当前主体概率最高对应的物体类别名称
            obj = CLASSES[probas_obj[i].argmax()]  # 获取当前客体概率最高对应的物体类别名称
            if sub not in nodes:
                nodes.append(sub)                     # 将未出现过的新主体加入节点列表
                node_features.append(probas_sub[i])   # 将主体概率分布作为该节点的输入特征向量
            if obj not in nodes:
                nodes.append(obj)                     # 将未出现过的新客体加入节点列表
                node_features.append(probas_obj[i])   # 将客体概率分布作为该节点的输入特征向量
            edges.append((sub, obj))                  # 记录主体指向客体的有向交互边

        # 将节点名称转换为离散整数索引，构造 PyG 标准的二维边索引张量 edge_index [2, Num_Edges]
        edge_index = torch.tensor([[nodes.index(src), nodes.index(dst)] for src, dst in edges],
                                  dtype=torch.long).t().contiguous()
        # 将各节点特征列表堆叠为二维矩阵张量 [Num_Nodes, 151]
        node_features = torch.stack(node_features, dim=0)
        # 封装为 PyG 标准的 Data 图结构对象
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

        # 3. 提取图数据中的节点特征、边索引与边属性
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        # 逐层执行 MaskGTN 图卷积前向聚合与拓扑更新
        for layer in self.conv_layers:
            x = layer(x, edge_index, edge_attr)

        # 4. 全图均值池化 (Global Mean Pooling): 将任意数量的变长物体节点压缩为固定长度的图拓扑表征 [1, 4096]
        x = gnn.global_mean_pool(x, torch.arange(0, x.size(0), dtype=torch.long, device=x.device))

        # 5. 线性投影映射至 4096 维输出特征空间
        x = self.linear(x)
        # 返回最终场景图拓扑特征向量 X_s
        return x


def build_scene_tower():
    """构建场景图塔 (Scene Tower) - 2层GTN图卷积网络"""
    num_layers = 2          # 图卷积层数设定为 2 层
    in_channels = 151       # 节点输入特征维度为 151 (对应 Visual Genome 151 个物体类别概率分布)
    hidden_channels = 4096  # 中间隐藏特征通道维度为 4096 (对齐大语言模型特征空间)
    out_channels = 4096     # 输出场景图特征向量维度为 4096
    edge_attr_dim = 51      # 有向边谓词属性维度为 51 (对应 RelTR 51 个交互关系类别概率分布)
    # 实例化并返回基于 PyG 的场景图变换网络 (GTN)
    return GTN(num_layers, in_channels, hidden_channels, out_channels, edge_attr_dim)


def build_scene_projector():
    """构建场景特征后处理投影层 (4096 -> 4096)"""
    # 实例化并返回用于场景图特征微调映射的线性投影层
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
        # 调用父类 nn.Module 基础构造函数
        super().__init__()
        # 记录数值稳定常数 epsilon，防止除零错误
        self.eps = eps
        # 可学习的通道缩放仿射参数 gamma，初始化为全 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算均方根倒数并完成归一化: x / sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 升至 float32 计算提高数值稳定性，再还原为原始张量精度 (如 fp16/bf16)
        output = self._norm(x.float()).type_as(x)
        # 乘以通道缩放权重 gamma 并返回归一化后的张量
        return output * self.weight


@dataclass
class ModelArgs:
    """MoE 专家 TransformerBlock 网络结构超参数配置"""
    dim: int = 4096          # 隐藏层特征维度 (与 Vicuna-7B 保持一致)
    n_layers: int = 8        # 网络层数
    n_heads: int = 8         # 基础注意力头数 (在 MoE 专家内会被深拷贝并覆盖为 16)
    vocab_size: int = -1     # 词表大小 (-1 表示不直接用于语言解码词表)
    multiple_of: int = 256   # 保证 SwiGLU 隐藏层维度为 256 的倍数以最大化 GPU 算力利用率
    norm_eps: float = 1e-5   # 归一化微小偏置项 epsilon
    max_batch_size: int = 8  # 推理最大批次大小 (Batch Size)
    max_seq_len: int = 256   # 最大序列长度限制


# 默认线性层权重初始化策略: Xavier 均匀分布初始化
default_linear_init = nn.init.xavier_uniform_


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转位置编码 (RoPE, Rotary Position Embedding) 的复数旋转因子矩阵:
    公式: freqs_cis[m, i] = exp(i * m * theta^(-2(i-1)/dim))
    """
    # 计算每个通道偶数索引位置的基础角频率: theta^(-2(i-1)/dim)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成序列位置索引序列 [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # 计算位置向量与频率向量的外积得到各位置各维度的旋转角矩阵: [end, dim // 2]
    freqs = torch.outer(t, freqs).float()
    # 将极坐标 (幅值 1, 相位角 freqs) 转换为复数指数形式: cos(freqs) + i*sin(freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # 返回预计算的复数旋转位置编码矩阵
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """调整 RoPE 频率矩阵形状以支持广播运算"""
    # 获取输入张量 x 的总维度数 (通常为 4: [B, SeqLen, Heads, HeadDim])
    ndim = x.ndim
    # 断言张量维度数至少大于 1
    assert 0 <= 1 < ndim
    # 验证旋转因子形状与 [seq_len, dim//2] 严格相符
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # 构造广播形状 [1, seq_len, 1, dim//2]，仅在序列维与特征维保留原长度
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # 重塑复数旋转因子形状以便自动广播相乘
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
    # 将 Query 最后一个维度配对重塑为复数张量表达 [B, SeqLen, Heads, Dim//2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # 将 Key 最后一个维度配对重塑为复数张量表达 [B, SeqLen, Heads, Dim//2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 调整旋转因子矩阵形状以支持张量广播
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # 复数点乘实现逆时针旋转，再恢复为实数并在末尾展平
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    # 对 Key 同样执行复数旋转变换并在末尾展平
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    # 还原数据精度 (fp16/bf16) 并返回注入位置信息后的 (Q, K)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention):
    支持 RoPE 旋转位置编码、KV Cache 增量推理加速以及 FlashAttention 算子
    """
    def __init__(self, args: ModelArgs):
        # 调用父类构造函数
        super().__init__()
        # 记录本地注意力头数量
        self.n_local_heads = args.n_heads
        # 计算每个注意力头的特征维度: hidden_dim / n_heads
        self.head_dim = args.dim // args.n_heads

        # Query 线性投射矩阵 W_q
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim)
        # Key 线性投射矩阵 W_k
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim)
        # Value 线性投射矩阵 W_v
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim)
        # 输出线性合并投射矩阵 W_o
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim)

        # 标记启用 FlashAttention 加速模式
        self.flash = True
        # 初始化 KV Cache 缓存引用为 None
        self.k_cache, self.v_cache = None, None

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],
                prompt=None):
        # 提取输入张量的批次大小与序列长度
        bsz, seqlen, _ = x.shape
        # 1. 线性投射生成 Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 拆分为多头形式: [B, SeqLen, NumHeads, HeadDim]
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # 2. 注入 RoPE 旋转位置编码
        if freqs_cis is not None:
            # 应用 RoPE 旋转位置编码
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 3. 管理 KV Cache (用于自回归推理加速)
        if self.k_cache is None or self.v_cache is None:
            # 若未激活 KV Cache，则直接使用当前步生成的全部 Key 和 Value
            keys, values = xk, xv
        else:
            # 确保 K 缓存精度与计算设备与输入对齐
            self.k_cache = self.k_cache.to(xk)
            # 确保 V 缓存精度与计算设备与输入对齐
            self.v_cache = self.v_cache.to(xv)
            # 将新步产生的 Key 写入缓存切片
            self.k_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xk
            # 将新步产生的 Value 写入缓存切片
            self.v_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xv
            # 取出包含历史全部上下文的 Key 矩阵
            keys = self.k_cache[:bsz, :start_pos + seqlen]
            # 取出包含历史全部上下文的 Value 矩阵
            values = self.v_cache[:bsz, :start_pos + seqlen]

        # 4. 执行注意力打分与加权聚合 (优先调用 FlashAttention 算子计算注意力得分与加权融合)
        output = flash_attn_func(
            xq, keys, values, dropout_p=0.0, causal=mask is not None)
        # 展平多头并拼接还原为隐藏层向量 [B, SeqLen, Dim]
        output = output.contiguous().view(bsz, seqlen, -1)

        # 5. 经过输出矩阵 W_o 投射并返回最终注意力表征
        return self.wo(output)

    def allocate_kv_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        """显式分配 KV Cache 显存空间"""
        # 定义 KV Cache 的四维存储尺寸: [BatchSize, SeqLen, Heads, HeadDim]
        kv_cache_shape = (max_batch_size, max_seq_len, self.n_local_heads, self.head_dim)
        # 若 Key 缓存未分配或形状发生变化，则重新分配
        if self.k_cache is None or self.k_cache.size() != kv_cache_shape:
            self.k_cache = torch.empty(kv_cache_shape)
        # 若 Value 缓存未分配或形状发生变化，则重新分配
        if self.v_cache is None or self.v_cache.size() != kv_cache_shape:
            self.v_cache = torch.empty(kv_cache_shape)

    def destroy_kv_cache(self) -> None:
        """释放 KV Cache 显存"""
        # 释放 KV 缓存的张量引用，方便系统垃圾回收释放 GPU 显存
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
        # 调用父类 nn.Module 基础构造函数
        super().__init__()
        # SwiGLU 架构经验压缩系数: 取原隐藏层维度的 2/3
        hidden_dim = int(2 * hidden_dim / 3)
        # 向上取整为 multiple_of (256) 的整数倍，便于 Tensor Core 矩阵乘法硬件加速
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 门控分支线性层: 将输入映射至门控空间
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 下投影线性层: 将门控激活后的特征降维回模型维度 dim
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 上投影线性层: 与门控分支相乘的另一条线性通路
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _silu_gating(self, x, y):
        # SwiGLU 核心操作: 门控信号经过 SiLU 激活后与伴随分支逐元素相乘
        return F.silu(x) * y

    def forward(self, x):
        # 执行 W2 * (SiLU(W1 * x) * W3 * x) 计算并返回前馈结果
        return self.w2(self._silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    """
    标准 Pre-LayerNorm Transformer 解码层:
    组成: RMSNorm -> Attention -> 残差连接 -> RMSNorm -> SwiGLU FFN -> 残差连接
    在 Hawkeye 中作为 B-H MoE 专家网络 (Projection Expert) 的重采样器 (Resampler)。
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        # 调用父类 nn.Module 构造函数
        super().__init__()
        # 记录当前层的注意力头数量
        self.n_heads = args.n_heads
        # 记录模型隐藏层维度 (4096)
        self.dim = args.dim
        # 记录单头特征维度
        self.head_dim = args.dim // args.n_heads
        # 实例化多头注意力子模块
        self.attention = Attention(args)
        # 实例化 SwiGLU 门控前馈网络子模块
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        # 记录该 Transformer 层在网络中的深度编号
        self.layer_id = layer_id
        # 注意力子层的前置均方根归一化层
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 前馈网络子层的前置均方根归一化层
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _forward_ffn(self, h):
        # 执行前置 RMSNorm + SwiGLU FFN + 残差连接相加
        return h + self.feed_forward(self.ffn_norm(h))

    def _forward_attention(self, x, start_pos, freqs_cis, mask, prompt):
        # 执行前置 RMSNorm + Multi-Head Attention + 残差连接相加
        return x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, prompt)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],
                prompt=None):
        # 1. 依次执行注意力前置归一化、多头注意力与残差连接
        h = self._forward_attention(x, start_pos, freqs_cis, mask, prompt)
        # 2. 依次执行前馈网络前置归一化、SwiGLU FFN 与残差连接
        out = self._forward_ffn(h)
        # 返回经 TransformerBlock 深度编码后的上下文特征张量
        return out


class Mlp(nn.Module):
    """
    多层感知机 (MLP):
    用于 B-H MoE 中的模态路由器 (Modality Router R)，将融合输入映射为专家门控打分
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        # 调用父类 nn.Module 基础构造函数
        super().__init__()
        # 若未显式传入输出维度，则默认与输入维度保持一致
        out_features = out_features or in_features
        # 若未显式传入隐藏维度，则默认与输入维度保持一致
        hidden_features = hidden_features or in_features

        # 第一层线性变换: 从输入特征维度映射至隐藏维度
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 实例化非线性激活函数层 (默认为 GELU 激活函数)
        self.act = act_layer()
        # 第二层线性变换: 从隐藏维度映射至专家门控打分维度
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # 输入张量通过第一层全连接变换
        x = self.fc1(x)
        # 经过非线性激活函数提取高阶特征
        x = self.act(x)
        # 通过第二层全连接变换投射出最终打分
        x = self.fc2(x)
        # 返回 MLP 门控路由网络输出
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
    平衡异构混合专家网络 (Balanced Heterogeneous Mixture of Experts, B-H MoE):
    论文映射: 第 3.2 节 Balanced Heterogeneous Mixture of Experts
    输入: 
      - pose_feat: 人体动作敏感骨骼点特征张量，形状 [B, 1, 4096]
      - scene_feat: 物体关系敏感场景拓扑图特征张量，形状 [B, 1, 4096]
    输出:
      - 经自适应专家路由软门控加权融合后的场景+动作综合表征向量，形状 [B, 4096]
    """
    def __init__(self, params):
        # 调用父类 nn.Module 基础构造函数
        super(MOE, self).__init__()
        # 存储不同专家分支重采样层的 ModuleDict 字典
        self.resample_layers = nn.ModuleDict()
        # 论文设定：N=2 个投影专家 (Expert 0 为动作主导专家，Expert 1 为场景拓扑专家)
        self.num_experts = 2
        # 每个投影专家内包含的重采样层 (TransformerBlock) 层数，默认为 1 层
        self.num_resample_layers = 1
        
        # 依次构建 N=2 个独立的投影专家网络 (Projection Experts, E_i)
        for expert in range(self.num_experts):
            # 将专家编号转换为字符串键名 ('0', '1')，用于 ModuleDict 索引
            expert = str(expert)
            # 为当前专家初始化一个重采样层列表容器
            self.resample_layers[expert] = nn.ModuleList()
            # 深拷贝模型超参数对象，防止多专家间超参互相干扰覆盖
            resampler_params = copy.deepcopy(params)
            # 为重采样 TransformerBlock 指定 16 个注意力头
            resampler_params.n_heads = 16
            # 循环构建当前专家的每一层 TransformerBlock 重采样器
            for layer_id in range(self.num_resample_layers):
                # 实例化 TransformerBlock 并追加到当前专家的层列表中
                self.resample_layers[expert].append(
                    TransformerBlock(layer_id, resampler_params))

        # 存放不同模态的可学习查询 Token 的 ParameterDict 字典
        self.resample_tokens = nn.ParameterDict()
        # 存放模态路由门控网络 (Modality Router R) 的 ModuleDict 字典
        self.routers = nn.ModuleDict()
        # 存放前置特征投影层的 ModuleDict 字典
        self.clip_proj1 = nn.ModuleDict()
        # 存放后置融合投影归一化层的 ModuleDict 字典 (对应公式 5 外层的 LayerNorm)
        self.clip_proj2 = nn.ModuleDict()
        # 存放多模态起始特殊标记可学习嵌入的 ParameterDict 字典
        self.start_tag = nn.ParameterDict()
        # 存放多模态结束特殊标记可学习嵌入的 ParameterDict 字典
        self.end_tag = nn.ParameterDict()

        # 初始化以动作与场景交互为主的融合模态分支 (标识为 'pose')
        for modal in ['pose']:
            # 实例化模态路由器 R: 基于 MLP 实现自适应软门控权重计算，输入 4096 维，输出 2 个专家的门控打分
            self.routers[modal] = Mlp(
                4096, 4096 * 4, self.num_experts)

            # 初始化可学习查询 Token (Resample Tokens，序列长度 30，通道维度 resampler_params.dim)
            self.resample_tokens[modal] = nn.Parameter(
                torch.empty([1, 30, resampler_params.dim]))
            # 对可学习查询 Token 采用标准差 0.02 的正态分布进行随机权重初始化
            nn.init.normal_(self.resample_tokens[modal], std=0.02)

            # 构建前置线性投影与归一化序列层 (4096 -> resampler_params.dim)
            self.clip_proj1[modal] = nn.Sequential(
                nn.Linear(4096, resampler_params.dim),
                nn.LayerNorm(resampler_params.dim))

            # 构建融合后置投影与归一化序列层: 对应论文公式 (5) 外层的 LayerNorm 操作
            self.clip_proj2[modal] = nn.Sequential(
                nn.Linear(resampler_params.dim, params.dim),
                nn.LayerNorm(params.dim))

            # 初始化起始标记嵌入参数 (可学习向量，形状 [1, 1, params.dim])
            self.start_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))
            # 初始化结束标记嵌入参数 (可学习向量，形状 [1, 1, params.dim])
            self.end_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))

        # 遍历 MoE 模块内所有注册的可学习参数
        for param in self.parameters():
            # 显式设置需要梯度更新 (可参与端到端微调)
            param.requires_grad = True

    def forward(self, pose_feat, scene_feat):
        # 1. 跨模态特征拼接: 在序列维度 (dim=1) 拼接动作骨骼特征与场景图特征，张量形状变为 [B, 2, 4096]
        image_feats = torch.cat((pose_feat, scene_feat), dim=1)
        
        # 2. 论文公式 (5) 路由计算: 将融合特征输入模态路由器 R，并通过 Sigmoid 激活函数将输出映射到 (0, 1) 区间
        routing_weights = self.routers['pose'](image_feats).sigmoid()
        # 对门控打分在专家维度 (dim=-1) 进行 L1 归一化，使得各个专家的门控权重和恒等于 1
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        # 初始化列表容器，用于收集各个专家网络的加权输出特征
        image_feats_experts = []

        # 3. 遍历 N=2 个投影专家 (Expert 0 与 Expert 1)，分别计算专家重采样输出 E_i(h) 并与权重 R(h)_i 相乘
        for expert_id in range(self.num_experts):
            # 将拼接的多模态特征赋值给当前专家局部变量
            image_feats_expert = image_feats
            # 遍历当前专家的各层重采样 TransformerBlock 进行特征自注意力重编码
            for layer in self.resample_layers[str(expert_id)]:
                # 调用 TransformerBlock 执行前向传播计算 (start_pos=0, mask=None, prompt=None)
                image_feats_expert = layer(image_feats_expert, 0, None, None)
            # 截取重采样序列长度与可学习查询 Token 长度对齐
            image_feats_expert = image_feats_expert[:, :self.resample_tokens['pose'].size(1)]
            # 提取当前专家在对应序列长度上的门控路由权重切片，形状为 [B, L]
            routing_weight = routing_weights[:, :self.resample_tokens['pose'].size(
                1), expert_id]
            # 论文公式 (5) 核心: 将专家特征张量 [B, L, D] 与其门控权重 [B, L, 1] 逐元素相乘，实现自适应贡献度缩放
            image_feats_expert = image_feats_expert * routing_weight[:, :, None]
            # 将加权缩放后的专家特征存入列表容器
            image_feats_experts.append(image_feats_expert)
            
        # 4. 论文公式 (5) 加权求和: 将所有专家的加权特征累加求和: \sum_{i=1}^N R(h)_i E_i(h)
        image_feats = sum(image_feats_experts)
        # 论文公式 (5) 外层归一化: 经过最终的线性投影与 LayerNorm 归一化层: y = LayerNorm(...)
        image_feats = self.clip_proj2['pose'](image_feats)

        # 将融合表征展平重塑为标准二维张量 [B, 4096]，方便与视频主干 Token 拼接
        return image_feats.reshape(-1, 4096)


def build_moe():
    """
    构建平衡异构混合专家网络 (B-H MoE) 实例:
    负责实例化 MOE 模型并将其内部所有参数激活梯度，准备进行多模态融合训练
    """
    # 使用默认 LLaMA 解码器配置参数 (ModelArgs) 实例化 MOE 网络
    moe = MOE(ModelArgs())
    # 遍历 MoE 网络中的每一个参数张量
    for param in moe.parameters():
        # 显式将参数的 requires_grad 属性置为 True，确保在微调阶段能够更新梯度
        param.requires_grad = True
    # 返回构建好的 MoE 混合专家网络实例
    return moe


def build_moe_projector():
    """
    构建 MoE 融合特征后处理投影层:
    用于在多模态路由融合后，对输出向量施加可学习线性变换，保证与大语言模型 (Vicuna-7B) 词嵌入空间完美对齐
    输入维度: 4096, 输出维度: 4096
    """
    # 实例化并返回输入维度 4096、输出维度 4096 的全连接线性投影层
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
    """
    Hawkeye 多模态元模型基类 (LlavaMetaModel):
    主要职责:
      1. 统一管理并构建多模态子塔 (Towers):
         - image_tower: 静态图像编码器 (LanguageBind Image)
         - video_tower: 全局视频编码器 (LanguageBind Video, 负责提取宏观动态时序特征)
         - pose_tower: 动作敏感图骨骼姿态塔 (HigherHRNet, 论文 §3.1.1 ASG)
         - scene_tower: 物体交互拓扑图塔 (RelTR + MaskGTN, 论文 §3.1.2 ORG)
         - moe: 平衡异构混合专家网络 (论文 §3.2 B-H MoE)
      2. 维护各模态到大语言模型 (Vicuna-7B) 维度对齐的投影层 (Projectors)
      3. 提供各子模块的解包提取方法 (Getters)，自动适配 FSDP/ZeRO 显存分布式封装
      4. 提供各子模块的初始化接口 (initialize_*_modules)，支持预训练权重挂载与参数配置
    """
    def __init__(self, config):
        # 调用父类构造函数完成基础模型配置初始化
        super(LlavaMetaModel, self).__init__(config)

        # 判断配置中是否声明了图像特征塔 (mm_image_tower)
        if hasattr(config, "mm_image_tower"):
            # 延迟构建 LanguageBind 静态图像视觉编码塔 (设置 delay_load=True 优化分布式加载显存)
            self.image_tower = build_image_tower(config, delay_load=True)
            # 构建图像特征投影层 (将视觉维度映射至语言模型 4096 维度)
            self.mm_projector = build_vision_projector(config)
        # 判断配置中是否声明了视频特征塔 (mm_video_tower)
        if hasattr(config, "mm_video_tower"):
            # 延迟构建 LanguageBind 视频时序编码塔
            self.video_tower = build_video_tower(config, delay_load=True)
            # 构建视频特征多模态对齐投影层 (映射至 4096 维)
            self.mm_projector = build_vision_projector(config)
        # 判断配置中是否声明了人体姿态特征塔 (mm_pose_tower) - 论文 §3.1.1 动作敏感图
        if hasattr(config, "mm_pose_tower"):
            # 实例化 HigherHRNet 动作骨骼敏感图特征塔 (pose_feat)
            self.pose_tower = build_pose_tower()
            # 实例化动作特征对齐投影层 (pose_projector: 4096 -> 4096)
            self.pose_projector = build_pose_projector()
        # 判断配置中是否声明了场景拓扑关系塔 (mm_scene_tower) - 论文 §3.1.2 物体关系图
        if hasattr(config, "mm_scene_tower"):
            # 实例化 RelTR + MaskGTN 场景拓扑图编码塔 (scene_tower)
            self.scene_tower = build_scene_tower()
            # 实例化场景拓扑特征对齐投影层 (scene_projector: 4096 -> 4096)
            self.scene_projector = build_scene_projector()
        # 判断配置中是否声明了平衡异构专家混合网络 (mm_moe) - 论文 §3.2 B-H MoE
        if hasattr(config, "mm_moe"):
            # 实例化平衡异构混合专家网络 (包含 N=2 个 Projection Experts 与 Modality Router)
            self.moe = build_moe()
            # 实例化 MoE 融合输出特征投影层 (moe_projector: 4096 -> 4096)
            self.moe_projector = build_moe_projector()

    # --- 统一的组件 Getter 接口 (自动解包分布式 FSDP / DeepSpeed 列表封装) ---
    def get_moe(self):
        """获取 MoE 混合专家网络实例，若被 FSDP 包装为单元素列表则自动解包取出模型主体"""
        # 从模型实例属性中安全获取 moe 对象，若不存在则返回 None
        moe = getattr(self, 'moe', None)
        # 判断 moe 是否被分布式训练框架 (如 PyTorch FSDP) 封装为 list
        if type(moe) is list:
            # 提取列表中的首个元素作为真实计算模块
            moe = moe[0]
        # 返回解包后的 MoE 实例
        return moe

    def get_image_tower(self):
        """获取静态图像视觉编码塔实例，自动解包 FSDP 列表包装"""
        # 从模型实例属性中安全获取 image_tower 对象
        image_tower = getattr(self, 'image_tower', None)
        # 判断是否被分布式框架包装为列表容器
        if type(image_tower) is list:
            # 取出列表内的实际模型对象
            image_tower = image_tower[0]
        # 返回解包后的 image_tower 实例
        return image_tower

    def get_video_tower(self):
        """获取全局视频时序编码塔实例，自动解包 FSDP 列表包装"""
        # 从模型实例属性中安全获取 video_tower 对象
        video_tower = getattr(self, 'video_tower', None)
        # 判断是否被分布式框架包装为列表容器
        if type(video_tower) is list:
            # 取出列表内的实际模型对象
            video_tower = video_tower[0]
        # 返回解包后的 video_tower 实例
        return video_tower

    def get_pose_tower(self):
        """获取人体姿态动作骨骼塔实例 (ASG §3.1.1)，自动解包 FSDP 列表包装"""
        # 从模型实例属性中安全获取 pose_tower 对象
        pose_tower = getattr(self, 'pose_tower', None)
        # 判断是否被分布式框架包装为列表容器
        if type(pose_tower) is list:
            # 取出列表内的实际模型对象
            pose_tower = pose_tower[0]
        # 返回解包后的 pose_tower 实例
        return pose_tower

    def get_scene_tower(self):
        """获取场景物体拓扑关系图塔实例 (ORG §3.1.2)，自动解包 FSDP 列表包装"""
        # 从模型实例属性中安全获取 scene_tower 对象
        scene_tower = getattr(self, 'scene_tower', None)
        # 判断是否被分布式框架包装为列表容器
        if type(scene_tower) is list:
            # 取出列表内的实际模型对象
            scene_tower = scene_tower[0]
        # 返回解包后的 scene_tower 实例
        return scene_tower

    def initialize_image_modules(self, model_args, fsdp=None):
        """
        初始化静态图像编码器及其多模态线性投影层:
        参数:
          - model_args: 包含视觉塔路径、层数选择及超参的配置参数字典/命名空间
          - fsdp: 是否启用 Fully Sharded Data Parallel 分布式并行
        """
        # 读取配置中的图像视觉编码器路径或预训练标识
        image_tower = model_args.image_tower
        # 读取要选取的 ViT 特征层索引号 (例如倒数第二层 -2)
        mm_vision_select_layer = model_args.mm_vision_select_layer
        # 读取要截取的视觉特征类型 (例如 'patch' 或 'cls_patch')
        mm_vision_select_feature = model_args.mm_vision_select_feature
        # 读取预训练阶段投影层权重保存路径 (若有)
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        # 将图像塔标识存入模型全局配置中
        self.config.mm_image_tower = image_tower
        # 真正载入并实例化图像特征提取器
        image_tower = build_image_tower(model_args)

        # 检查是否处于 FSDP 并行分布式训练环境下
        if fsdp is not None and len(fsdp) > 0:
            # 在 FSDP 下封装为单元素列表，延迟梯度分块同步
            self.image_tower = [image_tower]
        else:
            # 单卡或标准 DDP 下直接赋给成员变量
            self.image_tower = image_tower

        # 标记当前模型启用了多模态投影层
        self.config.use_mm_proj = True
        # 记录投影层架构类型 (例如 'linear' 或 'mlp2x_gelu')
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        # 记录图像特征编码器的隐藏层特征维度 (如 1024)
        self.config.mm_hidden_size = image_tower.hidden_size
        # 记录视觉特征抽取的层数选择配置
        self.config.mm_vision_select_layer = mm_vision_select_layer
        # 记录视觉特征抽取的特征方式配置
        self.config.mm_vision_select_feature = mm_vision_select_feature

        # 按照配置构建视觉特征对齐投影层 (mm_projector)
        self.mm_projector = build_vision_projector(self.config)

        # 若提供了预训练好的投影层权重 checkpoint，则执行定向加载
        if pretrain_mm_mlp_adapter is not None:
            # 从本地硬盘加载权重字典到 CPU 内存
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            # 定义辅助闭包函数: 根据关键词提取并裁剪权重张量的键名前缀
            def get_w(weights, keyword):
                # 剥离前缀 keyword 后的名称并构建新的字典
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # 载入投影层预训练权重参数
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_video_modules(self, model_args, fsdp=None):
        """
        初始化全局视频编码器模块及其多模态对齐投影层 (LanguageBind Video):
        负责提取宏观全局视频序列的动态视觉 Token
        """
        # 读取配置中的视频编码塔标识路径
        video_tower = model_args.video_tower
        # 读取要选取的视频 ViT 特征层索引号
        mm_vision_select_layer = model_args.mm_vision_select_layer
        # 读取要提取的视频特征形式
        mm_vision_select_feature = model_args.mm_vision_select_feature
        # 读取多模态预训练投影权重文件路径
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        # 将视频塔信息同步至模型配置
        self.config.mm_video_tower = video_tower
        # 实例化构建 LanguageBind Video 视频时序特征提取塔
        video_tower = build_video_tower(model_args)

        # 判断是否处于 FSDP 分布式模式
        if fsdp is not None and len(fsdp) > 0:
            # 列表包装以规避 FSDP 根模块自动包裹递归冲突
            self.video_tower = [video_tower]
        else:
            # 正常赋值
            self.video_tower = video_tower

        # 设置启用投影层标记
        self.config.use_mm_proj = True
        # 设置投影器类型 (默认 'linear')
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        # 记录视频编码塔输出特征维度 (如 1024)
        self.config.mm_hidden_size = video_tower.hidden_size
        # 写入特征层选择参数
        self.config.mm_vision_select_layer = mm_vision_select_layer
        # 写入特征选取类型
        self.config.mm_vision_select_feature = mm_vision_select_feature

        # 实例化视频特征到大模型维度的投影转换层 (如 Linear(1024, 4096))
        self.mm_projector = build_vision_projector(self.config)

        # 若存在预训练投影层权重，则进行定向载入
        if pretrain_mm_mlp_adapter is not None:
            # 读取预训练权重文件
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            # 定义前缀剥离函数
            def get_w(weights, keyword):
                # 提取包含特定关键字的键并剔除前缀
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # 载入投影层状态字典
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_pose_modules(self, model_args, fsdp=None):
        """
        初始化动作敏感图模块 (ASG / HigherHRNet 姿态塔: 论文 §3.1.1):
        负责提取视频帧中人体的多尺度关键点骨骼动作特征
        """
        # 获取姿态塔标识名称
        pose_tower = model_args.pose_tower
        # 记录在模型配置字典中
        self.config.mm_pose_tower = pose_tower

        # 实例化动作敏感特征塔 (HigherHRNet pose_feat 封装)
        pose_tower = build_pose_tower()
        # 适配 FSDP 分布式显存封装
        if fsdp is not None and len(fsdp) > 0:
            # 分布式模式下打包为单元素列表
            self.pose_tower = [pose_tower]
        else:
            # 正常存储姿态塔实例
            self.pose_tower = pose_tower

        # 实例化动作特征线性投影层 (4096 -> 4096)
        self.pose_projector = build_pose_projector()

    def initialize_scene_modules(self, model_args, fsdp=None):
        """
        初始化物体关系敏感图模块 (ORG / MaskGTN 场景拓扑图塔: 论文 §3.1.2):
        负责利用可学习注意力图神经网络聚合 RelTR 提取的物体与动作交互图
        """
        # 获取场景拓扑塔标识名称
        scene_tower = model_args.scene_tower
        # 记录在模型配置字典中
        self.config.mm_scene_tower = scene_tower

        # 实例化物体关系图神经网络塔 (基于 PyG MessagePassing 的 MaskGTN)
        scene_tower = build_scene_tower()
        # 适配 FSDP 分布式显存封装
        if fsdp is not None and len(fsdp) > 0:
            # 分布式模式下打包为单元素列表
            self.scene_tower = [scene_tower]
        else:
            # 正常存储场景图神经网络塔实例
            self.scene_tower = scene_tower

        # 实例化场景特征线性投影层 (4096 -> 4096)
        self.scene_projector = build_scene_projector()

    def initialize_moe_modules(self, model_args, fsdp=None):
        """
        初始化平衡异构混合专家网络模块 (B-H MoE: 论文 §3.2):
        负责自适应平衡融合来自 ASG 的动作特征与来自 ORG 的场景拓扑交互特征
        """
        # 获取 MoE 模块标识名称
        moe = model_args.moe
        # 记录在模型配置字典中
        self.config.mm_moe = moe
        # 实例化构建 B-H MoE 网络结构 (2 个投影专家 + 门控路由器)
        moe = build_moe()
        # 适配 FSDP 分布式显存封装
        if fsdp is not None and len(fsdp) > 0:
            # 分布式模式下打包为单元素列表
            self.moe = [moe]
        else:
            # 正常存储 MoE 实例
            self.moe = moe

        # 实例化 MoE 融合后处理投影层 (4096 -> 4096)
        self.moe_projector = build_moe_projector()



class LlavaMetaForCausalLM(ABC):
    """
    Hawkeye 多模态因果语言模型 (Causal LM) 抽象元类:
    继承自 Python abc.ABC 抽象基类，用于规范所有多模态因果语言模型子类的公共接口。
    核心职责:
      1. 定义下层骨干 LLM 模型提取接口 get_model()
      2. 代理多模态特征塔 (Image, Video, Pose, Scene, MoE) 的获取函数
      3. 封装各模态的前向特征抽取管道:
         - encode_images: 静态图像编码与多模态投影
         - encode_videos: 全局动态视频编码与投影
         - encode_poses: 人体动作敏感骨骼点编码 (ASG §3.1.1)
         - encode_scenes: 物体关系拓扑图编码 (ORG §3.1.2)
         - moe_route: 动作与场景异构专家的平衡加权融合 (B-H MoE §3.2)
      4. 统一执行多模态与文本序列的对齐拼装与训练掩码生成 (prepare_inputs_labels_for_multimodal)
    """

    @abstractmethod
    def get_model(self):
        """
        抽象方法: 强制所有继承该基类的具体子类实现此方法以返回内部的 LlavaMetaModel 模型实例
        若未实现则在实例化具体模型类时直接抛出 TypeError 异常
        """
        pass

    def get_image_tower(self):
        """代理方法: 从内部核心底层模型中获取静态图像视觉编码塔实例"""
        # 调用子类实现的 get_model() 获取底层 LlavaMetaModel，并调用其 get_image_tower()
        return self.get_model().get_image_tower()

    def get_video_tower(self):
        """代理方法: 从内部核心底层模型中获取全局动态视频视觉编码塔实例"""
        # 调用底层 LlavaMetaModel 获取 video_tower 模块实例
        return self.get_model().get_video_tower()

    def get_pose_tower(self):
        """代理方法: 从内部核心底层模型中获取人体姿态动作骨骼塔实例 (ASG §3.1.1)"""
        # 调用底层 LlavaMetaModel 获取 pose_tower 模块实例
        return self.get_model().get_pose_tower()

    def get_scene_tower(self):
        """代理方法: 从内部核心底层模型中获取场景物体拓扑关系图塔实例 (ORG §3.1.2)"""
        # 调用底层 LlavaMetaModel 获取 scene_tower 模块实例
        return self.get_model().get_scene_tower()

    def get_moe(self):
        """代理方法: 从内部核心底层模型中获取平衡异构混合专家网络实例 (B-H MoE §3.2)"""
        # 调用底层 LlavaMetaModel 获取 moe 专家网络实例
        return self.get_model().get_moe()

    def get_all_tower(self, keys):
        """
        动态按需批量获取所有指定模态特征塔的字典映射:
        参数:
          - keys: 模态名称可迭代集合 (如 {'video', 'pose', 'scene'})
        返回:
          - 键为模态名、值为对应特征塔提取方法的字典
        """
        # 使用字典推导式，动态通过 getattr 反射获取各模态的 get_{key}_tower 成员函数
        tower = {key: getattr(self, f'get_{key}_tower') for key in keys}
        # 返回动态构建的特征塔方法字典映射
        return tower

    def encode_images(self, images):
        """
        静态图像特征编码管道:
        输入:
          - images: 输入图像张量，形状 [B, C, H, W]
        处理:
          1. 经过 LanguageBind Image Tower 提取 ViT 视觉 Patch Token
          2. 经过 mm_projector 线性映射至大语言模型词嵌入维度 4096
        返回:
          - image_features: 形状为 [B, num_patches, 4096] 的对齐视觉 Token
        """
        # 送入图像视觉编码器进行特征抽取，获取原始视觉隐层表征
        image_features = self.get_model().get_image_tower()(images)
        # 送入多模态投影层 (Linear 或 MLP)，将其通道投影至 LLM 的隐藏维度 4096
        image_features = self.get_model().mm_projector(image_features)
        # 返回映射对齐后的图像 Token 张量
        return image_features

    def encode_videos(self, videos):
        """
        全局动态视频特征编码管道:
        输入:
          - videos: 输入视频帧序列张量，形状 [B, C, T, H, W]
        处理:
          1. 经过 LanguageBind Video Tower 提取时空联合注意力表征
          2. 经过 mm_projector 线性映射至 4096 维
        返回:
          - video_features: 形状为 [B, T_tokens, 4096] 的全局动态时序视觉 Token
        """
        # 送入视频时序编码器，利用 3D 时空卷积/时序自注意力抽取视频宏观特征
        video_features = self.get_model().get_video_tower()(videos)
        # 通过多模态视频投影层映射至大语言模型统一特征空间 (4096 维)
        video_features = self.get_model().mm_projector(video_features)
        # 返回视频特征 Token 张量
        return video_features

    def encode_poses(self, poses):
        """
        动作敏感骨骼姿态特征编码管道 (Action-Sensitive Graph, ASG):
        论文映射: 第 3.1.1 节 Action-Sensitive Graph
        输入:
          - poses: 人体姿态关键点骨骼序列张量 (HigherHRNet 输出)
        处理:
          1. 经过 pose_tower (pose_feat 模块) 提取多尺度空间骨骼图嵌入
          2. 经过 pose_projector 映射至 4096 维
        返回:
          - pose_features: 动作敏感特征向量，形状 [1, 4096]
        """
        # 送入 HigherHRNet 姿态特征塔提取骨骼图动作表征
        pose_features = self.get_model().get_pose_tower()(poses)
        # 通过动作特征专用投影层对齐到大模型词嵌入空间维度 (4096 维)
        pose_features = self.get_model().pose_projector(pose_features)
        # 返回动作敏感特征向量
        return pose_features

    def encode_scenes(self, scenes):
        """
        物体拓扑场景图特征编码管道 (Object-Relation Graph, ORG):
        论文映射: 第 3.1.2 节 Object-Relation Graph
        输入:
          - scenes: 场景图交互三元组特征 (RelTR 提取的 353 维主体-客体-谓词交互表征)
        处理:
          1. 动态构图并通过 MaskGTN 图神经网络 (GNN) 进行关系聚合与注意力自环更新
          2. 经过全局均值池化与 scene_projector 投影至 4096 维
        返回:
          - scene_features: 物体关系敏感特征向量，形状 [1, 4096]
        """
        # 送入 RelTR + MaskGTN 场景图网络完成拓扑图交互传播与全局图池化
        scene_features = self.get_model().get_scene_tower()(scenes)
        # 通过场景特征专用投影层映射至大模型统一维度 (4096 维)
        scene_features = self.get_model().scene_projector(scene_features)
        # 返回场景拓扑交互特征向量
        return scene_features

    def moe_route(self, pose_feat, scene_feat):
        """
        B-H MoE 专家自适应门控平衡路由融合:
        论文映射: 第 3.2 节 Balanced Heterogeneous Mixture of Experts
        数学公式: y = LayerNorm( \sum_{i=1}^N R(h)_i E_i(h) )
        输入:
          - pose_feat: 动作骨骼敏感特征向量 [1, 4096]
          - scene_feat: 场景拓扑交互特征向量 [1, 4096]
        处理:
          1. 扩展 batch 维度并在 B-H MoE 中通过软门控路由器动态权衡两模态权重
          2. 经专家重采样后加权求和并执行 LayerNorm
          3. 经 moe_projector 变换输出与视频特征拼接准备
        返回:
          - moe_featers: 形状为 [1, 4096] 的细粒度平衡融合 Token
        """
        # 将骨骼特征与场景特征各自在首维增加 batch 维度 (unsqueeze(0)) 并输入 MoE 混合专家网络
        moe_featers = self.get_model().get_moe()(pose_feat.unsqueeze(0), scene_feat.unsqueeze(0))
        # 通过 MoE 融合特征投影层进行后续线性表征微调映射 (4096 -> 4096)
        moe_featers = self.get_model().moe_projector(moe_featers)
        # 返回自适应平衡融合后的细粒度多模态 Token
        return moe_featers

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, X_modalities
    ):
        """
        多模态序列切分、对齐拼装与训练掩码生成核心函数:
        论文核心映射:
          - 将 ASG 骨骼动作特征与 ORG 物体拓扑图特征送入 B-H MoE 进行软门控融合
          - 将融合后的细粒度多模态 Token 与 LanguageBind 全局视频视觉 Token 拼接
          - 将拼装后的多模态特征替换纯文本序列中的 <video> 占位符
          - 自动为多模态 Token 位置生成 IGNORE_INDEX (-100) 训练标签掩码，确保模型仅对文本预测计算交叉熵损失
          - 动态补齐批次 (Batch) 内样本长度差异 (Dynamic Padding) 并对齐注意力掩码 (Attention Mask)
        参数:
          - input_ids: 文本输入的 Token 索引张量，形状 [B, L_text]
          - attention_mask: 注意力掩码张量，形状 [B, L_text]
          - past_key_values: 自回归生成时的 KV 缓存 (若有)
          - labels: 训练监督目标标签张量，形状 [B, L_text]
          - X_modalities: 多模态输入元组 (Xs, poses, scenes, keys)
        """
        # 解包多模态输入元组: Xs(视频/图像帧), poses(骨骼关键点), scenes(场景交互三元组), keys(模态类型标识)
        Xs, poses, scenes, keys = X_modalities

        # 若提供了模态标识，则动态获取对应的多模态塔映射；否则置为 None
        all_tower = self.get_all_tower(set(keys)) if len(keys) > 0 else None
        # 边界检查: 若无多模态塔、无有效模态数据、或处于自回归文本单步解码阶段 (序列长度==1)
        if all_tower is None or X_modalities[0][0] is None or input_ids.shape[1] == 1:
            # 在自回归单步解码且存在 KV 缓存时，将 attention_mask 动态扩充 1 个步长
            if past_key_values is not None and all_tower is not None and Xs is not None and input_ids.shape[1] == 1:
                # 构造全 1 的注意力掩码，长度为已有缓存长度 + 1
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            # 直接返回原始输入参数，跳过多模态特征嵌入
            return input_ids, attention_mask, past_key_values, None, labels
        try:
            # 1. 尝试以视频形式批量编码全局时序视觉特征: 输入 [1, C, T, H, W]，展平为 [T_tokens, 4096]
            X_features_video = [getattr(self, 'encode_videos')(X.unsqueeze(0)).flatten(0, 1) for X in
                                Xs]  # 逐个样本扩展 batch 维度编码并展平为 Token 序列
                
        except Exception as e:
            # 若输入非时序视频或为静态图片，则平滑降级调用图像编码器抽取特征
            X_features_video = [getattr(self, 'encode_images')(X.unsqueeze(0)).flatten(0, 1) for X in Xs]

        # 初始化融合后的多模态特征列表容器
        X_features = []

        # =================================================================================================
        # Hawkeye 论文核心方法映射：
        # 第 3.1 节 (ASG/ORG 特征抽取) 与 第 3.2 节 (B-H MoE 专家自适应平衡融合)
        # 将视频全局特征 (Macro) 与经 MoE 门控平衡后的细粒度交互特征 (Micro) 进行跨维度拼接
        # =================================================================================================
        for i in range(len(X_features_video)):

            # 判断当前视频样本是否存在对应的人体骨骼姿态数据
            if poses[i] != None:
                # 动作敏感特征编码: 论文 §3.1.1 姿态塔抽取多尺度骨骼图特征 [1, 4096]
                X_features_pose = getattr(self, 'encode_poses')(poses[i])
                # 物体拓扑场景图编码: 论文 §3.1.2 场景塔聚合 RelTR 交互三元组 [1, 4096]
                X_features_scene = getattr(self, 'encode_scenes')(scenes[i])
                # 平衡异构混合专家门控路由融合: 论文 §3.2 B-H MoE 计算公式 (5) [1, 4096]
                X_moe_feat = getattr(self, 'moe_route')(X_features_pose, X_features_scene)

                # 将宏观时序视频特征序列与微观场景+动作 MoE 细粒度 Token 沿序列维度 (dim=0) 拼接，送入大语言模型
                X_features.append(torch.cat((X_features_video[i], X_moe_feat), dim=0))
            else:
                # 若无细粒度姿态数据，则仅保留基础全局视频特征
                X_features.append(X_features_video[i])


        # ---------------------------------------------------------------------------------------------
        # 2. 遍历 Batch 中的每一个样本，将多模态特征嵌入文本序列 (替换 <video> 占位符)
        # ---------------------------------------------------------------------------------------------
        # 存放批次中各个样本融合后的嵌入张量列表
        new_input_embeds = []
        # 存放批次中各个样本对齐后的监督标签列表 (若无监督标签则保持为 None)
        new_labels = [] if labels is not None else None
        # 多模态特征计数索引指针
        cur_X_idx = 0

        # 遍历批次中的每个文本序列
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 判断当前文本样本中是否包含多模态占位符 (例如 <video> Token ID)
            if (torch.any(torch.stack([cur_input_ids == X_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)).sum() == 0:
                # 兼容性分支: 若当前样本未包含任何多模态占位标记 (如纯文本微调或 DeepSpeed ZeRO-3 虚拟切片)
                half_len = cur_input_ids.shape[0] // 2
                # 获取当前多模态特征切片
                cur_X_features = X_features[cur_X_idx]
                # 计算前半段纯文本词嵌入
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                # 计算后半段纯文本词嵌入
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                # 拼接纯文本词嵌入与空的特征张量以保证梯度反向传播连通性
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_X_features[0:0], cur_input_embeds_2], dim=0)
                # 追加到批次嵌入列表中
                new_input_embeds.append(cur_input_embeds)
                # 同步追加标签
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                # 模态索引指针递增
                cur_X_idx += 1
                # 跳过后续占位符切分逻辑，处理下一条样本
                continue

            # 定位文本序列中所有多模态占位符出现的一维索引位置
            X_token_indices = torch.where(
                torch.any(torch.stack([cur_input_ids == X_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)
            )[0]
            # 初始化当前样本的重组词嵌入列表
            cur_new_input_embeds = []
            # 若处于模型训练状态，初始化当前样本的重构标签列表
            if labels is not None:
                # 获取当前样本对应的原始标签张量
                cur_labels = labels[batch_idx]
                # 初始化新的重组标签容器
                cur_new_labels = []
                # 严格断言标签序列长度与文本序列长度完全一致
                assert cur_labels.shape == cur_input_ids.shape

            # 循环遍历并切分替换序列中所有的多模态占位符
            while X_token_indices.numel() > 0:
                # 获取当前匹配的多模态融合特征张量 (视频 + MoE)
                cur_X_features = X_features[cur_X_idx]
                # 取出当前排在最前面的占位符起始下标
                X_token_start = X_token_indices[0]

                # 分支 A: 启用了 Start/End 特殊标记包装 (如 <video_start> <video_patch>... <video_end>)
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end', False):
                    # 截取并嵌入占位符前面的文本 Token (不需要更新梯度因此执行 detach)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:X_token_start - 1]).detach())
                    # 嵌入多模态开始前缀标记 (<video_start>)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[X_token_start - 1:X_token_start]))
                    # 插入核心多模态特征 Token 序列
                    cur_new_input_embeds.append(cur_X_features)
                    # 嵌入多模态结束后缀标记 (<video_end>)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[X_token_start + 1:X_token_start + 2]))
                    # 同步切分标签张量
                    if labels is not None:
                        # 保留多模态标记前面的文本标签
                        cur_new_labels.append(cur_labels[:X_token_start])
                        # 视觉多模态特征 Token 位置填充 IGNORE_INDEX (-100)，训练时不计算其自回归损失
                        cur_new_labels.append(torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                                         dtype=labels.dtype))
                        # 保留后续单个标记的标签
                        cur_new_labels.append(cur_labels[X_token_start:X_token_start + 1])
                        # 截断更新剩余文本标签切片
                        cur_labels = cur_labels[X_token_start + 2:]
                # 分支 B: 标准直接占位替换 (Hawkeye 官方默认执行路径)
                else:
                    # 嵌入占位符之前的纯文本 Token
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:X_token_start]))
                    # 插入包含视频全局特征 + MoE 细粒度场景特征的多模态张量
                    cur_new_input_embeds.append(cur_X_features)
                    # 同步构建监督标签
                    if labels is not None:
                        # 前序文本标签直接复制
                        cur_new_labels.append(cur_labels[:X_token_start])
                        # 关键机制: 多模态 Token 对应位置填充 IGNORE_INDEX (-100)，模型只监督后续生成的文本回答
                        cur_new_labels.append(torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                                         dtype=labels.dtype))
                        # 剥离已消费的前序标签，保留剩余标签
                        cur_labels = cur_labels[X_token_start + 1:]

                # 多模态特征索引用量自增 1
                cur_X_idx += 1
                # 根据是否启用了 Start/End 标记更新剩余 input_ids 文本序列
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end', False):
                    # 跳过占位符及结束标记
                    cur_input_ids = cur_input_ids[X_token_start + 2:]
                else:
                    # 仅跳过单个多模态占位符
                    cur_input_ids = cur_input_ids[X_token_start + 1:]
                # 重新定位剩余文本中是否还包含后续多模态占位符
                X_token_indices = torch.where(
                    torch.any(torch.stack([cur_input_ids == X_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0]

            # 处理末尾剩余的纯文本 Token
            if cur_input_ids.numel() > 0:
                # 判断是否包含特定微调配置
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end', False):
                    # 嵌入剩余文本并冻结梯度
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    # 正常执行词嵌入映射
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                # 追加剩余监督文本标签
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            # 将当前样本的所有词嵌入张量移动至当前模型所在设备
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            # 沿序列维度拼接当前样本所有切片，得到完整的输入嵌入向量 [Seq_len, 4096]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            # 收集到批次列表中
            new_input_embeds.append(cur_new_input_embeds)
            # 沿序列维度拼接当前样本所有切片的标签 [Seq_len]
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                # 收集到批次标签列表中
                new_labels.append(cur_new_labels)

        # ---------------------------------------------------------------------------------------------
        # 3. 动态 Padding 对齐: 处理同一批次 (Batch) 中不同视频/文本序列长度不一致的问题
        # ---------------------------------------------------------------------------------------------
        # 检测批次内各样本张量的序列长度是否存在差异
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            # 计算批次内所有样本的最大序列长度 max_len
            max_len = max(x.shape[0] for x in new_input_embeds)

            # 初始化对齐后的输入词嵌入列表
            new_input_embeds_align = []
            # 遍历每个样本的词嵌入张量
            for cur_new_embed in new_input_embeds:
                # 在序列末尾填充全 0 向量，补齐至批次最大长度 max_len
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                # 存入对齐列表
                new_input_embeds_align.append(cur_new_embed)
            # 将列表沿 Batch 维度堆叠为 3D 张量 [B, max_len, 4096]
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            # 对应补齐监督标签张量
            if labels is not None:
                # 初始化对齐后的标签列表
                new_labels_align = []
                # 备份原始未补齐标签列表以供 Attention Mask 计算使用
                _new_labels = new_labels
                # 遍历各个样本的标签
                for cur_new_label in new_labels:
                    # 在末尾填充 IGNORE_INDEX (-100)，确保补齐的 Padding Token 不产生损失反向传播
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    # 存入对齐标签列表
                    new_labels_align.append(cur_new_label)
                # 沿 Batch 维度堆叠为 2D 标签张量 [B, max_len]
                new_labels = torch.stack(new_labels_align, dim=0)

            # 对应左右动态补齐注意力掩码 (Attention Mask)
            if attention_mask is not None:
                # 初始化对齐注意力掩码列表
                new_attention_mask = []
                # 联合遍历原始掩码、融合后标签与对齐后标签
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    # 左侧填充 True (代表插入的多模态 Token 允许参与自注意力交互)
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    # 右侧填充 False (代表右侧末尾 Padding 补齐的 Token 被屏蔽，不参与注意力计算)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    # 沿序列维度拼接左侧掩码、原始文本掩码与右侧补齐掩码
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    # 收集对齐后的注意力掩码
                    new_attention_mask.append(cur_new_attention_mask)
                # 沿 Batch 维度堆叠为 2D 注意力掩码张量 [B, max_len]
                attention_mask = torch.stack(new_attention_mask, dim=0)
                # 严格断言注意力掩码维度与标签维度完全契合
                assert attention_mask.shape == new_labels.shape
        else:
            # 若批次内所有样本序列长度天然相同，直接沿 Batch 维度堆叠
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            # 若存在监督标签，直接沿 Batch 维度堆叠
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            # 若存在注意力掩码，仅需在左侧填充 True 以涵盖新插入的多模态 Token
            if attention_mask is not None:
                # 构造左侧补全的 True 掩码张量
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                # 在左侧拼接新掩码
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                # 严格断言掩码形状与词嵌入前两维完全一致
                assert attention_mask.shape == new_input_embeds.shape[:2]

        # 返回处理完毕的元组: input_ids 置 None (已替换为 new_input_embeds), attention_mask, past_key_values, new_input_embeds, new_labels
        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_X_tokenizer(self, model_args, tokenizer):
        """
        初始化多模态 Tokenizer 与词嵌入权重 (Embedding Layer 扩容与微调控制):
        核心步骤:
        1. 向 Tokenizer 分词器注册各模态特殊标记 (例如 <video>, <image>, <video_start>, <video_end>)
        2. 调用 resize_token_embeddings 扩容模型词表嵌入矩阵，使其尺寸与分词器词表完全对齐
        3. 对新增的特殊标记执行词嵌入均值初始化 (Mean Initialization)，避免冷启动时由于随机初始化引发剧烈梯度爆炸
        4. 根据微调模式 (tune_mm_mlp_adapter) 灵活控制输入词嵌入层与输出线性分类头 (LM Head) 的梯度更新开关
        5. 若指定了预训练的多模态适配器权重，则自动读取并加载新 Token 的预训练嵌入向量
        """
        # 判断配置中是否启用了多模态 Patch 占位标记 (例如 <video_patch>)
        if model_args.mm_use_x_patch_token:
            # 遍历支持的多模态列表 (例如 ['video', 'image'])
            for x in model_args.X:
                # 向分词器添加对应模态的 Patch 特殊标记 (special_tokens=True 确保不被二次拆分子词)
                tokenizer.add_tokens([DEFAULT_X_PATCH_TOKEN[x.upper()]], special_tokens=True)
            # 扩展模型词嵌入层 (Embedding Layer) 矩阵大小以匹配增加后的词表长度
            self.resize_token_embeddings(len(tokenizer))

        # 判断配置中是否启用了模态起止边界特殊标记 (例如 <video_start>, <video_end>)
        if model_args.mm_use_x_start_end:
            # 初始化新增特殊 Token 计数器
            num_new_tokens = 0
            # 遍历各模态
            for x in model_args.X:
                # 向分词器注册该模态的 Start 与 End 标记，并累加实际新增的 Token 数量
                num_new_tokens += tokenizer.add_tokens(
                    [DEFAULT_X_START_TOKEN[x.upper()], DEFAULT_X_END_TOKEN[x.upper()]], special_tokens=True)
            # 根据最新词表大小调整模型的 Token 词嵌入矩阵尺寸
            self.resize_token_embeddings(len(tokenizer))

            # 均值初始化策略 (Mean Initialization): 若存在新增加的特殊标记
            if num_new_tokens > 0:
                # 提取模型输入词嵌入层 (Input Embeddings) 的底层张量数据指针
                input_embeddings = self.get_input_embeddings().weight.data
                # 提取模型输出词嵌入层 (LM Head / Output Embeddings) 的底层张量数据指针
                output_embeddings = self.get_output_embeddings().weight.data

                # 计算除新增 Token 之外的所有已有词嵌入向量在通道维度的均值向量 [1, Hidden_size]
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                # 计算输出层已有权重的均值向量 [1, Hidden_size]
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                # 将输入嵌入层新增的末尾 Token 权重切片赋值为均值向量
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                # 将输出嵌入层新增的末尾 Token 权重切片赋值为均值向量
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # 参数微调梯度控制分支: 若仅对多模态投影层及相关标记进行轻量微调 (tune_mm_mlp_adapter)
            if model_args.tune_mm_mlp_adapter:
                # 遍历输入词嵌入层的所有参数
                for p in self.get_input_embeddings().parameters():
                    # 激活输入词嵌入梯度的更新，允许学习新注册的视觉标记表征
                    p.requires_grad = True
                # 遍历输出词嵌入分类头的所有参数
                for p in self.get_output_embeddings().parameters():
                    # 冻结输出分类头参数，避免语言模型预测词表分布发生灾难性遗忘
                    p.requires_grad = False

            # 若提供了预训练多模态投影适配器权重路径，则从 checkpoint 中加载新标记权重
            if model_args.pretrain_mm_mlp_adapter:
                # 从文件中加载预训练权重字典到 CPU 内存
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                # 提取其中的词嵌入权重字典项
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                # 严格断言新增标记数量为 2 (<video_start> 与 <video_end>)
                assert num_new_tokens == 2
                # 情形 1: 若预训练权重与当前嵌入层完整维度完全一致
                if input_embeddings.shape == embed_tokens_weight.shape:
                    # 仅复制末尾新增 Token 的嵌入向量
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                # 情形 2: 若预训练权重仅单独保存了新增的 2 个 Token 向量
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    # 直接将全部预训练权重赋值给当前嵌入层的最后切片
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    # 若形状无法匹配则抛出明确的异常报错
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        # 分支: 若未启用 Start/End 仅启用了 Patch Token
        elif model_args.mm_use_x_patch_token:
            # 若处于轻量适配器微调模式
            if model_args.tune_mm_mlp_adapter:
                # 冻结输入词嵌入层所有参数
                for p in self.get_input_embeddings().parameters():
                    # 冻结输入嵌入参数
                    p.requires_grad = False
                # 冻结输出词嵌入层所有参数
                for p in self.get_output_embeddings().parameters():
                    # 冻结输出嵌入参数
                    p.requires_grad = False