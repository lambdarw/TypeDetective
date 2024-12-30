import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiGAT(nn.Module):
    """
    多图注意力网络
    """

    def __init__(self, args):
        super(MultiGAT, self).__init__()

        self.args = args
        self.dropout = nn.Dropout(args.dropout)  # dropout，防止过拟合
        self.input_dim_MLP = args.gat_hidden_size  # MLP 中输入层的维度 768
        # 使用情感信息
        if self.args.use_senticnet:
            self.input_dim_MLP = args.embedding_dim

        # 4个GAT，分别用于4个不同的维度
        self.gat1 = GAT(args, num_layers=args.gat_num_layers)
        self.gat2 = GAT(args, num_layers=args.gat_num_layers)
        self.gat3 = GAT(args, num_layers=args.gat_num_layers)
        self.gat4 = GAT(args, num_layers=args.gat_num_layers)

        # MLP
        # 具有激活函数的全连接层
        layers = [nn.Linear(self.input_dim_MLP, args.final_hidden_size), nn.ReLU()]  # 768 -> 128
        # 通过循环创建额外的具有激活函数的全连接层
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size, args.final_hidden_size), nn.ReLU()]
        # 创建序列模型，用于对图卷积输出进行进一步处理
        self.fcs1 = nn.Sequential(*layers)
        self.fcs2 = nn.Sequential(*layers)
        self.fcs3 = nn.Sequential(*layers)
        self.fcs4 = nn.Sequential(*layers)
        # 将最终处理后的特征映射到输出类别上进行最终的分类
        self.fc_final1 = nn.Linear(args.final_hidden_size, args.num_classes)
        self.fc_final2 = nn.Linear(args.final_hidden_size, args.num_classes)
        self.fc_final3 = nn.Linear(args.final_hidden_size, args.num_classes)
        self.fc_final4 = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, p_mask, feature, c_node):
        # 获取 batch size
        B = p_mask.size(0)
        # 如果不禁用特殊节点，通过在 p_mask 最后添加一个全 1 的特殊节点，得到新的 new_p_mask  (B, N+1)
        new_p_mask = torch.cat((p_mask, torch.ones(B, 1).cuda()), 1) if not self.args.no_special_node else p_mask

        # 如果不禁用特殊节点，将特殊节点特征 c_node 与原始节点特征 feature 进行连接
        if not self.args.no_special_node:
            feature1 = torch.cat((feature, c_node), 1)  # (B, N+1, D)
            feature2 = torch.cat((feature, c_node), 1)
            feature3 = torch.cat((feature, c_node), 1)
            feature4 = torch.cat((feature, c_node), 1)

        # 将节点特征和节点注意掩码输入到 GAT，获得输出
        gat_out1 = self.gat1(new_p_mask, feature1)  # (B, N+1, D)
        gat_out2 = self.gat2(new_p_mask, feature2)
        gat_out3 = self.gat3(new_p_mask, feature3)
        gat_out4 = self.gat4(new_p_mask, feature4)

        # 如果不禁用特殊节点，选择所有样本的最后一个节点的特征
        if not self.args.no_special_node:
            out1 = gat_out1[:, -1]  # (B, D)
            out2 = gat_out2[:, -1]
            out3 = gat_out3[:, -1]
            out4 = gat_out4[:, -1]
        # 如果禁用特殊节点，取倒数第二个维度（节点维度）上的均值
        else:
            out1 = gat_out1.mean(dim=-2)  # (B, D)
            out2 = gat_out2.mean(dim=-2)
            out3 = gat_out3.mean(dim=-2)
            out4 = gat_out4.mean(dim=-2)

        x1 = self.dropout(out1)  # dropout，防止过拟合
        x1 = self.fcs1(x1)  # 进一步做特征变换
        # logit 是指神经网络模型在输出层的线性输出，通常在没有应用激活函数（如 sigmoid）之前的值（在分类问题中是用于计算类别概率的原始分数）
        logit1 = self.fc_final1(x1)
        x2 = self.dropout(out2)
        x2 = self.fcs2(x2)
        logit2 = self.fc_final2(x2)
        x3 = self.dropout(out3)
        x3 = self.fcs3(x3)
        logit3 = self.fc_final3(x3)
        x4 = self.dropout(out4)
        x4 = self.fcs4(x4)
        logit4 = self.fc_final4(x4)

        return logit1, logit2, logit3, logit4


class GAT(nn.Module):
    """
    图注意力网络
    """

    def __init__(self, args, num_layers):
        super(GAT, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.d_model = args.embedding_dim  # 768 788
        self.n_head = 6
        # 模型的总维度d_model被均匀地分割到每个头，每个头可以并行地学习不同的信息
        assert (self.d_model % self.n_head == 0)
        # d_head为每个头分到的维度（n_head * d_head = D）
        self.d_head = int(self.d_model / self.n_head)

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Linear(args.d_model, args.d_model, bias=False)
        self.attn = nn.Linear(self.d_head * 2, self.n_head, bias=False)
        self.actv = nn.Tanh()  # 激活函数
        self.dropout = nn.Dropout(args.gat_dropout)  # dropout，减少过拟合

    def forward(self, p_mask, feature):
        B = p_mask.size(0)
        new_p_mask = p_mask.unsqueeze(-1)  # (B,N+1,1)
        # 生成邻接矩阵 (B,N+1,1)*(B,1,N+1)
        adj = new_p_mask.bmm(new_p_mask.transpose(2, 1))  # (B,N+1,N+1)

        for l in range(self.num_layers):
            # 当前层的输入特征被保存为“残差”
            residual = feature  # (B,N+1,D)
            # 创建一个索引列表，表示每个注意力头的索引
            head_list = list(range(self.n_head))  # [0,1,2,3,4,5]

            # 通过一个线性层self.W变换特征并重新整形，为每个头产生 Q 和 K (Query 和 Key)：(B,N+1,D) -> (B,N+1,n_head,d_head)
            qk = self.W(feature).view(B, self.args.max_post + 1, self.n_head, self.d_head)
            # mh_q 和 mh_k 是对qk的扩展版本：(B,N+1,N+1,n_head,d_head) (B,N+1,N+1,n_head,d_head)
            mh_q, mh_k = qk[:, :, None, :, :].expand(-1, -1, self.args.max_post + 1, -1, -1), \
                qk[:, None, :, :, :].expand(-1, self.args.max_post + 1, -1, -1, -1)
            # 连接mh_q和mh_k并通过self.attn线性层，为每对节点和每个注意力头计算权重
            # (B, N+1, N+1, n_head, 2*d_head) -> (B, N+1, N+1, n_head, n_head) -> (B, N+1, N+1, n_head)
            mh_attn = self.attn(torch.cat([mh_q, mh_k], dim=-1))[:, :, :, head_list, head_list]  # (B, N+1, N+1, n_head)
            # 应用激活函数tanh
            mh_attn = self.actv(mh_attn)  # (B, N+1, N+1, n_head)

            # 应用掩码，将邻接矩阵adj中为0的位置的注意力权重设置为一个非常小的负值（-1e-8）（这样在应用softmax时，这些位置的权重将接近于0）
            mh_attn = mh_attn.masked_fill((1 - adj)[:, :, :, None].expand_as(mh_attn).bool(), -1e-8)
            # 使用softmax进行归一化，确保权重之和为1
            mh_attn = F.softmax(mh_attn, dim=-2)
            mh_attn = self.dropout(mh_attn)  # dropout，防止过拟合

            # mh_attn: (B, N+1, N+1, n_head)  qk: (B, N+1, n_head, d_head)
            # 加权的特征聚合：mh_attn为qk中的每个查询、头和特征维度提供加权系数，这是多头注意力机制的关键组成部分
            # (B, N+1, n_head, d_head)
            mh_hid = torch.tanh(torch.einsum('bqkn,bknd->bqnd', mh_attn, qk))
            # 输出被重新整形并与残差（原始特征）相加，并整形，得到该层的最终输出：(B, N+1, D)
            feature = residual + mh_hid.reshape(B, self.args.max_post + 1, -1)

        return feature


class AttentionGRU(nn.Module):
    """
    GRU + 注意力机制
    """

    def __init__(self, args):
        super(AttentionGRU, self).__init__()
        self.args = args
        self.input_dim = args.embedding_dim  # 768 788
        self.hidden_dim = args.embedding_dim
        # dropout，防止过拟合
        self.dropout = nn.Dropout(args.dropout)
        # GRU + Attention
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.attention = nn.Linear(self.hidden_dim, 1)
        # BiGRU + Attention
        # self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        # self.attention = nn.Linear(self.hidden_dim, 1)  # 在类定义中添加一个全连接层用于维度降低
        # self.dim_reduction = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, p_mask, feature):
        # GRU 层
        output, _ = self.gru(feature)  # output形状：(B, N, D) 或 (B, N, 2D)
        # output = self.dim_reduction(output)  # 将输出维度从 2D 降至 D
        # 注意力权重
        attention_weights = F.softmax(self.attention(output), dim=1)  # (B, N, 1)
        # dropout，防止过拟合
        attention_weights = self.dropout(attention_weights)
        # 使用帖子掩码（unsqueeze(-1) 用来确保掩码在乘法操作中的维度与 attention_weights 一致）
        attention_weights = attention_weights * p_mask.unsqueeze(-1)
        # 加权求和（带注意力）
        weighted_output = torch.sum(output * attention_weights, dim=1)  # (B, D)
        return weighted_output  # (B, D)


class InteractionAttention(nn.Module):
    """
    人格特质互动层
    """

    def __init__(self, args):
        super(InteractionAttention, self).__init__()
        self.feature_dim = args.embedding_dim  # 768 788
        self.query = nn.Linear(self.feature_dim, self.feature_dim)
        self.key = nn.Linear(self.feature_dim, self.feature_dim)
        self.value = nn.Linear(self.feature_dim, self.feature_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.gat_dropout)  # dropout，减少过拟合

    def forward(self, feature):
        # 当前输入特征被保存为“残差”
        residual = feature  # (B, N, D)
        # feature: (B, N, D)
        Q = self.query(feature)  # (B, N, D)
        K = self.key(feature)  # (B, N, D)
        V = self.value(feature)  # (B, N, D)
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(feature.size(-1))  # Scale by sqrt(D)
        attention = self.softmax(scores)  # (B, N, N)
        attention = self.dropout(attention)  # dropout，防止过拟合
        # 应用注意力权重
        output = torch.matmul(attention, V)  # (B, N, D)
        # 输出与残差（原始特征）相加得到最终输出
        output = residual + output  # (B, N, D)
        return output


class Multi_AttnGRU(nn.Module):
    """
    多 Attention_GRU
    """

    def __init__(self, args):
        super(Multi_AttnGRU, self).__init__()

        self.args = args
        self.dropout = nn.Dropout(args.dropout)  # dropout，防止过拟合
        self.input_dim_MLP = args.gat_hidden_size  # MLP 中输入层的维度 768
        # 使用情感信息
        if self.args.use_senticnet:
            self.input_dim_MLP = args.embedding_dim

        # 使用结合注意力机制的 GRU
        self.attention_gru1 = AttentionGRU(args)
        self.attention_gru2 = AttentionGRU(args)
        self.attention_gru3 = AttentionGRU(args)
        self.attention_gru4 = AttentionGRU(args)
        # 初始化交互注意力模块
        self.interaction_attention = InteractionAttention(args)

        # MLP
        # 具有激活函数的全连接层
        layers = [nn.Linear(self.input_dim_MLP, args.final_hidden_size), nn.ReLU()]  # 768 -> 128
        # 通过循环创建额外的具有激活函数的全连接层
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size, args.final_hidden_size), nn.ReLU()]
        # 创建序列模型，用于对图卷积输出进行进一步处理
        self.fcs1 = nn.Sequential(*layers)
        self.fcs2 = nn.Sequential(*layers)
        self.fcs3 = nn.Sequential(*layers)
        self.fcs4 = nn.Sequential(*layers)
        # 将最终处理后的特征映射到输出类别上进行最终的分类
        self.fc_final1 = nn.Linear(args.final_hidden_size, args.num_classes)
        self.fc_final2 = nn.Linear(args.final_hidden_size, args.num_classes)
        self.fc_final3 = nn.Linear(args.final_hidden_size, args.num_classes)
        self.fc_final4 = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, p_mask, feature):
        # 使用结合注意力机制的 GRU
        gru_feature1 = self.attention_gru1(p_mask, feature)  # (B, D)
        gru_feature2 = self.attention_gru2(p_mask, feature)
        gru_feature3 = self.attention_gru3(p_mask, feature)
        gru_feature4 = self.attention_gru4(p_mask, feature)
        # 应用交互注意力模块，交互注意力学习人格特质间的潜在关系
        trait_features = torch.cat([gru_feature1.unsqueeze(1), gru_feature2.unsqueeze(1),
                                    gru_feature3.unsqueeze(1), gru_feature4.unsqueeze(1)], dim=1)
        trait_interaction_features = self.interaction_attention(trait_features)
        final_gru_feature1, final_gru_feature2, final_gru_feature3, final_gru_feature4 = torch.unbind(
            trait_interaction_features, dim=1)  # (B, D)

        x1 = self.dropout(final_gru_feature1)  # dropout，防止过拟合
        x1 = self.fcs1(x1)  # 进一步做特征变换
        # logit 是指神经网络模型在输出层的线性输出，通常在没有应用激活函数（如 sigmoid）之前的值（在分类问题中是用于计算类别概率的原始分数）
        logit1 = self.fc_final1(x1)
        x2 = self.dropout(final_gru_feature2)
        x2 = self.fcs2(x2)
        logit2 = self.fc_final2(x2)
        x3 = self.dropout(final_gru_feature3)
        x3 = self.fcs3(x3)
        logit3 = self.fc_final3(x3)
        x4 = self.dropout(final_gru_feature4)
        x4 = self.fcs4(x4)
        logit4 = self.fc_final4(x4)

        return logit1, logit2, logit3, logit4


class PD_HAN(nn.Module):
    """
    异构图注意力网络
    """

    def __init__(self, args):
        super(PD_HAN, self).__init__()

        self.args = args
        self.dropout = nn.Dropout(args.dropout)  # dropout，防止过拟合
        self.input_dim_MLP = args.gat_hidden_size  # MLP 中输入层的维度 768
        # 使用情感信息
        if self.args.use_senticnet:
            self.input_dim_MLP = args.embedding_dim

        # 帖子交互的 GAT
        self.post_gat = P_GAT(args, num_layers=args.p_gat_num_layers)
        # 用户和帖子交互的 GAT
        self.user_post_gat1 = UP_GAT(args, num_layers=args.up_gat_num_layers)
        self.user_post_gat2 = UP_GAT(args, num_layers=args.up_gat_num_layers)
        self.user_post_gat3 = UP_GAT(args, num_layers=args.up_gat_num_layers)
        self.user_post_gat4 = UP_GAT(args, num_layers=args.up_gat_num_layers)
        # 使用用户人格特质级别的交互
        if self.args.use_user_interaction:
            # 用户人格特质间交互的 GAT
            self.user_gat = U_GAT(args, num_layers=args.u_gat_num_layers)

        # 使用结合注意力机制的 GRU
        if self.args.attention_gru:
            self.input_dim_MLP = self.input_dim_MLP * 2
            self.attention_gru1 = AttentionGRU(args)
            self.attention_gru2 = AttentionGRU(args)
            self.attention_gru3 = AttentionGRU(args)
            self.attention_gru4 = AttentionGRU(args)
            # 初始化交互注意力模块
            self.interaction_attention = InteractionAttention(args)

        # MLP
        # 具有激活函数的全连接层
        layers = [nn.Linear(self.input_dim_MLP, args.final_hidden_size), nn.ReLU()]  # 768 -> 128
        # 通过循环创建额外的具有激活函数的全连接层
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size, args.final_hidden_size), nn.ReLU()]
        # 创建序列模型，用于对图卷积输出进行进一步处理
        self.fcs1 = nn.Sequential(*layers)
        self.fcs2 = nn.Sequential(*layers)
        self.fcs3 = nn.Sequential(*layers)
        self.fcs4 = nn.Sequential(*layers)
        # 将最终处理后的特征映射到输出类别上进行最终的分类
        self.fc_final1 = nn.Linear(args.final_hidden_size, args.num_classes)
        self.fc_final2 = nn.Linear(args.final_hidden_size, args.num_classes)
        self.fc_final3 = nn.Linear(args.final_hidden_size, args.num_classes)
        self.fc_final4 = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, p_mask, feature, c_node):
        # 获取 batch size
        B = p_mask.size(0)
        # 将帖子节点特征和节点注意掩码输入到 P_GAT，获得输出（新的帖子特征）
        new_p_feature = self.post_gat(p_mask, feature)  # (B, N, D)

        # 通过在 p_mask 最后添加一个全 1 的特殊节点，得到新的 new_p_mask  (B, N+1)
        new_p_mask = torch.cat((p_mask, torch.ones(B, 1).cuda()), 1)
        # 将特殊节点特征 c_node 与帖子节点特征进行连接
        combined_feature = torch.cat((new_p_feature, c_node), 1)  # (B, N+1, D)
        # 获得新的用户特征
        new_u_feature1 = self.user_post_gat1(new_p_mask, combined_feature, c_node)  # (B, D)
        new_u_feature2 = self.user_post_gat2(new_p_mask, combined_feature, c_node)
        new_u_feature3 = self.user_post_gat3(new_p_mask, combined_feature, c_node)
        new_u_feature4 = self.user_post_gat4(new_p_mask, combined_feature, c_node)
        # 最终输入到 MLP 的用户人格特征
        final_u_feature1 = new_u_feature1  # (B, D)
        final_u_feature2 = new_u_feature2
        final_u_feature3 = new_u_feature3
        final_u_feature4 = new_u_feature4

        # 使用用户人格特质级别的交互
        if self.args.use_user_interaction:
            # 将用户的四个人格特质特征输入到 U_GAT，获得输出（新的人格特质特征）
            new_u_feature = [new_u_feature1, new_u_feature2, new_u_feature3, new_u_feature4]
            new_u_feature = torch.stack(new_u_feature, dim=1)  # 堆叠特征 (B, 4, D)
            new_new_u_feature = self.user_gat(new_u_feature)  # (B, 4, D)
            # 拆分和重构特征
            slices = torch.split(new_new_u_feature, 1, dim=1)
            new_new_u_feature_list = [s.squeeze(dim=1) for s in slices]
            final_u_feature1 = new_new_u_feature_list[0]  # (B, D)
            final_u_feature2 = new_new_u_feature_list[1]
            final_u_feature3 = new_new_u_feature_list[2]
            final_u_feature4 = new_new_u_feature_list[3]

        # 使用结合注意力机制的 GRU
        if self.args.attention_gru:
            gru_feature1 = self.attention_gru1(p_mask, feature)  # (B, D)
            gru_feature2 = self.attention_gru2(p_mask, feature)
            gru_feature3 = self.attention_gru3(p_mask, feature)
            gru_feature4 = self.attention_gru4(p_mask, feature)
            # 应用交互注意力模块，交互注意力学习人格特质间的潜在关系
            trait_features = torch.cat([gru_feature1.unsqueeze(1), gru_feature2.unsqueeze(1),
                                        gru_feature3.unsqueeze(1), gru_feature4.unsqueeze(1)], dim=1)
            trait_interaction_features = self.interaction_attention(trait_features)
            final_gru_feature1, final_gru_feature2, final_gru_feature3, final_gru_feature4 = torch.unbind(
                trait_interaction_features, dim=1)  # (B, D)
            final_u_feature1 = torch.cat([final_u_feature1, final_gru_feature1], dim=1)  # (B, 2D)
            final_u_feature2 = torch.cat([final_u_feature2, final_gru_feature2], dim=1)  # (B, 2D)
            final_u_feature3 = torch.cat([final_u_feature3, final_gru_feature3], dim=1)  # (B, 2D)
            final_u_feature4 = torch.cat([final_u_feature4, final_gru_feature4], dim=1)  # (B, 2D)

        x1 = self.dropout(final_u_feature1)  # dropout，防止过拟合
        x1 = self.fcs1(x1)  # 进一步做特征变换
        # logit 在分类问题中是用于计算类别概率的原始分数
        logit1 = self.fc_final1(x1)
        x2 = self.dropout(final_u_feature2)
        x2 = self.fcs2(x2)
        logit2 = self.fc_final2(x2)
        x3 = self.dropout(final_u_feature3)
        x3 = self.fcs3(x3)
        logit3 = self.fc_final3(x3)
        x4 = self.dropout(final_u_feature4)
        x4 = self.fcs4(x4)
        logit4 = self.fc_final4(x4)

        return logit1, logit2, logit3, logit4


class P_GAT(nn.Module):
    """
    帖子间交互的图注意力网络
    """

    def __init__(self, args, num_layers):
        super(P_GAT, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.d_model = args.embedding_dim  # 768 788
        self.n_head = args.p_num_heads
        self.max_post = args.max_post  # 最大帖子数
        # 模型的总维度 d_model 被均匀地分割到每个头，每个头可以并行地学习不同的信息
        assert (self.d_model % self.n_head == 0)
        # d_head为每个头分到的维度（n_head * d_head = D）
        self.d_head = int(self.d_model / self.n_head)

        # 定义可训练参数，即论文中的 W 和 a
        self.W = nn.Linear(self.d_model, self.d_model, bias=False)
        self.attn = nn.Linear(self.d_head * 2, self.n_head, bias=False)
        self.actv = nn.Tanh()  # 激活函数
        self.dropout = nn.Dropout(args.gat_dropout)  # dropout，减少过拟合

    def forward(self, p_mask, feature):
        B = p_mask.size(0)
        new_p_mask = p_mask.unsqueeze(-1)  # (B, N, 1)
        # 生成邻接矩阵 (B, N, 1) * (B, 1, N)
        adj = new_p_mask.bmm(new_p_mask.transpose(2, 1))  # (B, N, N)

        for l in range(self.num_layers):
            # 当前层的输入特征被保存为“残差”
            residual = feature  # (B, N, D)
            # 创建一个索引列表，表示每个注意力头的索引
            head_list = list(range(self.n_head))  # [0, 1, 2, 3, 4, 5]

            # 通过一个线性层 self.W 变换特征并重新整形，为每个头产生 Q 和 K (Query 和 Key)：(B, N, D) -> (B, N, n_head, d_head)
            qk = self.W(feature).view(B, self.max_post, self.n_head, self.d_head)
            # mh_q 和 mh_k 是对 qk 的扩展版本：(B, N, N, n_head, d_head) (B, N, N, n_head, d_head)
            mh_q, mh_k = qk[:, :, None, :, :].expand(-1, -1, self.max_post, -1, -1), \
                qk[:, None, :, :, :].expand(-1, self.max_post, -1, -1, -1)
            # 连接 mh_q 和 mh_k 并通过 self.attn 线性层，为每对节点和每个注意力头计算权重
            # (B, N, N, n_head, 2 * d_head) -> (B, N, N, n_head, n_head) -> (B, N, N, n_head)
            mh_attn = self.attn(torch.cat([mh_q, mh_k], dim=-1))[:, :, :, head_list, head_list]  # (B, N, N, n_head)
            # 应用激活函数tanh
            mh_attn = self.actv(mh_attn)  # (B, N, N, n_head)

            # 应用掩码，将邻接矩阵 adj 中为 0 的位置的注意力权重设置为一个非常小的负值（-1e-8）（这样在应用 softmax 时，这些位置的权重将接近于 0）
            mh_attn = mh_attn.masked_fill((1 - adj)[:, :, :, None].expand_as(mh_attn).bool(), -1e-8)
            # 使用 softmax 进行归一化，确保权重之和为 1
            mh_attn = F.softmax(mh_attn, dim=-2)
            mh_attn = self.dropout(mh_attn)  # dropout，防止过拟合

            # mh_attn: (B, N, N, n_head)  qk: (B, N, n_head, d_head)
            # 加权的特征聚合：mh_attn 为 qk 中的每个查询、头和特征维度提供加权系数，这是多头注意力机制的关键组成部分
            # (B, N, n_head, d_head)
            mh_hid = torch.tanh(torch.einsum('bqkn,bknd->bqnd', mh_attn, qk))
            # 输出被重新整形并与残差（原始特征）相加，得到该层的最终输出：(B, N, D)
            feature = residual + mh_hid.reshape(B, self.max_post, -1)

        return feature


class U_GAT(nn.Module):
    """
    用户人格特质间交互的图注意力网络
    """

    def __init__(self, args, num_layers):
        super(U_GAT, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.d_model = args.embedding_dim  # 768 788
        self.n_head = args.u_num_heads
        # 模型的总维度 d_model 被均匀地分割到每个头，每个头可以并行地学习不同的信息
        assert (self.d_model % self.n_head == 0)
        # d_head 为每个头分到的维度（n_head * d_head = D）
        self.d_head = int(self.d_model / self.n_head)
        self.n_user_trait = 4

        # 定义可训练参数，即论文中的 W 和 a
        self.W = nn.Linear(self.d_model, self.d_model, bias=False)
        self.attn = nn.Linear(self.d_head * 2, self.n_head, bias=False)
        self.actv = nn.Tanh()  # 激活函数
        self.dropout = nn.Dropout(args.gat_dropout)  # dropout，减少过拟合

    def forward(self, feature):
        if len(feature.shape) == 2:
            feature = feature.view(-1, self.n_user_trait, self.d_model)
        B = feature.size(0)

        for l in range(self.num_layers):
            # 当前层的输入特征被保存为“残差”
            residual = feature  # (B, N, D)
            # 创建一个索引列表，表示每个注意力头的索引
            head_list = list(range(self.n_head))  # [0, 1, 2, 3, 4, 5]

            # 通过一个线性层 self.W 变换特征并重新整形，为每个头产生 Q 和 K (Query 和 Key)：(B, N, D) -> (B, N, n_head, d_head)
            qk = self.W(feature).view(B, self.n_user_trait, self.n_head, self.d_head)
            # mh_q 和 mh_k 是对qk的扩展版本：(B, N, N, n_head, d_head) (B, N, N, n_head, d_head)
            mh_q, mh_k = qk[:, :, None, :, :].expand(-1, -1, self.n_user_trait, -1, -1), \
                qk[:, None, :, :, :].expand(-1, self.n_user_trait, -1, -1, -1)

            # 连接 mh_q 和 mh_k 并通过 self.attn 线性层，为每对节点和每个注意力头计算权重
            # (B, N, N, n_head, 2 * d_head) -> (B, N, N, n_head, n_head) -> (B, N, N, n_head)
            mh_attn = self.attn(torch.cat([mh_q, mh_k], dim=-1))[:, :, :, head_list, head_list]  # (B, N, N, n_head)
            # 应用激活函数 tanh
            mh_attn = self.actv(mh_attn)  # (B, N, N, n_head)
            # 使用 softmax 进行归一化，确保权重之和为 1
            mh_attn = F.softmax(mh_attn, dim=-2)
            mh_attn = self.dropout(mh_attn)  # dropout，防止过拟合

            # mh_attn: (B, N, N, n_head)  qk: (B, N, n_head, d_head)
            # 加权的特征聚合：mh_attn 为 qk 中的每个查询、头和特征维度提供加权系数，这是多头注意力机制的关键组成部分
            # (B, N, n_head, d_head)
            mh_hid = torch.tanh(torch.einsum('bqkn,bknd->bqnd', mh_attn, qk))
            # 输出被重新整形并与残差（原始特征）相加，并整形，得到该层的最终输出：(B, N, D)
            feature = residual + mh_hid.reshape(B, self.n_user_trait, -1)

        return feature


class UP_GAT(nn.Module):
    """
    用户和帖子交互的图注意力网络
    """

    def __init__(self, args, num_layers):
        super(UP_GAT, self).__init__()
        self.args = args
        self.d_model = args.embedding_dim  # 768 788
        self.num_layers = num_layers
        self.num_heads = args.up_num_heads
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model * self.num_heads, self.d_model)
        self.dropout = nn.Dropout(args.gat_dropout)  # dropout，减少过拟合
        self.actv = nn.Tanh()  # 激活函数

    def forward(self, p_mask, p_feature, u_feature):
        for l in range(self.num_layers):
            # 计算多个头的注意力
            multihead_attn = torch.cat(
                [self.attention_head(p_mask, p_feature, u_feature) for _ in range(self.num_heads)],
                dim=-1)  # (B, 1, D * n_head)
            # 使用权重矩阵将多头注意力结果投影到原始维度
            new_u_feature = self.W_o(multihead_attn)  # (B, 1, D)
        return new_u_feature.squeeze(1)

    def attention_head(self, p_mask, p_feature, u_feature):
        """
        计算一个头的注意力
        """
        similarity_scores = self.calculate_similarity(u_feature, p_feature)  # (B, 1, N)
        # 将掩码中为 0 的位置的相似性得分设置为一个非常小的负值（-1e-8）（这样在应用softmax时，这些位置的权重将接近于0）
        similarity_scores = similarity_scores.masked_fill((1 - p_mask)[:, None, :].bool(), 1e-8)
        # 对相似性得分进行 softmax 操作，以获得注意力系数
        attention_scores = torch.softmax(similarity_scores, dim=2)  # (B, 1, N)

        if self.args.use_filter:
            # 计算每个批次的注意力系数的阈值
            k = int(0.1 * attention_scores.size(2))  # 10% 的位置
            threshold_scores, _ = torch.topk(attention_scores, k, dim=2, largest=False)  # 取最小的k个值
            threshold = threshold_scores[:, :, -1:]  # 取最小的 k 个值中的最大值作为阈值
            # 将符合条件的注意力系数置为 0
            # mask = attention_scores < threshold
            # mask = attention_scores < 0.02
            mask = (attention_scores < 0.02) & (attention_scores < threshold)
            attention_scores = attention_scores.masked_fill(mask, 1e-8)

        attention_scores = self.dropout(attention_scores)  # dropout，防止过拟合
        # 通过注意力系数加权用户和帖子张量，得到用户的表示
        new_u_feature = torch.matmul(attention_scores, p_feature)  # (B, 1, D)
        return new_u_feature

    def calculate_similarity(self, u_feature, p_feature):
        """
        计算相似性
        """
        q = self.W_q(u_feature)  # (B, 1, D)
        k = self.W_k(p_feature)  # (B, N, D)
        similarity_scores = torch.matmul(q, k.transpose(1, 2)) / self.d_model ** 0.5  # (B, 1, N)
        return similarity_scores
