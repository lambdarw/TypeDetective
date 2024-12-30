import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import AlbertModel, AlbertConfig
from transformers import XLNetConfig, XLNetModel
from transformers import RobertaConfig, RobertaModel

from model_utils import MultiGAT, PD_HAN, Multi_AttnGRU


class MyModel(nn.Module):
    """
    PretrainedModel + Multi_GAT
    """

    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.args.embedding_dim = args.d_model  # 768

        # 初始化相应类型的预训练模型（配置信息也从预训练模型的文件中加载）
        if args.pretrain_type == 'bert':
            config = BertConfig.from_pretrained(args.model_dir)
            self.pretrain_models = BertModel.from_pretrained(args.model_dir, config=config, from_tf=False)
        elif args.pretrain_type == 'xlnet':
            config = XLNetConfig.from_pretrained(args.model_dir)
            self.pretrain_models = XLNetModel.from_pretrained(args.model_dir, config=config, from_tf=False)
        elif args.pretrain_type == 'roberta':
            config = RobertaConfig.from_pretrained(args.model_dir)
            self.pretrain_models = RobertaModel.from_pretrained(args.model_dir, config=config, from_tf=False)
        elif args.pretrain_type == 'albert':
            config = AlbertConfig.from_pretrained(args.model_dir)
            self.pretrain_models = AlbertModel.from_pretrained(args.model_dir, config=config, from_tf=False)
        else:
            raise NotImplementedError

        # 多 GAT
        self.multi_gat = MultiGAT(args)

    def forward(self, post_tokens_id, polarities):
        if self.args.pretrain_type == 'bert':
            pad_id = 0
        elif self.args.pretrain_type == 'xlnet':
            pad_id = 5
        elif self.args.pretrain_type == 'roberta':
            pad_id = 1
        elif self.args.pretrain_type == 'albert':
            pad_id = 0
        else:
            raise NotImplementedError

        # 创建 attention mask，将不是填充符号的位置设为 1，填充符号的位置设为 0
        a_mask = (post_tokens_id != pad_id).float()  # (B, N, L)
        # 创建 post mask，判断每个样本是否包含有效的 tokens，将包含有效 tokens 的位置设为 1，否则为 0
        p_mask = (a_mask.sum(-1) > 0).float()  # (B, N)
        # 将 post_tokens_id (B, N, L) 重塑为二维张量，方便后续处理
        input_ids = post_tokens_id.view(-1, self.args.max_len)  # (B*N, L)
        # 将 attention mask (B, N, L) 也重塑为二维张量，方便后续处理
        att_mask = a_mask.view(-1, self.args.max_len)  # (B*N, L)

        # 使用预训练模型处理数据
        # [0] 取返回元组的第一个元素，[:, :1]进行切片，将每个样本的第一个 token（通常是 [CLS] 标记，包含整个序列的汇总信息）提取出来。
        cls_token = self.pretrain_models(input_ids=input_ids, attention_mask=att_mask)[0][:, :1]
        # 将切片的结果重新变形为一个三维张量，形状为 (B, N, D)，其中 B 是批大小，N 是最大帖子数，D 是表示的维度（768）
        # 这个结果表示了每个样本中 [CLS] 标记的表示，可以用作后续模型的输入
        cls_token = cls_token.view(-1, self.args.max_post, self.args.embedding_dim)  # (B, N, D)

        # 根据所有帖子节点表示计算特殊节点表示：
        #   masked_fill 函数将 cls_token 中对应位置的无效节点（由 p_mask 标识）替换为 0  (B, N, D)
        #   sum(dim=1) 对每个样本，对有效节点的表示进行求和  (B, D)
        #   将求和后的 cls_token 除以 p_mask 的求和结果，表示每个批次中的文本序列的平均表示  (B, D)
        #   最后使用 unsqueeze(1) 将结果的维度扩展到 (B, 1, D)
        c_node = (cls_token.masked_fill((1 - p_mask[:, :, None].expand_as(cls_token)).bool(), 0).sum(dim=1) /
                  p_mask.sum(dim=-1)[:, None].expand(-1, self.args.embedding_dim)).unsqueeze(1)  # (B, 1, D)

        # p_mask (B, N) 判断每个样本是否包含有效的 tokens
        # cls_token (B, N, D) 包含所有帖子的上下文表示
        # c_node (B, 1, D) 为用户节点的初始化表示
        logit1, logit2, logit3, logit4 = self.multi_gat(p_mask, cls_token, c_node)

        return logit1, logit2, logit3, logit4


class MyModel2(nn.Module):
    """
    PretrainedModel + SenticNet + Multi_GRU + PD_HAN
    """

    def __init__(self, args):
        super(MyModel2, self).__init__()
        self.args = args
        self.embedding_dim = args.d_model
        # 使用情感信息
        if self.args.use_senticnet:
            self.embedding_dim = args.embedding_dim

        # 初始化相应类型的预训练模型（配置信息也从预训练模型的文件中加载）
        if args.pretrain_type == 'bert':
            config = BertConfig.from_pretrained(args.model_dir)
            self.pretrain_models = BertModel.from_pretrained(args.model_dir, config=config, from_tf=False)
        elif args.pretrain_type == 'xlnet':
            config = XLNetConfig.from_pretrained(args.model_dir)
            self.pretrain_models = XLNetModel.from_pretrained(args.model_dir, config=config, from_tf=False)
        elif args.pretrain_type == 'roberta':
            config = RobertaConfig.from_pretrained(args.model_dir)
            self.pretrain_models = RobertaModel.from_pretrained(args.model_dir, config=config, from_tf=False)
        elif args.pretrain_type == 'albert':
            config = AlbertConfig.from_pretrained(args.model_dir)
            self.pretrain_models = AlbertModel.from_pretrained(args.model_dir, config=config, from_tf=False)
        else:
            raise NotImplementedError

        # PD_HAN
        self.pd_han = PD_HAN(args)

    def forward(self, post_tokens_id, polarities):
        if self.args.pretrain_type == 'bert':
            pad_id = 0
        elif self.args.pretrain_type == 'xlnet':
            pad_id = 5
        elif self.args.pretrain_type == 'roberta':
            pad_id = 1
        elif self.args.pretrain_type == 'albert':
            pad_id = 0
        else:
            raise NotImplementedError

        # 创建 attention mask，将不是填充符号的位置设为 1，填充符号的位置设为 0
        a_mask = (post_tokens_id != pad_id).float()  # (B, N, L)
        # 创建 post mask，判断每个样本是否包含有效的 tokens，将包含有效 tokens 的位置设为 1，否则为 0
        p_mask = (a_mask.sum(-1) > 0).float()  # (B, N)
        # 将 post_tokens_id (B, N, L) 重塑为二维张量，方便后续处理
        input_ids = post_tokens_id.view(-1, self.args.max_len)  # (B*N, L)
        # 将 attention mask (B, N, L) 也重塑为二维张量，方便后续处理
        att_mask = a_mask.view(-1, self.args.max_len)  # (B*N, L)

        # 使用预训练模型处理数据
        # [0] 取返回元组的第一个元素，[:, :1]进行切片，将每个样本的第一个 token（通常是 [CLS] 标记，包含整个序列的汇总信息）提取出来
        cls_token = self.pretrain_models(input_ids=input_ids, attention_mask=att_mask)[0][:, :1]
        # 将切片的结果重新变形为一个三维张量，形状为 (B, N, D)，其中 B 是批大小，N 是最大帖子数，D 是表示的维度（768）
        # 这个结果表示了每个样本中 [CLS] 标记的表示，可以用作后续模型的输入
        cls_token = cls_token.view(-1, self.args.max_post, self.args.d_model)  # (B, N, D)

        # 使用情感信息
        if self.args.use_senticnet:
            cls_token = torch.cat((cls_token, polarities), dim=2)

        # 根据所有帖子节点表示计算特殊节点表示：
        #   masked_fill 函数将 cls_token 中对应位置的无效节点（由 p_mask 标识）替换为 0  (B, N, D)
        #   sum(dim=1) 对每个样本，对有效节点的表示进行求和  (B, D)
        #   将求和后的 cls_token 除以 p_mask 的求和结果，表示每个批次中的文本序列的平均表示  (B, D)
        #   最后使用 unsqueeze(1) 将结果的维度扩展到 (B, 1, D)
        c_node = (cls_token.masked_fill((1 - p_mask[:, :, None].expand_as(cls_token)).bool(), 0).sum(dim=1) /
                  p_mask.sum(dim=-1)[:, None].expand(-1, self.embedding_dim)).unsqueeze(1)  # (B, 1, D)

        # p_mask (B, N) 判断每个样本是否包含有效的 tokens
        # cls_token (B, N, D) 包含所有帖子的上下文表示
        # c_node (B, 1, D) 为用户节点的初始化表示
        logit1, logit2, logit3, logit4 = self.pd_han(p_mask, cls_token, c_node)

        return logit1, logit2, logit3, logit4


class MyModel3(nn.Module):
    """
    PretrainedModel + SenticNet + Multi_GRU
    """

    def __init__(self, args):
        super(MyModel3, self).__init__()
        self.args = args
        self.embedding_dim = args.d_model
        # 使用情感信息
        if self.args.use_senticnet:
            self.embedding_dim = args.embedding_dim

        # 初始化相应类型的预训练模型（配置信息也从预训练模型的文件中加载）
        if args.pretrain_type == 'bert':
            config = BertConfig.from_pretrained(args.model_dir)
            self.pretrain_models = BertModel.from_pretrained(args.model_dir, config=config, from_tf=False)
        elif args.pretrain_type == 'xlnet':
            config = XLNetConfig.from_pretrained(args.model_dir)
            self.pretrain_models = XLNetModel.from_pretrained(args.model_dir, config=config, from_tf=False)
        elif args.pretrain_type == 'roberta':
            config = RobertaConfig.from_pretrained(args.model_dir)
            self.pretrain_models = RobertaModel.from_pretrained(args.model_dir, config=config, from_tf=False)
        elif args.pretrain_type == 'albert':
            config = AlbertConfig.from_pretrained(args.model_dir)
            self.pretrain_models = AlbertModel.from_pretrained(args.model_dir, config=config, from_tf=False)
        else:
            raise NotImplementedError

        # Multi_AttnGRU
        self.multi_attn_gru = Multi_AttnGRU(args)

    def forward(self, post_tokens_id, polarities):
        if self.args.pretrain_type == 'bert':
            pad_id = 0
        elif self.args.pretrain_type == 'xlnet':
            pad_id = 5
        elif self.args.pretrain_type == 'roberta':
            pad_id = 1
        elif self.args.pretrain_type == 'albert':
            pad_id = 0
        else:
            raise NotImplementedError

        # 创建 attention mask，将不是填充符号的位置设为 1，填充符号的位置设为 0
        a_mask = (post_tokens_id != pad_id).float()  # (B, N, L)
        # 创建 post mask，判断每个样本是否包含有效的 tokens，将包含有效 tokens 的位置设为 1，否则为 0
        p_mask = (a_mask.sum(-1) > 0).float()  # (B, N)
        # 将 post_tokens_id (B, N, L) 重塑为二维张量，方便后续处理
        input_ids = post_tokens_id.view(-1, self.args.max_len)  # (B*N, L)
        # 将 attention mask (B, N, L) 也重塑为二维张量，方便后续处理
        att_mask = a_mask.view(-1, self.args.max_len)  # (B*N, L)

        # 使用预训练模型处理数据
        # [0] 取返回元组的第一个元素，[:, :1]进行切片，将每个样本的第一个 token（通常是 [CLS] 标记，包含整个序列的汇总信息）提取出来
        cls_token = self.pretrain_models(input_ids=input_ids, attention_mask=att_mask)[0][:, :1]
        # 将切片的结果重新变形为一个三维张量，形状为 (B, N, D)，其中 B 是批大小，N 是最大帖子数，D 是表示的维度（768）
        # 这个结果表示了每个样本中 [CLS] 标记的表示，可以用作后续模型的输入
        cls_token = cls_token.view(-1, self.args.max_post, self.args.d_model)  # (B, N, D)

        # 使用情感信息
        if self.args.use_senticnet:
            cls_token = torch.cat((cls_token, polarities), dim=2)

        # p_mask (B, N) 判断每个样本是否包含有效的 tokens
        # cls_token (B, N, D) 包含所有帖子的上下文表示
        logit1, logit2, logit3, logit4 = self.multi_attn_gru(p_mask, cls_token)

        return logit1, logit2, logit3, logit4
