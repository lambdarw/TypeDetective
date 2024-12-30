import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from transformers import BertTokenizer
from transformers import AlbertTokenizer
from transformers import XLNetTokenizer
from transformers import RobertaTokenizer
from collections import OrderedDict

from datasets import load_split_datasets
from model import MyModel, MyModel2, MyModel3
from trainer import train, get_labels
# from contrast_model import BERT_Mean, BERT_CNN, BERT_Concat, SN_Atnn, BGRU

logger = logging.getLogger(__name__)


def set_seed(args):
    """
    设置随机种子，以确保在同样的种子下每次运行程序得到的随机结果是一致的。
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def check_args(args):
    """
    检查参数
    """
    logger.info(vars(args))


def parse_args():
    """
    命令行参数解析函数
    :return: 将命令行中的参数解析为一个 Python 字典，并返回该字典
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--option', type=str, default='train', choices=['train', 'test'],
                        help='train or test the model')
    parser.add_argument('--continue_train', action='store_true', default=False)
    parser.add_argument('--task', type=str, default='kaggle',
                        choices=['kaggle', 'kaggle_test', 'pandora', 'pandora_test'],
                        help='task name')
    parser.add_argument('--output_dir', type=str, default='output/kaggle/0223',
                        help='directory to store intermedia data, such as models, records')
    parser.add_argument('--pth_name', type=str, default='best_f1_pdhan_kaggle.pth')
    parser.add_argument('--max_post', type=int, default=50, choices=[50, 100],
                        help='Number of max post. decide by task')
    parser.add_argument('--seed', type=int, default=321,
                        help='random seed for initialization')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes of each personality dimension')
    parser.add_argument('--max_len', type=int, default=70,
                        help='number of max len')

    # Model parameters
    parser.add_argument('--model_dir', type=str,
                        default='/nfs/huggingfacehub/models--bert-base-cased',
                        # default='/nfs/huggingfacehub/models--roberta-base',
                        # default='/nfs/huggingfacehub/models--albert-base-v2',
                        # default='/nfs/huggingfacehub/models--xlnet-base-cased',
                        # default='/nfs/huggingfacehub/roberta-large',
                        # default='/mntc/huangxinhui/pretrained_models/models--bert-base-cased',
                        # default='/home/huangxinhui/PersonalityDetection/data/MLM-data-preprocessing/roberta-model-kaggle',
                        # default='/root/a_huangxinhui/pretrained_models/models--bert-base-cased',
                        help='path to pre-trained model')
    parser.add_argument('--pretrain_type', type=str, default='bert',
                        choices=['bert', 'xlnet', 'roberta', 'albert', 'roberta-large'])
    parser.add_argument('--use_roberta_large', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.2,  # 0.2
                        help='dropout rate for embedding')
    parser.add_argument('--d_model', type=int, default=768,
                        help='model dimension of Bert or other pretrained language models')
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='number of mlps in the last of model')
    parser.add_argument('--final_hidden_size', type=int, default=128,
                        help='hidden size of attention')
    # GAT
    parser.add_argument('--gat_hidden_size', type=int, default=768,
                        help='hidden_size for gat')
    parser.add_argument('--gat_dropout', type=float, default=0.2,  # 0.2
                        help='dropout rate for gat')
    parser.add_argument('--gat_num_layers', type=int, default=2)
    parser.add_argument('--p_gat_num_layers', type=int, default=2)
    parser.add_argument('--u_gat_num_layers', type=int, default=2)
    parser.add_argument('--up_gat_num_layers', type=int, default=2)
    parser.add_argument('--p_num_heads', type=int, default=4)
    parser.add_argument('--u_num_heads', type=int, default=4)
    parser.add_argument('--up_num_heads', type=int, default=4)

    # Training parameters
    parser.add_argument('--all_gpu_train_batch_size', type=int, default=8,  # 8 4 2
                        help='batch size per GPU/CPU for training')
    parser.add_argument('--all_gpu_eval_batch_size', type=int, default=32,  # 32 16 8
                        help='batch size per GPU/CPU for evaluation')
    parser.add_argument('--num_train_epochs', type=float, default=10.0,  # 30.0 25.0
                        help='total number of training epochs to perform')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument("--max_steps", type=int, default=-1,
                        help='If > 0: set total number of training steps(that update the weights) to perform.'
                             'Override num_train_epochs.')
    parser.add_argument('--logging_steps', type=int, default=25,  # 25
                        help='log every X updates steps')
    # Learning rates
    parser.add_argument('--plm_learning_rate', type=float, default=1e-5,  # 2e-5
                        help='the initial learning rate for pretrain language models')
    parser.add_argument('--gm_learning_rate', default=1e-5, type=float,
                        help='graph network learning rate')
    parser.add_argument('--gru_learning_rate', default=1e-5, type=float,
                        help='gru network learning rate')
    parser.add_argument('--other_learning_rate', type=float, default=1e-4,  # 1e-4
                        help='the initial learning rate for other components')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6,
                        help='epsilon for Adam optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='max gradient norm')
    # parser.add_argument('--weight_decay', type=float, default=1e-5,
    #                     help='weight decay if we apply some')

    # Ablation experiments parameters
    parser.add_argument('--no_special_node', action='store_true', default=False,
                        help='remove-special node variant for ablation experiments')
    parser.add_argument('--no_dart', action='store_true', default=True,
                        help='post-training')
    parser.add_argument('--use_user_interaction', action='store_true', default=True)
    parser.add_argument('--attention_gru', action='store_true', default=True)
    parser.add_argument('--use_filter', action='store_true', default=False)
    parser.add_argument('--use_senticnet', action='store_true', default=True)
    parser.add_argument('--senticnet_dim', type=int, default=20)
    parser.add_argument('--use_data_augmentation', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    # 解析参数
    args = parse_args()
    if args.task == 'kaggle':
        args.max_post = 50
    elif args.task == 'pandora':
        args.max_post = 100
        args.pth_name = 'best_f1_pdhan_pandora.pth'
    elif args.task == 'kaggle_test':
        args.max_post = 50
        args.logging_steps = 1
    elif args.task == 'pandora_test':
        args.max_post = 100
        args.logging_steps = 1
        args.pth_name = 'best_f1_pdhan_pandora.pth'
    # 使用情感知识
    if args.use_senticnet:
        args.embedding_dim = args.d_model + args.senticnet_dim
    # 使用 roberta-large
    if args.use_roberta_large:
        args.d_model = 1024
        args.embedding_dim = 1044
    check_args(args)

    # 日志设置
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    # 创建输出文件夹
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # 设备设置
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用 GPU 0 和 GPU 1
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    # 设置随机种子
    set_seed(args)

    # 加载预训练模型和相应的分词器
    if args.pretrain_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_dir)
        args.tokenizer = tokenizer
    elif args.pretrain_type == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained(args.model_dir)
        args.tokenizer = tokenizer
    elif args.pretrain_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_dir)
        args.tokenizer = tokenizer
    elif args.pretrain_type == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(args.model_dir)
        args.tokenizer = tokenizer
    else:
        raise NotImplementedError

    if args.option == 'train':
        # 加载数据集
        train_dataset, eval_dataset, test_dataset = load_split_datasets(args)
        # 建立模型
        model = MyModel2(args)

        # 加载后训练的 BERT 模型参数，并将它们映射到当前模型的相应位置，从而初始化模型的权重
        if not args.no_dart:
            # 定义一个前缀 name_ptm，用于将原始预训练模型参数中的键值映射到当前模型中的对应键值
            name_ptm = 'pretrain_models.'
            # 加载预训练的BERT模型参数。
            original_msd = torch.load(os.path.join('bert-pretrained', 'bert-pretrained.pth'), map_location='cpu')
            # 创建一个空字典，用于存储仅与BERT相关的模型参数
            only_bert_msd = {}
            # 遍历预训练模型参数字典的键值对。
            for key, values in original_msd.items():
                # 将预训练参数字典中的键 key 修改为当前模型中的对应键，然后将值 values 存储到新的字典 only_bert_msd 中
                if 'module.ptm.bert.' in key:
                    only_bert_msd[key.replace('module.ptm.bert.', name_ptm)] = values
            # 获取模型的所有参数。
            total_msd = model.state_dict()
            # 将仅与 BERT 相关的参数从字典 only_bert_msd 更新到当前模型的参数字典 total_msd 中
            total_msd.update(only_bert_msd)
            # 使用更新后的参数字典 total_msd 加载模型的权重（参数）
            model.load_state_dict(total_msd)

        # 进行多 GPU 并行处理
        model = nn.DataParallel(model)
        # 将模型移到指定的计算设备上
        model.to(args.device)

        # 训练模型
        global_step, ave_training_loss, all_eval_results = train(args, model, train_dataset, eval_dataset, test_dataset)

    elif args.option == 'test':
        # 加载数据集
        train_dataset, eval_dataset, test_dataset = load_split_datasets(args)
        # 建立模型
        model = MyModel2(args)
        # 单 GPU 测试
        model.to(args.device)

        # 从指定路径加载之前保存的模型权重
        model_state_dict = torch.load(os.path.join(args.output_dir, 'best_f1_pdhan_kaggle.pth'))
        # 创建一个新的有序字典，用于存储处理后的模型权重
        new_state_dict = OrderedDict()
        # 遍历之前加载的模型权重字典中的键值对
        for k, v in model_state_dict.items():
            # 移除了键中的 'module' 前缀（通常在多 GPU 训练时出现）
            name = k[7:]
            # 将处理后的权重数据存储在新的有序字典中
            new_state_dict[name] = v
        # 将处理后的权重字典加载到当前的模型中，以完成权重的恢复
        model.load_state_dict(new_state_dict)
        # 获取预测标签和真实标签
        preds1, preds2, preds3, preds4, out_label_ids1, out_label_ids2, out_label_ids3, out_label_ids4, ave_f1 = get_labels(
            args, model, test_dataset)

        print('*' * 40)
        print('ave_f1: ', ave_f1)
        print('*' * 40)

        # 创建 Pandas DataFrame，用于存储预测标签和真实标签
        save_data = pd.DataFrame(out_label_ids1, columns=['T1'])
        save_data['T2'] = out_label_ids2
        save_data['T3'] = out_label_ids3
        save_data['T4'] = out_label_ids4
        # save_data['P1'] = preds1
        # save_data['P2'] = preds2
        # save_data['P3'] = preds3
        # save_data['P4'] = preds4
        # 将内容保存到 Excel 文件
        save_data.to_excel('../output/test_kaggle.xlsx')

    else:
        raise NotImplementedError
