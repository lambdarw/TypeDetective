import logging
import os
import random
import math

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import trange

from datasets import my_collate
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR

logger = logging.getLogger(__name__)


def train(args, model, train_dataset, eval_dataset, test_dataset):
    """
    训练模型
    """

    # 创建一个 TensorBoard 的 SummaryWriter，用于记录训练过程中的各种信息
    tb_writer = SummaryWriter()
    # 将参数信息写入 TensorBoard
    write_tb_writer(tb_writer, args)

    print('=' * 40 + 'training' + '=' * 40)

    # 将训练批次大小设置为多个GPU的训练批次大小（在多GPU并行训练时保持相同的训练批次大小）
    args.train_batch_size = args.all_gpu_train_batch_size
    # 创建一个随机采样器（从训练数据集中随机采样训练样本）
    train_sampler = RandomSampler(train_dataset)
    # 获取一个数据加载器的合并函数（将一个批次的样本组织成一个批次的 PyTorch 张量）
    collate_fn = get_collate_fn()
    # 创建一个训练数据加载器（使用随机采样器对训练数据集进行采样，并使用指定的批次大小和合并函数）
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps  # 训练的总步数（模型更新权重的总次数）
        # 计算在预设的最大步数内可以进行的训练轮数。因为可能会有部分步骤未被完全利用，所以会在总轮数上再加 1
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        # len(train_dataloader)：训练数据加载器中的批次数量；args.gradient_accumulation_steps：梯度累积的步数；args.num_train_epochs：训练的总轮数。
        # 计算需要训练的总步数 t_total（训练将进行的总迭代次数）
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # 根据传入的参数和模型来获取优化器对象
    optimizer = get_optimizer(args, model, 'pd_han')
    # optimizer = get_optimizer(args, model, 'multi_attn_gru')
    # optimizer = get_optimizer(args, model, 'gat')
    # optimizer = get_optimizer2(args, model)
    # 学习率调度器
    # scheduler = ExponentialLR(optimizer, gamma=0.95)
    # 如果在3个epoch内验证损失没有下降，学习率乘以0.1
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    logger.info('=' * 40 + "Running training" + '=' * 40)
    logger.info("  Num examples = %d", len(train_dataset))  # 训练集中样本总数
    logger.info("  Num Epochs = %d", args.num_train_epochs)  # 总训练轮数
    logger.info("  Instantaneous batch size per GPU = %d", args.all_gpu_train_batch_size)  # 每个GPU上即时批量大小
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)  # 梯度累积步数
    logger.info("  Total optimization steps = %d", t_total)  # 总的优化步数（训练的总步数）

    global_step = 0  # 全局训练步数
    training_loss, logging_loss = 0.0, 0.0  # 训练过程中的累计损失；用于日志的损失
    best_eval_ave_f1 = 0.0  # 到目前为止在验证集上的最佳平均 F1 分数
    best_test_ave_f1 = 0.0  # 到目前为止在测试集上的最佳平均 F1 分数
    best_eval_epoch = 0  # 获得最佳验证集结果的训练迭代次数
    best_eval_results = None  # 最佳验证集上的评价结果
    best_test_results = None  # 最佳测试集上的评价结果
    all_eval_results = []  # 验证集评估结果列表
    all_test_results = []  # 测试集评估结果列表
    model.zero_grad()  # 梯度清零，为新的训练迭代做准备
    # 使用 tqdm.trange 创建一个迭代器，用于遍历每个训练 Epoch
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # 设置随机种子

    '''
    遍历训练的epochs  Start
    '''
    for train_epoch, _ in enumerate(train_iterator):
        preds1, preds2, preds3, preds4 = None, None, None, None  # 模型预测的类别
        out_labels1, out_labels2, out_labels3, out_labels4 = None, None, None, None  # 真实的类别标签
        results1, results2, results3, results4 = {}, {}, {}, {}  # 训练结果
        step_loss = 0.0  # 当前 epoch 内的损失总和
        n_steps = 0  # 当前 epoch 内已经执行的步数

        '''
        遍历训练数据集中的批次  Start
        '''
        # step: 当前批次的索引；batch: 当前批次的数据
        for step, batch in enumerate(train_dataloader):
            n_steps += 1  # 当前 epoch 内已经执行的步数+1
            model.train()  # 将模型设置为训练模式
            batch = tuple(t.to(args.device) for t in batch)  # 将批次中的每个张量移动到 GPU 上

            # 从批次数据中提取输入和标签信息作为模型的输入
            inputs, label1, label2, label3, label4, polarities = get_input_from_batch(batch)
            # 将输入信息传递给模型，得到四个任务的预测结果
            logit1, logit2, logit3, logit4 = model(**inputs, **polarities)
            # 记录每个训练批次的预测结果和真实标签
            preds1, preds2, preds3, preds4, out_labels1, out_labels2, out_labels3, out_labels4 = record_predictions_and_labels(
                logit1, logit2, logit3, logit4, label1, label2, label3, label4,
                preds1, preds2, preds3, preds4, out_labels1, out_labels2, out_labels3, out_labels4)

            # 根据预测值和标签值计算损失
            batch_ave_loss = compute_batch_ave_loss(logit1, logit2, logit3, logit4, label1, label2, label3, label4)
            # step_loss 用于跟踪当前 epoch 内所有批次的总损失
            step_loss += batch_ave_loss.item()

            # 检查损失是否为 NaN（Not a Number），NaN 可能会在训练过程中出现，通常表示出现了数值不稳定的情况
            if math.isnan(batch_ave_loss.item()):
                # 如果损失值为 NaN，代码使用 pdb.set_trace() 调用进入调试模式，以便开发者调查问题
                import pdb
                pdb.set_trace()

            # 梯度累积：将多个小批次的梯度累积起来，然后一次性进行参数更新（这对于内存受限或训练过程中需要更大批次大小的情况很有用）
            if args.gradient_accumulation_steps > 1:
                # 平均每个累积的损失，确保参数更新的稳定性
                batch_ave_loss = batch_ave_loss / args.gradient_accumulation_steps

            # 计算损失对模型参数的梯度
            batch_ave_loss.backward()
            # 执行梯度裁剪，以防止梯度爆炸问题（args.max_grad_norm 是裁剪梯度的阈值）
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 累计训练过程中的损失
            training_loss += batch_ave_loss.item()

            '''
            在每经过一定数量的梯度累积步数后执行一些操作  Start
            '''
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # 更新模型的参数
                model.zero_grad()  # 将梯度清零，为下一次梯度计算做准备
                global_step += 1  # 增加全局步数计数器，跟踪训练的总步数

                # 记录指标
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # 进行验证操作，得到验证结果
                    eval_results, eval_loss, eval_ave_f1 = evaluate_or_test(args, model, eval_dataset, 'evaluate')
                    # 如果当前验证结果的平均 F1 分数更好
                    if eval_ave_f1 > best_eval_ave_f1:
                        # Save model checkpoint 保存模型的状态字典（权重参数）到指定的文件路径（/output/~.pth）
                        torch.save(model.state_dict(), os.path.join(args.output_dir, args.pth_name))
                        best_eval_epoch = train_epoch  # 记录最佳平均 F1 分数时的迭代次数（验证集）
                        best_eval_ave_f1 = eval_ave_f1  # 记录最佳平均 F1 分数（验证集）
                        best_eval_results = eval_results  # 记录最佳平均 F1 分数时的其他指标结果（验证集）
                        # 进行测试操作，得到测试结果
                        test_results, test_loss, test_ave_f1 = evaluate_or_test(args, model, test_dataset, 'test')
                        best_test_ave_f1 = test_ave_f1  # 记录最佳平均 F1 分数（测试集）
                        best_test_results = test_results  # 记录最佳平均 F1 分数时的其他指标结果（测试集）

                        # 将当前测试结果追加到所有测试结果列表中
                        all_test_results.append(test_results)
                        # 将测试结果写入 TensorBoard 日志
                        tensorboard_results(tb_writer, test_results[0], test_results[1],
                                            test_results[2], test_results[3], global_step, 'test')
                        tb_writer.add_scalar('test_loss', test_loss, global_step)
                        tb_writer.add_scalar('test_ave_f1', test_ave_f1, global_step)

                    # 将验证结果追加到所有验证结果列表中
                    all_eval_results.append(eval_results)
                    # 将验证结果写入 TensorBoard 日志
                    tensorboard_results(tb_writer, eval_results[0], eval_results[1],
                                        eval_results[2], eval_results[3], global_step, 'evaluate')
                    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    tb_writer.add_scalar('eval_ave_f1', eval_ave_f1, global_step)

                    # 每个日志步骤的平均损失
                    tb_writer.add_scalar('train_loss', (training_loss - logging_loss) / args.logging_steps, global_step)
                    # 当前的训练损失赋值给日志损失，以便在下一个日志步骤计算平均损失
                    logging_loss = training_loss
            '''
            在每经过一定数量的梯度累积步数后执行一些操作  End
            '''
        '''
        遍历训练数据集中的批次  End
        '''

        # scheduler.step()  # 更新优化器的学习率

        # 根据预测值和标签值计算最终结果
        result1, result2, result3, result4 = compute_final_results(preds1, preds2, preds3, preds4,
                                                                   out_labels1, out_labels2, out_labels3, out_labels4)
        # 将 result 字典中的键值对更新到 results 字典中
        results1.update(result1)
        results2.update(result2)
        results3.update(result3)
        results4.update(result4)
        # 将训练结果写入 TensorBoard 日志
        tensorboard_results(tb_writer, results1, results2, results3, results4, global_step, 'train')

        logger.info('=' * 40 + "Results" + '=' * 40)
        # 设置训练结果文件的路径（/output/train_results.txt）
        output_eval_file = os.path.join(args.output_dir, 'train_results.txt')
        # 设置打开文件的模式（文件不存在：写入模式；文件已存在：追加写入模式）
        file_mode = 'w' if not os.path.exists(output_eval_file) else 'a+'
        # 将训练结果写入训练结果文件
        with open(output_eval_file, file_mode) as writer:
            logger.info('*' * 20 + ' Train results ' + '*' * 20)
            logger.info("current_train_epoch= %s", str(train_epoch))
            logger.info("train loss: %s", str(step_loss / (n_steps)))
            writer.write(("current_train_epoch= %s\n" % str(train_epoch)))
            # 记录四个维度人格检测的结果
            process_results(writer, results1, results2, results3, results4)

        # 将最好验证结果写入文件（/output/best_eval_results.txt）
        best_record_file = os.path.join(args.output_dir, 'best_eval_results.txt')
        with open(best_record_file, file_mode) as writer:
            logger.info('*' * 20 + ' Best results ' + '*' * 20)
            logger.info("current_train_epoch = %s", str(train_epoch))
            logger.info("best_eval_epoch = %s", str(best_eval_epoch))
            logger.info("best_eval_ave_f1 = %s", str(best_eval_ave_f1))
            writer.write(("current_train_epoch = %s\n" % str(train_epoch)))
            writer.write(("best_eval_epoch = %s\n" % str(best_eval_epoch)))
            writer.write(("best_eval_ave_f1 = %s\n" % str(best_eval_ave_f1)))
            # 记录四个维度人格检测的结果
            process_results(writer, best_eval_results[0], best_eval_results[1],
                            best_eval_results[2], best_eval_results[3])

        # 将最好测试结果写入文件（/output/best_test_results.txt）
        best_test_record_file = os.path.join(args.output_dir, 'best_test_results.txt')
        with open(best_test_record_file, file_mode) as writer:
            logger.info('*' * 20 + ' Test results ' + '*' * 20)
            logger.info("current_train_epoch = %s", str(train_epoch))
            logger.info("test_ave_f1 = %s", str(best_test_ave_f1))
            writer.write(("current_train_epoch = %s\n" % str(train_epoch)))
            writer.write(("test_ave_f1 = %s\n" % str(best_test_ave_f1)))
            # 记录四个维度人格检测的结果
            process_results(writer, best_test_results[0], best_test_results[1],
                            best_test_results[2], best_test_results[3])
    '''
    遍历训练的epochs  End
    '''

    tb_writer.close()  # 关闭 TensorBoard 写入器

    # 返回训练总步数、平均训练损失、所有验证结果的列表
    return global_step, training_loss / global_step, all_eval_results


def evaluate_or_test(args, model, dataset, choice):
    """
    验证或测试模型
    """

    eval_or_test_batch_size = args.all_gpu_eval_batch_size  # 验证/测试时的批大小，与训练时不同
    # 顺序采样器（按顺序采样验证数据集中的样本）
    eval_or_test_sampler = SequentialSampler(dataset)
    # 用于将样本组织成批次
    collate_fn = get_collate_fn()
    # 数据加载器（根据给定的采样器、批大小和数据组织函数，加载数据集中的数据并组织成批次）
    eval_or_test_dataloader = DataLoader(dataset,
                                         sampler=eval_or_test_sampler,
                                         batch_size=eval_or_test_batch_size,
                                         collate_fn=collate_fn)

    if choice == 'evaluate':
        logger.info('=' * 40 + "Running evaluate" + '=' * 40)
    if choice == 'test':
        logger.info('=' * 40 + "Running test" + '=' * 40)
    logger.info('Num examples = %d', len(dataset))
    logger.info('Batch size = %d', eval_or_test_batch_size)

    preds1, preds2, preds3, preds4 = None, None, None, None  # 模型预测的类别 [1 1 1 1]
    out_labels1, out_labels2, out_labels3, out_labels4 = None, None, None, None  # 真实的类别标签 [0 0 1 0]
    results1, results2, results3, results4 = {}, {}, {}, {}  # 验证结果
    eval_or_test_loss = 0.0  # 用于累积验证过程中的损失值
    n_eval_or_test_steps = 0  # 用于计数验证/测试过程中的步数

    # 遍历数据集中的每个批次
    for batch in eval_or_test_dataloader:
        model.eval()  # 将模型设置为验证模式
        batch = tuple(t.to(args.device) for t in batch)  # 将当前批次中的所有张量移动到GPU上

        with torch.no_grad():  # 不需要计算梯度
            # 从当前批次中获取输入数据和标签 tensor([0,0,1,1])
            inputs, label1, label2, label3, label4, polarities = get_input_from_batch(batch)
            # 将输入数据传递给模型，得到四个任务的预测 tensor([[0.2,0.8],[0.2,0.8],[0.2,0.8],[0.2,0.8]])
            logit1, logit2, logit3, logit4 = model(**inputs, **polarities)
            # 根据预测值和标签值计算损失
            batch_ave_loss = compute_batch_ave_loss(logit1, logit2, logit3, logit4, label1, label2, label3, label4)
            # 将当前批次的验证损失值添加到总的验证损失中
            eval_or_test_loss += batch_ave_loss.item()

        # 记录每个批次的预测结果和真实标签
        preds1, preds2, preds3, preds4, out_labels1, out_labels2, out_labels3, out_labels4 = record_predictions_and_labels(
            logit1, logit2, logit3, logit4, label1, label2, label3, label4,
            preds1, preds2, preds3, preds4, out_labels1, out_labels2, out_labels3, out_labels4)

    # 根据预测值和标签值计算最终结果
    result1, result2, result3, result4 = compute_final_results(preds1, preds2, preds3, preds4,
                                                               out_labels1, out_labels2, out_labels3, out_labels4)
    # 计算所有任务的 F1 分数的平均值
    ave_f1 = (result1['f1'] + result2['f1'] + result3['f1'] + result4['f1']) / 4.0
    # 将 result 字典中的键值对更新到 results 字典中
    results1.update(result1)
    results2.update(result2)
    results3.update(result3)
    results4.update(result4)

    # 计数验证/测试过程中的步数
    n_eval_or_test_steps += 1
    # 计算验证/测试阶段的平均损失
    eval_or_test_loss = eval_or_test_loss / n_eval_or_test_steps

    # 设置结果文件的路径
    if choice == 'evaluate':
        output_file = os.path.join(args.output_dir, 'eval_results.txt')  # /output/eval_results.txt
    if choice == 'test':
        output_file = os.path.join(args.output_dir, 'test_results.txt')  # /output/test_results.txt
    # 设置打开文件的模式（文件不存在：写入模式；文件已存在：追加写入模式）
    file_mode = 'w' if not os.path.exists(output_file) else 'a+'

    # 将结果写入结果文件
    with open(output_file, file_mode) as writer:
        if choice == 'evaluate':
            logger.info('*' * 20 + ' Evaluate results ' + '*' * 20)
            logger.info('evaluate_loss = %s', str(eval_or_test_loss))
            logger.info('evaluate_ave_f1 = %s', str(ave_f1))
            writer.write(('evaluate_ave_f1 = %s\n' % str(ave_f1)))
        if choice == 'test':
            logger.info('*' * 20 + ' Test results ' + '*' * 20)
            logger.info('test_loss = %s', str(eval_or_test_loss))
            logger.info('test_ave_f1 = %s', str(ave_f1))
            writer.write(('test_ave_f1 = %s\n' % str(ave_f1)))
        # 记录四个维度人格检测的结果
        process_results(writer, results1, results2, results3, results4)

    return [results1, results2, results3, results4], eval_or_test_loss, ave_f1


def get_labels(args, model, test_dataset):
    """
    获取预测标签和真实标签
    """

    # 用于将样本组织成批次
    collate_fn = get_collate_fn()
    # 数据加载器（根据给定的采样器、批大小和数据组织函数，加载测试数据集中的数据并组织成批次）
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.all_gpu_eval_batch_size,
                                 collate_fn=collate_fn)
    preds1, preds2, preds3, preds4 = None, None, None, None  # 模型预测的类别
    out_labels1, out_labels2, out_labels3, out_labels4 = None, None, None, None  # 真实的类别标签

    # 遍历测试数据集中的每个批次
    for batch in test_dataloader:
        model.eval()  # 将模型设置为验证模式
        batch = tuple(t.to(args.device) for t in batch)  # 将当前批次中的所有张量移动到GPU上

        with torch.no_grad():  # 不需要计算梯度
            # 从当前批次中获取输入数据和标签
            inputs, label1, label2, label3, label4, polarities = get_input_from_batch(batch)
            # 将输入数据传递给模型，得到四个任务的预测
            logit1, logit2, logit3, logit4 = model(**inputs, **polarities)

            # 记录每个批次的预测结果和真实标签
            preds1, preds2, preds3, preds4, out_labels1, out_labels2, out_labels3, out_labels4 = record_predictions_and_labels(
                logit1, logit2, logit3, logit4, label1, label2, label3, label4,
                preds1, preds2, preds3, preds4, out_labels1, out_labels2, out_labels3, out_labels4)

    # 根据预测值和标签值计算最终结果
    result1, result2, result3, result4 = compute_final_results(preds1, preds2, preds3, preds4,
                                                               out_labels1, out_labels2, out_labels3,
                                                               out_labels4)
    # 计算所有任务的 F1 分数的平均值
    ave_f1 = (result1['f1'] + result2['f1'] + result3['f1'] + result4['f1']) / 4.0

    # 返回预测标签和真实标签
    return preds1, preds2, preds3, preds4, out_labels1, out_labels2, out_labels3, out_labels4, ave_f1


def change_logit_dimension(logit1, logit2, logit3, logit4):
    """
    训练最后一个 step 时，当出现 logit 是一维的情况，把 logit 拓展成二维的
    """

    if len(logit1.shape) == 1:
        # logit1 = torch.unsqueeze(logit1, 0)
        logit1 = logit1.view(-1, 2)
    if len(logit2.shape) == 1:
        # logit2 = torch.unsqueeze(logit2, 0)
        logit2 = logit2.view(-1, 2)
    if len(logit3.shape) == 1:
        # logit3 = torch.unsqueeze(logit3, 0)
        logit3 = logit3.view(-1, 2)
    if len(logit4.shape) == 1:
        # logit4 = torch.unsqueeze(logit4, 0)
        logit4 = logit4.view(-1, 2)
    return logit1, logit2, logit3, logit4


def compute_batch_ave_loss(logit1, logit2, logit3, logit4, label1, label2, label3, label4):
    """
    根据预测值和标签值计算损失
    """

    # 训练最后一个 step 时，出现 logit 是一维的情况，把 logit 拓展成二维的
    logit1, logit2, logit3, logit4 = change_logit_dimension(logit1, logit2, logit3, logit4)

    # 计算四个子任务的交叉熵损失（交叉熵损失是常用于多类别分类问题的损失函数，用于衡量模型的预测与真实标签之间的差异）
    # logit 是模型的输出值（预测值），label 是真实标签，计算损失时对每个样本的损失值进行求和
    loss1 = F.cross_entropy(logit1, label1, reduction='sum')
    loss2 = F.cross_entropy(logit2, label2, reduction='sum')
    loss3 = F.cross_entropy(logit3, label3, reduction='sum')
    loss4 = F.cross_entropy(logit4, label4, reduction='sum')

    # 当前批次中的样本数 B
    batch_samples = label1.size(0)
    # 当前批次的平均损失（将四个任务的损失值相加并进行平均）
    batch_ave_loss = (loss1 + loss2 + loss3 + loss4) / (4.0 * batch_samples)

    return batch_ave_loss


def record_predictions_and_labels(logit1, logit2, logit3, logit4, label1, label2, label3, label4,
                                  preds1, preds2, preds3, preds4, out_labels1, out_labels2, out_labels3, out_labels4):
    """
    记录每个批次的预测结果和真实标签
    """

    # 训练最后一个 step 时，出现 logit 是一维的情况，把 logit 拓展成二维的
    logit1, logit2, logit3, logit4 = change_logit_dimension(logit1, logit2, logit3, logit4)

    if preds1 is None:
        preds1 = logit1.detach().cpu().numpy()
        preds2 = logit2.detach().cpu().numpy()
        preds3 = logit3.detach().cpu().numpy()
        preds4 = logit4.detach().cpu().numpy()
        out_labels1 = label1.detach().cpu().numpy()
        out_labels2 = label2.detach().cpu().numpy()
        out_labels3 = label3.detach().cpu().numpy()
        out_labels4 = label4.detach().cpu().numpy()
    else:
        # 在行方向上追加
        preds1 = np.append(preds1, logit1.detach().cpu().numpy(), axis=0)
        preds2 = np.append(preds2, logit2.detach().cpu().numpy(), axis=0)
        preds3 = np.append(preds3, logit3.detach().cpu().numpy(), axis=0)
        preds4 = np.append(preds4, logit4.detach().cpu().numpy(), axis=0)
        out_labels1 = np.append(out_labels1, label1.detach().cpu().numpy(), axis=0)
        out_labels2 = np.append(out_labels2, label2.detach().cpu().numpy(), axis=0)
        out_labels3 = np.append(out_labels3, label3.detach().cpu().numpy(), axis=0)
        out_labels4 = np.append(out_labels4, label4.detach().cpu().numpy(), axis=0)

    return preds1, preds2, preds3, preds4, out_labels1, out_labels2, out_labels3, out_labels4


def compute_final_results(preds1, preds2, preds3, preds4, out_labels1, out_labels2, out_labels3, out_labels4):
    """
    根据预测值和标签值计算最终结果
    """

    # 将预测结果数组 preds 转换为对每个样本最可能的类别索引（通过找到每一行中具有最大值的索引，即在第一维上取最大值的索引）
    out_preds1 = np.argmax(preds1, axis=1)
    out_preds2 = np.argmax(preds2, axis=1)
    out_preds3 = np.argmax(preds3, axis=1)
    out_preds4 = np.argmax(preds4, axis=1)
    # 计算任务的评估指标（准确率和 F1 分数）
    result1 = compute_metrics(out_preds1, out_labels1)
    result2 = compute_metrics(out_preds2, out_labels2)
    result3 = compute_metrics(out_preds3, out_labels3)
    result4 = compute_metrics(out_preds4, out_labels4)

    return result1, result2, result3, result4


def process_results(writer, results1, results2, results3, results4):
    """
    记录四个维度人格检测的结果
    遍历 result 字典中的所有键（即指标名称），对每个指标进行处理
    """

    # 遍历 result 字典中的所有键（即指标名称），对每个指标进行处理
    for key in sorted(results1.keys()):
        logger.info("1:  %s = %s", key, str(results1[key]))
        writer.write("1:  %s = %s\n" % (key, str(results1[key])))
        writer.write('----------------------------------\n')
    for key in sorted(results2.keys()):
        logger.info("2:  %s = %s", key, str(results2[key]))
        writer.write("2:  %s = %s\n" % (key, str(results2[key])))
        writer.write('----------------------------------\n')
    for key in sorted(results3.keys()):
        logger.info("3:  %s = %s", key, str(results3[key]))
        writer.write("3:  %s = %s\n" % (key, str(results3[key])))
        writer.write('-----------------------------------\n')
    for key in sorted(results4.keys()):
        logger.info("4:  %s = %s", key, str(results4[key]))
        writer.write("4:  %s = %s\n" % (key, str(results4[key])))
        writer.write('-----------------------------------\n')
    writer.write('\n')


def tensorboard_results(tb_writer, results1, results2, results3, results4, global_step, choice):
    """
    将结果写入 TensorBoard 日志
    """

    if choice == 'train':
        for key, value in results1.items():
            # 第一个参数：标签的名称，第二个参数：要记录的值，第三个参数：TensorBoard 中的 x 轴坐标
            tb_writer.add_scalar('train1_{}'.format(key), value, global_step)
        for key, value in results2.items():
            tb_writer.add_scalar('train2_{}'.format(key), value, global_step)
        for key, value in results3.items():
            tb_writer.add_scalar('train3_{}'.format(key), value, global_step)
        for key, value in results4.items():
            tb_writer.add_scalar('train4_{}'.format(key), value, global_step)
    if choice == 'test':
        for key, value in results1.items():
            tb_writer.add_scalar('test1_{}'.format(key), value, global_step)
        for key, value in results2.items():
            tb_writer.add_scalar('test2_{}'.format(key), value, global_step)
        for key, value in results3.items():
            tb_writer.add_scalar('test3_{}'.format(key), value, global_step)
        for key, value in results4.items():
            tb_writer.add_scalar('test4_{}'.format(key), value, global_step)
    if choice == 'evaluate':
        for key, value in results1.items():
            tb_writer.add_scalar('eval1_{}'.format(key), value, global_step)
        for key, value in results2.items():
            tb_writer.add_scalar('eval2_{}'.format(key), value, global_step)
        for key, value in results3.items():
            tb_writer.add_scalar('eval3_{}'.format(key), value, global_step)
        for key, value in results4.items():
            tb_writer.add_scalar('eval4_{}'.format(key), value, global_step)


def get_input_from_batch(batch):
    """
    从批次数据中提取输入和标签信息
    """
    inputs = {'post_tokens_id': batch[0]}  # B * 50 * 70
    label1 = batch[1]
    label2 = batch[2]
    label3 = batch[3]
    label4 = batch[4]
    polarities = {'polarities': batch[5]}  # B * 50 * 20
    return inputs, label1, label2, label3, label4, polarities


def get_optimizer(args, model, choice):
    """
    根据传入的参数和模型来获取优化器对象
    """

    # 获取在多个 GPU 上运行的原始模型
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # 计算模型中所有参数的总数量
    total_params_num = sum(x.numel() for x in model.parameters())
    # 计算预训练模型参数的数量
    bert_params_num = sum(x.numel() for x in model.pretrain_models.parameters())
    if choice == 'gat':
        # 计算图网络参数的数量
        gm_params_num = sum(x.numel() for x in model.multi_gat.gat1.parameters()) * 4
    if choice == 'pd_han':
        # 计算图网络参数的数量
        gm_params_num = sum(x.numel() for x in model.pd_han.post_gat.parameters())
        gm_params_num += sum(x.numel() for x in model.pd_han.user_post_gat1.parameters()) * 4
    # 计算其他部分的参数数量
    other_params_num = total_params_num - bert_params_num - gm_params_num
    # other_params_num = total_params_num - bert_params_num
    if choice == 'gat':
        logger.info('parameters of plm, gat and other: %d, %d, %d', bert_params_num, gm_params_num,
                    other_params_num)
    if choice == 'pd_han':
        logger.info('parameters of plm, pd_han and other: %d, %d, %d', bert_params_num, gm_params_num,
                    other_params_num)

    # 获取预训练模型参数的 ID 列表
    bert_params_id = list(map(id, model.pretrain_models.parameters()))
    # 获取图网络参数的 ID 列表
    gm_params_id = []
    # 获取 GRU 网络参数的 ID 列表
    gru_params_id = []
    if choice == 'gat':
        gm_params_id += list(map(id, model.multi_gat.gat1.parameters()))
        gm_params_id += list(map(id, model.multi_gat.gat2.parameters()))
        gm_params_id += list(map(id, model.multi_gat.gat3.parameters()))
        gm_params_id += list(map(id, model.multi_gat.gat4.parameters()))
    if choice == 'pd_han':
        gm_params_id += list(map(id, model.pd_han.post_gat.parameters()))
        gm_params_id += list(map(id, model.pd_han.user_post_gat1.parameters()))
        gm_params_id += list(map(id, model.pd_han.user_post_gat2.parameters()))
        gm_params_id += list(map(id, model.pd_han.user_post_gat3.parameters()))
        gm_params_id += list(map(id, model.pd_han.user_post_gat4.parameters()))
        # 使用结合注意力机制的 GRU
        if args.attention_gru:
            gru_params_id += list(map(id, model.pd_han.attention_gru1.parameters()))
            gru_params_id += list(map(id, model.pd_han.attention_gru2.parameters()))
            gru_params_id += list(map(id, model.pd_han.attention_gru3.parameters()))
            gru_params_id += list(map(id, model.pd_han.attention_gru4.parameters()))
    if choice == 'multi_attn_gru':
        gru_params_id += list(map(id, model.multi_attn_gru.attention_gru1.parameters()))
        gru_params_id += list(map(id, model.multi_attn_gru.attention_gru2.parameters()))
        gru_params_id += list(map(id, model.multi_attn_gru.attention_gru3.parameters()))
        gru_params_id += list(map(id, model.multi_attn_gru.attention_gru4.parameters()))

    # 通过 filter 函数来筛选出图网络参数
    gm_params = filter(lambda p: id(p) in gm_params_id, model.parameters())
    # 通过 filter 函数来筛选出 GRU 网络参数
    gru_params = filter(lambda p: id(p) in gru_params_id, model.parameters())
    # 通过 filter 函数来筛选其他参数
    other_params = filter(lambda p: id(p) not in bert_params_id + gm_params_id + gru_params_id, model.parameters())

    # 包含不同参数组的列表，每个参数包含参数列表和该参数组使用的学习率
    optimizer_grouped_parameters_with_lr = [
        {'params': model.pretrain_models.parameters(), 'lr': args.plm_learning_rate},
        {'params': gm_params, 'lr': args.gm_learning_rate},
        {'params': gru_params, 'lr': args.gru_learning_rate},
        {'params': other_params, 'lr': args.other_learning_rate}
    ]

    # 使用了 Adam 优化器，传递了之前创建的参数组列表 optimizer_grouped_parameters 以及参数 eps（即 epsilon，用于数值稳定性的小常数）
    model_optimizer = Adam(optimizer_grouped_parameters_with_lr, eps=args.adam_epsilon)
    # model_optimizer = Adam(optimizer_grouped_parameters_with_lr, eps=args.adam_epsilon, weight_decay=1e-5)

    return model_optimizer


def get_optimizer2(args, model):
    """
    根据传入的参数和模型来获取优化器对象
    """

    # 获取在多个 GPU 上运行的原始模型
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # 计算模型中所有参数的总数量
    total_params_num = sum(x.numel() for x in model.parameters())
    logger.info('total parameters: %d', total_params_num)
    # 创建了一个包含不同参数组的列表，每个参数包含参数列表和该参数组使用的学习率
    optimizer_grouped_parameters_with_lr = [
        {'params': model.parameters(), 'lr': 1e-5}
    ]

    # 使用了 Adam 优化器，传递了之前创建的参数组列表 optimizer_grouped_parameters 以及参数 eps（即 epsilon，用于数值稳定性的小常数）
    model_optimizer = Adam(optimizer_grouped_parameters_with_lr, eps=args.adam_epsilon)

    return model_optimizer


def write_tb_writer(tb_writer: SummaryWriter, args):
    """
    用于将训练过程中的参数和设置信息写入 TensorBoard 以供可视化
    """
    tb_writer.add_text('seed', str(args.seed))
    tb_writer.add_text('plm_lr', str(args.plm_learning_rate))
    tb_writer.add_text('gm_lr', str(args.gm_learning_rate))
    tb_writer.add_text('other_lr', str(args.other_learning_rate))
    tb_writer.add_text('gat_layers', str(args.gat_num_layers))
    tb_writer.add_text('final_hidden_size', str(args.final_hidden_size))
    tb_writer.add_text('bsz', str(args.all_gpu_train_batch_size))


def compute_metrics(preds, labels):
    """
    计算分类模型的准确率（accuracy）和 F1 分数（macro-averaged F1 score），这两个指标常用于评估分类模型的性能
    :param preds: 模型预测的类别标签
    :param labels: 真实的类别标签
    """

    # 模型预测正确的比例，即准确率
    acc = ((preds == labels).mean())
    # F1 分数：精确率和召回率的调和平均
    # f1_score 函数的 average 参数被设置为 'macro'，这表示计算各类别的 F1 分数，然后对所有类别的分数取平均（这用于处理不平衡类别分布的情况）
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1
    }


def set_seed(args):
    """
    设置随机种子，以确保训练过程的可复现性。
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_collate_fn():
    """
    接受一个批次的样本列表 batch，将一个批次的样本组织成一个批次的 PyTorch 张量。
    """
    return my_collate
