import pickle
import os
import random


def data_split(file, result_path, top):
    """
    数据集切割
    """

    with open(file, 'rb') as f:
        pandora = pickle.load(f)
    annotations = pandora['annotations']  # MBTI 标签
    posts_text = pandora['posts_text']  # 帖子文本
    posts_num = pandora['posts_num']  # 帖子数量
    user_num = len(annotations)  # 用户数量
    user_ids = list(range(user_num))  # 用户 id 列表
    # 随机打乱 user_ids 列表，以便后续的数据拆分
    random.shuffle(user_ids)

    # 将数据集分成训练集、验证集和测试集，并提取相应的数据
    train_ids, eval_ids, test_ids = (user_ids[:int(0.6 * user_num)],
                                     user_ids[int(0.6 * user_num):int(0.8 * user_num)],
                                     user_ids[int(0.8 * user_num):])
    train_text, train_num, train_annotations = ([keep_top(posts_text[i], top) for i in train_ids],
                                                [min(posts_num[i], top) for i in train_ids],
                                                [annotations[i] for i in train_ids])
    eval_text, eval_num, eval_annotations = ([keep_top(posts_text[i], top) for i in eval_ids],
                                             [min(posts_num[i], top) for i in eval_ids],
                                             [annotations[i] for i in eval_ids])
    test_text, test_num, test_annotations = ([keep_top(posts_text[i], top) for i in test_ids],
                                             [min(posts_num[i], top) for i in test_ids],
                                             [annotations[i] for i in test_ids])

    # 保存训练集、验证集和测试集数据
    save_data(train_text, train_num, train_annotations, result_path, 'train')
    save_data(eval_text, eval_num, eval_annotations, result_path, 'eval')
    save_data(test_text, test_num, test_annotations, result_path, 'test')


def keep_top(data, k, recent=True):
    """
    从给定的数据列表 data 中保留前 k 个元素
    """
    if len(data) <= k:
        return data
    else:
        if recent:
            return data[-k:]
        else:
            return data[:k]


def save_data(posts_text, posts_num, annotations, result_path, option):
    """
    将指定的数据保存到文件中
    """
    with open(os.path.join(result_path, option + '.pkl'), 'wb') as f:
        data = {'posts_text': posts_text, 'posts_num': posts_num, 'annotations': annotations}
        pickle.dump(data, f)


if __name__ == '__main__':
    top = 2000
    random.seed(0)
    file = 'preprocessed/pandora/pandora.pkl'
    result_path = 'preprocessed/pandora_test'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    data_split(file, result_path, top)
