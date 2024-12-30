import re
import pickle
import random
import pandas as pd

MBTIs = ('INTJ', 'INTP', 'INFP', 'ENTP', 'ISTP', 'ISFP', 'ESTJ', 'ISTJ',
         'ESTP', 'ISFJ', 'ENFP', 'ESFP', 'ESFJ', 'ENFJ', 'INFJ', 'ENTJ',
         'TYPE_MENTION', 'COGFUNC_MENTION')


def get_user_label():
    """
    获取用户 MBTI 标签字典
    """
    # 从 author_profiles.csv 中读取用户和对应的 MBTI 类型标签
    data = pd.read_csv('raw/pandora_comments/author_profiles.csv')
    user = list(data['author'])  # 用户
    label = list(data['mbti'])  # MBTI 类型标签
    user_dict = {}  # 用户字典
    # 遍历所有 MBTI 标签，如果标签在 MBTIs 列表中，将用户与其 MBTI 类型添加到用户字典中
    for i, t in enumerate(label):
        if str(t).upper() not in MBTIs:
            continue
        else:
            user_dict[user[i]] = t
    return user_dict


def find_all_MBTIs(post, mbti):
    """
    在给定的文本 post 中找到所有匹配指定 MBTI 类型 mbti 的位置
    """
    return [(match.start(), match.end()) for match in re.finditer(mbti, post)]


def data_preprocessing():
    # 获取用户 MBTI 标签字典
    user_dict = get_user_label()
    # 从 all_comments_since_2015.csv 文件中读取用户和评论内容
    data = pd.read_csv('raw/pandora_comments/all_comments_since_2015.csv')
    user = list(data['author'])  # 用户
    text = list(data['body'])  # 帖子文本
    token = ''  # '' or '<mask>'  # 用于文本处理的标记
    # 正则表达式对象（用于匹配文本中的链接）
    reg_link = re.compile('http\S+', flags=re.MULTILINE)
    # 帖子字典
    posts = {'annotations': [], 'posts_text': [], 'posts_num': [], 'max_len': [], 'users': []}

    # recorded_user = []
    # 逐行处理帖子文本，将它们按用户组织，并进行一些文本清洗和标记操作
    for i, t in enumerate(user):
        if t in user_dict.keys():
            if t not in posts['users']:  # new user
                posts['users'].append(t)  # 用户
                posts['posts_text'].append([])  # 帖子文本
                posts['annotations'].append(user_dict[t].upper())  # MBTI 标签
                posts['max_len'].append(0)  # 帖子最大长度
                posts['posts_num'].append(0)  # 帖子数量
            user_id = posts['users'].index(t)  # 用户 id

            # 使用正则表达式 reg_link 删除文本中的链接
            filter_text = reg_link.sub('', text[i])
            if filter_text != '':
                # 删除 MBTI类型内容
                for MBTI in MBTIs:
                    mbti_idx_list = find_all_MBTIs(filter_text.lower(), MBTI.lower())
                    delete_idx = 0
                    for start, end in mbti_idx_list:
                        filter_text = filter_text[:start - delete_idx] + token + filter_text[end - delete_idx:]
                        delete_idx += end - start + len(token)
                post_len = len(filter_text.split(' '))
                # 如果文本的单词数大于 5：将文本添加到用户的帖子列表中，并记录最大的帖子长度
                if post_len > 5:
                    posts['posts_text'][user_id].append(filter_text)
                    if posts['max_len'][user_id] < post_len:
                        posts['max_len'][user_id] = post_len
                    posts['posts_num'][user_id] += 1
                else:
                    continue

    # 将 MBTI 类别标签映射为二进制标签
    label_lookup = {'E': 0, 'I': 1, 'S': 0, 'N': 1, 'T': 0, 'F': 1, 'J': 0, 'P': 1}
    types = posts['annotations']
    # 四个 label 分别包含了每个样本在四个维度上的二进制标签
    label0, label1, label2, label3 = [], [], [], []
    for type in types:
        label0.append(label_lookup[list(type)[0]])
        label1.append(label_lookup[list(type)[1]])
        label2.append(label_lookup[list(type)[2]])
        label3.append(label_lookup[list(type)[3]])

    # 统计相关数据信息，并保存到 data_statistic.xlsx
    # save_data = pd.DataFrame(posts['annotations'], columns=['annotations'])
    # save_data['posts_num'] = posts['posts_num']
    # save_data['max_len'] = posts['max_len']
    # save_data['I_E'] = label0
    # save_data['N_S'] = label1
    # save_data['F_T'] = label2
    # save_data['P_J'] = label3
    # save_data.to_excel('./preprocessed/pandora/data_statistic.xlsx')

    # 将处理后的数据 posts 以二进制形式保存到 'pandora.pkl' 文件中，以便以后使用
    with open('./preprocessed/pandora/pandora.pkl', 'wb') as f:
        pickle.dump(posts, f)


if __name__ == '__main__':
    random.seed(0)
    data_preprocessing()
