import logging
import math

import torch
from torch.utils.data import Dataset

import pickle
from tqdm import tqdm
import string
from senticnet.senticnet import SenticNet
import nlpaug.augmenter.word as naw
from transformers import BertTokenizer, BertForMaskedLM

logger = logging.getLogger(__name__)


def load_split_datasets(args):
    # 加载训练集、验证集、测试集（从 pickle 格式的文件中读取数据）
    train = pickle.load(open('../data/preprocessed/' + args.task + '/train.pkl', 'rb'))
    train_text = train['posts_text']
    train_label = train['annotations']
    eval = pickle.load(open('../data/preprocessed/' + args.task + '/eval.pkl', 'rb'))
    eval_text = eval['posts_text']
    eval_label = eval['annotations']
    test = pickle.load(open('../data/preprocessed/' + args.task + '/test.pkl', 'rb'))
    test_text = test['posts_text']
    test_label = test['annotations']

    # 处理训练集、验证集、测试集（将文本数据和标签转换为模型需要的数据格式，并统计了标签的出现次数）
    # [{'posts': ['persona 5 !! type ze characters', "? mayb    1. i'm very straightforward and blunt with how i feel about anything. i'm oblivious to how others feel and sometimes this leads me to hurt others because i dwell in my own little...", 'Seems like it.', 'drank', 'troll :^)', ' Instinct. Switching to Mystic.  Sorry team.', "Anybody watching Love Live! Sunshine!! ? I'm really curious about their types, though we're only 3 episodes in so it may be a bit difficult...", "1. Fe?  1 1/2. Te.. 2. I think Ti? 3. Ne 4. wut... i guess you're trying to relate this to Si? lol 5. er not function related... or even personality related :p 6. Fi 7. Ne / lack of sensing......", 'no.  you could be fake.', 'heckkk yeahhh you mentioned some of my favorite anime youtubers!!  i would have met them yesterday at anime expo (hint: i went there yesterday...(july 1st)), but it was too crowded. :/ perhaps...', 'sure', 'Possibly', 'unknown = inaccurate   ', 'MEME. quite a rare personality type... way rarer than ..', 'You could be an ...', 's are nerds that barely have any acess to human emotions. the only people they care about is their anime waifus...', "i wish d.va was ... though she's most likely xsfp idk   do u play overwatch", 'I actually thought you were an ...', 's are such freaking trolls.... like just pls stop', "ur jokes aren't even funny...", 'leictreon ur avatar is really cute!!!  (for some reason i said accurate at first lol) -- (at above person)  ur a mystery', 'Man... This all seems really accurate! Props to ya! :D', 'yes u r', 'pfne', "Lever  (oh whoops... that's already been said. oh well >:D)", "THE LAST VID!!! That's on my liked videos playlist on youtube; I didn't think I would see it here!! Bossa Nova like that = perfection.   (Sorry I got a bit too excited lmao.)  Anyways, jazz is...", " I love the vibes and lyrics of this song. It's just so chill~", "Oh gosh, I'm such a Dreamer hehe. Maybe with a little mix of Idealist but mostly Dreamer :p", 'Agnostic theist born in a Christian fanily. :o', ' :p', 'Thanks for joining this topic! :o  Hmm, I relate to both of you, but I think I identify with you more.  I also took the quiz you suggested a billion of times because I tried to answer all of the...', 'i present to you an ', 'You see more . :o', ' :) Pretty noice', "Good thing you mentioned that. :P My thoughts do tend to be very personal, so I'm not always one to share different ideas with a bunch a random people. If I'm comfortable with an individual and I...", 'o-o-oh my god.... there was just one period in that sentence... my head is spinning  i think just had a gag reflex', 'i read turnip as trump lmao', 'Definitely inferior-Te. :p', "ennifer Maybe I could explain my intuition function without directly saying which statement applies to Ni or Ne. I'm pretty much just gonna ramble lol :p.   I am very open to possibilty;...", "Besides the stereotypes, I'm still not sure if I use Ni or Ne :/", 'That.... That confused me. :', '|quite accurate m8 :^)', "i thought i was but i just ??? don't know", 'screw pokemon and digimon BEYBLADE LET IT RIP', "i used to  wasn't it on cartoon network", 'u guys have taken over the thread', "Once Upon A Time  ennifer Thanks for explaining :) I don't think I should really rely on stereotypes; they will definitely confuse me when it comes to figuring out my mbti type. I should continue...", 'b-b-b-b-b-b-b-b-b-b-b-baka!!!!', ', of course :p', "Heh, ya never know :P  s are noted for being very creative and I'm just... not. s are called artistic, poetic, and all that jazz, but all I do is play video games, read, and watch TV. I...'"],
    # 'label0': 0, 'label1': 0, 'label2': 0, 'label3': 0, 'polarity': [[],[]]}]
    deal_train_data = process_data(args, train_text, train_label, 'train')
    deal_eval_data = process_data(args, eval_text, eval_label, 'eval')
    deal_test_data = process_data(args, test_text, test_label, 'test')

    # 创建训练集、验证集、测试集的数据加载器（将处理后的数据转换为适合模型输入的数据集对象）
    # (tensor([[ 101,  146, 1567,  ...,    0,    0,    0],
    #         [ 101,  146, 1341,  ...,    0,    0,    0],
    #         [ 101, 1422, 1954,  ...,    0,    0,    0],
    #         ...,
    #         [   0,    0,    0,  ...,    0,    0,    0],
    #         [   0,    0,    0,  ...,    0,    0,    0],
    #         [   0,    0,    0,  ...,    0,    0,    0]]), tensor(0), tensor(0), tensor(0), tensor(1))
    train_dataset = MBTI_Dataset(deal_train_data, args)
    eval_dataset = MBTI_Dataset(deal_eval_data, args)
    test_dataset = MBTI_Dataset(deal_test_data, args)

    return train_dataset, eval_dataset, test_dataset


def process_data(args, poster, label, option):
    """
    主要用于处理数据，将输入的帖子文本和标签转换为适合模型输入的数据格式（还统计了各个标签出现的次数）
    """

    sn = SenticNet()
    label_lookup = {'E': 1, 'I': 0, 'S': 1, 'N': 0, 'T': 1, 'F': 0, 'J': 1, 'P': 0}
    # poster_data 列表每个元素都是一个字典，表示一个用户的帖子及其标签的信息、帖子感情极性信息
    poster_data = [{'posts': t,
                    'label0': label_lookup[list(label[i])[0]],
                    'label1': label_lookup[list(label[i])[1]],
                    'label2': label_lookup[list(label[i])[2]],
                    'label3': label_lookup[list(label[i])[3]],
                    'polarities': get_text_polarity_feature(args, sn, t)}
                   for i, t in enumerate(poster)]
    # print(poster_data[:2])

    if args.use_data_augmentation:
        if option == 'train':
            # 复制 label1 为 1 的数据，并对复制的 posts 进行处理
            new_poster_data = []
            for data in poster_data:
                if data['label1'] == 1:
                    # 深拷贝以避免修改原始数据
                    new_data = data.copy()
                    # new_data['posts'] = [synonym_replacement(post, n=1) for post in data['posts']]
                    new_data['posts'] = mlm_data_augmentation(data['posts'])
                    new_poster_data.append(new_data)
            # 将处理后的数据拼接到原始数据后面
            poster_data = poster_data + new_poster_data

    # 根据整体标签统计 I、E、S、N、T、F、P、J 标签的出现次数。
    # I, E, S, N, T, F, P, J = 0, 0, 0, 0, 0, 0, 0, 0
    # total = 0
    # for t in label:
    #     I, E, S, N, T, F, P, J, total = counter(t, I, E, S, N, T, F, P, J, total)
    #     if 'S' in t:
    #         I, E, S, N, T, F, P, J, total = counter(t, I, E, S, N, T, F, P, J, total)
    # print('I', I)
    # print('E', E)
    # print('S', S)
    # print('N', N)
    # print('T', T)
    # print('F', F)
    # print('P', P)
    # print('J', J)
    # print('total', total)

    # 返回处理后的数据，即帖子文本和数值标签的信息。
    # [{'posts': ['persona 5 !! type ze characters', "? mayb    1. i'm very straightforward and blunt with how i feel about anything. i'm oblivious to how others feel and sometimes this leads me to hurt others because i dwell in my own little...", 'Seems like it.', 'drank', 'troll :^)', ' Instinct. Switching to Mystic.  Sorry team.', "Anybody watching Love Live! Sunshine!! ? I'm really curious about their types, though we're only 3 episodes in so it may be a bit difficult...", "1. Fe?  1 1/2. Te.. 2. I think Ti? 3. Ne 4. wut... i guess you're trying to relate this to Si? lol 5. er not function related... or even personality related :p 6. Fi 7. Ne / lack of sensing......", 'no.  you could be fake.', 'heckkk yeahhh you mentioned some of my favorite anime youtubers!!  i would have met them yesterday at anime expo (hint: i went there yesterday...(july 1st)), but it was too crowded. :/ perhaps...', 'sure', 'Possibly', 'unknown = inaccurate   ', 'MEME. quite a rare personality type... way rarer than ..', 'You could be an ...', 's are nerds that barely have any acess to human emotions. the only people they care about is their anime waifus...', "i wish d.va was ... though she's most likely xsfp idk   do u play overwatch", 'I actually thought you were an ...', 's are such freaking trolls.... like just pls stop', "ur jokes aren't even funny...", 'leictreon ur avatar is really cute!!!  (for some reason i said accurate at first lol) -- (at above person)  ur a mystery', 'Man... This all seems really accurate! Props to ya! :D', 'yes u r', 'pfne', "Lever  (oh whoops... that's already been said. oh well >:D)", "THE LAST VID!!! That's on my liked videos playlist on youtube; I didn't think I would see it here!! Bossa Nova like that = perfection.   (Sorry I got a bit too excited lmao.)  Anyways, jazz is...", " I love the vibes and lyrics of this song. It's just so chill~", "Oh gosh, I'm such a Dreamer hehe. Maybe with a little mix of Idealist but mostly Dreamer :p", 'Agnostic theist born in a Christian fanily. :o', ' :p', 'Thanks for joining this topic! :o  Hmm, I relate to both of you, but I think I identify with you more.  I also took the quiz you suggested a billion of times because I tried to answer all of the...', 'i present to you an ', 'You see more . :o', ' :) Pretty noice', "Good thing you mentioned that. :P My thoughts do tend to be very personal, so I'm not always one to share different ideas with a bunch a random people. If I'm comfortable with an individual and I...", 'o-o-oh my god.... there was just one period in that sentence... my head is spinning  i think just had a gag reflex', 'i read turnip as trump lmao', 'Definitely inferior-Te. :p', "ennifer Maybe I could explain my intuition function without directly saying which statement applies to Ni or Ne. I'm pretty much just gonna ramble lol :p.   I am very open to possibilty;...", "Besides the stereotypes, I'm still not sure if I use Ni or Ne :/", 'That.... That confused me. :', '|quite accurate m8 :^)', "i thought i was but i just ??? don't know", 'screw pokemon and digimon BEYBLADE LET IT RIP', "i used to  wasn't it on cartoon network", 'u guys have taken over the thread', "Once Upon A Time  ennifer Thanks for explaining :) I don't think I should really rely on stereotypes; they will definitely confuse me when it comes to figuring out my mbti type. I should continue...", 'b-b-b-b-b-b-b-b-b-b-b-baka!!!!', ', of course :p', "Heh, ya never know :P  s are noted for being very creative and I'm just... not. s are called artistic, poetic, and all that jazz, but all I do is play video games, read, and watch TV. I...'"],
    # 'label0': 0, 'label1': 0, 'label2': 0, 'label3': 0}]
    return poster_data


def get_text_polarity_feature(args, sn, sentence_list):
    """
    获取文本的情感极性特征
    """
    feature_list = []
    for sentence in sentence_list:
        # 去掉文本中的标点符号
        cleaned_sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        # 将文本拆分称单词列表
        words = cleaned_sentence.split()
        # 计算文本（单词列表）的极性值
        sentence_polarity_value = 0.0
        word_polarity_value_num = 0
        for word in words:
            try:
                word_polarity_value = float(sn.polarity_value(word.lower()))
                sentence_polarity_value = sentence_polarity_value + word_polarity_value
                word_polarity_value_num += 1
            except KeyError:
                continue
        # 文本的极性值取所有单词极性值和的平均值
        if word_polarity_value_num != 0:
            # sentence_polarity_value = sentence_polarity_value / word_polarity_value_num
            sentence_polarity_value = sentence_polarity_value / len(words)
        # 文本的极性值转换成情感等级，共有 20 个级别：[0, 1, ..., 19]
        sentence_polarity_class = math.floor((sentence_polarity_value + 1) * 10)
        # 将每个情感等级映射到一个 20 维向量中（在相应的位置值为 1，在所有其他位置值为 0），作为文本情感极性的指示
        sentence_polarity_feature = [1 if i == sentence_polarity_class else 0 for i in range(args.senticnet_dim)]
        # sentence_polarity_feature = torch.zeros(20)
        # sentence_polarity_feature[sentence_polarity_class] = 1
        feature_list.append(sentence_polarity_feature)
    return feature_list


class MBTI_Dataset(Dataset):
    """
    自定义的 MBTI 数据集类，用于在 PyTorch 中加载数据并准备模型训练、评估和测试所需的数据
    """

    def __init__(self, data, args):
        self.data = data
        self.args = args
        # 针对不同的预训练模型类型，根据分词器（Tokenizer）将 [PAD]、[CLS] 等特殊标记转换为对应的数值标识
        # 这些数值标识将在后续数据处理中用于数据的填充和分类
        if self.args.pretrain_type == 'bert':
            self.pad, self.cls = self.args.tokenizer.convert_tokens_to_ids(['[PAD]', '[CLS]'])
        elif self.args.pretrain_type == 'xlnet':
            self.pad, self.cls = self.args.tokenizer.convert_tokens_to_ids(['<pad>', '<cls>'])
        elif self.args.pretrain_type == 'roberta':
            self.pad, self.cls = self.args.tokenizer.convert_tokens_to_ids(['<pad>', '<s>'])
        elif self.args.pretrain_type == 'albert':
            self.pad, self.cls = self.args.tokenizer.convert_tokens_to_ids(['<pad>', '[CLS]'])
        else:
            raise NotImplementedError

        # 将用户的帖子文本转换为对应的数值标识（token ids）
        self.convert_features()

    def __len__(self):
        """
        返回数据集中样本的数量
        """
        return len(self.data)

    def _tokenize(self, text):
        """
        定义了一个私有方法 _tokenize，用于将文本转换为对应的 Token IDs
        """
        # 首先使用预训练模型的分词器（Tokenizer）的 tokenize 方法将文本拆分为标记（tokens）
        # 然后，使用分词器的 convert_tokens_to_ids 方法将标记转换为对应的整数 Token IDs
        # 最后，使用分词器的 build_inputs_with_special_tokens 方法将得到的 Token IDs 构建成适合模型输入的格式，包括添加 [CLS] 和 [SEP] 等特殊标记
        return self.args.tokenizer.build_inputs_with_special_tokens(
            self.args.tokenizer.convert_tokens_to_ids(self.args.tokenizer.tokenize(text)))

    def __getitem__(self, idx):
        """
        通过定义 __getitem__ 方法，可以通过索引从数据集中获取一个样本，以便在训练和推断过程中使用
        例如，在使用 PyTorch 的 DataLoader 时，可以通过调用 dataset[idx] 获取数据集中的第 idx 个样本
        """
        e = self.data[idx]
        items = e['post_tokens_id'], e['label0'], e['label1'], e['label2'], e['label3'], e['polarities']
        items_tensor = tuple(torch.tensor(t) for i, t in enumerate(items))
        # 返回一个包含 Tensor 对象的元组，表示一个样本的输入特征和各个标签。
        # (tensor([[ 101,  146, 1567,  ...,    0,    0,    0],
        #         [ 101,  146, 1341,  ...,    0,    0,    0],
        #         [ 101, 1422, 1954,  ...,    0,    0,    0],
        #         ...,
        #         [   0,    0,    0,  ...,    0,    0,    0],
        #         [   0,    0,    0,  ...,    0,    0,    0],
        #         [   0,    0,    0,  ...,    0,    0,    0]]),
        # tensor(0), tensor(0), tensor(0), tensor(1), tensor([[]]))
        return items_tensor

    def convert_feature(self, i):
        """
        将第 i 个用户的帖子文本转换为模型输入的特征表示，其中主要是将文本转换为对应的 Token IDs
        """

        post_tokens_id = []
        # 遍历每个 post（即指定数量样本中的每一篇帖子）做处理
        for post in self.data[i]['posts'][:self.args.max_post]:
            # 将文本标记转换为对应的 Token IDs
            input_ids = self._tokenize(post)
            pad_len = self.args.max_len - len(input_ids)

            # 如果剩余的序列长度可以填充（pad_len > 0），则根据不同的预训练模型类型，在序列末尾添加相应数量的填充标记
            # 如果剩余的序列长度不足以填充（pad_len <= 0），则根据不同的预训练模型类型，截断或保留特殊标记
            if pad_len > 0:
                if self.args.pretrain_type == 'bert':
                    input_ids += [self.pad] * pad_len
                elif self.args.pretrain_type == 'xlnet':
                    input_ids = [input_ids[-1]] + input_ids[:-1]
                    input_ids += [self.pad] * pad_len
                elif self.args.pretrain_type == 'roberta':
                    input_ids += [self.pad] * pad_len
                elif self.args.pretrain_type == 'albert':
                    input_ids += [self.pad] * pad_len
                else:
                    raise NotImplementedError
            else:
                if self.args.pretrain_type == 'bert':
                    input_ids = input_ids[:self.args.max_len - 1] + input_ids[-1:]
                elif self.args.pretrain_type == 'xlnet':
                    input_ids = [input_ids[-1]] + input_ids[:self.args.max_len - 2] + [input_ids[-2]]
                elif self.args.pretrain_type == 'roberta':
                    input_ids = input_ids[:self.args.max_len - 1] + input_ids[-1:]
                elif self.args.pretrain_type == 'albert':
                    input_ids = input_ids[:self.args.max_len - 1] + input_ids[-1:]
                else:
                    raise NotImplementedError

            # 通过一系列处理后，确保每个句子的 Token IDs 长度与 args.max_len 相同
            assert (len(input_ids) == self.args.max_len)
            # 将处理后的 Token IDs 列表存储到数据的 post_tokens_id 属性中
            post_tokens_id.append(input_ids)

        # 如果实际帖子数量小于 args.max_post，则使用填充标记填充剩余的帖子，确保每个样本都有相同数量的帖子
        real_post = len(post_tokens_id)
        for j in range(self.args.max_post - real_post):
            post_tokens_id.append([self.pad] * self.args.max_len)

        # 更新 post_tokens_id
        self.data[i]['post_tokens_id'] = post_tokens_id

    def append_polarities(self, i):
        """
        确保每个用户样本都有相同数量的帖子文本感情极性特征
        """

        polarities = []
        # 把 polarities 的数量控制在指定范围内
        for polarity in self.data[i]['polarities'][:self.args.max_post]:
            polarities.append(polarity)
        # 如果实际极性值数量小于 args.max_post，则填充剩余的极性值，确保每个样本都有相同数量的感情极性值
        real_num = len(polarities)
        for j in range(self.args.max_post - real_num):
            polarities.append([self.pad] * self.args.senticnet_dim)

        # 更新 polarities
        self.data[i]['polarities'] = polarities

    def convert_features(self):
        """
        将所有用户的帖子文本转换为对应的数值标识（token ids），并规范用户的帖子文本及其感情极性值数量
        """
        for i in tqdm(range(len(self.data))):
            self.convert_feature(i)
            self.append_polarities(i)


def my_collate(batch):
    """
    接受一个批次的样本列表 batch，组织成一个批次的 PyTorch 张量
    """

    # 将批次中的每个样本解压，并将每个样本的不同特征分别存储在指定变量中
    post_tokens_id, label0, label1, label2, label3, polarities = zip(*batch)  # from Dataset.__getitem__()
    # 将 post_tokens_id 中的 Tensor 进行堆叠，从而形成一个批次的张量
    post_tokens_id = torch.stack(post_tokens_id)
    # 将 label 中的列表转换为 PyTorch 张量
    label0 = torch.tensor(label0)
    label1 = torch.tensor(label1)
    label2 = torch.tensor(label2)
    label3 = torch.tensor(label3)
    polarities = torch.stack(polarities)

    # 返回包含所有转换后的张量的元组，表示一个完整的批次
    return post_tokens_id, label0, label1, label2, label3, polarities


def mlm_data_augmentation(posts, model_name='/nfs/huggingfacehub/models--bert-base-cased', mask_rate=0.15):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    inputs = tokenizer(posts, return_tensors="pt", max_length=512, truncation=True, padding='max_length',
                       pad_to_max_length=True)
    # 根据 mask_rat e在 inputs.input_ids 的每个位置上生成遮蔽的布尔索引
    mask_indices = torch.bernoulli(torch.full(inputs.input_ids.shape, mask_rate)).bool()
    # 将选中的位置替换为 BERT 的遮蔽符号 [MASK] 的 ID
    inputs.input_ids[mask_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits

    # 遍历每个文本
    for i in range(inputs.input_ids.shape[0]):
        # 找到每个文本中被掩码的位置
        masked_positions = mask_indices[i].nonzero(as_tuple=False).squeeze()
        # 为每个掩码位置选择最有可能的 Token ID
        predicted_token_ids = predictions[i, masked_positions].argmax(dim=-1)
        # 用预测的 Token ID 替换原来的掩码 Token ID
        inputs.input_ids[i, masked_positions] = predicted_token_ids
    # 将 Token IDs 解码为文本
    new_posts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in inputs.input_ids]
    return new_posts


def synonym_replacement(text, n):
    aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.6)
    augmented_text = aug.augment(text, n=n)
    return augmented_text[0] if augmented_text else text


def random_insertion(text, n):
    aug = naw.ContextualWordEmbsAug(model_path='/nfs/huggingfacehub/models--bert-base-uncased',
                                    action="insert",
                                    aug_p=0.2)
    augmented_text = aug.augment(text, n=n)
    return augmented_text[0] if augmented_text else text


# 随机交换
def random_swap(text, n):
    aug = naw.RandomWordAug(action="swap", aug_p=0.2)
    augmented_text = aug.augment(text, n=n)
    return augmented_text[0] if augmented_text else text


# 随机删除
def random_deletion(text, n):
    aug = naw.RandomWordAug(action="delete", aug_p=0.2)
    augmented_text = aug.augment(text, n=n)
    return augmented_text[0] if augmented_text else text


def counter(t, I, E, S, N, T, F, P, J, total):
    if 'I' in t:
        I += 1
    if 'E' in t:
        E += 1
    if 'S' in t:
        S += 1
    if 'N' in t:
        N += 1
    if 'T' in t:
        T += 1
    if 'F' in t:
        F += 1
    if 'P' in t:
        P += 1
    if 'J' in t:
        J += 1
    total += 1
    return I, E, S, N, T, F, P, J, total
