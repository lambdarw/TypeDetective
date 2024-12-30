import csv
import re
import pickle
from collections import Counter
import random
import os

token = ''  # '' or '<mask>'
MBTIs = ('INTJ', 'INTP', 'INFP', 'ENTP', 'ISTP', 'ISFP', 'ESTJ', 'ISTJ',
         'ESTP', 'ISFJ', 'ENFP', 'ESFP', 'ESFJ', 'ENFJ', 'INFJ', 'ENTJ')


def find_all_MBTIs(post, mbti):
    """
    在给定的文本 post 中找到所有匹配指定 MBTI 类型 mbti 的位置
    """
    return [(match.start(), match.end()) for match in re.finditer(mbti, post)]


def save_kaggle(data_kaggle, path):
    """
    对一些文本数据进行预处理，然后将处理后的数据保存到一个 pickle 文件中
    """
    # 创建一个空字典用来存放处理后的数据
    kaggle = {'annotations': [], 'posts_text': [], 'posts_num': []}
    for data in data_kaggle:
        # 使用正则表达式找到所有的帖子，并以'|||'为分割符进行分割
        raw_posts = patten_posts.findall(data[1])[0].split('|||')
        filter_posts = []
        # 删除文本中的链接（以 "http" 开头的部分）以及删除特定 MBTI 类型的内容
        for text in raw_posts:
            # 删除 http:/
            filter_text = ' '.join(filter(lambda x: x[:4] != 'http', text.split(' ')))
            if filter_text != '':
                # 删除 MBTI类型内容
                for MBTI in MBTIs:
                    mbti_idx_list = find_all_MBTIs(filter_text.lower(), MBTI.lower())
                    delete_idx = 0
                    for start, end in mbti_idx_list:
                        filter_text = filter_text[:start - delete_idx] + token + filter_text[end - delete_idx:]
                        delete_idx += end - start + len(token)

                # 将处理后的文本添加到 filter_posts 列表
                filter_posts.append(filter_text)
        # 将每个用户的类别、处理后的帖子列表及帖子数量分别添加到字典
        kaggle['annotations'].append(data[0])
        kaggle['posts_text'].append(filter_posts)
        kaggle['posts_num'].append(len(filter_posts))

    # 打印出部分处理后的数据以供检查
    # [['persona 5 !! type ze characters', "? mayb    1. i'm very straightforward and blunt with how i feel about anything. i'm oblivious to how others feel and sometimes this leads me to hurt others because i dwell in my own little...", 'Seems like it.', 'drank', 'troll :^)', ' Instinct. Switching to Mystic.  Sorry team.', "Anybody watching Love Live! Sunshine!! ? I'm really curious about their types, though we're only 3 episodes in so it may be a bit difficult...", "1. Fe?  1 1/2. Te.. 2. I think Ti? 3. Ne 4. wut... i guess you're trying to relate this to Si? lol 5. er not function related... or even personality related :p 6. Fi 7. Ne / lack of sensing......", 'no.  you could be fake.', 'heckkk yeahhh you mentioned some of my favorite anime youtubers!!  i would have met them yesterday at anime expo (hint: i went there yesterday...(july 1st)), but it was too crowded. :/ perhaps...', 'sure', 'Possibly', 'unknown = inaccurate   ', 'MEME. quite a rare personality type... way rarer than ..', 'You could be an ...', 's are nerds that barely have any acess to human emotions. the only people they care about is their anime waifus...', "i wish d.va was ... though she's most likely xsfp idk   do u play overwatch", 'I actually thought you were an ...', 's are such freaking trolls.... like just pls stop', "ur jokes aren't even funny...", 'leictreon ur avatar is really cute!!!  (for some reason i said accurate at first lol) -- (at above person)  ur a mystery', 'Man... This all seems really accurate! Props to ya! :D', 'yes u r', 'pfne', "Lever  (oh whoops... that's already been said. oh well >:D)", "THE LAST VID!!! That's on my liked videos playlist on youtube; I didn't think I would see it here!! Bossa Nova like that = perfection.   (Sorry I got a bit too excited lmao.)  Anyways, jazz is...", " I love the vibes and lyrics of this song. It's just so chill~", "Oh gosh, I'm such a Dreamer hehe. Maybe with a little mix of Idealist but mostly Dreamer :p", 'Agnostic theist born in a Christian fanily. :o', ' :p', 'Thanks for joining this topic! :o  Hmm, I relate to both of you, but I think I identify with you more.  I also took the quiz you suggested a billion of times because I tried to answer all of the...', 'i present to you an ', 'You see more . :o', ' :) Pretty noice', "Good thing you mentioned that. :P My thoughts do tend to be very personal, so I'm not always one to share different ideas with a bunch a random people. If I'm comfortable with an individual and I...", 'o-o-oh my god.... there was just one period in that sentence... my head is spinning  i think just had a gag reflex', 'i read turnip as trump lmao', 'Definitely inferior-Te. :p', "ennifer Maybe I could explain my intuition function without directly saying which statement applies to Ni or Ne. I'm pretty much just gonna ramble lol :p.   I am very open to possibilty;...", "Besides the stereotypes, I'm still not sure if I use Ni or Ne :/", 'That.... That confused me. :', '|quite accurate m8 :^)', "i thought i was but i just ??? don't know", 'screw pokemon and digimon BEYBLADE LET IT RIP', "i used to  wasn't it on cartoon network", 'u guys have taken over the thread', "Once Upon A Time  ennifer Thanks for explaining :) I don't think I should really rely on stereotypes; they will definitely confuse me when it comes to figuring out my mbti type. I should continue...", 'b-b-b-b-b-b-b-b-b-b-b-baka!!!!', ', of course :p', "Heh, ya never know :P  s are noted for being very creative and I'm just... not. s are called artistic, poetic, and all that jazz, but all I do is play video games, read, and watch TV. I...'"]]
    print(kaggle['posts_text'][:1])
    # 5205
    print(len(kaggle['annotations']))
    # Counter({50: 2915, 49: 738, 48: 339, 47: 216, 46: 153, 45: 95, 44: 84, 42: 67, 41: 58, 43: 56, 40: 41, 39: 39, 35: 33, 36: 31, 37: 29, 38: 29, 32: 22, 30: 20, 33: 18, 31: 16, 25: 16, 26: 15, 34: 14, 27: 12, 29: 12, 23: 11, 22: 11, 51: 11, 28: 10, 21: 10, 24: 10, 53: 7, 52: 5, 9: 5, 15: 5, 55: 5, 54: 4, 19: 4, 20: 4, 58: 4, 18: 4, 17: 3, 8: 3, 10: 3, 14: 3, 11: 2, 56: 2, 7: 2, 4: 2, 12: 2, 16: 2, 1: 1, 5: 1, 57: 1})
    print(Counter(kaggle['posts_num']))

    # 将处理后的数据保存到一个 '.pkl' 文件中（文件使用 pickle 格式进行序列化，这样就可以在以后重新加载这个文件，恢复原始的数据结构）
    with open(os.path.join('preprocessed/kaggle_test2', path + '.pkl'), 'wb') as f:
        pickle.dump(kaggle, f)


def data_preprocessing():
    """
    数据预处理
    从"MBTI_kaggle.csv"文件中加载数据，对数据进行随机洗牌，然后将数据分割成训练集、验证集和测试集，并分别保存到 pkl 文件中
    """
    with open('raw/MBTI_kaggle.csv', 'r', encoding='utf-8') as f:
        # 读取CSV文件，并将结果转换成列表形式，去掉标题行
        f_csv = list(csv.reader(f))[1:]
        # 对列表进行随机洗牌，以便数据分布均匀
        random.shuffle(f_csv)
        # 计算训练数据集的长度（占总数据的60%）
        train_len = int(0.6 * len(f_csv))
        # 计算验证数据集的长度（占总数据的20%）
        eval_len = int(0.2 * len(f_csv))
        # 根据计算出的长度，划分训练集、验证集和测试集
        train_kaggle, eval_kaggle, test_kaggle = f_csv[:train_len], f_csv[train_len:train_len + eval_len], f_csv[
                                                                                                           train_len + eval_len:]
        # 分别保存训练集、验证集和测试集到pkl文件中
        save_kaggle(train_kaggle, 'train')
        save_kaggle(eval_kaggle, 'eval')
        save_kaggle(test_kaggle, 'test')


if __name__ == '__main__':
    random.seed(0)
    patten_posts = re.compile(r'\"{0,1}\'{0,1}(.*)\'{0,1}\"{0,1}')
    data_preprocessing()
