## gpt2 进行文言文翻译
from bert_seq2seq import load_gpt
import torch
from tqdm import tqdm
import time
import  glob
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq import Tokenizer, load_chinese_base_vocab

vocab_path = "./state_dict/gpt2通用中文模型/vocab.txt"
model_path = "./state_dict/gpt2通用中文模型/pytorch_model.bin"
model_save_path = "./state_dict/gpt_ancient_trans_model.bin"
batch_size = 8
lr = 1e-5
word2idx = load_chinese_base_vocab(vocab_path)

def read_corpus():
    """
    读原始数据
    """
    src = []
    tgt = []
    data_path = glob.glob("./corpus/文言文翻译/*")
    for p in data_path:
        dir = p.split("/")[:-1]
        dir = "/".join(dir)
        # print(dir)
        name = p.split("/")[-1]
        if "翻译" in name:
            # 找到了一个翻译文件
            tgt_name = name
            src_name = name[:-2]
            with open(dir + "/" + src_name) as fs:
                lines = fs.readlines()
                for line in lines:
                    src.append(line.strip("\n").strip())

            with open(dir + "/" + tgt_name) as ft:
                lines = ft.readlines()
                for line in lines:
                    tgt.append(line.strip("\n").strip())

        else:
            pass

    return src, tgt

class SeqDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, sents_src, sents_tgt):
        ## 一般init函数是加载所有数据
        super(SeqDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

        self.idx2word = {k: v for v, k in word2idx.items()}
        self.tokenizer = Tokenizer(word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids, _ = self.tokenizer.encode(src, tgt, max_length=256)
        output = {
            "token_ids": token_ids,
        }
        return output

    def __len__(self):
        return len(self.sents_src)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])

    token_ids_padded = padding(token_ids, max_length)
    target_ids_padded = token_ids_padded.clone()
    target_ids_padded[target_ids_padded == 0] = -100

    return token_ids_padded, target_ids_padded


class Trainer:
    def __init__(self):
        # 加载数据
        self.sents_src, self.sents_tgt = read_corpus()

        # 判断是否有可用GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.gpt_model = load_gpt(word2idx)
        ## 加载预训练的模型参数～
        self.gpt_model.load_pretrain_params(model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.gpt_model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.gpt_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = SeqDataset(self.sents_src, self.sents_tgt)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        # 一个epoch的训练
        self.gpt_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)

    def save(self, save_path):
        """
        保存模型
        """
        self.gpt_model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        report_loss = 0
        start_time = time.time()  ## 得到当前时间
        step = 0
        # for token_ids, target_ids in tqdm(dataloader, position=0, leave=True):
        for token_ids, target_ids in dataloader:
            step += 1
            if step % 4000 == 0:
                self.gpt_model.eval()
                test_data = ["遂入颍川。", "会日暝，结陈相持。", "一言兴邦，斯近之矣。"]
                for text in test_data:
                    print(self.gpt_model.sample_generate(text, add_eos=True))
                self.gpt_model.train()
                print("report loss is " + str(report_loss))
                report_loss = 0.0

            # 因为传入了target标签，因此会计算loss并且返回
            loss, _ = self.gpt_model(token_ids,
                                                labels=target_ids,
                                                )
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()
            report_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch) + ". loss is " + str(total_loss) + ". spend time is " + str(spend_time))
        # 保存模型
        self.save(model_save_path)


if __name__ == '__main__':

    trainer = Trainer()
    train_epoches = 100
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
