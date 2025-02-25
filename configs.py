import argparse
import random

# 设计随机种子 或者输入random以随机的方式， 或者则指定[0, 9999]的数字作为随机种子
def seed(s):
    if isinstance(s, int):
        if 0 <= s <= 9999:
            return s
        else:
            raise argparse.ArgumentTypeError("Seed must be between 0 and 2**32 - 1. Received {0}".format(s))
    elif s == "random":
        return random.randint(0, 9999)
    else:
        raise argparse.ArgumentTypeError("Integer value is expected. Recieved {0}".format(s))


# 设置一些超参数 用于调参使用
# 创建 ArgumentParser() 对象   调用add_argument() 方法添加参数
parser = argparse.ArgumentParser(description='Multimodal Sentiment Analysis')
parser.add_argument('--model', type=str, default="xlnet-base-cased", help=' -- model name')
""" 指定数据集 """
parser.add_argument('--dataset', type=str, default='mosi', help='default: mosei/mosi/')
# 句子的最大长度s
parser.add_argument("--max_seq_length", type=int, default=50)
# 随机种子的选择
# parser.add_argument("--seed", type=seed, default="random")
# MOSI 随机种子 高的离谱
parser.add_argument("--seed", type=seed, default=6820)
# 学习率相关
parser.add_argument("--learning_rate", type=float, default=1e-5, help='1e-5')
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
# batch_size
# 总共的训练轮数
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
# 模型的超参数
parser.add_argument("--Use_EFusion", type=bool, default=True)
parser.add_argument("--Use_LFusion", type=bool, default=True)
parser.add_argument("--Use_Mag", type=bool, default=False)

# 添加序列模型选择参数
parser.add_argument("--sequence_model", type=str, default='transformer',
                   choices=['bilstm', 'gru', 'transformer', 'tcn'],
                   help='sequence model type for fusion')

parser.add_argument("--drop", type=float, default=0.1)  # 0.1 / 0.5
parser.add_argument("--scaling_factor", type=float, default=0.5)  # 0.5 / 1.0

args = parser.parse_args()  # 使用 parse_args() 解析添加的参数

if args.dataset == 'mosi':
    # MOSI SETTING
    ACOUSTIC_DIM = 74
    VISUAL_DIM = 47
    TEXT_DIM = 768
elif args.dataset == 'mosei':
    # MOSEI SETTING
    ACOUSTIC_DIM = 74
    VISUAL_DIM = 35
    TEXT_DIM = 768



print(f"Dataset: {args.dataset}")
print(f"Acoustic Dimension: {ACOUSTIC_DIM}, Visual Dimension: {VISUAL_DIM}, Text Dimension: {TEXT_DIM}")
print(f"Batch Sizes: Train={args.train_batch_size}, Dev={args.dev_batch_size}, Test={args.test_batch_size}")
