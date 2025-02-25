import torch
from configs import args # 从configs模块导入args，包含了训练配置参数的对象或变量。
from src import train # 从src模块导入train函数，这个函数可能包含了训练模型的主要逻辑。
from src.utils import set_up_data_loader, set_random_seed
# 从src.utils模块导入set_up_data_loader和set_random_seed函数。这些函数可能用于设置数据加载器和设置随机种子以确保结果的可复现性。


# 调用set_random_seed函数，传入args.seed作为参数，设置随机种子
set_random_seed(args.seed)

# 使用 cuda进行运算。设置默认的张量类型为32位浮点数。
torch.set_default_tensor_type('torch.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#根据是否有可用的CUDA设备，设置计算设备为GPU或CPU。

train_loader, valid_loader, test_loader, num_train_optimization_steps = set_up_data_loader(args)
# 调用set_up_data_loader函数，传入args参数，获取训练、验证和测试数据加载器以及训练优化步数。
hyp_params = args # 将args赋值给hyp_params，包含了超参数。
hyp_params.device = device # 将计算设备赋值给hyp_params的device属性。
hyp_params.name = args.model + str(args.Use_EFusion) + str(args.Use_LFusion) + str(args.Use_Mag)
# 设置hyp_params的name属性，模型的名称。
# torch._C._jit_set_nesting_tensor(False)  # 禁用嵌套张量


#Python的常规用法，用于判断当前脚本是否作为主程序运行。
if __name__ == '__main__':
    train.train(hyp_params, train_loader, valid_loader, test_loader, num_train_optimization_steps)
# 如果脚本是主程序，调用train.train函数，传入超参数、数据加载器和训练优化步数，开始训练过程。
