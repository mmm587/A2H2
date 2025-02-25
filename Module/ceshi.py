import pickle

# 直接查看pkl文件内容
with open("datasets/ch-simsv2s.pkl", "rb") as f:
    data = pickle.load(f)
    # 打印训练集的详细信息
    print("\nTrain data details:")
    print(f"Type: {type(data['train'])}")
    if isinstance(data['train'], dict):
        print(f"Keys: {data['train'].keys()}")
        print(f"Sample value: {next(iter(data['train'].values()))}")