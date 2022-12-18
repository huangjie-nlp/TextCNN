from framework.framework import Framework
from config.config import Config
import torch
import numpy as np
import pandas as pd


seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if __name__ == '__main__':
    config = Config()
    print(config.train_file)
    fw = Framework(config)
    fw.train()
    print("="*50+"test"+"="*50)
    precision, recall, f1_score, accuracy, predict, target, sentence= fw.test()
    print('recall:{:5.4f} precision:{:5.4f} f1_score:{:5.4f} accuracy:{:5.4f}'.format(recall, precision, f1_score, accuracy))
    pd.DataFrame({"sentence": sentence, "label": target, "predict": predict}).to_csv(config.test_result)

