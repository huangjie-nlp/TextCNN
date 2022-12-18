class Config():
    def __init__(self):
        self.num_label = 7
        self.filter_size = [2, 3, 4]
        self.vocab_size = 1408
        self.emb_dim = 64
        self.filter_num = 128
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.train_file = "dataset/train_data.csv"
        self.dev_file = "dataset/dev_data.csv"
        self.test_file = "dataset/test_data.csv"
        self.schema = "dataset/level1_schema.json"
        self.vocab = "dataset/vocab.json"
        self.epochs = 100
        self.test_result = "test_result/test_result.csv"
        self.save_model = "checkpoint/textcnn_bs_{}_lr_{}.pt".format(self.batch_size, self.learning_rate)