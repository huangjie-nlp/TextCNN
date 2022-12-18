
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.emb_dim)
        self.module_list = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=self.config.filter_num,
                                                    kernel_size=(kernel, self.config.emb_dim)) for kernel in self.config.filter_size])

        self.fc = nn.Linear(len(self.config.filter_size) * self.config.filter_num, self.config.num_label)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        """
        filter_num = out_channel
        """
        input_ids = data["input_ids"].to(self.device)

        # [batch_size, seq_len, emb_dim]
        emb = self.embedding(input_ids)
        # each cnn block: [batch_size, filter_num, seq_len - filter_sizes + 1, 1]
        input_cnn = [cnn(emb.unsqueeze(dim=1)) for cnn in self.module_list]

        activate = [F.relu(x) for x in input_cnn]

        # each pooled block: [batch_size, filter_num, 1, 1]
        pooled = [F.max_pool2d(x, kernel_size=(x.size(2), x.size(3))) for x in activate]
        # [batch_size, filter_num * 1 * 1]
        flatten = [x.view(x.size(0), -1) for x in pooled]

        # [batch_size, filter_num * len(filter_size)]
        x = torch.cat(flatten, 1)
        # [batch_size, num_label]
        output = self.fc(x)
        return F.sigmoid(output)

