import torch
import json
from dataloader.dataloader import MyDataset, collate_fn
from models.models import TextCNN
# from models.models import TextCnn as TextCNN
from torch.utils.data import DataLoader
from tqdm import tqdm

class Framework():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(self.config.schema, "r", encoding="utf-8") as f:
            self.id2label = json.load(f)[1]

    def train(self):
        dataset = MyDataset(self.config, self.config.train_file)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.config.batch_size,
                                collate_fn=collate_fn, pin_memory=True)

        dev_dataset = MyDataset(self.config, self.config.dev_file)
        dev_dataloader = DataLoader(dev_dataset, batch_size=1,
                                    collate_fn=collate_fn, pin_memory=True)

        model = TextCNN(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        loss_fn = torch.nn.BCELoss()
        best_F1 = 0
        accuracy = 0
        best_epoch = 0
        recall, precision = 0, 0
        global_step, global_loss = 0, 0
        for epoch in range(self.config.epochs):
            for data in tqdm(dataloader):
                logits = model(data)
                model.zero_grad()

                loss = loss_fn(logits, data["target"].to(self.device))
                global_loss += loss.item()

                loss.backward()
                optimizer.step()
                if global_step % 1000 == 0:
                    print("epoch: {} global_step: {} global_loss: {:5.4f}".format(epoch, global_step, global_loss))
                    global_loss = 0
                global_step += 1
            p, r, f, a = self.evaluate(model, dev_dataloader)
            if f > best_F1:
                best_F1 = f
                precision = p
                recall = r
                accuracy = a
                best_epoch = epoch
                print("save_model......")
                torch.save(model.state_dict(), self.config.save_model)
                print("best_epoch: {} precision: {:5.4f} recall: {:5.4f} best_f1: {:5.4f} accuracy: {:5.4f}".format(best_epoch, precision, recall, best_F1, accuracy))
                torch.save(model.state_dict(), self.config.save_model)
        print("best_epoch: {} precision: {:5.4f} recall: {:5.4f} best_f1: {:5.4f} accuracy: {:5.4f}".format(best_epoch, precision, recall, best_F1, accuracy))

    def evaluate(self, model, dataloader):
        model.eval()
        correct_num, predict_num, gold_num = 0, 0, 0
        acc = 0
        print("model is deving......")
        predict = []
        target = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                logits = model(data)
                logits = logits.cpu()[0].tolist()
                temp_predict = []
                taregt_data = data["label"][0]
                for k, v in enumerate(logits):
                    if v > 0.5:
                        temp_predict.append(self.id2label[str(k)])
                predict_num += len(temp_predict)
                if type(taregt_data) != float:
                    gold_num += len(taregt_data.split("、"))
                if len(temp_predict) != 0 and type(taregt_data) != float:
                    correct_num += len(set(temp_predict) & set(taregt_data.split("、")))
                    if len(set(temp_predict) - set(taregt_data.split("、"))) == 0 and len(set(taregt_data.split("、")) - set(temp_predict)) == 0:
                        acc += 1
                if temp_predict == [] and type == float:
                    acc += 1
                predict.append("、".join(temp_predict))
                target.append(taregt_data)

        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy = acc / len(target)
        print("precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f} accuracy:{:5.4f}".format(precision, recall, f1_score, accuracy))
        model.train()
        return precision, recall, f1_score, accuracy

    def test(self, file=None):
        model = TextCNN(self.config).to(self.device)
        model.load_state_dict(torch.load(self.config.save_model, map_location=self.device))
        model.eval()

        if file == None:
            test_file = self.config.test_file
        else:
            test_file = file
        dataset = MyDataset(self.config, test_file)
        dataloader = DataLoader(dataset, batch_size=1,
                                collate_fn=collate_fn, pin_memory=True)

        correct_num, predict_num, gold_num = 0, 0, 0
        acc = 0

        predict = []
        target = []
        sentence = []
        temp = ""
        with torch.no_grad():
            for k, data in tqdm(enumerate(dataloader)):
                if k == 0:
                    temp = data
                try:
                    logits = model(data)
                except:
                    print(temp)
                    print(data)
                    exit()
                logits = logits.cpu().tolist()[0]
                temp_predict = []
                taregt_data = data["label"][0]
                for k, v in enumerate(logits):
                    if v > 0.5:
                        temp_predict.append(self.id2label[str(k)])
                predict_num += len(temp_predict)
                if type(taregt_data) != float:
                    gold_num += len(taregt_data.split("、"))
                if len(temp_predict) != 0 and type(taregt_data) != float:
                    correct_num += len(set(temp_predict) & set(taregt_data.split("、")))
                    if len(set(temp_predict) - set(taregt_data.split("、"))) == 0 and len(set(taregt_data.split("、")) - set(temp_predict)) == 0:
                        acc += 1
                if temp_predict == [] and type == float:
                    acc += 1
                predict.append("、".join(temp_predict))
                target.append(taregt_data)
                sentence.append(data["sentence"][0])

        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy = acc / len(target)
        print("precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f} accuracy:{:5.4f}".format(precision, recall, f1_score, accuracy))
        return precision, recall, f1_score, accuracy, predict, target, sentence

