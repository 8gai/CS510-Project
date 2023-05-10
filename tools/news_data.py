import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertAdam, BertForPreTraining, BertForMaskedLM

from datasets import TextDataSet
from bert import SimpleBert
from server import Log
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(file):
    data = []
    label = {'POLITICS':0, 'WELLNESS':1, 'ENTERTAINMENT':2, 'TRAVEL':3, 'STYLE & BEAUTY':4, 'PARENTING':5,
             'HEALTHY LIVING':6, 'QUEER VOICES':7, 'FOOD & DRINK':8, 'BUSINESS':9, 'COMEDY':10, 'SPORTS':11,
             'BLACK VOICES':12, 'HOME & LIVING':13, 'PARENTS':14}
    total_label = 15
    with open(file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            item = json.loads(line)
            category = item["category"]
            if category not in label:
                continue
            else:
                the_label = label[category]
            sentence = item["headline"].strip() + ". " + item["short_description"].strip()
            data.append([sentence, the_label])
    return data, label, total_label


def get_token(data_list, language='english'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' if language == "english" else 'bert-base-chinese')
    max_len = 350
    token_list = []
    index_list = []
    mask_list = []
    label_list = []
    length_list = []
    data_index_path = 'archive/index_short.txt'
    data_mask_path = 'archive/mask_short.txt'
    data_label_path = 'archive/label_short.txt'
    for i, item in enumerate(data_list):
        tokens = tokenizer.tokenize(item[0])
        tokens_len = len(tokens)
        if tokens_len > max_len-2:
            del tokens[max_len-2:]
            tokens_len = max_len-2
        tokens.insert(0, "[CLS]")
        tokens.append("[SEP]")
        tokens_len += 2
        pad_num = max_len - tokens_len
        tokens.extend(["[PAD]"] * pad_num)
        mask = [1] * tokens_len + [0] * pad_num
        index = tokenizer.convert_tokens_to_ids(tokens)
        token_list.append(tokens)
        index_list.append(index)
        mask_list.append(mask)
        label_list.append(item[1])
        length_list.append(tokens_len)
        if (i + 1) % 10000 == 0:
            print("Processed Data {} / {}".format(i + 1, len(data_list)))

    if data_index_path is not None:
        with open(data_index_path, "w", encoding="UTF-8") as fp:
            for item in index_list:
                fp.write("{}\n".format(item))

    if data_mask_path is not None:
        with open(data_mask_path, "w", encoding="UTF-8") as fp:
            for item in mask_list:
                fp.write("{}\n".format(item))

    if data_label_path is not None:
        with open(data_label_path, "w", encoding="UTF-8") as fp:
            for item in label_list:
                fp.write("{}\n".format(item))
    return token_list, index_list, mask_list, label_list, length_list

def f1_count(tf_matrix, label_count, prediction_count, lg):
    num_class = label_count.shape[0]
    lg.log("TF MATRIX TABLE\n{}".format(tf_matrix))
    tp = torch.zeros([num_class])
    fp = torch.zeros([num_class])
    fn = torch.zeros([num_class])
    for i in range(num_class):
        tp[i] = tf_matrix[i][i]
        fp[i] = prediction_count[i] - tp[i]
        fn[i] = label_count[i] - tp[i]
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    macro_p = p.mean()
    macro_r = r.mean()
    macro_f1 = (2 * macro_p * macro_r) / (macro_p + macro_r)
    micro_p = tp.mean() / (tp.mean() + fp.mean())
    micro_r = tp.mean() / (tp.mean() + fn.mean())
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r)
    return macro_f1.item(), micro_f1.item()

def train():
    task_name = 'CS510'
    torch.cuda.empty_cache()
    lg = Log("CS510")
    # load training data and indexing texts
    lg.log("Indexing Training Data......")

    train_set = torch.load('archive/train_set_short.pt')

    trainloader = DataLoader(
            train_set,
            batch_size=16, shuffle=True)
    t_batch = len(trainloader)
    num_class = 15
    lg.log("Index Training Data Done.")

    # prepare BERT model and set hyper params
    model = SimpleBert(350, num_class, language='english').to(device)
    lg.log("choosing BERT + Linear model.")
    
    model.train()

    init_epoch = 0
    t_epoch = 4
    lr = 2e-5
    t_total = t_batch * t_epoch
    warmup = 0.1

    criterion = nn.CrossEntropyLoss()
    optimizer = BertAdam(model.parameters(), lr=lr, t_total=t_total, warmup=warmup)
    state_path = None
    if state_path is not None:
        init_state = torch.load(state_path)
        model.load_state_dict(init_state['state_dict'])
        optimizer.load_state_dict(init_state['optimizer'])
        init_epoch = init_state['epoch'] + 1
        lg.log("Read model checkpoint in epoch {}. Training will be initiated from epoch {}".format(init_epoch,
                                                                                                    init_epoch + 1))

    lg.log("Model Config Done.")

    # fine tuning BERT
    lg.log("Start Training.")
    start_time = time.time()
    last_time = start_time
    for epoch in range(init_epoch, t_epoch):
        batch_num = 0
        total_loss = 0.0
        for inputs, mask, label, length in trainloader:
            inputs = inputs.to(device)
            mask = mask.to(device)
            label = label.to(device)
            length = length.to(device)
            output = model(inputs, mask) 
            # N * output_size (after softmax, represent probability)  eg. N * 2
            loss = criterion(output, label)
            if (batch_num + 1) % 50 == 0 or (batch_num + 1) == t_batch:
                lg.log("epoch {}/{}, batch {}/{}, loss = {:.6f}".format(epoch + 1, t_epoch, batch_num + 1, t_batch,
                                                                        loss.item()))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_num += 1

        this_time = time.time()
        lg.log(
            "epoch {}/{}, training done. Average loss = {:.6f}, Time Elapse this epoch : {}".format(epoch + 1, t_epoch,
                                                                                                    total_loss / t_batch,
                                                                                                    time.strftime(
                                                                                                        "%H:%M:%S",
                                                                                                        time.gmtime(
                                                                                                            this_time - last_time))),
            message=False)
        cur_state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(cur_state, "checkpoint/{}_TRAINING_EPOCH_{}.pb".format(task_name, epoch))
        lg.log("epoch {}/{}, model checkpoint saved.".format(epoch + 1, t_epoch))
        last_time = time.time()

    lg.log("Saving Model......")
    torch.save(model.state_dict(), "checkpoint/{}.pb".format(task_name))
    lg.log("Model saved.")
    final_time = time.time()
    lg.log("Training Done. Time elapsed: {}".format(time.strftime("%H:%M:%S", time.gmtime(final_time - start_time))),
           message=False)
    lg.writelog()
    

def evaluate():
    task_name = 'CS510_eval'
    torch.cuda.empty_cache()
    lg = Log(task_name)
    # load testing data and indexing texts
    lg.log("Indexing Testing Data......")

    val_set = torch.load('archive/val_set_short.pt')

    testloader = DataLoader(
            val_set,
            batch_size=24, shuffle=True)
    t_batch = len(testloader)
    num_class = 15
    lg.log("Index Testing Data Done.")

    # prepare BERT model and set hyper params
    lg.log("Model Config......")
    model = SimpleBert(350, num_class, language='english').to(device)
    lg.log("choosing BERT + Linear model.")
    model_path = 'checkpoint/CS510.pb'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.CrossEntropyLoss()

    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size(), ':', parameters)

    # evaluate
    lg.log("Testing......")
    val_loss = 0.0
    val_total = 0
    val_cor = 0
    tf_matrix = torch.zeros([num_class, num_class])
    label_count = torch.zeros([num_class])
    predict_count = torch.zeros([num_class])
    batch_num = 0
    with torch.no_grad():
        for inputs, mask, label, length in testloader:
            inputs = inputs.to(device)
            mask = mask.to(device)
            label = label.to(device)
            length = length.to(device)
            # output = model(inputs, mask) if model_name in ["bert_linear", "bert_lstm"] else \
            #     (model(inputs) if model_name == "textcnn" else model(inputs, length))
           
            output = model(inputs, mask)
           
            # N * output_size (after softmax, represent probability)  eg. N * 2
            loss = criterion(output, label)
            val_loss += loss.item()

            prediction = output.argmax(dim=-1)
            answer = label.view(-1)
            val_total += prediction.shape[0]
            val_cor += prediction[prediction == answer].shape[0]
            for i in range(prediction.shape[0]):
                tf_matrix[answer[i]][prediction[i]] += 1
                label_count[answer[i]] += 1
                predict_count[answer[i]] += 1
            if (batch_num + 1) % 50 == 0 or (batch_num + 1) == t_batch:
                lg.log("Testing {} / {} done.".format(batch_num + 1, t_batch))
            batch_num += 1

    val_loss = val_loss / t_batch
    acc = val_cor / val_total
    macro_f1, micro_f1 = f1_count(tf_matrix, label_count, predict_count, lg)
    lg.log("Test Result: {} / {} correct, {} accuracy, {} average loss, {} macro_f1, {} micro_f1".format(val_cor,
                                                                                                         val_total, acc,
                                                                                                         val_loss,
                                                                                                         macro_f1,
                                                                                                         micro_f1),
           message=False)

    lg.writelog()

if __name__ == '__main__':
    evaluate()
    


