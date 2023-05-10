import torch
from pytorch_pretrained_bert import BertTokenizer

from bert import SimpleBert

from location_date import extract_location

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'checkpoint/CS510.pb'

def construct_model():
    num_class = 15
    model = SimpleBert(350, num_class, language='english').to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def tokenize_text(text):
    max_len = 350
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
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
    return tokens, index, mask

def get_category_prob(model, text):
    tokens, index, mask = tokenize_text(text)
    t_index = torch.LongTensor(index).unsqueeze(0).to(device)
    t_mask = torch.LongTensor(mask).unsqueeze(0).to(device)
    output = model(t_index, t_mask).squeeze(0)
    labels = ['POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY', 'PARENTING',
             'HEALTHY LIVING', 'QUEER VOICES', 'FOOD & DRINK', 'BUSINESS', 'COMEDY', 'SPORTS',
             'BLACK VOICES', 'HOME & LIVING', 'PARENTS']
    idx = torch.argsort(output, descending=True)
    ret = []
    for i in range(15):
        ret.append([labels[idx[i]], output[idx[i]].item()])
    return ret

def get_location(url):
    return extract_location(url)