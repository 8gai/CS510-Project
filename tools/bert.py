import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleBert(nn.Module):
    def __init__(self, seq_len, output_size, language="english", feat=0):
        super(SimpleBert, self).__init__()
        self.seq_len = seq_len
        self.output_size = output_size
        self.d_model = 768
        self.bert = BertModel.from_pretrained('bert-base-uncased' if language == "english" else 'bert-base-chinese')
        self.feat = feat
        # 12-layer, 768-hidden, 12-heads, 110M parameters
        # output: [batch_size, sequence_length, hidden_size]
        # choose the hidden of first token [CLS]
        if feat == 2:
            self.params = nn.ParameterList(
                [nn.Parameter(torch.tensor([1 / 12], device=device), requires_grad=True) for _ in range(12)])
        self.linear = nn.Linear(self.d_model, self.output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, mask):
        '''
        :param inputs: N * seq_len
        :param mask: N * seq_len
        :var bert_output: N * seq_len * hidden_size
        :return: N * output_size (after softmax, represent probability)
            classification logits
        '''
        bert_feature, _ = self.bert(inputs, attention_mask=mask)
        if self.feat == 0:
            bert_output = bert_feature[11]
            bert_output = bert_output[:, 0, :]
        elif self.feat == 1:
            temp = torch.cat([item[:, 0, :].unsqueeze(0) for item in bert_feature], dim=0)
            bert_output = torch.mean(temp, dim=0)
        elif self.feat == 2:
            temp = torch.cat([item[:, 0, :].unsqueeze(0) for item in bert_feature], dim=0)  # 12, N, hidden
            bert_output = self.params[0] * temp[0] + self.params[1] * temp[1] + self.params[2] * temp[2] + self.params[
                3] * temp[3] + self.params[4] * temp[4] + self.params[5] * temp[5] + self.params[6] * temp[6] + self.params[
                7] * temp[7] + self.params[8] * temp[8] + self.params[9] * temp[9] + self.params[10] * temp[10] + self.params[
                11] * temp[11]
        else:
            raise ValueError
        context = self.linear(bert_output)
        outputs = self.softmax(context)
        return outputs

