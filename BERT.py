#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd 
import numpy as np 
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report


# In[21]:


pd.set_option('display.max_columns', None)
df = pd.read_csv("smalldataset.csv")
print(df.head())


# In[22]:


from collections import Counter
print(Counter(df['label'].values))


# In[23]:


df_fake = df[df['label'] == 'fake']
df_real = df[df['label'] == 'real']
df_real = df_real.sample(n=len(df_fake))
df = df_real.append(df_fake)
df = df.sample(frac=1, random_state = 24).reset_index(drop=True)
print(Counter(df['label'].values))


# In[24]:


train_data = df.head(20)
test_data = df.tail(20)


# In[25]:


train_data = [{'text': text, 'label': type_data } for text in list(train_data['text']) for type_data in list(train_data['label'])]
test_data = [{'text': text, 'label': type_data } for text in list(test_data['text']) for type_data in list(test_data['label'])]


# In[26]:


train_texts, train_labels = list(zip(*map(lambda d: (d['text'], d['label']), train_data)))
test_texts, test_labels = list(zip(*map(lambda d: (d['text'], d['label']), test_data)))


# In[27]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], train_texts))
test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], test_texts))
train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
test_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))
train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=512, truncating="post", padding="post", dtype="int")
test_tokens_ids = pad_sequences(test_tokens_ids, maxlen=512, truncating="post", padding="post", dtype="int")


# In[28]:


train_y = np.array(train_labels) == 'fake'
test_y = np.array(test_labels) == 'fake'


# In[29]:


class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokens, masks=None):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba


# In[30]:


train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]
train_masks_tensor = torch.tensor(train_masks)
test_masks_tensor = torch.tensor(test_masks)


# In[31]:


train_tokens_tensor = torch.tensor(train_tokens_ids)
train_y_tensor = torch.tensor(train_y.reshape(-1, 1)).float()
test_tokens_tensor = torch.tensor(test_tokens_ids)
test_y_tensor = torch.tensor(test_y.reshape(-1, 1)).float()


# In[32]:


BATCH_SIZE = 1
EPOCHS = 4

train_dataset =  torch.utils.data.TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
train_sampler =  torch.utils.data.RandomSampler(train_dataset)
train_dataloader =  torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
test_dataset =  torch.utils.data.TensorDataset(test_tokens_tensor, test_masks_tensor, test_y_tensor)
test_sampler =  torch.utils.data.SequentialSampler(test_dataset)
test_dataloader =  torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)


# In[33]:


bert_clf = BertBinaryClassifier()
optimizer = torch.optim.Adam(bert_clf.parameters(), lr=0.001)
for epoch_num in range(EPOCHS):
    bert_clf.train()
    train_loss = 0
    for step_num, batch_data in enumerate(train_dataloader):
        token_ids, masks, labels = tuple(t for t in batch_data)
        probas = bert_clf(token_ids, masks)
        loss_func = nn.BCELoss()
        batch_loss = loss_func(probas, labels)
        train_loss += batch_loss.item()
        bert_clf.zero_grad()
        batch_loss.backward()
        optimizer.step()
        print('Epoch: ', epoch_num + 1)
        print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(train_data) / BATCH_SIZE, train_loss / (step_num + 1)))


# In[34]:


bert_clf.eval()
bert_predicted = []
all_logits = []
with torch.no_grad():
    for step_num, batch_data in enumerate(test_dataloader):
      token_ids, masks, labels = tuple(t for t in batch_data)
      logits = bert_clf(token_ids, masks)
      loss_func = nn.BCELoss()
      loss = loss_func(logits, labels)
      numpy_logits = logits.cpu().detach().numpy()
        
      bert_predicted += list(numpy_logits[:, 0] > 0.5)
      all_logits += list(numpy_logits[:, 0])
        
print(classification_report(test_y, bert_predicted))


# In[ ]:




