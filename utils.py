import numpy as np # linear algebra
import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def build_word_vocab(corpus,pad_index=0,pad_token="<PAD>",unk_index=1,unk_token="<UNK>"):  
# Builds word level vocabulary dictionary from given corpus    

  word_to_index = {}
  index_to_word = {}
  word_to_index[pad_token] = pad_index  
  word_to_index[unk_token] = unk_index
  index_to_word[pad_index] = pad_token
  index_to_word[unk_index] = unk_token   
  index = 0
  if index == pad_index:
    index += 1
    if index == unk_index:
      index += 1 
  if index == unk_index:
    index += 1
    if index == pad_index:
      index += 1 
  for string in corpus:
    # tokens = word_tokenize(str(string).lower())
    tokens = string
    for token in tokens:
      if token not in word_to_index:      
        word_to_index[token] = index
        index_to_word[index] = token
        index += 1
        if index == pad_index:
          index += 1
          if index == unk_index:
            index += 1 
        if index == unk_index:
          index += 1
          if index == pad_index:
            index += 1 
  return word_to_index,index_to_word

def build_char_vocab(corpus,pad_index=0,pad_token="<PAD>",unk_index=1,unk_token="<UNK>"):  
# Builds character level vocabulary dictionary from given corpus      
    
    char_to_index = {}
    index_to_char = {}
    char_to_index[pad_token] = pad_index
    char_to_index[unk_token] = unk_index    
    index = 0
    if index == pad_index:
        index += 1
        if index == unk_index:
            index += 1 
    if index == unk_index:
        index += 1
        if index == pad_index:
            index += 1 
    for string in corpus:
        tokens = string
        for token in tokens:
              for char in token:
                if char not in char_to_index:      
                    char_to_index[char] = index
                    index_to_char[index] = char
                    index += 1
                    if index == pad_index:
                        index += 1
                        if index == unk_index:
                            index += 1 
                    if index == unk_index:
                        index += 1
                        if index == pad_index:
                            index += 1 
    return char_to_index,index_to_char




def to_padded_list_word(sentences,word_to_index,labels=[],pos=[],pad_cat=0,pad_token='<PAD>',
                        unk_token='<UNK>',pos_pad=0,max_seq_len=1000):
# Creates a 2d word level padded sequence from a given list of tokenized sentences. This also creates
# masks to be propagated through the neural net. Masks are important to avoid processing the pads in 
# a padded sequence, and helps in stopping backprop loss from pads. This code is NOT optimized for 
# performance.
    
    seq_len = 0
    for tokens in sentences:
        if len(tokens) > seq_len:
            seq_len = len(tokens)
    # print(seq_len)
    if max_seq_len < seq_len:
        seq_len = max_seq_len

    pad_index = word_to_index[pad_token]
    unk_index = word_to_index[unk_token]
    sents = []
    labels_sents = []
    pos_sents = []
    mask_sents = []
    p = 0
    for tokens in sentences:
        if len(tokens) < 0:
            continue    
        sent = []
        labels_words = []
        pos_words = []
        mask_words = []
        q = 0
        for word in tokens:
            try:    
                sent.append(word_to_index[word])
            except KeyError:
                sent.append(unk_index)
            if len(labels) > 0:
                labels_words.append(labels[p][q])
            if len(pos) > 0:
                pos_words.append(pos[p][q])          
            mask_words.append(True)
            if len(sent) == seq_len:
                sents.append(sent)
                labels_sents.append(labels_words)
                pos_sents.append(pos_words)
                mask_sents.append(mask_words)
                sent = []
                labels_words = []
                pos_words = []
            q += 1
        if len(sent) > 0:
            pad_len = seq_len-len(sent)
            sent.extend(pad_len*[pad_index])
            labels_words.extend(pad_len*[pad_cat])
            mask_words.extend(pad_len*[False])
            pos_words.extend(pad_len*[pos_pad])
            sents.append(sent)
            labels_sents.append(labels_words)
            mask_sents.append(mask_words)
            pos_sents.append(pos_words)
        p += 1  

    return(sents,labels_sents,mask_sents,pos_sents)



def to_padded_list_char(sentences,char_to_index,labels=[],pos=[],pad_cat=0,pad_token='<PAD>',
                        unk_token='<UNK>',pos_pad=0,max_seq_len=1000,max_word_len=15): 
# Creates a 3d character level padded sequence from a given list of tokenized sentences. This also creates
# masks to be propagated through the neural net. Masks are important to avoid processing the pads in 
# a padded sequence, and helps in stopping backprop loss from pads. This code is NOT optimized for 
# performance.
        
    seq_len = 0
    for string in sentences:
        if len(string) > seq_len:
            seq_len = len(string)
    if max_seq_len < seq_len:
        seq_len = max_seq_len
    max_len = []
    pad_index = char_to_index[pad_token]
    unk_index = char_to_index[unk_token]
    sents = []
    pos_sents = []
    labels_sents = []
    mask_sents = []
    p = 0
    for string in sentences:
        sent_len = seq_len    
        tokens = string
        sent = []
        pos_words = []
        labels_words = []
        mask_words = []
        q = 0
        for word in tokens:                      
            word_char = []
            for char in word:        
                try:                      
                    word_char.append(char_to_index[char])
                except KeyError:
                    word_char.append(unk_index) 
                if len(word_char) == max_word_len:
                    sent.append(word_char)
                    if len(labels) > 0:
                        labels_words.append(labels[p][q])
                    if len(pos) > 0:
                        pos_words.append(pos[p][q])         
                    mask_words.append(True)
                    word_char = []
            if len(sent) == seq_len:
                sents.append(sent)
                labels_sents.append(labels_words)
                pos_sents.append(pos_words)
                mask_sents.append(mask_words)
                sent = []
                labels_words = []
                pos_words = []
                mask_words = []
            if len(word_char) > 0:
                for i in range(max_word_len-len(word_char)):                      
                    word_char.append(pad_index)
                sent.append(word_char)
                if len(labels) > 0:
                    labels_words.append(labels[p][q])
                if len(pos) > 0:
                    pos_words.append(pos[p][q])        
                mask_words.append(True)
            q += 1

        if len(sent) > 0:
            sent_len = len(sent)
            for i in range(seq_len-sent_len):
                word_char = []
                for j in range(max_word_len):
                    word_char.append(pad_index)
                sent.append(word_char)
                labels_words.append(pad_cat)
                pos_words.append(pos_pad)
                mask_words.append(False)
            sents.append(sent)
            labels_sents.append(labels_words)            
            pos_sents.append(pos_words)
            mask_sents.append(mask_words)
        p += 1  
  
    return(sents,labels_sents,mask_sents,pos_sents)

def datagen_char(corpus,corpus_labels,corpus_pos,vocab,pad_cat,batch_size):
# Creates a character based generator to fed to the neural network

    df = pd.DataFrame({'sent':corpus})
    df['labels'] = corpus_labels
    if len(corpus_pos) > 0:
        df['pos'] = corpus_pos
    else:
        df['pos'] = corpus_labels
    while(1):
        data = df.sample(frac = 1)
        prev_index = 0
        for index in range(batch_size,len(data),batch_size):
            x = data.iloc[prev_index:index]['sent'] 
            y = data.iloc[prev_index:index]['labels'].to_list()
            pos = data.iloc[prev_index:index]['pos'].to_list()
            x,y,mask,pos = to_padded_list_char(x,vocab,y,pos,pad_cat)
            yield ({'input_ids': np.array(x,dtype=np.float64),
                    'attention_masks': np.array(mask),
                    'pos_tags': np.array(pos,dtype=np.float64)},
                   np.array(y,dtype=np.int32))
            prev_index = index
        if prev_index < len(data):
            x = data.iloc[prev_index:len(data)]['sent']
            y = data.iloc[prev_index:len(data)]['labels'].to_list()
            pos = data.iloc[prev_index:len(data)]['pos'].to_list()
            x,y,mask,pos = to_padded_list_char(x,vocab,y,pos,pad_cat)
            yield ({'input_ids': np.array(x,dtype=np.float64),
                    'attention_masks': np.array(mask),
                    'pos_tags': np.array(pos,dtype=np.float64)},
                   np.array(y,dtype=np.int32))


def datagen_word(corpus,corpus_labels,corpus_pos,vocab,pad_cat,batch_size):
# Creates a word based generator to fed to the neural network    

    df = pd.DataFrame({'sent':corpus})
    df['labels'] = corpus_labels
    df['pos'] = corpus_pos
    while(1):
        data = df.sample(frac = 1)
        prev_index = 0
        for index in range(batch_size,len(data),batch_size):
            x = data.iloc[prev_index:index]['sent'] 
            y = data.iloc[prev_index:index]['labels'].to_list()
            pos = data.iloc[prev_index:index]['pos'].to_list()
            x,y,mask,pos = to_padded_list_word(x,vocab,y,pos,pad_cat)
            yield ({'input_ids': np.array(x,dtype=np.float64),
                    'attention_masks': np.array(mask),
                    'pos_tags': np.array(pos,dtype=np.float64)},
                   np.array(y,dtype=np.int32))
            prev_index = index
        if prev_index < len(data):
            x = data.iloc[prev_index:len(data)]['sent']
            y = data.iloc[prev_index:len(data)]['labels'].to_list()
            pos = data.iloc[prev_index:len(data)]['pos'].to_list()
            x,y,mask,pos = to_padded_list_word(x,vocab,y,pos,pad_cat)
            yield ({'input_ids': np.array(x,dtype=np.float64),
                    'attention_masks': np.array(mask),
                    'pos_tags': np.array(pos,dtype=np.float64)},
                   np.array(y,dtype=np.int32))