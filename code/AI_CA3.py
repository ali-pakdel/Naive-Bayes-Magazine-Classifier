#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Classifier
# 
# ### Ali Pakdel Samadi
# 
# #### 810198368

# Read csv files.

# In[1]:


import pandas as pd
from hazm import *
import string
from collections import Counter
import numpy as np
import math

train_df = pd.read_csv("./Data/train.csv")
test_df = pd.read_csv("./Data/test.csv")

train_df['content'] = train_df.fillna({'content':''})
display(train_df)
display(test_df)


# Read stop words file and store them in a set.

# In[2]:


with open('stop_words.txt') as f:
    lines = [line.rstrip() for line in f]
stop_words = set(lines)


# # First Phase: Pre-processing Data
# 
# Pre-processing the dataframes and removing stop words from them.

# In[3]:


def remove_stopwords(df):
    i = 0
    while i < df.shape[0]:
        j = 0
        j_lim = len(df["content"].iloc[i])
        while j < j_lim:
            if df["content"].iloc[i][j] in stop_words:
                df["content"].iloc[i].pop(j)
                j -= 1
                j_lim -= 1
            j += 1
        i += 1
                
    return df
                

def pre_process(df):
    df = df.copy(deep = True)
    normalizer = Normalizer()
    df["content"] = df.apply(lambda line: normalizer.normalize(line["content"]), axis=1)
    df["content"] = df.apply(lambda line: word_tokenize(line["content"]), axis=1)
    df = remove_stopwords(df)
    lemmatizer = Lemmatizer()
    df["content"] = df.apply(lambda line: [lemmatizer.lemmatize(word) for word in line["content"]], axis=1)
    
    return df

pp_train = pre_process(train_df)
pp_test = pre_process(test_df)

display(pp_train)
display(pp_test)


# ### Question 1.
# 
# Example of lemmatization and stemming

# In[4]:


lemmatizer = Lemmatizer()
display(lemmatizer.lemmatize ('خوابیدم'))

stemmer = Stemmer()
display(stemmer.stem ( 'آرزو ها' ))


# # Second Phase: Problem Process

# In[5]:


def word_counts(row, counts):
    d = dict(Counter(row))
    for key,value in d.items():
        if key in counts.keys():
            counts[key] += value
        else:
            counts[key] = value
    return counts

def label_dict(df, label):
    df = df[df["label"] == label]
    counts = {}
    for i in range(df.shape[0]):
        counts = word_counts(df.iloc[i]["content"],counts)
    return counts

tech_dict = label_dict(pp_train, "علم و تکنولوژی")
art_dict = label_dict(pp_train, "هنر و سینما")
game_dict = label_dict(pp_train, "بازی ویدیویی")
health_dict = label_dict(pp_train, "سلامت و زیبایی")
del health_dict['شد#شو']
del health_dict['کرد#کن']

tech_sums = sum(tech_dict.values())
art_sums = sum(art_dict.values())
game_sums = sum(game_dict.values())
health_sums = sum(health_dict.values())

prior_tech = sum(pp_train["label"] == "علم و تکنولوژی") / len(pp_train)
prior_art = sum(pp_train["label"] == "هنر و سینما") / len(pp_train)
prior_game = sum(pp_train["label"] == "بازی ویدیویی") / len(pp_train)
prior_health = sum(pp_train["label"] == "سلامت و زیبایی") / len(pp_train)


# In[6]:


def calculate_prob(word, count_label):
    if word in count_label:
        return count_label[word]
    else:
        count_label[word] = 0
        return 0
        
def NBC():
    predict = []
    for i in range(pp_test.shape[0]):
        probs = [[prior_tech, "علم و تکنولوژی"], [prior_art, "هنر و سینما"], [prior_game, "بازی ویدیویی"], [prior_health, "سلامت و زیبایی"]]
        for word in pp_test["content"].iloc[i]:
            
            prob_tech = calculate_prob(word, tech_dict)
            probs[0][0] *= (prob_tech) / tech_sums
                
            prob_art = calculate_prob(word, art_dict)
            probs[1][0] *= (prob_art) / art_sums
            
            prob_game = calculate_prob(word, game_dict)
            probs[2][0] *= (prob_game) / game_sums
            
            prob_health = calculate_prob(word, health_dict)
            probs[3][0] *= (prob_health) / health_sums

        probs = sorted(probs,key=lambda l:l[0], reverse=True)
        predict.append(probs[0][1])

    
    pp_test["predict"] = predict
    return pp_test
        
def smoothing_NBC():
    predict = []
    for i in range(pp_test.shape[0]):
        probs = [[prior_tech, "علم و تکنولوژی"], [prior_art, "هنر و سینما"], [prior_game, "بازی ویدیویی"], [prior_health, "سلامت و زیبایی"]]
        for word in pp_test["content"].iloc[i]:
            
            prob_tech = calculate_prob(word, tech_dict)
            probs[0][0] += math.log((prob_tech + 1) / tech_sums)
                
            prob_art = calculate_prob(word, art_dict)
            probs[1][0] += math.log((prob_art + 1) / art_sums)
            
            prob_game = calculate_prob(word, game_dict)
            probs[2][0] += math.log((prob_game + 1) / game_sums)
            
            prob_health = calculate_prob(word, health_dict)
            probs[3][0] += math.log((prob_health + 1) / health_sums)

        probs = sorted(probs,key=lambda l:l[0], reverse=True)
        predict.append(probs[0][1])
    
    pp_test["predict"] = predict
    return pp_test


# ## Verification
# 
# ### Question 6.

# In[7]:


from matplotlib import pyplot as plt
from operator import itemgetter

labels = ['علم و تکنولوژی', 'هنر و سینما', 'بازی ویدیویی', 'سلامت و زیبایی']
count_labels = [tech_dict, art_dict, game_dict, health_dict]

for i in range(4):
    items = dict(sorted(count_labels[i].items(), key = itemgetter(1), reverse = True)[0:5])
    keys = items.keys()
    values = items.values()
    plt.bar(keys, values, color = 'green')
    plt.title(labels[i])
    plt.show()


# # Third Phase: Evaluation

# In[8]:


def check_corrects(df):
    df['correct'] = np.where(df['label'] == df['predict'] , 'T', 'F')

def accuracy(df):   
    count = (df['correct'] == 'T').sum()
    return count / len(df)
    
def precision(df):
    precision_lables = {}
    for l in labels:
        corr_det_class = 0
        all_det_class = 0
        for i in range(len(df["label"])):
            if df["predict"][i] == l:
                all_det_class += 1
                if df["correct"][i] == 'T':
                    corr_det_class += 1
        precision_lables[l] = corr_det_class / all_det_class
    return precision_lables

def recall(df):
    recall_lables = {}
    for l in labels:
        corr_det_class = 0
        total_class = 0
        for i in range(len(df["label"])):
            if df["predict"][i] == l and df["correct"][i] == 'T':
                corr_det_class += 1
            if df["label"][i] == l:
                total_class += 1
        recall_lables[l] = corr_det_class / total_class
    return recall_lables
    
def F1(df):
    p = precision(df)
    r = recall(df)
    f1 = {}
    for l in labels:
        f1[l] = 2 * ((p[l] * r[l]) / (p[l] + r[l]))
    return f1


# In[9]:


def macro(df):
    data = F1(df)
    s = sum(data.values())
    return s / len(data)

def weighted(df):
    data = F1(df)
    s = 0
    counts = 0
    for l in labels:
        w = len(df[df['label'] == l])
        s += w * data[l]
        counts += w
    return s / counts

def micro(df):
    return accuracy(df)


# In[10]:


def print_outputs(df):
    check_corrects(df)
    
    precision_dict = precision(df)
    recall_dict = recall(df)
    F1_dict = F1(df)

    print("Precision")
    print("\tScience Technology: ", round(precision_dict['علم و تکنولوژی'] * 100,2))
    print("\tArt Cinema: ", round(precision_dict['هنر و سینما'] * 100, 2))
    print("\tVideo Game: ", round(precision_dict['بازی ویدیویی'] * 100,2))
    print("\tHealth Beauty: ", round(precision_dict['سلامت و زیبایی'] *100,2))

    print("Recall")
    print("\tScience Technology: ", round(recall_dict['علم و تکنولوژی'] *100,2))
    print("\tArt Cinema: ", round(recall_dict['هنر و سینما'] * 100,2))
    print("\tVideo Game: ", round(recall_dict['بازی ویدیویی'] *100,2))
    print("\tHealth Beauty: ", round(recall_dict['سلامت و زیبایی'] *100,2))

    print("F1-Score")
    print("\tScience Technology: ", round(F1_dict['علم و تکنولوژی'] *100,2))
    print("\tArt Cinema: ", round(F1_dict['هنر و سینما'] *100,2))
    print("\tVideo Game: ", round(F1_dict['بازی ویدیویی'] *100,2))
    print("\tHealth Beauty: ", round(F1_dict['سلامت و زیبایی'] *100,2))

    print("\nAccuracy: ", round(accuracy(df) * 100, 2))
    print("Marco Avg: ", round(macro(df) * 100, 2))
    print("Micro Avg: ", round(micro(df) * 100, 2))
    print("Weighted Avg: ", round(weighted(df) * 100, 2))


# ### Question 10.
# #### A) With Additive Smoothing

# In[11]:


print_outputs(smoothing_NBC())


# #### B) Without Additive Smoothing

# In[12]:


print_outputs(NBC())


# ### Question 12.

# In[13]:


df = smoothing_NBC()
false = df[df["label"] != df["predict"]]
idx = false.head().index.values.tolist()
for i in idx:
    print("Content: ", test_df["content"][i])
    print("Main Label: ", false["label"][i])
    print("Predicted Label: ", false["predict"][i])
    print("-------------------")
    print("-------------------")

