#!/usr/bin/env python
# coding: utf-8

# **Review**
# 
# Hi, my name is Dmitry and I will be reviewing your project.
#   
# You can find my comments in colored markdown cells:
#   
# <div class="alert alert-success">
#   If everything is done successfully.
# </div>
#   
# <div class="alert alert-warning">
#   If I have some (optional) suggestions, or questions to think about, or general comments.
# </div>
#   
# <div class="alert alert-danger">
#   If a section requires some corrections. Work can't be accepted with red comments.
# </div>
#   
# Please don't remove my comments, as it will make further review iterations much harder for me.
#   
# Feel free to reply to my comments or ask questions using the following template:
#   
# <div class="alert alert-info">
#   For your comments and questions.
# </div>
#   
# First of all, thank you for turning in the project! You did a pretty good job overall! There are no problems apart from the driver error for BERT, I left a comment below to help you with that. Let me know if you have questions!

# # Project Statement

# The Film Junky Union, a new edgy community for classic movie enthusiasts, is developing a system for filtering and categorizing movie reviews. The goal is to train a model to automatically detect negative reviews. You'll be using a dataset of IMBD movie reviews with polarity labelling to build a model for classifying positive and negative reviews. It will need to have an F1 score of at least 0.85.

# ## Initialization

# In[8]:


import math

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from tqdm.auto import tqdm


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
# the next line provides graphs of better quality on HiDPI screens
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

plt.style.use('seaborn')


# In[10]:


# this is to use progress_apply, read more at https://pypi.org/project/tqdm/#pandas-integration
tqdm.pandas()


# ## Load Data

# In[11]:


df_reviews = pd.read_csv('/datasets/imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})


# In[12]:


df_reviews.info()


# In[13]:


df_reviews.head(7)


# In[14]:


# CHECKING FOR NULLS
df_reviews.isna().sum()


# In[15]:


df_reviews.isna().sum().sum()


# In[16]:


# Finding the rows with nulls
df_reviews[df_reviews.isnull().any(axis=1)]


# In[17]:


# Checking the rows related to the nulls
df_reviews[df_reviews['original_title'] == 'Mila Ass Painting']


# In[18]:


df_reviews['average_rating'].describe()


# In[19]:


# Fill the nulls in the average_rating column with the median
df_reviews['average_rating'].fillna(value=df_reviews['average_rating'].median(), inplace=True)


# In[20]:


df_reviews['votes'].describe()


# In[21]:


# Fill the nulls of the votes column with the minimum since there are so few reviews
df_reviews['votes'].fillna(value=df_reviews['votes'].min(), inplace=True)


# In[22]:


df_reviews.sample(7)


# In[23]:


df_reviews[df_reviews.duplicated()]


# In[24]:


df_reviews['end_year'].value_counts()


# In[25]:


df_reviews['average_rating'].value_counts()


# In[26]:


df_reviews['average_rating'].unique()


# In[27]:


df_reviews['rating'].unique()


# In[28]:


df_reviews['ds_part'].unique()


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The data was loaded and inspected
# 
# </div>

# ## EDA

# Let's check the number of movies and reviews over years.

# In[22]:


fig, axs = plt.subplots(2, 1, figsize=(16, 8))

ax = axs[0]

dft1 = df_reviews[['tconst', 'start_year']].drop_duplicates() \
    ['start_year'].value_counts().sort_index()
dft1 = dft1.reindex(index=np.arange(dft1.index.min(), max(dft1.index.max(), 2021))).fillna(0)
dft1.plot(kind='bar', ax=ax)
ax.set_title('Number of Movies Over Years')

ax = axs[1]

dft2 = df_reviews.groupby(['start_year', 'pos'])['pos'].count().unstack()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)

dft2.plot(kind='bar', stacked=True, label='#reviews (neg, pos)', ax=ax)

dft2 = df_reviews['start_year'].value_counts().sort_index()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)
dft3 = (dft2/dft1).fillna(0)
axt = ax.twinx()
dft3.reset_index(drop=True).rolling(5).mean().plot(color='orange', label='reviews per movie (avg over 5 years)', ax=axt)

lines, labels = axt.get_legend_handles_labels()
ax.legend(lines, labels, loc='upper left')

ax.set_title('Number of Reviews Over Years')

fig.tight_layout()


# Let's check the distribution of number of reviews per movie with the exact counting and KDE (just to learn how it may differ from the exact counting)

# In[23]:


fig, axs = plt.subplots(1, 2, figsize=(16, 5))

ax = axs[0]
dft = df_reviews.groupby('tconst')['review'].count() \
    .value_counts() \
    .sort_index()
dft.plot.bar(ax=ax)
ax.set_title('Bar Plot of #Reviews Per Movie')

ax = axs[1]
dft = df_reviews.groupby('tconst')['review'].count()
sns.kdeplot(dft, ax=ax)
ax.set_title('KDE Plot of #Reviews Per Movie')

fig.tight_layout()


# The number of reviews per movie is left skewed, with most movies receiving few reviews as shown in the bar and KDE plots.

# In[24]:


df_reviews['pos'].value_counts()


# In[25]:


fig, axs = plt.subplots(1, 2, figsize=(12, 4))

ax = axs[0]
dft = df_reviews.query('ds_part == "train"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('The train set: distribution of ratings')

ax = axs[1]
dft = df_reviews.query('ds_part == "test"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('The test set: distribution of ratings')

fig.tight_layout()


# The distribution of ratings for the train and test sets are just about the same.

# Distribution of negative and positive reviews over the years for two parts of the dataset

# In[26]:


fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=(2, 1), height_ratios=(1, 1)))

ax = axs[0][0]

dft = df_reviews.query('ds_part == "train"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('The train set: number of reviews of different polarities per year')

ax = axs[0][1]

dft = df_reviews.query('ds_part == "train"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('The train set: distribution of different polarities per movie')

ax = axs[1][0]

dft = df_reviews.query('ds_part == "test"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('The test set: number of reviews of different polarities per year')

ax = axs[1][1]

dft = df_reviews.query('ds_part == "test"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('The test set: distribution of different polarities per movie')

fig.tight_layout()


# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# Would be nice if you added some comments to the plots
# 
# </div>

# <div class="alert alert-info">
#   Updated and included comments on the plots.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Great!
# 
# </div>

# In terms of polarities per year, the train and test sets are near identical, and thus a good way to split the data.

# In[27]:


print(df_reviews.groupby(['pos', 'rating'])['rating'].count())  # ['rating', 'review']


# In[106]:


#df_reviews.groupby(['pos', 'average_rating'])['average_rating'].count().plot()


# There is a direct relationship between the rating and pos values. Ratings below 5 have a negative (0) review and above 5 have a positive (1) review.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Right!
# 
# </div>

# ## Evaluation Procedure

# Composing an evaluation routine which can be used for all models in this project

# In[7]:


import sklearn.metrics as metrics

def evaluate_model(model, train_features, train_target, test_features, test_target):
    
    eval_stats = {}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6)) 
    
    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        
        eval_stats[type] = {}
    
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]
        
        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]
        
        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps
        
        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # F1 Score
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'F1 Score') 

        # ROC
        ax = axs[1]    
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')        
        ax.set_title(f'ROC Curve')
        
        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')        

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)
    
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1', 'APS', 'ROC AUC'))
    
    print(df_eval_stats)
    
    return


# ## Normalization

# We assume all models below accepts texts in lowercase and without any digits, punctuations marks etc.

# In[29]:


# import spacy
# import re

# nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# def clear_text(text):
#     pattern = r"[^a-zA-Z']"
#     cleared = re.sub(pattern, ' ', text)
#     cleared = " ".join(cleared.split())
#     doc = nlp(cleared.lower())  # .lower()

#     lemmas = []
#     for token in doc:
#         lemmas.append(token.lemma_)

#     return ' '.join(lemmas)


# In[29]:


import re

def normalize(text):
    # convert to lower case
    lower_string = text.lower()
 
    # remove numbers
    no_number_string = re.sub(r'\d+','',lower_string)
 
    # remove all punctuation except words and space
    no_punc_string = re.sub(r'[^\w\s]','', no_number_string) 
 
    # remove white spaces
    no_wspace_string = no_punc_string.strip()

    return no_wspace_string 


# In[30]:


normalize(df_reviews.loc[6435, 'review'])


# In[31]:


# for i in range(df_reviews.shape[0]):
#     df_reviews.loc[i, 'review_norm'] = clear_text(df_reviews.loc[i, 'review'])
df_reviews['review_norm'] = df_reviews['review'].apply(lambda x: normalize(x))


# In[32]:


df_reviews['review_norm']


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Texts were normalized successfully!
# 
# </div>

# ## Train / Test Split

# Luckily, the whole dataset is already divided into train/test one parts. The corresponding flag is 'ds_part'.

# In[33]:


df_reviews_train = df_reviews.query('ds_part == "train"').copy()
df_reviews_test = df_reviews.query('ds_part == "test"').copy()

train_target = df_reviews_train['pos']
test_target = df_reviews_test['pos']

print(df_reviews_train.shape)
print(df_reviews_test.shape)


# In[34]:


# df_reviews_train
print(train_target, test_target)


# In[35]:


# features
train_features_reviews = df_reviews_train['review_norm']
test_features_reviews = df_reviews_test['review_norm']
print(train_features_reviews.shape)
print(test_features_reviews.shape)


# In[37]:


# train_features_reviews


# ## Working with models

# ### Model 0 - Constant

# In[36]:


from sklearn.dummy import DummyClassifier


# In[39]:


dummy_clf = DummyClassifier(strategy="uniform")  # , constant=0)
dummy_clf.fit(df_reviews_train, train_target)
predict_dummy = dummy_clf.predict(df_reviews_test)


# In[40]:


metrics.f1_score(test_target, predict_dummy)


# In[41]:


evaluate_model(dummy_clf, train_features_reviews, train_target, test_features_reviews, test_target)


# With the dummy model, the F1 score is 0.50 to set a baseline for the other models.

# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# Yeah, not a very useful baseline :) Maybe `strategy='uniform'` would work better in this case as a baseline (predicting uniformly at random instead of a constant 0)
# 
# </div>

# <div class="alert alert-info">
#   Updated the strategy to uniform for dummy model and updated the baseline F1 score.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Awesome! Beating a baseline with F1 score of 0.5 is a lot more meaningful than beating a baseline with F1 score equal to 0, right? :)
# 
# </div>

# ### Model 1 - NLTK, TF-IDF and LR

# TF-IDF

# In[42]:


import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords


# In[43]:


# TF-IDF
stop_words = set(stopwords.words('english'))
count_tf_idf = TfidfVectorizer(stop_words=stop_words)
#tf_idf = count_tf_idf.fit_transform(corpus)


# In[44]:


# NLTK, TF-IDF
corpus = df_reviews['review_norm']


# In[45]:


tf_idf = count_tf_idf.fit_transform(corpus)
print(tf_idf.shape)  # (47331, 157856)


# In[46]:


# For section 8
tfidf_vectorizer_2 = count_tf_idf.fit(corpus)


# In[47]:


print(tfidf_vectorizer_2)


# In[48]:


# print(tfidf_vectorizer_2.transform(corpus))


# In[49]:


train_features_reviews.index


# In[50]:


train_features_1 = tf_idf[train_features_reviews.index]
print(train_features_1.shape)


# In[51]:


test_features_1 = tf_idf[test_features_reviews.index]
print(test_features_1.shape)


# In[52]:


model_1 = LogisticRegression(random_state=12345)  # solver='liblinear', 
model_1.fit(train_features_1, train_target)


# In[53]:


pred_test = model_1.predict(test_features_1)
print(pred_test)


# In[54]:


metrics.f1_score(test_target, pred_test)


# In[55]:


# evaluate_model(model_1b, tf_idf_train, train_target, tf_idf_test, test_target)


# In[56]:


evaluate_model(model_1, train_features_1, train_target, test_features_1, test_target)


# ### Model 3 - spaCy, TF-IDF and LR

# In[58]:


import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# In[59]:


def text_preprocessing_3(text):
    
    doc = nlp(text)
    #tokens = [token.lemma_ for token in doc if not token.is_stop]
    tokens = [token.lemma_ for token in doc]
    
    return ' '.join(tokens)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# It's nice that you tried lemmatization
# 
# </div>

# In[59]:


# Applying spaCy
corpus_spacy = df_reviews['review_norm'].apply(lambda x: text_preprocessing_3(x))


# In[60]:


corpus_spacy


# In[61]:


# TF-IDF
tf_idf = count_tf_idf.fit_transform(corpus_spacy)
print(tf_idf.shape)  #


# In[86]:


# For section 8
tfidf_vectorizer_3 = count_tf_idf.fit(corpus_spacy)


# In[63]:


# tfidf_vectorizer_3.transform(corpus_spacy)


# In[64]:


train_features_3 = tf_idf[train_features_reviews.index]
test_features_3 = tf_idf[test_features_reviews.index]
print(train_features_3.shape)  # (23796, 139742)
print(test_features_3.shape)  # (23535, 139742)


# In[65]:


model_3 = LogisticRegression(random_state=12345)  # solver='liblinear', 
model_3.fit(train_features_3, train_target)


# In[66]:


pred_3 = model_3.predict(test_features_3)
print(pred_3)


# In[67]:


metrics.f1_score(test_target, pred_3)


# In[68]:


evaluate_model(model_3, train_features_3, train_target, test_features_3, test_target)


# In[ ]:





# In[ ]:





# ### Model 4 - spaCy, TF-IDF and LGBMClassifier

# In[69]:


from lightgbm import LGBMClassifier


# In[70]:


df_reviews['review_norm']


# In[71]:


# Use variables already transformed by spaCy and TF-IDF
train_features_4 = train_features_3.copy()
test_features_4 = test_features_3.copy()
print(train_features_4.shape)
print(test_features_4.shape)


# In[87]:


# For section 8
tfidf_vectorizer_4 = count_tf_idf.fit(corpus_spacy)


# In[73]:


model_4 = LGBMClassifier()
model_4.fit(train_features_4, train_target)


# In[74]:


pred_4 = model_4.predict(test_features_4)
print(pred_3)


# In[75]:


metrics.f1_score(test_target, pred_4)


# In[76]:


evaluate_model(model_4, train_features_4, train_target, test_features_4, test_target)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Great, you vectorized text data using TF-IDF and trained a couple of different models using the resulting features
# 
# </div>

# ###  Model 9 - BERT

# In[77]:


import torch
import transformers


# In[78]:


tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
config = transformers.BertConfig.from_pretrained('bert-base-uncased')
model = transformers.BertModel.from_pretrained('bert-base-uncased')


# In[79]:


def BERT_text_to_embeddings(texts, max_length=512, batch_size=10, force_device=None, disable_progress_bar=False):
    
    ids_list = []
    attention_mask_list = []

    # text to padded ids of tokens along with their attention masks
    for input_text in texts.iloc[:batch_size]:
        ids = tokenizer.encode(input_text.lower(), add_special_tokens=True, truncation=True, max_length=max_length)
        padded = np.array(ids + [0]*(max_length - len(ids)))
        attention_mask = np.where(padded != 0, 1, 0)
        ids_list.append(padded)
        attention_mask_list.append(attention_mask)
    # <put your code here to create ids_list and attention_mask_list>
    
    if force_device is not None:
        device = torch.device(force_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.to(device)
    if not disable_progress_bar:
        print(f'Using the {device} device.')
    
    # gettings embeddings in batches

    embeddings = []

    for i in tqdm(range(math.ceil(len(ids_list)/batch_size)), disable=disable_progress_bar):
            
        ids_batch = torch.LongTensor(ids_list[batch_size*i:batch_size*(i+1)]).to(device)
        # <put your code here to create attention_mask_batch
        attention_mask_batch = torch.LongTensor(attention_mask_list[batch_size * i:batch_size * (i + 1)]).to(device)
            
        with torch.no_grad():            
            model.eval()
            batch_embeddings = model(input_ids=ids_batch, attention_mask=attention_mask_batch)   
        embeddings.append(batch_embeddings[0][:,0,:].detach().cpu().numpy())
        
    return np.concatenate(embeddings)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The code for generating BERT embeddings is correct
# 
# </div>

# In[55]:


# Attention! Running BERT for thousands of texts may take long run on CPU, at least several hours
# train_features_9 = BERT_text_to_embeddings(df_reviews_train['review_norm'], force_device='cuda')


# Kept getting the following error, not sure how to proceed:
# 
# RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx

# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# Yeah, if you're running on the platform, there is no GPU, so forcing device 'cuda' results in an error. Running it on CPU is also not an option, as it will take a very long time. You can try running this locally if you have an nvidia GPU and know how to install CUDA, or on google colab/kaggle notebooks which give you a notebook environment with a GPU and preinstalled CUDA for free with some limitations.
#     
# You can also skip training this model if you'd like, as this section is optional.
# 
# </div>

# <div class="alert alert-info">
#     I'll opt out of the BERT sections since I only work on a laptop for now by commenting them out, but good to know the code is correct for future use.
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Ok, no problem!
# 
# </div>

# In[ ]:


#print(df_reviews_train['review_norm'].shape)
#print(train_features_9.shape)
#print(train_target.shape)


# In[97]:


#print(df_reviews_train['review_norm'].shape)
#print(train_target)


# In[56]:


#model_9 = LogisticRegression(random_state=12345)
#model_9.fit(train_features_9, train_target)


# 

# In[ ]:


# if you have got the embeddings, it's advisable to save them to have them ready if 
# np.savez_compressed('features_9.npz', train_features_9=train_features_9, test_features_9=test_features_9)

# and load...
# with np.load('features_9.npz') as data:
#     train_features_9 = data['train_features_9']
#     test_features_9 = data['test_features_9']


# In[ ]:





# In[ ]:





# In[ ]:





# ## My Reviews

# In[81]:


# feel free to completely remove these reviews and try your models on your own reviews, those below are just examples

my_reviews = pd.DataFrame([
    'I did not simply like it, not my kind of movie.',
    'Well, I was bored and felt asleep in the middle of the movie.',
    'I was really fascinated with the movie',    
    'Even the actors looked really old and disinterested, and they got paid to be in the movie. What a soulless cash grab.',
    'I didn\'t expect the reboot to be so good! Writers really cared about the source material',
    'The movie had its upsides and downsides, but I feel like overall it\'s a decent flick. I could see myself going to see it again.',
    'What a rotten attempt at a comedy. Not a single joke lands, everyone acts annoying and loud, even kids won\'t like this!',
    'Launching on Netflix was a brave move & I really appreciate being able to binge on episode after episode, of this exciting intelligent new drama.'
], columns=['review'])

my_reviews['review_norm'] = my_reviews['review'].apply(lambda x: normalize(x)) # <put here the same normalization logic as for the main dataset>

# my_reviews['review_norm'] = my_reviews['review_norm'].apply(lambda x: text_preprocessing_3(x)) 

my_reviews


# In[103]:


pos = my_reviews.copy()
pos['pos'] = [0, 0, 1, 0, 1, 1, 0, 1]
pos


# ### Model 2

# In[84]:


texts = my_reviews['review_norm']

my_reviews_pred_prob = model_1.predict_proba(tfidf_vectorizer_2.transform(texts))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# 6 of 8 correct

# ### Model 3

# In[90]:


texts = my_reviews['review_norm']

my_reviews_pred_prob = model_3.predict_proba(tfidf_vectorizer_3.transform(texts.apply(lambda x: text_preprocessing_3(x))))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# 5 of 8 correct

# ### Model 4

# In[89]:


texts = my_reviews['review_norm']

tfidf_vectorizer_4 = tfidf_vectorizer_3
my_reviews_pred_prob = model_4.predict_proba(tfidf_vectorizer_4.transform(texts.apply(lambda x: text_preprocessing_3(x))))[:, 1]

for i, review in enumerate(texts.str.slice(0, 100)):
    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# 8 of 8 correct

# ### Model 9

# In[57]:


#texts = my_reviews['review_norm']

#my_reviews_features_9 = BERT_text_to_embeddings(texts, disable_progress_bar=True)

#my_reviews_pred_prob = model_9.predict_proba(my_reviews_features_9)[:, 1]

#for i, review in enumerate(texts.str.slice(0, 100)):
#    print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The models were applied to new reviews correctly
# 
# </div>

# ## Conclusions

# Note: Unfortunately, I could not get the Bert model (model 9) working correctly since it kept giving an error related to an NVIDIA driver. Any guidance would be appreciated.

# Not including the BERT model, the Film Junky Union can use model 1, 3, or 4 to help predict positive and negative reviews since they achieved a higher F1 score of 0.85. These three models were relatively close in F1 scores.
# 
# On the My Reviews section, only model 4 was able to get all eight of the reviews correctly assigned as positive or negative.
# 
# With these points in mind, model 4 could be the best model to use for Film Junky Union.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Nice summary!
# 
# </div>

# <div class="alert alert-info">
#     Appreciate you taking the time to give your suggestions and feedback!
#     Thank you Dmitry! 
# </div>

# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# You're welcome! The project is now accepted. Good luck on the next sprint!
# 
# </div>

# # Checklist

# - [x]  Notebook was opened
# - [ ]  The text data is loaded and pre-processed for vectorization
# - [ ]  The text data is transformed to vectors
# - [ ]  Models are trained and tested
# - [ ]  The metric's threshold is reached
# - [ ]  All the code cells are arranged in the order of their execution
# - [ ]  All the code cells can be executed without errors
# - [ ]  There are conclusions

# In[ ]:




