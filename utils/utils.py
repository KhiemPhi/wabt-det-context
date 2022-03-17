import re

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
space_pattern = '\s+'
giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
mention_regex = '@[\w\-]+'
emoji_regex = '&#[0-9]{4,6};'

def get_data_loader(train_data, val_data, batch_size=5):
      
    # Create DataLoader for training data    
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data    
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader


def scatter_tSNE(features, labels, file_path="vis/tSNE/test-SNE.jpg"):

    num_classes = 2 
    
    embeddings = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(features)
   
    palette = np.array(sns.color_palette("hls", num_classes))
    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(embeddings[:,0], embeddings[:,1], lw=0, s=40, c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(embeddings[labels == i, :], axis=0)
        if i == 0:
            txt = ax.text(xtext, ytext, "N-WABT", fontsize=24)
        else: 
            txt = ax.text(xtext, ytext, "WABT", fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.savefig(file_path)


def load_comments(csv_path):
    df = pd.read_csv(csv_path) #pd.read_csv(csv_path) 
    df = df.drop_duplicates(subset=["Comments"], keep='last', inplace=False)
    
        
    all_comments = df[['Comments']].values.squeeze()
    all_labels = df[['Label']].values.squeeze()
    all_topic = df[['Topic']].values.squeeze()
    all_title = df[['Title']].values.squeeze()
    all_id = df[["ID"]].values.squeeze()
    #all_transcripts = df[["Transcript"]].values.squeeze()
    all_transcripts = []

    topics, comment_count_by_topic = np.unique(df.index, return_counts=True)
    
    '''
    plt.figure()    
    plt.xticks(rotation=10)
    plt.xlabel("Topic")
    plt.ylabel("No. of comments")
    plt.title("Dataset Comments Per Topic")
    plt.bar(topics, comment_count_by_topic)
    plt.savefig("vis/topic_dataset_summary.jpg")
    plt.close("all")
    

    labels, comment_count_by_labels = np.unique(all_labels, return_counts=True)
    labels = ["Not Whataboutism", "Whataboutism"]

    plt.figure()    
    plt.xticks(rotation=10)
    plt.xlabel("Label")
    plt.ylabel("No. of comments")
    plt.title("Dataset Comments Per Label")
    plt.bar(labels, comment_count_by_labels)
    plt.savefig("vis/label_dataset_summary.jpg")
    plt.close("all")
    '''

    
    unique_id = np.unique(all_id)

    df = df.reset_index().set_index("Title")  
    
    return all_comments, all_labels, all_topic, all_title, all_id, unique_id, all_transcripts, df

def get_items_from_split(idx, all_comments, all_labels, all_topics, all_titles, all_ids, all_transcripts, feats):
    
    return feats[idx], all_comments[idx], all_labels[idx], all_topics[idx], all_titles[idx], all_ids[idx], all_transcripts[idx]


def train_split_balance(all_comments, all_topics, all_labels):
    test_df = pd.DataFrame( list(zip(all_comments, all_topics, all_labels)), columns=["Comment", "Topic", "Labels"] )

    vid_topics = test_df["Topic"].unique()

    train_idx_all = []
    test_idx_all = []

    for topic in vid_topics:
        topic_index = test_df[ test_df["Topic"] == topic ].index
        topic_labels = test_df[ test_df["Topic"] == topic ]["Labels"].values
        train_idx, val_idx, _, _ = train_test_split(topic_index, topic_labels, test_size=0.4, random_state=42)

        train_idx_all.append(train_idx)
        test_idx_all.append(val_idx)

    train_idx_all = np.hstack(train_idx_all)
    test_idx_all = np.hstack(test_idx_all)

    train_idx_all = np.unique(train_idx_all)
    test_idx_all = np.unique(test_idx_all)

    # Plot the dataset-split train vs. test
    
    '''
    labels = ["Not Whataboutism", "Whataboutism"]
    _, train_comment_count_by_labels = np.unique(all_labels[train_idx_all], return_counts=True)
    plt.figure()    
    plt.xticks(rotation=10)
    plt.xlabel("Label")
    plt.ylabel("No. of comments")
    plt.title("Train Set Comments Per Label")
    plt.bar(labels, train_comment_count_by_labels)
    plt.savefig("vis/train_label_dataset_summary.jpg")
    plt.close("all")

    _, test_comment_count_by_labels = np.unique(all_labels[test_idx_all], return_counts=True)
    plt.figure()    
    plt.xticks(rotation=10)
    plt.xlabel("Label")
    plt.ylabel("No. of comments")
    plt.title("Test Set Comments Per Label")
    plt.bar(labels, test_comment_count_by_labels)
    plt.savefig("vis/test_label_dataset_summary.jpg")
    plt.close("all")
    '''
   
    return train_idx_all, test_idx_all 

def sort_lda_words(tup): 
  
    # reverse = None (Sorts in Ascending order) 
    # key is set to sort using second element of 
    # sublist lambda has been used 
    tup.sort(key = lambda x: x[1], reverse=True) 
    return tup


def get_unique_words(lst):
    d = {}
    for tpl in lst:
        first,  last = tpl
        if first not in d or last > d[first][-1]:
            d[first] = tpl
    
    return [*d.values()]



def get_token_split(train_idx, test_idx,  tokens):

    return [tokens[i] for i in train_idx], [tokens[i] for i in test_idx]


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub('RT','', parsed_text) #Some RTs have !!!!! in front of them
    parsed_text = re.sub(emoji_regex,'',parsed_text) #remove emojis from the text
    parsed_text = re.sub('…','',parsed_text) #Remove the special ending character is truncated
    parsed_text = re.sub('#[\w\-]+', '',parsed_text)
    # parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text


def preprocess_clean(text_string, remove_hashtags=True, remove_special_chars=True):
    # Clean a string down to just text
    # text_string=preprocess(text_string)

    parsed_text = preprocess(text_string)
    # parsed_text = parsed_text.lower()
    parsed_text = re.sub('\'', '', parsed_text)
    parsed_text = re.sub('|', '', parsed_text)
    parsed_text = re.sub(':', '', parsed_text)
    parsed_text = re.sub(',', '', parsed_text)
    parsed_text = re.sub('/', ' ', parsed_text)
    parsed_text = re.sub("\*", '', parsed_text)
    parsed_text = re.sub(';', '', parsed_text)
    parsed_text = re.sub('\.', '', parsed_text)
    parsed_text = re.sub('&amp', '', parsed_text)
    parsed_text = re.sub('ð', '', parsed_text)

    if remove_hashtags:
        parsed_text = re.sub('#[\w\-]+', '', parsed_text)
    if remove_special_chars:
        # parsed_text = re.sub('(\!|\?)+','.',parsed_text) #find one or more of special char in a row, replace with one '.'
        parsed_text = re.sub('(\!|\?)+','',parsed_text)
    return parsed_text

def add_augmentation(comments, labels, topics, titles, ids, aug_path="../dataset/augment.csv"):
    
    wabt_comments = np.where(labels==1)[0]
    aug_df = pd.read_csv(aug_path, index_col="Comments")
    
    for i in wabt_comments: 
        comment = comments[i]
        try: 
            aug_comments = aug_df.loc[comment][6:14].values
            topic = aug_df.loc[comment]["Topic"]
           
            title = aug_df.loc[comment]["Title"]
            id = aug_df.loc[comment]["ID"]

            extend_labels = np.repeat(1, aug_comments.shape)
            extend_topic = np.repeat(topic, aug_comments.shape)
            extend_title = np.repeat(title, aug_comments.shape)
            extend_id = np.repeat(id, aug_comments.shape)

            comments = np.hstack((comments, aug_comments))
            labels = np.hstack((labels, extend_labels))
            topics = np.hstack((topics, extend_topic))
            titles = np.hstack((titles, extend_title))
            ids = np.hstack((ids, extend_id))
         
        except Exception as e :
            continue
    
    return comments, labels, topics, titles, ids

def train_test_split_helper(comments, titles, labels, topics, ids):
    train_idx, test_idx  = train_split_balance(comments, titles, labels)
    
    train_comments = comments[train_idx]
    test_comments = comments[test_idx]
    
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    
    train_topics = topics[train_idx]
    test_topics = topics[test_idx]

    train_titles = titles[train_idx]
    test_titles = titles[test_idx]

    train_ids = ids[train_idx]
    test_ids = ids[test_idx]

    return train_comments, train_labels, train_topics, train_titles, train_ids, test_comments, test_labels, test_topics, test_titles, test_ids