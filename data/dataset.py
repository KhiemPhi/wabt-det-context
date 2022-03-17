import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import random


class WhataboutismDataset(Dataset):

    def __init__(self, comments, labels, topics, titles, ids, context, df, test):

        self.labels = torch.tensor(labels, dtype=torch.long)
       
        self.comments = comments
        self.topics = topics
        self.titles = titles
        self.ids = ids

        self.on_topic_related = 0
        self.on_topic_whataboutism = 0

        self.pos_counts = len( np.where(labels==1)[0] )
        self.neg_counts = len( np.where(labels==0)[0] )

        self.context = context
        
      
        self.df = df 
        
        self.test = test
        


  
    def __getitem__(self, idx:int):

        label = self.labels[idx]
        comment = self.comments[idx]
     
        topic = self.titles[idx]

        # Generate another random comment
        if not self.test:
            same_topic_comments = self.df.loc[topic]["Comments"].values
            same_topic_labels = self.df.loc[topic]["Label"].values
            different_label_idx = np.where(same_topic_labels != label.item())[0]
            context_comments = same_topic_comments[different_label_idx]
            context = np.random.choice(context_comments, size=1)[0]
            
            return comment, label, context
        else: 

            same_topic_comments = self.df.loc[topic]["Comments"].values
            context = np.random.choice(same_topic_comments, size=1)[0]
            
            return comment, label, context
    def __len__(self):

        return len(self.titles)

