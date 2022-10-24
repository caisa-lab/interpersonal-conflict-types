import re
from tqdm import tqdm
from constants import NTA_KEYWORDS, YTA_KEYWORDS
from utils.clusters_utils import ListDict
import os
import pickle as pkl
import string
from abc import ABC, abstractmethod
                          
                
class Dataset(ABC):
    def __init__(self):
        self.verdictToId = {}
        self.idToVerdict = []
        self.verdictToText = {}
        self.verdictToParent = {}
        self.verdictToCleanedText = {}
        self.verdictToLabel = {}
        self.postIdToTitle = {}
        self.postIdToText = {}
        self.idTopostId = []
        self.postIdToId = {}
        self.authorsToVerdicts = ListDict()
        self.verdictToAuthor = {}
        self.aita_labels = ['NTA','YTA']

    @abstractmethod
    def load_maps(self, sc, sn):
        pass
    
class SocialNormDataset:
    """Creates the verdicts datastructures to use through out the project
    """
    def __init__(self, sc, sn, cond=5):
        self.verdictToId = {}
        self.idToVerdict = []
        self.verdictToText = {}
        self.verdictToParent = {}
        self.verdictToCleanedText = {}
        self.verdictToLabel = {}
        self.verdictToTokensLength = {}
        self.postIdToTitle = {}
        self.postIdToText = {}
        self.idTopostId = []
        self.postIdToId = {}
        self.filtering_cond = cond
        self.authorsToVerdicts = ListDict()
        self.verdictToAuthor = {}
        self.postToVerdicts = ListDict()
        #self.aita_labels = ['NTA','YTA','NAH','ESH','INFO']
        self.aita_labels = ['NTA','YTA']
        

        self.load_maps(sc, sn)


    def load_maps(self, sc, sn):
        for _, row in tqdm(sn.iterrows(), desc='Creating situations maps'):
            post_id = row['post_id']
            self.postIdToTitle[post_id] = row['situation']
            self.postIdToText[post_id] = row['fulltext']
            self.idTopostId.append(post_id)
            self.postIdToId[post_id] = len(self.idTopostId) - 1
        
        for _, row in tqdm(sc.dropna(subset=['author_name', 'author_fullname']).iterrows(), desc="Creating filtered verdict-authors maps by condition {}".format(self.filtering_cond)):
            label = row['label']
            verdict = row['id']
            if label in self.aita_labels and row['author_name'] != 'Judgement_Bot_AITA':
                self.verdictToAuthor[verdict] = row['author_name']    
                self.authorsToVerdicts.append(row['author_name'], verdict)

        authorsToCount = {k: len(v) for k, v in self.authorsToVerdicts.items()}
        filtering_cond = 5

        for a, c in authorsToCount.items():
            if c <= filtering_cond:
                verdicts = self.authorsToVerdicts.pop(a)
                for v in verdicts:
                    del self.verdictToAuthor[v]         
                    
        print("After filtering, we are left with {} authors and {} verdicts.".format(len(self.authorsToVerdicts), len(self.verdictToAuthor)))   
        
        for _, row in tqdm(sc.iterrows(), desc="Creating comments maps"):
            label = row['label']
            parent = row['parent_id']
            verdict = row['id']
            
            if label in self.aita_labels and parent in self.postIdToId and verdict in self.verdictToAuthor:
                text = row['body']
                self.idToVerdict.append(verdict)
                self.verdictToId[verdict] = len(self.idToVerdict) - 1
                self.verdictToText[verdict] = text
                self.verdictToLabel[verdict] = self.aita_labels.index(label)
                self.verdictToParent[verdict] = parent
        
        self._clean_keywords_from_verdicts()
        self.verdictToTokensLength = {k: len(v.split(' ')) for k, v in self.verdictToCleanedText.items()}
        
        for v, s in self.verdictToParent.items():
            self.postToVerdicts.append(s, v)

    def _clean_keywords_from_verdicts(self):
        keywords_rep = {'ampx200b': "", 'x200b': "", 'AITA': "", 'aita': ""}
        
        for key in NTA_KEYWORDS + YTA_KEYWORDS:
            keywords_rep[key] = ""
        keywords_rep = dict(sorted(keywords_rep.items(), key=lambda k: len(k[0]), reverse=True))

        rep = dict((re.escape(k), v) for k, v in keywords_rep.items()) 
        pattern = re.compile("|".join(rep.keys()))

        for verdict, text in tqdm(self.verdictToText.items(), desc="Removing keywords from verdicts"):
            self.verdictToCleanedText[verdict] = pattern.sub(lambda m: rep[re.escape(m.group(0))], text.lower())
            self.verdictToCleanedText[verdict] = self.verdictToCleanedText[verdict].translate(str.maketrans('', '', string.punctuation))       
    
    def clean_single_text(self, text):
        keywords_rep = {'ampx200b': "", 'x200b': "", 'AITA': "", 'aita': "", '[removed]': "", '[REMOVED]': "", '[deleted]': "", '[DELETED]': ""}
        rep = dict((re.escape(k), v) for k, v in keywords_rep.items()) 
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text.lower())
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def clean_single_verdict(self, text):
        keywords_rep = {'ampx200b': "", 'x200b': "", 'AITA': "", 'aita': "", '[removed]': "", '[REMOVED]': "", '[deleted]': "", '[DELETED]': ""}
        for key in NTA_KEYWORDS + YTA_KEYWORDS:
            keywords_rep[key] = ""
        keywords_rep = dict(sorted(keywords_rep.items(), key=lambda k: len(k[0]), reverse=True))
        
        rep = dict((re.escape(k), v) for k, v in keywords_rep.items()) 
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text.lower())
        return text.translate(str.maketrans('', '', string.punctuation))    
    
    
    def get_authors_from_situations(self, situations):
        authors = set()
        for sit in situations:
            if sit in self.postToVerdicts:
                verdicts = self.postToVerdicts[sit]
                for v in verdicts:
                    authors.add(self.verdictToAuthor[v])
        
        return authors
    
 