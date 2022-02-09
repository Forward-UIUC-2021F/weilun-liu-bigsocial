import pandas as pd
import json
import re
import pickle
from tqdm import tqdm
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer


# data for classification
# TODO: change file name
f = open('misRwcTreeContents.json')
rwc = json.load(f)

sid = SentimentIntensityAnalyzer()
# use bert model
model = SentenceTransformer('bert-base-nli-mean-tokens')


# replace special characters in sentence by space
def processString(original):
    original = original.lower()
    original = ''.join(e for e in original if (e.isalnum() or e == ' '))
    original = original.replace("@", '')
    original = original.replace("#", '')
    words = original.split(' ')
    return words


# previously trained models
embeddingFile = 'bestEmbedding.sav'
svmFile = 'bestSVM.sav'

clfEmbedding = pickle.load(open(embeddingFile, 'rb'))
clfSVM = pickle.load(open(svmFile, 'rb'))

output = []
save = []

# begin classifying
for tree in tqdm(rwc.values()):
    # input()
    for original in tree[1:]:
        line = [original]

        # preprocess
        words = original.split(' ')
        for i, word in enumerate(words):
            if '#' in word and '#' == word[0]:
                newword = re.findall('[A-Z][^A-Z]*', word[1:])
                if len(newword) > 0:
                    words[i] = " ".join(newword)
        content = " ".join(words)
        content = content.lower()
        content = content.replace("cancer", 'thing')
        content = content.replace("@", '')
        content = content.replace("#", '')
        while 'http://' in content:
            j = 0
            k = 0
            for i in range(0, len(content)-4):
                if content[i:i+4] == 'http':
                    k = i
                    while i+j < len(content) and content[i+j] != ' ':
                        j += 1
            content = content[:k] + content[k+j:]
        while 'https://' in content:
            j = 0
            k = 0
            for i in range(0, len(content)-5):
                if content[i:i+5] == 'https':
                    k = i
                    while i+j < len(content) and content[i+j] != ' ':
                        j += 1
            content = content[:k] + content[k+j:]
        ss = sid.polarity_scores(content)

        # sentence embedding
        words = processString(original)
        test = model.encode(' '.join(words))
        res = clfEmbedding.predict([test.tolist()])[0]

        # svm
        temp = [ss['pos']-ss['neg'], TextBlob(content).sentiment[0], res]
        # tag num
        temp.append(original.count('#'))
        # ? num
        temp.append(original.count('?'))
        # ! num
        temp.append(original.count('!'))
        # " num
        temp.append(original.count('"'))


        # label
        label = clfSVM.predict([temp])[0]
        
        # append to output
        line.append(label)
        output.append(line)

# save results
output = pd.DataFrame(output)
output.to_csv("rwcMisLabled.csv", header=None, index=None)


