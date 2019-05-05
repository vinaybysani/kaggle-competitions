import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from scipy.stats import skew, kurtosis
from fuzzywuzzy import fuzz
import gensim
from nltk.corpus import stopwords
from nltk import word_tokenize
from tqdm import tqdm
import sys, traceback
import pickle

def wmd(model, s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)

def sent2vec(model, s):
    '''
    Average of Word2Vec vectors : take the average of all the word vectors in a sentence. This average vector will represent your sentence vector.
    '''
    words = str(s).lower()
    # words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

def run():
    try:
        print("Reading csv")
        train = pd.read_csv('data/train.csv')
        train = train.drop(['id', 'qid1', 'qid2'], axis=1)
        

        model = gensim.models.KeyedVectors.load_word2vec_format('data/w2v_googlenews/GoogleNews-vectors-negative300.bin.gz', binary=True)
        train['wmd'] = train.apply(lambda x: wmd(model, x['question1'], x['question2']), axis=1) # Word movers distance

        question1_vectors = np.zeros((train.shape[0], 300))
        question2_vectors = np.zeros((train.shape[0], 300))

        print("Generating question vectors")
        for i, q in tqdm(enumerate(train.question1.values)):
            question1_vectors[i, :] = sent2vec(model, q)

        for i, q in tqdm(enumerate(train.question2.values)):
            question2_vectors[i, :] = sent2vec(model, q)
        

        # Simple features
        print("Calculating simple features")
        train['len_q1'] = train.question1.apply(lambda x: len(str(x)))
        train['len_q2'] = train.question2.apply(lambda x: len(str(x)))
        train['diff_len'] = train.len_q1 - train.len_q2
        train['len_char_q1'] = train.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        train['len_char_q2'] = train.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        train['len_word_q1'] = train.question1.apply(lambda x: len(str(x).split()))
        train['len_word_q2'] = train.question2.apply(lambda x: len(str(x).split()))
        train['common_words'] = train.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
        train['fuzz_qratio'] = train.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
        train['fuzz_WRatio'] = train.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
        train['fuzz_partial_ratio'] = train.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
        train['fuzz_partial_token_set_ratio'] = train.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        train['fuzz_partial_token_sort_ratio'] = train.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        train['fuzz_token_set_ratio'] = train.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        train['fuzz_token_sort_ratio'] = train.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        
        # Strong features (Various Distances)
        print("Calculating strong features")
        train['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
        train['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
        train['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
        train['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
        train['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
        train['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
        train['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

        # For normally distributed data, the skewness should be about 0. 
        train['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
        train['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]

        # Kurtosis is a measure of the combined weight of a distribution's tails relative to the center of the distribution.
        # Like skewness, kurtosis is a statistical measure that is used to describe the distribution. 
        # kurtosis is a measure that describes the shape of a distribution's tails in relation to its overall shape.
        train['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
        train['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

        # Writing pickle file
        pickle.dump(np.nan_to_num(question1_vectors), open('data/q1_w2v.pkl', 'wb'), -1)
        pickle.dump(np.nan_to_num(question2_vectors), open('data/q2_w2v.pkl', 'wb'), -1)
        print("Pickle write successful")

        # Writing features to csv file
        train.to_csv('data/features.csv', index=False)
        print("CSV write successful")
    except:
        print("!!! Exception caught")
        traceback.print_exc(file=sys.stdout)

run()