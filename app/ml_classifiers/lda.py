import pandas as pd
import numpy as np
import tqdm
import gensim
from gensim.models import CoherenceModel, TfidfModel
from gensim.corpora import Dictionary

def compute_coherence_values(corpus, texts, dictionary, k, a, b):
    """
    Calcula el valor de coherencia del modelo con los parametros indicados
    """
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b,
                                           workers=3)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')

    return coherence_model_lda.get_coherence()

def fine_tuning(X_tfidf, y):

    texts = [text for text in X_tfidf if text != []]

    dictionary = Dictionary(texts)      # espera recibir el corpus preprocesado

    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = TfidfModel(corpus)
    corpus = tfidf[corpus]

    # corpus = X_tfidf
    corpus = X_tfidf

    grid = {}
    grid['Validation_Set'] = {}

    # Topics range
    min_topics = 2
    max_topics = 11
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)

    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')

    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')

    # Validation sets
    num_of_docs = len(corpus)
    corpus_sets = [  # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25),
        # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
        gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
        corpus]
    # para dividir el dataset mejor usar StratifiedShuffleSplit,
    # ClippedCorpus solo devuelve el head del corpus que se le facilita
    corpus_title = ['75% Corpus', '100% Corpus']
    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                     }
    # Can take a long time to run
    if 1 == 1:
        pbar = tqdm.tqdm(total=540)

        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                # iterate through alpha values
                for a in alpha:
                    # iterare through beta values
                    for b in beta:
                        # get the coherence score for the given parameters
                        cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=dictionary,
                                                      k=k, a=a, b=b)
                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)

                        pbar.update(1)

        pd.DataFrame(model_results).to_csv(f"{str(path)}files/lda_tuning_results.csv', index=False")
        pbar.close()