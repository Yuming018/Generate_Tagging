#https://github.com/NeuroSYS-pl/coreference-resolution/blob/main/allennlp_coreference_resolution.ipynb

from allennlp.predictors.predictor import Predictor
from collections import defaultdict

def corf_resolution(text):
    model_url = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz'
    predictor = Predictor.from_path(model_url)
    prediction = predictor.predict(document=text)
    
    document, clusters = prediction['document'], prediction['clusters']
    coreference = defaultdict(list)
    for cluster in clusters:
        print(get_span_words(cluster[0], document) + ': ', end='')
        print(f"[{'; '.join([get_span_words(span, document) for span in cluster])}]")
        for span in cluster:
            coreference[get_span_words(cluster[0], document)].append(get_span_words(span, document))
    
    return coreference

def get_span_words(span, document):
    return ' '.join(document[span[0]:span[1]+1])

if __name__ == '__main__':
    text = "quite unlike other boys , kintaro , grew up all alone in the mountain wilds . as he had no companions he made friends with all the animals and learned to understand them and to speak their strange talk . by degrees they all grew quite tame and looked upon kintaro as their master , and he used them as his servants and messengers . but his special retainers were the bear , the deer , the monkey and the hare ."
    coreference = corf_resolution(text)
    # print('i' in 'was a pretty lttle fellow once')
