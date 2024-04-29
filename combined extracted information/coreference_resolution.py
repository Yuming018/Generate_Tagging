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
        for span in cluster:
            coreference[get_span_words(cluster[0], document)].append(get_span_words(span, document))
    
    return coreference

def get_span_words(span, document):
    return ' '.join(document[span[0]:span[1]+1])

if __name__ == '__main__':
    # text = "' away , away ! ' barked the yard - dog . ' i 'll tell you ; they said i was a pretty little fellow once . then i used to lie in a velvet - covered chair , up at the master 's house , and sit in the mistress 's lap . they used to kiss my nose , and wipe my paws with an embroidered handkerchief , and i was called ' ami , dear ami , sweet ami . ' but after a while i grew too big for them , and they sent me away to the housekeeper 's room . so i came to live on the lower story . you can look into the room from where you stand , and see where i was master once . i was indeed master to the housekeeper . it was certainly a smaller room than those up stairs . but i was more comfortable , for i was not being continually taken hold of and pulled about by the children as i had been . i received quite as good food , or even better . i had my own cushion , and there was a stove -- it is the finest thing in the world at this season of the year . i used to go under the stove , and lie down quite beneath it . ah , i still dream of that stove . away , away ! '"
    # coreference = corf_resolution(text)
    print('i' in 'was a pretty lttle fellow once')
