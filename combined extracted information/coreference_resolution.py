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
        # print(get_span_words(cluster[0], document) + ': ', end='')
        # print(f"[{'; '.join([get_span_words(span, document) for span in cluster])}]")
        # print(get_span_words(cluster[0], document))
        # print(cluster)
        # input()
        coreference[get_span_words(cluster[0], document)] = cluster
        # for span in cluster:
            # coreference[get_span_words(cluster[0], document)].append(get_span_words(span, document))
    
    allennlp_pred = defaultdict(list)
    allennlp_pred['document'] = document
    allennlp_pred['coreference'] = coreference
    return allennlp_pred

def get_span_words(span, document):
    # print(document[span[0]:span[1]+1])
    # input()
    return ' '.join(document[span[0]:span[1]+1])

if __name__ == '__main__':
    text = "then said culain , ' have all thy retinue come in , o conchubar ? ' and when the king said that they were all there , culain bade one of his apprentices go out and let loose the great mastiff that guarded the house . now , this mastiff was as large as a calf and exceedingly fierce , and he guarded all the smith 's property outside the house , and if anyone approached the house without beating on the gong , which was outside the foss and in front of the drawbridge , he was accustomed to rend him . then the mastiff , having been let loose , careered three times round the liss , baying dreadfully , and after that remained quiet outside his kennel , guarding his master 's property . but , inside , they devoted themselves to feasting and merriment , and there were many jests made concerning culain , for he was wo nt to cause laughter to conchubar mac nessa and his knights , yet he was good to his own people and faithful to the crave rue , and very ardent and skilful in the practice of his art . but as they were amusing themselves in this manner , eating and drinking , a deep growl came from without , as it were a note of warning , and after that one yet more savage ; but where he sat in the champion 's seat , fergus mac roy struck the table with his hand and rose straightway , crying out , ' it is setanta . ' but ere the door could be opened they heard the boy 's voice raised in anger and the fierce yelling of the dog , and a scuffling in the bawn of the liss ."
    allennlp_pred = corf_resolution(text)
    print(allennlp_pred)
