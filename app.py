import gradio as gr
from ernie.ernie import SentenceClassifier
from ernie import helper

file_list=['config.json', 'tf_model.h5', 'tokenizer.json', 'vocab.txt', 'special_tokens_map.json', 'tokenizer_config.json']

for f in file_list:
    helper.download_from_hub(repo_id='jeang/bert-finetuned-sentence-classification-toy', filename=f, cache_dir='model/')
    
classifier = SentenceClassifier(model_path='model/', max_length=128, labels_no=2)

def classify(sentence):
    probability = classifier.predict_one(sentence)[1]
    
    return 'probability = ' + str(probability) + ' (' + ('positive' if probability >= 0.5 else 'negative') +  ')'
            

iface = gr.Interface(fn=classify, inputs="text", outputs="text")
iface.launch()
