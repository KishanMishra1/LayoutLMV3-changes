import os
import json
import token
import cv2
import numpy as np
import pickle
from PIL import Image
import warnings


folder_dir='dataset/training_data/annotations/'


def address(json_file):
    image_dir='dataset/training_data/images/'
    image_file=str(json_file)[:-5]+'.jpeg'
    image = cv2.imread(image_dir+image_file)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return img
    



#NER_TAGS
label2id={}
id2label={0:'cess_amount',1:'cgst_amount',2: "customer_phone_number",3:"receipt_number",4:"savings",5:'sgst_amount',
6:'sub_total_amount',7:'total_amount',8:'total_items_number',9:'transaction_date',10:'transaction_time',11:'vendor_address',
12:'vendor_gst',13:'vendor_name',14:'vendor_phone_number'}
for i,j in id2label.items():
    label2id[j]=i
kx=[]
final={}

count=0
ids=[]
tokens2,bboxes2,ner_tags2,ids2,image2=[],[],[],[],[]
for i in os.listdir(folder_dir):
    if str(i).endswith(".json"):
        
        tokens,bboxes,ner_tags,image=[],[],[],[]
        with open(folder_dir+i, 'r') as k:
            res = json.load(k)
        ids.append(count)
        for itr1 in range(len(res['form'])):
            for itr2 in range(len(res['form'][itr1]['words'])):
                tokens.append(res['form'][itr1]['words'][itr2]['text'])
                bboxes.append(res['form'][itr1]['words'][itr2]['bbox'])
                ner_tags.append(label2id[res['form'][itr1]['label']])
            image.append(address(i))

        tokens2.append(tokens)
        bboxes2.append(bboxes)
        ner_tags2.append(ner_tags)
        image2.append(image)

    count+=1
ans={
    'id':ids,
    'tokens':tokens2,
    'bboxes':bboxes2,
    'ner_tags':ner_tags2,
    'image':image2

}
print(0)