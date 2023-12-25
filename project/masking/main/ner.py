from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from collections import defaultdict
import torch
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
model = AutoModelForTokenClassification.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
model.to(device)

# 송장이라 추정되는부분을 craft에 통과시키고 text 가 있는부분을 크롭해서 trocr로 text를 그 영역에 뽑아낸이후 프로세스입니다.
# 뽑힌 text에 대한 class를 판별합니다. 
# text에 대한 class가 "사람이름 PS", "도로/건물 이름 AF", "주소 LC" 에 속하면 1을 반환하여 이후 모자이크 하도록합니다. 
# ner 모델은 text를 어절 마다 쪼개서 각 단어에 대한 class를 반환합니다.
# 이 때 모든 단어에 대한 class를 고려하다보면 infer speed 가 매우느려서 최소한 하나라도 ps,af,lc 클래스 해당 단어가 있으면 1 반환하도록합니다. 

def check_entity(entities):
    for entity_info in entities:
        entity_value = entity_info.get('entity', '').upper()
        if 'LC' in entity_value or 'PS' in entity_value or 'AF' in entity_value:
            return 1
    return 0
def ner(example):
  ner = pipeline("ner", model=model, tokenizer=tokenizer,device=device)
  ner_results = ner(example)
  ner_results=check_entity(ner_results)
  return ner_results



# 하나
# def find_longest_value_key(input_dict):
#     max_length = 0
#     max_length_keys = []

#     for key, value in input_dict.items():
#         current_length = len(value)
#         if current_length > max_length:
#             max_length = current_length
#             max_length_keys = [key]
#         elif current_length == max_length:
#             max_length_keys.append(key)

#     if len(max_length_keys) == 1:
#         return 0
#     else:
#         return 1



# def find_longest_value_key2(input_dict):
#     if not input_dict:
#         return None

#     max_key = max(input_dict, key=lambda k: len(input_dict[k]))
#     return max_key


# def find_most_frequent_entity(entities):
#     entity_counts = defaultdict(list)

#     for item in entities:
#         split_entity = item['entity'].split('-')

#         entity_type = split_entity[1]
#         entity_counts[entity_type].append(item['score'])
#     number=find_longest_value_key(entity_counts)
#     if number==1:
#       max_entities = []
#       max_score_average = -1

#       for entity, scores in entity_counts.items():
#           score_average = sum(scores) / len(scores)

#           if score_average > max_score_average:
#               max_entities = [entity]
#               max_score_average = score_average
#           elif score_average == max_score_average:
#               max_entities.append(entity)
#       if len(max_entities)>0:
#            return max_entities if len(max_entities) > 1 else max_entities[0]
#       else:
#            return "Do not mosaik"
#     else:
#       A=find_longest_value_key2(entity_counts)

#       return A




# 하나라도 ps 나 lc 가 있으면 바로 ps , lc 꺼내기 


#   label=filtering(ner_results)
#   if label.find("PS")>-1 or label.find("LC")>-1:
#     return 1
#   else:
#     return 0
#print(ner("홍길동"))




#label=check_label(example)


