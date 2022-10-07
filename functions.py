from collections import Counter


def item_class(tokens, mentions):
    item_list1 = []
    item_list2 = []
    for ment in mentions:
        start = ment["start"]
        end = ment["end"]
        token = tokens[start:end]
        label = ment["labels"]
        token = ' '.join(map(str, token))
        item_list1 = [token, label, start, end]    
        item_list2.append(item_list1)    
    return item_list2

def majority_item(ann1, ann2, ann3, ann4):
    list1 = []
    list2 = []
    list3 = []
    i = 2
    list1.append(ann1[i][0])
    list1.append(ann2[i][0])
    list1.append(ann3[i][0])
    list1.append(ann4[i][0])
    majority_word = most_frequent(list1)
    # list1.clear()

    # list2.append(ann1[i][1])
    # list2.append(ann2[i][1])
    # list2.append(ann3[i][1])
    # list2.append(ann4[i][1])
    # majority_item = most_frequent(list2)
    # list2.clear()

    # word_item = [majority_word, majority_item]
    list3.append(majority_word)     
        
    return list3

def most_frequent(List) :
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]



