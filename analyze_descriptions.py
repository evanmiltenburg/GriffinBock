import json
import csv
import spacy
from operator import itemgetter
from collections import defaultdict
from numpy import median


def load_data(filename):
    "Load the annotation data."
    with open(filename) as f:
        data = json.load(f)
    return data


def count_nouns(doc):
    "Count the number of nouns in a sentence, merging compound nouns."
    num_nouns = 0
    preceding = None
    for tok in doc:
        if tok.pos_ in {'NOUN','PROPN'} and preceding not in {'NOUN','PROPN'}:
            if tok.orth_ not in {'voorgrond', 'achtergrond', 'midden', 'voorkant', 
                                 'achterkant', 'donker'}:
                num_nouns += 1
        preceding = tok.pos_
    return num_nouns


nlp = spacy.load('nl_core_news_sm')

def enrich(data):
    "Enrich the data by parsing the normalized descriptions and counting the nouns."
    for entry in data:
        entry['doc'] = nlp(entry['normalized_description'])
        entry['num_nouns'] = count_nouns(entry['doc'])
        entry['pos'] = ' '.join([tok.pos_ for tok in entry['doc']])


def write_data(data, filename, fieldnames):
    "Write data to a file."
    with open(filename,'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            selection = {key:val for key, val in entry.items() if key in fieldnames}
            writer.writerow(selection)


def median_nouns(data):
    "Compute the median number of nouns per image."
    index = defaultdict(list)
    for entry in data:
        index[entry['image']].append(entry['num_nouns'])
    for image, noun_counts in index.items():
        index[image] = median(noun_counts)
    return index

data = load_data('Resources/annotations_final.json')
enrich(data)

data = sorted(data, key=itemgetter('num_nouns'))
write_data(data, 
           filename='Output/annotated_data.csv', 
           fieldnames=['image', 'participant', 'normalized_description', 
                       'num_nouns', 'pos', 'filename'])

counts = median_nouns(data)
with open('Output/noun_counts.json','w') as f:
    json.dump(counts, f, indent=2)
