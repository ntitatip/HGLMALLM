import os
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, BertConfig
from tqdm.auto import tqdm
import pickle




def read_csv(file_path):
    try:
        df = pd.read_csv(file_path, sep=",", header=None)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def process_sequence(sequence, k):
    UtoT_sequence = sequence.replace('U', 'T')
    kmers = [ UtoT_sequence[i:i+k] for i in range(len(UtoT_sequence) - k + 1)]
    return kmers


def extract_bert_features(path, sequences_list):
    max_token_length = 510 
    processed_kmers = []  
    original_indices = [[] for _ in sequences_list]

    for idx, sequence in enumerate(sequences_list): 
        if len(sequence) <= max_token_length:
            processed_kmers.append(sequence)
            original_indices[idx].append(len(processed_kmers)-1)
        else:
            for i in range(0, len(sequence), 50):
                end_idx = i + max_token_length
                if end_idx > len(sequence):
                    break
                sub_text = sequence[i:end_idx]
                processed_kmers.append(sub_text)
                original_indices[idx].append(len(processed_kmers)-1)

    processed_sequences = [' '.join(words) for words in processed_kmers]

    model_path = os.path.join(path, 'pytorch_model.bin')
    config_path = os.path.join(path, 'config.json')
    vocab_path = path

    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    config = BertConfig.from_pretrained(config_path)

    model = BertModel.from_pretrained(model_path, config=config)

    model.eval()
    model.cuda()

    encoding = tokenizer(processed_sequences, padding=True, max_length=512, truncation=True, return_tensors="pt")
    encoding_gpu = {k: v.to('cuda') for k, v in encoding.items()}


    with torch.no_grad():
        outputs = model(**encoding_gpu)

    
    outputs_cpu = {key: value.to('cpu') for key, value in outputs.items()}

    pooled_results = []

    for row in original_indices:

        processed_indices = row  
    

        selected_states = torch.index_select(outputs_cpu['pooler_output'], 0, torch.tensor(processed_indices))
    

        pooled, _ = torch.max(selected_states, dim=0)
        pooled_results.append(pooled)




    del outputs
    torch.cuda.empty_cache()  

    return pooled_results


def extract_features(file_path):
    data_name = "_".join(os.path.basename(file_path).split("_")[:2]) + "_"
    df = read_csv(file_path)
    df[2] = df[1].apply(lambda x: process_sequence(x, 3))

    directories_models = [d for d in os.listdir(model_directory) if os.path.isdir(os.path.join(model_directory, d))]

    for model_names in directories_models:
        model_path = os.path.join(model_directory, model_names)
        if os.path.isdir(model_path):
            sequence_vector = []
            for i in tqdm(range(0, len(df[2]), 5), desc="Processing", leave=True):
                sliding_sentences = extract_bert_features(model_path, df[2][i:i+5])
                sequence_vector.extend(sliding_sentences)

            results_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'feature')
            results_path = os.path.join(results_file, data_name + model_names.split("-")[:1][0] + ".pkl")


            with open(results_path, 'wb') as file:
                pickle.dump(sequence_vector, file)

# 获取文件所在的目录



pyfile_path = os.path.dirname(os.path.realpath(__file__))
raw_data_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'raw_data')
model_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')





directories_data = [d for d in os.listdir(raw_data_directory) if os.path.isdir(os.path.join(raw_data_directory, d))]
   



for directory in directories_data:
    dir_path = os.path.join(raw_data_directory, directory)
    if os.path.isdir(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            print(f"Processing file: {file_path}")



