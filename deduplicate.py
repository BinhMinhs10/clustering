import os
import csv
import time
import xxhash
import pandas as pd
from tqdm import tqdm


def num_lines_in_file(fname):
    """
    Returns the number of lines in a file.
    """
    with open(fname, 'r') as f:
        for i, _ in enumerate(f):
            pass
    return i + 1


def dedup_file(input_file_lang_1, output_file_lang_1):
    print()
    print('====================================')
    print('========== De-duplicate ============')
    print('====================================')
    print()
    num_lines = num_lines_in_file(input_file_lang_1)
    hashes = set()
    num_output_lines = 0
    with open(input_file_lang_1, 'r') as f_lang1, open(output_file_lang_1, 'w') as f_out_lang1:
        for line_1 in tqdm(f_lang1, total=num_lines, desc=f"Deduplicating files"):
            parallel_hash = xxhash.xxh64((line_1.strip()).encode('utf-8')).hexdigest()
            if parallel_hash not in hashes:
                hashes.add(parallel_hash)
                f_out_lang1.write(line_1.strip() + '\n')
                num_output_lines += 1
            else:
                print(line_1)

    print(f"Kept {num_output_lines} out of {num_lines} after deduplication")


def dedup_couple_file(input_file_lang_1, input_file_lang_2, output_file_lang_1, output_file_lang_2):
    print()
    print('====================================')
    print('========== De-duplicate ============')
    print('====================================')
    print()
    num_lines = num_lines_in_file(input_file_lang_1)
    hashes = set()
    num_output_lines = 0
    with open(input_file_lang_1, 'r') as f_lang1, \
        open(input_file_lang_2, 'r')  as f_lang2, \
        open(output_file_lang_1, 'w') as f_out_lang1, \
        open(output_file_lang_2, 'w') as f_out_lang2:
        for line_1, line_2 in tqdm(zip(f_lang1, f_lang2), total=num_lines, desc=f"Deduplicating files"):
            parallel_hash = xxhash.xxh64((line_1.strip() + '\t' + line_2.strip()).encode('utf-8')).hexdigest()
            if parallel_hash not in hashes:
                hashes.add(parallel_hash)
                f_out_lang1.write(line_1.strip() + '\n')
                f_out_lang2.write(line_2.strip() + '\n')
                num_output_lines += 1

    print(f"Kept {num_output_lines} out of {num_lines} after deduplication")


if __name__ == "__main__":
    dataset_path = "data/news_20k.csv"
    output_path = "data/news_20k.txt"
    max_corpus_size = 50000  # We limit our corpus to only the first 50k questions

    # Get all unique sentences from the file
    corpus_sentences = set()
    corpus_sentences = pd.read_csv(dataset_path, usecols=['DESCRIPTION'])

    # corpus_sentences = list(corpus_sentences)
    corpus_sentences.loc[corpus_sentences['DESCRIPTION'].notnull(), 'value_is_NaN'] = 'No'
    corpus_sentences = corpus_sentences.loc[corpus_sentences['value_is_NaN'] == 'No']

    textfile = open(output_path, "w")
    for element in corpus_sentences['DESCRIPTION']:
        textfile.write(element + "\n")
    textfile.close()

    dedup_file(
        output_path,
        "data/hash_deduplicate.txt"
    )