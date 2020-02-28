# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 00:51:12 2019

@author: agilist
"""

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords 
import re
import os
import math
from statistics import mean

def frquency_matrix(sentences):
    matrix = {}
    stop_wrd = stopwords.words("english")
    stemmer = PorterStemmer()

    for sentence in sentences:
        sent_freq = {}
        words = word_tokenize(sentence)
        for word in words:
            word = word.lower()
            word = stemmer.stem(word)
            if word in stop_wrd:
                continue
            elif word in sent_freq.keys():
                sent_freq[word] += 1
            else:
                sent_freq[word] = 1

        matrix[sentence[:10]] = sent_freq
    return matrix


def term_freq_matrix(matrix_freq):
    term_freq = {}
    for s, table in matrix_freq.items():
        sent_table = {}

        word_count_in_sent = len(table)

        for word, freq in table.items():
            sent_table[word] = freq / word_count_in_sent

        term_freq[s] = sent_table

    return term_freq

def total_word_count(matrix_freq):
    total_word_freq = {}
    for sent, wtable in matrix_freq.items():
        for word, count in wtable.items():
            if word in total_word_freq.keys():
                total_word_freq[word] += 1
            else:
                total_word_freq[word] = 1

    return total_word_freq


def idf_matrix(matrix_freq, word_count, num_sent):
    idf = {}

    for sentence, freq in matrix_freq.items():
        idf_sent = {}

        for word in freq.keys():
            idf_sent[word] = math.log10(num_sent/ float(word_count[word]))

        idf[sentence] = idf_sent

    return idf

def tf_idf_matrix(term_freq_mat, matrix_idf):
    matrix_tf_idf = {}

    for (sentence1, tfreq1), (sentence2, tfreq2) in zip(term_freq_mat.items(), matrix_idf.items()):
        tf_idf_sent = {}

        for word, freq in tfreq1.items():
            freq2 = tfreq2[word]
            tf_idf_sent[word] = float(freq * freq2)

        matrix_tf_idf[sentence1] = tf_idf_sent

    return matrix_tf_idf

def sentence_scores(matrix_tf_idf):
    sent_score = {}

    for sent, tf_idf_matrix in matrix_tf_idf.items():
        total_score = 0
        word_count = len(tf_idf_matrix)

        for word, score in tf_idf_matrix.items():
            total_score = total_score + score

        sent_score[sent] = round(total_score/word_count, 2)

    return sent_score

def find_average_score(scores):
    avg_score = mean(scores[sent] for sent in scores)

    return round(avg_score, 2)

def generate_summary(sentences, scores, threshold):
    count = 0
    summary = ''

    print(round(threshold, 2))

    for sentence in sentences:
        if sentence[:10] in scores and scores[sentence[:10]] >= round(threshold, 2):
            summary = summary + " " + sentence
            count += 1

    return summary


def main():
    with open('article.txt', encoding="utf8") as f:
        text = f.read()
    sentences = sent_tokenize(text)
    num_sent = len(sentences)
    # print(sentences)

    matrix_freq = frquency_matrix(sentences)
    # for k,v in matrix_freq.items():
    #     print(k)
    #     print(v)
    term_freq_mat = term_freq_matrix(matrix_freq)
    # for k,v in term_freq_mat.items():
    #     print(k)
    #     print(v)
    word_count = total_word_count(matrix_freq)
    # for k,v in word_doc_count.items():
    #     print(k)
    #     print(v)
    matrix_idf = idf_matrix(matrix_freq, word_count, num_sent)
    # for k,v in matrix_idf.items():
    #     print(k)
    #     print(v)
    matrix_tf_idf = tf_idf_matrix(term_freq_mat, matrix_idf)
    # for k, v in matrix_tf_idf.items():
    #     print(k)
    #     print(v)
    scores = sentence_scores(matrix_tf_idf)
    for k, v in scores.items():
        print(k)
        print(v)
    avg_score = find_average_score(scores)

    summary = generate_summary(sentences, scores, 0.9*avg_score)

    print(summary)







if __name__ == '__main__':
    main()