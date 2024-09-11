import json
import os
import math
import random

import pandas as pd
import numpy as np
from collections import defaultdict


def tokenize(text):
    return text.lower().split()


def tokenize_with_boundaries(text):
    tokens = text.lower().split()
    return ['<s>', '<s>'] + tokens + ['</s>', '</s>']


# Generate unigrams, bigrams, and trigrams
def generate_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams


# Function to generate trigrams from the tokenized text
def generate_trigrams(tokens):
    trigrams = []
    for i in range(len(tokens) - 2):
        trigrams.append((tokens[i], tokens[i+1], tokens[i+2]))
    return trigrams


def build_ngram_models(dataset):
    unigram_model = defaultdict(int)
    bigram_model = defaultdict(int)
    trigram_model = defaultdict(int)
    
    for document in dataset:
        tokens = tokenize_with_boundaries(document)
        
        unigrams = generate_ngrams(tokens, 1)
        bigrams = generate_ngrams(tokens, 2)
        trigrams = generate_ngrams(tokens, 3)
        
        # Count unigrams, bigrams, and trigrams
        for unigram in unigrams:
            unigram_model[unigram] += 1
        
        for bigram in bigrams:
            bigram_model[bigram] += 1
        
        for trigram in trigrams:
            trigram_model[trigram] += 1
    
    total_unigrams = sum(unigram_model.values())
    return unigram_model, bigram_model, trigram_model, total_unigrams


# Function to create the trigram language model
def create_trigram_model(dataset):
    trigram_model = defaultdict(int)

    for document in dataset:
        tokens = tokenize(document)
        trigrams = generate_trigrams(tokens)
        for trigram in trigrams:
            trigram_model[trigram] += 1

    return trigram_model


# Function to calculate unigram, bigram, and trigram probabilities
def calculate_probabilities(unigram_model, bigram_model, trigram_model, total_unigrams):
    unigram_prob = {unigram: count / total_unigrams for unigram, count in unigram_model.items()}
    bigram_prob = {bigram: count / unigram_model[(bigram[0],)] for bigram, count in bigram_model.items()}
    trigram_prob = {trigram: count / bigram_model[(trigram[0], trigram[1])] for trigram, count in trigram_model.items()}
    
    return unigram_prob, bigram_prob, trigram_prob


# Function to perform interpolation
def interpolate(trigram_prob, bigram_prob, unigram_prob, trigram, lambdas):
    w1, w2, w3 = trigram  # unpack trigram
    lambda3, lambda2, lambda1 = lambdas
    
    # Trigram probability P(w3 | w1, w2)
    trigram_p = trigram_prob.get(trigram, 0)
    
    # Bigram probability P(w3 | w2)
    bigram_p = bigram_prob.get((w2, w3), 0)
    
    # Unigram probability P(w3)
    unigram_p = unigram_prob.get((w3,), 0)
    
    # Interpolated probability
    interpolated_p = lambda3 * trigram_p + lambda2 * bigram_p + lambda1 * unigram_p
    return interpolated_p



# Function to calculate log-probabilities for unigram, bigram, and trigram models
def calculate_log_probabilities(unigram_model, bigram_model, trigram_model, total_unigrams):
    unigram_prob = {unigram: math.log(count / total_unigrams) for unigram, count in unigram_model.items()}
    bigram_prob = {bigram: math.log(count / unigram_model[(bigram[0],)]) for bigram, count in bigram_model.items()}
    trigram_prob = {trigram: math.log(count / bigram_model[(trigram[0], trigram[1])]) for trigram, count in trigram_model.items()}
    
    return unigram_prob, bigram_prob, trigram_prob

# Function to perform log-interpolation
def log_interpolate(trigram_prob, bigram_prob, unigram_prob, trigram, lambdas):
    w1, w2, w3 = trigram
    lambda3, lambda2, lambda1 = lambdas

    # Get the log-probabilities, using a default of negative infinity if not found
    trigram_log_prob = trigram_prob.get(trigram, float('-inf'))
    bigram_log_prob = bigram_prob.get((w2, w3), float('-inf'))
    unigram_log_prob = unigram_prob.get((w3,), float('-inf'))

    # Weighted sum of log-probabilities
    log_interpolated_prob = math.log(lambda3) + trigram_log_prob
    log_interpolated_prob = np.logaddexp(log_interpolated_prob, math.log(lambda2) + bigram_log_prob)
    log_interpolated_prob = np.logaddexp(log_interpolated_prob, math.log(lambda1) + unigram_log_prob)

    return log_interpolated_prob


def sample_next_word(trigram_model, bigram_model, unigram_model, w1, w2, lambdas):
    candidates = [(trigram, count) for trigram, count in trigram_model.items() if trigram[0] == w1 and trigram[1] == w2]
    
    # If no trigram found, backoff to bigram or unigram
    if candidates:
        # Calculate interpolated probabilities for candidates
        total_count = sum(count for _, count in candidates)
        probabilities = []
        for (w1, w2, w3), count in candidates:
            trigram_prob = count / total_count
            bigram_prob = bigram_model.get((w2, w3), 0)
            unigram_prob = unigram_model.get((w3,), 0)
            interpolated_prob = lambdas[0] * trigram_prob + lambdas[1] * bigram_prob + lambdas[2] * unigram_prob
            probabilities.append((w3, interpolated_prob))
        
        # Normalize probabilities to sum to 1
        total_prob = sum(p for _, p in probabilities)
        probabilities = [(word, p / total_prob) for word, p in probabilities]
        
        # Sample the next word based on the probabilities
        next_word = random.choices([w for w, _ in probabilities], [p for _, p in probabilities])[0]
        return next_word
    else:
        # If no trigram candidates, fallback to bigram or unigram generation
        bigram_candidates = [(bigram, count) for bigram, count in bigram_model.items() if bigram[0] == w2]
        if bigram_candidates:
            next_word = random.choice(bigram_candidates)[0][1]
            return next_word
        else:
            # Fallback to a random word from unigrams if no bigram matches
            next_word = random.choice(list(unigram_model.keys()))[0]
            return next_word

# Function to generate text using the trigram model
def generate_text(trigram_model, bigram_model, unigram_model, lambdas, max_length=20):
    # Start with the boundary tokens
    generated_sequence = ["<s>", "<s>"]
    
    while len(generated_sequence) < max_length:
        w1, w2 = generated_sequence[-2], generated_sequence[-1]
        next_word = sample_next_word(trigram_model, bigram_model, unigram_model, w1, w2, lambdas)
        
        if next_word == "</s>":
            break
        generated_sequence.append(next_word)
    
    return ' '.join(generated_sequence[2:])  # Omit the starting boundary tokens


# Function to calculate the log probability of a sequence using the model
def log_probability_of_sequence(sequence, trigram_prob, bigram_prob, unigram_prob, lambdas):
    # print(trigram_prob, bigram_prob, unigram_prob)
    log_prob_sum = 0
    num_trigrams = 0
    
    # Add start symbols to the sequence
    sequence = ['<s>', '<s>'] + sequence + ['</s>', '</s>']
    
    # Iterate through the trigrams in the sequence
    for i in range(len(sequence) - 2):
        trigram = (sequence[i], sequence[i+1], sequence[i+2])
        log_prob = log_interpolate(trigram_prob, bigram_prob, unigram_prob, trigram, lambdas)
        log_prob_sum += log_prob
        num_trigrams += 1
    
    return log_prob_sum, num_trigrams


# Function to calculate perplexity for a set of documents
def calculate_perplexity(test_documents, trigram_prob, bigram_prob, unigram_prob, lambdas):
    total_log_prob = 0
    total_trigrams = 0
    
    for document in test_documents:
        # Tokenize the document
        tokens = tokenize_with_boundaries(document)
        log_prob_sum, num_trigrams = log_probability_of_sequence(tokens, trigram_prob, bigram_prob, unigram_prob, lambdas)
        
        print(log_prob_sum)
        total_log_prob += log_prob_sum
        total_trigrams += num_trigrams
    
    # Calculate perplexity
    print(total_log_prob, total_trigrams)
    avg_log_prob = total_log_prob / total_trigrams
    perplexity = math.exp(-avg_log_prob)
    
    return perplexity


if __name__ == '__main__':
    df = None
    df = pd.read_csv('./datasets/train.csv')
    
    n = 1000
    t = 10
    train_items = df.sample(n=n, random_state=42)
    test_items = df.copy().drop(train_items.index).sample(n=10, random_state=42)
    train_samples = []
    test_samples = []

    i = 0
    while i < n:
        curr_path = os.path.join(
            os.getcwd(), 'datasets', 'train', train_items.iloc[i]['Id'] + '.json')

        with open(curr_path, 'r') as file:
            curr_json = json.load(file)
            train_samples.append(''.join([cj['text'] for cj in curr_json]))
        i += 1

    i = 0
    while i < t:
        curr_path = os.path.join(
            os.getcwd(), 'datasets', 'train', test_items.iloc[i]['Id'] + '.json')

        with open(curr_path, 'r') as file:
            curr_json = json.load(file)
            test_samples.append(''.join([cj['text'] for cj in curr_json]))
        i += 1

    # trigram_model = create_trigram_model(samples)
    # for t, c in trigram_model.items():
    #     if c > 1:
    #         print(t, c)
    unigram_model, bigram_model, trigram_model, total_unigrams = build_ngram_models(train_samples)

    # unigram_prob, bigram_prob, trigram_prob = calculate_probabilities(unigram_model, bigram_model, trigram_model, total_unigrams)

    # lambdas = (0.7, 0.2, 0.1)  # weights for trigram, bigram, unigram (prioritize trigram)
    # trigram = ("non-tested", "grades", "and")
    # probability = interpolate(trigram_prob, bigram_prob, unigram_prob, trigram, lambdas)
    # print(probability)

    # # Generate text using the trigram model
    # generated_text = generate_text(trigram_model, bigram_model, unigram_model, lambdas)
    # print("Generated Text: ", generated_text)

    # Sample usage: Compute log-probabilities and interpolate
    unigram_log_prob, bigram_log_prob, trigram_log_prob = calculate_log_probabilities(unigram_model, bigram_model, trigram_model, total_unigrams)

    # Example trigram to interpolate probability for: ("<s>", "<s>", "this")
    lambdas = (0.7, 0.2, 0.1)  # weights for trigram, bigram, and unigram
    trigram = ("a", "bb", "ccc")
    log_probability = log_interpolate(trigram_log_prob, bigram_log_prob, unigram_log_prob, trigram, lambdas)

    # Convert back to probability if needed (though usually you'd stay in log-space)
    probability = math.exp(log_probability)
    print(f"Log-Interpolated probability for {trigram}: {log_probability}")
    print(f"Interpolated probability for {trigram}: {probability}")

    # print(test_items)
    test_samples = [
        "the quick brown fox jumps over the lazy dog"
    ]
    perplexity_score = calculate_perplexity(test_samples, trigram_log_prob, bigram_log_prob, unigram_log_prob, lambdas)
    print(f"Perplexity Score: {perplexity_score}")

