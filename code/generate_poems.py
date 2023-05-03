import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_poem_naive_n_grams(model, tokenizer, starter_poem, max_len = 17, length = 20, deterministic= True, seed = 42, markdown = False):
    line1 = starter_poem
    while syllable_count(line1) < 5:
        token_list = tokenizer.texts_to_sequences([line1])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        if deterministic:
            np.random.seed(seed)
        probs = model.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
        # predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        line1 += " " + output_word
    line2 = ""
    while syllable_count(line2) < 7:
        token_list = tokenizer.texts_to_sequences([line1 + line2])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        if deterministic:
            np.random.seed(seed)
        probs = model.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        line2 += " " + output_word
    line3 = ""
    while syllable_count(line3) < 5:
        token_list = tokenizer.texts_to_sequences([line1 + line2 + line3])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        if deterministic:
            np.random.seed(seed)
        probs = model.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
        # predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        line3 += " " + output_word
    if markdown:
        return "\\\n".join([line1.strip(), line2.strip(), line3.strip()])
    return "\n".join([line1.strip(), line2.strip(), line3.strip()])

def generate_poem_line_by_line(model1, model2, model3, tokenizer, starter_poem, deterministic= True, seed = 42, markdown = False):
    line1 = starter_poem
    while syllable_count(line1) < 5:
        token_list = tokenizer.texts_to_sequences([line1])[0]
        token_list = pad_sequences([token_list], maxlen=5-1, padding='pre')
        if deterministic:
            np.random.seed(seed)
        probs = model1.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
        # predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        line1 += " " + output_word
    line2 = ""
    while syllable_count(line2) < 7:
        token_list = tokenizer.texts_to_sequences([line1 + line2])[0]
        token_list = pad_sequences([token_list], maxlen=12-1, padding='pre')
        if deterministic:
            np.random.seed(seed)
        probs = model2.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        line2 += " " + output_word
    line3 = ""
    while syllable_count(line3) < 5:
        token_list = tokenizer.texts_to_sequences([line1 + line2 + line3])[0]
        token_list = pad_sequences([token_list], maxlen=17-1, padding='pre')
        if deterministic:
            np.random.seed(seed)
        probs = model3.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
        # predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        line3 += " " + output_word
    if markdown:
        return "\\\n".join([line1.strip(), line2.strip(), line3.strip()])
    return "\n".join([line1.strip(), line2.strip(), line3.strip()])

def generate_poem_line_by_line_with_stops(model1, model2, model3, tokenizer, starter_poem, deterministic= True, seed = 42, markdown = False):
    line1 = starter_poem
    output_word = ""
    while output_word != "newline":
        token_list = tokenizer.texts_to_sequences([line1])[0]
        token_list = pad_sequences([token_list], maxlen=5-1, padding='pre')
        if deterministic:
            np.random.seed(seed)
        probs = model1.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
        # predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        line1 += " " + output_word
    line2 = ""
    output_word = ""
    while output_word != "newline":
        token_list = tokenizer.texts_to_sequences([line1 + line2])[0]
        token_list = pad_sequences([token_list], maxlen=12-1, padding='pre')
        if deterministic:
            np.random.seed(seed)
        probs = model2.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        line2 += " " + output_word
    line3 = ""
    output_word = ""
    while output_word != "newline":
        token_list = tokenizer.texts_to_sequences([line1 + line2 + line3])[0]
        token_list = pad_sequences([token_list], maxlen=17-1, padding='pre')
        if deterministic:
            np.random.seed(seed)
        probs = model3.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
        # predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        line3 += " " + output_word
    if markdown:
        fulllines =  "\\\n".join([line1.strip(), line2.strip(), line3.strip()])
    else:
        fulllines = "\n".join([line1.strip(), line2.strip(), line3.strip()])
    fulllines = fulllines.replace("newline", "")
    return fulllines



def syllable_count(word):
    # "source https://stackoverflow.com/questions/46759492/syllable-count-in-python"
    if word.strip() == "":
        return 0
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if count == 0:
        count += 1
    return count