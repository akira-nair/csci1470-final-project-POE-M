import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_poem_naive_n_grams(model, tokenizer, starter_poem: str, max_len: int = 17, deterministic: bool = True, seed: int = 42, markdown: int = False)->str:
    """
    Generates a poem for naive n-gram model.

    Args:
        model: Tensorflow model trained
        tokenizer: Tokenizer used for training
        starter_poem (str): A primer that serves as context to generate the rest
        of the haiku
        max_len (int, optional): Max length used for model. Defaults to 17.
        deterministic (bool, optional): If true, generates poem based on seed. If false, 
        uses a random seed. Defaults to True.
        seed (int, optional): Seed for sampling. Defaults to 42.
        markdown (int, optional): If true, returns text in format for markdown file. 
        Defaults to False.

    Returns:
        str: Generated poem.
    """
    line1 = starter_poem
    # generate line 1 words until # of syllables reaches 5
    while syllable_count(line1) < 5:
        token_list = tokenizer.texts_to_sequences([line1])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        if deterministic:
            np.random.seed(seed)
        # get probabilities across corpus
        probs = model.predict(token_list, verbose=0)[-1]
        # sample from the probability distribution
        # this gets an index between 0 and vocab_size - 1
        predicted = np.random.choice(len(probs), p=probs)
        # predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        # loop through the dictionary mapping words to indices
        for word, index in tokenizer.word_index.items():
            # find the match, and return the key (word)
            if index == predicted:
                output_word = word
                break
        line1 += " " + output_word
    # repeat for lines 2 and 3...
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
    # separate lines with \n
    return "\n".join([line1.strip(), line2.strip(), line3.strip()])

def generate_poem_line_by_line(model1, model2, model3, tokenizer, starter_poem, deterministic= True, seed = 42, markdown = False) -> str:
    """
    Generates a poem for line by line model, without stops.

    Args:
        model1: Tensorflow model trained on line 1.
        model2: Tensorflow model trained on line 2.
        model3: Tensorflow model trained on line 3.
        tokenizer: Tokenizer used for training
        starter_poem (str): A primer that serves as context to generate the rest
        of the haiku
        deterministic (bool, optional): If true, generates poem based on seed. If false, 
        uses a random seed. Defaults to True.
        seed (int, optional): Seed for sampling. Defaults to 42.
        markdown (int, optional): If true, returns text in format for markdown file. 
        Defaults to False.

    Returns:
        str: Generated poem.
    """
    line1 = starter_poem
    # Similar structure to naive, except calling the corresponding model
    # when generating each line
    while syllable_count(line1) < 5:
        token_list = tokenizer.texts_to_sequences([line1])[0]
        token_list = pad_sequences([token_list], maxlen=5-1, padding='pre')
        if deterministic:
            np.random.seed(seed)
        probs = model1.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
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
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        line3 += " " + output_word
    if markdown:
        return "\\\n".join([line1.strip(), line2.strip(), line3.strip()])
    return "\n".join([line1.strip(), line2.strip(), line3.strip()])

def generate_poem_line_by_line_with_stops(model1, model2, model3, tokenizer, starter_poem, deterministic= True, seed = 42, markdown = False) -> str:
    """
    Generates a poem for line by line model, without stops.

    Args:
        model1: Tensorflow model trained on line 1.
        model2: Tensorflow model trained on line 2.
        model3: Tensorflow model trained on line 3.
        tokenizer: Tokenizer used for training
        starter_poem (str): A primer that serves as context to generate the rest
        of the haiku
        deterministic (bool, optional): If true, generates poem based on seed. If false, 
        uses a random seed. Defaults to True.
        seed (int, optional): Seed for sampling. Defaults to 42.
        markdown (int, optional): If true, returns text in format for markdown file. 
        Defaults to False.

    Returns:
        str: Generated poem.
    """
    line1 = starter_poem
    output_word = ""
    # instead of using syllable counts to cut lines,
    # we wait until the model generates "newline"
    while output_word != "newline":
        token_list = tokenizer.texts_to_sequences([line1])[0]
        token_list = pad_sequences([token_list], maxlen=5-1, padding='pre')
        if deterministic:
            np.random.seed(seed)
        probs = model1.predict(token_list, verbose=0)[-1]
        predicted = np.random.choice(len(probs), p=probs)
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



def syllable_count(word: str) -> int:
    """
    Calculates syllable count of a string

    Args:
        word (str): input string

    Returns:
        int: number of syllables
    Code source:
      https://stackoverflow.com/questions/46759492/syllable-count-in-python
    """

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