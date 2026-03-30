import random
import string

words = []
for i in range(3000):
    w = ""
    for j in range(random.randint(3, 8)):
        w += random.choice(string.ascii_lowercase)
    words.append(w)

def make_sentence():
    """
    Make a s.strip.
    """
    s = ""
    n = random.randint(6, 14)
    for i in range(n):
        s += random.choice(words) + " "
    s = s.strip()
    return s.capitalize() + "."

def make_paragraph():
    """
    Make a p.strip.
    """
    p = ""
    for i in range(random.randint(4, 8)):
        p += make_sentence() + " "
    return p.strip()

def make_block():
    """
    make_block.
    """
    b = ""
    for i in range(5):
        b += make_paragraph() + "\n\n"
    return b

def transform(text):
    """
    transform text into a text.
    
    Args:
        text: text.
    """
    t = text.upper()
    parts = t.split()
    random.shuffle(parts)
    return " ".join(parts)

def generate_data(n):
    """
    n returns a list of the recursors.
    
    Args:
        n: n.
    """
    raw = []
    processed = []
    for i in range(n):
        p = make_paragraph()
        raw.append(p)
        processed.append(transform(p))
    return raw, processed

# Counts words.
# Body: 2 loop(s), 1 conditional(s).
# Cyclomatic complexity: 4 (moderate).
def count_words(texts):
    """
    Returns a number of texts.
    
    Args:
        texts: texts.
    """
    freq = {}
    for t in texts:
        for w in t.split():
            if w not in freq:
                freq[w] = 0
            freq[w] += 1
    return freq

def top_words(freq):
    """
    Return a list of items.sort.
    
    Args:
        freq: freq.
    """
    items = list(freq.items())
    items.sort(key=lambda x: -x[1])
    return items[:20]

def lengths(texts):
    """
    Returns the length of the text.
    
    Args:
        texts: texts.
    """
    arr = []
    for t in texts:
        arr.append(len(t))
    return arr

raw, processed = generate_data(100)

freq = count_words(processed)
top = top_words(freq)
lens = lengths(processed)

print("Top words:")
for w, c in top:
    print(w, c)

print("\nLengths:")
print(lens[:20])

print("\nSample:\n")
print(make_block())