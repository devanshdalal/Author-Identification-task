import hashlib
import re

def sanitize(text):
    """
    Return a list of tuples from text of the form (i, c) where 
    i is the index of an individual character within text
    and c is the character itself.
    """
    tuples = zip(xrange(len(text)), text)
    exp = re.compile(r'[^a-zA-Z]')
    return [(t[0], t[1].lower()) for t in tuples if exp.match(t[1]) == None]


def kgram_gen(sanitized_text, k=5):
    """
    A generator that loops through sanitized_text and returns character
    tuples k at a time.
    """
    for x in xrange(len(sanitized_text)-k+1):
        yield sanitized_text[x:x+k]

def hash(chars):    
    """
    Return a hashed value for some number of chars.
    """
    hs = hashlib.sha1(chars.encode('utf-8'))
    hs = hs.hexdigest()[-4:]
    return int(hs, 16)


def compute_kgram_hashes(sanitized, kgram=5):
    """
    Compute all k-gram hashes over a list of sanitized k-grams and return a
    list of tuples containing the original text character indicies, values and
    hahed value for each k-gram.
    """
    final = []
    for kg in kgram_gen(sanitized, k=kgram):
        unpacked = zip(*kg)
        final.append([unpacked[0], unpacked[1], hash(''.join(unpacked[1]))])

    return final

def winnow_kgrams(hashed_kgrams, window=4):
    """
    Cycle through window kgrams at a time in intervals and take the smallest
    kgram hash in each window as the fingerprint if that hash is not the same 
    hash that was identified as the last fingerprint.

    Note: this is where issues with hash collisions could impact results,
    and we could implement a different hashing algorithm by hand or from 
    another library, but I would be shocked if this ended up being a big 
    enough problem to justify the expense as the project is concieved now.

    Return a list of hashes representing a fingerprint identity for the
    document.
    """
    fingerprints = []
    for k in xrange(len(hashed_kgrams)-window+1):
        min_hash = min(hashed_kgrams[k:k+window], key=lambda x: x[2])
        try:
            if fingerprints[-1] != min_hash:
                fingerprints.append(min_hash)
        except IndexError:
                fingerprints.append(min_hash)

    return fingerprints

def winnow(text, k=5, w=4):
    """
    Implement the winnowing algorithm as described in Schleimer, Wilkerson
    and Aiken 2003.

    Parameters
    ----------
    k (default 5): the size of each k-gram (in characters) to hash 
    w (default 4): the size of each window (in k-grams) to fingerprint
    """
    sanitized = sanitize(text)
    kgram_hashes = compute_kgram_hashes(sanitized, kgram=k)

    return winnow_kgrams(kgram_hashes, window=w)
    