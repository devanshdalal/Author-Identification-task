from __future__ import division
import sys
import hashlib
from time import time
import math as m
from collections import OrderedDict

class FingerprintMatcher:
    """
    Acknowledgements
    ----------------
    With thanks to David Harvey for his suggestions for improving
    the matching algorithm.
    """
    def __init__(self, fingerprint1, fingerprint2, threshold=5, progress=True):
        if len(fingerprint1) == 0 or len(fingerprint2) == 0:
            sys.exit('FAILED: Attempted to recieve 0-length fingerprint.')
        self.fp1 = fingerprint1
        self.fp2 = fingerprint2
        self.threshold = threshold
        self.progress = progress

    def __repr__(self):
        return "\n-- FINGERPRINT MATCHER -- \
                \nText 1 fingerprinted length: {} \
                \nText 2 fingerprinted length: {} \
                \nFingerprint match threshold: {}".format(len(self.fp1), len(self.fp2), self.threshold) 

    def _hash(self, chunk):
        hs = hashlib.sha1(chunk.encode('utf-8')).hexdigest()
        return int(hs, 16)

    def _assemble_hash_dict(self, fp):
        """
        Given a fingerprinted text 'fp',  return an ordered dict of hash values
        based on the combined hash values of each fingerprint set [x:x+threshold] 
        within 'fp'.

        TODO: Explain this better - probably needs to be a diagram somewhere of this
            data structure.
        """
        composite_hashes = OrderedDict()

        for x in xrange(len(fp)-self.threshold):
            cphash = self._hash(''.join([str(v[2]) for v in fp[x:x+self.threshold]]))
            
            if cphash not in composite_hashes:
                composite_hashes[cphash] = [(fp[x][0][0], fp[x+self.threshold][0][-1])]
            else:
                composite_hashes[cphash].append((fp[x][0][0], fp[x+self.threshold][0][-1]))

        return composite_hashes

    def match(self):
        """
        Return a list of matches of length 'threshold' between text fingerprints 
        fp1 and fp2, identified by their start and end position in the two 
        original comparison texts.
        """
        #print 'Threshold is {}'.format(self.threshold)
        ngram_hash_dict_1 = self._assemble_hash_dict(self.fp1)
        ngram_hash_dict_2 = self._assemble_hash_dict(self.fp2)

        results_list = []
        start_time = time()
        if self.progress:
            import pyprind
            prbar = pyprind.ProgBar(
                                len(ngram_hash_dict_1), 
                                stream=sys.stdout, 
                                track_time=False
                              )

        for hash_key, location_list in ngram_hash_dict_1.items():
            if hash_key in ngram_hash_dict_2:
                results_list += [
                    (item[0], item[1], match[0], match[1]) 
                        for match in ngram_hash_dict_2[hash_key] 
                            for item in location_list                   
                    ]
            if self.progress:
                prbar.update()

        return results_list

    def _within(self, n, index_1, index_2):
        if m.fabs(index_1 - index_2) <= n:
            return True
        else:
            return False

    def naive_reduce_result_complexity(self, results, diff=5):
        summary_counts = []
        for result in results:
            s = {
            'st': result[0],
            'en': result[1],
            'match': 1
            }
            if len(summary_counts) == 0:
                summary_counts.append(s)
            else:
                for summ in summary_counts:
                    if self._within(diff, summ['st'], s['st']) and self._within(diff, summ['en'], s['en']):
                        summ['match'] += 1
                        break
                else: 
                    summary_counts.append(s)
        return summary_counts


