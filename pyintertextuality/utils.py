import os
import re
import sys
from fuzzywuzzy import fuzz

def read_source_file(infile):
    if not os.path.isfile(infile):
        sys.exit('Source file not found: \'{}\''.format(infile))
    
    with open(infile,'rU') as inf:
        return inf.read().replace('\r\n',' ').replace('\n',' ')

def remove_duplicate_matches(speech1, speech2, result_pairs, str_threshold = 80):
    ok_matches = []
    prev_cycle = None

    for idx, (t1st, t1end, t2st, t2end) in enumerate(result_pairs):
        exp = re.compile(r'[^a-zA-Z]')
        if idx == 0:
            ok_matches.append(((t1st, t1end, t2st, t2end)))
            prev_cycle = (t1st, t1end, t2st, t2end)
        elif (fuzz.ratio(re.sub(exp, '', speech1[t1st:t1end+1]), re.sub(exp, '', speech1[prev_cycle[0]:prev_cycle[1]+1])) < str_threshold) and \
             (fuzz.ratio(re.sub(exp, '', speech2[t2st:t2end+1]), re.sub(exp, '', speech1[prev_cycle[2]:prev_cycle[3]+1])) < str_threshold):
                ok_matches.append((t1st, t1end, t2st, t2end))
                prev_cycle = (t1st, t1end, t2st, t2end)
        else:
            prev_cycle = (t1st, t1end, t2st, t2end)
    
    return ok_matches
