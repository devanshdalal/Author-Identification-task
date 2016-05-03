
class SchinkeStemmer:
    """
    Implements the stemming algorithm for latin texts first described in 
    Schinke et al. 1996 and further outlined by Martin Porter in his snowball
    documentation website.

    References
    -----------
    Porter, M. "The Schinke Latin stemming algorithm":
        http://snowball.tartarus.org/otherapps/schinke/intro.html

    Schinke R, Greengrass M, Robertson AM and Willett P (1996) A stemming 
        algorithm for Latin text databases. Journal of Documentation, 
        52: 172-187.
    """

    def __init__(self, text, min_token_size=4):
        self.src = text
        self.min = min_token_size
        self.QUE_PRESERVE = ['atque', 'quoque', 'neque', 'itaque', 'absque', 
            'apsque', 'abusque', 'adaeque', 'adusque', 'denique', 'deque', 
            'susque', 'oblique', 'peraeque', 'plenisque', 'quandoque', 
            'quisque', 'quaeque', 'cuiusque', 'cuique', 'quemque', 'quamque', 
            'quaque', 'quique', 'quorumque', 'quarumque', 'quibusque', 'quosque', 
            'quasque', 'quotusquisque', 'quousque', 'ubique', 'undique', 
            'usque', 'uterque', 'utique', 'utroque', 'utribique', 'torque', 
            'coque', 'concoque', 'contorque', 'detorque', 'decoque', 'excoque', 
            'extorque', 'obtorque', 'optorque', 'retorque', 'recoque', 
            'attorque', 'incoque', 'intorque', 'praetorque']

    def _longer(self, n, v):
        """
        Return the longer of two strings.
        """
        return v if len(v) > len(n) else n

    def _remove_noun_suffix(self, token):
        """
        Strip common noun suffixes from a token.
        """
        n_suffixes = ['ibus', 'ius', 'ae', 'am', 'as', 'em', 'es', 'ia',
                'is', 'nt', 'os', 'ud', 'um', 'us', 'a', 'e', 'i', 'o', 'u']

        for suff in n_suffixes:
            if token.endswith(suff):
                return token[:-len(suff)]
        else:
            return token
   
    def _noun_stem(self, token):
        """
        Return the candidate word stem for token, treating token as if it 
        were a noun.
        """
        ntoken = token[:]
        original = token[:]

        if len(ntoken) <= self.min or ntoken.lower() in self.QUE_PRESERVE:
            return ntoken

        elif len(ntoken) > self.min and ntoken.endswith('que'):
            ntoken = ntoken[:-3]

        ntoken = self._remove_noun_suffix(ntoken)

        return ntoken if len(ntoken) >= 2 else original

    def _remove_verb_suffix(self, token):
        """
        Strip common verb suffixes from a token, replacing specific suffixes 
        where applicable.
        """
        v_suffixes = ['iuntur', 'beris', 'erunt', 'untur', 'iunt', 'mini', 
        'ntur', 'stis', 'bor', 'ero', 'mur', 'mus', 'ris', 'sti', 'tis', 'tur',
        'unt',  'bo', 'ns', 'nt', 'ri', 'm', 'r', 's', 't']

        i_replace = ['iuntur','erunt','untur','iunt','unt']
        bi_replace = ['beris','bor','bo']

        for suff in v_suffixes:
            if token.endswith(suff):
                if suff in i_replace:
                    return token[:-len(suff)] + 'i'
                elif suff in bi_replace:
                    return token[:-len(suff)] + 'bi'
                elif suff == 'ero':
                    return token[:-len(suff)] + 'eri'
                else:
                    return token[:-len(suff)]
        else:
            return token

    def _verb_stem(self, token):
        """
        Return a candidate word stem for token, treating token as if it
        were a verb.
        """
        vtoken = token[:]
        original = vtoken[:]

        if len(original) <= self.min or original.lower() in self.QUE_PRESERVE:
            return vtoken

        elif len(vtoken) > self.min and vtoken.endswith('que'):
            vtoken = vtoken[:-3]

        vtoken = self._remove_verb_suffix(vtoken)

        return vtoken if len(vtoken) >= 2 else original

    def stem(self):
        """
        Create candidate stems for all words
        """
        stemmed = [token.replace('j','i').replace('v','u') for token in self.src]

        for idx, word in enumerate(stemmed):
            stemmed[idx] = self._longer(self._noun_stem(word), self._verb_stem(word))

        return stemmed
