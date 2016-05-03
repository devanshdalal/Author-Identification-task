class NaiveStemmer:
    def __init__(self, tokens, strip=3, offset=3):
        """
        Set stemming preferences.

        Parameters 
        ----------
        tokens (list): a list of candidate tokens for stemming. The naive 
            stemmer provides no validation, so users must ensure that 
            tokens are as desired before passing them to the stemmer.

            NOTE: Ensure that this is not passed as a list of lists, as
            no flattening is applied.

        offset (int, optional): number of characters beyond 'strip' to 
            require before a token can be stemmed. Ex: offset=3 and 
            strip=3 means that a token must be 6 chars or longer before
            stemming is applied.

        strip (int, optional): number of characters to strip to create
            word stems.
        """
        self.offset = offset
        self.strip = strip
        self.tokens = tokens

    def stem(self):
        """ 
        Remove a fixed number of characters from any token of 
        sufficient length.

        Returns
        --------
        A list containing one possibly-stemmed token for each token in
        the input tokens.
        """
        return [token[:-self.strip] if len(token) >= self.strip + self.offset else token for token in self.tokens]