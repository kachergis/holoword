import numpy
#from scipy.spatial.distance import pdist, squareform
from nltk.corpus import cmudict

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_']
phonemes = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
casedAlphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def permute(a, perm, n):
    toReturn = a
    for i in xrange(n): toReturn = toReturn[perm]
    return toReturn

def hammingSim(a, b):
    h = float(numpy.sum((a > 0) * (b > 0))) / float(numpy.sum((a > 0) + (b > 0)))
    return h

def normalize(a):
    '''
    Normalize a vector to length 1.
    '''
    return a / numpy.sum(a**2.0)**0.5

def maj(p = .5, *args):
    '''
    The majority-rule operation for binary vectors.
    '''
    if len(args) == 0:
        raise ArgumentError('Need something to work with!')
    if len(args) == 1:
        argSum = args[0]
        #argSum[argSum == 0] = numpy.random.randint(0, 2, numpy.sum(argSum==0))*2.0 - 1.0
    else:
        argSum = reduce(lambda a,b: a+b, args)
        argSum[argSum == -2*p + 1] = (numpy.roll(args[0], 1) * numpy.roll(args[len(args)-1], 1))[argSum == -2*p + 1]
    
    argSum[argSum < -2*p + 1] = -1.0
    argSum[argSum > -2*p + 1] = 1.0
    
    return argSum
    
def xor(a, b):
    '''
    The X-OR operation for binary vectors.
    '''
    return -(a * b)

def entropy(p):
    p /= numpy.sum(p)
    return -numpy.dot(p, numpy.log(p))

def cconv(a, b):
    '''
    Computes the circular convolution of the (real-valued) vectors a and b.
    '''
    return numpy.fft.ifft(numpy.fft.fft(a) * numpy.fft.fft(b)).real

def ccorr(a, b):
    '''
    Computes the circular correlation (inverse convolution) of the real-valued
    vector a with b.
    '''
    return cconv(numpy.roll(a[::-1], 1), b)

def convpow(a, p):
    '''
    Computes the convolutive power of the real-valued vector a, to the
    (real-valued) power p.
    '''
    return numpy.fft.ifft(numpy.fft.fft(a)**p).real

def cosine(a,b):
    '''
    Computes the cosine of the angle between the vectors a and b.
    '''
    sumSqA = numpy.sum(a**2.0)
    sumSqB = numpy.sum(b**2.0)
    if sumSqA == 0.0 or sumSqB == 0.0: return 0.0
    return numpy.dot(a,b) * (sumSqA * sumSqB)**-0.5

def euclidean_distance(a,b):
    return numpy.sum((a - b)**2.0)**0.5

def dl(str1, str2):
    """Computes the Damerau-Levenshtein distance between the two given strings."""
    f = numpy.zeros((len(str1) + 1,len(str2) + 1), dtype='int')
    cost = 0
    
    for i in range(1, f.shape[0]): f[i][0] = i
    for j in range(1, f.shape[1]): f[0][j] = j
    
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]: cost = 0
            else: cost = 1
            f[i][j] = min(f[i - 1][j - 1] + cost, f[i - 1][j] + 1, f[i][j - 1] + 1)
            if i > 1 and j > 1 and str1[i - 1] == str2[j - 2] and str1[i - 2] == str2[j - 1]:
                f[i][j] = min(f[i][j], f[i - 2, j - 2] + cost)
    
    return f[f.shape[0]-1][f.shape[1]-1]

def ld(str1, str2):
    """Computes the Levenshtein distance between the two given strings."""
    f = numpy.zeros((len(str1) + 1,len(str2) + 1), dtype='int')
    cost = 0
    
    for i in range(1, f.shape[0]): f[i][0] = i
    for j in range(1, f.shape[1]): f[0][j] = j
    
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]: cost = 0
            else: cost = 1
            f[i][j] = min(f[i - 1][j - 1] + cost, f[i - 1][j] + 1, f[i][j - 1] + 1)
            #if i > 1 and j > 1 and str1[i - 1] == str2[j - 2] and str1[i - 2] == str2[j - 1]:
            #    f[i][j] = min(f[i][j], f[i - 2, j - 2] + cost)
    
    return f[f.shape[0]-1][f.shape[1]-1]

def extractNGrams(s, sizes = range(1,5), tr = True, onlyEnds = False, penalizeEnds = True, bothEnds = True, counts=False, spaces=True, internalMarker=False, openNGrams = False, maxGap = 1000, frontGap = False):
    '''
    tr - also employ "terminal-relative" encoding, so terminal letters are
    considered within the receptive field, no matter where the field is.
    
    onlyEnds - any n-gram greater than size 1 MUST be encoded with at least one
    terminal letter
    
    penalizeEnds - if using TR encoding, should the terminal letters count
    against the current n-gram size?
    
    bothEnds - allows n-grams to be formed using letters from both ends, not
    just one or the other.
    
    counts - return the number of n-grams in which each letter appears
    
    spaces - put an underscore in place of a gap in a non-contiguous n-gram
    
    internalMarker - place an underscore on one side (for terminal n-grams)
    or both sides (for internal n-grams) "marking" whether the n-gram is
    internal or not.
    
    openNGrams - Allows for non-contiguous n-grams.
    '''
    ngrams = set([])
    
    if counts:
        letterCount = numpy.zeros((len(s)))
        
        for i in xrange(len(s)):
            for x, size in enumerate(sizes):
                #if size > len(s): continue
                
                if i+size <= len(s) and (not onlyEnds or (onlyEnds and (size==1 or (i == 0 or i+size>=len(s))))):
                    if ('_' if internalMarker and i > 0 else '')+s[i:i+size]+('_' if internalMarker and i+size < len(s) else '') not in ngrams: letterCount[i:i+size] += 1
                    ngrams.add(('_' if internalMarker and i > 0 else '')+s[i:min(i+size, len(s))]+('_' if internalMarker and i+size < len(s) else ''))
                if tr and ((penalizeEnds and size > 1) or not penalizeEnds):
                    if bothEnds:
                        for b in xrange(max(0, i+size-len(s)), min(i+1, (size if penalizeEnds else max(sizes)))):
                            for e in xrange(min(len(s)-i, (size-b if penalizeEnds else max(sizes)))):
                                if b+e==0: continue
                                if penalizeEnds:
                                    newNGram = s[:b]+('_' if (b > 0 and i > b and spaces) or (b==0 and i > 0 and internalMarker) else '')+s[i:(i+size-b-e)]+('_' if (e > 0 and len(s)-e > i+size-b-e and spaces) or (e == 0 and i+size-b-e < len(s) and internalMarker) else '')+s[max(i+size-b-e, len(s)-e):len(s)]
                                    if newNGram not in ngrams:
                                        letterCount[:b] += 1
                                        letterCount[i:(i+size-b-e)] += 1
                                        letterCount[max(i+size-b-e, len(s)-e):len(s)] += 1
                                else:
                                    newNGram = s[:b]+('_' if (b > 0 and i > b and spaces) or (b==0 and i > 0 and internalMarker) else '')+s[i:(i+size)]+('_' if (e > 0 and len(s)-e > i+size and spaces) or (e == 0 and i+size-b-e < len(s) and internalMarker) else '')+s[max(i+size, len(s)-e):len(s)]
                                    if newNGram not in ngrams:
                                        letterCount[:b] += 1
                                        letterCount[i:(i+size)] += 1
                                        letterCount[max(i+size, len(s)-e):len(s)] += 1
                                ngrams.add(newNGram)
                    else:
                        for b in xrange(1, min(i+1, (size if penalizeEnds else max(sizes)))):
                            if penalizeEnds:
                                newNGram = s[:b]+('_' if (b > 0 and i > b and spaces) or (b==0 and i > 0 and internalMarker) else '')+s[i:(i+size-b)]
                                if newNGram not in ngrams:
                                    letterCount[:b] += 1
                                    letterCount[i:(i+size-b)] += 1
                            else:
                                newNGram = s[:b]+('_' if (b > 0 and i > b and spaces) or (b==0 and i > 0 and internalMarker) else '')+s[i:(i+size)]
                                if newNGram not in ngrams:
                                    letterCount[:b] += 1
                                    letterCount[i:(i+size)] += 1
                            ngrams.add(newNGram)
                        for e in xrange(1, min(len(s)-i, (size if penalizeEnds else max(sizes)))):
                            if penalizeEnds:
                                newNGram = s[i:(i+size-e)]+('_' if (e > 0 and len(s)-e > i+size-e and spaces) or (e == 0 and i+size-b-e < len(s) and internalMarker) else '')+s[max(i+size-e, len(s)-e):len(s)]
                                if newNGram not in ngrams:
                                    letterCount[i:(i+size-e)] += 1
                                    letterCount[max(i+size-e, len(s)-e):len(s)] += 1
                            else:
                                newNGram = s[i:(i+size)]+('_' if (e > 0 and len(s)-e > i+size and spaces) or (e == 0 and i+size-b-e < len(s) and internalMarker) else '')+s[max(i+size, len(s)-e):len(s)]
                                if newNGram not in ngrams:
                                    letterCount[i:(i+size)] += 1
                                    letterCount[max(i+size, len(s)-e):len(s)] += 1
                            ngrams.add(newNGram)
        
        return ngrams, letterCount
    else:
        for i in xrange(len(s)):
            for size in sizes:
                #if size > len(s): continue
                
                if not onlyEnds or (onlyEnds and (size==1 or (i == 0 or i+size>=len(s)))):
                    ngrams.add(('_' if internalMarker and i > 0 else '')+s[i:min(i+size, len(s))]+('_' if internalMarker and i+size < len(s) else ''))
                if tr and ((penalizeEnds and size > 1) or not penalizeEnds):
                    if bothEnds:
                        for b in xrange(min(i+1, (size if penalizeEnds else max(sizes)))):
                            for e in xrange(min(len(s)-i, (size-b if penalizeEnds else max(sizes)))):
                                if b+e==0: continue
                                if penalizeEnds:
                                    ngrams.add(s[:b]+('_' if (b > 0 and i > b and spaces) or (b==0 and i > 0 and internalMarker) else '')+s[i:(i+size-b-e)]+('_' if (e > 0 and len(s)-e > i+size-b-e and spaces) or (e == 0 and i+size-b-e < len(s) and internalMarker) else '')+s[max(i+size-b-e, len(s)-e):len(s)])
                                else:
                                    ngrams.add(s[:b]+('_' if (b > 0 and i > b and spaces) or (b==0 and i > 0 and internalMarker) else '')+s[i:(i+size)]+('_' if (e > 0 and len(s)-e > i+size and spaces) or (e == 0 and i+size-b-e < len(s) and internalMarker) else '')+s[max(i+size, len(s)-e):len(s)])
                    else:
                        for b in xrange(1, min(i+1, (size if penalizeEnds else max(sizes)))):
                            if penalizeEnds:
                                ngrams.add(s[:b]+('_' if (b > 0 and i > b and spaces) or (b==0 and i > 0 and internalMarker) else '')+s[i:(i+size-b)])
                            else:
                                ngrams.add(s[:b]+('_' if (b > 0 and i > b and spaces) or (b==0 and i > 0 and internalMarker) else '')+s[i:(i+size)])
                        for e in xrange(1, min(len(s)-i, (size if penalizeEnds else max(sizes)))):
                            if penalizeEnds:
                                ngrams.add(s[i:(i+size-e)]+('_' if (e > 0 and len(s)-e > i+size-e and spaces) or (e == 0 and i+size-b-e < len(s) and internalMarker) else '')+s[max(i+size-e, len(s)-e):len(s)])
                            else:
                                ngrams.add(s[i:(i+size)]+('_' if (e > 0 and len(s)-e > i+size and spaces) or (e == 0 and i+size-b-e < len(s) and internalMarker) else '')+s[max(i+size, len(s)-e):len(s)])
                if openNGrams:
                    thisMaxGap = maxGap if (not frontGap or (frontGap and i == 0)) else 0
                    for b in xrange(1, size):
                        for e in xrange(1, min(len(s)-i-size, thisMaxGap)+1):
                            ngrams.add(('_' if internalMarker and i > 0 else '')+s[i:i+b]+('_' if spaces else '')+s[i+b+e:i+e+size]+('_' if internalMarker and i+e+size < len(s) else ''))
                    if frontGap and size > 1 and i+size < len(s):
                        for b in xrange(1, size): ngrams.add(('_' if internalMarker and i > 0 else '')+s[i:i+size-b]+('_' if spaces else '')+s[-b:])
        
        return ngrams

class HoloWordRep:
    '''
    Spawns objects capable of generating holographic word-form representations
    in both the visual and auditory modalities. Auditory representations depend
    on the given word being present in the CMU Pronouncing Dictionary.
    '''
    
    def __init__(self, d = 1024, vis_scale = [1, 2, 4], aud_scale = [1, 2]):
        '''
        Creates a new holographic word-form representation generator.
        d - dimensionality of the representations (typically quite large)
        vis_scale - a list of the size of successive chunks in visual word
            representations (in # of letters)
        aud_scale - a list of the size of successive chunks in auditory word
            representations (in # of phonemes)
        '''
        
        self.d = d
        
        # Permutation operators that scramble the vectors in a
        # convolution operation; this makes the operation non-commutative and
        # thus allows it to encode order.
        self.place1 = numpy.random.permutation(d)
        self.place2 = numpy.random.permutation(d)
        
        self.invplace1 = numpy.zeros((d), dtype='int')
        self.invplace2 = numpy.zeros((d), dtype='int')
        
        for i in xrange(d):
            self.invplace1[self.place1[i]] = i
            self.invplace2[self.place2[i]] = i
        
        self.vis_scale = vis_scale
        self.aud_scale = aud_scale
        
        # Create random vectors representing individual letters and phonemes
        # These will be convolved and superposed to create word-form
        # representations.
        self.letters = dict(zip(alphabet, [numpy.random.randn(d) * d**-0.5 for letter in alphabet]))
        self.sounds = dict(zip(phonemes, [numpy.random.randn(d) * d**-0.5 for phoneme in phonemes]))
    
    def make_rep(self, word, modality = 'vis'):
        '''
        Returns a holographic representation of the given word in the visual
        ('v') or auditory ('a' or 's') modality. An auditory representation can
        only be created if the given word is present in the CMU Pronouncing
        Dictionary.
        '''
        
        rep = numpy.zeros(self.d)       # The representation to be returned
        word = word.strip().lower()     # Clean up the given word string
        
        # Create a visual word form representation based on the letters present
        # in the word.
        if modality.lower().startswith('v'):
            for i in xrange(len(word)):
                for scale in self.vis_scale:
                    if i+scale > len(word): continue
                
                    rep += cconv(self.letters[word[i]][self.place1], reduce(lambda a, b: cconv(a[self.place1], b[self.place2]), [self.letters[l] for l in word[i+1:i+scale]], numpy.eye(self.d)[0]))
        # Create an auditory word-form representation based on the phonemes
        # in the word (as determined from the CMU pronouncing dictionary)
        elif modality.lower().startswith('a') or modality.lower().startswith('s'):
            try:
                pronounciations = cmudict.dict()[word]
            except:
                return None
            
            # Return a representation that is the superposition of the
            # representations of all possible pronounciations
            for pronounciation in pronounciations:
                # Ignore stress for now (coded as numbers with each vowel phone)
                for p, phoneme in enumerate(pronounciation):
                    if phoneme[-1] == '0' or phoneme[-1] == '1' or phoneme[-1] == '2':
                        pronounciation[p] = phoneme[:-1]
                
                for scale in self.aud_scale:
                    if scale > len(pronounciation):
                        break
                    for i in xrange(len(pronounciation) - scale + 1):
                        toAdd = numpy.zeros(self.d)
                        toAdd[0] = 1.0
                        for j in xrange(i, i + scale):
                            try:
                                toAdd = cconv(toAdd[self.place1], self.sounds[pronounciation[j]][self.place2])
                            except:
                                pass
                        rep += toAdd# / float(len(pronounciations))
        else:
            return None
        
        return rep
    
    def probe_rep(self, probe, word, direction = 'l', modality = 'vis'):
        if type(probe) == type(''):
            #probe = reduce(lambda a,b: cconv(a[self.place1], b[self.place2]), [self.letters[i] for i in probe], numpy.eye(self.d)[0])
            probe = self.make_rep(probe, modality)
        
        if type(word) == type(''): word = self.make_rep(word, modality)
        
        if direction == 'l':
            echo = ccorr(probe[self.place1], word)
            echo = echo[self.invplace2]
        else:
            echo = ccorr(probe[self.place2], word)
            echo = echo[self.invplace1]
        
        if modality.lower().startswith('a') or modality.lower().startswith('s'):
            return sorted([(cosine(echo, phonemeRep), phoneme) for phoneme, phonemeRep in self.sounds.iteritems()])

        return sorted([(cosine(echo, letterRep), letter) for letter, letterRep in self.letters.iteritems()])

class HoloWordRep2:
    '''
    Spawns objects capable of generating holographic word-form representations
    in both the visual and auditory modalities. Auditory representations depend
    on the given word being present in the CMU Pronouncing Dictionary.
    '''
    
    def __init__(self, d = 1024, vis_scale=range(1,5), aud_scale=[1,2]):
        '''
        Creates a new holographic word-form representation generator.
        d - dimensionality of the representations (typically quite large)
        vis_scale - a list of the size of successive chunks in visual word
            representations (in # of letters)
        aud_scale - a list of the size of successive chunks in auditory word
            representations (in # of phonemes)
        '''
        
        self.d = d
        self.vis_scale = vis_scale
        self.aud_scale = aud_scale
        
        # Permutation operators that scramble the vectors in a
        # convolution operation; this makes the operation non-commutative and
        # thus allows it to encode order.
        self.place1 = numpy.random.permutation(d)
        self.place2 = numpy.random.permutation(d)
        
        self.invplace1 = numpy.zeros((d), dtype='int')
        self.invplace2 = numpy.zeros((d), dtype='int')
        
        for i in xrange(d):
            self.invplace1[self.place1[i]] = i
            self.invplace2[self.place2[i]] = i
        
        # Create random vectors representing individual letters and phonemes
        # These will be convolved and superposed to create word-form
        # representations.
        self.letters = dict(zip(alphabet, [numpy.random.randn(d) * d**-0.5 for letter in alphabet]))
        self.sounds = dict(zip(phonemes, [numpy.random.randn(d) * d**-0.5 for phoneme in phonemes]))
    
    def make_rep(self, word, modality = 'vis', segmentation = None):
        '''
        Returns a holographic representation of the given word in the visual
        ('v') or auditory ('a' or 's') modality. An auditory representation can
        only be created if the given word is present in the CMU Pronouncing
        Dictionary.
        '''
        
        rep = numpy.zeros(self.d)       # The representation to be returned
        word = word.strip().lower()     # Clean up the given word string

        if segmentation == None: segmentation = [ [word] ]
        
        # Create a visual word form representation based on the letters present
        # in the word.
        if modality.lower().startswith('v'):
            for form in segmentation:
                segReps = []

                for seg in form:
                    segRep = numpy.zeros(self.d)
                    #ngrams = extractNGrams(seg, self.vis_scale)
                    ngrams = []
                    for size in self.vis_scale:
                        if size > numpy.floor(float(len(seg)) / 2.0) + 1: continue
                        for i in xrange(len(seg)-size+1):
                            if '_' in seg[i:i+size]: continue
                            
                            ngrams.append(seg[i:i+size])
                            #for b in self.vis_scale:
                            #    if b >= size: continue
                            #    ngrams.append(seg[:b] + ('_' if i > b else '') + seg[max(i,b):i+size])
                            #for e in self.vis_scale:
                            #    if e >= size: continue
                            #    ngrams.append(seg[i:i+size] + ('_' if i+size < len(seg)-e else '') + seg[max(i+size,len(seg)-e):])
                            if seg[0] != '_':
                                if i > 0: ngrams.append(seg[0] + ('_' if i > 1 else '') + seg[i:i+size])
                                else: ngrams.append(seg[i:i+size])
                            if seg[-1] != '_':
                                if i + size < len(seg): ngrams.append(seg[i:i+size] + ('_' if i+size < len(seg)-1 else '') + seg[-1])
                                else: ngrams.append(seg[i:i+size])
                    #ngrams = [letter for letter in seg] + [seg[0] + seg[1:i] + ('_' if i < len(seg)-1 else '') + seg[-1] for i in xrange(1, len(seg))]
                    #ngrams = set(ngrams)
                    #print ngrams
                    for ngram in ngrams:
                        if len(ngram) == 1: segRep += self.letters[ngram]
                        else:
                            toAdd = reduce(lambda a,b: normalize(cconv(a[self.place1], b[self.place2])), [self.letters[l] for l in ngram])
                            #toAdd = reduce(lambda a,b: cconv(a[self.place2], b[self.place1]), [self.letters[l] for l in ngram[::-1]])
                            segRep += toAdd# / numpy.sum(toAdd**2.0)**0.5
                    segReps.append(normalize(segRep))
                
                if len(form) > 1:
                    formRep = numpy.zeros(self.d)
                    segReps = dict(zip(map(chr, range(len(segReps))), segReps))                
                    segGrams = extractNGrams(''.join(map(chr, range(len(segReps)))), range(1,len(segReps)), tr = False)
                    
                    for segGram in segGrams:
                        if len(segGram) == 1: formRep += segReps[segGram]
                        else:
                            toAdd = reduce(lambda a,b: normalize(cconv(a[self.place1], b[self.place2])), [segReps[s] for s in segGram])
                            formRep += toAdd# / numpy.sum(toAdd**2.0)**0.5
                    rep += normalize(formRep) / float(len(form))
                else:
                    rep += segReps[0]
            
            #for ngram in ngrams:
            #    rep += reduce(lambda a,b: cconv(a[self.place1], b[self.place2]), [self.letters[l] for l in ngram], numpy.eye(self.d)[0])
        # Create an auditory word-form representation based on the phonemes
        # in the word (as determined from the CMU pronouncing dictionary)
        elif modality.lower().startswith('a') or modality.lower().startswith('s'):
            pass
        else:
            return None
        
        return normalize(rep)
    
    def probe_rep(self, probe, word, direction = 'l', modality = 'vis'):
        if type(probe) == type(''): probe = self.make_rep(probe, modality)#reduce(lambda a,b: cconv(a, b), [self.letters[letter] for letter in probe], numpy.eye(self.d)[0])
        
        if type(word) == type(''): word = self.make_rep(word, modality)
        
        if direction == 'l':
            echo = ccorr(probe[self.place1], word)
            echo = echo[self.invplace2]
        else:
            echo = ccorr(probe[self.place2], word)
            echo = echo[self.invplace1]
        
        if modality.lower().startswith('a') or modality.lower().startswith('s'):
            return sorted([(cosine(echo, phonemeRep), phoneme) for phoneme, phonemeRep in self.sounds.iteritems()])

        return sorted([(cosine(echo, letterRep), letter) for letter, letterRep in self.letters.iteritems()])

class HoloWordRepAdd:
    '''
    Spawns objects capable of generating holographic word-form representations
    in both the visual and auditory modalities. Auditory representations depend
    on the given word being present in the CMU Pronouncing Dictionary.
    '''
    
    def __init__(self, d = 1024, vis_scale=range(1,5), aud_scale=[1,2]):
        '''
        Creates a new holographic word-form representation generator.
        d - dimensionality of the representations (typically quite large)
        vis_scale - a list of the size of successive chunks in visual word
            representations (in # of letters)
        aud_scale - a list of the size of successive chunks in auditory word
            representations (in # of phonemes)
        '''
        
        self.d = d
        self.vis_scale = vis_scale
        self.aud_scale = aud_scale
        
        # Permutation operators that scramble the vectors in a
        # convolution operation; this makes the operation non-commutative and
        # thus allows it to encode order.
        self.place1 = normalize(numpy.random.randn(d) * d**-0.5)
        self.place2 = normalize(numpy.random.randn(d) * d**-0.5)
        
        # Create random vectors representing individual letters and phonemes
        # These will be convolved and superposed to create word-form
        # representations.
        self.letters = dict(zip(alphabet, [normalize(numpy.random.randn(d) * d**-0.5) for letter in alphabet]))
        self.sounds = dict(zip(phonemes, [normalize(numpy.random.randn(d) * d**-0.5) for phoneme in phonemes]))
    
    def make_rep(self, word, modality = 'vis', segmentation = None):
        '''
        Returns a holographic representation of the given word in the visual
        ('v') or auditory ('a' or 's') modality. An auditory representation can
        only be created if the given word is present in the CMU Pronouncing
        Dictionary.
        '''
        
        rep = numpy.zeros(self.d)       # The representation to be returned
        word = word.strip().lower()     # Clean up the given word string
        
        if segmentation == None: segmentation = [ [word] ]
        
        # Create a visual word form representation based on the letters present
        # in the word.
        if modality.lower().startswith('v'):
            for form in segmentation:
                segReps = []
                
                for seg in form:
                    segRep = numpy.zeros(self.d)
                    #ngrams = extractNGrams(seg, range(1,max(2, len(seg))))
                    #ngrams = [letter for letter in seg] + [seg[0] + seg[1:i] + ('_' if i < len(seg)-1 else '') + seg[-1] for i in xrange(1, len(seg))]
                    ngrams = extractNGrams(seg, self.vis_scale)
                    for ngram in ngrams:
                        if len(ngram) == 1: segRep += self.letters[ngram]
                        else:
                            #toAdd = reduce(lambda a,b: cconv(a, self.place2) + cconv(b, self.place1), [self.letters[l] for l in ngram[::-1]])
                            toAdd = reduce(lambda a,b: cconv(a, self.place1) + cconv(b, self.place2), [self.letters[l] for l in ngram])
                            #toAdd = numpy.zeros((len(ngram)-1, self.d))
                            #toAdd[0] = normalize(cconv(self.letters[ngram[0]], self.place1) + cconv(self.letters[ngram[1]], self.place2))
                            #for i in xrange(1, len(ngram)-1):
                            #    toAdd[i] = normalize(cconv(toAdd[i-1], self.place1) + cconv(self.letters[ngram[i+1]], self.place2))
                            #segRep += toAdd
                            segRep += normalize(toAdd)
                    segReps.append(normalize(segRep))
                
                if len(form) > 1:
                    formRep = numpy.zeros(self.d)
                    segReps = dict(zip(map(chr, range(len(segReps))), segReps))                
                    segGrams = extractNGrams(''.join(map(chr, range(len(segReps)))), range(1, len(segReps)), tr = False)
                    
                    for segGram in segGrams:
                        if len(segGram) == 1: formRep += segReps[segGram]
                        else:
                            toAdd = reduce(lambda a,b: cconv(a, self.place1) + cconv(b, self.place2), [segReps[s] for s in segGram])
                            formRep += toAdd
                    rep += normalize(formRep) / float(len(form))
                else:
                    rep += segReps[0]
                
                #places = [convpow(self.place1, i) for i in xrange(len(ngram))]
                #rep += reduce(lambda a,(b,p): a + cconv(b, p), zip([self.letters[l] for l in ngram], places), numpy.zeros(self.d))
                #rep += reduce(lambda a,(b,p): cconv(a, cconv(b, p)), zip([self.letters[l] for l in ngram], places), numpy.eye(self.d)[0])
        # Create an auditory word-form representation based on the phonemes
        # in the word (as determined from the CMU pronouncing dictionary)
        elif modality.lower().startswith('a') or modality.lower().startswith('s'):
            pass
        else:
            return None
        
        return rep

class HoloWordRepBSC:
    '''
    Spawns objects capable of generating holographic word-form representations
    in both the visual and auditory modalities. Auditory representations depend
    on the given word being present in the CMU Pronouncing Dictionary.
    '''
    
    def __init__(self, d = 1024, p = .5, vis_scale=range(1,4), aud_scale=[1,2]):
        '''
        Creates a new holographic word-form representation generator.
        d - dimensionality of the representations (typically quite large)
        vis_scale - a list of the size of successive chunks in visual word
            representations (in # of letters)
        aud_scale - a list of the size of successive chunks in auditory word
            representations (in # of phonemes)
        '''
        
        self.d = d
        self.p = p
        self.vis_scale = vis_scale
        self.aud_scale = aud_scale
        
        # Permutation operators that scramble the vectors in a
        # convolution operation; this makes the operation non-commutative and
        # thus allows it to encode order.
        #self.place1 = numpy.random.randint(0, 2, d)* 2.0 - 1.0
        #self.place2 = numpy.random.randint(0, 2, d)* 2.0 - 1.0
        self.place1 = (numpy.random.rand(d) < p) * 2.0 - 1.0
        self.place2 = (numpy.random.rand(d) < p) * 2.0 - 1.0
        self.perm = numpy.random.permutation(d)
        
        # Create random vectors representing individual letters and phonemes
        # These will be convolved and superposed to create word-form
        # representations.
        self.letters = dict(zip(alphabet, [(numpy.random.rand(d) < p) * 2.0 - 1.0 for letter in alphabet]))
        self.sounds = dict(zip(phonemes, [(numpy.random.rand(d) < p) * 2.0 - 1.0 for phoneme in phonemes]))
    
    def make_rep(self, word, modality = 'vis', segmentation = None):
        '''
        Returns a holographic representation of the given word in the visual
        ('v') or auditory ('a' or 's') modality. An auditory representation can
        only be created if the given word is present in the CMU Pronouncing
        Dictionary.
        '''
        
        rep = []                       # The representation to be returned
        word = word.strip().lower()     # Clean up the given word string
        
        if segmentation == None: segmentation = [ [word] ]
        
        # Create a visual word form representation based on the letters present
        # in the word.
        if modality.lower().startswith('v'):
            for form in segmentation:
                segReps = []
                
                for seg in form:
                    segRep = []
                    ngrams = extractNGrams(seg, self.vis_scale)
                    for ngram in ngrams:
                        if len(ngram) == 1: segRep.append(self.letters[ngram])
                        else:
                            #toAdd = [xor(self.letters[ngram[-2]], self.place1), xor(self.letters[ngram[-1]], self.place2)]
                            toAdd = [xor(self.letters[ngram[0]], self.place1), xor(self.letters[ngram[1]], self.place2)]
                            #toAdd = [maj(xor(self.letters[ngram[0]], self.place1), xor(self.letters[ngram[1]], self.place2))]
                            #toAdd = [xor(self.letters[ngram[0]], self.place1) + xor(self.letters[ngram[1]], self.place2)]
                            #for i in xrange(len(ngram)-3, -1, -1):
                            for i in xrange(2, len(ngram)):
                                #toAdd.append(xor(toAdd[i-2], self.place1) + xor(self.letters[ngram[i]], self.place2))
                                #toAdd.append(maj(xor(toAdd[i-2], self.place1), xor(self.letters[ngram[i]], self.place2)))
                                #toAdd.append(maj(xor(maj(*toAdd), self.place1), xor(self.letters[ngram[i]], self.place2)))
                                toAdd.extend([xor(maj(self.p, *toAdd), self.place1), xor(self.letters[ngram[i]], self.place2)])
                                #toAdd.extend(map(lambda a: xor(a, self.place1), toAdd) + [xor(self.letters[ngram[i]], self.place2)])
                                #toAdd.extend([xor(self.letters[ngram[i]], self.place1), xor(maj(self.p, *toAdd), self.place2)])
                            #toAdd = map(lambda (a,i): xor(a, permute(self.place1, self.perm, i)), zip([self.letters[l] for l in ngram], range(len(ngram))))
                            #toAdd = reduce(lambda a,b: xor(a, self.place1) + xor(b, self.place2), [self.letters[l] for l in ngram])
                            #print toAdd
                            segRep.append(maj(self.p, *toAdd))
                            #segRep.extend(toAdd)
                    segReps.append(maj(self.p, *segRep))
                
                if len(form) > 1:
                    formRep = []
                    segReps = dict(zip(map(chr, range(len(segReps))), segReps))                
                    segGrams = extractNGrams(''.join(map(chr, range(len(segReps)))), range(1, len(segReps)), tr = False)
                    
                    for segGram in segGrams:
                        if len(segGram) == 1: formRep.append(segReps[segGram])
                        else:
                            toAdd = [xor(segReps[segGram[0]], self.place1), xor(segReps[segGram[1]], self.place2)]
                            for i in xrange(2, len(segGram)):
                                toAdd.extend([xor(maj(self.p, *toAdd), self.place1), xor(segReps[segGram[i]], self.place2)])
                            #toAdd = reduce(lambda a,b: xor(a, self.place1) + xor(b, self.place2), [segReps[s] for s in segGram])
                            formRep.extend(toAdd)
                    
                    rep.append(maj(self.p, *formRep) / float(len(form)))
                else:
                    rep.append(segReps[0])
                
                #places = [convpow(self.place1, i) for i in xrange(len(ngram))]
                #rep += reduce(lambda a,(b,p): a + cconv(b, p), zip([self.letters[l] for l in ngram], places), numpy.zeros(self.d))
                #rep += reduce(lambda a,(b,p): cconv(a, cconv(b, p)), zip([self.letters[l] for l in ngram], places), numpy.eye(self.d)[0])
        # Create an auditory word-form representation based on the phonemes
        # in the word (as determined from the CMU pronouncing dictionary)
        elif modality.lower().startswith('a') or modality.lower().startswith('s'):
            pass
        else:
            return None
        
        return maj(self.p, *rep)

def EvalConstraints(repType = 2, n = 1, d = 1024, vis_scale = range(1,4)):
    pairs = [('abcde', 'abcde'), ('abde', 'abcde'), ('abccde', 'abcde'), ('abcfde', 'abcde'), ('abfge', 'abcde'), ('afcde', 'abcde'), ('abgdef', 'abcdef'), ('abgdhf', 'abcdef'), ('fbcde', 'abcde'), ('abfde', 'abcde'), ('abcdf', 'abcde'), ('abdce', 'abcde'), ('badcfehg', 'abcdefgh'), ('abedcf', 'abcdef'), ('acfde', 'abcde'), ('abcde', 'abcdefg'), ('cdefg', 'abcdefg'), ('acdeg', 'abcdefg'), ('abcbef', 'abcbdef'), ('abcdef', 'abcbdef')]
    sim = numpy.zeros((len(pairs)))
    simSq = numpy.zeros((len(pairs)))
    
    for i in xrange(n):
        if repType == 1:
            h = HoloWordRep(d = d, vis_scale=vis_scale)
        elif repType == 2:
            h = HoloWordRep2(d = d, vis_scale=vis_scale)
        elif repType == 3:
            h = HoloWordRepAdd(d = d, vis_scale=vis_scale)
        else:
            h = HoloWordRepBSC(d = d, vis_scale=vis_scale)
        newData = numpy.array([cosine(h.make_rep(prime), h.make_rep(target)) for prime, target in pairs])
        sim += newData
        simSq += newData**2.0
    
    sim /= float(n)
    sd = ((simSq - n * sim**2.0) / float(n - 1))**0.5
    
    trends = [sim[0] == numpy.max(sim), sim[1] < sim[0], sim[2] < sim[0], sim[3] < sim[0], sim[4] < sim[0], sim[5] < sim[0], sim[6] < sim[0], sim[7] < sim[6], sim[8] < sim[9], sim[9] < sim[0], sim[10] < sim[9], sim[11] > sim[4], sim[12] == numpy.min(sim), sim[13] < sim[6] and sim[13] > sim[7], sim[14] < sim[5], sim[15] > numpy.min(sim), sim[16] > numpy.min(sim), sim[17] > numpy.min(sim), sim[18] > numpy.min(sim), numpy.abs(sim[19] - sim[18]) == numpy.min(numpy.abs(sim[19] - sim[:19]))]
    # (old last trend) == numpy.min(numpy.abs(numpy.tile(sim, [len(sim),1]) - numpy.transpose(numpy.tile(sim, [len(sim), 1]))) + 1000*numpy.eye(len(sim)))
    
    return sim, sd, trends

def Substitutions(length = 7, repType = 2, numsims = 1, d = 1024, vis_scale = range(1,4)):
    target = ''.join(map(chr, range(97, 97+min(length, 25))))
    primes = [target[:i]+'z'+target[i+1:] for i in xrange(len(target))]
    sim = numpy.zeros((len(primes)))
    simSq = numpy.zeros((len(primes)))
    
    for n in xrange(numsims):
        if repType == 1:
            h = HoloWordRep(d = d, vis_scale=vis_scale)
        elif repType == 2:
            h = HoloWordRep2(d = d, vis_scale=vis_scale)
        elif repType == 3:
            h = HoloWordRepAdd(d = d, vis_scale=vis_scale)
        else:
            h = HoloWordRepBSC(d = d, vis_scale=vis_scale)
        
        targetRep = h.make_rep(target)
        newData = numpy.array([cosine(h.make_rep(prime), targetRep) for prime in primes])
        sim += newData
        simSq += newData**2.0
    
    sim /= float(numsims)
    sd = ((simSq - numsims * sim**2.0) / float(numsims - 1))**0.5
    
    return sim, sd

def MakeReps(words, repType = 2, d = 1024, vis_scale=range(1,4), segment = False):
    if repType == 1:
        h = HoloWordRep(d = d, vis_scale=vis_scale)
    elif repType == 2:
        h = HoloWordRep2(d = d, vis_scale=vis_scale)
    elif repType == 3:
        h = HoloWordRepAdd(d = d, vis_scale=vis_scale)
    else:
        h = HoloWordRepBSC(d = d, vis_scale=vis_scale)
    
    if segment:
        forms = SegmentWords(words)
        reps = numpy.array([h.make_rep(word, segmentation = forms[w]) for w, word in enumerate(words)])
    else:
        reps = numpy.array([h.make_rep(word) for word in words])            # Make representations for all words in the list
    
    return reps

def SimMatrix(words, repType = 2, d = 1024, vis_scale=range(1,4), numsims=1, normalize=True, numToCompare = None, segment = False):
    if numToCompare == None:
        sim = numpy.zeros((len(words), len(words)))
    else:
        sim = numpy.zeros((len(words), numToCompare))

    for n in xrange(numsims):
        reps = MakeReps(words, repType, d, vis_scale, segment)
        
        if normalize:
            reps /= numpy.reshape(numpy.sum(reps**2.0, 1)**0.5, (len(reps), 1)) # Normalize the representations
        
        if numToCompare == None:
            sim += numpy.dot(reps, numpy.transpose(reps))    # Computes the dot product of all representations
        else:
            for i in xrange(len(words)):
                temp_sims = numpy.array([numpy.dot(reps[i], reps[j]) for j in xrange(len(words))])
                sim[i] += numpy.sort(temp_sims)[(-2):(-2-numToCompare):-1]
    
    return sim / float(numsims)

def DistMatrix(words, repType = 2, d = 1024, vis_scale=range(1,4), numsims=1, normalize=True, numToCompare = None, segment = False):
    dist = numpy.zeros((len(words), len(words)))
    
    for n in xrange(numsims):
        reps = MakeReps(words, repType, d, vis_scale, segment)
        
        if normalize:
            reps /= numpy.reshape(numpy.sum(reps**2.0, 1)**0.5, (len(reps), 1)) # Normalize the representations
        
        dist += squareform(pdist(reps))
    
    return dist / float(numsims)

def strCombos(toAdd, start=''):
    if len(toAdd) == 1: return [start+toAdd[0]]
    combos = []
    for i, item in enumerate(toAdd):
        combos.append(start + item)
        combos.extend(strCombos(toAdd[:i]+toAdd[(i+1):], start+item))
    return combos

def SegmentWords(words):
    forms = [[] for word in words]
    
    for w, word in enumerate(words):
        forms[w].extend(segmentWord(word, words[:w] + words[(w+1):]))
    
    return forms
        
def segmentWord(word, lexicon):
    segs = [[word]]
    if len(word) < 3: return segs
    
    for i in xrange(len(word)-2):
        for j in xrange(2, len(word)-i):
            if word[i:i+j] in lexicon:
                suffixes = segmentWord(word[i+j:], lexicon)
                for suffix in suffixes:
                    if i > 0: toAdd = [word[:i], word[i:i+j]]
                    else: toAdd = [word[i:i+j]]
                    toAdd.extend(suffix)
                    segs.append(toAdd)
                #if i > 0: toAdd = [word[:i], word[i:i+j]]
                #else: toAdd = [word[i:i+j]]
                #toAdd.append(word[i+j:])
                #segs.append(toAdd)
    
    i = 0
    while i < len(segs):
        if segs[i] in segs[(i+1):]:
            del segs[i]
        else:
            i += 1
    
    return segs

def TopNSim(n = 20, words = 'elp_trimmed_words.txt', repType = 2, d = 1024, vis_scale=range(1,5), numsims=1, filename=None, normalize=True, segment=False, useFreq = False):
    freq = []
    
    if type(words) == type(0):   # Then, words are the power set of the first 'words' letters
        words = strCombos([letter for letter in alphabet[:min(words, 26)]])
    if type(words) == type(''):
        word_list = []
        FIN = open(words, 'r')
        for line in FIN:
            line = line.strip().lower()
            line = line.split()
            word_list.append(line[0])
            if len(line) > 1:
                freq.append(float(line[1]))
        FIN.close()
        words = word_list
    
    if len(freq) == 0 or not useFreq: freq = numpy.ones((len(words)))
    freq = numpy.array(freq)
    
    if n >= len(words): n = len(words) - 1
    
    aboveZero = numpy.zeros((len(words)))
    allMean = numpy.zeros((len(words)))
    sorted_sim = numpy.zeros((len(words), n))
    sorted_freq = numpy.zeros((len(words), n))
    closest_words = []
    
    if len(words) < 12000:
        sim = SimMatrix(words, repType, d, vis_scale, numsims, normalize, segment=segment)
        sim -= 2.0*numpy.eye(len(sim))*sim
        for i in xrange(len(words)):
            aboveZero[i] = numpy.dot(sim[i, sim[i] > 0], freq[sim[i] > 0] / numpy.sum(freq[sim[i]>0]))
            allMean[i] = (numpy.dot(sim[i, 0:i], freq[0:i]) + numpy.dot(sim[i, (i+1):], freq[(i+1):])) / (numpy.sum(freq[0:i]) + numpy.sum(freq[(i+1):]))
            topSims = numpy.argsort(sim[i])[(len(sim[i])-1):(len(sim[i])-1-n):-1]
            sorted_sim[i] = sim[i][topSims]
            sorted_freq[i] = freq[topSims]
            closest_words.append(words[topSims[0]])
        #sorted_sim = numpy.array([numpy.sort(simrow)[(len(simrow)-1):(len(simrow)-1-n):-1] for simrow in sim])
    else:
        for s in xrange(numsims):
            reps = MakeReps(words, repType, d, vis_scale, segment)
            for i in xrange(len(words)):
                temp_sims = numpy.array([cosine(reps[i], reps[j]) for j in xrange(len(words))])
                temp_sims[i] = -1.0
                aboveZero[i] += numpy.mean(temp_sims[temp_sims > 0])
                allMean[i] += (numpy.sum(temp_sims[:i]) + numpy.sum(temp_sims[(i+1):])) / float(len(temp_sims)-1)
                sorted_sim[i] += numpy.sort(temp_sims)[(-2):(-2-n):-1]
        
        sorted_sim /= float(numsims)
        closest_words = ['NA']*len(sorted_sim)
        aboveZero /= float(numsims)
        allMean /= float(numsims)
    
    if filename != None:
        FOUT = open(filename, 'w')
        FOUT.write('Word,ClosestWord,' + ','.join(['sim'+str(i) for i in xrange(sorted_sim.shape[1])]) + ',' + ','.join(['freq'+str(i) for i in xrange(sorted_sim.shape[1])]) + ',AboveZero,AllMean\n')
        for i, word in enumerate(words):
            FOUT.write(word+','+closest_words[i]+','+ ','.join([str(s) for s in sorted_sim[i]]) + ',' + ','.join([str(s) for s in sorted_freq[i]]) +','+str(aboveZero[i])+','+str(allMean[i])+'\n')
        FOUT.close()
    
    return words, closest_words, sorted_sim, aboveZero, allMean

def TopNDist(n = 20, words = 'elp_trimmed_words.txt', repType = 2, d = 1024, vis_scale=range(1,5), numsims=1, filename=None, normalize=True, segment=False, useFreq = False):
    freq = []
    
    if type(words) == type(0):   # Then, words are the power set of the first 'words' letters
        words = strCombos([letter for letter in alphabet[:min(words, 26)]])
    if type(words) == type(''):
        word_list = []
        FIN = open(words, 'r')
        for line in FIN:
            line = line.strip().lower()
            line = line.split()
            word_list.append(line[0])
            if len(line) > 1:
                freq.append(float(line[1]))
        FIN.close()
        words = word_list
    
    if len(freq) == 0 or not useFreq: freq = numpy.ones((len(words)))
    freq = numpy.array(freq)
    
    if n >= len(words): n = len(words) - 1
    
    allMean = numpy.zeros((len(words)))
    sorted_dist = numpy.zeros((len(words), n))
    sorted_freq = numpy.zeros((len(words), n))
    closest_words = []
    
    dist = DistMatrix(words, repType, d, vis_scale, numsims, normalize, segment=segment)
    for i in xrange(len(words)):
        allMean[i] = (numpy.dot(dist[i, 0:i], freq[0:i]) + numpy.dot(dist[i, (i+1):], freq[(i+1):])) / (numpy.sum(freq[0:i]) + numpy.sum(freq[(i+1):]))
        topDists = numpy.argsort(dist[i])[1:(n+1)]
        sorted_dist[i] = dist[i][topDists]
        sorted_freq[i] = freq[topDists]
        closest_words.append(words[topDists[0]])
    
    if filename != None:
        FOUT = open(filename, 'w')
        FOUT.write('Word,ClosestWord,' + ','.join(['dist'+str(i) for i in xrange(sorted_dist.shape[1])]) + ',' + ','.join(['freq'+str(i) for i in xrange(sorted_dist.shape[1])]) + ',AllMean\n')
        for i, word in enumerate(words):
            FOUT.write(word+','+closest_words[i]+','+ ','.join([str(s) for s in sorted_dist[i]]) + ',' + ','.join([str(s) for s in sorted_freq[i]]) +','+str(allMean[i])+'\n')
        FOUT.close()
    
    return words, closest_words, sorted_dist, allMean

def OptimalFixation(words, repType = 3, d = 1024, vis_scale = range(1,5), numsims = 1, segment = False):
    if type(words) == type(0):   # Then, words are the power set of the first 'words' letters
        words = strCombos([letter for letter in alphabet[:min(words, 26)]])
    if type(words) == type(''):
        FIN = open(words, 'r')
        word_list = [line.strip().lower() for line in FIN]
        word_lengths = numpy.array([len(word) for word in words])
        FIN.close()
        words = word_list
    
    for n in xrange(numsims):
        reps = MakeReps(words, repType, d, vis_scale, segment)
        
        
