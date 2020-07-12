import numpy, holoword, time

badChars = ['\r', '\n', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', '-', ',', '.', '/', ':', ';', '=', '?', '@', '[', '\\', ']', '^', '<', '>', '`', '{', '|', '}', '~']

standardStop = ['a', 'about', 'above', 'accordingly', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anywhere', 'apart', 'appear', 'appropriate', 'are', 'around', 'as', 'aside', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', 'came', 'can', 'cannot', 'cant', 'cause', 'causes', 'certain', 'changes', 'co', 'come', 'consequently', 'contain', 'containing', 'contains', 'corresponding', 'could', 'currently', 'd', 'day', 'described', 'did', 'different', 'do', 'does', 'doing', 'done', 'down', 'downwards', 'during', 'e', 'each', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'given', 'gives', 'go', 'gone', 'good', 'got', 'great', 'h', 'had', 'hardly', 'has', 'have', 'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'hither', 'how', 'howbeit', 'however', 'i', 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'kept', 'know', 'l', 'last', 'latter', 'latterly', 'least', 'less', 'lest', 'life', 'like', 'little', 'long', 'ltd', 'm', 'made', 'make', 'man', 'many', 'may', 'me', 'meanwhile', 'men', 'might', 'more', 'moreover', 'most', 'mostly', 'mr', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'near', 'necessary', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'of', 'off', 'often', 'oh', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'people', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'probably', 'provides', 'q', 'que', 'quite', 'r', 'rather', 'really', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'second', 'secondly', 'see', 'seem', 'seemed', 'seeming', 'seems', 'self', 'selves', 'sensible', 'sent', 'serious', 'seven', 'several', 'shall', 'she', 'should', 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'specified', 'specify', 'specifying', 'state', 'still', 'sub', 'such', 'sup', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'time', 'to', 'together', 'too', 'toward', 'towards', 'twice', 'two', 'u', 'under', 'unless', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'was', 'way', 'we', 'well', 'went', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'world', 'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero']

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

def normalize(a):
    '''
    Normalize a vector to length 1.
    '''
    norm2 = numpy.sum(a**2.0)
    if norm2 <= 0.0: return a
    return a / norm2**0.5

def cosine(a,b):
    '''
    Computes the cosine of the angle between the vectors a and b.
    '''
    sumSqA = numpy.sum(a**2.0)
    sumSqB = numpy.sum(b**2.0)
    if sumSqA == 0.0 or sumSqB == 0.0: return 0.0
    return numpy.dot(a,b) * (sumSqA * sumSqB)**-0.5

#def noise(a, p):
#    toChange = numpy.random.random(a.shape) < p
#    noised = a.copy()
#    noised[toChange] = numpy.random.randn(toChange.sum()) * (a.size**-0.5)
#    return normalize(noised)

def noise(a, p):
    noise = numpy.random.normal(0, a.size**-0.5, a.shape)
    noised = p * noise + (1 - p) * a
    return normalize(noised)

class RandomWordGen:
    def __init__(self, d = 1024, *args):
        self.d = d
    
    def make_rep(self, *args):
        return normalize(numpy.random.randn(self.d) * (self.d)**-0.5)

class BEAGLE:
    def __init__(self, d = 2048, windowSize = 7, segment = False, seed = None, **args):
        self.d = d
        self.windowSize = windowSize
        self.segment = segment
        if self.windowSize < 1: self.windowSize = 2
        
        self.seed = seed
        if self.seed == None: self.seed = numpy.random.randint(-32768, 32768)
        
        self.words = []     # A list of words known to BEAGLE
        self.env = numpy.zeros((0, d))  # Stores environment vectors for words
        self.con = numpy.zeros((0, d))  # Stores memory vectors for words
        self.ord = numpy.zeros((0, d))  # Stores context (order) vectors for words
        
        self.place1 = numpy.random.permutation(d)
        self.place2 = numpy.random.permutation(d)
        
        self.invplace1 = numpy.zeros((d), dtype='int')
        self.invplace2 = numpy.zeros((d), dtype='int')
        
        for i in xrange(d):
            self.invplace1[self.place1[i]] = i
            self.invplace2[self.place2[i]] = i
        
        self.phi = normalize(numpy.random.randn(d) * d**-0.5)
        
        self.wordGenArgs = args
        
        numpy.random.seed(self.seed)
        
        if 'r' in args:
            self.wordRepGen = RandomWordGen(d = d)
        else:
            self.wordRepGen = holoword.HoloWordRep(d = d, **args)
    
    def defineLexicon(self, lex, files = True, otherWords=None):
        if files:
            self.words = []

            if type(lex) == type(''):
                FIN = open(lex, 'r')
                for line in FIN:
                    line = line.strip().lower()
                    i = 0
                    while i < len(line):
                        if line[i] not in holoword.alphabet + [' ']: line = line.replace(line[i], '')
                        else: i += 1
                    #for badChar in badChars: line = line.replace(badChar, '')
                    self.words.extend(line.split())
                FIN.close()
            else:
                for filename in lex:
                    FIN = open(filename, 'r')
                    for line in FIN:
                        line = line.strip().lower()
                        i = 0
                        while i < len(line):
                            if line[i] not in holoword.alphabet + [' ']: line = line.replace(line[i], '')
                            else: i += 1
                        self.words.extend(line.split())
                    FIN.close()
        elif type(lex) == type([]):
            self.words = [word for word in lex]
        
        if otherWords!=None:
            self.words.extend(otherWords)
        
        self.words = list(set(self.words))  # Makes sure there are no repeats in the word-list
        
        self.env = numpy.zeros((len(self.words), self.d))
        self.con = numpy.zeros((len(self.words), self.d))
        self.ord = numpy.zeros((len(self.words), self.d))
        
        print 'Done allocating vectors for', len(self.words), 'words. (', time.ctime(), ')'
        
        if self.segment:
            forms = holoword.SegmentWords(self.words)
            for i, form in enumerate(forms):
                self.env[i] = self.wordRepGen.make_rep(form)
        else:
            for i, word in enumerate(self.words):
                self.env[i] = self.wordRepGen.make_rep(word)
        
        print 'Done creating environmental vectors. (', time.ctime(), ')'
    
    def readFile(self, fileName):
        if type(fileName) == type(''):
            fileName = [fileName]
        
        for fname in fileName:
            try:
                FIN = open(fname, 'r')
            except:
                print 'Could not find file \''+fname+'\'!'
                continue
            
            l = 0
            for line in FIN:
                l += 1
                if l % 500 == 0: print l, line, time.ctime()
                
                line = line.strip().lower()
                for badChar in badChars: line = line.replace(badChar, '')
                line = line.split()
                
                self.update(line)
            
            FIN.close()
    
    def update(self, charWords):
        wordInds = []
        nonStopInds = []
        
        for word in charWords:
            try:
                newIndex = self.words.index(word)
                wordInds.append(newIndex)
                if word not in standardStop: nonStopInds.append(newIndex)
            except:
                pass
        
        memToAdd = reduce(lambda a,b: a+b, [self.env[ind] for ind in nonStopInds], numpy.zeros((self.d)))
        
        for i, wordInd in enumerate(wordInds):
            if wordInd in nonStopInds: self.con[wordInd] += normalize(memToAdd - self.env[wordInd])
            else: self.con[wordInd] += normalize(memToAdd)
            
            ordInfo = numpy.zeros_like(self.ord[wordInd])
            
            for b in xrange(max(0, i - self.windowSize + 1), i+1):
                for e in xrange(i+1, min(self.windowSize + b, len(wordInds))+1):
                    if b == i and e == i+1: continue
                    ngram = reduce(lambda a, b: cconv(a[self.place1], b[self.place2]), [self.env[ind] for ind in wordInds[b:i]] + [self.phi] + [self.env[ind] for ind in wordInds[(i+1):e]])
                    #print charWords[b:i]+['_']+charWords[(i+1):e]
                    ordInfo += ngram
            
            self.ord[wordInd] += normalize(ordInfo)
    
    def sim(self, w1, w2, envWeight = 1.0, conWeight = 1.0, ordWeight = 1.0):
        try:
            ind1 = self.words.index(w1)
            env1 = self.env[ind1]
            con1 = self.con[ind1]
            ord1 = self.ord[ind1]
        except:
            env1 = self.wordRepGen.make_rep(w1)
            con1 = numpy.zeros((self.d))
            ord1 = numpy.zeros((self.d))
        
        try:
            ind2 = self.words.index(w2)
            env2 = self.env[ind2]
            con2 = self.con[ind2]
            ord2 = self.ord[ind2]
        except:
            env2 = self.wordRepGen.make_rep(w2)
            con2 = numpy.zeros((self.d))
            ord2 = numpy.zeros((self.d))
        
        return cosine(env1, env2)*envWeight + cosine(con1, con2)*conWeight + cosine(ord1, ord2)*ordWeight
    
    def probe(self, probeSeq, target=None, partial=None, envWeight = 0.5, conWeight = 0.5, ordWeight = 0.5, envNoise = 0.0, conNoise = 0.0, ordNoise = 0.0):
        if target != None:
            try:
                targetInd = self.words.index(target)
            except:
                raise ArgumentError('Could not find target "'+str(target)+'" in lexicon!')
        
        wordInds = []
        envProbe = numpy.zeros((self.d))
        conProbe = numpy.zeros((self.d))
        ordResProbe = numpy.zeros((self.d))
        #ordDecProbe = numpy.zeros((self.d))
        
        if type(probeSeq) == type(''):
            probeSeq = probeSeq.strip().split()
        
        if partial != None:
            if type(partial) == type(''):
                if partial in self.words: envProbe = self.env[self.words.index(partial)]
                else: envProbe = self.wordRepGen.make_rep(partial)
            elif type(partial) == type(numpy.array([0])):
                envProbe = normalize(partial)
        else:
            envWeight = 0.0
        
        for word in probeSeq:
            try:
                wordInds.append(self.words.index(word))
                if word not in standardStop: conProbe += self.env[wordInds[-1]]
            except:
                pass
        
        for i in xrange(max(0, len(wordInds) - self.windowSize), len(wordInds)):
            ordResProbe += reduce(lambda a,b: cconv(a[self.place1], b[self.place2]), [self.env[ind] for ind in wordInds[i:]] + [self.phi])
            #for j in xrange(max(0, len(wordInds) - self.windowSize), i):
            #    ordDecProbe += ccorr(self.ord[wordInds[i]], reduce(lambda a,b: cconv(a[self.place1], b[self.place2]), [self.env[ind] for ind in wordInds[j:i]] + [self.phi] + [self.env[ind] for ind in wordInds[(i+1):]])[self.place1])[self.invplace2]
        
        envProbe = noise(normalize(envProbe), envNoise)
        conProbe = noise(normalize(conProbe), conNoise)
        ordResProbe = noise(normalize(ordResProbe), ordNoise)
        
        weightSum = envWeight + conWeight + ordWeight
        envWeight /= weightSum
        conWeight /= weightSum
        ordWeight /= weightSum
        
        compositeProbe = envWeight * envProbe + (1 - envWeight) * normalize(conWeight * conProbe + ordWeight * ordResProbe)
        
        if target == None:
            sim = numpy.zeros((len(self.words)))
            for i in xrange(len(self.words)):
                compositeTrace = envWeight * self.env[i] + (1 - envWeight) * normalize(conWeight * self.con[i] + ordWeight * self.ord[i])
                sim[i] = cosine(compositeTrace, compositeProbe)
                #envWeight * cosine(self.env[i], envProbe) + (1 - envWeight) * cosine(compositeTrace, compositeProbe)
                # + .5 * ordWeight * (cosine(compositeTrace, ordResProbe) + cosine(self.env[i], ordDecProbe))
                #sim[i] = cosine(envWeight * self.env[i] + conWeight * self.con[i] + ordWeight * self.ord[i], compositeProbe)
        else:
            compositeTrace = envWeight * self.env[targetInd] + (1 - envWeight) * normalize(conWeight * self.con[targetInd] + ordWeight * self.ord[targetInd])
            sim = cosine(compositeTrace, compositeProbe)
            #envWeight * cosine(self.env[targetInd], envProbe) + (1 - envWeight) * cosine(compositeTrace, compositeProbe)
            # + .5 * ordWeight * (cosine(compositeTrace, ordResProbe) + cosine(self.env[targetInd], ordDecProbe))
            #sim = cosine(envWeight * self.env[targetInd] + conWeight * self.con[targetInd] + ordWeight * self.ord[targetInd], compositeProbe)

        return sim
    
    def decode(self, sequence):
        inds = []
        decoded = numpy.zeros((self.d))
        
        for i, word in enumerate(sequence):
            if word == '_':
                blankPos = i
            else:
                try:
                    inds.append(self.words.index(word))
                except:
                    pass
        
        for i, word in enumerate(sequence):
            if abs(blankPos - i) > self.windowSize:
                continue
    
    def fromFile(self, prefix):
        FIN = open(prefix + '_words.txt', 'r')
        self.words = [word.strip() for word in FIN]
        FIN.close()
        
        FIN = open(prefix + '_params.txt', 'r')
        self.windowSize = int(FIN.readline())
        self.segment = bool(FIN.readline())
        self.seed = int(FIN.readline())
        numpy.random.seed(self.seed)
        for line in FIN:
            key, t, argType, arg = line.split()
            if 'str' in argType: self.wordGenArgs[key] = arg
            elif 'int' in argType: self.wordGenArgs[key] = int(arg)
            elif 'float' in argType: self.wordGenArgs[key] = float(arg)
            elif 'list' in argType:
                arg = arg[1:-1].split(',')
                self.wordGenArgs[key] = [int(item) for item in arg]
        FIN.close()
        
        data = numpy.load(prefix + '_data.npz', 'r')
        self.env = data['env']
        self.con = data['con']
        self.ord = data['ord']
        self.phi = data['phi']
        self.place1 = data['place1']
        self.place2 = data['place2']
        self.invplace1 = data['invplace1']
        self.invplace2 = data['invplace2']
        self.d = len(self.phi)
        
        numpy.random.seed(self.seed)
        
        if 'r' in self.wordGenArgs:
            self.wordRepGen = RandomWordGen(d = self.d)
        else:
            self.wordRepGen = holoword.HoloWordRep(d = self.d, **self.wordGenArgs)
    
    def toFile(self, prefix):
        FOUT = open(prefix + '_words.txt', 'w')
        FOUT.write('\n'.join(self.words))
        FOUT.close()
        
        FOUT = open(prefix + '_params.txt', 'w')
        FOUT.write(str(self.windowSize) + '\n')
        FOUT.write(str(self.segment) + '\n')
        FOUT.write(str(self.seed) + '\n')
        for key, arg in self.wordGenArgs:
            FOUT.write(str(key) + ' ' + str(type(arg)) + ' ' + str(arg) + '\n')
        FOUT.close()
        
        savez(prefix + '_data.npz', env = self.env, con = self.con, ord = self.ord, phi = self.phi, place1 = self.place1, place2 = self.place2, invplace1 = self.invplace1, invplace2 = self.invplace2)
        
def TMB(b, stimuli, contextSizes, noiseLevels, numSims, envWeight = 1., conWeight = 1., ordWeight = 1., conNoise = 0., ordNoise = 0.):
    data = numpy.zeros((len(stimuli), len(contextSizes), len(noiseLevels)))
    
    for i, (context, target) in enumerate(stimuli):
        try: targetInd = b.words.index(target)
        except:
            print 'Target word "'+target+'" not in lexicon!'
            continue
        
        if type(context) == type(''): context = context.strip().split()
        
        goOn = True
        for word in context:
            if word not in b.words:
                print 'Context word "'+word+'" not in lexicon!'
                goOn = False
                break
        
        if goOn == False: continue
        
        for j, size in enumerate(contextSizes):
            for k, noiseLevel in enumerate(noiseLevels):
                for n in xrange(numSims):
                    results = b.probe(context[(len(context)-size):], None, None, envWeight, conWeight, ordWeight, noiseLevel, conNoise, ordNoise)
                    #data[i][j][k] += numpy.sum(results <= results[targetInd]) / float(len(results))
                    data[i][j][k] += (results[targetInd] - numpy.mean(results)) / numpy.std(results)
                    #data[i][j][k] += b.probe(context[(len(context)-size):], target, target, envWeight, conWeight, ordWeight, noiseLevel, conNoise, ordNoise)
                    #if numpy.isnan(data[i][j][k]):
                    #    print context, target, size, noiseLevel, n
                    #    break
    
    return data / float(numSims)

def TG(b, stimuli, contextSizes, numSims, toSearch, threshold = .5, envWeight = 1., conWeight = 1., ordWeight = 1., envNoise = 0., conNoise = 0., ordNoise = 0.):
    data = numpy.zeros((len(stimuli), len(contextSizes), 2))
    
    for i, (context, target) in enumerate(stimuli):
        try: targetInd = b.words.index(target)
        except:
            print 'Target word "'+target+'" not in lexicon!'
            continue
        
        if type(context) == type(''): context = context.strip().split()
        
        goOn = True
        for word in context:
            if word not in b.words:
                print 'Context word "'+word+'" not in lexicon!'
                goOn = False
                break
        
        if goOn == False: continue
        
        for j, size in enumerate(contextSizes):
            for n in xrange(numSims):
                toPick = numpy.random.randint(0, len(stimuli)-1)
                if toPick >= i: toPick += 1
                incongruent = stimuli[toPick][0]
                if type(incongruent) == type(''): incongruent = incongruent.strip().split()
                
                #results = b.probe(context[(len(context)-size):], None, target, envWeight, conWeight, ordWeight, envNoise, conNoise, ordNoise)
                #data[i][j][0] += results[targetInd] / numpy.std(results)
                for noiseLevel in toSearch:
                    sim = b.probe(context[(len(context)-size):], target, target, envWeight, conWeight, ordWeight, min(noiseLevel + envNoise, 1.0), conNoise, ordNoise)
                    if sim >= threshold:
                        data[i][j][0] += noiseLevel
                        break
                
                #results = b.probe(incongruent[(len(incongruent)-size):], None, target, envWeight, conWeight, ordWeight, envNoise, conNoise, ordNoise)
                #data[i][j][1] += results[targetInd] / numpy.std(results)
                for noiseLevel in toSearch:
                    sim = b.probe(incongruent[(len(incongruent)-size):], target, target, envWeight, conWeight, ordWeight, min(noiseLevel + envNoise, 1.0), conNoise, ordNoise)
                    if sim >= threshold:
                        data[i][j][1] += noiseLevel
                        break
    
    return data / float(numSims)

def savez(file, *args, **kwds):
    # Import is postponed to here since zipfile depends on gzip, an optional
    # component of the so-called standard library.
    import zipfile
    # Import deferred for startup time improvement
    import tempfile
    import numpy as np
    import os
    
    if isinstance(file, basestring):
        if not file.endswith('.npz'):
            file = file + '.npz'
    
    namedict = kwds
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            raise ValueError, "Cannot use un-named variables and keyword %s" % key
        namedict[key] = val
    
    zip = zipfile.ZipFile(file, mode="w", allowZip64 = True)

    # Stage arrays in a temporary file on disk, before writing to zip.
    fd, tmpfile = tempfile.mkstemp(suffix='-numpy.npy')
    os.close(fd)
    try:
        for key, val in namedict.iteritems():
            fname = key + '.npy'
            fid = open(tmpfile, 'wb')
            try:
                numpy.lib.format.write_array(fid, np.asanyarray(val))
                fid.close()
                fid = None
                zip.write(tmpfile, arcname=fname)
            finally:
                if fid:
                    fid.close()
    finally:
        os.remove(tmpfile)

    zip.close()
