import numpy, beagle

numpy.random.seed(37916487)

# Blocking Prime	Positive Prime	Fragment	No. of Letters	Frag. Freq.	         PP	         CP	         BP	
# Note.  Frag. Freq. = fragment solution frequency/per million.  For Fragment Completion: PP = positive prime; CP = control prime; BP = blocking prime.  For Priming: Positive = PP - CP; Blocking = BP - CP.	

# below from save_beagle_vectors2.py (bottom from block_prime_test2.py)

def remove_spaces(s):
    ch = s.split(' ')
    st = ""
    for c in ch:
        st = st + str(c)
    return st

def get_blocking_data():
    ofile = open("rass07_clean.txt", 'w')
    infile = open("Rass2007.txt", 'r')
    ofile.write("block\tprime\tfragment\tposprime\tctrlprime\tblckprime\n")
    block_primes = []
    primes = []
    fragments = []
    completion = [] # (posprime, ctrlprime, blckprime)
    for l in infile:
        w = l.strip().split('\t')
        print(w[1]+'\t'+w[2]+'\t'+w[3]+'\t'+w[6]+'\t'+w[7]+'\t'+w[8])
        block_primes.append(w[1].lower())
        primes.append(w[2].lower())
        fragments.append(remove_spaces(w[3]).lower())
        completion.append((float(w[6]), float(w[7]), float(w[8])))
        ofile.write(w[1].lower()+'\t'+w[2].lower()+'\t'+w[3].lower()+'\t'+w[6]+'\t'+w[7]+'\t'+w[8]+'\n')
    infile.close()
    ofile.close()
    return block_primes, primes, fragments, completion

def get_lexicon():
	infile = open("../ELP_HAL.txt", 'r')
	# [index] Word I_Mean_RT I_NMG_Mean_RT Freq_HAL Ortho_N Length Pos_Sum Pos_Count Mean_Pos_Sim
	raw_words = []
	for l in infile:
		w = l.split()
		if ("'" not in w[1]) and (w[1] not in raw_words):
			raw_words.append(w[1].lower())
	print "Done building corpus. Doing something else."
	infile.close()
	print(str(len(raw_words)) + " words in corpus.")
	return raw_words

B = beagle.BEAGLE(windowSize=3)

block_primes, primes, fragments, completion = get_blocking_data()

raw_words = primes + block_primes + fragments

full_lex = raw_words + get_lexicon()
B.defineLexicon(full_lex)
B.readFile("../tasa_magnus_lemmatized/tasa.cln") # "../tasa.lemma.fixed"

ind = 0
indices = {}
for w in B.words:
    if w in raw_words:
        indices[w] = ind
    ind +=1

print(indices)

env = numpy.zeros((len(raw_words), 2048))
i = 0
for w in raw_words:
    env[i] = B.env[indices[w]]
    i +=1
#numpy.savetxt('sm_env.csv', env, delimiter=';')
numpy.savez('sm_env', env)
print 'saved env vectors'

mem = numpy.zeros((len(raw_words), 2048))
i = 0
for w in raw_words:
    mem[i] = B.mem[indices[w]]
    i +=1
#numpy.savetxt('sm_mem.csv', mem, delimiter=';')
numpy.savez('sm_mem', mem)
print 'saved mem vectors'

con = numpy.zeros((len(raw_words), 2048))
i = 0
for w in raw_words:
    con[i] = B.con[indices[w]]
    i +=1
#numpy.savetxt('sm_order.csv', mem, delimiter=';')
numpy.savez('sm_order', mem)
print 'saved order vectors (con)'


# below from block_prime_test2.py
#env = numpy.loadtxt('sm_env.csv', delimiter=';')
#mem = numpy.loadtxt('sm_mem.csv', delimiter=';')
# raw_words = primes + block_primes + fragments

offset = len(primes)
of = open('blocking_sims_redo.txt', 'w')
of.write('prime\tblock\tfragment\tfrag2prim\tfrag2block\tfrag2primSem\tfrag2blockSem\tposprim\tctrlprime\tblckprime\n')
for i, prime in enumerate(primes):
    #print raw_words[i] +'\t'+ raw_words[i+offset]  +'\t'+ raw_words[i+2*offset]
    block = block_primes[i]
    print prime +'\t'+ block +'\t'+ fragments[i]
    frag_env = B.env[indices[fragments[i]]] #env[i+2*offset]
    block_env = B.env[indices[block]]
    frag2prim = str(beagle.cosine(frag_env, B.env[indices[prime]])) #str(cosine(frag_env, env[i])) 
    #print cosine(frag_env, env[i])
    frag2block = str(beagle.cosine(frag_env, block_env)) #str(cosine(frag_env, env[i+offset]))
    primSem = B.env[indices[prime]] + beagle.normalize(B.mem[indices[prime]]) #env[i]+mem[i]
    blockSem = block_env + beagle.normalize(B.mem[indices[block]]) #env[i+offset]+mem[i+offset]
    frag2primSem = str(beagle.cosine(frag_env, primSem))
    frag2blockSem = str(beagle.cosine(frag_env, blockSem))
    line = prime+'\t'+block+'\t'+fragments[i] +'\t'+ frag2prim +'\t'+ frag2block +'\t'+ frag2primSem +'\t'+ frag2blockSem  +'\t'+ str(completion[i][0]) +'\t'+ str(completion[i][1]) +'\t'+ str(completion[i][2]) +'\n'
    #print line + str(i)
    of.write(line)

of.close()
# compare block_primes to fragments and primes to fragments




"""
of = open("words.txt", 'w')
for w in B.words:
    of.write(w+'\n')
of.close()

numpy.savetxt('env.csv', B.env, delimiter=';')
numpy.savetxt('mem.csv', B.mem, delimiter=';')
numpy.savetxt('con.csv', B.con, delimiter=';')
"""
