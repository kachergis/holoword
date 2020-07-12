import numpy, beagle

# Blocking Prime	Positive Prime	Fragment	No. of Letters	Frag. Freq.	         PP	         CP	         BP	
# Note.  Frag. Freq. = fragment solution frequency/per million.  For Fragment Completion: PP = positive prime; CP = control prime; BP = blocking prime.  For Priming: Positive = PP - CP; Blocking = BP - CP.	

#block_primes = ['analogy', 'brigade', 'cottage', 'charter', 'cluster', 'crumpet', 'density', 'fixture', 'holster', 'tonight', 'trilogy', 'voyager']
#primes = ['allergy', 'baggage', 'catalog', 'charity', 'country', 'culprit', 'dignity', 'failure', 'history', 'tangent', 'tragedy', 'voltage']
#fragments = ['a_l__gy', 'b_g_a_e', 'c_ta__g', 'char_t_', 'c_u_tr_', 'cu_p__t', 'd__nity', 'f_i_ure', 'h_st_r_', 't_ng__t', 'tr_g__y', 'vo__age']

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
        block_primes.append(w[1])
        primes.append(w[2])
        fragments.append(remove_spaces(w[3]))
        completion.append((float(w[6]), float(w[7]), float(w[8])))
        ofile.write(w[1]+'\t'+w[2]+'\t'+w[3]+'\t'+w[6]+'\t'+w[7]+'\t'+w[8]+'\n')
    infile.close()
    ofile.close()
    return block_primes, primes, fragments, completion

def get_lexicon():
	infile = open("ELP_HAL.txt", 'r')
	# [index] Word I_Mean_RT I_NMG_Mean_RT Freq_HAL Ortho_N Length Pos_Sum Pos_Count Mean_Pos_Sim
	raw_words = []
	for l in infile:
		w = l.split()
		if ("'" not in w[1]) and (w[1] not in raw_words):
			raw_words.append(w[1])
	print "Done building corpus. Doing something else."
	infile.close()
	print(str(len(raw_words)) + " words in corpus.")
	return raw_words

B = beagle.BEAGLE(windowSize=3)

block_primes, primes, fragments, completion = get_blocking_data()

raw_words = primes + block_primes + fragments

full_lex = raw_words + get_lexicon()
B.defineLexicon(full_lex)
B.readFile("tasa_magnus_lemmatized/tasa.cln")

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
numpy.savetxt('sm_env.csv', env, delimiter=';')
print 'saved env vectors'

mem = numpy.zeros((len(raw_words), 2048))
i = 0
for w in raw_words:
    mem[i] = B.mem[indices[w]]
    i +=1
numpy.savetxt('sm_mem.csv', mem, delimiter=';')
print 'saved mem vectors'

con = numpy.zeros((len(raw_words), 2048))
i = 0
for w in raw_words:
    mem[i] = B.con[indices[w]]
    i +=1
numpy.savetxt('sm_order.csv', mem, delimiter=';')
print 'saved order vectors (con)'

"""
of = open("words.txt", 'w')
for w in B.words:
    of.write(w+'\n')
of.close()

numpy.savetxt('env.csv', B.env, delimiter=';')
numpy.savetxt('mem.csv', B.mem, delimiter=';')
numpy.savetxt('con.csv', B.con, delimiter=';')
"""
