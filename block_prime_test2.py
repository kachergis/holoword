import numpy, holoword
from holoword import cosine

def remove_spaces(s):
    ch = s.split(' ')
    st = ""
    for c in ch:
        st = st + str(c)
    return st

def get_blocking_data():
    infile = open("rass07_clean.txt", 'r')
    # block prime fragment posprime ctrlprime blckprime
    block_primes = []
    primes = []
    fragments = []
    completion = [] # (posprime, ctrlprime, blckprime)
    for l in infile:
        w = l.strip().split('\t')
        block_primes.append(w[0])
        primes.append(w[1])
        fragments.append(remove_spaces(w[2]))
        completion.append((w[3], w[4], w[5]))
    infile.close()
    return block_primes, primes, fragments, completion


block_primes, primes, fragments, completion = get_blocking_data()

raw_words = primes + block_primes + fragments
print 'Raw words: '+str(len(raw_words))
print 'Primes: '+str(len(primes))

env = numpy.loadtxt('sm_env.csv', delimiter=';')
mem = numpy.loadtxt('sm_mem.csv', delimiter=';')

offset = len(primes)
of = open('block_prime_sims2.txt', 'w')
of.write('prime\tblock\tfragment\tfrag2prim\tfrag2block\tfrag2primSem\tfrag2blockSem\tposprim\tctrlprime\tblckprime\n')
for i, prime in enumerate(primes):
    print raw_words[i] +'\t'+ raw_words[i+offset]  +'\t'+ raw_words[i+2*offset]
    block = block_primes[i]
    frag_env = env[i+2*offset]
    frag2prim = str(cosine(frag_env, env[i])) 
    #print cosine(frag_env, env[i])
    frag2block = str(cosine(frag_env, env[i+offset]))
    primSem = env[i]+mem[i]
    blockSem = env[i+offset]+mem[i+offset]
    frag2primSem = str(cosine(frag_env, primSem))
    frag2blockSem = str(cosine(frag_env, blockSem))
    line = prime+'\t'+block+'\t'+fragments[i] +'\t'+ frag2prim +'\t'+ frag2block +'\t'+ frag2primSem +'\t'+ frag2blockSem  +'\t'+ str(completion[i][0]) +'\t'+ str(completion[i][1]) +'\t'+ str(completion[i][2]) +'\n'
    #print line + str(i)
    of.write(line)

of.close()
# compare block_primes to fragments and primes to fragments


