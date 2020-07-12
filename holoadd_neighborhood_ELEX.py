''' 
uses Greg's new holoword.py 
to make principled representations of words

now we want to look at what schemes for making orthographic
representations (env. vectors for words) are closely correlated
to Levenshtein distance

It would probably make sense to have it be able
to print nearest neighbors to environmental vectors so that we can
show how the nearest neighbors of a word are all words that look very
orthographically similar to it. And it might be good if it can, given
a word w, report the average orthographic similarity of the X most
orthographically similar words to w, so that you could calculate a
metric analogous to Yarkoni Balota & Yap's OLD20 and see if it
correlates with lexical decision and naming times. 
'''

from holoword import *

h = HoloWordRep2()

NUM_CLOSEST = 20

class Word:
	def __init__(self, word, meanRT, nmg_meanRT, freq_HAL, ortho_N, length):
		self.string = word
		self.meanRT = meanRT
		self.nmg_meanRT = nmg_meanRT
		self.freq_HAL = freq_HAL
		self.ortho_N = ortho_N
		self.length = length
		#self.aud_rep = h.make_rep(word, 'a')
		self.vis_rep = h.make_rep(word, 'v')
		self.closest = [(-1,'5','NA')] # sorted list of tuples: (cos_dist, lev_dist, word)
		self.pos_sum = 0.0 # summed cos similarity to all >0 similarity words
		self.pos_count = 0 # number of other words with positive similarity
	
	def update_closest(self, word):
		if word.string==self.string:
			return
		sim = cosine(word.vis_rep, self.vis_rep)
		
		if sim > 0:
			self.pos_sum += sim
			self.pos_count += 1
		if sim > self.closest[len(self.closest)-1][0] and (sim, word.string) not in self.closest:
			self.closest.append((sim, word.string))
			self.closest.sort(reverse=True)
			if len(self.closest)>NUM_CLOSEST:
				#self.closest = self.closest[:len(self.closest)-1]
				del self.closest[-1]
			#word.update_closest(self.word) # you were close to me, I'm close to you...
	
	def get_closest(self):
		out = map(lambda(sim, lev, w): str(sim)+"\t"+w+"\t"+self.string, self.closest)
		return out
	
	def get_line(self):
		m = 0
		for c in self.closest:
			m += c[0]
		m = m / len(self.closest)
		mean_pos = float(self.pos_sum) / float(self.pos_count)
		out = self.string+'\t'+str(m)+'\t'+self.closest[0][1]+'\t'+str(self.closest[0][0])+'\t'+str(self.meanRT)+'\t'+str(self.nmg_meanRT)+'\t'+str(self.freq_HAL)+'\t'+str(self.ortho_N)+'\t'+str(self.length)+'\t'+str(self.pos_sum)+'\t'+str(self.pos_count)+'\t'+str(mean_pos)
		return out

infile = open("../ELP_HAL.txt", 'r')
# [index] Word I_Mean_RT I_NMG_Mean_RT Freq_HAL Ortho_N Length Pos_Sum Pos_Count Mean_Pos_Sim

raw_words = []
corpus = []
for l in infile:
	w = l.split()
	if ("'" not in w[1]) and (w[1] not in raw_words):
		corpus.append(Word(w[1],w[2],w[3],w[4],w[5],w[6]))
		raw_words.append(w[1])
print "Done building corpus. Finding orthographic neighbors."
infile.close()
print(str(len(corpus)) + " words in corpus.")

for i in range(len(corpus)):
	if i%10==0:
		print "i: " +str(i)
	for j in range(len(corpus)):
		#print " j: "+str(j)
		if i!=j:
			corpus[i].update_closest(corpus[j])

outfile = open("holo2_neighborhood_ELEX.txt", 'w')
outfile.write("Word\tMeanCosSim\tClosestWord\tClosestWordSim\tmeanRT\tnmg_meanRT\tfreq_HAL\tortho_N\tlength\tpos_sum\tpos_count\tmean_pos_sim\n")
for word in corpus:
	line = word.get_line()
	outfile.write(line+'\n')
