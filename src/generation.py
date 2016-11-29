import sys
import math
import clusterGraphics as CG
from sklearn.cluster import KMeans
from scipy.stats import mstats
import clusterFileHandler as CFH
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
from nltk.stem.porter import PorterStemmer
import gensim
import os
import numpy as np
import networkx as netx
from networkx.drawing.nx_agraph import graphviz_layout
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import time

#Generates a dictionary based on a log of messages
def generateDictionary(logFile):
	if logFile:
		dictionary = []
		for line in logFile:
			for word in line:
				if word not in dictionary:
					dictionary.append(word)
		print 'Dictionary generated'
		return dictionary
	else:
		raise IOError

#Appends a line to the log, dictioanry, and matrix
def append(log, dictionary, matrix, avgClusters, entry):
	tokenizer = RegexpTokenizer(r'\w+')
	en_stop = get_stop_words('en')
	stemmer = PorterStemmer()
	tokens = tokenizer.tokenize(entry.decode('utf-8').lower()) #Tokenizes the line
	stop_tokens = [i for i in tokens if not i in en_stop] #Removes common words
	stem_tokens = [i for i in tokens if not i in en_stop and i.isalnum()] #Stems each word
	entry = [i for i in stem_tokens if len(i) > 1] #Removes non-alphanumeric words and single letter words
	matrixLength = len(matrix)
	count = 0

	for word in entry:
		if word not in dictionary: #If word is not in dictionary add it
			print 'Word is being added!'
			count += 1
			dictionary.append(word)
			for i in range(len(matrix)):
				matrix[i].append(0.0) #Increase size of each row of matrix

	for i in range(count):
			for i in range(len(avgClusters)):

				if(i > 0):
					if len(avgClusters[i]) < len(avgClusters[0]) and avgClusters[i] is not matrix[i]:
						avgClusters[i].append(0)
				else:
					if avgClusters[i] is not matrix[i]:
						avgClusters[i].extend([0.0])

	matrix.append((vectorizeLine(dictionary, entry)))

	log.append(entry)
	return log, dictionary, matrix, avgClusters

#Turns a log message into a vector using the dictionary
#Words from the log message must have already been appeneded to the dictionary
def vectorizeLine(dictionary, entry):
	vec = [0.0] * len(dictionary)
	for word in entry:
		vec[dictionary.index(word)] = 1
	return normalizeRow(vec)

#Vectorizes a single log message
#Returns the normalized vector
def vectorizeSingle(dictionary, entry):
	tokenizer = RegexpTokenizer(r'\w+')
	en_stop = get_stop_words('en')
	stemmer = PorterStemmer()
	tokens = tokenizer.tokenize(entry.decode('utf-8').lower()) #Tokenizes the line
	stop_tokens = [i for i in tokens if not i in en_stop] #Removes common words
	stem_tokens = [i for i in tokens if not i in en_stop and i.isalnum()] #Stems each word
	entry = [i for i in stem_tokens if len(i) > 1] #Removes non-alphanumeric words and single letter words
	print entry
	vec = [0.0] * len(dictionary)
	for word in entry:
		if word in dictionary:
			vec[dictionary.index(word)] = 1
	try:
		norm = normalizeRow(vec)	
	except:
		norm = [0.0] * len(dictionary)
	return norm

#Creates a matrix based on a dictinary and a log of messages (vecotrizes each message)
#Matrix is normalized
def generateMatrix(log, dictionary, logLength):
	matrix = [[0 for i in range(len(dictionary))]for j in range(logLength)]
	lineNum = 0
	for line in log:
		for word in line:
			matrix[lineNum][dictionary.index(word)] = 1
		matrix[lineNum] = normalizeRow(matrix[lineNum]) #Normalize matrix row
		lineNum += 1

	print 'Matrix generated'
	return matrix

#Generates a matrix of distances between rows of input matrix (Distance between vectors)
def generateSimilarties(matrix):
	logLen = len(matrix)
	simMatrix = [[0 for i in range(logLen)] for j in range(logLen)]
	for i in range(len(simMatrix)):
		for j in range(len(simMatrix[i])):
 			
			simMatrix[i][j] = np.arccos(dot(matrix[i], matrix[j]))

			if math.isnan(simMatrix[i][j]):
				simMatrix[i][j] = 0.0


	print 'Similarity matrix generated'
	return simMatrix

#Normalizes a matrix
def normalize(matrix):
	matrixLength = len(matrix)
	vecLength = len(matrix[0])
	for i in range(matrixLength):
		magnitude = mag(matrix[i])
		for j in range(vecLength):

			matrix[i][j] /= magnitude


	print 'Matrix normalization complete!'
	return matrix

#Normalizes a vector
def normalizeRow(vector):
	magnitude = mag(vector)
	for i in range(len(vector)):
		vector[i] /= magnitude
	return vector

#Calculates the magnitude of a vector
def mag(vector):
	summation = 0;
	for i in range(len(vector)):
		summation += (vector[i]*vector[i])
	return math.sqrt(summation)

#Performs a dot product on two rows of a  normalized matrix
def dot(vec1, vec2):
	dot = 0;
	vecLength = len(vec1)
	for i in range(vecLength):
		dot += (vec1[i] * vec2[i])
	return dot

#Averages all the vectors in a cluster
#Takes an array of the log vectors that are contained in the cluster
def averageCluster(cluster):
	avgVec = cluster[0] #initialize average
	for i in range(1, len(cluster)):
		avgVec = averageVectors(avgVec, cluster[i])
	return avgVec

#Averages two vectors
#Vectors must be of the same dimensionality
def averageVectors(vec1, vec2):
	if(len(vec1) != len(vec2)):
		return None
	else:
		avgVec = []
		vecLength = len(vec1)
		for i in range(vecLength):
			avg = (float(vec1[i]) + float(vec2[i])) / 2.0 #average indicies
			avgVec.append(avg)
		return avgVec

#Calculates the kmeans for a certain matrix
#Returns the labels ofthe calculation as an array
#Takes the log vectors as the matrix, not the similarity matrix
def KMeansCalc(matrix, K):
	numClusters = int(K)
	kmeans = KMeans(n_clusters=numClusters)
	kmeans.fit_predict(matrix)
	zscore,pvalue = mstats.normaltest(kmeans.labels_)
	#plotGaussianCheck(kmeans.labels_)
	'''while(pvalue < 0.05):
		numClusters += 1
		print pvalue, numClusters
		kmeans = KMeans(n_clusters=numClusters)                                                            
		kmeans.fit_predict(matrix)
		for i in kmeans.labels_:
			kmeans.labels_[i] += 1
		zscore,pvalue = mstats.normaltest(kmeans.labels_)'''
		#plotGaussianCheck(kmeans.labels_)

	return kmeans.labels_

#Prints out the topic associated with a cluster
def topicModel(log, clusters):
	topics = []
	maxCluster = max(clusters)
	#file = open('cluster_0.txt', 'w')
	text = []
	for clusterNum in range(maxCluster + 1):
		for i in range(len(clusters)):
			if clusters[i] == clusterNum:
				text.append(log[i])
		dictionary = corpora.Dictionary(text)
		corpus = [dictionary.doc2bow(items) for items in text]
		lda = gensim.models.ldamodel.LdaModel(corpus, num_topics = 2, id2word = dictionary, passes = 20)
		topics.append(lda.print_topics(num_topics = 3, num_words = 3))

	return topics

#Prints out the topics for all clusters
def graphTopics(log, clusters):
	topics = []
	for cluster in clusters:
		if len(cluster) > 0:
			text = []
			for i in range(len(cluster)):
				text.append(log[cluster[i]])
			dictionary = corpora.Dictionary(text)
			corpus = [dictionary.doc2bow(items) for items in text]
			lda = gensim.models.ldamodel.LdaModel(corpus, num_topics = 2, id2word = dictionary, passes = 20)
			topics.append(lda.print_topics(num_topics = 3, num_words = 3))

	return topics

#Analyzes the sentiment of a log message
def sentimentAnalysis(log, clusters):

	maxCluster = max(clusters)
	for clusterNum in range(maxCluster + 1):
		file = open('cluster.txt', 'w')
		for i in range(len(clusters)): #Checks the labels for the log messages
			if clusters[i] == clusterNum:
				try:
					file.write(' '.join(str(word.encode('utf-8')) for word in log[i]) + '. ') #Write the log messages to a file if they are in a certain cluster
				except:
					print log[i]
					sys.exit(1)

		file.close()
		os.system('mv cluster.txt /home/cje0613/Code/scnlp') #Move cluster log to coreNLP directory
		
		#Start sentiment analysis
		ret_dir = os.getcwd()
		os.chdir('/home/cje0613/Code/scnlp') #Change to coreNLP directory
		os.system('java -cp "*" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -file cluster.txt 1> ../CosineSim/out.txt 2> err') #Execute sentiment analysis
		os.chdir(ret_dir)
		file = open('out.txt', 'r')
		count = 0
		sentimentArr = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
		sentimentArrVals = [0] * 5
		for line in file:
			if count % 2 == 1:
				try:
					sentimentArrVals[sentimentArr.index(line.strip())] += 1 #Increases sentiment counters
				except: #Just incase some funky output is encountered
					pass 
			count += 1
		print sentimentArrVals
		print sentimentArr[sentimentArrVals.index(max(sentimentArrVals))]

#Generates a graph of the nodes
#Returns the clusters in the graph and the adjacency matrix of the graph
def graphCluster(log, matrix, logVecs):
	
	#Creates the adjacency matrix
	THRESHOLD = math.pi / 2.3
	for i in range(len(matrix)):
		for j in range(len(matrix)):
			if(matrix[i][j] < THRESHOLD):
				matrix[i][j] = 1
			else:
				matrix[i][j] = 0 

	G= netx.from_numpy_matrix(np.array(matrix)) #Creates the graph from the matrix
	nodes =  netx.connected_components(G) #List of the nodes
	clusters = [] #Holds the clusters of nodes
	for node in nodes:
		logCluster= []
		clusters.append(list(node)) #Stores the clusters generated from the graph
		

	#print properClusters
	return clusters, matrix #Returns the clusters and the adjacency matrix

#Detects clusters of log messages and assigns each message to a cluster
#Returns an array with each index being a cluster and each index within the cluster being a log message
def ACD(log, logVecs):
	avgClusters = []
	logLength = len(logVecs)
	avgLength = 0
	THRESHOLD = math.pi / 2.3
	for i in range(logLength): #Cycles through the messages in the log
		found = 0
		for j in range(avgLength): #Cycles through the averages
			if np.arccos(min(dot(logVecs[i], avgClusters[j]), 1)) < THRESHOLD: #If similar
				avgClusters[j] = averageVectors(avgClusters[j], logVecs[i])
				found = 1
		if found == 0:
			avgClusters.append(logVecs[i])
			avgLength += 1
	
	clusters = [[] for i in range(avgLength)]
	for i in range(logLength): #Cycles through log vectors
		clusterIndex = - 1
		minDist = math.pi
		vector = logVecs[i]
		for j in range(avgLength):
			distTemp = np.arccos(min(dot(avgClusters[j], vector), 1))
			if(distTemp < minDist):
				minDist = distTemp
				clusterIndex = j
		clusters[clusterIndex].append(i)
	#print clusters
	#print len(clusters)
	#clusters = [x for x in clusters if x != []]
	return clusters, avgClusters

#Appends a log entry to the clusters
def ACDAppend(log, logVecs, clusters, avgClusters, topics):
	logLength = len(logVecs)
	avgLength = len(avgClusters)
	vec = logVecs[logLength - 1]
	entry = log[logLength - 1]
	originalLength = len(clusters)
	found = 0
	for j in range(avgLength): #Cycles through the averages
		if np.arccos(min(dot(vec, avgClusters[j]), 1)) < math.pi / 2.0: #If similar
			avgClusters[j] = averageVectors(avgClusters[j], vec)
			found = 1
	if found == 0:
		avgClusters.append(vec)
		clusters.append([])
		avgLength += 1

	
	clusterIndex = -1
	minDist = math.pi
	for j in range(avgLength):
		distTemp = np.arccos(min(dot(avgClusters[j], vec), 1))
		if(distTemp < minDist):
			minDist = distTemp
			clusterIndex = j
	clusters[clusterIndex].append(logLength - 1)

	if clusterIndex < originalLength:
		print 'This new message is closest to cluster', clusterIndex,
		print 'This cluster contains:'
		print clusterIndex, len(clusters), clusters
		print len(avgClusters)
	else:
		print 'This message was not close to any cluster'
		print 'Creating new cluster based on entry'
		print 'Entry has been placed in cluster', clusterIndex

	printCluster(log, clusters[clusterIndex])

	return clusters, avgClusters

#Matches a log entry to an existing clusters
#An entry may not match with any existing clusters
#Returns the number of the cluster to which the entry was assigned
#Will be -1 if the entry does not fit with any cluster
def clusterMatch(entry, log, logVecs, clusters, avgClusters):
	logLength = len(logVecs)
	avgLength = len(avgClusters)
	vec = vectorizeSingle(dictionary, entry)
	THRESHOLD = math.pi / 2.0
	clusterIndex = -1
	minDist = math.pi
	for j in range(avgLength):
		distTemp = np.arccos(min(dot(avgClusters[j], vec), 1))
		if(distTemp < minDist and distTemp < THRESHOLD):
			minDist = distTemp
			clusterIndex = j

	if clusterIndex != -1:
		print 'This new message is closest to cluster', clusterIndex,
		print 'This cluster contains:'
		print clusterIndex, len(clusters), clusters
		print len(avgClusters)
		printCluster(log, clusters[clusterIndex])
	else:
		print 'This entry does not match any clusters!'

	return clusterIndex

#Prints the log entries in a cluster
def printCluster(log, cluster):
	for index in cluster:
		print log[index]

#TRIAL FUNCTION
#Enumerates a vector
#Converts the binary representation of the vector into decimal
def enumerate(vector):
	for i in range(len(vector)):
		if vector[i] != 0:
			vector[i] = 1
		else:
			vector[i] = 0
	vector = ''.join([str(x) for x in vector])
	num = int(vector, 2)
	return num

#Searches an array for the value that most closely matches val
def cSearch(array, val):
	if(len(array)) == 1:
		return 0
	
	split = len(array) / 2
	if(abs(int(math.log10(val)) - int(math.log10(array[split - 1])) < abs(int(math.log10(val)) - int(math.log10(array[split]))))):
		return cSearch(array[:split], val)
	elif(abs(int(math.log10(val)) - int(math.log10(array[split - 1])) > abs(int(math.log10(val)) - int(math.log10(array[split]))))):
		return split + cSearch(array[split:], val)
	else:
		if(abs(val - array[split - 1]) < abs(val - array[split])):
			return cSearch(array[:split], val)
		else:
			return split + cSearch(array[split:], val)

#Prints a matrix nicely
def printMatrix(matrix):
	for vector in matrix:
		for element in vector:
			print str(element) +'\t',
		print ''

#Runs the distributed algorithm in Go
def GDACD(matrixFile, clusterFile, threadNum):
	ret_dir = os.getcwd()
	try:
		os.system('go run ' + sys.path[0] + '/GDACD.go ' + str(matrixFile) + ' ' + str(clusterFile) + ' ' + str(threadNum))
	except Exception as E:
		print 'Something failed!', E


if __name__ == "__main__":
	
	log = CFH.loadLog(sys.argv[1])
	#DO NOT TRY TO SAVE THE LOG

	dictionary = generateDictionary(log)

	#Sets the matrix
	#matrix = generateMatrix(dictionary, len(log), log)
	#matrix = normalize(matrix)
	CFH.saveMatrix(matrix, sys.argv[2])

	GDACD(sys.argv[2], sys.argv[3], 3)
	