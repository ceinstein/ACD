import argparse
import generation as gen
import clusterGraphics as CG
import clusterFileHandler as CFH
import sys


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Cluster Argument Handler')
	LOG = []
	DICT = []
	MATRIX = []
	SIM = []
	CLUSTERS = []
	TOPICS = []
	LABELS = []
	THREADS = 1

	parser.add_argument('-n', action="store", dest="name") #Specifies the name of the log file to cluster (without the .log)
	parser.add_argument('-t', action="store", dest="threadNum") #Specifies the number of threads for the distribution
	parser.add_argument('-s', action="store_true", default=False) #Save topics
	parser.add_argument('-g', action='store_true', default=False) #Run go algorithm
	parser.add_argument('-a', action='store_true', default=False) #Full Runthrough of pipeline
	parser.add_argument('-l', action='store_true', default=False) #Generate dictionary and matrix

	path = sys.path[0]

	r = parser.parse_args()
	logFile = path + '/../Logs/' + r.name + '.log'
	dictFile = path + '/../Dictionaries/' + r.name + '.dict'
	matrixFile = path + '/../Matricies/' + r.name + '.matrix'
	simFile = path + '/../Matricies/' + r.name + '.smatrix'
	clusterFile = path + '/../Clusters/' + r.name + '.clstrs'
	topicFile = path + '/../Topics/' + r.name + '.tpcs'
	labelsFile = path + '/../Labels/' + r.name + '.lbls'

	if r.threadNum:
		THREADS = int(r.threadNum)

	if r.g: #Skip to algorithm
		try:
			gen.GDACD(matrixFile, clusterFile, THREADS)
		except Exception as E:
			print 'Algorithm needs a matrix file, cluster file, and a certain thread number'
			print E

	if r.s: #Save topics
		try:
			LOG = CFH.loadLog(logFile)
			CLUSTERS = CFH.loadClusters(clusterFile)
			TOPICS = gen.graphTopics(LOG, CLUSTERS)
			CFH.saveTopics(topicFile, TOPICS)
		except Exception as E:
			print 'Need a log file and a cluster file to save topics'
			print E


	if r.l: #Generates dictionary and matrix
		try:
			LOG = CFH.loadLog(logFile)
			DICT = gen.generateDictionary(LOG)
			try:
				CFH.saveDictionary(dictFile, DICT)
			except Exception as E:
				print 'Could Not save the dictionary'
				print E
			MATRIX = gen.generateMatrix(LOG, DICT, len(LOG))
			CFH.saveMatrix(matrixFile, MATRIX)
		except Exception as E:
			print 'Need a log file to generate matrix and dictionary'
			print E


	if r.a: #Full runthrough of pipeline
		try:
			LOG = CFH.loadLog(logFile)
			DICT = gen.generateDictionary(LOG)
			try:
				CFH.saveDictionary(dictFile, DICT)
			except Exception as E:
				print 'Could Not save the dictionary'
				print E
			MATRIX = gen.generateMatrix(LOG, DICT, len(LOG))
			CFH.saveMatrix(matrixFile, MATRIX)
			gen.GDACD(matrixFile, clusterFile, THREADS)
			CLUSTERS = CFH.loadClusters(clusterFile)
			TOPICS = gen.graphTopics(LOG, CLUSTERS)
			CFH.saveTopics(topicFile, TOPICS)
		except Exception as E:
			print 'Need a log file for the pipeline'
			print E
