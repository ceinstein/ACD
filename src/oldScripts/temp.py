from threading import Thread
import threading
import time
import subprocess
import sys
import cosSimFileHandler as CSFH
import cosSim as cSim
import Queue as Q
import math
import numpy as np

DONE = 0
LOCK = threading.Lock()
MAXTHREADS = 0
WORKERS = Q.Queue(maxsize=MAXTHREADS)
WORKERS2 = Q.Queue(maxsize=MAXTHREADS)
AVERAGES = [[]]
CLUSTERS = [[]]#Class that handles the jobs
LOGLEN = 0


def distributedACD(log, dictionary, matrix, nThreads):
	global LOCK
	global WORKERS
	global WORKERS2
	global AVERAGES
	global CLUSTERS
	global MAXTHREADS
	global LOGLEN
	#print 'MAXTHREADS', MAXTHREADS
	MAXTHREADS = int(nThreads)
	WORKERS = Q.Queue()#maxsize=MAXTHREADS)
	WORKERS2 = Q.Queue()#maxsize=MAXTHREADS)
	globIndex = 0
	LOGLEN = len(log)
	
	#Creates objects which contain a log and a vector
	class AvJobs(object):
		def __init__(self, _log, _averages):
			self.log = _log
			self.averages = _averages

		def printInfo(self):
			print self.log
			print self.matrix

	#Creates objects which contain an index and a vector
	class Assigners(object):
		def __init__(self, _index, _vector):
			self.index = _index
			self.vector = _vector

		def printInfo(self):
			print self.index
			print self.vector

	#Ensures that the amount of threads is not greater than the length of the log
	if MAXTHREADS > len(log):
		MAXTHREADS = len(log)

	#Function that launches a thread
	def launchAverageThread():
		thread = Thread(target=average)
		thread.setDaemon(True) #Terminates with master thread
		thread.start()

	def launchAssignThread():
		thread = Thread(target=assign)
		thread.setDaemon(True)
		thread.start()
	#Function that is called that performs the work
	def average():
		global DONE
		global LOCK
		global WORKERS
		global LOGLEN
		global AVERAGES

		#Finds the averages clusters from all averages given
		def findAverages(worker):
			THRESHOLD = math.pi / 2.2
			found = 0
			for j in range(len(AVERAGES)): #Cycles through worker 2 averages
				if np.arccos(min(cSim.dot(worker.averages[0], AVERAGES[j]), 1)) < THRESHOLD: #If similar
					AVERAGES[j] = cSim.averageVectors(AVERAGES[j], worker.averages[0])
					found = 1
			if found == 0:
				AVERAGES.append(worker.averages[0])

		worker = WORKERS.get()
		avgs = findAverages(worker)
		WORKERS.task_done()
		launchAverageThread()

	#Initializes the averages list
	AVERAGES[0] = matrix[0]
	#Creates multiple threads to handle the workers 

	#print 'Beginning Part 1!'
	#Creates the threads for the averaging
	for i in range(MAXTHREADS):
		try:
			launchAverageThread()
		except Exception as E:
			print E

	#Assigns the initial jobs to the workers
	for i in range(len(matrix)):
		#print i
		WORKERS.put(AvJobs([log[i]], [matrix[i]]))

	#Waits for part 1 to finish
	print 'Waiting for part 1 to finish'
	WORKERS.join()
	#print 'Finished Part 1!'

	#Part 2
	#print 'Beginning Part 2!'
	CLUSTERS = [[] for i in range(len(AVERAGES))]

	def assign():
		global LOGLEN
		global LOCK
		global CLUSTERS
		global AVERAGES
		global WORKERS2

		def assignLog(worker):
			clusterIndex = -1
			minDist = math.pi
			vector = worker.vector
			for j in range(len(AVERAGES)):
				distTemp = np.arccos(min(cSim.dot(AVERAGES[j], vector), 1))
				if(distTemp < minDist):
					minDist = distTemp
					clusterIndex = j
			CLUSTERS[clusterIndex].append(worker.index)
		worker = WORKERS2.get()
		assignLog(worker)
		WORKERS2.task_done()
		launchAssignThread()

	for i in range(MAXTHREADS):
		try:
			launchAssignThread()
		except Exception as E:
			print E

	for i in range(len(matrix)):
		WORKERS2.put(Assigners(i, matrix[i]))

	print 'Waiting for part 2 to finish'
	WORKERS2.join()
	#print 'Finished Part 2!'

	return AVERAGES, CLUSTERS

if __name__ == "__main__":
	#Sets the log and dictionary
	log = CSFH.loadLog(sys.argv[1])
	
	dictionary = cSim.generateDictionary(log)

	#Sets the matrix
	matrix = cSim.generateMatrix(dictionary, len(log), log)

	from sklearn import decomposition
	from sklearn import datasets
	pca = decomposition.PCA(n_components=5)
	pca.fit(matrix)
	matrix = pca.transform(matrix)
	

	matrix = cSim.normalize(matrix)

	avgs, clstrs = distributedACD(log, dictionary, matrix, sys.argv[2])
	cSim.graphTopics(log, clstrs)