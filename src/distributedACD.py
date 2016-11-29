from threading import Thread
import threading
import time
import subprocess
import sys
import cosSimFileHandler as CSFH
import generation as gen
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
	
	#Creates objects which contain a log and a matrix
	class AvJobs(object):
		def __init__(self, _log, _averages):
			self.log = _log
			self.averages = _averages

		def printInfo(self):
			print self.log
			print self.matrix

	class Assigners(object):
		def __init__(self, _index, _vector):
			self.index = _index
			self.vector = _vector

		def printInfo(self):
			print self.index
			print self.vector

	#Sets the log and dictionary
	#log = CSFH.loadLog(sys.argv[1])
	
	#dictionary = gen.generateDictionary(log)

	#Sets the matrix
	#matrix = gen.generateMatrix(dictionary, len(log), log)
	#matrix = gen.normalize(matrix)

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
		def findAverages(worker1, worker2):
			avgClusters = worker1.averages
			retlog = worker1.log
			retlog.append(worker2.log)
			logLength = len(worker1.log)
			avgLength = len(avgClusters)
			THRESHOLD = math.pi / 2.3
			for i in range(len(worker2.averages)): #Cycles through worker 1 averages
				found = 0
				for j in range(avgLength): #Cycles through worker 2 averages
					if np.arccos(min(gen.dot(worker2.averages[i], avgClusters[j]), 1)) < THRESHOLD: #If similar
						avgClusters[j] = gen.averageVectors(avgClusters[j], worker2.averages[i])
						found = 1
				if found == 0:
					avgClusters.append(worker2.averages[i])
					avgLength += 1
			#print len(avgClusters)
			return retlog, avgClusters #Returns the concatenation of the logs and the clusters


		LOCK.acquire()
	   	worker1 = WORKERS.get() #Waits to receive work
	   #	print len(worker1.log)
	   #	print worker1.log
	   	worker2 = 0
	   	compute = 0
	   	repeat = 0
	   	if(not WORKERS.empty()): #If there is more than one job remaining
	   		worker2 = WORKERS.get() #Get the second job
	   		compute = 1 #Indicate that computation needs to be performed
	   	else: #If there are no jobs currently in the queue
	   		if(len(worker1.log) == LOGLEN): #If all log messages have been processed
	   			#print '****THE FINISH!****'
	   			AVERAGES = worker1.averages
	   			WORKERS.task_done()
	   		else: #If jobs are running slowly
	   			#WORKERS.task_done()
	   			WORKERS.task_done()
	   			WORKERS.put(worker1)
	   			repeat = 1 #Loop and wait for new jobs


		LOCK.release() #Allow other threads to access the queue
	   	if(compute == 1):
	   		LOCK.acquire()
	   		#print len(worker1.log)
	   		LOCK.release()
	   		newLog, avgClusters = findAverages(worker1, worker2)
	   		
	   		AVERAGES = avgClusters
			WORKERS.task_done() #Finishes worker 1 task
			WORKERS.task_done() #Finishes worker 2 task
			#print 'finish two'
			WORKERS.put(AvJobs(newLog, avgClusters)) #Put the new job in the queue
			#print 'mateeee'
			
			launchAverageThread()
			#work() #Ready for work!
		if (repeat == 1):
			#work()
			launchAverageThread()
			pass


	#Creates multiple threads to handle the workers 
	for i in range(MAXTHREADS):
		try:
			launchAverageThread()
		except Exception as E:
			print E

	#Assigns the initial jobs to the workers
	for i in range(len(matrix)):
		WORKERS.put(AvJobs([log[i]], [matrix[i]]))

	#Waits for part 1 to finish

	WORKERS.join()

	#print AVERAGES
	#print WORKERS.qsize(), 'YO!!!!'

	#Part 2
	CLUSTERS = [[] for i in range(len(AVERAGES))]

	def assign():
		global LOGLEN
		global LOCK
		global CLUSTERS
		global AVERAGES
		global WORKERS2

		def assignLog(worker):
			clusterIndex = - 1
			minDist = math.pi
			vector = worker.vector
			for j in range(len(AVERAGES)):
				distTemp = np.arccos(min(gen.dot(AVERAGES[j], vector), 1))
				if(distTemp < minDist):
					minDist = distTemp
					clusterIndex = j
			#LOCK.acquire()
			#CLUSTERS[clusterIndex].append(worker.index)
			#LOCK.release()
			#print CLUSTERS
			#print len(CLUSTERS)
			#clusters = [x for x in clusters if x != []]
			#print worker.index
			return clusterIndex
		#LOCK.acquire()
		worker = WORKERS2.get()
		ind = assignLog(worker)
		CLUSTERS[ind].append(worker.index)
		#LOCK.release()
		WORKERS2.task_done()
		launchAssignThread()

	#print MAXTHREADS
	for i in range(MAXTHREADS):
		try:
			launchAssignThread()
		except Exception as E:
			print E

	for i in range(len(matrix)):
		WORKERS2.put(Assigners(i, matrix[i]))

	WORKERS2.join()

	#print CLUSTERS

	#gen.graphTopics(log, CLUSTERS)
	#print 'We finished?'
	#print AVERAGES

	#aCluster, avgClusters = ACD(log, matrix, AVERAGES)
	#topics = gen.graphTopics(log, aCluster)

	return AVERAGES, CLUSTERS

if __name__ == "__main__":
	#Sets the log and dictionary
	log = CSFH.loadLog(sys.argv[1])
	#DO NOT TRY TO SAVE THE LOG

	dictionary = gen.generateDictionary(log)

	#Sets the matrix
	matrix = gen.generateMatrix(dictionary, len(log), log)
	matrix = gen.normalize(matrix)
	CSFH.saveMatrix(matrix, sys.argv[2])

	#distributedACD(log, dictionary, matrix, sys.argv[4])