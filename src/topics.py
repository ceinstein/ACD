import generation as gen
import clusterFileHandler as CFH
import sys


if __name__ == '__main__':
	log = CFH.loadLog(sys.argv[1])
	clusters = CFH.loadClusters(sys.argv[2])
	topics = gen.graphTopics(log, clusters)
	CFH.saveTopics(sys.argv[3], topics)