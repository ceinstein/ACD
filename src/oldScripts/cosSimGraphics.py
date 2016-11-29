from mayavi import mlab
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from tkFileDialog import *
from Tkinter import *
import sys
import networkx as netx
from networkx.drawing.nx_agraph import graphviz_layout

#Handles the GUI
#Has nested functions which handle error detection and call functions
def displaySet():
	import generation as gen
	import cosSimFileHandler as CSFH


	#Loads a log
	def getLog():
		global LOG
		try:
			filename = askopenfilename()
			LOG = CSFH.loadLog(filename)
			log.set(filename)
			event.set('Log Loaded!')
		except:
			#print E
			event.set('Log file could not be loaded!')

	#Loads a dictionary
	def getDict():
		global DICT
		try:
			dic.set(askopenfilename())
			DICT = CSFH.loadDictionary(dic.get())
			event.set('Dictionary loaded!')
		except E:
			print E
			dic.set('No Dictionary')
			event.set('Dictionary file could not be loaded!')

	#Saves a dictionary
	def saveDict():
		global DICT
		try:
			if len(DICT) < 2:
				raise IOError
			filename = asksaveasfilename(defaultextension='.dict')
			CSFH.saveDictionary(DICT, filename)
			dic.set(filename)
			event.set('Dictionary Saved!')
		except:
			#print E
			event.set('Dictionary file could not be saved!')

	#Loads a matrix
	def getMatrix():
		global MATRIX
		try:
			filename = askopenfilename()
			if filename.endswith('.matrix'):
				MATRIX = CSFH.loadMatrix(filename)
				matrix.set(filename)
				event.set('Matrix loaded!')
			else:
				raise IOError
		except:
			#print E
			event.set('Matrix file could not be loaded!')

	#Saves a matrix
	def saveMat():
		global MATRIX
		try:
			if len(MATRIX) < 2:
				raise IOError
			filename = asksaveasfilename(defaultextension='.matrix')
			CSFH.saveMatrix(MATRIX, filename)
			matrix.set(filename)
			event.set('Matrix Saved!')
		except:
			#print E
			event.set('Matrix file could not be saved!')

	#Loads a similarity matrix
	def getSimMatrix():
		global SIM
		try:
			filename = askopenfilename()
			if filename.endswith('.smatrix'):
				SIM = CSFH.loadMatrix(filename)
				simMatrix.set(filename)
				event.set('Similarity Matrix loaded')
			else:
				raise IOError
		except:
			#print E		
			event.set('Simialrity Matrix could not be loaded!')

	#Saves a similarity matrix
	def saveSimMatrix():
		global SIM
		try:
			if len(SIM) < 2:
				raise IOError
			filename = asksaveasfilename(defaultextension='.smatrix')
			CSFH.saveMatrix(SIM, filename)
			simMatrix.set(filename)
			event.set('Similarity Matrix Saved!')
		except E:
			print E
			event.set('Similarity Matrix file could not be saved!')

	#Loads the KMeans labels
	def getLabels():
		global LABELS
		try:
			filename = askopenfilename()
			if filename.endswith('.lbls'):
				LABELS = CSFH.loadLabels(filename)
				labelHolder.set(filename)
				event.set('Labels loaded!')

			else:
				raise IOError
		except:
			#print E
			event.set('Labels could not be loaded!')

	#Saves the KMeans labels
	def saveLabs():
		global LABELS
		try:
			if len(LABELS) < 2:
				raise IOError
			filename = asksaveasfilename(defaultextension='.lbls')
			CSFH.saveLabels(LABELS, filename)
			labelHolder.set(filename)
			event.set('Labels Saved!')
		except:
			#print E
			event.set('Label file could not be saved!')


	#Generates the dictionary, normalized matrix, and similarity matrix from a log file
	def generateFromLog():
		global DICT
		global LOG
		global MATRIX
		global SIM
		try:
			if len(LOG) < 2:
				raise IOError

			DICT = gen.generateDictionary(LOG)
			MATRIX = gen.generateMatrix(DICT, len(LOG), LOG)
			MATRIX = gen.normalize(MATRIX)
			#Temporarily Turned off for speed!!
			SIM = gen.generateSimilarties(MATRIX) 
			event.set('Dictionary, Matrix, and Similarity Matrix Generated!')
			matrix.set('New Matrix')
			dic.set('New Dictionary')
			simMatrix.set('New Similarity Matrix')
		except:
			event.set('Generation Incomplete... Aborting!')
			#print E
			

	#Generates a similarity matrix
	def genSimMat():
		global DICT
		global LOG
		global MATRIX
		global SIM
		try:
			if(len(MATRIX) < 2):
				raise IOError
			gen.generateSimilarties(MATRIX)
			event.set('New Similarity Matrix created')
			simMatrix.set('New Similarity Matrix')
		except:
			#print E
			event.set('Similiarity Matrix could not be generated... Aborting!')


	#Loads KMeans labels
	def genLabels():
		global MATRIX
		global DICT
		global LABELS
		try:
			if len(MATRIX) < 2:
				raise IOError
			LABELS = gen.KMeansCalc(MATRIX)
			labelHolder.set('New KMeans Labels')
			event.set('KMeans labels acquired')

		except:
			#print E
			event.set('KMeans could not be calculated')


	#Plots a scatter plot of the similarity matrix
	def plot3DButton():
		global SIM
		global LOG
		try:
			plot3D(SIM, LOG)
		except E:
			print E
			event.set('There is no similarity matrix to plot!')

	#Plots a scatter plot of the log IDs vs a vector of all ones
	def plot2DButton():
		global DICT
		global MATRIX
		try:
			if len(DICT) < 2 or len(MATRIX) < 2:
				raise IOError

			plot2D(DICT, MATRIX)
		except:
			#print E
			event.set('The dictionary and matrix are not set!')

	#Plot KMeans
	def plotKMeanButton():
		global LOG
		global LABELS
		try:
			if(len(LABELS) > 1):
				plotKMeans2d(LOG, LABELS)
			else:
				raise IOError
		except:
			#print E
			event.set('Could not plot KMeans')

	#Resets the global variables
	def reset():
		global DICT
		global MATRIX
		global LOG
		global SIM
		DICT = []
		MATRIX = [[]]
		LOG = []
		SIM = []
		log.set('Empty Log')
		dic.set('Empty Dictionary')
		matrix.set('Empty Matrix')
		simMatrix.set('Empty Similarity Matrix')
		labelHolder.set('Empty Labels')
		event.set('Data has been reset!')

	#Quits the display and terminates the program
	def quit():
		display.quit()
		sys.exit(1)


	display = Tk()
	display.title('Matrix Generator')
	display.minsize(width = 1000, height = 400)
	#display.maxsize(width = 500, height = 800)

	title = StringVar()
	title.set('**Cluster Detection**')
	title_label = Label(display, textvariable = title)


	#EVENT HANDLING
	event = StringVar()
	event.set('Welcome')
	event_label= Label(display, textvariable = event)
	

	#LOG
	log = StringVar()
	log.set('Empty Log')
	logButton = Button(text = 'Load Log', command = getLog)
	log_label = Label(display, textvariable = log)
	

	#Dictionary
	dic = StringVar()
	dic.set('Empty Dictionary')
	dictButton = Button(text = 'Load Dictionary', command = getDict)
	dic_label = Label(display, textvariable = dic)
	

	#MATRIX
	matrix = StringVar()
	matrix.set('Empty Matrix')
	matrixButton = Button(text = 'Load Matrix', command = getMatrix)
	matrix_label = Label(display, textvariable = matrix)

	#SIMILARITY MATRIX
	simMatrix = StringVar()
	simMatrix.set('Empty Similarity Matrix')
	simMatrixButton = Button(text = 'Load Similarity Matrix', command = getSimMatrix)
	sim_matrix_label = Label(display, textvariable = simMatrix)
	

	#LABELS
	labelHolder = StringVar()
	labelHolder.set('Empty Labels')
	labelButton = Button(text = 'Load Labels', command = getLabels)
	labelHolder_label = Label(display, textvariable = labelHolder)
	

	#Generates from a log file
	generateFL = Button(text = 'Generate Dictionary and Matricies from Log', command = generateFromLog)
	

	#Generates from a log file
	generateSimMat = Button(text = 'Generate Similarity Matrix', command = genSimMat)

	KMeansButton = Button(text = 'Generate KMeans Labels', command = genLabels)
	

	plot3 = Button(text = 'Plot 3D', command = plot3DButton)
	
	plot2 = Button(text = 'Plot 2D', command = plot2DButton)
	
	plotKMeans = Button(text = 'Plot KMean Clusters', command = plotKMeanButton)

	saveDButton = Button(text = 'Save Dictionary', command = saveDict)
	
	saveMButton = Button(text = 'Save Matrix', command = saveMat)
	
	saveSMButton = Button(text = 'Save Similarity Matrix', command = saveSimMatrix)
	
	saveLabelsButton = Button(text = 'Save Labels', command = saveLabs)
	
	resetButton = Button(text = 'Reset', command = reset)
	
	quitButton = Button(text = 'Quit', command = quit)
	

	

	title_label.grid(row = 0, column = 1)
	event_label.grid(row = 1, column = 1)

	logButton.grid(row = 2, column = 0)
	log_label.grid(row = 2, column = 1)

	dictButton.grid(row = 3, column = 0)
	dic_label.grid(row = 3, column = 1)
	saveDButton.grid(row = 3, column = 2)

	matrixButton.grid(row = 4, column = 0)
	matrix_label.grid(row = 4, column = 1)
	saveMButton.grid(row = 4, column = 2)

	simMatrixButton.grid(row = 5, column = 0)
	sim_matrix_label.grid(row = 5, column = 1)
	saveSMButton.grid(row = 5, column = 2)

	labelButton.grid(row = 6, column = 0)
	labelHolder_label.grid(row = 6, column = 1)
	saveLabelsButton.grid(row = 6, column = 2)

	generateFL.grid(row = 8, column = 0)
	generateSimMat.grid(row = 8, column = 1)
	KMeansButton.grid(row = 8, column = 2)
	
	plot2.grid(row = 10, column = 0)
	plot3.grid(row = 10, column = 1)
	plotKMeans.grid(row = 10, column = 2)

	resetButton.grid(row = 12, column = 1)
	quitButton.grid(row = 13, column = 1)



	display.mainloop()

#Plots a scatterplot of each entry and their similarities
def plot3D(log, matrix):
	print matrix
	print log
	x = []
	y = []
	z = []
	for i in range(len(matrix) - 1):
		for j in range(i + 1, len(matrix[i])):
			if(matrix[i][j] < float(1.0) and matrix[i][j] > float(0.0)):
				x.append(i)
				y.append(j)
				z.append(matrix[i][j])

	plot = plt.figure()
	ax = plot.add_subplot(111, projection='3d')

	ax.set_xlim(xmin=0, xmax=max(x))
	ax.set_ylim(ymin=0, ymax=max(y))
	ax.set_zlim(zmin=0, zmax=max(z))

	ax.scatter(x, y, z, picker = 10)
	def select(event):
		ind = event.ind
		print '\n*****\nShowing', len(ind), 'data points\n*****\n'
		for i in range(len(ind)):
			print '*****Difference = ', z[ind[i]]
			print '*****LINE 1:', log[x[ind[i]]], '\n\n*****LINE 2:', log[y[ind[i]]], '\n'

	plot.canvas.mpl_connect('pick_event', select)
	ax.set_xlabel('Log ID')
	ax.set_ylabel('Log ID')
	ax.set_zlabel('Difference')
	plt.show()

#Plots the distance of all the rows of a matrix against a vector of all ones
def plot2D(log, matrix):
	import cosSim as gen
	nullVec = np.ones(len(log))
	nullVec = gen.normalizeRow(nullVec)
	x = []
	y = []
	for i in range(len(matrix)):
		x.append(i)
		y.append(gen.dot(nullVec, matrix[i]))

	plot = plt.figure()
	ax = plt.subplot()
	ax.scatter(x, y, picker = 10)
	def select(event):
		ind = event.ind
		print '\n*****\nShowing', len(ind), 'data points\n*****\n'
		for i in range(len(ind)):
			print '*****Difference = ', y[ind[i]]
			print '*****LINE:', log[x[ind[i]]], '\n'

	plot.canvas.mpl_connect('pick_event', select)
	ax.set_xlabel('Log ID')
	ax.set_ylabel('Difference')
	plt.show()

#Plots the which log IDs are in which cluster
def plotKMeans2d(log, clusters):
	#print log
	#print clusters
	x = []
	y = range(len(log))
	for cluster in clusters:
		x.append(cluster)


	plot = plt.figure()
	ax = plt.subplot()
	ax.scatter(x, y, picker = 10)

	def select(event):
		ind = event.ind
		print '\n*****\nShowing', len(ind), 'data points\n*****\n'
		for i in range(len(ind)):
			print '*****Cluster = ', x[ind[i]]
			print '*****LINE:', log[y[ind[i]]], '\n'

	plot.canvas.mpl_connect('pick_event', select)
	ax.set_xlabel('Cluster')
	ax.set_ylabel('Log ID')
	plt.show()

#Plots the distribution of the KMeans clusters
def plotGaussianCheck(labels):
	y = []
	#Counts number of logs in certain cluster
	found = 0
	looking = 0
	while found == 0:
		count = 0
		for num in labels:
			if num == looking:
				count += 1
		if count == 0:
			found = 1
		else:
			y.append(count)
			looking += 1
	x = range(looking)
	plot = plt.figure()
	ax = plt.subplot()
	ax.scatter(x,y)
	plt.plot(x, y, alpha=0.5, color='purple')
	ax.set_xlabel('CLuster ID')
	ax.set_ylabel('Number of Logs')
	plt.show()

#Creates a 3D visualization of a graph
def networkGraphics(log, matrix):
	
	#plot = plt.figure()
	#ax = plt.subplot()
	#ax.set_title('YO!')

	
	def select(event):
		print event
		ind = event.ind
		print ind
		print 'dude'
	#plot.canvas.mpl_connect('pick_event', select)

	G= netx.from_numpy_matrix(np.array(matrix))
	#netx.draw(G)
	#print list(netx.bfs_edges(G,0))
	netx.draw_networkx(G, pos=graphviz_layout(G))
	scalars=np.array(G.nodes())+5

	#plt.show()
	graph_pos = netx.spring_layout(G, dim=3)
	#xyz=np.array([graph_pos[v] for v in sorted(G)])
	xyz=np.array([graph_pos[v] for v in G])
	#print netx.is_connected(G)
	nodes =  netx.connected_components(G)

	clusters = []
	for node in nodes:
		#print list(node)
		clusters.append(list(node))
	#return clusters

	mlab.figure(1, bgcolor=(0,0,0))
	figure = mlab.gcf()
	mlab.clf()
	pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
        scalars,
        scale_factor=0.03,
        scale_mode='none',
        colormap='winter',
        resolution=20)

	outline = mlab.outline(line_width=3)
	outline.outline_mode = 'cornered'
	outline.bounds = (xyz[:,0][0]-0.02, xyz[:,0][0]+0.02,
		xyz[:,1][0]-0.02, xyz[:,1][0]+0.02,
		xyz[:,2][0]-0.02, xyz[:,2][0]+0.02)

	glyph_points = pts.glyph.glyph_source.glyph_source.output.points.to_array()



	#Allows nodes to be selected
	def picker_callback(picker):
		if picker.actor in pts.actor.actors:
			# Find which data point corresponds to the point picked:
			# we have to account for the fact that each data point is
			# represented by a glyph with several points
			point_id = picker.point_id/glyph_points.shape[0]
			# If the no points have been selected, we have '-1'
			if point_id != -1:
				# Retrieve the coordinnates coorresponding to that data
				# point
				x, y, z = xyz[:,0][point_id], xyz[:,1][point_id], xyz[:,2][point_id]
				# Move the outline to the data point.
				outline.bounds = (x-0.02, x+0.02,
									y-0.02, y+0.02,
									z-0.02, z+0.02)
				print log[point_id], '\n'

	picker = figure.on_mouse_pick(picker_callback)

	# Decrease the tolerance, so that we can more easily select a precise
	# point.
	picker.tolerance = 10

	pts.mlab_source.dataset.lines = np.array(G.edges())
	tube = mlab.pipeline.tube(pts, tube_radius=0.002)
	mlab.pipeline.surface(tube, color=(1,1,1))
	mlab.show()
	
#Creates an animation of the ACD algorithm
def animateClusters(log, logVecs, dictionary, matrix):
	import cosSim as gen
	import math
	import time
	import matplotlib.animation as animation
	from sklearn import decomposition
	from sklearn import datasets
	pca = decomposition.PCA(n_components=3)
	pca.fit(logVecs)
	sClust = pca.transform(logVecs)
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.set_xlim(0.06)
	ax.set_ylim(0.06)
	ax.set_zlim(0.06)


	avgClusters = []
	sClust = [[]]
	cClust = pca.transform(logVecs)
	
	#clust = ax.scatter(x, y, z)
	i = 0
	#ax.scatter(X[:,0], X[:,1])#, X[:,2])
	sClust = pca.transform(logVecs[0])
	#clusters = ax.plot(avgClusters)


	def animate(i):
		ax.set_xlim(0.06)
		ax.set_ylim(0.06)
		ax.set_zlim(0.06)
		#dude = cClust
		#avgClusters = []
		logLength = len(logVecs)
		avgLength = len(avgClusters)
		if(i >= logLength):
			newLog = gen.append(log, dictionary, matrix, raw_input('put in a thing!!!!'))
			
		if i < logLength* 2: #Cycles through the messages in the lgo
			found = 0
			for j in range(avgLength): #Cycles through the averages
				if i < logLength:
					if np.arccos(min(gen.dot(logVecs[i], avgClusters[j]), 1)) < math.pi / 2.3: #If similar
						avgClusters[j] = gen.averageVectors(avgClusters[j], logVecs[i])
						found = 1
				else:
					if np.arccos(min(gen.dot(newLog, avgClusters[j]), 1)) < math.pi / 2.3: #If similar
						avgClusters[j] = gen.averageVectors(avgClusters[j], newLog)
						found = 1	
			if found == 0:
				if i < logLength:
					avgClusters.append(logVecs[i])
					avgLength += 1
				else:
					avgClusters.append(newLog)
					avgLength += 1
			time.sleep(0.2)
			ax.cla()
			sClust = pca.transform(avgClusters)
			ax.scatter(sClust[:,0], sClust[:,1], sClust[:,2], s=60000, color='purple')
		else:
			animate2(logLength, avgLength, avgClusters, i, cClust)
		#cClust = _cClust
		#cClust = dude
		#print len(avgClusters)
			
	#			clusters.append([i])
	def animate2(logLength, avgLength, avgClusters, i, cClust):

		if i < logLength * 2:
			ax.set_autoscale_on(False)
			ax.cla()
			time.sleep(0.5)
			num = i - logLength
			sClust = pca.transform(avgClusters)
			ax.scatter(sClust[:,0], sClust[:,1], sClust[:,2], s=60000, color='purple')
			ax.scatter(cClust[:num+1][:,0], cClust[:num+1][:,1], cClust[:num+1][:,2], s = 300, color='red', marker='.', picker = 10)
			i += 1

	def select(event):
		ind = event.ind
		for index in ind:
			print log[index]

	fig.canvas.mpl_connect('pick_event', select)
	ani = animation.FuncAnimation(fig, animate, frames = (len(logVecs) * 2), repeat=False)

	plt.show()


	ani.save('trial.mp4')

#Global Variables to be used with the graphics
DICT = []
LOG = []
MATRIX = []
SIM = []
LABELS = []
