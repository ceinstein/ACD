package main

import "fmt"
import "os"
import "strings"
import "bytes"
import "math"
import "strconv"
import "errors"
import "bufio"
import "time"
import "io/ioutil"

//Checks if an error has occured
//If it has, end the program
func errCheck(e error){
	if e != nil{ //If an error has been encountered
		panic(e)
	}
}


func main(){

	//log,_ := loadFile(os.Args[1])
	//strMat, data := loadFile(os.Args[1])
	//matrix := stoiAll(strMat, data)
	
	matrix, _:= loadMatrix(os.Args[1])
	fmt.Println("Starting ACD")
	t1 := time.Now() 
	threadNum, err := strconv.Atoi(os.Args[3])
	errCheck(err)
	clusters, _ := ACD(matrix, threadNum)
	t2 := time.Since(t1)
	fmt.Println(t2)
	//fmt.Println(clusters)
	//printMatrix(avgClusters)
	saveClusters(os.Args[2], clusters)

}

//Prints a matrix nicely
func printMatrix(matrix [][]float64){
	for i := 0; i < len(matrix); i++{
		for j := 0; j < len(matrix[i]); j++{
			fmt.Printf("%f ", matrix[i][j])
		}
		fmt.Println( "E!!")
	}
}

//Saves the clusters to a file
func saveClusters(filename string, clusters [][]int) {
	file, err := os.Create(filename)
	errCheck(err)
	//length, err := file.Write(clusters)
	errCheck(err)
	writer := bufio.NewWriter(file)
	for i := 0; i < len(clusters); i++ {
		for j := 0; j < len(clusters[i]); j	++ {
			_, err := writer.WriteString(strconv.Itoa(clusters[i][j]))
			errCheck(err)
			_, err = writer.WriteString(" ")

		}
		if len(clusters[i]) > 0{
			writer.WriteString("\n")
		}
	}
	writer.Flush()
}
//Converts a string slice to a float64 slice
func stof(line []string) []float64 {
	ret := make([]float64, len(line))
	var err error
	for i := 0; i < len(line); i++{
		ret[i], err = strconv.ParseFloat(line[i], 64)
		errCheck(err)
	}
	return ret
}

//Converts a 2d string slice to a 2d float64 slice
func stofAll(strings [][]string, data []string) [][]float64 {
	rows, err := strconv.Atoi(data[0])
	errCheck(err)

	ret := [][]float64{}
	for i := 0; i < rows; i++{
		ret = append(ret, stof(strings[i]))
	}

	return ret
}

//Takes the dot product of two vectors
func dot(vec1 []float64, vec2 []float64) float64 {

	if len(vec1) != len(vec2){
		panic(errors.New("Vectors do not have the same dimensions"))
	}
	dot := 0.0

	for i:= 0; i < len(vec1); i++{
		dot += (vec1[i] * vec2[i])
	}

	return dot
}

//Averages two vectors together
func averageVectors(vec1 []float64, vec2 []float64) []float64 {
	
	if len(vec1) != len(vec2){
		panic(errors.New("Vectors do not have the same dimensions"))
	}

	avgVec := make([]float64, len(vec1))
	vecLength := len(avgVec)
	for i := 0; i < vecLength; i++ {
		avgVec[i] = (vec1[i] + vec2[i]) / 2.0 
	}

	return avgVec

}

//Loads a file into a 2d slice. First line of file contains metadata
//Returns the 2d slice and the metadata
//SLOW OLD VERSION
func loadFile(path string) ([][]string, []string) {
	//Opens the file
	file, err := os.Open(path)
	errCheck(err)

	//Gathers stats on the file
	fileStats, err := file.Stat()
	errCheck(err)

	//Reads the file into a buffer
	data := make([]byte, fileStats.Size())
	dataLen, err := file.Read(data)
	errCheck(err)

	//Assigns the log messages to the 2d slice
	lines := [][]string{} //Holds the log messages
	var buffer bytes.Buffer //Reads in the log messages
	for i := 0; i < dataLen; i++ {
		fmt.Println(i, dataLen)
		if(data[i] == 10){ //Checks for a new line (indicating end of a log message)
			lines = append(lines, strings.Fields(buffer.String()))
			buffer.Reset()
		}else{ //Part of same message
			buffer.WriteString(string(data[i]))
		}
	}

	return lines[1:], lines[0]
} 

//Loads a matrix
//Returns the matrix, and metadata about the matrix
func loadMatrix(path string) ([][]float64, []float64) {
	data, err := ioutil.ReadFile(path)
	errCheck(err)

	matrix := [][]float64{}
	vectors := strings.Split(string(data), "\n")
	for i := 0; i < len(vectors); i++ { 
		if (len(vectors[i]) > 0){
			matrix = append(matrix, stof(strings.Fields(vectors[i])))
		}
		
	}
	return matrix[1:len(matrix)], matrix[0]
}

//Struct that holds the averaging jobs
type AverageJobs struct {
	Averages [][]float64 //List of the current averages
}

//Struct that holds the cluster jobs
type ClusterJobs struct {
	Index int //Index of the vector
	Vector []float64

}
//Finds the number of threads that are created for the averaging portion of ACD
func averageCycles(length int) int{
	if length == 3{
		return 1
	}
	if length < 3{
		return 0
	}
	newLen := ((length / 2) + (length % 2))
	return  length / 2 + averageCycles(newLen)
}

//Detects clusters of log messages and assigns each message to a cluster
//Returns an array with each index being a cluster and each index within the cluster being a log message
func ACD(logVecs [][]float64, threadNum int) ([][]int, [][]float64){
	t := time.Now()
	//Channels that allows program to wait for threads to finish
	average_complete := make(chan int) //Complete average jobs
	cluster_complete := make(chan int) //Complete cluster jobs


	workers := make(chan int, threadNum) //Handles number of concurrent threads

	average_jobs := make(chan AverageJobs) //Stores the averaging jobs
	cluster_jobs := make(chan ClusterJobs) //Stores the cluster jobs

	avgClusters := [][]float64{}
	logLength := len(logVecs)
	avgLength := 0
	THRESHOLD := math.Pi / 2.3 //How lenient the averaging is

	//Function that handles the averaging portion of the algorithm
	average := func(job1 AverageJobs, job2 AverageJobs){
		//Finds the averages
		for i:= 0; i < len(job1.Averages); i++ {
			found := 0
			for j := 0; j < len(job2.Averages); j++ { //Checks if average is close to any other average
				if math.Acos(math.Min(dot(job1.Averages[i], job2.Averages[j]), 1.0)) < THRESHOLD{
					job2.Averages[j] = averageVectors(job2.Averages[j], job1.Averages[i])
					found = 1
				}
			}
			if found == 0{ //Creates a new average
				job2.Averages = append(job2.Averages, job1.Averages[i])
				avgLength++
			}
		}
		
		avgClusters = job2.Averages //Create new master avgCluster
		
		average_jobs <- AverageJobs{job2.Averages} //Add the new averages to the jobs list
		<- workers //Free worker
		average_complete <- 3 //Indicate a job is done
	}

	//Recycles an average job
	recycleAverageJob := func(job AverageJobs){
		average_jobs <- job
	}

	//Buffer that stores two averaging jobs
	jobBuffer := []AverageJobs{}
	
	//Handles the job distribution
	go func(){
		count := 0
		for job := range average_jobs{
			
			if count < 2{ //If job buffer is not full
				jobBuffer = append(jobBuffer, job)
				count++
			}else{
				go recycleAverageJob(job)
				if len(workers) != cap(workers){ //If workers are available
					workers <- 1 //Take a worker
					go average(jobBuffer[0], jobBuffer[1]) //Average two jobs
					jobBuffer = []AverageJobs{} //Clear the job buffer
					count = 0	
				}
			}
		}
		
	}()

	//Assigns the log vectors to the averaging jobs
	go func(){
		for i := 0; i < len(logVecs); i++{		
			average_jobs <- AverageJobs{[][]float64{logVecs[i]}}
		}
	}()

	//Waits for the average jobs to finish
	for i := 0; i < averageCycles(logLength); i++{
		 <- average_complete
	}

	fmt.Println("Done with Part 1!", time.Since(t))
	t = time.Now()

	//Constructs the cluster 2D array
	clusters := [][]int{}
	for i := 0; i < avgLength; i++ {

		clusters = append(clusters, []int{})
	}

	//Function that handles the clustering portion of the algorithm
	cluster := func(job ClusterJobs){
		clusterIndex := -1
		minDist := math.Pi
		avgLength := len(avgClusters)
		for j := 0; j < avgLength; j++ {
			distTemp := math.Acos(math.Min(dot(avgClusters[j], job.Vector), 1.0))
			if(distTemp < minDist){ //If message is closer to a cluster
				minDist = distTemp
				clusterIndex = j
			}
		}
		//fmt.Println(clusterIndex)
		clusters[clusterIndex] = append(clusters[clusterIndex], job.Index)
		<- workers //Free a worker
		cluster_complete <- 3
	}

	//Recylces a job
	recycleClusterJob := func(job ClusterJobs){
		cluster_jobs <- job
	}

	//Handles the cluster job distribution
	go func(){ //Launches thread
		for job := range cluster_jobs{ //As jobs come in
			if len(workers) == cap(workers){ //If all workers are busy
				go recycleClusterJob(job)
			}else{ //If a worker is available
				workers <- 1
				go cluster(job) //Do job
			}
		}
	}()
	
	//Assigns the log messages to the cluster jobs
	go func(){
		for i := 0; i < logLength; i++ {
			cluster_jobs <- ClusterJobs{i, logVecs[i]}
		}


	}()

	//Waits for cluster jobs to finish
	for i := 0; i < logLength; i++ {
		<- cluster_complete
	}

	fmt.Println("Done with Part 2!", time.Since(t))

	return clusters, avgClusters
}

