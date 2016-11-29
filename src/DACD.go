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
	//printMatrix(matrix)
	fmt.Println("Starting ACD")
	//fmt.Println(matrix)
	//fmt.Println(log)
	//fmt.Println(math.Acos(math.Min(dot(matrix[0], matrix[0]), 1.0)))
	//fmt.Printf(os.Args[1])
	//fmt.Println(matrix[0], matrix[1])
	//fmt.Println(averageVectors(matrix[0], matrix[18]))
	t1 := time.Now() 
	clusters, _ := ACD(matrix)
	t2 := time.Since(t1)
	fmt.Println(t2)
	//fmt.Println(clusters)
	//printMatrix(avgClusters)
	saveClusters(os.Args[2], clusters)


	//fmt.Println(lines[0][1])
}

func printMatrix(matrix [][]float64){

	for i := 0; i < len(matrix); i++{
		for j := 0; j < len(matrix[i]); j++{
			fmt.Printf("%f ", matrix[i][j])
		}
		fmt.Println( "E!!")
	}
}

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
func stoi(line []string) []float64 {
	ret := make([]float64, len(line))
	var err error
	for i := 0; i < len(line); i++{
		ret[i], err = strconv.ParseFloat(line[i], 64)
		errCheck(err)
	}
	return ret
}

//Converts a 2d string slice to a 2d float64 slice
func stoiAll(strings [][]string, data []string) [][]float64 {
	rows, err := strconv.Atoi(data[0])
	errCheck(err)

	ret := [][]float64{}
	for i := 0; i < rows; i++{
		ret = append(ret, stoi(strings[i]))
	}

	return ret
}

//Takes the dot product of two vectors
func dot(vec1 []float64, vec2 []float64) float64 {

	if len(vec1) != len(vec2){
		fmt.Println(vec1)
		fmt.Println(vec2)
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

func loadMatrix(path string) ([][]float64, []float64) {
	data, err := ioutil.ReadFile(path)
	errCheck(err)

	matrix := [][]float64{}
	vectors := strings.Split(string(data), "\n")
	for i := 0; i < len(vectors); i++ {
		if (len(vectors[i]) > 0){
			matrix = append(matrix, stoi(strings.Fields(vectors[i])))
		}
		
	}

	return matrix[1:len(matrix)-1], matrix[0]



}

//Detects clusters of log messages and assigns each message to a cluster
//Returns an array with each index being a cluster and each index within the cluster being a log message
func ACD(logVecs [][]float64) ([][]int, [][]float64){
	t1 := time.Now()
	avgClusters := [][]float64{}
	logLength := len(logVecs)
	avgLength := 0
	THRESHOLD := math.Pi / 2.3

	//Finds the averages
	for i:= 0; i < logLength; i++ {
		//fmt.Println(i)
		found := 0
		for j := 0; j < avgLength; j++ {
			
			if math.Acos(math.Min(dot(logVecs[i], avgClusters[j]), 1.0)) < THRESHOLD{
				avgClusters[j] = averageVectors(avgClusters[j], logVecs[i])
				found = 1
			}
		}
		if found == 0{
			avgClusters = append(avgClusters, logVecs[i])
			avgLength++
		}
	}
	fmt.Println("Done with Part 1!", time.Since(t1))
	t1 = time.Now()
	//Assigns the log messages to the clusters
	clusters := [][]int{}
	for i := 0; i < avgLength; i++ {
		clusters = append(clusters, []int{})
	}
	for i := 0; i < logLength; i++ {
		clusterIndex := -1
		minDist := math.Pi
		vector := logVecs[i]
		for j := 0; j < avgLength; j++ {
			distTemp := math.Acos(math.Min(dot(avgClusters[j], vector), 1.0))
			if(distTemp < minDist){
				minDist = distTemp
				clusterIndex = j
			}
		}
		clusters[clusterIndex] = append(clusters[clusterIndex], i)
	}
	fmt.Println("Done with Part 2!", time.Since(t1))

	return clusters, avgClusters
}