#include "Matrix.h"


int main(int argc, char** argv){
  Matrix neo;
  printf("Starting!\n");
  neo.generate(argv[1]);
  neo.normalize();
  neo.similarities();
  //neo.dictionary.save("dude");
  printf("DONE\n");
  return 1;

}

//Empty Constructor
Matrix::Matrix(){
  this->rows = -1;
  this->columns = -1;
}

//Initialize matrix elements to 0
void Matrix::initialize(){
  for(int i = 0; i < this->getRows(); i++){
    for(int j = 0; j < this->getColumns(); j++){
      this->matrix[i][j] = 0;
    }
  }
}


//Generates the matrix from the Matrix dictionary and a given log 
void Matrix::generate(const char * filename){
  Dictionary dict1;
  dict1.generate(filename);
  //dict1.save("animals");
  //dict1.print();
  long numRows = dict1.getlogLength();
  if(numRows >= 0){ //If the log exists

    this->matrix = (double **) malloc(numRows * sizeof(double *));
    for(int i = 0; i < dict1.getLength(); i++){
        this->matrix[i] = (double *) malloc(dict1.getLength() * sizeof(double));
    }
    this->setRows(numRows);
    this->setColumns(dict1.getLength());
    this->initialize();
    
    std::ifstream log;
    log.open(filename);
    std::string entry;

    std::string words[numRows][120];
    long lineNum = 0;
    int i;
    //Reads through everyline of the log
    
    while(std::getline(log, entry)){
      i = 0;
      std::stringstream parse(entry);
      
      //Parses the words
      while(parse.good() && i < sizeof(entry)){
        parse >> words[lineNum][i];
        printf("%s\n", words[lineNum][i].c_str());
        printf("%s\n", dict1.dictionary[i]);
        i++;

      }
      
      lineNum++;
    }

    log.close();
    Dictionary dict;
    dict.generate(filename);



    //Loops through the words of a line of the log
    long line = 0;
    long place;
  //  dict.print();
    while(line < lineNum){
      for(i = 0; i < 120; i++){
        if(strcmp(words[line][i].c_str(), "") != 0 && strcmp(words[line][i].c_str(), " ") != 0){
         place = dict.contains((const char *)words[line][i].c_str());
         if(place != -1){
           this->matrix[line][place] = this->getElement(line, place) + (double)1;
          }             
        }
      }
      line++;
    }
  }
}

//Normalizes the rows of the matrix
void Matrix::normalize(){


//  this->print();
  long rowSum = 0;
  for(int i = 0; i < this->rows; i++){
    rowSum = 0;
    for(int j = 0; j < this->columns; j++){
      rowSum += (this->getElement(i , j) * this->getElement(i, j));
    }
    for(int j = 0; j < this->columns; j++){
      this->matrix[i][j] = this->getElement(i, j) / sqrt(rowSum);
    }
  }
  printf("Done with normalization\n");
}

//Takes the dot product of a normalized vector with a vector with 1s for all words
double Matrix::dot(long row1, long row2){
  double summation = 0;
  for(long i = 0; i < this->columns; i++){
    summation += (this->matrix[row1][i] * this->matrix[row2][i]); //(a1 * b1) + (a2 * b2) + ...
  }

 // printf("%f %f\n", summation, (double)acos(summation));
  return acos(summation);
}

//Finds the cosine similariies between each entry
void Matrix::similarities(){
  float dot;
  FILE *file;
  file = fopen("data.dat", "w");
  for(long i = 0; i < this->rows; i++){
    for(long j = 0; j < this->rows; j++){
      dot = this->dot(i,j);
      if(isnan(dot)){
        dot = (double)0;
      }
      //std::cout << i << "\t" << j << "\t" << dot << std::endl;
   //   datafile << i << "\t" << j << "\t" << dot;
      fprintf(file, "%ld\t%ld\t%f\n", i, j, dot);
      if(dot < (double)1.0){
        //printf("Row %ld and Row %ld\tSimilarity: %f\n",i, j, dot);
      }
    }
  }

  fclose(file);


}



//Sets the number of rows of the matrix
void Matrix::setRows(long numRows){
  this->rows = numRows;
}

//Gets the number of rows of the matrix
long Matrix::getRows(){
  return this->rows;
}

//Sets the number of columns of the matrix
void Matrix::setColumns(long numColumns){
  this->columns = numColumns;
}

//Gets the number of columns of the matrix
long Matrix::getColumns(){
  return this->columns;
}


//Gets the value of a certain element in the matrix
double Matrix::getElement(long row, long column){
  return this->matrix[row][column];
}

void Matrix::print(){
  if(rows != -1 && columns != -1){
    for(long i = 0; i < this->getRows(); i++){
      for(long j = 0; j < this->getColumns(); j++){
        printf("%f ", this->getElement(i, j));
      }
      printf("\n");
    }
  }else{
    printf("The matrix has not been initalized! Nothing to print...\n");
  }

}
