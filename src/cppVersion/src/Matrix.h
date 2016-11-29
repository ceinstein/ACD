#ifndef MATRIX
#define MATRIX

#include "Dictionary.h"

class Matrix{

 private:
  double ** matrix;
  long rows;
  long columns;

 public:

  Dictionary * dictionary;
  
  //Empty Constructor
  Matrix();
  
  //Initialize matrix elements to 0
  void initialize();
  
  //Generates the matrix from the Matrix dictionary and a given log
  void generate(const char * filename);

  //Normalizes the rows of the matrix
  void normalize();

  //Takes the dot product of a vector with a vector with 1s for all words
  double dot(long row1, long row2);

  //Finds the cosine similarities between each entry
  void similarities();

  //Sets the number of rows of the matrix
  void setRows(long numRows);

  //Gets the number of rows of the matrix
  long getRows();

  //Sets the numer of columns of the matrix
  void setColumns(long numColumns);

  //Gets the number of columns of the matrix
  long getColumns();

  //Gets the value of a certain element in the matrix
  double getElement(long row, long column);

  //Prints the matrix
  void print();
};

#endif
