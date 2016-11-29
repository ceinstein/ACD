/*
 *Author: Craig Einstein, einstein@bu.edu
 *Date Created: 06/21/16
 *
 *Description:
 *    Contains the prototypes and descriptions of the functions used in Dictionary.cpp
 *
 */
#ifndef DICTIONARY
#define DICTIONARY

#include<map>
#include<string.h>
#include<iostream>
#include<fstream>
#include<stdio.h>
#include<stdlib.h>
#include<sstream>
#include<math.h>

class Dictionary{

 private:
  //std::map<long, const char *> dictionary;
  std::map<long, const char *>::iterator iter;
  long length;
  long logLen;

  long ** matrix;
  long rows;
  long columns;

 public:

    std::map<long, const char *> dictionary;

  //Empty Constructor
  Dictionary();
  
  //Checks if a value is contained within the dictionary
  long contains(const char * word);
  
  //Adds a key/value pair to the dictionary
  void add(const char * value);

  //Sets the new length of the dictionary
  void setLength(long length);

  //Gets the length of the dictionary
  long getLength();

  //Saves a dictionary to a txt file (do not include .txt)
  int save(const char * filename);

  //Loads a dictionary into a Dictionary object
  void load(const char * filename);

  //Adds to dictionary from a log
  void generate(const char * logFile);

  //Calculates the length of a log
  long logLength(const char * filename);

  //Returns the value of logLength
  long getlogLength();

  //Appends one dictionary to another
  void append(Dictionary * dict);
  
  //Clears the dictionary
  void clear();

  //Prints the dictionary to the terminal
  void print();

  /*//Initialize matrix elements to 0
  void initialize();
  
  //Generates the matrix from the Matrix dictionary and a given log
  void generateMatrix(const char * filename);

  //Sets the number of rows of the matrix
  void setRows(long numRows);

  //Gets the number of rows of the matrix
  long getRows();

  //Sets the numer of columns of the matrix
  void setColumns(long numColumns);

  //Gets the number of columns of the matrix
  long getColumns();

  //Gets the value of a certain element in the matrix
  long getElement(long row, long column);

  //Prints the matrix
  void printMatrix();*/
};


#endif
