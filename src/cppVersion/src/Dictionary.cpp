/*
 *Author: Craig Einstein, einstein@bu.edu
 *Date Created: 06/21/16
 *
 *Description
 *    Defines the functions that the dictionary class uses. Function descriptions are contained in the
 *    'Dictionary.h' file.
 *
 */

#include "Dictionary.h"


//int main(int argc, char** argv){
  Dictionary dict;
  //dict.generateMatrix(argv[1]);
 // dict.save("kangaroo");
 // return 1;
//}

//Empty Constructor
Dictionary::Dictionary(){
  this->length = 0; 
}

//Checks if a value is contained within the dictionary 
long Dictionary::contains(const char * word){

  //Iterates through the map to check for the value

  for(this->iter = this->dictionary.begin(); this->iter != this->dictionary.end(); this->iter++){
    if(strcmp(this->iter->second, word) == 0){ //If value is in the map
      return this->iter->first;
    }
  }
  return -1;
}


//Adds a key/value pair to the dictionary
void Dictionary::add(const char * value){
  if(this->contains(value) == -1 && strcmp(value, "") != 0 && strcmp(value, "\n") != 0 && strcmp(value, " ") != 0 && strcmp(value, "\t") != 0){
    long length = this->getLength();
    this->dictionary[this->getLength()] = value;
    this->setLength(length + 1);
  }
}

//Sets the new length of the dictionary
void Dictionary::setLength(long _length){
  this->length = _length;
}

//Gets the length of the dictionary 
long Dictionary::getLength(){
  return this->length;
}

//Saves a dictionary to a txt file (do not include .txt)  
int Dictionary::save(const char *filename){

  char name[sizeof(filename) + 1000];
  strcpy(name, filename);
  strcat(name, ".txt");

  std::ofstream savefile;  
  std::ifstream check;
  check.open(name);
  if(name){ //Removes the file if it already exists
    check.close();
    remove(name);
  }

  savefile.open(name);
  savefile << this->getLength() << "\n"; //Adds the length to the top
  
  //Iterates throught the map and adds to the file

  for(this->iter = this->dictionary.begin(); this->iter != this->dictionary.end(); this->iter++){
    savefile << this->iter->first << "\t" << this->iter->second << "\n";
  }
  savefile.close();

  std::cout << "\"" <<name << "\" Successfully Saved!\n";
  return 1;

}


//Loads a dictionary into a Dictionary object
void Dictionary::load(const char *filename){
  printf("sdfasdfasafds\n");
  std::ifstream loadfile;
  loadfile.open(filename);
  long length;
  if(loadfile){ //Checks if the file exists
    //Gets the length of the dictionary
    std::string line;
    std::getline(loadfile, line);
    length = (long)atoi(line.c_str()); //Sets the length of the dictionary

    //Iterates through the file and loads the dictionary
    char * key;
    key = (char *) malloc(50 * sizeof(char));
    char** values;
    values = (char **) malloc(length * sizeof(char *));
    for(int i = 0; i < length; i++){
      values[i] = (char *) malloc(100 * sizeof(char));
    }
    char * lineParse;
    lineParse = (char *) malloc(100 * sizeof(char));
    int tab = 0;
    int inc = 0;
    int valInc = 0;
    //Reads the lines of the file line by line
    while(std::getline(loadfile, line)){      
      lineParse = (char *)line.c_str();
      tab = 0;
      inc = 0;
      key[0] = '\0';
      values[valInc][0] = '\0';
      //Parse each line
      for(int i = 0; i < 100; i++){
	if(lineParse[i] == '\t'){ //If character is a tab
	  tab = 1;
	  key[inc] = '\0';
	  inc = 0;
	}else if(lineParse[i] == '\0'){
	  break;
	}else{ //If the character is not a tab
	  if(tab == 0){ //Before the tab
	    key[inc++] = lineParse[i];
	  }else{ //After the tab
	    values[valInc][inc++] = lineParse[i];
	  }
	}
      }
      values[valInc][inc++] = '\0';
      this->add((char *)values[valInc++]);
    }

  }else{
    std::cout<< "The file does not exist!\n";
    return;
  }

  if(length != this->getLength()){
    printf("Something went wrong! Exitting the loading process...\n");
    Dictionary dict;
    this->dictionary.clear();
    return; 
  }
  
  std::cout << "\"" << filename << "\" Successfully Loaded!\n";
}

//Creates a dictionary from a log
void Dictionary::generate(const char * logFile){
  
  long logLenth = this->logLength(logFile);
  this->logLen = logLenth;
  std::ifstream log;
  //log.open(logFile);
  if(logLen >= 0){
    log.open(logFile);
    std::string entry; //Creates a string to store the input
    long lineNum = 0; //Tracks the line number
    int counter = 0;
    

    std::string words[120];
    for(long j = 0; j < 120; j++){
      words[j] = "";
    }
    //long i;
    //Reads through everyline of the log
    long count;
    while(std::getline(log, entry)){
      count = 0;
      std::stringstream parse(entry);
      //Parses the words
      while(parse.good() && count < 120){
	       parse >> words[count++];
      }
      //Loops through the words of a line of the log
      for(long i = 0; i < 120; i++){
        if(strcmp(words[i].c_str(), "") != 0 && strcmp(words[i].c_str(), " ") != 0){
          this->add((char *)words[i].c_str()); //Add the word to the dictionary
          printf("%s\n", this->dictionary[i]);                
        }
      }
      
    }
    log.close();
  }else{
    log.close();
    std::cout << "That log file does not exist! Exitting...\n";
    return;
  }

  printf("Generation Complete\n");

}

//Calculates the length of a log (used in the creation of the dictionary)
long Dictionary::logLength(const char * filename){

  long counter = 0;
  std::ifstream log;
  log.open(filename);
  if(log){
    std::string entry;

    while(std::getline(log, entry)){
      counter++;

    }
    

  }else{
    log.close();
    return -1;
  }
  log.close();
  printf("Log Length complete\n");
  return 3;
  
}

long Dictionary::getlogLength(){
  return this->logLen;
}

//Appends one dictionary to another
void Dictionary::append(Dictionary * dict){
  
  for(dict->iter = dict->dictionary.begin(); dict->iter != dict->dictionary.end(); dict->iter++){
    this->add((char *)dict->iter->second);
  }
}


//Clears the dictionary
void Dictionary::clear(){
  this->dictionary.clear();
}

  
//Prints the dictionary to the terminal 
void Dictionary::print(){

  std::cout << "Length: " << this->getLength() << "\n";
  for(this->iter=this->dictionary.begin(); this->iter != this->dictionary.end(); iter++){

    std::cout << "Key: " << iter->first << "\tValue: " << iter->second << "\n";

  }
}