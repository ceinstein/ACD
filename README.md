# ACD - Automatic Cluster Detection
## Installations

To run the ACD algorith python 2.7 and Go must be installed

_A full list of Python packages used will also be included_

## Running

To run the pipeline the `cluster.py` script is run. The following flags can be supplied:


	-n is used to specify the name of the log file (name is given without the .log) 

	-t specifies the number of threads to use in the distribution

	-s saves the topics generated (topics are saved in a full runthrough of the pipeline)

	-g runs the go algorithm on certain inputs

	-a full runthrough of pipeline on the specified log file

	-l Generates the dictionary and matrix for a certain log file


Below is an example run of a full pipeline runthrough for the distinctAnimal log. This is being run from the main ACD directory

`
	python ./src/cluster.py -n distinctAnimal -a
`

## Output

A full runthrough of the pipeline will produce:

* A matrix file
* A dictionary file
* A cluster file (shows the row numbers of the logs that are grouped together)
	* Each cluster is seperated by a new line
* A topic file (shows the topic generated for each cluster)

---
###### Craig Einstein