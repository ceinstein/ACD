import sys
import time

infile = open(sys.argv[1], 'r')
outfile = open(sys.argv[2], 'w')
count = 0
formattedLine = ''
for line in infile:
    splitLine = line.split('\n') #Splits each line of the log
    for segments in splitLine:
        words = segments.split() #Splits the messages in the lines of the log
        #Counts the spaces between the lines in the log
        if len(words) == 0: #If a line is empty
            count += 1
        else:
            count = 0
            if((not(words[0][-1] == ':' and (words[0][:-1] == 'Author' or words[0][:-1] == 'Date' or words[0][:-1] == 'Reviewed-by'))) and words[0] != 'commit'): #Exclude metadata
                for word in words: #Splits the messages of each line of the log
                    if (word[-1] == '.' or word[-1] == ','): #Removes punctuation
                        formattedLine += word[:-1] + ' '
                    else:
                        formattedLine += word + ' '
        #If the commit message is over
        if count == 3:
            formattedLine = formattedLine[:-1] + '\n'
            if len(formattedLine) > 1:
                outfile.write(formattedLine)
            formattedLine = ''

