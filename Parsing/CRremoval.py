import sys

infile = open(sys.argv[1], 'r')
outfile = open(sys.argv[2], 'w')

parsedLine = ''
for line in infile:
    for i in range(len(line)):
        if line[i] == '\r':
            break
        parsedLine += line[i]

    parsedLine += '\n'
    outfile.write(parsedLine)
    parsedLine = ''

