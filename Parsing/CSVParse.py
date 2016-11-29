import sys



#COMMAND LINE INPUTS:
	#INFLINE
	#OUTFILE
	#NUMBER OF ROWS (LEAVE BLANK FOR ALL)
if __name__ == '__main__':
	infile = open(sys.argv[1], 'r')
	outfile = open(sys.argv[2], 'w')
	length = 0
	try:
		length = int(sys.argv[3])
	except:
		length = len(infile.read().split('\n'))
		infile = open(sys.argv[1], 'r')
	print length
	count = 0

	for line in infile:

		splitline = line.split(',')

		i = 2
		while i < len(splitline):
			outfile.write(splitline[i] + ' ')
			i += 1
		outfile.write('\n')

		count += 1
		
		if count > length:
			sys.exit(1)
		