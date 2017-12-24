from lxml import etree


def parse(infilename, outfilename):
	outfile = open(outfilename,'w')

	tree = etree.parse(infilename)
	root = tree.getroot()
	sentence = root[0][0]
	for token in sentence:
		if 'ph' in token.keys():
			outfile.write(token.attrib['ph'] + '\n')

parse('example.xml','parsed_example.txt')
