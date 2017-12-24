from lxml import etree


def parse(infilename, outfilename):
	outfile = open(outfilename,'w')

	tree = etree.parse(infilename)
	root = tree.getroot()
	sentences = root[0]
	for sentence in sentences:
		for token in sentence:
			if 'ph' in token.keys():
				outfile.write(token.attrib['ph'] + '\n')
