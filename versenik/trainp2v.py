import gensim
import logging
import io
import datetime
import plac


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class Phonseqs(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with io.open(self.filename,
                     encoding='utf8') as inf:
            for line in inf:
                phonetics = line.split()[2:]
                yield phonetics


def main(filename):
    model = gensim.models.Word2Vec(Phonseqs(filename), min_count=1)
    model.save('models/phon{}'.format(datetime.datetime.utcnow().isoformat()))


if __name__ == "__main__":
    plac.call(main)
