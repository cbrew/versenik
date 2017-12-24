import requests
import urllib
from bs4 import BeautifulSoup
import sys
import gensim.downloader as api
import plac


def to_phon(text):
    tx = urllib.parse.quote_plus(text)
    r = requests.get(('http://localhost:59125/process?INPUT_TEXT={}' +
                      '&INPUT_TYPE=TEXT' +
                      '&OUTPUT_TYPE=PHONEMES&LOCALE=en_US').format(tx))
    soup = BeautifulSoup(r.text, 'lxml')
    for sent in soup('s'):
        for token in sent('t'):
            try:
                yield dict(text=token.text.strip(),
                            pos=token['pos'],
                            phonetics=token['ph'])
            except:
                pass


def text8_to_phon():
    show_every = 5000
    dataset = api.load('text8')
    total = 0
    for chunk in dataset:
        for i in range(0, len(chunk), 100):
            text = " ".join(chunk[i:i+100])
            if text.strip():
                for token in to_phon(text):
                    print(token['text'],
                          token['pos'],
                          token['phonetics'])
            total += 100
            if total % show_every == 0:
                print(total, text, file=sys.stderr)


if __name__ == "__main__":
    plac.call(text8_to_phon)
