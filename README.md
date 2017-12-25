# versenik
Tunable word similarity balancing sound and meaning.

## Tools used
MaryTTS [https://github.com/marytts/marytts] to extract a phonemic representation from arbitrary text.
Gensim [https://radimrehurek.com/gensim/] to train a phonemic embedding.
## Installation

1. Clone the marytts repository [https://github.com/marytts/marytts] and follow the startup instructions. This will start a 
a TTS service on localhost:59125 which you can play with using a browser or access from other tools.
2. Get a recent version of Python 3. We suggest Anaconda [https://www.anaconda.com/]. 
3. Clone this repository and change directory to versenik.
4. Install the requirements with pip -r requirements.txt
