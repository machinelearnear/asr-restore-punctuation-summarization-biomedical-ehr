# Automatically detect medical entities from speech

Through this guide, you will learn how to do automatic speech recognition in your language, fix the grammar from that transcribed speech, restore its punctuation, detect biomedical or clinical entities from that text, get a summary of it, and finally how to put everything together.  

## Getting started with Automatic Speech Recognition
- [Intro to Automatic Speech Recognition on 🤗](https://huggingface.co/tasks/automatic-speech-recognition)
- [Robust Speech Challenge Results on 🤗](https://huggingface.co/spaces/speech-recognition-community-v2/FinalLeaderboard)
- [Mozilla Common Voice 9.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_9_0)
- [Thunder-speech, A Hackable speech recognition library](https://scart97.github.io/thunder-speech/Ultimate%20guide/)
- [SpeechBrain - PyTorch powered speech toolkit](https://speechbrain.github.io/)
- [Neural building blocks for speaker diarization: speech activity detection, speaker change detection, overlapped speech detection, speaker embedding](https://github.com/pyannote/pyannote-audio)
- [SPEECH RECOGNITION WITH WAV2VEC2](https://pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html)
- [How to add timestamps to ASR output](https://github.com/huggingface/transformers/issues/11307)

## Requirements
- [SageMaker Studio Lab](https://studiolab.sagemaker.aws/) account. See this [explainer video](https://www.youtube.com/watch?v=FUEIwAsrMP4) to learn more about this.
- Python=3.9
- PyTorch>=1.10
- Hugging Face Transformers
- Several audio processing libraries (see `environment.yml`)

## Step by step tutorial

### Clone repo and install dependencies

There are 3 main notebooks to follow, but you can start from `0_speech_recognition.ipynb` [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/machinelearnear/asr-restore-punctuation-summarization-biomedical-ehr/blob/main/0_speech_recognition.ipynb)

Click on `Copy to project` in the top right corner. This will open the Studio Lab web interface and ask you whether you want to clone the entire repo or just the Notebook. Clone the entire repo and click `Yes` when asked about building the `Conda` environment automatically. You will now be running on top of a `Python` environment with all libraries already installed.

### Transcribe speech from audio (YouTube)

Open `0_speech_recognition.ipynb` and run all steps. For more information, please refer back to [this other repo from machinelearnear](https://github.com/machinelearnear/long-audio-transcription-spanish). You will basically download the audio from a YouTube video by providing the VideoID and then generate a transcript that will be saved locally to `/transcripts`.

### Fix grammar and restore punctuation

Open `1_grammar_punctuation_correction.ipynb` and load your transcribed speech. What we want to do now is to first fix the grammar errors and then base out of that fix the punctuation. This order of doing things is random, try it on your own to see what brings better results.

I have tested a number of libraries to do spellchecking and ended up with `autocorrect` and `pyspellchecker`. Both of them allow for the addition of custom vocabularies to the spell checker (see [this for example](https://github.com/filyp/autocorrect/issues/17)) so here is where you could use your very own list of relevant words in your domain e.g. radiology, pathology, etc. The way that you would run it is as follows:

```python
from spellchecker import SpellChecker
spell_py = SpellChecker(language='es', distance=2)  # Spanish dictionary
processed_text = spell_py.correction(input_text)
```

```python
from autocorrect import Speller
spell_autocorrect = Speller(lang='es',only_replacements=True)
processed_text = spell_autocorrect(input_text)
```

Once we have our corrected text, we apply a model to restore punctuation. There are a number of them, and you can see many links at the bottom of the notebook, but I short-listed it to 2: [deepmultilingualpunctuation](https://github.com/oliverguhr/deepmultilingualpunctuation) and [Silero](https://github.com/snakers4/silero-models#text-enhancement). Both of them allow for the fine-tuning to a specific language. The first library is the one that performs the best even though it was not even trained in Spanish. I'm using a multi-lingual model.


```python
from deepmultilingualpunctuation import PunctuationModel
model = PunctuationModel(model='oliverguhr/fullstop-punctuation-multilingual-base')
result = model.restore_punctuation(output_text)
```

### Detect medical entities (NER) and run summarisation

To detect medical entities, we are going to be using [Stanza](https://stanfordnlp.github.io/stanza/), "a collection of accurate and efficient tools for the linguistic analysis of many human languages. Starting from raw text to syntactic analysis and entity recognition, Stanza brings state-of-the-art NLP models to languages of your choosing". There are medical NLP models available in Hugging Face through [the Spanish Government's National NLP Plan](https://huggingface.co/PlanTL-GOB-ES) but they are not yet fine-tuned to detect clinical entities such as `disease`, `treatment`, etc.

```python
import stanza
# download and initialize a mimic pipeline with an i2b2 NER model
stanza.download('en', package='mimic', processors={'ner': 'i2b2'})
nlp = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'})
# annotate clinical text
doc = nlp(input_text)
# print out all entities
for ent in doc.entities:
    print(f'{ent.text}\t{ent.type}')
```

## Keep reading
- [Pyctcdecode & Speech2text decoding](https://www.youtube.com/watch?v=mp7fHMTnK9A&t=5s)
- [XLS-R: Large-Scale Cross-lingual Speech Representation Learning on 128 Languages](https://www.youtube.com/watch?v=ic_J7ZCROBM)
- [Unlocking global speech with Mozilla Common Voice](https://www.youtube.com/watch?v=Vvn984QmAVg)
- [Reconocimiento automático de voz con Python y HuggingFace en segundos (+ Repo)](https://www.youtube.com/watch?v=wFjPxz22MEs)
- [“SomosNLP”, red internacional de estudiantes, profesionales e investigadores acelerando el avance del NLP en español](https://somosnlp.org/)
- [How to Write a Spelling Corrector](https://norvig.com/spell-correct.html )
- [Build Spell Checking Models For Any Language In Python](https://medium.com/mlearning-ai/build-spell-checking-models-for-any-language-in-python-aa4489df0a5f )
- [Grammatical Error Correction](http://nlpprogress.com/english/grammatical_error_correction.html )
- [FullStop: Multilingual Deep Models for Punctuation Prediction](http://ceur-ws.org/Vol-2957/sepp_paper4.pdf)
- [BioMedIA: Abstractive Question Answering for the BioMedical Domain in Spanish](https://huggingface.co/spaces/hackathon-pln-es/BioMedIA)
- [PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer](https://huggingface.co/PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer )
- [Host Hugging Face transformer models using Amazon SageMaker Serverless Inference](https://aws.amazon.com/de/blogs/machine-learning/host-hugging-face-transformer-models-using-amazon-sagemaker-serverless-inference/)

## Citations
```bibtex
Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton and Christopher D. Manning. 2020. Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In Association for Computational Linguistics (ACL) System Demonstrations. 2020. [pdf][bib]

Yuhao Zhang, Yuhui Zhang, Peng Qi, Christopher D. Manning, Curtis P. Langlotz. Biomedical and Clinical English Model Packages in the Stanza Python NLP Library, Journal of the American Medical Informatics Association. 2021.
```

## Disclaimer
- The content provided in this repository is for demonstration purposes and not meant for production. You should use your own discretion when using the content.
- The ideas and opinions outlined in these examples are my own and do not represent the opinions of AWS.