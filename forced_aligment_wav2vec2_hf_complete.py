
from grid import IntervalTier
from grid import TextGrid
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

from jiwer import cer, wil
from utils import *
import numpy as np

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sampling_rate = 16_000

#replace this with Yaroslav's. HF. submisssion
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

SPEECH_FILE = "/content/must_apologize.wav"
forced_transcript = "|".join("I must apologize for dragging you all here in such an uncommon hour".upper().split())



labels = processor.tokenizer.get_vocab()
with torch.inference_mode():
    waveform, _ = torchaudio.load(SPEECH_FILE)
    emissions = model(waveform.to(device)).logits
    # directly decoding to text
    decoded_transcription_logits, decoded_transcription_ids = emissions.max(dim=-1)
    # for buidlinng the trellis
    emissions = torch.log_softmax(emissions, dim=-1)

emission = emissions[0].cpu().detach()

free_transcript = "|".join(processor.batch_decode(decoded_transcription_ids)[0].split())


dictionary = {c: i for i, c in enumerate(labels)}

forced_tokens = [dictionary[c] for c in forced_transcript]
free_tokens = [dictionary[c] for c in free_transcript]


forced_trellis = get_trellis(emission, forced_tokens)
free_trellis = get_trellis(emission, free_tokens)

forced_path = backtrack(forced_trellis, emission, forced_tokens)
free_path = backtrack(free_trellis, emission, free_tokens)

forced_segments = merge_repeats(forced_path, forced_transcript)
free_segments = merge_repeats(free_path, free_transcript)

forced_word_segments = merge_words(forced_segments)
free_word_segments = merge_words(free_segments)

forced_ratio = (waveform.size(1) / (forced_trellis.size(0) - 1)) / sampling_rate
free_ratio = (waveform.size(1) / (free_trellis.size(0) - 1)) / sampling_rate


free_score = np.mean([w.score for w in free_word_segments])

forced_score = np.mean([w.score for w in forced_word_segments])

print("GOP score: {}".format(1 - np.abs(free_score - forced_score)))
print("CER-based GOP", 1- cer(free_transcript, forced_transcript))


forced_tier = IntervalTier('forcedWords')
forced_tier_char = IntervalTier('forcedChars')
free_tier = IntervalTier('freeWords')
free_tier_char = IntervalTier('freeChars')
txtgrid = TextGrid()

# output forced words
for w in forced_word_segments:
  word = w.label
  start_time = (w.start*forced_ratio)
  end_time = (w.end*forced_ratio) 
  duration = end_time - start_time
  forced_tier.add(start_time, end_time, word)

# output forced chars
for w in forced_segments:
  word = w.label
  start_time = (w.start*forced_ratio)
  end_time = (w.end*forced_ratio) 
  duration = end_time - start_time
  forced_tier_char.add(start_time, end_time, word)

# output free words

for w in free_word_segments:
  word = w.label
  start_time = (w.start*free_ratio)
  end_time = (w.end*free_ratio) 
  duration = end_time - start_time
  free_tier.add(start_time, end_time, word)

# output free chars
for w in free_segments:
  word = w.label
  start_time = (w.start*free_ratio)
  end_time = (w.end*free_ratio) 
  duration = end_time - start_time
  free_tier_char.add(start_time, end_time, word)


txtgrid.append(forced_tier)
txtgrid.append(forced_tier_char)
txtgrid.append(free_tier)
txtgrid.append(free_tier_char)
txtgrid.write('must_apologize.TextGrid')