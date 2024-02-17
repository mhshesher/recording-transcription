import os
import json
import shutil
import torch
# import argparse
import numpy as np
from datetime import datetime
from pyannote.audio import Pipeline
from data_loader import dataset_load
from audio_segmentation import segment_audio
from transformers import AutoProcessor, SeamlessM4Tv2Model
from speech_transcription_seamless import do_transcription
from speaker_diarization_2 import do_diarize


def load_asr_model(model_name):

    processor=AutoProcessor.from_pretrained(model_name)
    model=SeamlessM4Tv2Model.from_pretrained(model_name)

    return processor, model


def load_diarization_model(model_name):

    return Pipeline.from_pretrained(model_name, use_auth_token="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")

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

src_path=os.path.join(root_dir, 'data', 'recordings')
seg_path=os.path.join(root_dir, 'data', 'segments')
txt_path=os.path.join(root_dir, 'data', 'transcripts')
dia_path=os.path.join(root_dir, 'data', 'diarizations')
lf_path=os.path.join(root_dir, 'data', 'large_files')
completed_path=os.path.join(root_dir, 'data', 'completed')


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Available device:", device)


# get_audios(aud_inven, src_path, args.audio_count)
audio_names=os.listdir(src_path)
audio_names.sort()
print(len(audio_names), 'audio files have been loaded...')


# loading ASR model
asr_model_name='facebook/seamless-m4t-v2-large'
processor, model=load_asr_model(asr_model_name)
print('ASR model has been loaded...')


# loading diarization model
diarization_model_name='pyannote/speaker-diarization-3.1'
diarization_pipeline=load_diarization_model(diarization_model_name)
print('Diarization model has been loaded...')

start_time=datetime.now().strftime("%H:%M:%S")

for audio in audio_names:

    large_file=False
    
    audio_name=audio.split('.')[0]

    print('\nTranscribing '+audio_name+'...')

    transcripts_saved=os.listdir(txt_path)
    if audio_name+'_seamless.txt' in transcripts_saved:
        print('Transcription exists.')
        shutil.move(os.path.join(src_path, audio), os.path.join(completed_path, audio))
        continue

    diarizations=os.listdir(dia_path)
    diarizations.sort()
    if audio_name+'.json' not in diarizations:
        print('Diarization not found.')
        sample=dataset_load(os.path.join(src_path, audio))

        diarization=do_diarize(sample.copy(), diarization_pipeline)

        diarization_name=audio_name+'.json'
        with open(os.path.join(dia_path, diarization_name), 'w') as f:
            json.dump(diarization, f)
    else:
        print('Diarization has been loaded.')
        diarization_name=audio_name+'.json'
        with open(os.path.join(dia_path, diarization_name), 'r') as f:
            diarization=json.load(f)

    if os.path.exists(os.path.join(seg_path, audio_name))==False:
        print('Audio segments not found.')
        large_file=segment_audio(os.path.join(src_path, audio), os.path.join(seg_path, audio_name), diarization)
        print('Audio segmentation has been completed.')
    else:
        print('Audio segments has been found.')


    chunks_path=os.path.join(seg_path, audio_name)
    fnames_chunk=os.listdir(chunks_path)
    fnames_chunk.sort()

    transcripts=[]
    for fname_chunk in fnames_chunk:
        sample_chunk=dataset_load(os.path.join(chunks_path, fname_chunk))

        try:
            transcript=do_transcription(sample_chunk, processor, model)
        except:
            transcript='<error>ASR could not transcribe<error>'
            print('error at', fname_chunk)

        key=fname_chunk.split('.')[0]

        transcripts.append('-'.join([key, diarization[key]['speaker'], str(np.around(diarization[key]['start'], decimals=2)), str(np.around(diarization[key]['end'], decimals=2)), transcript]))



    txt_name=audio_name+'_seamless.txt'
    f=open(os.path.join(txt_path, txt_name), 'a+', encoding="utf-8")
    for line in transcripts:
        f.write(line)
        f.write('\n')
    f.close()

    if large_file==True:
        shutil.move(os.path.join(src_path, audio), os.path.join(lf_path, audio))
        print('Moved to large file directory')
    else:
        shutil.move(os.path.join(src_path, audio), os.path.join(completed_path, audio))



end_time=datetime.now().strftime("%H:%M:%S")

print('Start time:', start_time)
print('End time:', end_time)
