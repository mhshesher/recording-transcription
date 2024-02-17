import os
from pydub import AudioSegment

def segment_audio(aud_path, des_path, diarization):

	if diarization is None:
		print('No diarization found.')

	os.makedirs(des_path, exist_ok=True)

	audio=AudioSegment.from_file(aud_path, 'mp3')
	audio=audio.set_frame_rate(16000)

	audio_format='wav'
	large_file=False

	for key, val in diarization.items():
		st=int(diarization[key]['start']*1000)
		ed=int(diarization[key]['end']*1000)
		chunk=audio[st:ed+1]
		audio_name='.'.join([key,audio_format])
		if chunk.duration_seconds>60:
			large_file=True
		chunk.export(os.path.join(des_path, audio_name), format=audio_format)

	return large_file