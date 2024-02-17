import os
import math
from pydub import AudioSegment

src_path='/home/csedu01/Downloads/Segment 2-20240207T180907Z-001 (1)/Segment 2'

fnames=os.listdir(src_path)
fnames.sort()

print('Total audio files:', len(fnames))

total_dur=0
idx=0
for fname in fnames:

	audio=AudioSegment.from_file(os.path.join(src_path, fname), 'mp3')
	audio_dur=audio.duration_seconds
	total_dur+=math.ceil(audio_dur)

	idx+=1

	if idx%500==0:
		print(idx)

duration=total_dur
hours=duration//3600
duration=duration%3600
minutes=duration//60
seconds=duration%60


print('Total duration in seconds:', total_dur, 'seconds')
print('Total duration: %d h %d m %d s'%(hours, minutes, seconds))