import torch


def do_diarize(sample, pipeline):

	# device = "cuda:0" if torch.cuda.is_available() else "cpu"
	if torch.cuda.is_available():
		pipeline.to(torch.device("cuda"))

	input_tensor=torch.from_numpy(sample["audio"]["array"][None, :]).float()
	outputs=pipeline({"waveform": input_tensor, "sample_rate": sample["audio"]["sampling_rate"]})

	diarization={}
	speaker_buff=-1
	start_buff=-1
	end_buff=-1
	idx=1
	idx_len=0

	for segment, track, label in outputs.itertracks(yield_label=True):
		if label!=speaker_buff:
			if speaker_buff!=-1:
				prefix=str(idx)
				if idx<10: prefix='0'+prefix
				diarization['turn_'+prefix]={'speaker': speaker_buff, 'start': start_buff, 'end': end_buff}
				idx+=1
			speaker_buff=label
			start_buff=segment.start
		end_buff=segment.end

		if idx_len==len(outputs)-1:
			prefix=str(idx)
			if idx<10: prefix='0'+prefix
			diarization['turn_'+prefix]={'speaker': speaker_buff, 'start': start_buff, 'end': end_buff}
			idx+=1
		idx_len+=1

	return diarization
