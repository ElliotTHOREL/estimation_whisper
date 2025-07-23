from pydub import AudioSegment

import torchaudio

from pydub.silence import detect_nonsilent
import torch
import time
import os

from services.database.audio import get_audio_path
from services.database.batch_audio import get_batch_audio_path


def get_full_path(id_audio, batch_audio):
    path_batch = get_batch_audio_path(batch_audio)
    path_spe_audio = get_audio_path(id_audio, batch_audio)
    return os.path.join(path_batch, path_spe_audio)



def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)  
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    waveform = waveform.mean(dim=0)  # mono
    return waveform.numpy(), target_sr



class Audio_file:
    def __init__(self, id_audio, batch_audio):
        self.path_audio = get_full_path(id_audio, batch_audio)
        self.audio_data, self.sr = load_audio(self.path_audio, target_sr=16000)
        self.segment = AudioSegment.from_file(self.path_audio)
        self.liste_chunks = None


    def get_liste_chunks(self):
        if self.liste_chunks == None:
            self.liste_chunks = safe_chunking(self)
        return self.liste_chunks

class Chunk_audio:
    #Un objet avec un Audiofile, une duree, un début, une fin
    def __init__(self, audio_file_origin : Audio_file, start, end):
        self.audio_file_origin = audio_file_origin
        self.sr = audio_file_origin.sr

        self.start_ms = start
        self.end_ms = end
        self.duration_ms = end - start

        self.start_sample = None
        self.end_sample = None

        self.audio_data = None  #lazy loading
        self.segment = None #lazy loading
        
    def get_audio_data (self):
        if self.audio_data == None:
            self.start_sample = int(self.start_ms * self.audio_file_origin.sr / 1000)
            self.end_sample = min (int(self.end_ms * self.audio_file_origin.sr / 1000), len(self.audio_file_origin.audio_data))
            self.audio_data = self.audio_file_origin.audio_data [self.start_sample:self.end_sample]
        return self.audio_data

    def get_segment(self):
        if self.segment == None:
            self.segment = self.audio_file_origin.segment[self.start_ms:self.end_ms]
        return self.segment
       



def safe_chunking(audio_file: Audio_file, min_silence_len=500, silence_thresh=-50 , max_chunk_duration=25000, preferred_duration=20000):
    #Découpage initial
    initial_liste_chunks = []
    start = 0
    print("safe_chunking", time.time())
    nonsilent_ranges = detect_nonsilent(audio_file.segment, 
                                     min_silence_len=min_silence_len,
                                     silence_thresh=silence_thresh)
    print("safe_chunking 2", time.time())

    initial_liste_chunks = [
        Chunk_audio(audio_file, start, end)
        for (start, end) in nonsilent_ranges
    ]
    print("safe_chunking 3", len(initial_liste_chunks), time.time())

    #On améliore le découpage en forçant la redécoupe des chunks trop longs
    final_liste_chunks = []
    for chunk in initial_liste_chunks:
        print("safe_chunking 4", chunk.duration_ms , time.time())
        if chunk.duration_ms > max_chunk_duration:
            sub_chunks = force_split_chunk(chunk, preferred_duration)
            final_liste_chunks.extend(sub_chunks)
        else:
            final_liste_chunks.append(chunk)

    print("safe_chunking 5", len(final_liste_chunks), time.time())
    return final_liste_chunks

def force_split_chunk(chunk: Chunk_audio, target_duration):
    """Division forcée avec recherche de points optimaux"""
    if chunk.duration_ms <= target_duration:
        return [chunk]
    
    chunks = []
    start = chunk.start_ms
    
    while start < chunk.end_ms:
        potential_end = min(start + target_duration, chunk.end_ms)
        
        if potential_end < chunk.end_ms:
            print(chunk.start_ms, chunk.end_ms, potential_end)
            # Chercher un point de coupure moins brutal
            search_window = chunk.audio_file_origin.segment[potential_end-1000:potential_end+1000]  # ±1s
            print(len(search_window)) 
            volume_levels = [search_window[i:i+100].rms 
                           for i in range(0, len(search_window)-100, 100)]
            print(len(volume_levels))
            
            
            # Couper au point le plus silencieux
            min_volume_idx = volume_levels.index(min(volume_levels))
            optimal_cut = potential_end - 1000 + (min_volume_idx * 100)
            end = max(start + target_duration//2, optimal_cut)  # Sécurité
        else:
            end = potential_end

        chunks.append(Chunk_audio(chunk.audio_file_origin, start, end))
        start = end
    
    return chunks




