import numpy as np
import torch
import logging
import threading
import time
import wave
from typing import List, Optional
from queue import SimpleQueue, Empty

from whisperlivekit.timed_objects import SpeakerSegment

logger = logging.getLogger(__name__)

try:
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
except ImportError:
    raise SystemExit("""Please use `pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"` to use the Sortformer diarization""")


class StreamingSortformerState:
    """
    This class creates a class instance that will be used to store the state of the
    streaming Sortformer model.

    Attributes:
        spkcache (torch.Tensor): Speaker cache to store embeddings from start
        spkcache_lengths (torch.Tensor): Lengths of the speaker cache
        spkcache_preds (torch.Tensor): The speaker predictions for the speaker cache parts
        fifo (torch.Tensor): FIFO queue to save the embedding from the latest chunks
        fifo_lengths (torch.Tensor): Lengths of the FIFO queue
        fifo_preds (torch.Tensor): The speaker predictions for the FIFO queue parts
        spk_perm (torch.Tensor): Speaker permutation information for the speaker cache
        mean_sil_emb (torch.Tensor): Mean silence embedding
        n_sil_frames (torch.Tensor): Number of silence frames
    """

    def __init__(self):
        self.spkcache = None  # Speaker cache to store embeddings from start
        self.spkcache_lengths = None
        self.spkcache_preds = None  # speaker cache predictions
        self.fifo = None  # to save the embedding from the latest chunks
        self.fifo_lengths = None
        self.fifo_preds = None
        self.spk_perm = None
        self.mean_sil_emb = None
        self.n_sil_frames = None


class SortformerDiarization:
    def __init__(self, model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2"):
        """
        Stores the shared streaming Sortformer diarization model. Used when a new online_diarization is initialized.
        """
        self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        """Load and configure the Sortformer model for streaming."""
        try:
            self.diar_model = SortformerEncLabelModel.from_pretrained(model_name)
            self.diar_model.eval()

            if torch.cuda.is_available():
                self.diar_model.to(torch.device("cuda"))
                print("Using CUDA for Sortformer model")
            else:
                print("Using CPU for Sortformer model")

            self.diar_model.sortformer_modules.chunk_len = 10
            self.diar_model.sortformer_modules.subsampling_factor = 10
            self.diar_model.sortformer_modules.chunk_right_context = 0
            self.diar_model.sortformer_modules.chunk_left_context = 10
            self.diar_model.sortformer_modules.spkcache_len = 188
            self.diar_model.sortformer_modules.fifo_len = 188
            self.diar_model.sortformer_modules.spkcache_update_period = 72  # Reduced from 144 for more frequent updates
            self.diar_model.sortformer_modules.log = False
            # Force the model to consider multiple speakers
            self.diar_model.sortformer_modules.n_spk = 4  # Ensure 4-speaker capability is active
            self.diar_model.sortformer_modules._check_streaming_parameters()
                        
        except Exception as e:
            logger.error(f"Failed to load Sortformer model: {e}")
            raise
 
class SortformerDiarizationOnline:
    def __init__(self, shared_model, sample_rate: int = 16000):
        """
        Initialize the streaming Sortformer diarization system.
        
        Args:
            sample_rate: Audio sample rate (default: 16000)
            model_name: Pre-trained model name (default: "nvidia/diar_streaming_sortformer_4spk-v2")
        """
        self.sample_rate = sample_rate
        self.speaker_segments = []
        self.buffer_audio = np.array([], dtype=np.float32)
        self.segment_lock = threading.Lock()
        self.global_time_offset = 0.0
        self.processed_time = 0.0
        self.debug = False
                
        self.diar_model = shared_model.diar_model
             
        self.audio2mel = AudioToMelSpectrogramPreprocessor(
            window_size=0.025,
            normalize="NA",
            n_fft=512,
            features=128,
            pad_to=0
        )
        
        # Move audio preprocessor to the same device as the model
        if hasattr(self.diar_model, 'device'):
            self.audio2mel = self.audio2mel.to(self.diar_model.device)
            logger.info(f"Moved audio2mel preprocessor to device: {self.diar_model.device}")
        
        self.chunk_duration_seconds = (
            self.diar_model.sortformer_modules.chunk_len * 
            self.diar_model.sortformer_modules.subsampling_factor * 
            self.diar_model.preprocessor._cfg.window_stride
        )
        
        self._init_streaming_state()
        
        self._previous_chunk_features = None
        self._chunk_index = 0
        self._len_prediction = None
        
        # Audio buffer to store PCM chunks for debugging
        self.audio_buffer = []
        
        # Buffer for accumulating audio chunks until reaching chunk_duration_seconds
        self.audio_chunk_buffer = []
        self.accumulated_duration = 0.0
        
        logger.info("SortformerDiarization initialized successfully")


    def _init_streaming_state(self):
        """Initialize the streaming state for the model."""
        batch_size = 1
        device = self.diar_model.device
        
        self.streaming_state = StreamingSortformerState()
        self.streaming_state.spkcache = torch.zeros(
            (batch_size, self.diar_model.sortformer_modules.spkcache_len, self.diar_model.sortformer_modules.fc_d_model), 
            device=device
        )
        self.streaming_state.spkcache_preds = torch.zeros(
            (batch_size, self.diar_model.sortformer_modules.spkcache_len, self.diar_model.sortformer_modules.n_spk), 
            device=device
        )
        self.streaming_state.spkcache_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.streaming_state.fifo = torch.zeros(
            (batch_size, self.diar_model.sortformer_modules.fifo_len, self.diar_model.sortformer_modules.fc_d_model), 
            device=device
        )
        self.streaming_state.fifo_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.streaming_state.mean_sil_emb = torch.zeros((batch_size, self.diar_model.sortformer_modules.fc_d_model), device=device)
        self.streaming_state.n_sil_frames = torch.zeros((batch_size,), dtype=torch.long, device=device)
        
        # Initialize total predictions tensor
        self.total_preds = torch.zeros((batch_size, 0, self.diar_model.sortformer_modules.n_spk), device=device)
        
        # Track how many predictions we've already processed
        self._processed_pred_count = 0

    def insert_silence(self, silence_duration: float):
        """
        Insert silence period by adjusting the global time offset.
        
        Args:
            silence_duration: Duration of silence in seconds
        """
        with self.segment_lock:
            self.global_time_offset += silence_duration
        logger.debug(f"Inserted silence of {silence_duration:.2f}s, new offset: {self.global_time_offset:.2f}s")

    async def diarize(self, pcm_array: np.ndarray):
        """
        Process audio data for diarization in streaming fashion.
        
        Args:
            pcm_array: Audio data as numpy array
        """
        try:
            if self.debug:
                self.audio_buffer.append(pcm_array.copy())

            threshold = int(self.chunk_duration_seconds * self.sample_rate)
            
            self.buffer_audio = np.concatenate([self.buffer_audio, pcm_array.copy()])
            if not len(self.buffer_audio) >= threshold:
                return
            
            audio = self.buffer_audio[:threshold]
            self.buffer_audio = self.buffer_audio[threshold:]
            
            audio_signal_chunk = torch.tensor(audio).unsqueeze(0).to(self.diar_model.device)
            audio_signal_length_chunk = torch.tensor([audio_signal_chunk.shape[1]]).to(self.diar_model.device)
            
            processed_signal_chunk, processed_signal_length_chunk = self.audio2mel.get_features(
                audio_signal_chunk, audio_signal_length_chunk
            )
            
            if self._previous_chunk_features is not None:
                to_add = self._previous_chunk_features[:, :, -99:]
                total_features = torch.concat([to_add, processed_signal_chunk], dim=2)
            else:
                total_features = processed_signal_chunk
            
            self._previous_chunk_features = processed_signal_chunk
            
            chunk_feat_seq_t = torch.transpose(total_features, 1, 2)
            
            with torch.inference_mode():
                left_offset = 8 if self._chunk_index > 0 else 0
                right_offset = 8
                
                # Debug: log chunk processing
                print(f"🎵 DIARIZATION: Processing chunk {self._chunk_index} (time: {self._chunk_index * self.chunk_duration_seconds:.2f}s)")
                
                self.streaming_state, self.total_preds = self.diar_model.forward_streaming_step(
                    processed_signal=chunk_feat_seq_t,
                    processed_signal_length=torch.tensor([chunk_feat_seq_t.shape[1]], device=self.diar_model.device),
                    streaming_state=self.streaming_state,
                    total_preds=self.total_preds,
                    left_offset=left_offset,
                    right_offset=right_offset,
                )
                
            # Convert predictions to speaker segments
            self._process_predictions()
            
            self._chunk_index += 1
            
        except Exception as e:
            logger.error(f"Error in diarize: {e}")
            raise
            
        # TODO: Handle case when stream ends with partial buffer (accumulated_duration > 0 but < chunk_duration_seconds)

    def _process_predictions(self):
        """Process model predictions and convert to speaker segments."""
        try:
            preds_np = self.total_preds[0].cpu().numpy()
            total_pred_count = len(preds_np)
            
            # Only process new predictions since last time
            if total_pred_count <= self._processed_pred_count:
                return
                
            # Get only the new predictions
            new_preds = preds_np[self._processed_pred_count:]
            active_speakers = np.argmax(new_preds, axis=1)
            
            # Debug logging to see what speakers are being predicted
            unique_speakers = np.unique(active_speakers)
            print(f"🔍 DIARIZATION DEBUG: Predicted speakers in this chunk: {unique_speakers}")
            print(f"🔍 DIARIZATION DEBUG: New predictions: {len(new_preds)}, Total predictions: {total_pred_count}")
            print(f"🔍 DIARIZATION DEBUG: Speaker distribution: {[(spk, np.sum(active_speakers == spk)) for spk in unique_speakers]}")
            
            # Force model to consider multiple speakers if stuck on one
            if len(unique_speakers) == 1 and self._chunk_index > 3:
                print(f"⚠️ DIARIZATION WARNING: Only detecting speaker {unique_speakers[0]} for multiple chunks")
                # Show raw prediction probabilities for debugging
                if len(new_preds) > 0:
                    last_frame_probs = new_preds[-1]
                    print(f"🔍 DIARIZATION DEBUG: Raw probabilities for last frame: {last_frame_probs}")
                    print(f"🔍 DIARIZATION DEBUG: Max probability: {np.max(last_frame_probs):.3f} for speaker {np.argmax(last_frame_probs)}")
            
            # Calculate frame duration based on the model's subsampling factor and window stride
            # The model outputs one prediction per frame after subsampling
            # Frame duration = subsampling_factor * window_stride
            frame_duration = (self.diar_model.sortformer_modules.subsampling_factor * 
                            self.diar_model.preprocessor._cfg.window_stride)
            frames_per_chunk = len(new_preds)
            print(f"🔍 DIARIZATION DEBUG: Frame duration: {frame_duration:.3f}s, Frames in chunk: {frames_per_chunk}")
            
            with self.segment_lock:
                # Process predictions into segments
                # Use the actual time based on processed predictions
                base_time = self._processed_pred_count * frame_duration + self.global_time_offset
                
                for idx, spk in enumerate(active_speakers):
                    start_time = base_time + idx * frame_duration
                    end_time = base_time + (idx + 1) * frame_duration
                    
                    # Check if this continues the last segment or starts a new one
                    if (self.speaker_segments and 
                        self.speaker_segments[-1].speaker == spk and 
                        abs(self.speaker_segments[-1].end - start_time) < frame_duration * 1.0):  # Reduced tolerance for better speaker separation
                        # Continue existing segment
                        self.speaker_segments[-1].end = end_time
                    else:
                        # Create new segment
                        new_segment = SpeakerSegment(
                            speaker=spk,
                            start=start_time,
                            end=end_time
                        )
                        self.speaker_segments.append(new_segment)
                        
                        # Log when a new speaker is detected
                        if not self.speaker_segments[:-1] or all(seg.speaker != spk for seg in self.speaker_segments[:-1][-10:]):
                            print(f"🎤 DIARIZATION: New speaker {spk} detected at {start_time:.2f}s")
                
                # Merge very short segments with adjacent ones from the same speaker
                self._consolidate_segments()
                
                # Update processed prediction count
                self._processed_pred_count = total_pred_count
                
                # Update processed time
                self.processed_time = max(self.processed_time, base_time + frames_per_chunk * frame_duration)
                
                logger.debug(f"Processed chunk {self._chunk_index}, total segments: {len(self.speaker_segments)}")
                print(f"🔍 DIARIZATION DEBUG: Processed {len(new_preds)} new predictions, total processed: {self._processed_pred_count}")
                
        except Exception as e:
            logger.error(f"Error processing predictions: {e}")

    def _consolidate_segments(self):
        """Consolidate short segments and merge adjacent segments from the same speaker."""
        if len(self.speaker_segments) < 2:
            return
            
        consolidated = []
        current = self.speaker_segments[0]
        
        for next_segment in self.speaker_segments[1:]:
            # If same speaker and close timing, merge them
            if (current.speaker == next_segment.speaker and 
                next_segment.start - current.end < 0.5):  # Gap less than 0.5 seconds
                current.end = next_segment.end
            else:
                consolidated.append(current)
                current = next_segment
        
        consolidated.append(current)
        self.speaker_segments = consolidated

    def assign_speakers_to_tokens(self, tokens: list, use_punctuation_split: bool = False) -> list:
        """
        Assign speakers to tokens based on timing overlap with speaker segments.
        
        Args:
            tokens: List of tokens with timing information
            use_punctuation_split: Whether to use punctuation for boundary refinement
            
        Returns:
            List of tokens with speaker assignments
        """
        with self.segment_lock:
            segments = self.speaker_segments.copy()
        
        if not segments or not tokens:
            logger.debug("No segments or tokens available for speaker assignment")
            return tokens
        
        print(f"🔍 DIARIZATION DEBUG: Assigning speakers to {len(tokens)} tokens using {len(segments)} segments")
        
        # Debug logging: show available segments
        for i, segment in enumerate(segments[-5:]):  # Show last 5 segments
            print(f"🔍 DIARIZATION DEBUG: Segment {i}: Speaker {segment.speaker} from {segment.start:.2f}s to {segment.end:.2f}s")
        
        use_punctuation_split = False
        if not use_punctuation_split:
            # Simple overlap-based assignment with improved logic
            for token in tokens:
                token.speaker = -1  # Default to no speaker
                best_overlap = 0.0
                best_speaker = -1
                
                for segment in segments:
                    # Calculate overlap between token and segment
                    overlap_start = max(token.start, segment.start)
                    overlap_end = min(token.end, segment.end)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    
                    if overlap_duration > best_overlap:
                        best_overlap = overlap_duration
                        best_speaker = segment.speaker + 1  # Convert to 1-based indexing
                
                # If we found any overlap, assign the speaker
                if best_overlap > 0:
                    token.speaker = best_speaker
                    print(f"🔍 DIARIZATION DEBUG: Token '{token.text}' ({token.start:.2f}-{token.end:.2f}s) assigned to Speaker {token.speaker} (overlap: {best_overlap:.2f}s)")
                else:
                    # If no overlap, assign to the closest segment
                    closest_distance = float('inf')
                    closest_speaker = -1
                    
                    for segment in segments:
                        # Distance to segment
                        if token.end <= segment.start:
                            distance = segment.start - token.end
                        elif token.start >= segment.end:
                            distance = token.start - segment.end
                        else:
                            distance = 0  # Token is within segment
                        
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_speaker = segment.speaker + 1
                    
                    # If very close to a segment (within 1 second), assign to it
                    if closest_distance < 1.0 and closest_speaker != -1:
                        token.speaker = closest_speaker
                        print(f"🔍 DIARIZATION DEBUG: Token '{token.text}' ({token.start:.2f}-{token.end:.2f}s) assigned to closest Speaker {token.speaker} (distance: {closest_distance:.2f}s)")
                    else:
                        print(f"🔍 DIARIZATION DEBUG: Token '{token.text}' ({token.start:.2f}-{token.end:.2f}s) got no speaker assignment (closest distance: {closest_distance:.2f}s)")
        else:
            # Use punctuation-aware assignment (similar to diart_backend)
            tokens = self._add_speaker_to_tokens_with_punctuation(segments, tokens)
        
        return tokens

    def _add_speaker_to_tokens_with_punctuation(self, segments: List[SpeakerSegment], tokens: list) -> list:
        """
        Assign speakers to tokens with punctuation-aware boundary adjustment.
        
        Args:
            segments: List of speaker segments
            tokens: List of tokens to assign speakers to
            
        Returns:
            List of tokens with speaker assignments
        """
        punctuation_marks = {'.', '!', '?'}
        punctuation_tokens = [token for token in tokens if token.text.strip() in punctuation_marks]
        
        # Convert segments to concatenated format
        segments_concatenated = self._concatenate_speakers(segments)
        
        # Adjust segment boundaries based on punctuation
        for ind, segment in enumerate(segments_concatenated):
            for i, punctuation_token in enumerate(punctuation_tokens):
                if punctuation_token.start > segment['end']:
                    after_length = punctuation_token.start - segment['end']
                    before_length = segment['end'] - punctuation_tokens[i - 1].end if i > 0 else float('inf')
                    
                    if before_length > after_length:
                        segment['end'] = punctuation_token.start
                        if i < len(punctuation_tokens) - 1 and ind + 1 < len(segments_concatenated):
                            segments_concatenated[ind + 1]['begin'] = punctuation_token.start
                    else:
                        segment['end'] = punctuation_tokens[i - 1].end if i > 0 else segment['end']
                        if i < len(punctuation_tokens) - 1 and ind - 1 >= 0:
                            segments_concatenated[ind - 1]['begin'] = punctuation_tokens[i - 1].end
                    break
        
        # Ensure non-overlapping tokens
        last_end = 0.0
        for token in tokens:
            start = max(last_end + 0.01, token.start)
            token.start = start
            token.end = max(start, token.end)
            last_end = token.end
        
        # Assign speakers based on adjusted segments
        ind_last_speaker = 0
        for segment in segments_concatenated:
            for i, token in enumerate(tokens[ind_last_speaker:]):
                if token.end <= segment['end']:
                    token.speaker = segment['speaker']
                    ind_last_speaker = i + 1
                elif token.start > segment['end']:
                    break
        
        return tokens

    def _concatenate_speakers(self, segments: List[SpeakerSegment]) -> List[dict]:
        """
        Concatenate consecutive segments from the same speaker.
        
        Args:
            segments: List of speaker segments
            
        Returns:
            List of concatenated speaker segments
        """
        if not segments:
            return []
            
        segments_concatenated = [{"speaker": segments[0].speaker + 1, "begin": segments[0].start, "end": segments[0].end}]
        
        for segment in segments[1:]:
            speaker = segment.speaker + 1
            if segments_concatenated[-1]['speaker'] != speaker:
                segments_concatenated.append({"speaker": speaker, "begin": segment.start, "end": segment.end})
            else:
                segments_concatenated[-1]['end'] = segment.end
                
        return segments_concatenated

    def get_segments(self) -> List[SpeakerSegment]:
        """Get a copy of the current speaker segments."""
        with self.segment_lock:
            return self.speaker_segments.copy()

    def clear_old_segments(self, older_than: float = 30.0):
        """Clear segments older than the specified time."""
        with self.segment_lock:
            current_time = self.processed_time
            self.speaker_segments = [
                segment for segment in self.speaker_segments 
                if current_time - segment.end < older_than
            ]
            logger.debug(f"Cleared old segments, remaining: {len(self.speaker_segments)}")

    def close(self):
        """Close the diarization system and clean up resources."""
        logger.info("Closing SortformerDiarization")
        with self.segment_lock:
            self.speaker_segments.clear()
        
        if self.debug:
            concatenated_audio = np.concatenate(self.audio_buffer)
            audio_data_int16 = (concatenated_audio * 32767).astype(np.int16)                
            with wave.open("diarization_audio.wav", "wb") as wav_file:
                wav_file.setnchannels(1)  # mono audio
                wav_file.setsampwidth(2)   # 2 bytes per sample (int16)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data_int16.tobytes())
            logger.info(f"Saved {len(concatenated_audio)} samples to diarization_audio.wav")


def extract_number(s: str) -> int:
    """Extract number from speaker string (compatibility function)."""
    import re
    m = re.search(r'\d+', s)
    return int(m.group()) if m else 0


if __name__ == '__main__':
    import asyncio
    import librosa
    
    async def main():
        """TEST ONLY."""
        an4_audio = 'audio_test.mp3'
        signal, sr = librosa.load(an4_audio, sr=16000)
        signal = signal[:16000*30]

        print("\n" + "=" * 50)
        print("ground truth:")
        print("Speaker 0: 0:00 - 0:09")
        print("Speaker 1: 0:09 - 0:19") 
        print("Speaker 2: 0:19 - 0:25")
        print("Speaker 0: 0:25 - 0:30")
        print("=" * 50)
        
        diarization = SortformerDiarization(sample_rate=16000)        
        chunk_size = 1600
        
        for i in range(0, len(signal), chunk_size):
            chunk = signal[i:i+chunk_size]
            await diarization.diarize(chunk)
            print(f"Processed chunk {i // chunk_size + 1}")
        
        segments = diarization.get_segments()
        print("\nDiarization results:")
        for segment in segments:
            print(f"Speaker {segment.speaker}: {segment.start:.2f}s - {segment.end:.2f}s")
    
    asyncio.run(main())
