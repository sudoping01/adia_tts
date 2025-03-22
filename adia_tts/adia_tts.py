import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import os
import uuid
from typing import List, Tuple, Dict, Optional, Any

class AdiaTTS:
    """
    This class provides functionality for converting text to speech in Wolof language,
    with support for long text segmentation and audio concatenation.
    """
    
    def __init__(self, 
                 model_id: str = "CONCREE/Adia_TTS", 
                 output_dir: str = "/tmp/tts_output",
                 hf_token: Optional[str] = None):

        self.model_id = model_id
        self.output_dir = output_dir
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.MAX_SEGMENT_LENGTH = 200
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_model()
    
    def load_model(self) -> None:
        try:
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.model_id,
                token=self.hf_token
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=self.hf_token
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def synthesize(self, 
                  text: str, 
                  description: str = "A warm and natural voice, with a conversational flow", 
                  config: Optional[Dict[str, Any]] = None) -> Tuple[str, np.ndarray]:
        """
            Synthesize speech from text of any length
        """
        if config is None:
            config = {
                "temperature": 0.01,
                "max_new_tokens": 1000,
                "do_sample": True,
                "top_k": 50,
                "repetition_penalty": 1.2
            }
        
        try:
   
            segments = self.segment_text(text)
            
            if len(segments) == 1:
                return self._synthesize_single_segment(segments[0], description, config)
            
            else:
                output_files = []
                for segment in segments:
                    output_path, _ = self._synthesize_single_segment(segment, description, config)
                    output_files.append(output_path)
                
                combined_path = self.concatenate_audio_files(output_files)
                
                self.cleanup_temp_files(output_files)
                
                return combined_path, np.array([]) # I will return the array in here to avoid the memory issue
                
        except Exception as e:
            raise Exception(f"Speech synthesis failed: {e}")
    
    def _synthesize_single_segment(self,
                                  text: str,
                                  description: str,
                                  config: Dict[str, Any]) -> Tuple[str, np.ndarray]:
        """
        Synthesize a single text segment
        """
        try:
           
            if len(text) > self.MAX_SEGMENT_LENGTH:
                text = self._truncate_to_natural_boundary(text)
            
            input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
            prompt_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            
            audio = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_ids,
                **config
            )
            
            audio_np = audio.cpu().numpy().squeeze()
            
            output_path = os.path.join(self.output_dir, f"{uuid.uuid4()}.wav")
            sf.write(output_path, audio_np, self.model.config.sampling_rate)
            
            return output_path, audio_np
        except Exception as e:
            raise Exception(f"Single segment synthesis failed: {e}")
    
    def _truncate_to_natural_boundary(self, text: str) -> str:
        """
        Truncate text to a natural boundary within maximum length
        """
        max_pos = min(self.MAX_SEGMENT_LENGTH, len(text))
        pause_chars = ['.', '!', '?', ',', ';', ':', '…']
        
        last_pause = 0
        for char in pause_chars:
            pos = text[:max_pos].rfind(char)
            if pos > last_pause:
                last_pause = pos
        
        if last_pause > 0:
            return text[:last_pause + 1]
        
        last_space = text[:max_pos].rfind(' ')
        if last_space > 0:
            return text[:last_space]
        
        return text[:self.MAX_SEGMENT_LENGTH]
    
    def segment_text(self, text: str) -> List[str]:
        """
        Split text into segments at natural boundaries
        
        This method divides text at natural pause points
        to ensure the most natural-sounding speech when segments are
        concatenated later.
        """
        if len(text) <= self.MAX_SEGMENT_LENGTH:
            return [text]
            
        segments = []
        remaining_text = text.strip()
        
        sentence_end_chars = ['.', '!', '?']
        pause_chars = [',', ';', ':', '…']
        
        while remaining_text:

            if len(remaining_text) <= self.MAX_SEGMENT_LENGTH:
                segments.append(remaining_text)
                break

            last_sentence_end = -1
            for i in range(min(self.MAX_SEGMENT_LENGTH, len(remaining_text))):
                if remaining_text[i] in sentence_end_chars:
                    last_sentence_end = i
            
            if last_sentence_end != -1:
                segment_text = remaining_text[:last_sentence_end + 1]
                remaining_text = remaining_text[last_sentence_end + 1:].strip()
            else:
           
                last_punct = -1
                for i in range(min(self.MAX_SEGMENT_LENGTH, len(remaining_text))):
                    if remaining_text[i] in pause_chars:
                        last_punct = i
                
                if last_punct != -1:
                    segment_text = remaining_text[:last_punct + 1]
                    remaining_text = remaining_text[last_punct + 1:].strip()
                else:
                    
                    text_to_check = remaining_text[:self.MAX_SEGMENT_LENGTH]
                    last_space = text_to_check.rfind(' ')
                    
                    if last_space != -1:
                        segment_text = remaining_text[:last_space]
                        remaining_text = remaining_text[last_space + 1:].strip()
                    else:
                       
                        segment_text = remaining_text[:self.MAX_SEGMENT_LENGTH]
                        remaining_text = remaining_text[self.MAX_SEGMENT_LENGTH:].strip()
            
            segments.append(segment_text)
        
        return segments
    
    def synthesize_from_file(self, 
                           file_path: str, 
                           description: str = "A warm and natural voice, with a conversational flow", 
                           config: Optional[Dict[str, Any]] = None) -> str:
        """
        Synthesize speech from a text file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
            
            audio_path, _ = self.synthesize(
                text=text_content,
                description=description,
                config=config
            )
            
            return audio_path
            
        except Exception as e:
            raise Exception(f"File synthesis failed: {e}")
    
    def concatenate_audio_files(self, file_paths: List[str]) -> str:
        """
        Concatenate multiple audio files 
        """
        if not file_paths:
            raise ValueError("No audio files provided for concatenation")
        
        sample_rate = None
        audio_segments = []
        
 
        for file_path in file_paths:
            data, sr = sf.read(file_path)
            
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                pass # I will come back to here to write the resampling logic but for now it's keep it like 
            
            audio_segments.append(data)
        
        normalized_segments = []
        for segment in audio_segments:
            if np.abs(segment).max() < 0.001:
                normalized_segments.append(segment)
                continue
                
            max_amp = np.abs(segment).max()
            target_amp = 0.7  
            normalized = segment * (target_amp / max_amp)
            normalized_segments.append(normalized)
        

        crossfade_ms = 150  # milliseconds 
        crossfade_samples = int((crossfade_ms / 1000) * sample_rate)
        

        result = normalized_segments[0]
        
 
        for i in range(1, len(normalized_segments)):
            current_segment = normalized_segments[i]
            
            if len(result) < crossfade_samples or len(current_segment) < crossfade_samples:
                actual_crossfade = min(len(result), len(current_segment), crossfade_samples)
            else:
                actual_crossfade = crossfade_samples
            
            if actual_crossfade <= 0:
                result = np.concatenate([result, current_segment])
                continue
            

            t = np.linspace(0, np.pi, actual_crossfade)
            fade_out = np.cos(t) * 0.5 + 0.5  # Smooth fade out from 1 to 0
            fade_in = np.sin(t) * 0.5 + 0.5   # Smooth fade in from 0 to 1
            
            result_end = result[-actual_crossfade:]
            segment_start = current_segment[:actual_crossfade]
            crossfade_region = (result_end * fade_out) + (segment_start * fade_in)
            

            result = np.concatenate([result[:-actual_crossfade], crossfade_region, current_segment[actual_crossfade:]])
        
        output_path = os.path.join(self.output_dir, f"combined_{uuid.uuid4()}.wav")
        sf.write(output_path, result, sample_rate)
    
        return output_path
    
    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model": self.model_id
        }
    
    def cleanup_temp_files(self, file_paths: List[str]) -> None:
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Failed to remove temporary file {file_path}: {e}")