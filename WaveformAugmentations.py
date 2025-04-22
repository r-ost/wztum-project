import numpy as np
import torch
import torchaudio.transforms as T
import random
import torchaudio.functional as F


class WaveformAugmentations:
    @staticmethod
    def time_shift(waveform, shift_limit=0.4):
        shift = int(np.random.uniform(-shift_limit, shift_limit) * waveform.shape[0])
        if shift > 0:
            waveform = torch.cat([waveform[shift:], torch.zeros(shift)], dim=0)
        elif shift < 0:
            shift = -shift
            waveform = torch.cat([torch.zeros(shift), waveform[:-shift]], dim=0)
        return waveform
    
    @staticmethod
    def add_noise(waveform, noise_level=0.005):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    @staticmethod
    def pitch_shift(waveform, sample_rate, pitch_shift_range=(-5.0, 5.0)):
        n_steps = np.random.uniform(pitch_shift_range[0], pitch_shift_range[1])
        pitch_shifter = T.PitchShift(sample_rate=sample_rate, n_steps=n_steps)
        shifted_waveform = pitch_shifter(waveform)
        return shifted_waveform.detach()
    
    @staticmethod
    def volume_control(waveform, gain_range=(0.5, 1.5)):
        """Adjusts the volume of the audio by a random factor"""
        gain = np.random.uniform(gain_range[0], gain_range[1])
        waveform_float = waveform.float()
        augmented = torch.clamp(waveform_float * gain, -1.0, 1.0)
        return augmented
    
    @staticmethod
    def speed_change(waveform, sample_rate, speed_range=(0.8, 1.2)):
        """Changes the speed and pitch of the audio simultaneously"""
        speed_factor = np.random.uniform(speed_range[0], speed_range[1])
        resampler = T.Resample(orig_freq=sample_rate, new_freq=int(sample_rate / speed_factor))
        waveform = resampler(waveform)
        target_length = int(len(waveform) * speed_factor)
        if len(waveform) < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - len(waveform)), mode='constant', value=0)
        else:
            waveform = waveform[:target_length]
        return waveform
            
    @staticmethod
    def reverb(waveform, sample_rate, reverb_range=(0.5, 0.8)):
        reverb_amount = np.random.uniform(reverb_range[0], reverb_range[1])
        delay_samples = int(0.02 * sample_rate)  # 20ms delay
        decay_factor = reverb_amount
        reverb_waveform = waveform.clone()
        for i in range(3):  # Apply echoes
            delay = delay_samples * (i + 1)
            attenuation = decay_factor ** (i + 1)
            padded = torch.zeros_like(waveform)
            padded[delay:] = waveform[:-delay] * attenuation if delay < len(waveform) else torch.zeros(len(waveform) - delay)
            reverb_waveform += padded
        return reverb_waveform / (1 + reverb_amount * 1.5)
    
    @staticmethod
    def mix_background(waveform, background_waveform, mix_ratio_range=(0.05, 0.2)):
        mix_ratio = np.random.uniform(mix_ratio_range[0], mix_ratio_range[1])
        if len(background_waveform) >= len(waveform):
            # Randomly choose a start position in background
            start = random.randint(0, len(background_waveform) - len(waveform))
            bg_segment = background_waveform[start:start+len(waveform)]
        else:
            # Pad background to match waveform length
            bg_segment = torch.zeros_like(waveform)
            bg_segment[:len(background_waveform)] = background_waveform
        
        return (1 - mix_ratio) * waveform + mix_ratio * bg_segment

    @staticmethod
    def convolution_reverb(waveform, rir_waveform):
        """Apply convolution reverb using a room impulse response (RIR)"""
        rir_waveform = rir_waveform / torch.linalg.vector_norm(rir_waveform, ord=2)
        augmented = F.fftconvolve(waveform.unsqueeze(0), rir_waveform.unsqueeze(0)).squeeze(0)
        
        # Ensure output has same length as input
        if len(augmented) > len(waveform):
            augmented = augmented[:len(waveform)]
        elif len(augmented) < len(waveform):
            augmented = torch.nn.functional.pad(augmented, (0, len(waveform) - len(augmented)))
        return augmented