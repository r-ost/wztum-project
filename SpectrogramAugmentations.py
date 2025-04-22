import torchaudio.transforms as T


class SpectrogramAugmentations:
    """Collection of spectrogram augmentation techniques"""
    
    @staticmethod
    def time_masking(spectrogram, time_mask_param=10):
        """Apply time masking to the spectrogram"""
        return T.TimeMasking(time_mask_param)(spectrogram)
    
    @staticmethod
    def freq_masking(spectrogram, freq_mask_param=10):
        """Apply frequency masking to the spectrogram"""
        return T.FrequencyMasking(freq_mask_param)(spectrogram)
