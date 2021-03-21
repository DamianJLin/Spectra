from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import re


class WaveData:
    def __init__(self, fname):
        if isinstance(fname, str):
            rate, data = None, None
            try:
                rate, data = wavfile.read(fname)
            except FileNotFoundError:
                exit('File Not Found')
            assert rate is not None and data is not None

            self.fname = re.sub(r'.*/', '', fname)

            self.data = data
            self.sample_rate = rate
            self.sample_count = data.shape[0]
            self.duration = data.shape[0] / rate
            self.sample_period = 1 / rate

    def decompose(self, subduration=2, f_max=20):
        """
        Decompose the wave data into varying harmonics.
        :param subduration: Approximate duration for each subsection to undergo fourier analysis.
        :param f_max: Maximum frequency to be plotted in the spectrum.
        :return: (ax, (fq, sp)) where ax is the matplotlib.pyplot axis object for the plot, fq is the frequency data and
        sp is the spectral data.
        """
        if subduration > self.duration:
            raise ValueError('Split duration longer than original duration.')

        # Split the wave data into sections of duration ~subduration.
        n_parts = int(np.round(self.duration / subduration))

        # Truncate the data to split into equal parts.
        trunc_size = self.data.size - (self.data.size % n_parts)
        trunc_data = self.data[0:trunc_size]

        # Error thrown by np.split if segments are of non-equal size. This guarantees that the fft frequency bins will
        # be the same for all segments.
        segs = np.split(trunc_data, n_parts)

        def fourier_transform(seg):

            win = signal.windows.gaussian(
                seg.size,
                0.15 * seg.size
            )

            data = win * seg

            return np.abs(np.fft.fft(data)) ** 2

        fq = np.fft.fftshift(
            np.fft.fftfreq(trunc_size // n_parts, d=self.sample_period)
        )

        specs = list(map(fourier_transform, segs))

        specs = list(map(np.fft.fftshift, specs))

        sp = np.mean(specs, axis=0)

        fig, ax = plt.subplots()

        ax.plot(fq, sp)
        ax.set_yscale('log')
        ax.grid('minor')
        ax.set_xlabel('Frequency $f$ (Hz)')
        ax.set_ylabel('Amplitude $Y$ (arb. u.)')
        ax.set_title(f'Spectrum of Alpha Waves Candidate {self.fname}')
        ax.set_xlim(0, f_max)

        return ax, (fq, sp)
