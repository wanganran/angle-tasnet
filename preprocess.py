import numpy as np
from scipy.signal import stft, istft
sr = 24000
n_mic = 6
n_pair = 6
filter_length = 64
hop_length = 32
window = 'hann'
win_length = 64

pairs = ((0, 3), (1, 4), (2, 5), (0, 1), (2, 3), (4, 5))


m_total = filter_length//2+1
frequency_vector = np.arange(0, m_total)/filter_length*sr # in Hz
n_grid = 36
V = 343
#freqs = np.expand_dims(frequency_vector.reshape((1,1,-1)), (n_mic, n_grid, m_total)) # n_mic, n_grid, frequency


def Prep(angle, mixture, R):
    # mixture: C, L
    input=mixture

    mic_array_layout = R - np.tile(R[:, 0].reshape((3, 1)), (1, n_mic))
    
    
    
    spec = np.stack([stft(input[i], nperseg=filter_length)[2][:m_total] for i in range(n_mic)], axis=0) # C, F, T

    # IPD
    IPD_list = []
    for h, m in enumerate(pairs):
        com_u1 = spec[m[0]]
        com_u2 = spec[m[1]]
        IPD = np.angle(com_u1*com_u2.conj()) # F, T
        IPD_list.append(IPD)
    IPD = np.stack(IPD_list, axis=0) # P, F, T real

    # AF
    AF_up=[]
    AF_down=[]
    for h,m in enumerate(pairs):
        v=__get_h(angle, m, mic_array_layout) #F,
        AF = v.reshape((-1, 1)) * np.exp(1j*IPD[h]) # F,T
        AF_up.append(AF)
        AF_down.append(np.abs(AF))

    AF=np.real(sum([AF_up[i]/AF_down[i] for i in range(n_pair)])) # F, T real

    # DPR
    delay = np.zeros((n_grid, n_mic)) # in meter
    for h in range(n_mic):
        dx = mic_array_layout[0, h] - mic_array_layout[0, 0]
        dy = mic_array_layout[1, h] - mic_array_layout[1, 0]
        delay[:, h] = dx * np.cos(np.arange(n_grid) * 2 * np.pi / n_grid) + dy * np.sin(np.arange(n_grid) * 2 * np.pi / n_grid)
    delay=delay.reshape((n_grid, n_mic, 1)) # D, C, 1
    w = np.exp(-2j * np.pi * frequency_vector.reshape((1,1,-1)) * delay / V) # D, C, F; steering_vector^H, delay-and-sum: y=wx
    
    dprs=np.einsum('ijk,jkl->ikl', w, spec) # D, F, T
    dpr=np.abs(dprs)**2
    DPR=dpr/np.sum(dpr, axis=0, keepdims=True) #D, F, T real
    DPR=DPR[round(angle/2/np.pi*n_grid)]

    T=spec.shape[-1]
    print(AF.shape, IPD.shape, DPR.shape)
    feature_list = [AF.reshape((-1,T)), IPD.reshape((-1,T)), DPR.reshape((-1,T))]

    fusion = np.concatenate(feature_list, axis=0)
    return np.expand_dims(fusion, 0)

def __get_h(angle, pair, mic_array_layout):
    # get delay
    dx = mic_array_layout[0, pair[1]] - mic_array_layout[0, pair[0]]
    dy = mic_array_layout[1, pair[1]] - mic_array_layout[1, pair[0]]
    delay = dx * np.cos(angle) + dy * np.sin(angle)

    return np.exp(1j*2*np.pi*frequency_vector*delay/V) # Fx1; w such that x_0=w^Hx_1

def generate_mic_array(mic_radius: float, n_mics: int, pos):
    """
    Generate a list of Microphone objects
    Radius = 50th percentile of men Bitragion breadth
    (https://en.wikipedia.org/wiki/Human_head)
    """
    import pyroomacoustics as pra
    R = pra.circular_2D_array(center=[pos[0], pos[1]], M=n_mics, phi0=0, radius=mic_radius)
    R = np.concatenate((R, np.ones((1, n_mic)) * pos[2]), axis=0)
    return R


if __name__ == "__main__":
    # global mic array:
    R = generate_mic_array(0.06, 6, (0, 0, 0))
    fusion=Prep(1., np.random.randn(6, 72000), R)
    print(fusion.shape)