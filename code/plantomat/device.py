from __future__ import annotations

import torch


def select_device(requested: str = 'auto') -> torch.device:
    requested = (requested or 'auto').lower()
    if requested == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(requested)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(device)
        print(f'Используется CUDA-устройство: {gpu_name}')
        print('Если eGPU подключена и видна системе как CUDA, обучение пойдёт на ней автоматически.')
    else:
        print('CUDA не обнаружена. Используется CPU.')
    return device
