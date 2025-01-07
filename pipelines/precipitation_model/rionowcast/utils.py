# -*- coding: utf-8 -*-
"""
Utils file
"""


import numpy as np


def calculate_opacity(pixel):
    """Função para calcular a transparência dependendo da proximidade com o branco"""
    # Distância do pixel ao branco (255, 255, 255)
    distance_to_white = np.linalg.norm(np.array(pixel[:3]) - np.array([255, 255, 255]))

    # Normalizar a distância para a faixa [0, 255]
    # (quanto mais próximo do branco, maior a transparência)
    return int(np.clip(distance_to_white, 0, 255))
