#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
desc:

@author: admin
@date: 2024/3/26
"""

size_factors = ['ln_capital',]
style_factors = ['BETA', 'BTOP', 'EARNYILD', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'MOMENTUM', 'RESVOL', 'SIZE']
industry_factors = [f'ZX_Class_{x+1}' for x in range(30)]