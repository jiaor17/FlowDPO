#!/usr/bin/python
# -*- coding:utf-8 -*-
from .diffusion.model import DiffAb
from .MEAN.mc_att_model import MEANDock

import utils.register as R

def create_model(config: dict, **kwargs):
    return R.construct(config, **kwargs)