#!/usr/bin/env scons

import os

from os.path import join
from nestly.scons import SConsWrap
from nestly import Nest
import SCons.Script as sc

# Command line options

sc.AddOption('--output', type='string', help="output folder", default='_output')

env = sc.Environment(
        ENV=os.environ,
        output=sc.GetOption('output'),
        )

sc.Export('env')

env.SConsignFile()

flag = 'exp_cub_birds_existing'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'exp_mimic'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'exp_clinical_notes'
sc.SConscript(flag + '/sconscript', exports=['flag'])