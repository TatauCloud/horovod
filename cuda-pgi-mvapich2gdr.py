#!/usr/bin/env python

from __future__ import print_function

import argparse
import hpccm
from hpccm.building_blocks import gnu, mlnx_ofed, openmpi, pgi, gdrcopy, ucx, mvapich2_gdr, xpmem, knem
from hpccm.primitives import baseimage

parser = argparse.ArgumentParser(description='HPCCM Tutorial')
parser.add_argument('--format', type=str, default='docker',
                    choices=['docker', 'singularity'],
                    help='Container specification format (default: docker)')

args = parser.parse_args()

Stage0 = hpccm.Stage()

py_ver = '3.6'
mlnx_ofed_version = '4.2-1.0.0.0' # '4.5-1.0.1.0'

#compiler = gnu()

compiler = pgi(eula=True)

Stage0 += baseimage(image='nvidia/cuda:9.0-cudnn7-devel-centos7')
Stage0 += mlnx_ofed(version=mlnx_ofed_version)
# Stage0 += gdrcopy(version='1.3')
Stage0 += compiler
# Stage0 += xpmem()
# Stage0 += knem()
# Stage0 += ucx(version='1.5.0', cuda=True, gdrcopy=True)
# Stage0 += openmpi(cuda=True, infiniband=True, toolchain=compiler.toolchain, version='3.1.2', ucx='/usr/local/ucx')
Stage0 += mvapich2_gdr(version='2.3', cuda_version='9.0', pgi=True, gnu=False)

hpccm.config.set_container_format(args.format)

print(Stage0)
