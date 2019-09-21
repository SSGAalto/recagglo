#============================================================================================
# Name        : parsing.py
# Author      : Samuel Marchal, Sebastian Szyller
# Version     : 1.0
# Copyright   : Copyright (C) Secure Systems Group, Aalto University {https://ssg.aalto.fi/}
# License     : This code is released under Apache 2.0 license
#============================================================================================

import argparse


class Parser:
    def __init__(self):
        parser = argparse.ArgumentParser(description= \
            "RecAgglo clustering. To run: python3 main.py --infile XYZ --outfile XYZ. Override default args as necessary.")

        required= parser.add_argument_group('Required input/output file arguments')
        required.add_argument(
            "--infile", "-i",
            type=str,
            dest="infile",
            required=True,
            help="CSV file containing input data.")

        required.add_argument(
            "--outfile", "-o",
            type=str,
            dest="outfile",
            required=True,
            help="CSV output file containing input data + cluster number.")

        parser.add_argument(
            "--algorithm",
            metavar="INT",
            type=self.__check_positive_int,
            dest="algo",
            action="store",
            default=0,
            help="clustering algorithm to use: 0=RecAgglo, 1=SampleClust, 2=AggloClust")

        parser.add_argument(
            "--weight",
            type=str,
            dest="weight",
            action="store",
            default="[1.]",
            help="List of weight for each attribute [0,0.5,...,2.5] (default weights = 1.)")

        parser.add_argument(
            "--delta_a",
            dest="delta_a",
            metavar="INT",
            type=self.__check_positive_int,
            action="store",
            default=1000,
            help="Threshold for cluster sampling (default = 1000). Must be > 0.")

        parser.add_argument(
            "--delta_fc",
            dest="delta_fc",
            metavar="INT",
            type=self.__check_positive_int,
            action="store",
            default=1,
            help="Threshold for full clustering (default = 1). Must be > 0.")

        parser.add_argument(
            "--d_max",
            dest="d_max",
            metavar="FLOAT",
            type=self.__check_positive_float,
            action="store",
            default=0.5,
            help="Distance threshold to split clusters (default = 0.5). Must be > 0.")

        parser.add_argument(
            "--rho_mc",
            dest="rho_mc",
            metavar="FLOAT",
            type=self.__check_positive_float,
            action="store",
            default=6.0,
            help="Divider of max cluster according to n samples, max_clust = n/mclust (default = 6.0). Must be > 0.")

        parser.add_argument(
            "--rho_s",
            dest="rho_s",
            metavar="FLOAT",
            type=self.__check_positive_float,
            action="store",
            default=0.5,
            help="Multiplier of sqrt(n) for sample size (default = 0.5). Must be > 0.")

        parser.add_argument(
            "--skip_index",
            dest="skip_index",
            default = False,
            action="store_true",
            help="Skip first column if it is an index and not a attribute."
        )

        parser.add_argument(
            "--verbose",
            dest="verbose",
            default = False,
            action="store_true",
            help="Verbose printing."
        )

        self.parser = parser

        self.args = self.parser.parse_args()

    def __check_positive_int(self, value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError("%s is not a positive int value" % value)
        return ivalue

    def __check_positive_float(self, value):
        ivalue = float(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError("%s is not a positive float value" % value)
        return ivalue
