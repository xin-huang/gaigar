# GNU General Public License v3.0
# Copyright 2024 Xin Huang
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, please see
#
#    https://www.gnu.org/licenses/gpl-3.0.en.html


def cal_pr(ntruth_tracts, ninferred_tracts, ntrue_positives):
    """
    """
    precision = np.nan if float(ninferred_tracts) == 0 else ntrue_positives / float(ninferred_tracts) * 100
    recall = np.nan if float(ntruth_tracts) == 0 else ntrue_positives / float(ntruth_tracts) * 100

    return precision, recall
