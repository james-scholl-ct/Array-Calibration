# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:08:22 2026

@author: uconn
"""

from lcm_board import DeloreanBoard
import numpy as np

#himax rail numbers indexed from 1 that correspond to hb/lb element locations on 288 element dual patch antenna
hb_map = [[797, 793, 795, 799, 3, 7, 5, 1], [781, 777, 779, 783, 19, 23, 21, 17], [773, 769, 771, 775, 27, 31, 29, 25], [757, 753, 755, 759, 43, 47, 45, 41], [749, 745, 747, 751, 51, 55, 53, 49], [733, 729, 731, 735, 67, 71, 69, 65], [725, 721, 723, 727, 75, 79, 77, 73], [709, 705, 707, 711, 91, 95, 93, 89], [701, 697, 699, 703, 99, 103, 101, 97], [685, 681, 683, 687, 115, 119, 117, 113], [677, 673, 675, 679, 123, 127, 125, 121], [661, 657, 659, 663, 139, 143, 141, 137], [649, 653, 655, 651, 151, 147, 145, 149], [633, 637, 639, 635, 167, 163, 161, 165], [625, 629, 631, 627, 175, 171, 169, 173], [609, 613, 615, 611, 191, 187, 185, 189], [601, 605, 607, 603, 199, 195, 193, 197], [585, 589, 591, 587, 215, 211, 209, 213], [577, 581, 583, 579, 223, 219, 217, 221], [561, 565, 567, 563, 239, 235, 233, 237], [553, 557, 559, 555, 247, 243, 241, 245], [537, 541, 543, 539, 263, 259, 257, 261], [529, 533, 535, 531, 271, 267, 265, 269], [513, 517, 519, 515, 287, 283, 281, 285]]
lb_map = [[789, 785, 787, 791, 11, 15, 13, 9], [765, 761, 763, 767, 35, 39, 37, 33], [741, 737, 739, 743, 59, 63, 61, 57], [717, 713, 715, 719, 83, 87, 85, 81], [693, 689, 691, 695, 107, 111, 109, 105], [669, 665, 667, 671, 131, 135, 133, 129], [641, 645, 647, 643, 159, 155, 153, 157], [617, 621, 623, 619, 183, 179, 177, 181], [593, 597, 599, 595, 207, 203, 201, 205], [569, 573, 575, 571, 231, 227, 225, 229], [545, 549, 551, 547, 255, 251, 249, 253], [521, 525, 527, 523, 279, 275, 273, 277]]
interposer_map = [x for x in range(1,1022, 2)] #Himax rail numbers starting from 1 that correspond to DATA1, DATA2, ... DATA511 on interposer

class InvalidTypeError(Exception):
    pass

def make_steering_array(voltages, band='lb'):
    steering_arr = np.full(1021, 9)
    voltages = np.array(voltages)
    if band == 'lb':
        if voltages.shape != (12,8):
            raise InvalidTypeError("LB array must be (12,8)")
        for i, row in enumerate(lb_map):
            for j, col in enumerate(row):
                steering_arr[col-1] = voltages[i][j]
    elif band == 'hb':
        if voltages.shape != (24,8):
            raise InvalidTypeError("HB array must be (24,8)")
        for i, row in enumerate(hb_map):
            for j, col in enumerate(row):
                steering_arr[col-1] = voltages[i][j]
    else:
        raise InvalidTypeError('Select hb or lb')
    return steering_arr

def make_steering_array_dualband(hb_voltages, lb_voltages):
    steering_arr = np.full(1021, 9)
    hb_voltages = np.array(hb_voltages)
    lb_voltages = np.array(lb_voltages)
    
    if hb_voltages.shape != (12,8) or hb_voltages.shape != (24,8):
        raise InvalidTypeError("LB array must be (12,8), HB array must be (24,8)")
    for i, row in enumerate(lb_map):
        for j, col in enumerate(row):
            steering_arr[col-1] = lb_voltages[i][j]
    for i, row in enumerate(hb_map):
        for j, col in enumerate(row):
            steering_arr[col-1] = hb_voltages[i][j]    
            
#uses interposer map to crate an array that drives a specifc DATA pin on chip on flex interposer board
def make_element_driving_array(voltages):
    voltages = np.array(voltages)
    if len(voltages)>511 or voltages.ndim != 1:
        raise InvalidTypeError("There are only 511 voltage outputs on interposer baord, array should be 1D")
    driving_arr = np.full(1021, 9)
    for index, v in enumerate(voltages):
        driving_arr[interposer_map[index]-1] = v
    return driving_arr

class DB:
    def __init__(self):
        self.db = DeloreanBoard('dl-32')
     
    def steer(self, pattern, channel='tx'):
        # steer LCM to order 300
        self.db.write_pattern_v(v_pattern=pattern,
                                tx_or_rx=channel)
        
    def standby(self):
        # put LCM in standby mode (e.g. when taking a short break)
        self.db.lcm_standby_mode()
    
    def shutdown(self):
        # shut down hardware (e.g. when taking a long break)
        self.db.shutdown()

def main(): 
    voltages = np.ones((24,8))
    voltages[0][7] = 5
    print(make_steering_array(voltages, 'hb'))   
    print(len(interposer_map))

if __name__ == "__main__":
    main()