#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Shuai Xu  date:2020/3/22  time:17:38
#         School Of Life Sciences, Tsinghua University
#         Beijing Advanced Innovation Center For Structural Biology

from CustomizedFunctions import *
import lumicks.pylake as lk
import os
from bluelake import trap1, trap2, stage, fluidics, pause, power, timeline, reset_force, time, get_force_calibration, \
    piezo_tracker
import sys
import numpy as np
from enum import IntEnum

sys.path.append(r'C:\LUMICKS\Bluelake\LDAT\forcecalibration')
import pscalibration

# Preparations:
# 1.Setup CatchingBeads/FormingTether positions on the Controller PC GUI, FormingTether position should be in the center of the red rectangle
# 2.Select new image templates for bead1 and bead2 on the Controller PC GUI
# 3.Set pressure deviation=0.05 bar
# 4.Make sure the variable names of channel 1/3/4 are the same with ones on GUI
# 5.Make sure the download icons of HF 1x/1y/2x/2y, Bead diameter Template 1/2 are all activated, since they will be used in the ForceCalibration function.

# Customized parameters
PAUSE_SECONDS_FORM_TETHER = 5  # sleep time to form tether at a closed position
REPEAT_TIMES_CHECK_TETHER = 1  # repeat times to check tethers
INTERVAL_SECONDS_FORM_TETHER = 5  # sleep time while forming tether
INTERVAL_SECONDS_PULL_REFOLD_TETHER = 10  # sleep time between pulling and refolding tether
MATCH_REJECTION = 92  # template match score, bead with low score will be dropped
FORCE_THRESHOLD_INIT = 5  # force threshold while Bead2(right) trying to move closer to the Bead1(left)
FORCE_THRESHOLD_FINAL = 25  # force threshold for two beads almost being stuck together
FORCE_STEP_FORM_TETHER = 1.0  # force step size when the beads move closer
DISTANCE_LEFT_LIMITATION = -0.4  # left distance limitation when the two beads are stuck
DISTANCE_OF_STAGE_LEFT = 0.3  # stage's left safety position, stage: curve describing where sample structure starts to open
DISTANCE_OF_STAGE_RIGHT = 1.0  # stage's right safety position
PRESSURE_CATCHING_BEADS = 0.1  # air pressure while catching beads
PRESSURE_FORMING_TETHER = 0.1  # air pressure while forming tether
CHECKPOINT_1_DISTANCE = 0.15  # the first check point, where distance=0.15um
CHECKPOINT_1_FORCE = 25
CHECKPOINT_2_DISTANCE = 0.5  # the second check point
CHECKPOINT_2_FORCE = 5
CHECKPOINT_2_FORCE_THRESHOLD = 20  # force threshold to check tether status at checkpoint 2
CHECKPOINT_3_DISTANCE_START_POINT = 0.55
CHECKPOINT_3_DISTANCE_STOP_POINT = 0.9
CHECKPOINT_3_FORCE__LOWER = 5  # the lower limitation of the third force checkpoint
CHECKPOINT_3_FORCE_UPPER = 30  # the upper limitation of the third force checkpoint
CHECKPOINT_4_DELTA_DISTANCE = 0.2  # the fourth checkpoint to keep two beads far away, using template match function to check if beads lost
CHECKPOINT_SINGLE_TETHER_DISTANCE = 1.1  # the right distance limitation when re-pulling a single tether
CHECKPOINT_SINGLE_TETHER_FORCE = 10  # the right limitation of force while re-pull single tether
FORCE_LIMITATION_SINGLE_TETHER = 65  # the upper limitation of force while pulling single tether
REPEAT_TIMES_PULL_SINGLE_TETHER = 3  # repeat times of re-pulling single tether
INIT_SPEED_CHECK_TETHER_TYPE = 0.05  # init speed
FORCE_RECORD_POINT = 10  # the checkpoint where distinguishing single or multiple tether for the first time, force is the reference value
K_threshold = 9999  # the threshold to distinguish single or multiple tether
SPEED_PULL_SINGLE_TETHER = 0.01  # speed of pulling single tether
EXPORT_BASE_PATH = r'D:\AutoScript'  # H5 file export path
Experiment_Round = 100  # experiment repeat times
ROUNDS_INTERVAL_CALI_DISTANCE_OFFSET = 20  # interval times( of experiment rounds) to re-cali distance offset
STIFFNESS_LOWER_THRESHOLD = 0.4  # lower threshold of stiffness value calculated by force calibration automatically
STIFFNESS_UPPER_THRESHOLD = 1.1
CHN1_NAME = "dsDNA(0.1)"  # channel1's name defined on the Controller PC GUI
CHN3_NAME = "Buffer3"
CHN4_NAME = "Streptavidin"
TRAPPING_LASER = 100
OVERALL_TRAPPING_POWER = 15
TRAP1_SPLIT = 50
BRIGHT_FIELD_LED = 30

# Global Parameters
global marker_file_path
global name_prefix
global flag_record_baseline
global distance_offset
global flag_cali_distance_offset

distance_offset = 0  # the distance offset between the experimental and theoretical curves, which will be changed automatically after original_position_calibration() function
flag_cali_distance_offset = True
flag_record_baseline = True  # flag of recording baseline


class TetherType(IntEnum):
    InvalidTether = 0
    SingleTether = 1
    MultipleTether = 2
    BeadsStuck = 3
    BeadsLost = 4


class TetherStatus(IntEnum):
    Fine = 0
    Broken = 1
    MultipleTether = 2


class ExperimentError_Continue(RuntimeError):
    pass


class ExperimentError_Break(RuntimeError):
    pass


class PowerSpectrumCalibrationError(RuntimeError):
    pass


def CatchBead1InTrap2():
    """
    Catch a bead with trap2( the left trap)
    :return:
    """
    stage.move_to(CHN1_NAME)
    pause(1)
    stage.move_to(CHN1_NAME)  # this repeated function will reduce the stage deviation.
    print("Start to catch bead1...")
    trap1.clear()  # the two traps must be cleared simultaneously, since the match score function can't distinguish the two beads
    trap2.clear()

    counter_reset_trap1 = 0
    counter_reset_trap2 = 0
    counter_failure = 0

    while match_score_bead1.latest_value < MATCH_REJECTION:
        if match_score_bead1.latest_value > 1:  # multi-beads or bead doesn't match
            trap2.clear()
            counter_reset_trap2 = 0
            counter_reset_trap1 += 1
            if counter_reset_trap1 > 3:  # Three consecutive occurrences of valid values
                trap1.clear()  # to avoid the situation: low-quality bead has been trapped by Trap1 while Trap2 is still empty. So Bead1 has a low score.
        else:
            counter_reset_trap1 = 0
        pause(0.01)  # a blink time waiting for the next check round
        counter_reset_trap2 += 1
        if counter_reset_trap2 > 500:  # reset trap2 for every 5 seconds
            trap2.clear()  # to avoid situation: Trap2 already has multi-beads but is not recognised.
            counter_reset_trap2 = 0
            counter_failure += 1
            if counter_failure > 5:  # re-do at most for 5 times
                raise ExperimentError_Continue("Failed to catch bead1 for more than 5 times!")
    print(f"AD bead is matched, last match score:{match_score_bead1.latest_value:.2f}")


def CatchBead2InTrap1():
    """
    Catch a bead with trap1( the right trap)
    :return:
    """
    stage.move_to(CHN4_NAME)
    pause(1)
    stage.move_to(CHN4_NAME)
    print("Start to catch bead2...")
    trap1.clear()
    if match_score_bead1.latest_value == 0:
        raise ExperimentError_Continue("Bead1 lost before catching Bead2!")

    attempt_interval = 0
    catch_attempts = 0
    frames_without_bead1 = 0
    while match_score_bead2.latest_value < MATCH_REJECTION:
        if match_score_bead2.latest_value > 1:
            trap1.clear()
            attempt_interval = 0
        pause(0.01)  # wait for next bead
        attempt_interval += 1
        if attempt_interval > 500:
            trap1.clear()
            attempt_interval = 0
            catch_attempts += 1
            if catch_attempts > 5:
                raise ExperimentError_Continue("Failed to catch bead2 for more than 5 times! Restart a new round.")
        frames_without_bead1 = frames_without_bead1 + 1 if match_score_bead1.latest_value < MATCH_REJECTION else 0  # Bead1 mustn't lost continuously while catching Bead2
        if frames_without_bead1 >= 5:
            raise ExperimentError_Continue("Bead1 lost while catching Bead2!")
        else:
            frames_without_bead1 = 0
    print(f"SA bead is matched, last match score:{match_score_bead2.latest_value:.2f}")
    stage.move_to(CHN3_NAME)
    pause(1)
    stage.move_to(CHN3_NAME)


def RecordBaseline():
    """
    Get baseline data
    :return:
    """
    print("Try to record baseline.")
    print("Stop flow, reset force.")
    fluidics.stop_flow()
    pause(2)
    reset_force()
    print("Move the beads closer to each other.")
    trap1.move_by(dx=-3, speed=0.1)
    pause(2)

    time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    baseline_name = time_str + "_Baseline"
    timeline.mark_begin(baseline_name)
    print("BaseLine Marker begins...")
    trap1.move_by(dx=1, speed=0.1)
    full_path_baseline = os.path.join(EXPORT_BASE_PATH, baseline_name + ".h5")
    timeline.mark_end(export=True, filepath=full_path_baseline)
    print("BaseLine Marker is over.")


def CaliDistanceOffset():
    """
    Calibrate the distance offset( where force is about 20pN)
    :return:
    """
    global flag_record_baseline
    global distance_offset

    print("Move the beads closer to calibrate distance offset.")
    fluidics.open(1, 2, 3, 4, 6)
    SetPressure(PRESSURE_FORMING_TETHER)

    if distance_beads.latest_value <= 0:  # check if bead lost suddenly before moving closer, here if_beads_lost() function can't be used, since template match score is invalid when beads are very close.
        print("Bead lost before moving closer.")
        return TetherType.BeadsLost

    print("Move quickly towards left when the beads are far away.")
    while distance_beads.latest_value > 1:  # move quickly when the beads are far away
        distance_start_pos = distance_beads.latest_value
        trap1.move_by(dx=-0.1, speed=10)
        distance_stop_pos = distance_beads.latest_value
        print(f"Start position:{distance_start_pos:.2f}  Stop position:{distance_stop_pos:.2f}")
        if distance_start_pos <= distance_stop_pos:  # check if beads lost by comparing start/stop position
            print("Beads lost while moving closer.")
            return TetherType.BeadsLost

    print("Reset Force.")
    reset_force()
    pause(0.1)

    print(f"Move slowly to the force threshold={FORCE_THRESHOLD_INIT}pN.")
    while True:  # move towards left direction slowly to the force threshold
        current_force = force.latest_value
        if current_force < FORCE_THRESHOLD_INIT - 5:
            trap1.move_by(dx=-0.01, speed=0.1)  # large step
            print("A large step towards left.")
        elif current_force <= FORCE_THRESHOLD_INIT:
            trap1.move_by(dx=-0.001, speed=0.01)  # small step
            print("A small step towards left.")
        elif distance_beads.latest_value <= DISTANCE_LEFT_LIMITATION:
            print("The two beads are stuck together.")
            return TetherType.BeadsStuck
        else:
            print(f"Move to the force threshold={FORCE_THRESHOLD_INIT}pN successfully.")
            break

    distance_offset = distance_beads.latest_value
    print(f"Distance offset:{distance_offset:.2f}")
    print("Move the beads far away from each other.")
    trap1.move_by(dx=3, speed=0.1)
    if CheckIfBeadsLost():
        print("Beads lost after moving away.")
        return TetherType.BeadsLost
    elif force.latest_value >= 5:
        print("There seems to be a tether formed, as the force={force.latest_value:.2f}pN.")
        return TetherType.SingleTether

    if flag_record_baseline:  # The baseline only needs to be done once when the script starts to run.
        RecordBaseline()
        flag_record_baseline = False  # first run into flag, self-kill after once used
        if CheckIfBeadsLost():
            return TetherType.BeadsLost

    return TetherType.InvalidTether


# def check_offset_cali():
#     step = 0.005
#     distance = CHECKPOINT_1_DISTANCE - distance_offset
#     times = round(distance / step)
#     for i in range(times):  # the first check segment, checking by force ramp
#         trap1.move_by(dx=step, speed=0.02)
#         if force.latest_value > (FORCE_THRESHOLD_INIT + 1):
#             print("Beads are stuck whiling checking offset calibration.")
#             return TetherType.BeadsStuck
#
#     distance = CHECKPOINT_3_DISTANCE_STOP_POINT - CHECKPOINT_1_DISTANCE
#     times = round(distance / step)
#     for i in range(times):  # the second check segment, checking by force threshold
#         trap1.move_by(dx=step, speed=0.02)
#         if force.latest_value > CHECKPOINT_3_FORCE__LOWER:
#             print("Tether formed whiling checking offset calibration.")
#             return TetherType.SingleTether
#
#     return TetherType.InvalidTether


def FormingTether(init_force):
    """
    Forming tether
    :return:
    """
    print("Try to form tether...")
    print("Open selected channels")
    fluidics.open(1, 2, 3, 4, 6)
    SetPressure(PRESSURE_FORMING_TETHER)

    if distance_beads.latest_value <= 0:  # bead lost suddenly before moving closer
        print("Bead lost before forming tether.")
        return TetherType.BeadsLost

    while distance_beads.latest_value > DISTANCE_OF_STAGE_RIGHT:  # move towards left direction
        print(
            f"Beads actual distance {distance_beads.latest_value:.2f} > {DISTANCE_OF_STAGE_RIGHT:.2f}, move closer by step of 1 um")
        distance_tag_start = distance_beads.latest_value
        trap1.move_by(dx=-1, speed=1)
        distance_tag_end = distance_beads.latest_value
        if distance_tag_start <= distance_tag_end:  # check if beads lost by comparing start/stop position
            print(f"Beads lost while moving closer to {DISTANCE_OF_STAGE_RIGHT:.2f}")
            return TetherType.BeadsLost, 0
    print("Reset force and have a little break...")
    reset_force()
    pause(0.1)

    force_threshold = init_force
    cnt = 0
    print(f"Current force threshold:{force_threshold:.2f}")
    while True:
        if force.latest_value > FORCE_THRESHOLD_FINAL:
            print(f'Force > {FORCE_THRESHOLD_FINAL} pN. (Beads are stuck.)')
            return TetherType.BeadsStuck, 0
        elif force.latest_value > force_threshold:
            print(f'Force > {force_threshold} pN. (Tether)')
            print(f"Start to pull tether, waiting for {PAUSE_SECONDS_FORM_TETHER:d} seconds...")
            pause(PAUSE_SECONDS_FORM_TETHER)
            result_type = CheckTetherType(distance_beads.latest_value)
            if result_type == TetherType.InvalidTether:
                cnt += 1
                print(f'Repeat times: {cnt}')
                if cnt >= REPEAT_TIMES_CHECK_TETHER:  # repeat times at the same position
                    force_threshold += FORCE_STEP_FORM_TETHER
                    cnt = 0
                    print(f"No tether formed, move closer by {FORCE_STEP_FORM_TETHER:.2f} pN")
                else:
                    print("No tether formed.")
                print(f"Current force threshold:{force_threshold:.2f}")
            elif result_type == TetherType.SingleTether:
                print("A single tether formed!")
                break
            elif result_type == TetherType.MultipleTether:
                print("A multiple tether formed!")
                break
            elif result_type == TetherType.BeadsLost:
                print("Beads lost while pulling tether!")
                break
            elif result_type == TetherType.BeadsStuck:
                print("Beads are stuck together!")
                break
        else:
            print(f'Force < {force_threshold} pN. (Nothing)')
            if force.latest_value < force_threshold - 3:
                print('A large step towards left.')
                trap1.move_by(dx=-0.01, speed=0.1)  # large step
            else:
                print('A small step towards left.')
                trap1.move_by(dx=-0.001, speed=0.001)  # small step
    print('Formingtether result: ' + str(result_type) + ' Force threshold: ' + str(force_threshold))
    return result_type, force_threshold


def CheckTetherType(delta_distance):
    """
    Check tether type while pulling tether(beads move far away from each other)
    :param delta_distance:
    :return:
    """
    print("Check tether type while pulling tether.")
    speed = INIT_SPEED_CHECK_TETHER_TYPE
    delta_distance_1 = CHECKPOINT_1_DISTANCE - delta_distance
    delta_distance_2 = CHECKPOINT_2_DISTANCE - CHECKPOINT_1_DISTANCE
    delta_distance_3 = CHECKPOINT_3_DISTANCE_START_POINT - CHECKPOINT_2_DISTANCE

    trap1.move_by(dx=delta_distance_1, speed=speed)
    if force.latest_value > CHECKPOINT_1_FORCE:
        speed = 0.5 * speed
    print(f'CheckPoint1, Refer: {CHECKPOINT_1_FORCE} pN, Force: {force.latest_value:.2f} pN')
    # pause(5)

    trap1.move_by(dx=delta_distance_2, speed=speed)
    force_checkpoint_2 = force.latest_value
    if force.latest_value > CHECKPOINT_2_FORCE:
        speed = 0.5 * speed
        print(f'CheckPoint2, Refer: {CHECKPOINT_1_FORCE} pN, Force: {force.latest_value:.2f} pN')
    # pause(5)

    trap1.move_by(dx=delta_distance_3, speed=speed)
    print(f'CheckPoint3, Refer: {CHECKPOINT_1_FORCE} pN, Force: {force.latest_value:.2f} pN')
    # pause(5)
    print('Check tether status')
    while distance_beads.latest_value < CHECKPOINT_3_DISTANCE_STOP_POINT:
        print('Move by a small step of 0.005.')
        trap1.move_by(dx=0.005, speed=0.02)
        # mean_force = GetAverageForce()
        current_force = force.latest_value
        if CHECKPOINT_3_FORCE__LOWER < current_force < CHECKPOINT_3_FORCE_UPPER:
            print('lower < force < upper in CP3.')
            if force_checkpoint_2 > CHECKPOINT_2_FORCE_THRESHOLD:  # 1.Beads2 is trapped by left trap2   2.Beads lost in the previous step
                print('force > threshold in CP2. (Beads lost)')
                result_type = TetherType.BeadsLost
            else:
                print('force < threshold in CP2. (Maybe a single tether)')
                result_type = TetherType.SingleTether
            return result_type
        elif current_force >= CHECKPOINT_3_FORCE_UPPER:
            print('force > upper in CP3.')
            if force_checkpoint_2 > CHECKPOINT_2_FORCE_THRESHOLD:
                print('force > threshold in CP2. (Beads are stuck)')
                result_type = TetherType.BeadsStuck
            else:
                print('force < threshold in CP2. (Tether is multiple)')
                result_type = TetherType.MultipleTether
            return result_type
    print("Move to CheckPoint4 to check if beads lost.")
    trap1.move_by(dx=CHECKPOINT_4_DELTA_DISTANCE, speed=INIT_SPEED_CHECK_TETHER_TYPE)
    if CheckIfBeadsLost():
        result_type = TetherType.BeadsLost
        return result_type
    else:
        result_type = TetherType.InvalidTether  # normal beads without tether
        reset_force()  # remove the vibration on Y-axis which appears after several cycles' pull action.
    print('CheckTetherType result: ' + str(result_type))
    return result_type


# def GetAverageForce(duration=0.5):
#     """
#     Get the mean value of force
#     :param duration:
#     :return:
#     """
#     t0 = timeline.current_time
#     pause(duration)
#     t1 = timeline.current_time
#     mean_force = np.mean(force[t0:t1].data)
#     print(f'Force mean value is: {mean_force:.2f} pN.')
#     return mean_force


def RePullSingleTether():
    """
    Try to pull and fold tether repeatedly
    :return:
    """
    print("Try to pull single tether.")
    while distance_beads.latest_value > DISTANCE_OF_STAGE_LEFT:  # go back to start point
        trap1.move_by(dx=-0.01, speed=SPEED_PULL_SINGLE_TETHER)
    ReleasePressure()
    reset_force()

    global name_prefix
    first_call_flag = True
    for times in range(1, REPEAT_TIMES_PULL_SINGLE_TETHER + 1):
        tag = time.strftime('%M%S', time.localtime())
        name_data = name_prefix + "_FD_%d_" % times + tag
        unfolding_curve_file = os.path.join(EXPORT_BASE_PATH, name_data + ".h5")
        print(f"Re-pull single tether,this is the {times:d} time")

        # positive Force-Extension Curve
        timeline.mark_begin(name_data)
        print("Marker begins...")
        while force.latest_value < FORCE_LIMITATION_SINGLE_TETHER:  # move the right trap by small steps
            print(
                f"Current force {force.latest_value:.2f} < Limit {FORCE_LIMITATION_SINGLE_TETHER:.2f}, pulling tether.")
            trap1.move_by(dx=0.1, speed=SPEED_PULL_SINGLE_TETHER)
            if distance_beads.latest_value > CHECKPOINT_SINGLE_TETHER_DISTANCE:  # double-check right trap's position by distance while moving
                print(
                    f"Current distance {distance_beads.latest_value:.2f} > Upper {CHECKPOINT_SINGLE_TETHER_DISTANCE:.2f} um, stop.")
                if force.latest_value < CHECKPOINT_SINGLE_TETHER_FORCE:  # if distance is long enough but force is still very low, we think it's broken
                    timeline.mark_end(export=True, filepath=unfolding_curve_file)
                    print("Single tether is broken!")
                    return TetherStatus.Broken
                break
            if force.latest_value > FORCE_RECORD_POINT and first_call_flag:  # record an appropriate data point to calculate curve's slope
                X_start_point = distance_beads.latest_value
                Y_start_point = force.latest_value
                first_call_flag = False
        timeline.mark_end(export=True, filepath=unfolding_curve_file)  # a complete curve is saved
        print("Marker ends.")

        X_end_point = distance_beads.latest_value  # try to distinguish tether type, single or multiple one
        Y_end_point = force.latest_value
        K = (Y_end_point - Y_start_point) / (X_end_point - X_start_point)
        print(f"\r\n\r\n\r\n\r\n******K Value:{K:.2f}******\r\n\r\n\r\n\r\n")
        if K >= K_threshold:
            return TetherStatus.MultipleTether

        if INTERVAL_SECONDS_PULL_REFOLD_TETHER:  # wait for seconds
            print(f"wait for {INTERVAL_SECONDS_PULL_REFOLD_TETHER:d} seconds to form the 3D structure.")
            pause(INTERVAL_SECONDS_PULL_REFOLD_TETHER)

        # reverse Force-Extension Curve
        folding_curve_file = os.path.join(EXPORT_BASE_PATH, name_data + "_R.h5")
        timeline.mark_begin(name_data)
        print("Mark begins...")
        while distance_beads.latest_value > DISTANCE_OF_STAGE_LEFT:
            trap1.move_by(dx=-0.01, speed=SPEED_PULL_SINGLE_TETHER)
        timeline.mark_end(export=True, filepath=folding_curve_file)
        print("Mark ends.")

    return TetherStatus.Fine


def SetPressure(pressure=0.1):
    """
    Set air pressure for every channel
    :param pressure:
    :return:
    """
    print(f"Set pressure to {pressure:.2f} bar.")
    pressure_lower_value = pressure
    pressure_upper_value = pressure + 0.1
    # Increase Pressure
    while fluidics.pressure < pressure_lower_value:
        fluidics.increase_pressure()
        print(f"Current pressure < {pressure_lower_value:.2f} bar, increasing pressure.")
        pause(0.2)
    while fluidics.pressure > pressure_upper_value:
        print(f"Current pressure > {pressure_upper_value:.2f} bar, decreasing pressure.")
        fluidics.decrease_pressure()
        pause(0.2)
    print('Pressure is well!')


def ReleasePressure():
    """
    Release air pressure slowly
    :return:
    """
    print('Release pressure.')
    while fluidics.pressure > 0.04:
        fluidics.decrease_pressure()
        pause(3)
    # fluidics.stop_flow()
    interval = 3
    fluidics.close(1)
    pause(interval)
    fluidics.close(2)
    pause(interval)
    fluidics.close(3)
    pause(interval)
    fluidics.close(4)
    pause(interval)
    fluidics.close(6)
    print('Pressure is released.')


def AlignBeads():
    """
    Align the two beads along the Y axis automatically
    :return:
    """
    bead_1y = timeline["Bead position"]["Bead 1 Y"]
    bead_2y = timeline["Bead position"]["Bead 2 Y"]
    # 这里使用match_score_bead1.latest_value作为判断条件更为合适，因为能识别：1.重叠 2.未识别的beads 3.丢失的beads，
    # 但是因为beads在靠近的过程中match效率低，无法正常使用该方法。暂时只能使用bead_1y.latest_value是否为0来判断
    # if match_score_bead1.latest_value < MATCH_REJECTION or match_score_bead2.latest_value < MATCH_REJECTION:
    #     print(f"-------AD bead is matched, last match score:{match_score_bead1.latest_value:.2f}-------")
    #     print(f"-------SA bead is matched, last match score:{match_score_bead2.latest_value:.2f}-------")
    if 0 == bead_1y.latest_value or 0 == bead_2y.latest_value:
        raise ExperimentError_Continue("Beads lost while aligning, return to a new round.")
    delta_y = bead_2y.latest_value - bead_1y.latest_value
    trap1.move_by(dy=delta_y, speed=1)
    print(f"Beads aligned, delta_y is {delta_y:.2f}")


def CheckIfBeadsLost():
    """ Check bead2's status( only useful when the two beads are far away)
    """
    t0 = timeline.current_time
    pause(1.0)
    t1 = timeline.current_time
    status = np.all(match_score_bead2[t0:t1].data < 50)
    if status:
        print('Beads check result: lost.')
    else:
        print('Beads check result: well.')
    return status


def DeleteFile(file):
    """
    Delete local file
    :param file:
    :return:
    """
    piezo_tracker.enabled = False
    if os.path.exists(file):
        try:
            os.remove(file)
        except Exception as e:
            print("Delete mark file failed:" + file)
            print((str(e)))


def ForceCalibration(file_path, which_channel, fit_range=(10, 23000), n_points_per_block=20, end_index=None,
                     start_index=None):
    f = lk.File(file_path)
    if which_channel == "1x":
        channel = f["Force HF"]["Force 1x"]
        calibration = f.force1x.calibration[0]
        bead_diameter = f["Bead diameter"]["Template 2"].data[0]
    elif which_channel == "1y":
        channel = f["Force HF"]["Force 1y"]
        calibration = f.force1y.calibration[0]
        bead_diameter = f["Bead diameter"]["Template 2"].data[0]
    elif which_channel == '2x':
        channel = f["Force HF"]["Force 2x"]
        calibration = f.force2x.calibration[0]
        bead_diameter = f["Bead diameter"]["Template 1"].data[0]
    elif which_channel == '2y':
        channel = f["Force HF"]["Force 2y"]
        calibration = f.force2y.calibration[0]
        bead_diameter = f["Bead diameter"]["Template 1"].data[0]
    if "Response (pN/V)" in calibration.keys():
        response = calibration["Response (pN/V)"]
    else:
        response = 1

    params = pscalibration.CalibrationParameters(bead_diameter=bead_diameter)
    settings = pscalibration.CalibrationSettings(fit_range=fit_range, n_points_per_block=n_points_per_block)
    pscal = pscalibration.PowerSpectrumCalibration(data=((channel.data[start_index:end_index]) / response),
                                                   sampling_rate=channel.sample_rate, params=params, settings=settings)
    pscal.run_fit()
    return pscal.results.kappa


def Step1():
    """ Open valves 1，2，3，4，6, increase pressure.
    """
    print('\n'*2+"Step 1: open valves, increase pressure.")
    if piezo_tracker.enabled:
        piezo_tracker.enabled = False
    print("Turn on valves...")
    fluidics.open(1, 2, 3, 4, 6)
    SetPressure(PRESSURE_CATCHING_BEADS)
    print("Step 1 is over.")


def Step2():
    """ Move the trap1 to the pre-defined CatchingBeads position.
    """
    print('\n'*2+"Step 2: move trap1 to the CatchingBeads position.")
    trap1.move_to("CatchingBeads")
    print("Step 2 is over.")


def Step3():
    """ Move the stage to the channel 1 for catching anti-dig bead,
        then move to the channel 4 for catching streptavidin bead.
    """
    print('\n'*2+"Step 3: catch bead1 and bead2.")
    CatchBead1InTrap2()
    CatchBead2InTrap1()
    print("Step 3 is over.")


def Step4():
    """ Move the trap1 to the pre-defined 'FormingTether' position, close valves, align beads.
    """
    print('\n'*2+"Step 4: move trap1 to the FormingTether position, close valves.")
    trap1.move_to("FormingTether")
    fluidics.stop_flow()
    pause(INTERVAL_SECONDS_FORM_TETHER)
    AlignBeads()
    print("Step 4 is over.")


def Step5(round_ind):
    """ Force calibration, distance calibration, baseline record.
    """
    print('\n'*2+"Step 5: force calibration.")
    name = time.strftime('%Y%m%d_%H%M%S', time.localtime()) + "_Cali"
    cali_file = os.path.join(EXPORT_BASE_PATH, name + ".h5")
    timeline.mark_begin(name)
    print('Cali begin...')
    print('Sleep for 11 sec.')
    pause(11)
    timeline.mark_end(export=True, filepath=cali_file)
    print('Cali end...')

    try:
        stiffness_1x = ForceCalibration(cali_file, "1x")  # Stiffness 1(refer to the trap1) corresponding to bead2
        print(f"Stiffness 1X:{stiffness_1x:.2f}")
        stiffness_1y = ForceCalibration(cali_file, "1y")
        print(f"Stiffness 1Y:{stiffness_1y:.2f}")
        stiffness_2x = ForceCalibration(cali_file, "2x")
        print(f"Stiffness 2X:{stiffness_2x:.2f}")
        stiffness_2y = ForceCalibration(cali_file, "2y")
        print(f"Stiffness 2Y:{stiffness_2y:.2f}")
    except:
        raise PowerSpectrumCalibrationError("Power Spectrum Calibration failed.")
    finally:
        """DeleteFile(cali_file)"""

    if stiffness_1x < STIFFNESS_LOWER_THRESHOLD or stiffness_1x > STIFFNESS_UPPER_THRESHOLD:
        raise PowerSpectrumCalibrationError(f"Stiffness 1X:{stiffness_1x:.2f} is invalid.")
    elif stiffness_1y < STIFFNESS_LOWER_THRESHOLD or stiffness_1y > STIFFNESS_UPPER_THRESHOLD:
        raise PowerSpectrumCalibrationError(f"Stiffness 1Y:{stiffness_1y:.2f} is invalid.")
    elif stiffness_2x < STIFFNESS_LOWER_THRESHOLD or stiffness_2x > STIFFNESS_UPPER_THRESHOLD:
        raise PowerSpectrumCalibrationError(f"Stiffness 2X:{stiffness_2x:.2f} is invalid.")
    elif stiffness_2y < STIFFNESS_LOWER_THRESHOLD or stiffness_2y > STIFFNESS_UPPER_THRESHOLD:
        raise PowerSpectrumCalibrationError(f"Stiffness 2Y:{stiffness_2y:.2f} is invalid.")

    # global flag_cali_distance_offset
    # if (
    #         (round_ind % ROUNDS_INTERVAL_CALI_DISTANCE_OFFSET) == 1) or flag_cali_distance_offset:  # re-cali distance offset until it works normally or cali between fix interval times
    #     print("Enable PT")
    #     if not piezo_tracker.enabled:  # activate PT first
    #         piezo_tracker.enabled = True
    #     tether_type = CaliDistanceOffset()
    #     if tether_type == TetherType.InvalidTether or tether_type == TetherType.SingleTether:
    #         flag_cali_distance_offset = False
    #         print("Calibrate distance offset successfully.")
    #     else:
    #         flag_cali_distance_offset = True
    #         raise ExperimentError_Continue("Distance offset calibration failed.")
    print("Step 5 is over.")


def Step6():
    """ Enable PT, record marker.
    """
    print('\n'*2+"Step 6: Enable PT, record marker.")
    global name_prefix
    name_prefix = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    name_marker = name_prefix + "_Marker"
    global marker_file_path
    marker_file_path = os.path.join(EXPORT_BASE_PATH, name_marker + ".h5")
    if not os.path.exists(EXPORT_BASE_PATH):
        try:
            os.makedirs(EXPORT_BASE_PATH)
        except e:
            print("Illegal directory:" + EXPORT_BASE_PATH)
            raise ExperimentError_Break()

    timeline.mark_begin(name_marker)
    print("Marker begin...")
    print("Enable PT")
    if not piezo_tracker.enabled:
        piezo_tracker.enabled = True
    pause(0.5)  # # the time from PT activate to PT Marker is about 0.383372~0.391360 sec by experiences.
    timeline.mark_end(export=True, filepath=marker_file_path)
    print("Marker end...")
    if CheckIfBeadsLost():
        raise ExperimentError_Continue("Beads lost, return to a new round.")
    print("Step 6 is over.")


def Step7():
    """ Try to form tether repeatedly, check if there's an idealized tether.
    """
    print('\n'*2+"Step 7: try to pull tether.")
    force_threshold = FORCE_THRESHOLD_INIT
    flag_single_tether = False
    global marker_file_path
    while True:
        ret, current_force = FormingTether(force_threshold)
        if ret == TetherType.SingleTether:
            result = RePullSingleTether()
            flag_single_tether = True
            if result == TetherStatus.Fine:
                piezo_tracker.enabled = False
                raise ExperimentError_Continue()
            elif result == TetherStatus.Broken:
                force_threshold = current_force  # continue move left by step from current position.
                continue
            elif result == TetherStatus.MultipleTether:
                print("This is a multiple tether, return to a new round.")
                DeleteFile(marker_file_path)
                raise ExperimentError_Continue()
        elif ret == TetherType.MultipleTether:
            print("A multiple tether is caught, return to a new round.")
            if ~flag_single_tether:
                DeleteFile(marker_file_path)
            raise ExperimentError_Continue()
        elif ret == TetherType.BeadsStuck:
            print("Beads are stuck together, return to a new round.")
            if ~flag_single_tether:
                DeleteFile(marker_file_path)
            raise ExperimentError_Continue()
        elif ret == TetherType.BeadsLost:
            print("Beads lost, return to a new round.")
            if ~flag_single_tether:
                DeleteFile(marker_file_path)
            raise ExperimentError_Continue()
    print("Step 7 is over.")

print(my_add(2,3))
# print('BlueLake version: ' + bluelake.__version__)
# print('Pylake version: ' + lk.__version__)
# distance_beads = timeline["Distance"]["Piezo Distance"]
# force = timeline["Force HF"]["Force 2x"]
# match_score_bead1 = timeline["Tracking Match Score"]["Bead 1"]  # bead in Trap2(the left trap)
# match_score_bead2 = timeline["Tracking Match Score"]["Bead 2"]  # bead in Trap1
#
#
# # Turn on laser, light, etc
# print("Try to turn on Laser, Trap, Light, etc...")
# power.trapping_laser = TRAPPING_LASER
# power.overall_trapping_power = OVERALL_TRAPPING_POWER
# power.trap1_split = TRAP1_SPLIT
# power.bright_field_led = BRIGHT_FIELD_LED
# print("Laser, Trap, Light, etc are turned on successfully.")
#
# if not os.path.exists(EXPORT_BASE_PATH):
#     try:
#         os.makedirs(EXPORT_BASE_PATH)
#     finally:
#         pass
#
# for round_index in range(1, Experiment_Round + 1):
#     print(f"----------This is round {round_index:d}----------")
#     try:
#         Step1()
#         Step2()
#         Step3()
#         Step4()
#         Step5(round_index)
#         Step6()
#         Step7()
#     except ExperimentError_Continue as e:
#         print(e)
#         continue
#     except PowerSpectrumCalibrationError as e:
#         print(e)
#         continue
#     except ExperimentError_Break as e:
#         print(e)
#         break
#     print(f"----------Round {round_index:d} is over.----------")
#
# print("Reset hardware.")
# SetPressure(0)
# fluidics.stop_flow()
# power.trapping_laser = 0
# power.overall_trapping_power = 0
# power.bright_field_led = 0
# print("Script runs over.")
