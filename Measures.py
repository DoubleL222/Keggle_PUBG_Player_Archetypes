import math as math
import numpy as np
from collections import Counter

weapon_types = \
    {
        'Sniper Rifle': [x.lower() for x in ['AWM', 'M24', 'Kar98k', 'Win94', 'MK14', 'VSS', 'SLR', 'MK14 EBR']],
        'Carbine': [x.lower() for x in ['SKS', 'Mini 14']],
        'Assault Rifle': [x.lower() for x in ['Groza', 'AKM', 'AUG A3', 'M16A4', 'M416', 'Scar-L', 'aug']],
        'LMG': [x.lower() for x in ['DP-28', 'M249']],
        'SMG': [x.lower() for x in ['Tommy Gun', 'UMP9', 'Vector', 'UZI', 'Micro UZI']],
        'Shotgun': [x.lower() for x in ['S686', 'S1897', 'S12k']],
        'Pistols and Sidearm': [x.lower() for x in ['Sawed-Off', 'R1895', 'R45', 'P1911', 'P92', 'P18C', 'death.weapsawnoff_c']],
        'Melee': [x.lower() for x in ['Pan', 'Machete', 'Crowbar', 'Sickle', 'Superman Punch', 'Punch']],
        'Crossbow': [x.lower() for x in ['Crossbow']],
        'Throwable': [x.lower() for x in ['Frag Grenade', 'Molotov Cocktail', 'Smoke Grenade', 'Stun Grenade', 'death.projmolotov_damagefield_c' ,'grenade', 'death.projmolotov_c']],
        'Vehicle': [x.lower() for x in ['buggy', 'uaz', 'motorbike', 'van', 'boat', 'pickup truck', 'hit by car', 'motorbike (sidecar)', 'dacia', 'death.pg117_a_01_c', 'aquarail']],
        'Environment': [x.lower() for x in ['falling', 'drown', 'redzone', 'death.redzonebomb_c']],
        'Zone': [x.lower() for x in ['bluezone']],
        'down and out': [x.lower() for x in ['down and out']],
    }


def travel_ratio(row):
    player_dist_total = row['player_dist_walk'] + row['player_dist_ride']

    if player_dist_total > 0:
        return row['player_dist_walk'] / player_dist_total
    else:
        return -1


def kill_knockdown_ratio(row):
    kill_knockout_total = row['player_kills'] + row['player_dbno']

    if kill_knockout_total > 0:
        return row['player_kills'] / kill_knockout_total
    else:
        return -1


def kill_distance(kill_rows):
    i = 0
    distanceSum = 0
    for index, single_row in kill_rows.iterrows():

        x1 = single_row['killer_position_x']
        y1 = single_row['killer_position_y']
        x2 = single_row['victim_position_x']
        y2 = single_row['victim_position_y']
        if not ((x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0)):
            distanceSum += math.hypot(x2 - x1, y2 - y1)
            i = i + 1
    if i == 0:
        return -1
    return distanceSum / i


def weapon_ratio(kill_rows):
    weapon_uses = []
    for index, single_row in kill_rows.iterrows():
        weapon_uses.append(single_row['killed_by'])
    if weapon_uses.__len__() > 0:
        most_common_weapon = Counter(weapon_uses).most_common(1)[0][0]
        return get_weapon_category(most_common_weapon.lower())
    else:
        return 'none'


def avg_player_placement(kill_rows):
    killer_placements = []
    for index, single_row in kill_rows.iterrows():
        killer_placements.append(single_row['killer_placement'])
    if killer_placements.__len__() > 0:
        return np.average(killer_placements)
    return -1


def get_weapon_category(weapon_name):
    for key, value in weapon_types.items():
        if weapon_name in value:
            return key.lower()
    return 'other'
