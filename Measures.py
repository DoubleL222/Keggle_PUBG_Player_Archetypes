import math as math
from collections import Counter

def travel_ratio(row):
    player_dist_total = row['player_dist_walk'] + row['player_dist_ride']

    if player_dist_total > 0:
        return row['player_dist_walk'] / player_dist_total
    else:
        return 0


def kill_knockdown_ratio(row):
    kill_knockout_total = row['player_kills'] + row['player_dbno']

    if kill_knockout_total > 0:
        return row['player_kills'] / kill_knockout_total
    else:
        return 0


def kill_distance(row, kill_rows):
    i = 0
    sum = 0
    for index, single_row in kill_rows.iterrows():
        i = i+1
        x1 = single_row['killer_position_x']
        y1 = single_row['killer_position_y']
        x2 = single_row['victim_position_x']
        y2 = single_row['victim_position_y']
        sum += math.hypot(x2 - x1, y2 - y1)

    if i == 0:
        return 0
    return sum/i


def weapon_ratio(row, kill_rows):
    weapon_uses = []
    for index, single_row in kill_rows.iterrows():
        weapon_uses.append(single_row['killed_by'])
    if weapon_uses.__len__() > 0:
        most_common_weapon = Counter(weapon_uses).most_common(1)[0][0]
        return most_common_weapon
    else:
        return 'none'
