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
    return 'Kill Distance (WIP)'


def weapon_ratio(row, kill_rows):
    return 'Weapon Ratio (WIP)'


