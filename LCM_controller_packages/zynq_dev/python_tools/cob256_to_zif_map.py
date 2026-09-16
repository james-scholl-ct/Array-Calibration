'''This script generates a map for ZIF-to-COB pins for the HX8175-K10 COB.
'''

import yaml
import numpy as np

_YAML_FILE = 'yaml/cob256_to_zif_map.yml'
_COB_WIDTH = 20
_COB_NULL_WIDTH = 12

chars = [v for v in range(ord('A'), ord('W') + 1)
         if chr(v) not in ('I', 'O', 'Q')]
_ZIF_ROW_IDX_MAP = {chr(v): i for i, v in enumerate(reversed(chars))}
_ZIF_COL_IDX_MAP = {i+1: i for i in range(_COB_WIDTH)}
_ZIF_ROW_IDX_IMAP = {v:k for k, v in _ZIF_ROW_IDX_MAP.items()}
_ZIF_COL_IDX_IMAP = {v:k for k, v in _ZIF_COL_IDX_MAP.items()}



def parse_yaml(file_path=_YAML_FILE):
    with open(file_path, 'r') as f:
        y = yaml.safe_load(f)
    return y


def validate_yaml(y):
    # Check that there are 20 rows and 20 columns for each row
    assert(len(y) == _COB_WIDTH)
    assert(all(len(col) == _COB_WIDTH for col in y.values()))

    # Create a flat, sorted list of non-null entries
    flat = []
    for col in y.values():
        flat.extend(col)
    flat = [e for e in flat if isinstance(e, int)]
    flat.sort()

    # Check for no duplicates
    assert(len(set(flat)) == len(flat))

    # Check the quantity of nulls (12x12 grid) vs non-nulls
    assert(len(flat) == _COB_WIDTH**2 - _COB_NULL_WIDTH**2)

    # Check that the flat list is a subset of possible pins
    assert(set(flat) < set(range(1, _COB_WIDTH**2 + 1)))


def get_map(y):
    my_map = []
    for k, v in y.items():
        for i, pin in enumerate(v):
            coord = k + str(i+1)
            pair = (coord, pin)
            my_map.append(pair)
    return my_map


def print_map_by_zif(y):
    my_map = get_map(y)
    for coord, pin in sorted(my_map):
        print("{},{}".format(coord, pin))


def print_map_by_cob(y):
    my_map = get_map(y)
    my_map = [(pin, coord) for coord, pin in my_map if isinstance(pin, int)]
    for pin, coord in sorted(my_map):
        print("{},{}".format(pin, coord))


def print_map_by_cob_rot(y):
    my_map = get_map(y)
    arr = np.zeros((20, 20), dtype=int)
    for coord, pin in my_map:
        letter = coord[0]
        number = coord[1:]
        r = _ZIF_ROW_IDX_MAP[letter]
        c = _ZIF_COL_IDX_MAP[int(number)]
        arr[r][c] = pin if isinstance(pin, int) else -1

    mat = np.matrix(arr)
    mat = np.rot90(mat)
    mat = np.rot90(mat)
    mat = np.rot90(mat)
    arr = np.array(mat)

    new_map = []
    for r in range(20):
        for c in range(20):
            pin = int(arr[r][c]) if arr[r][c] >= 0 else '.'
            letter = _ZIF_ROW_IDX_IMAP[r]
            number = _ZIF_COL_IDX_IMAP[c]
            coord = letter + str(number)
            new_map.append((pin, coord))

    new_map = [(pin, coord) for pin, coord in new_map if isinstance(pin, int)]
    for pin, coord in sorted(new_map):
        print("{},{}".format(pin, coord))


def main():
    y = parse_yaml()
    validate_yaml(y)
    print_map_by_cob_rot(y)


if __name__ == '__main__':
    main()
