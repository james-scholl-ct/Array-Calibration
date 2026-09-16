import pathlib
import pandas as pd


def load_standard_voltage_patterns(phi_i: float,
                                   delv_max: int):
    """
    Args:
        phi_i: laser incidence angle in degrees
        delv_max: max allowed delv between neighboring LCM rails

    Returns:
        voltage_patterns: pandas dataframe of voltage patterns, column = order

    """
    # Load standard voltage patterns
    current_dir = pathlib.Path(__file__).parent.parent
    if phi_i < 0:
        voltage_pattern_filepath = pathlib.Path(current_dir, 'delta',
                                                f'standard_voltage_patterns_clipped_ramp_'
                                                f'delv{delv_max}_flipFalse_rollTrue.csv')
    else:
        voltage_pattern_filepath = pathlib.Path(current_dir, 'delta',
                                                f'standard_voltage_patterns_clipped_ramp_'
                                                f'delv{delv_max}_flipTrue_rollTrue.csv')
    print(f'loaded patterns: {voltage_pattern_filepath.stem}')
    voltage_patterns = pd.read_csv(voltage_pattern_filepath, index_col=0)
    voltage_patterns.columns = voltage_patterns.columns.astype(int)

    return voltage_patterns
