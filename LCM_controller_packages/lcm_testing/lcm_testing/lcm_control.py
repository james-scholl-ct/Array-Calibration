from lcm_board import DeloreanBoard
from lcm_voltage_patterns.delta.load_patterns import load_standard_voltage_patterns


class DB:
    def __init__(self, angle):
        self.pattern = self.load(angle)


    def load(self, angle):
        # instantiate hardware
        self.db = DeloreanBoard('dl-32')

        # load standard voltage patterns
        pattern_df = load_standard_voltage_patterns(phi_i=angle, delv_max=4)
        return pattern_df
    
    
    def steer(self, order, channel='rx'):
        # steer LCM to order 300
        self.db.write_pattern_v(v_pattern=self.pattern[order],
                                tx_or_rx=channel)
    
    
    def standby(self):
        # put LCM in standby mode (e.g. when taking a short break)
        self.db.lcm_standby_mode()
    
    
    def shutdown(self):
        # shut down hardware (e.g. when taking a long break)
        self.db.shutdown()