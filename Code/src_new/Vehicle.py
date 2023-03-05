from Path import Path


# TODO: Add info about rewards (if using spaced rewards).
class Vehicle(object):
    """
    A Vehicle corresponds to a single decision making unit.

    In our formulation a learning agent corresponds to a single
    vehicle. It learns a value function based on the rewards it gets
    from the environment. It generates prefences for different actions
    using this value funciton and submits it to the CentralAgent for
    arbitration.
    """

    def __init__(self, veh_id: int, initial_location: int):
        # Initialising the state of the Vehicle
        self.id = veh_id
        #Check if initial location is of type VehicleLocation
        if isinstance(initial_location, VehicleLocation):
            self.position = initial_location
        else:
            self.position = VehicleLocation(initial_location)
        self.path: Path = Path()
        self.earning: float = 0.0
        self.discounted_value: float = 100.0 #Initialize a discounted value for smoothing
        self.discount: float = 0.9999 #Discount for past earnings

# TODO(?): Convert to NamedTuple
class VehicleLocation(object):
    """Define the current position of a Vehicle."""

    def __init__(self, next_location: int, time_to_next_location: float = 0):
        self.next_location = next_location
        self.time_to_next_location = time_to_next_location
