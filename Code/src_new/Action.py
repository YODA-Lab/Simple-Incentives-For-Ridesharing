from Request import Request
from Path import Path
from typing import Iterable, Optional


class Action(object):
    """
    An Action corresponds to a Vehicle accepting a given set
    of Requests.
    """

    def __init__(self, requests: Iterable[Request], veh_id) -> None:
        self.requests = frozenset(requests)
        self.new_path: Optional[Path] = None  # TODO: Get rid of this None condition
        self.veh_id = veh_id

    def __eq__(self, other):
        return (self.requests == other.requests)

    def __hash__(self):
        return hash(self.requests)
