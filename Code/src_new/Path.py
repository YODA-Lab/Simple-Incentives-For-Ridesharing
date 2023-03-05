from Request import Request
from typing import List, Tuple


# TODO: Make sure all the 'delay' nomenclature is standardised
class Path(object):
    """Define the things relevant to a taxi's motion and commitments."""

    NOT_APPLICABLE: int = -1

    def __init__(self):
        self.requests: List[RequestInfo] = []
        self.request_order: List[PathNode] = []
        self.total_delay = 0.0
        self.response_delay = 0.0
        self.current_capacity = 0

    def __repr__(self):
        path_str = ""
        for idx, node in enumerate(self.request_order):
            location, _ = self.get_info(node)
            path_str += str(location)
            if idx < len(self.request_order) - 1:
                path_str += " -> "
        return path_str

    def get_next_location(self) -> int:
        if (self.is_empty()):
            return self.NOT_APPLICABLE
        else:
            next_node = self.request_order[0]
            relevant_request = self.requests[next_node.relevant_request_id]

            if (next_node.is_dropoff):
                return relevant_request.request.dropoff
            else:
                return relevant_request.request.pickup

    def visit_next_location(self, current_time):
        if not self.is_empty():
            next_node = self.request_order.pop(0)
            relevant_request_id = next_node.relevant_request_id

            # If dropoff node has been visited, remove the associated request
            if next_node.is_dropoff:
                next_request = self.requests.pop(relevant_request_id)
                self.total_delay -= next_request.request.dropoff_deadline - next_node.expected_visit_time
                self.response_delay -= next_request.request.pickup_deadline - next_node.expected_visit_time

                # Update all the nodes in the request_order
                for node in self.request_order:
                    if (node.relevant_request_id > relevant_request_id):
                        node.relevant_request_id -= 1

                # Make sure that the dropoff deadline was met
                if not next_request.is_dummy:
                    assert next_request.request.dropoff_deadline >= current_time
            else:
                self.requests[relevant_request_id].has_been_picked_up = True

                # Make sure that the pickup deadline was met
                if not self.requests[relevant_request_id].is_dummy:
                    assert self.requests[relevant_request_id].request.pickup_deadline >= current_time

            self.current_capacity = next_node.current_capacity

    def is_empty(self) -> bool:
        return False if self.request_order else True

    def is_complete(self, request_order: List['PathNode'] = None) -> bool:
        # Count number of locations to visit
        num_locations_to_visit = 0
        for request in self.requests:
            if request.has_been_picked_up:
                num_locations_to_visit += 1
            else:
                num_locations_to_visit += 2

        # Check if it matches the number of locations you plan to visit
        if (request_order is None):
            request_order = self.request_order

        if (num_locations_to_visit == len(request_order)):
            return True
        else:
            return False

    def get_info(self, node: 'PathNode') -> Tuple[int, float]:
        relevant_request = self.requests[node.relevant_request_id]
        if node.is_dropoff:
            prev_location = relevant_request.request.dropoff
            deadline = relevant_request.request.dropoff_deadline
        else:
            prev_location = relevant_request.request.pickup
            deadline = relevant_request.request.pickup_deadline

        return prev_location, deadline


class RequestInfo(object):
    """docstring for RequestInfo"""

    def __init__(
        self,
        request: Request,
        has_been_picked_up: bool,
        is_dummy: bool
    ) -> None:

        self.request = request
        self.has_been_picked_up = has_been_picked_up
        self.is_dummy = is_dummy

    def __str__(self):
        return "({}, {})".format(self.request, self.has_been_picked_up)

    def __repr__(self):
        return str(self)


class PathNode(object):
    """Define the things relevant to the order in which requests are picked up and dropped off."""

    def __init__(self,
                 is_dropoff: bool,
                 relevant_request_id: int,
                 current_capacity: int = Path.NOT_APPLICABLE,
                 expected_visit_time: float = Path.NOT_APPLICABLE
                 ):

        self.is_dropoff = is_dropoff
        self.relevant_request_id = relevant_request_id
        self.current_capacity = current_capacity
        self.expected_visit_time = expected_visit_time

    def __str__(self):
        return "({}, {})".format(self.relevant_request_id, self.is_dropoff)

    def __repr__(self):
        return str(self)
