"""Auxiliary methods used in with the SysAdmin problem domain"""
from typing import Dict, Set


def get_connections(parser) -> Dict[str, Set[str]]:
    connections = {}
    for val in parser.state:
        if val[0] == 'conn':
            conn1 = val[1]
            conn2 = val[2]

            if conn1 in connections:
                connections[conn1].append(conn2)
            else:
                connections[conn1] = [conn2]
            
            if conn2 in connections:
                connections[conn2].append(conn1)
            else:
                connections[conn2] = [conn1]
    for key, val in connections.items():
        connections[key] = set(val) # Removes repetitions fron listed connections
    return connections