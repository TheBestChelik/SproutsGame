import numpy as np

from typing import List, Tuple, Set, Dict
from SproutGame.primitives import Vertex, Vector, Spot
from SproutGame.resources.constants import MAXIMUM_FORCE_MAGNITUDE


def apply_force_to_vertex(vertex: Vertex, force: Vector) -> None:

    if not force.magnitude:
        return

    normalized_force = (force / force.magnitude) * \
        min(force.magnitude, MAXIMUM_FORCE_MAGNITUDE)
    vertex.translate(normalized_force)


# calculate the force with which vertex2 is attracted to vertex1
def calculate_vertex_attraction_force(vertex1: Vertex, vertex2: Vertex,
                                      optimum_length: float,
                                      vertex_attraction_coefficient: float) -> Vector:

    resultant = Vector(vertex1.x, vertex1.y) - Vector(vertex2.x, vertex2.y)
    if resultant.magnitude <= optimum_length:
        return Vector(0.0, 0.0)

    return resultant * (resultant.magnitude / vertex_attraction_coefficient)


# calculate the force with which vertex2 is repulsed from vertex1
def calculate_vertex_repulsion_force(vertex1: Vertex, vertex2: Vertex, optimum_length: float, vertex_repulsion_coefficient: float) -> Vector:

    resultant = Vector(vertex2.x, vertex2.y) - Vector(vertex1.x, vertex1.y)

    if resultant.magnitude == 0:
        print("Distance is zeo!", vertex1, vertex2)
        return Vector(0, 0)
        raise RuntimeError("Distance between two poins is zero")

    if resultant.magnitude >= optimum_length:
        return Vector(0, 0)

    return resultant * (vertex_repulsion_coefficient / resultant.magnitude) ** 2


def calculate_edge_repulsion_force(vertex: Vertex, edge_start: Vertex, edge_end: Vertex, optimum_length: float, edge_repulsion_coefficient: float) -> Vector:

    projection = project_vertex_onto_edge(
        vertex, edge_start, edge_end)

    if projection is None:
        return Vector(0, 0)

    resultant = Vector(vertex.x, vertex.y) - projection

    if resultant.magnitude == 0:
        raise RuntimeError("Distance between vertex and edge is zero")

    if resultant.magnitude >= optimum_length:
        return Vector(0, 0)

    return (resultant / resultant.magnitude) * \
        ((edge_repulsion_coefficient - resultant.magnitude) ** 2)


# https://stackoverflow.com/questions/10301001/perpendicular-on-a-line-segment-from-a-given-point
def project_vertex_onto_edge(vertex: Vertex, edge_start: Vertex, edge_end: Vertex) -> Vector:
    # start to end vector
    resultant1 = Vector(edge_end.x, edge_end.y) - \
        Vector(edge_start.x, edge_start.y)

    # start to vertex vector
    resultant2 = Vector(vertex.x, vertex.y) - \
        Vector(edge_start.x, edge_start.y)

    # Calculate the dot product
    dot_product = resultant2.dot(resultant1)

    # Calculate the length of the line segment
    line_length_squared = resultant1.dot(resultant1)

    # Calculate the projection scalar
    projection_scalar = dot_product / line_length_squared

    # Check if the projection is outside the line segment
    if projection_scalar < 0 or projection_scalar > 1:
        return None

    # Calculate the projection vector
    projection = Vector(edge_start.x, edge_start.y) + \
        resultant1 * projection_scalar

    return projection


def calculate_boundary_repulsion_force(v, edge, l_opt,  bord_repuls_const) -> Vector:

    v_edge = project_vertex_onto_edge(v, edge[0], edge[1])

    if v_edge is None:
        # print("out of boundaries", v)
        direction = Vector(0.5, 0.5) - Vector(v.x, v.y)
        return (direction / direction.magnitude) * bord_repuls_const ** 2
        raise RuntimeError(
            f"Vertex {v} is outside of the boundary {edge[0]}; {edge[1]}")

    direction = Vector(v.x, v.y) - v_edge
    dist = direction.magnitude
    if dist == 0:
        direction = Vector(0.5, 0.5) - Vector(v.x, v.y)
        return (direction / direction.magnitude) * bord_repuls_const ** 2
    if dist >= l_opt:
        return Vector(0, 0)
    else:
        # return (Vector(0.5, 0.5) - Vector(v.x, v.y)) * bord_repuls_const
        return (direction / dist) * ((bord_repuls_const - dist) ** 2)


def calculate_resultant_forces(vertices: Set[Vertex], edges: Set[Tuple[Vertex, Vertex]], boundaries: Set[Tuple[Vertex, Vertex]], optimum_length: float) -> Dict[Vertex, Vector]:

    vertex_attraction_coefficient = optimum_length / 10
    vertex_repulsion_coefficient = optimum_length * 2
    edge_repulsion_coefficient = optimum_length * 8
    vertex_to_vertex_computed = set()
    resultant_forces = {}
    for v in vertices:
        resultant_forces.setdefault(v, Vector(0, 0))
    for vertex1 in vertices:
        # resultant_forces[vertex1] = Vector(0, 0)
        # vertex-to-vertex
        for vertex2 in vertices:
            if vertex1 == vertex2:
                continue
            if (vertex1, vertex2) in vertex_to_vertex_computed or (vertex2, vertex1) in vertex_to_vertex_computed:
                continue

            vertex_to_vertex_computed.add((vertex1, vertex2))

            if (type(vertex1) is not Spot and type(vertex2) is not Spot) and ((vertex1, vertex2) in edges or (vertex2, vertex1) in edges):
                # if ((vertex1, vertex2) in edges or (vertex2, vertex1) in edges):
                # pass
                vertex_attraction_force = calculate_vertex_attraction_force(
                    vertex2, vertex1, optimum_length, vertex_attraction_coefficient)
                resultant_forces[vertex1] += vertex_attraction_force
                resultant_forces[vertex2] += vertex_attraction_force * (-1.0)
            else:
                vertex_repulsion_force = calculate_vertex_repulsion_force(
                    vertex2, vertex1, optimum_length, vertex_repulsion_coefficient)
                resultant_forces[vertex1] += vertex_repulsion_force
                resultant_forces[vertex2] += vertex_repulsion_force * (-1.0)

        # edge-to-vertex
        for (edge_start, edge_end) in edges:
            if vertex1 in (edge_start, edge_end):
                continue

            edge_repulsion_force = calculate_edge_repulsion_force(
                vertex1, edge_start, edge_end,
                optimum_length, edge_repulsion_coefficient)
            resultant_forces[vertex1] += edge_repulsion_force

        # resultant_forces[vertex1] = sum

    return resultant_forces
