import unittest
import numpy as np

# Import the Node class and divide_line_recursive function
from Demo1 import Node, segment_line, merge_long_edge, project_node_onto_line_segment

class TestDivideLineRecursive(unittest.TestCase):

    def test_divide_line_recursive_base_case(self):
        l_opt = 1.0
        A = Node(0, 0)
        B = Node(1, 0)
        nodes = [A, B]
        result = segment_line(A, B, nodes, l_opt)
        exp = [(Node(0,0), Node(1,0))]
        self.assertEqual(nodes, [A,B])
        self.assertEqual(len(result), len(exp))
        self.assertEqual(result, exp)

        

    def test_divide_line(self):
        l_opt = 0.15
        A = Node(0, 0)
        B = Node(1, 0)
        nodes = [A,B]
        new_nodes = [A,B, Node(0.125,0), Node(0.25,0), Node(0.375,0), Node(0.5,0), Node(0.625,0), Node(0.75,0), Node(0.875,0)]
        result = segment_line(A, B, nodes, l_opt)

        exp = [(Node(0,0), Node(0.125,0)),
                (Node(0.125,0),Node(0.25,0)),
                (Node(0.25,0),Node(0.375,0)),
                (Node(0.375,0),Node(0.5,0)),
                (Node(0.5,0),Node(0.625,0)),
                (Node(0.625,0),Node(0.75,0)),
                (Node(0.75,0),Node(0.875,0)),
                (Node(0.875,0),Node(1,0))]

        self.assertEqual(len(result), len(exp))
        self.assertEqual(result, exp)
        self.assertEqual(len(nodes), len(new_nodes))
    

    def test_no_merge(self):
        l_opt = 0.1
        A = Node(0, 0)
        B = Node(0.1, 0)
        C = Node(0.2,0)
        D = Node(0.3,0)
        nodes = [A,B,C,D]
        long_edge = [(A,B), (B,C), (C,D)]
        new_long_edge = merge_long_edge(long_edge, nodes,  l_opt)
        self.assertEqual(len(long_edge), len(new_long_edge))
        self.assertEqual(long_edge, new_long_edge)        

    def test_simple_merge(self):
        l_opt = 0.1
        A = Node(0, 0)
        B = Node(0.1, 0)
        C = Node(0.15,0)
        D = Node(0.2,0)
        nodes = [A,B,C,D]
        long_edge = [(A,B), (B,C), (C,D)]
        new_long_edge = merge_long_edge(long_edge, nodes,  l_opt)
        expected = [(A,B), (B,D)]
        exp_nodes = [A,B,D]
        self.assertEqual(nodes, exp_nodes)
        self.assertEqual(new_long_edge, expected)

    def test_complicated_merge(self):
        l_opt = 0.1
        A = Node(0, 0)
        B = Node(0.1, 0)
        C = Node(0.15,0)
        D = Node(0.2,0)
        E = Node(0.21,0)
        nodes = [A,B,C,D,E]
        long_edge = [(A,B), (B,C), (C,D), (D,E)]
        new_long_edge = merge_long_edge(long_edge, nodes,  l_opt)
        expected = [(A,B), (B,E)]
        exp_nodes = [A,B,E]
        self.assertEqual(nodes, exp_nodes)
        self.assertEqual(new_long_edge, expected)


    def test_horizontal_line_segment(self):
        start_node = Node(1, 1)
        end_node = Node(4, 1)
        node_v = Node(2, 3)
        res = project_node_onto_line_segment(start_node, end_node, node_v)
        exp = Node(2, 1)
        self.assertAlmostEqual(res.x, exp.x)
        self.assertAlmostEqual(res.y, exp.y)

    def test_vertical_line_segment(self):
        start_node = Node(2, 2)
        end_node = Node(2, 5)
        node_v = Node(3, 4)
        res = project_node_onto_line_segment(start_node, end_node, node_v)
        exp = Node(2, 4)
        self.assertAlmostEqual(res.x, exp.x)
        self.assertAlmostEqual(res.y, exp.y)

    def test_diagonal_line_segment(self):
        start_node = Node(1, 1)
        end_node = Node(4, 4)
        node_v = Node(3, 2)
        res = project_node_onto_line_segment(start_node, end_node, node_v)
        exp = Node(2.5, 2.5)
        self.assertAlmostEqual(res.x, exp.x)
        self.assertAlmostEqual(res.y, exp.y)

    def test_start_node_equals_node_v(self):
        start_node = Node(1, 1)
        end_node = Node(4, 4)
        node_v = Node(1, 1)
        res = project_node_onto_line_segment(start_node, end_node, node_v)
        exp = Node(1, 1)
        self.assertAlmostEqual(res.x, exp.x)
        self.assertAlmostEqual(res.y, exp.y)
    def test_end_node_equals_node_v(self):
        start_node = Node(1, 1)
        end_node = Node(4, 4)
        node_v = Node(4, 4)
        res = project_node_onto_line_segment(start_node, end_node, node_v)
        exp = Node(4, 4)
        self.assertAlmostEqual(res.x, exp.x)
        self.assertAlmostEqual(res.y, exp.y)

    def test_projection_not_on_line(self):
        start_node = Node(1, 1)
        end_node = Node(3,1)  # Vertical line segment
        node_v = Node(4, 2)  # Node not aligned with the line
        projection = project_node_onto_line_segment(start_node, end_node, node_v)

        self.assertIsNone(projection)
        


if __name__ == '__main__':
    unittest.main()
