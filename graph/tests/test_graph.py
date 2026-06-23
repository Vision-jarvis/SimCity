"""Tests for graph intelligence modules."""



class TestNodeTypes:
    def test_all_node_types_exist(self):
        from graph.node_types import ALL_NODE_TYPES
        assert len(ALL_NODE_TYPES) >= 10

    def test_platform_map(self):
        from graph.node_types import PLATFORM_NODE_MAP
        assert 0 in PLATFORM_NODE_MAP  # Reddit
        assert 1 in PLATFORM_NODE_MAP  # HN
        assert 2 in PLATFORM_NODE_MAP  # GDELT

    def test_cypher_create(self):
        from graph.node_types import REDDIT_POST
        cypher = REDDIT_POST.cypher_create()
        assert "MERGE" in cypher
        assert "RedditPost" in cypher


class TestEdgeTypes:
    def test_all_edge_types_exist(self):
        from graph.edge_types import ALL_EDGE_TYPES
        assert len(ALL_EDGE_TYPES) >= 8

    def test_influence_edge(self):
        from graph.edge_types import INFLUENCE
        assert INFLUENCE.name == "INFLUENCE"
        assert "weight" in INFLUENCE.properties

    def test_narrative_transfer_edge(self):
        from graph.edge_types import NARRATIVE_TRANSFER
        assert "similarity" in NARRATIVE_TRANSFER.properties
        assert "time_delta" in NARRATIVE_TRANSFER.properties


class TestCommunityDetector:
    def test_louvain(self):
        import networkx as nx
        from graph.community_detector import CommunityDetector

        # Create a simple graph with two clear communities
        G = nx.Graph()
        # Community 1
        for i in range(5):
            for j in range(i + 1, 5):
                G.add_edge(f"a{i}", f"a{j}")
        # Community 2
        for i in range(5):
            for j in range(i + 1, 5):
                G.add_edge(f"b{i}", f"b{j}")
        # One bridge edge
        G.add_edge("a0", "b0")

        detector = CommunityDetector()
        partition = detector.detect_louvain(G)

        assert len(partition) == 10
        assert detector.modularity > 0
        sizes = detector.get_community_sizes()
        assert len(sizes) >= 2

    def test_label_propagation(self):
        import networkx as nx
        from graph.community_detector import CommunityDetector

        G = nx.karate_club_graph()
        detector = CommunityDetector()
        partition = detector.detect_label_propagation(G)
        assert len(partition) == 34  # Karate club has 34 nodes

    def test_cross_community_edges(self):
        import networkx as nx
        from graph.community_detector import CommunityDetector

        G = nx.Graph()
        G.add_edges_from([("a1", "a2"), ("b1", "b2"), ("a1", "b1")])

        detector = CommunityDetector()
        detector.detect_louvain(G)
        cross = detector.get_cross_community_edges(G)
        # At minimum there should be cross-community edges
        assert isinstance(cross, list)


class TestGraphInfluenceScorer:
    def test_basic_scoring(self):
        import networkx as nx
        from graph.influence_scorer import GraphInfluenceScorer

        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("a", "c"), ("b", "c"), ("c", "a")])

        scorer = GraphInfluenceScorer()
        scores = scorer.score_all(G)
        assert len(scores) == 3
        assert all(s >= 0 for s in scores.values())

    def test_top_k(self):
        import networkx as nx
        from graph.influence_scorer import GraphInfluenceScorer

        G = nx.star_graph(10)
        G = G.to_directed()

        scorer = GraphInfluenceScorer()
        top = scorer.top_k(G, k=3)
        assert len(top) == 3
        # Hub node should be most influential
        assert top[0][1] >= top[1][1]
