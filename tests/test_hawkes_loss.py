import torch

from models.hawkes import NeuralHawkesLoss, StreamingHawkesLoss
from models.virality_head import ViralityHead


def test_neural_hawkes_loss_is_finite_and_differentiable():
    loss_fn = NeuralHawkesLoss(num_platforms=3)
    t = torch.tensor([0.0, 120.0, 300.0, 900.0])
    platform = torch.tensor([0, 1, 1, 2])
    mu = torch.full((4, 3), 0.4, requires_grad=True)
    alpha = torch.full((4, 3, 3), 0.2, requires_grad=True)
    gamma = torch.full((4, 3, 3), 1.5, requires_grad=True)

    loss = loss_fn(t, platform, mu, alpha, gamma)

    assert torch.isfinite(loss)
    loss.backward()
    assert mu.grad is not None
    assert alpha.grad is not None
    assert gamma.grad is not None


def test_neural_hawkes_loss_sorts_events_chronologically():
    loss_fn = NeuralHawkesLoss(num_platforms=3)
    t = torch.tensor([0.0, 120.0, 300.0, 900.0])
    platform = torch.tensor([0, 1, 1, 2])
    mu = torch.full((4, 3), 0.4)
    alpha = torch.full((4, 3, 3), 0.2)
    gamma = torch.full((4, 3, 3), 1.5)
    permutation = torch.tensor([2, 0, 3, 1])

    ordered_loss = loss_fn(t, platform, mu, alpha, gamma)
    permuted_loss = loss_fn(
        t[permutation],
        platform[permutation],
        mu[permutation],
        alpha[permutation],
        gamma[permutation],
    )

    assert torch.allclose(ordered_loss, permuted_loss, atol=1e-6)


def test_virality_head_emits_positive_hawkes_parameters_with_gdelt_signal():
    batch_size = 5
    embedding_dim = 16
    num_platforms = 3
    head = ViralityHead(embedding_dim=embedding_dim, num_platforms=num_platforms)
    h_v = torch.randn(batch_size, embedding_dim)
    h_c = torch.randn(batch_size, embedding_dim)
    h_p = torch.randn(num_platforms, embedding_dim)
    gdelt_volume = torch.linspace(0.0, 50.0, batch_size)

    beta, mu, alpha, gamma, virality = head(h_v, h_c, h_p, gdelt_volume)

    assert beta.shape == (batch_size,)
    assert mu.shape == (batch_size, num_platforms)
    assert alpha.shape == (batch_size, num_platforms, num_platforms)
    assert gamma.shape == (batch_size, num_platforms, num_platforms)
    assert virality.shape == (batch_size,)
    assert (mu > 0).all()
    assert (alpha >= 0).all()
    assert (gamma > 0).all()


def test_streaming_hawkes_loss_matches_split_chronological_stream():
    t = torch.tensor([0.0, 120.0, 300.0, 900.0, 1200.0])
    platform = torch.tensor([0, 1, 1, 2, 0])
    mu = torch.full((5, 3), 0.4)
    alpha = torch.full((5, 3, 3), 0.2)
    gamma = torch.full((5, 3, 3), 1.5)

    one_shot = StreamingHawkesLoss(num_platforms=3)
    split = StreamingHawkesLoss(num_platforms=3)

    one_shot_loss = one_shot(t, platform, mu, alpha, gamma)
    split_loss_a = split(t[:2], platform[:2], mu[:2], alpha[:2], gamma[:2])
    split_loss_b = split(t[2:], platform[2:], mu[2:], alpha[2:], gamma[2:])
    weighted_split = (split_loss_a * 2 + split_loss_b * 3) / 5

    assert torch.allclose(one_shot_loss, weighted_split, atol=1e-6)
    assert torch.allclose(one_shot.excitation_state, split.excitation_state, atol=1e-6)
    assert torch.allclose(one_shot.last_time, split.last_time, atol=1e-6)


def test_streaming_hawkes_reset_clears_history():
    loss_fn = StreamingHawkesLoss(num_platforms=3)
    t = torch.tensor([0.0, 120.0, 300.0])
    platform = torch.tensor([0, 1, 2])
    mu = torch.full((3, 3), 0.4)
    alpha = torch.full((3, 3, 3), 0.2)
    gamma = torch.full((3, 3, 3), 1.5)

    _ = loss_fn(t, platform, mu, alpha, gamma)
    assert loss_fn.excitation_state.sum() > 0

    loss_fn.reset_state()

    assert loss_fn.excitation_state.sum() == 0
    assert torch.isnan(loss_fn.last_time)
