# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the cosim.py script."""

import pytest
import torch
import numpy as np

from cosim import (
    load_down_input,
    binarise_pct,
    compute_cosine_similarity,
    group_channels_greedy,
    group_channels_kmeans,
    group_channels_random,
    evaluate_grouping,
)


def test_binarise_pct():
    # 2 tokens, 4 channels
    acts = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [-4.0, -3.0, -2.0, -1.0]
    ], dtype=torch.float32)
    
    # 50th percentile is 2.5 for token 0, and 2.5 for token 1
    bin_acts = binarise_pct(acts, 50.0)
    
    expected = torch.tensor([
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0]
    ], dtype=torch.float32)
    
    assert torch.allclose(bin_acts, expected)


def test_load_down_input():
    # Test loading down_input from mock data dict
    data = {
        "layer0/down_input": np.arange(12, dtype=np.float32).reshape(3, 4)
    }
    
    tensor = load_down_input(data, layer_idx=0, device="cpu")
    assert tensor.shape == (3, 4)
    assert torch.allclose(tensor, torch.arange(12, dtype=torch.float32).reshape(3, 4))
    
    # Test KeyError for missing key
    with pytest.raises(KeyError, match="Expected key 'layer1/down_input' not found in NPZ data"):
        load_down_input(data, layer_idx=1, device="cpu")


def test_compute_cosine_similarity():
    # Create simple inputs with known cosine similarity
    # Channel 0: [1, 0]
    # Channel 1: [1, 1] -> cosim = 1 / sqrt(2) ≈ 0.7071
    # Channel 2: [0, 0] -> all zeros (dead channel), should be handled safely
    activations = torch.tensor([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    sim_matrix = compute_cosine_similarity(activations)
    
    assert sim_matrix.shape == (3, 3)
    # Check self-similarity is 1.0 (except dead channels which are safe-guarded to 0 or 1 depending on implementation)
    assert pytest.approx(sim_matrix[0, 0].item(), abs=1e-5) == 1.0
    assert pytest.approx(sim_matrix[1, 1].item(), abs=1e-5) == 1.0
    
    # Check similarity between Channel 0 and Channel 1
    expected_sim_0_1 = 1.0 / np.sqrt(2.0)
    assert pytest.approx(sim_matrix[0, 1].item(), abs=1e-5) == expected_sim_0_1
    assert pytest.approx(sim_matrix[1, 0].item(), abs=1e-5) == expected_sim_0_1
    
    # Check dead channel handling (should be safe from NaN/Inf)
    assert not torch.isnan(sim_matrix).any()
    assert not torch.isinf(sim_matrix).any()


def test_group_channels_greedy():
    # 128 channels, grouping into groups of 64 (2 groups total)
    C = 128
    group_size = 64
    
    # Construct a similarity matrix with 2 distinct clusters
    # Cluster 1: 0..63 are highly similar
    # Cluster 2: 64..127 are highly similar
    sim_matrix = torch.eye(C)
    sim_matrix[0:64, 0:64] += 0.8
    sim_matrix[64:128, 64:128] += 0.8
    # Clip to max 1.0
    sim_matrix = torch.clamp(sim_matrix, max=1.0)
    
    groups = group_channels_greedy(sim_matrix, group_size=group_size)
    
    assert len(groups) == 2
    for g in groups:
        assert len(g) == group_size
        
    # Flat list of all assigned indices
    flat_indices = [idx for g in groups for idx in g]
    assert len(flat_indices) == C
    assert len(set(flat_indices)) == C  # All channels assigned exactly once
    
    # Check that it correctly identified the cluster structures
    # (i.e. group 0 has either all elements from cluster 1 or all from cluster 2)
    g0 = set(groups[0])
    g1 = set(groups[1])
    
    cluster1 = set(range(64))
    cluster2 = set(range(64, 128))
    
    assert g0 == cluster1 or g0 == cluster2
    assert g1 == cluster1 or g1 == cluster2


def test_group_channels_greedy_invalid_size():
    # 100 channels cannot be divided into groups of 64
    sim_matrix = torch.eye(100)
    with pytest.raises(ValueError, match="Number of channels 100 is not divisible by group_size 64"):
        group_channels_greedy(sim_matrix, group_size=64)


def test_group_channels_kmeans():
    C = 128
    group_size = 64
    
    # 2 tokens, 128 channels
    normalized_channels = torch.randn(2, C)
    # L2 normalize
    norms = torch.norm(normalized_channels, dim=0, keepdim=True)
    normalized_channels = normalized_channels / torch.where(norms == 0.0, torch.ones_like(norms), norms)
    
    groups = group_channels_kmeans(normalized_channels, group_size=group_size, num_iters=2, seed=42)
    
    assert len(groups) == 2
    for g in groups:
        assert len(g) == group_size
        
    flat_indices = [idx for g in groups for idx in g]
    assert len(flat_indices) == C
    assert len(set(flat_indices)) == C


def test_group_channels_kmeans_invalid_size():
    normalized_channels = torch.randn(2, 100)
    with pytest.raises(ValueError, match="Number of channels 100 is not divisible by group_size 64"):
        group_channels_kmeans(normalized_channels, group_size=64)


def test_group_channels_random():
    C = 128
    group_size = 64
    groups = group_channels_random(C, group_size=group_size)
    
    assert len(groups) == 2
    for g in groups:
        assert len(g) == group_size
    flat_indices = [idx for g in groups for idx in g]
    assert len(flat_indices) == C
    assert len(set(flat_indices)) == C


def test_evaluate_grouping():
    # Simple test for evaluating groups
    sim_matrix = torch.eye(128)
    # Group 1: 0..63 (all identity, pairwise sims are 0.0)
    # Group 2: 64..127
    groups = [list(range(64)), list(range(64, 128))]
    
    stats = evaluate_grouping(sim_matrix, groups)
    assert stats["mean"] == 0.0
    assert stats["min"] == 0.0
    assert stats["max"] == 0.0
    assert stats["std"] == 0.0
