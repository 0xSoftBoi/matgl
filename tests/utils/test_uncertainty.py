"""Tests for MCDropoutWrapper: uncertainty-aware inference with MatGL models."""

from __future__ import annotations

import pytest
import torch
from pymatgen.core import Lattice, Structure

import matgl  # noqa: F401
from matgl.models._chgnet import CHGNet
from matgl.models._m3gnet import M3GNet
from matgl.utils.uncertainty import MCDropoutWrapper

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mos_structure():
    return Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


@pytest.fixture(scope="module")
def fe_structure():
    return Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])


# ---------------------------------------------------------------------------
# MCDropoutWrapper — construction
# ---------------------------------------------------------------------------


class TestMCDropoutWrapperConstruction:
    def test_chgnet_with_explicit_dropout(self, mos_structure):
        """CHGNet initialised with final_dropout > 0 should wrap without errors."""
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.1)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        assert len(wrapper._stochastic_modules) > 0

    def test_chgnet_identity_replaced(self, mos_structure):
        """CHGNet with default final_dropout=0 (nn.Identity) should have Identity replaced."""
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.0)
        MCDropoutWrapper(model, dropout_p=0.1)
        # The Identity should have been replaced with a real Dropout.
        assert isinstance(model.final_dropout, torch.nn.Dropout)
        assert model.final_dropout.p == pytest.approx(0.1)

    def test_m3gnet_wraps(self):
        """M3GNet with is_intensive=True uses an MLP final_layer that Dropout can be injected into."""
        model = M3GNet(element_types=("Mo", "S"), is_intensive=True)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        assert len(wrapper._stochastic_modules) > 0

    def test_invalid_dropout_p_zero(self):
        model = CHGNet(element_types=("Mo", "S"))
        with pytest.raises(ValueError, match="dropout_p must be in"):
            MCDropoutWrapper(model, dropout_p=0.0)

    def test_invalid_dropout_p_one(self):
        model = CHGNet(element_types=("Mo", "S"))
        with pytest.raises(ValueError, match="dropout_p must be in"):
            MCDropoutWrapper(model, dropout_p=1.0)

    def test_unknown_readout_attrs_raises(self):
        model = CHGNet(element_types=("Mo", "S"))
        with pytest.raises(ValueError, match="No dropout layers found"):
            MCDropoutWrapper(model, dropout_p=0.1, readout_attrs=("nonexistent_attr",))


# ---------------------------------------------------------------------------
# MCDropoutWrapper — predict_uncertainty
# ---------------------------------------------------------------------------


class TestPredictUncertainty:
    def test_output_shapes_single_structure(self, mos_structure):
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.1)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        mean, std = wrapper.predict_uncertainty(mos_structure, n_passes=5)
        assert mean.shape == torch.Size([])
        assert std.shape == torch.Size([])

    def test_output_shapes_multiple_structures(self, mos_structure, fe_structure):
        model = CHGNet(element_types=("Mo", "S", "Fe"), final_dropout=0.1)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        mean, std = wrapper.predict_uncertainty([mos_structure, fe_structure], n_passes=5)
        assert mean.shape == torch.Size([2])
        assert std.shape == torch.Size([2])

    def test_std_positive_with_dropout(self, mos_structure):
        """With dropout active, predictions across passes should differ (std > 0)."""
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.3)
        wrapper = MCDropoutWrapper(model, dropout_p=0.3)
        _, std = wrapper.predict_uncertainty(mos_structure, n_passes=20)
        assert std.item() > 0.0

    def test_std_zero_deterministic_baseline(self, mos_structure):
        """With dropout p=~0, all passes should agree (std ≈ 0)."""
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.0)
        # Use a tiny p so we can still wrap, but effectively no dropout signal
        wrapper = MCDropoutWrapper(model, dropout_p=1e-6)
        _, std = wrapper.predict_uncertainty(mos_structure, n_passes=10)
        assert std.item() < 1e-3

    def test_model_returns_to_eval_after_predict(self, mos_structure):
        """Model must be fully in eval() mode after predict_uncertainty returns."""
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.1)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        wrapper.predict_uncertainty(mos_structure, n_passes=3)
        for m in model.modules():
            assert not m.training, f"{m} still in training mode after predict_uncertainty"

    def test_m3gnet_uncertainty(self, mos_structure):
        model = M3GNet(element_types=("Mo", "S"), is_intensive=True)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        mean, std = wrapper.predict_uncertainty(mos_structure, n_passes=5)
        assert torch.isfinite(mean)
        assert std >= 0.0

    def test_mean_finite(self, mos_structure):
        model = CHGNet(element_types=("Mo", "S"), final_dropout=0.1)
        wrapper = MCDropoutWrapper(model, dropout_p=0.1)
        mean, _ = wrapper.predict_uncertainty(mos_structure, n_passes=5)
        assert torch.isfinite(mean)
