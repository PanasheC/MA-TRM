"""Composite MA-TRM objective compatible with the original pretraining loop."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.losses import IGNORE_LABEL_ID, stablemax_cross_entropy


class MATRMLossHead(nn.Module):
    r"""Optimize task accuracy, verification, specialization, and efficiency.

    The implemented objective is

    .. math::
        L = L_{ans} + \lambda_h L_{halt} + \lambda_v L_{verify}
            + \lambda_i L_{improve} + \lambda_c L_{cons}
            + \lambda_d L_{div} + \lambda_b L_{balance}
            + \lambda_e L_{compute} + \lambda_k L_{distill}.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_type: str = "stablemax_cross_entropy",
        halt_weight: float = 0.5,
        verifier_weight: float = 0.20,
        improvement_weight: float = 0.10,
        consistency_weight: float = 0.05,
        diversity_weight: float = 0.01,
        routing_balance_weight: float = 0.01,
        compute_weight: float = 0.002,
        distillation_weight: float = 0.10,
        improvement_margin: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        if loss_type != "stablemax_cross_entropy":
            raise ValueError(
                "MA-TRM currently uses stablemax_cross_entropy for direct TRM comparability."
            )
        self.loss_fn = stablemax_cross_entropy
        self.halt_weight = float(halt_weight)
        self.verifier_weight = float(verifier_weight)
        self.improvement_weight = float(improvement_weight)
        self.consistency_weight = float(consistency_weight)
        self.diversity_weight = float(diversity_weight)
        self.routing_balance_weight = float(routing_balance_weight)
        self.compute_weight = float(compute_weight)
        self.distillation_weight = float(distillation_weight)
        self.improvement_margin = float(improvement_margin)

    def initial_carry(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.initial_carry(*args, **kwargs)

    @staticmethod
    def _normalized_sample_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        divisor = valid_mask.sum(dim=-1).clamp_min(1)
        cell_loss = stablemax_cross_entropy(
            logits,
            labels,
            ignore_index=IGNORE_LABEL_ID,
            valid_mask=valid_mask,
        )
        return cell_loss.sum(dim=-1) / divisor

    @staticmethod
    def _consistency_loss(
        agent_logits: torch.Tensor,
        shared_logits: torch.Tensor,
        active_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        agent_log_probs = F.log_softmax(agent_logits.to(torch.float32), dim=-1)
        shared_probs = F.softmax(shared_logits.to(torch.float32), dim=-1).unsqueeze(1)
        kl = F.kl_div(agent_log_probs, shared_probs.expand_as(agent_log_probs), reduction="none")
        kl = kl.sum(dim=-1)
        weights = active_mask.to(kl.dtype).unsqueeze(-1) * valid_mask.to(kl.dtype).unsqueeze(1)
        per_sample = (kl * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1.0)
        return per_sample.sum()

    @staticmethod
    def _diversity_loss(
        pooled_states: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_agents, _ = pooled_states.shape
        if num_agents < 2:
            return pooled_states.new_zeros(())
        normalized = F.normalize(pooled_states.to(torch.float32), dim=-1)
        similarities = torch.einsum("bmd,bnd->bmn", normalized, normalized)
        eye = torch.eye(num_agents, device=pooled_states.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = (
            active_mask.to(torch.bool).unsqueeze(2)
            & active_mask.to(torch.bool).unsqueeze(1)
            & ~eye
        )
        squared = similarities.square()
        per_sample = (squared * pair_mask.to(squared.dtype)).sum(dim=(1, 2)) / pair_mask.sum(
            dim=(1, 2)
        ).clamp_min(1)
        return per_sample.sum()

    @staticmethod
    def _routing_balance_loss(router_probabilities: torch.Tensor) -> torch.Tensor:
        num_agents = router_probabilities.shape[-1]
        importance = router_probabilities.to(torch.float32).mean(dim=0)
        target = torch.full_like(importance, 1.0 / num_agents)
        return router_probabilities.shape[0] * (importance - target).square().sum()

    @staticmethod
    def _compute_loss(
        router_probabilities: torch.Tensor,
        cell_active_fraction: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = router_probabilities.to(torch.float32).clamp_min(1e-9)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        max_entropy = torch.log(
            torch.tensor(float(probabilities.shape[-1]), device=probabilities.device)
        ).clamp_min(1e-6)
        normalized_entropy = entropy / max_entropy
        return (normalized_entropy + cell_active_fraction.to(torch.float32)).sum()

    @staticmethod
    def _distillation_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        teacher_probabilities = F.softmax(teacher_logits.detach().to(torch.float32), dim=-1)
        student_log_probabilities = F.log_softmax(student_logits.to(torch.float32), dim=-1)
        kl = F.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="none",
        ).sum(dim=-1)
        valid = valid_mask.to(kl.dtype)
        return ((kl * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)).sum()

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs: Any,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        valid_mask = labels != IGNORE_LABEL_ID
        valid_count = valid_mask.sum(dim=-1)

        current_sample_loss = self._normalized_sample_loss(
            outputs["logits"], labels, valid_mask
        )
        answer_loss = current_sample_loss.sum()

        with torch.no_grad():
            predictions = outputs["logits"].argmax(dim=-1)
            outputs["preds"] = predictions
            cell_correct = valid_mask & (predictions == labels)
            sequence_correct = cell_correct.sum(dim=-1) == valid_count
            valid_metrics = new_carry.halted & (valid_count > 0)
            divisor = valid_count.clamp_min(1).to(torch.float32)

        halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            sequence_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )

        verifier_targets = cell_correct.to(outputs["cell_verify_logits"].dtype)
        verifier_cell_loss = F.binary_cross_entropy_with_logits(
            outputs["cell_verify_logits"],
            verifier_targets,
            reduction="none",
        )
        verifier_loss = (
            (verifier_cell_loss * valid_mask.to(verifier_cell_loss.dtype)).sum(dim=-1)
            / valid_count.clamp_min(1)
        ).sum()

        prior_sample_loss = self._normalized_sample_loss(
            outputs["prior_logits"].detach(), labels, valid_mask
        )
        prior_valid = outputs["prior_valid"].to(torch.bool)
        improvement_loss = (
            F.relu(current_sample_loss - prior_sample_loss + self.improvement_margin)
            * prior_valid.to(current_sample_loss.dtype)
        ).sum()

        consistency_loss = self._consistency_loss(
            agent_logits=outputs["agent_logits"],
            shared_logits=outputs["logits"],
            active_mask=outputs["router_mask"],
            valid_mask=valid_mask,
        )
        diversity_loss = self._diversity_loss(
            pooled_states=outputs["role_state_pooled"],
            active_mask=outputs["router_mask"],
        )
        routing_balance_loss = self._routing_balance_loss(
            outputs["router_probabilities"]
        )
        compute_loss = self._compute_loss(
            outputs["router_probabilities"], outputs["cell_active_fraction"]
        )

        distillation_loss = answer_loss.new_zeros(())
        if "student_logits" in outputs and "teacher_logits" in outputs:
            distillation_loss = self._distillation_loss(
                student_logits=outputs["student_logits"],
                teacher_logits=outputs["teacher_logits"],
                valid_mask=valid_mask,
            )

        continue_loss = answer_loss.new_zeros(())
        if "target_q_continue" in outputs:
            continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )

        total_loss = (
            answer_loss
            + self.halt_weight * (halt_loss + continue_loss)
            + self.verifier_weight * verifier_loss
            + self.improvement_weight * improvement_loss
            + self.consistency_weight * consistency_loss
            + self.diversity_weight * diversity_loss
            + self.routing_balance_weight * routing_balance_loss
            + self.compute_weight * compute_loss
            + self.distillation_weight * distillation_loss
        )

        with torch.no_grad():
            metrics: Dict[str, torch.Tensor] = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    cell_correct.to(torch.float32).sum(dim=-1) / divisor,
                    torch.zeros_like(divisor),
                ).sum(),
                "exact_accuracy": (valid_metrics & sequence_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics
                    & ((outputs["q_halt_logits"] >= 0) == sequence_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
                "mean_disagreement": torch.where(
                    valid_metrics,
                    outputs["disagreement"].to(torch.float32),
                    torch.zeros_like(outputs["disagreement"], dtype=torch.float32),
                ).sum(),
                "answer_change": torch.where(
                    valid_metrics,
                    outputs["answer_change"].to(torch.float32),
                    torch.zeros_like(outputs["answer_change"], dtype=torch.float32),
                ).sum(),
                "active_agents": torch.where(
                    valid_metrics,
                    outputs["active_agents"].to(torch.float32),
                    torch.zeros_like(outputs["active_agents"], dtype=torch.float32),
                ).sum(),
                "cell_active_fraction": torch.where(
                    valid_metrics,
                    outputs["cell_active_fraction"].to(torch.float32),
                    torch.zeros_like(outputs["cell_active_fraction"], dtype=torch.float32),
                ).sum(),
                "lm_loss": answer_loss.detach(),
                "q_halt_loss": halt_loss.detach(),
                "verifier_loss": verifier_loss.detach(),
                "improvement_loss": improvement_loss.detach(),
                "consistency_loss": consistency_loss.detach(),
                "diversity_loss": diversity_loss.detach(),
                "routing_balance_loss": routing_balance_loss.detach(),
                "compute_loss": compute_loss.detach(),
                "distillation_loss": distillation_loss.detach(),
                "total_loss": total_loss.detach(),
            }
            if "target_q_continue" in outputs:
                metrics["q_continue_loss"] = continue_loss.detach()

        detached_outputs = {
            key: outputs[key].detach() for key in return_keys if key in outputs
        }
        return (
            new_carry,
            total_loss,
            metrics,
            detached_outputs,
            new_carry.halted.all(),
        )
