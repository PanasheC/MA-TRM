"""MA-TRM-Lite, a parameter-controlled multi-agent extension of TRM.

The implementation preserves the original TRM training interface so that the
same datasets, evaluators, optimizer, and benchmark scripts can compare TRM and
MA-TRM directly. The default configuration contains one shared two-layer
backbone, four role-conditioned low-rank adapter banks, low-rank RecursiveLinks,
a sparse router, a shared latent workspace, cell-level recursive attention, and
a disagreement-aware adaptive-computation wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator

from models.agents import DEFAULT_AGENT_ROLES, RoleAdapterBank, validate_role_names
from models.collaboration import build_topology
from models.common import trunc_normal_init_
from models.halting import (
    answer_change_fraction,
    mean_predictive_uncertainty,
    multi_agent_js_divergence,
)
from models.layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)
from models.links import LowRankRecursiveLink
from models.routing import RoutingDecision, SparseDynamicRouter
from models.shared_workspace import CellRecursiveAttention, SharedLatentWorkspace
from models.sparse_embedding import CastedSparseEmbedding


IGNORE_LABEL_ID = -100


@dataclass
class MATRMLiteInnerCarry:
    """Private role states, shared workspace, and prior answer distribution."""

    workspace: torch.Tensor
    agent_states: torch.Tensor
    previous_logits: torch.Tensor
    previous_valid: torch.Tensor


@dataclass
class MATRMLiteCarry:
    inner_carry: MATRMLiteInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class MATRMLiteConfig(BaseModel):
    """Validated MA-TRM-Lite architecture and ACT configuration."""

    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # Recursive schedule. H_cycles are collaboration rounds. L_cycles are
    # repeated local backbone applications for each active role.
    H_cycles: int = 3
    L_cycles: int = 2
    H_layers: int = 0
    L_layers: int = 2
    gradient_rounds: int = 1

    hidden_size: int = 512
    expansion: float = 4.0
    num_heads: int = 8
    pos_encodings: str = "rope"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    halt_max_steps: int = 16
    halt_min_steps: int = 2
    halt_exploration_prob: float = 0.1
    halt_threshold: float = 0.0
    disagreement_halt_threshold: float = 0.025
    stability_halt_threshold: float = 0.0
    adaptive_eval: bool = True
    eval_halt_granularity: str = "batch"
    no_ACT_continue: bool = True

    forward_dtype: str = "bfloat16"
    mlp_t: bool = False
    puzzle_emb_len: int = 16

    num_agents: int = 4
    role_names: List[str] = Field(default_factory=lambda: list(DEFAULT_AGENT_ROLES))
    verifier_index: int = 3
    critic_index: int = 2
    student_index: int = 0
    topology: str = "sequential"

    adapter_rank: int = 8
    link_rank: int = 8
    router_hidden_size: int = 64
    active_agents: int = 2
    router_temperature: float = 1.0
    physical_sparse_eval: bool = True

    cell_attention: bool = True
    cell_attention_threshold: float = 0.35
    cell_attention_temperature: float = 0.10
    global_refresh_interval: int = 2

    @model_validator(mode="after")
    def validate_configuration(self) -> "MATRMLiteConfig":
        self.role_names = list(validate_role_names(self.role_names, self.num_agents))
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if self.gradient_rounds < 1 or self.gradient_rounds > self.H_cycles:
            raise ValueError("gradient_rounds must be in [1, H_cycles].")
        if self.halt_min_steps < 1 or self.halt_min_steps > self.halt_max_steps:
            raise ValueError("halt_min_steps must be in [1, halt_max_steps].")
        if self.verifier_index < 0 or self.verifier_index >= self.num_agents:
            raise ValueError("verifier_index is outside the configured role range.")
        if self.critic_index < 0 or self.critic_index >= self.num_agents:
            raise ValueError("critic_index is outside the configured role range.")
        if self.active_agents < 1 or self.active_agents > self.num_agents:
            raise ValueError("active_agents must be in [1, num_agents].")
        self.topology = self.topology.strip().lower()
        self.eval_halt_granularity = self.eval_halt_granularity.strip().lower()
        if self.eval_halt_granularity not in {"batch", "sample"}:
            raise ValueError("eval_halt_granularity must be batch or sample.")
        build_topology(self.topology, student_index=self.student_index)
        return self


class MATRMLiteBlock(nn.Module):
    """The shared TRM block used by every specialized role."""

    def __init__(self, config: MATRMLiteConfig, puzzle_emb_len: int) -> None:
        super().__init__()
        self.config = config
        self.puzzle_emb_len = puzzle_emb_len
        if config.mlp_t:
            self.mlp_t = SwiGLU(
                hidden_size=config.seq_len + puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            transposed = hidden_states.transpose(1, 2)
            transposed = rms_norm(
                transposed + self.mlp_t(transposed),
                variance_epsilon=self.norm_eps,
            )
            hidden_states = transposed.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states
                + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )
        return rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )


class SharedRoleConditionedBackbone(nn.Module):
    """One dense backbone with low-rank specialization at every layer."""

    def __init__(self, config: MATRMLiteConfig, puzzle_emb_len: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [MATRMLiteBlock(config, puzzle_emb_len) for _ in range(config.L_layers)]
        )
        self.adapters = RoleAdapterBank(
            num_roles=config.num_agents,
            num_layers=config.L_layers,
            hidden_size=config.hidden_size,
            rank=config.adapter_rank,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        role_index: int,
        cos_sin: CosSin,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
            hidden_states = self.adapters(
                hidden_states,
                role_index=role_index,
                layer_index=layer_index,
            )
        return hidden_states


class MATRMLiteInner(nn.Module):
    """Differentiable recursive collaboration engine."""

    def __init__(self, config: MATRMLiteConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            config.vocab_size,
            config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.cell_verifier_head = CastedLinear(config.hidden_size, 1, bias=True)
        self.q_head = CastedLinear(config.hidden_size + 3, 2, bias=True)

        self.puzzle_emb_len = (
            -(config.puzzle_emb_ndim // -config.hidden_size)
            if config.puzzle_emb_len == 0
            else config.puzzle_emb_len
        )
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers,
                config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len + self.puzzle_emb_len,
                config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )

        self.backbone = SharedRoleConditionedBackbone(config, self.puzzle_emb_len)
        self.role_embeddings = nn.Parameter(
            trunc_normal_init_(
                torch.empty(config.num_agents, config.hidden_size),
                std=1.0 / math.sqrt(config.hidden_size),
            )
        )
        self.links = nn.ModuleList(
            [
                LowRankRecursiveLink(
                    hidden_size=config.hidden_size,
                    rank=config.link_rank,
                    norm_eps=config.rms_norm_eps,
                )
                for _ in range(config.num_agents)
            ]
        )
        self.router = SparseDynamicRouter(
            hidden_size=config.hidden_size,
            num_agents=config.num_agents,
            top_k=config.active_agents,
            hidden_router_size=config.router_hidden_size,
            mandatory_agents=(config.verifier_index,),
            temperature=config.router_temperature,
        )
        self.workspace_update = SharedLatentWorkspace(
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
        )
        self.cell_attention = CellRecursiveAttention(
            threshold=config.cell_attention_threshold,
            temperature=config.cell_attention_temperature,
            global_refresh_interval=config.global_refresh_interval,
        )
        self.topology = build_topology(
            config.topology,
            student_index=config.student_index,
        )

        self.workspace_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1.0
            ),
            persistent=True,
        )
        self.agent_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(
                    config.num_agents,
                    config.hidden_size,
                    dtype=self.forward_dtype,
                ),
                std=1.0,
            ),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            if self.q_head.bias is not None:
                self.q_head.bias.fill_(-5.0)
            self.cell_verifier_head.weight.zero_()
            if self.cell_verifier_head.bias is not None:
                self.cell_verifier_head.bias.zero_()

    def _input_embeddings(
        self,
        inputs: torch.Tensor,
        puzzle_identifiers: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.embed_tokens(inputs.to(torch.int32))
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = (
                self.puzzle_emb_len * self.config.hidden_size
                - puzzle_embedding.shape[-1]
            )
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (
                    puzzle_embedding.view(
                        -1, self.puzzle_emb_len, self.config.hidden_size
                    ),
                    embedding,
                ),
                dim=-2,
            )
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> MATRMLiteInnerCarry:
        total_len = self.config.seq_len + self.puzzle_emb_len
        return MATRMLiteInnerCarry(
            workspace=torch.empty(
                batch_size,
                total_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            agent_states=torch.empty(
                batch_size,
                self.config.num_agents,
                total_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            previous_logits=torch.empty(
                batch_size,
                self.config.seq_len,
                self.config.vocab_size,
                dtype=self.forward_dtype,
            ),
            previous_valid=torch.zeros(batch_size, dtype=torch.bool),
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: MATRMLiteInnerCarry,
    ) -> MATRMLiteInnerCarry:
        batch_size = reset_flag.shape[0]
        workspace_init = self.workspace_init.view(1, 1, -1).expand_as(carry.workspace)
        agent_init = self.agent_init.view(
            1, self.config.num_agents, 1, self.config.hidden_size
        ).expand_as(carry.agent_states)
        previous_zeros = torch.zeros_like(carry.previous_logits)
        return MATRMLiteInnerCarry(
            workspace=torch.where(
                reset_flag.view(batch_size, 1, 1),
                workspace_init,
                carry.workspace,
            ),
            agent_states=torch.where(
                reset_flag.view(batch_size, 1, 1, 1),
                agent_init,
                carry.agent_states,
            ),
            previous_logits=torch.where(
                reset_flag.view(batch_size, 1, 1),
                previous_zeros,
                carry.previous_logits,
            ),
            previous_valid=torch.where(
                reset_flag,
                torch.zeros_like(carry.previous_valid),
                carry.previous_valid,
            ),
        )

    def _pooled(self, states: torch.Tensor) -> torch.Tensor:
        if self.puzzle_emb_len > 0:
            return states[:, : self.puzzle_emb_len].mean(dim=1)
        return states.mean(dim=1)

    def _active_roles(
        self,
        routing: RoutingDecision,
    ) -> Sequence[int]:
        if not self.training and self.config.physical_sparse_eval:
            return self.router.batch_active_agents(routing)
        # Dense execution during training preserves distributed gradient
        # synchronization. The straight-through mask still trains top-k routing.
        return tuple(range(self.config.num_agents))

    def _update_round(
        self,
        workspace: torch.Tensor,
        agent_states: torch.Tensor,
        input_embeddings: torch.Tensor,
        cos_sin: CosSin,
        round_index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        RoutingDecision,
        torch.Tensor,
    ]:
        current_logits = self.lm_head(workspace)[:, self.puzzle_emb_len :]
        cell_decision = self.cell_attention(
            logits=current_logits,
            prefix_len=self.puzzle_emb_len,
            round_index=round_index,
            enabled=self.config.cell_attention,
        )
        routing = self.router(self._pooled(workspace))
        active_roles = self._active_roles(routing)
        if not self.training and self.config.physical_sparse_eval:
            execution_mask = torch.zeros_like(routing.hard_mask)
            execution_mask[:, list(active_roles)] = 1.0
            routing = RoutingDecision(
                logits=routing.logits,
                probabilities=routing.probabilities,
                hard_mask=execution_mask,
                straight_through_mask=execution_mask,
            )
        stages = self.topology.plan(
            active_roles=active_roles,
            num_agents=self.config.num_agents,
            verifier_index=self.config.verifier_index,
            critic_index=self.config.critic_index,
        )

        state_list = list(agent_states.unbind(dim=1))
        for stage in stages:
            stage_workspace = workspace
            stage_messages: list[torch.Tensor] = []
            stage_weights: list[torch.Tensor] = []
            for role_index in stage.roles:
                old_state = state_list[role_index]
                role_embedding = self.role_embeddings[role_index].to(
                    input_embeddings.dtype
                ).view(1, 1, -1)
                injection = input_embeddings + stage_workspace + role_embedding
                proposed = old_state
                for _ in range(self.config.L_cycles):
                    proposed = self.backbone(
                        hidden_states=proposed,
                        input_injection=injection,
                        role_index=role_index,
                        cos_sin=cos_sin,
                    )
                proposed = (
                    cell_decision.mask * proposed
                    + (1.0 - cell_decision.mask) * old_state
                )
                role_weight = routing.straight_through_mask[:, role_index].to(
                    proposed.dtype
                ).view(-1, 1, 1)
                updated_state = role_weight * proposed + (1.0 - role_weight) * old_state
                state_list[role_index] = updated_state
                stage_messages.append(self.links[role_index](updated_state))
                stage_weights.append(role_weight[:, 0, 0])

            messages = torch.stack(stage_messages, dim=1)
            weights = torch.stack(stage_weights, dim=1)
            proposed_workspace = self.workspace_update(
                workspace=stage_workspace,
                messages=messages,
                weights=weights,
            )
            proposed_workspace = (
                cell_decision.mask * proposed_workspace
                + (1.0 - cell_decision.mask) * stage_workspace
            )
            stage_is_active = (weights.sum(dim=-1) > 0).view(-1, 1, 1)
            workspace = torch.where(
                stage_is_active,
                proposed_workspace,
                stage_workspace,
            )

        return (
            workspace,
            torch.stack(state_list, dim=1),
            routing,
            cell_decision.active_fraction,
        )

    def forward(
        self,
        carry: MATRMLiteInnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[
        MATRMLiteInnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        input_embeddings = self._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        )
        workspace = carry.workspace
        agent_states = carry.agent_states

        route_probabilities: list[torch.Tensor] = []
        route_masks: list[torch.Tensor] = []
        cell_fractions: list[torch.Tensor] = []

        for round_index in range(self.config.H_cycles):
            grad_enabled = round_index >= self.config.H_cycles - self.config.gradient_rounds
            with torch.set_grad_enabled(torch.is_grad_enabled() and grad_enabled):
                workspace, agent_states, routing, cell_fraction = self._update_round(
                    workspace=workspace,
                    agent_states=agent_states,
                    input_embeddings=input_embeddings,
                    cos_sin=cos_sin,
                    round_index=round_index,
                )
            route_probabilities.append(routing.probabilities)
            route_masks.append(routing.hard_mask)
            cell_fractions.append(cell_fraction)

        logits = self.lm_head(workspace)[:, self.puzzle_emb_len :]
        agent_logits = self.lm_head(agent_states)[:, :, self.puzzle_emb_len :]
        final_route_probabilities = torch.stack(route_probabilities, dim=0).mean(dim=0)
        final_route_mask = route_masks[-1]
        disagreement = multi_agent_js_divergence(
            agent_logits=agent_logits,
            active_mask=final_route_mask,
        )
        answer_change = answer_change_fraction(
            previous_logits=carry.previous_logits,
            current_logits=logits,
            previous_valid=carry.previous_valid,
        )
        uncertainty = mean_predictive_uncertainty(logits)

        verifier_state = agent_states[:, self.config.verifier_index]
        cell_verify_logits = self.cell_verifier_head(verifier_state)[
            :, self.puzzle_emb_len :, 0
        ].to(torch.float32)
        q_features = torch.cat(
            (
                self._pooled(workspace).to(torch.float32),
                disagreement.sequence.unsqueeze(-1),
                uncertainty.unsqueeze(-1),
                answer_change.unsqueeze(-1),
            ),
            dim=-1,
        )
        q_logits = self.q_head(q_features).to(torch.float32)

        pooled_role_states = agent_states.mean(dim=2).to(torch.float32)
        auxiliaries: Dict[str, torch.Tensor] = {
            "agent_logits": agent_logits,
            "router_probabilities": final_route_probabilities,
            "router_mask": final_route_mask,
            "active_agents": final_route_mask.sum(dim=-1),
            "cell_active_fraction": torch.stack(cell_fractions, dim=0).mean(dim=0),
            "disagreement": disagreement.sequence,
            "cell_disagreement": disagreement.cells,
            "answer_change": answer_change,
            "uncertainty": uncertainty,
            "cell_verify_logits": cell_verify_logits,
            "role_state_pooled": pooled_role_states,
            "prior_logits": carry.previous_logits,
            "prior_valid": carry.previous_valid,
        }
        if self.config.topology == "distillation":
            teacher_indices = [
                i for i in range(self.config.num_agents) if i != self.config.student_index
            ]
            auxiliaries["teacher_logits"] = agent_logits[:, teacher_indices].mean(dim=1)
            auxiliaries["student_logits"] = agent_logits[:, self.config.student_index]

        new_carry = MATRMLiteInnerCarry(
            workspace=workspace.detach(),
            agent_states=agent_states.detach(),
            previous_logits=logits.detach(),
            previous_valid=torch.ones_like(carry.previous_valid),
        )
        return new_carry, logits, (q_logits[..., 0], q_logits[..., 1]), auxiliaries


class MultiAgentTinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper compatible with the original TRM training pipeline."""

    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        self.config = MATRMLiteConfig(**config_dict)
        self.inner = MATRMLiteInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> MATRMLiteCarry:
        batch_size = batch["inputs"].shape[0]
        return MATRMLiteCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros(batch_size, dtype=torch.int32),
            halted=torch.ones(batch_size, dtype=torch.bool),
            current_data={key: torch.empty_like(value) for key, value in batch.items()},
        )

    def forward(
        self,
        carry: MATRMLiteCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[MATRMLiteCarry, Dict[str, torch.Tensor]]:
        if self.training:
            already_finished = torch.zeros_like(carry.halted)
            reset_flag = carry.halted
            data_replace_flag = carry.halted
            new_steps = torch.where(carry.halted, 0, carry.steps)
        else:
            # Evaluation may use sample-level halting. Completed samples must
            # remain completed instead of restarting the same puzzle while
            # other samples continue through the inherited ACT loop.
            already_finished = carry.halted & (carry.steps > 0)
            reset_flag = carry.halted & ~already_finished
            data_replace_flag = reset_flag
            new_steps = carry.steps

        new_inner_carry = self.inner.reset_carry(reset_flag, carry.inner_carry)
        new_current_data = {
            key: torch.where(
                data_replace_flag.view((-1,) + (1,) * (batch[key].ndim - 1)),
                batch[key],
                value,
            )
            for key, value in carry.current_data.items()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits), aux = self.inner(
            new_inner_carry, new_current_data
        )
        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            **aux,
        }

        if already_finished.any():
            batch_mask_3d = already_finished.view(-1, 1, 1)
            batch_mask_4d = already_finished.view(-1, 1, 1, 1)
            new_inner_carry = MATRMLiteInnerCarry(
                workspace=torch.where(
                    batch_mask_3d,
                    carry.inner_carry.workspace,
                    new_inner_carry.workspace,
                ),
                agent_states=torch.where(
                    batch_mask_4d,
                    carry.inner_carry.agent_states,
                    new_inner_carry.agent_states,
                ),
                previous_logits=torch.where(
                    batch_mask_3d,
                    carry.inner_carry.previous_logits,
                    new_inner_carry.previous_logits,
                ),
                previous_valid=torch.where(
                    already_finished,
                    carry.inner_carry.previous_valid,
                    new_inner_carry.previous_valid,
                ),
            )
            outputs["logits"] = torch.where(
                batch_mask_3d,
                carry.inner_carry.previous_logits,
                outputs["logits"],
            )
            repeated_final = carry.inner_carry.previous_logits.unsqueeze(1).expand_as(
                outputs["agent_logits"]
            )
            outputs["agent_logits"] = torch.where(
                batch_mask_4d,
                repeated_final,
                outputs["agent_logits"],
            )
            for key in ("disagreement", "answer_change", "uncertainty"):
                outputs[key] = torch.where(
                    already_finished,
                    torch.zeros_like(outputs[key]),
                    outputs[key],
                )

        with torch.no_grad():
            new_steps = torch.where(
                already_finished,
                carry.steps,
                new_steps + 1,
            )
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step | already_finished

            adaptive_mode = self.training or self.config.adaptive_eval
            if adaptive_mode and self.config.halt_max_steps > 1:
                if self.config.no_ACT_continue:
                    learned_halt = q_halt_logits > self.config.halt_threshold
                else:
                    learned_halt = q_halt_logits > q_continue_logits
                stable = outputs["answer_change"] <= self.config.stability_halt_threshold
                agreed = (
                    outputs["disagreement"]
                    <= self.config.disagreement_halt_threshold
                )
                learned_halt = (
                    learned_halt
                    & stable
                    & agreed
                    & (new_steps >= self.config.halt_min_steps)
                    & ~already_finished
                )
                if (
                    not self.training
                    and self.config.eval_halt_granularity == "batch"
                ):
                    learned_halt = learned_halt.all().expand_as(learned_halt)
                halted = halted | learned_halt

                if self.training and self.config.halt_exploration_prob > 0:
                    explore = (
                        torch.rand_like(q_halt_logits)
                        < self.config.halt_exploration_prob
                    )
                    random_min = torch.randint_like(
                        new_steps,
                        low=self.config.halt_min_steps,
                        high=self.config.halt_max_steps + 1,
                    )
                    halted = halted & (~explore | (new_steps >= random_min))

                if not self.config.no_ACT_continue:
                    _, _, (next_halt, next_continue), _ = self.inner(
                        new_inner_carry, new_current_data
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_halt,
                            torch.maximum(next_halt, next_continue),
                        )
                    )

        return (
            MATRMLiteCarry(
                inner_carry=new_inner_carry,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
            ),
            outputs,
        )


# Short aliases used by configs and external scripts.
MATRMLite = MultiAgentTinyRecursiveReasoningModel_ACTV1
MA_TRM_Lite = MultiAgentTinyRecursiveReasoningModel_ACTV1
