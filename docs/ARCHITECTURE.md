# MA-TRM Architecture Notes

## State definition

For batch size \(B\), sequence length \(L\), hidden width \(d\), vocabulary size \(C\), and \(M\) roles, MA-TRM maintains:

- shared workspace, `workspace`, shape `[B, L, d]`;
- private role states, `agent_states`, shape `[B, M, L, d]`;
- previous answer logits, shape `[B, L, C]`;
- routing probabilities and top-k masks, shape `[B, M]`.

The private states persist across ACT calls. They reset only for batch elements whose prior trajectory halted.

## Shared backbone and controlled specialization

Every role uses the same attention and SwiGLU backbone. Each role and each layer has a low-rank residual adapter:

$$
A_{i,\ell}(h)=h+s_{i,\ell}U_{i,\ell}\operatorname{SiLU}(D_{i,\ell}h).
$$

The default rank is eight. This adds 65,544 adapter parameters, including scalar scales, rather than four independent 6.8 million parameter networks.

## Training and physical sparsity

Per-sample sparse routing can produce different active roles on different distributed ranks. The inherited TRM training loop manually reduces gradients and assumes matching parameter participation. MA-TRM therefore uses dense masked execution during training. Every role executes, but inactive updates receive a straight-through top-k mask.

During evaluation, `physical_sparse_eval=true` chooses one batch-level top-k set, applies it to every sample in that batch, and executes only those roles. This provides real wall-time and memory measurements without distributed unused-gradient inconsistencies.

## Gradient truncation

`gradient_rounds` determines how many final collaboration rounds retain gradients. Earlier rounds execute under `torch.no_grad()`, following the memory-saving pattern used by TRM. Set `gradient_rounds=H_cycles` to backpropagate through every collaboration round.

## Adaptive halting

A trajectory halts when all of the following hold:

1. the learned halt logit exceeds `halt_threshold`;
2. mean Jensen-Shannon disagreement is at most `disagreement_halt_threshold`;
3. the answer change fraction is at most `stability_halt_threshold`;
4. `halt_min_steps` has been reached.

`halt_max_steps` is an unconditional safety bound. The default evaluation granularity is batch-level, so a batch halts only when every sample satisfies the learned conditions. This preserves the inherited ACT evaluation loop and prevents completed samples from restarting while other samples continue.

## Implementation cautions

- Batch-level physical routing measures actual sparsity but can underrepresent per-instance routing diversity.
- Disagreement can be low when all roles make the same error.
- The verifier must be calibrated before expected Hamming error is interpreted as a risk bound.
- A low-rank link preserves differentiability and uncertainty, but it is not guaranteed to reduce runtime when the vocabulary is small.
- Cell masks receive periodic global refreshes to avoid permanent local lock-in.
