"""JEPA joint training core (SPEC section 8, TASKS T5.3).

One joint step trains the online encoder E (the noisy tokenizer), its decoder D
and the generator G (backbone + head) on the same batch:

1. E encodes the batch: clean codes ``c_k``, chain codes ``c_k + sigma_k * eps``
   and ``zhat`` (SPEC 4.1/4.3).
2. Targets under ``no_grad`` (SPEC 8 step 2). ``jepa.target_chain = "online"``
   (default) teacher-forces E-bar onto the online chain:
   ``t_k = LayerNorm(Ebar.proj_k(down(zbar - zhat_{k-1}, l_k)))`` with the
   online ``zhat_{k-1}`` detached, before corruption. ``"ema_clean"`` keeps
   the v1.3 ``encode_clean`` chain (E-bar's own noise-free chain). When
   ``jepa.ema`` is false, E itself produces the targets, under stop-gradient.
3. G predicts level k from the corrupted online chain (SPEC 5.4), so the
   prediction loss reaches E through the conditioning (SPEC 8 step 3).
4. ``L_pred`` weights each level's head loss against ``sg(t_k)`` with the SPEC
   5.5 position weights; for ``flow`` the flow-matching target is ``t_k``.
5. ``L_dec = CE(D(zhat[cut]), x)`` with nested dropout; with
   ``jepa.decoder_through_E`` false the decoder input is detached.
6. ``L_vic`` is VICReg on the online ``c_k`` and on the predicted ``c-hat_k``.
7. ``L = L_dec + lambda_pred * L_pred + L_vic``.

``joint_loss`` consumes one explicit ``torch.Generator`` for every random draw,
in this order: online channel noise, conditioning corruption, the nested
dropout cutoff, flow-matching draws, the VICReg flow integration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from jncm.config import Config
from jncm.generator.backbone import GeneratorBackbone
from jncm.generator.corruption import corrupt_codes
from jncm.generator.heads import FlowHead, LevelHead, RegressHead, weighted_total
from jncm.jepa.vicreg import vicreg_loss
from jncm.levels import down
from jncm.tokenizer.trunk import MultiscaleTokenizer, sample_cutoff

JEPA_HEADS: tuple[str, ...] = ("regress", "flow")
TARGET_CHAINS: tuple[str, ...] = ("online", "ema_clean")


def encode_clean(
    tokenizer: MultiscaleTokenizer,
    x: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run the noisy encoder's residual chain with the channel noise off.

    ``MultiscaleTokenizer.encode`` always transmits ``c_k + sigma_k * eps``
    (SPEC 4.3: the noise is a channel, not a regularizer). The EMA target
    encoder must not see that noise (SPEC 8 step 2), so this repeats the SPEC
    4.1 chain with the clean code ``c_k = LayerNorm(Linear_k(u))``.

    Args:
        tokenizer: A ``noisy`` multiscale tokenizer (online or EMA copy).
        x: Token ids [B, L].

    Returns:
        ``(targets, zhat)`` where ``targets[k]`` is [B, l_k, d_k] and ``zhat``
        holds K+1 tensors [B, L, d], ``zhat[0] = 0``.
    """
    if tokenizer.quantizer_type != "noisy":
        raise ValueError(
            "JEPA joint mode requires the noisy quantizer (SPEC 8), got "
            f"quantizer={tokenizer.quantizer_type!r}"
        )
    if x.dim() != 2:
        raise ValueError(f"x must be [B, L] token ids, got {tuple(x.shape)}")

    z = tokenizer.encode_trunk(x)
    zhat: list[torch.Tensor] = [torch.zeros_like(z)]
    targets: list[torch.Tensor] = []
    for k in range(1, tokenizer.K + 1):
        residual = z - zhat[-1]
        u = down(residual, tokenizer.schedule[k - 1])
        quantizer = tokenizer.quantizers[k - 1]
        clean = quantizer.norm(quantizer.proj(u))
        targets.append(clean)
        zhat.append(zhat[-1] + tokenizer.contrib(k, clean))
    return targets, zhat


def encode_online_targets(
    target_tokenizer: MultiscaleTokenizer,
    x: torch.Tensor,
    zhat: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """SPEC 8 step 2 default chain: E-bar's embedding, online partial chain.

    ``t_k = LayerNorm(Ebar.proj_k(down(zbar - zhat_{k-1}, l_k)))`` with
    ``zbar = Ebar.encode_trunk(x)`` and the online partial reconstruction
    ``zhat_{k-1}`` (detached, before corruption). The targets then live in the
    chain the decoder is trained on, so finer levels correct the coarse channel
    noise.

    Args:
        target_tokenizer: E-bar (EMA copy), or the online encoder when
            ``jepa.ema`` is false.
        x: Token ids [B, L].
        zhat: Online chain reconstructions, K+1 tensors [B, L, d], zhat[0] = 0.

    Returns:
        One target [B, l_k, d_k] per level.
    """
    if target_tokenizer.quantizer_type != "noisy":
        raise ValueError(
            "JEPA joint mode requires the noisy quantizer (SPEC 8), got "
            f"quantizer={target_tokenizer.quantizer_type!r}"
        )
    if x.dim() != 2:
        raise ValueError(f"x must be [B, L] token ids, got {tuple(x.shape)}")
    expected = target_tokenizer.K + 1
    if len(zhat) != expected:
        raise ValueError(f"expected {expected} zhat tensors, got {len(zhat)}")

    zbar = target_tokenizer.encode_trunk(x)
    targets: list[torch.Tensor] = []
    for k in range(1, target_tokenizer.K + 1):
        residual = zbar - zhat[k - 1]
        u = down(residual, target_tokenizer.schedule[k - 1])
        quantizer = target_tokenizer.quantizers[k - 1]
        targets.append(quantizer.norm(quantizer.proj(u)))
    return targets


def integrate_flow(
    head: FlowHead,
    hidden: torch.Tensor,
    level: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Euler-integrate one level's flow without channel noise (c-hat_k).

    This is the deterministic part of ``FlowHead.sample_level``: SPEC 8 applies
    VICReg to ``c-hat_k``, so the channel noise is left out (it would make the
    variance term trivially large).

    Args:
        head: Flow head whose ``nets`` and ``flow_steps`` drive the integration.
        hidden: Level hidden state [B, l_k, H].
        level: 1-indexed level.
        generator: Optional RNG for the starting ``eps``.

    Returns:
        Predicted clean code [B, l_k, d_k].
    """
    b, length, _ = hidden.shape
    h_flat = hidden.reshape(-1, head.H)
    n = b * length
    x = torch.randn(
        (n, head.d_k),
        generator=generator,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    net = head.nets[level - 1]
    dt = 1.0 / head.flow_steps
    for step in range(head.flow_steps):
        t = torch.full((n,), step * dt, device=hidden.device, dtype=hidden.dtype)
        x = x + dt * net(x, t, h_flat)
    return x.reshape(b, length, head.d_k)


def predicted_codes(
    head: LevelHead,
    hidden: Sequence[torch.Tensor],
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Deterministic ``c-hat_k`` per level, without sampling noise (SPEC 8).

    ``regress`` uses the raw regression output (renormalization is a
    sampling-time correction); ``flow`` uses the noiseless Euler integration.

    Args:
        head: ``regress`` or ``flow`` head.
        hidden: One tensor [B, l_k, H] per level.
        generator: Optional RNG for the flow start ``eps``.

    Returns:
        One tensor [B, l_k, d_k] per level.
    """
    if len(hidden) != head.K:
        raise ValueError(f"expected {head.K} hidden tensors, got {len(hidden)}")
    out: list[torch.Tensor] = []
    for level, h in enumerate(hidden, start=1):
        if isinstance(head, RegressHead):
            out.append(head.heads[level - 1](h))
        elif isinstance(head, FlowHead):
            out.append(integrate_flow(head, h, level, generator=generator))
        else:
            raise ValueError(
                "JEPA joint mode needs a continuous head ('regress' or 'flow'), "
                f"got {type(head).__name__}"
            )
    return out


@dataclass
class JointStep:
    """Tensors and losses of one JEPA joint step (SPEC 8)."""

    loss: torch.Tensor
    loss_dec: torch.Tensor
    loss_pred: torch.Tensor
    loss_vic: torch.Tensor
    per_level_pred: list[torch.Tensor]
    codes: list[torch.Tensor]
    targets: list[torch.Tensor]
    hidden: list[torch.Tensor]
    predictions: list[torch.Tensor] = field(default_factory=list)
    cut: int = 0


def joint_loss(
    cfg: Config,
    tokenizer: MultiscaleTokenizer,
    backbone: GeneratorBackbone,
    head: LevelHead,
    x: torch.Tensor,
    target_tokenizer: torch.nn.Module | None = None,
    generator: torch.Generator | None = None,
) -> JointStep:
    """One SPEC 8 joint loss for a token batch ``x`` of shape [B, L].

    Args:
        cfg: Config; ``jepa.*`` flags and ``generator.*`` weights are used.
        tokenizer: Online noisy tokenizer (trainable encoder E and decoder D).
        backbone: Generator backbone (K levels).
        head: ``regress`` or ``flow`` head.
        x: Token ids [B, L].
        target_tokenizer: EMA copy E-bar, or None to take the targets from the
            online encoder under stop-gradient (``jepa.ema = false``).
            ``jepa.target_chain`` selects the online (default) or ``ema_clean``
            target chain.
        generator: Single RNG for all draws; a seed reproduces the step.

    Returns:
        :class:`JointStep` with the total loss and every component, including
        the detached c_k, t_k, hidden states and (when VICReg is on) c-hat_k.
    """
    if cfg.tokenizer.quantizer != "noisy":
        raise ValueError(
            "JEPA joint mode requires quantizer='noisy' (SPEC 8), got "
            f"{cfg.tokenizer.quantizer!r}"
        )
    if cfg.generator.head not in JEPA_HEADS:
        raise ValueError(
            f"JEPA joint mode needs head in {JEPA_HEADS}, got "
            f"{cfg.generator.head!r}"
        )

    # 1. Online encode: clean codes c_k, chain codes with channel noise, zhat.
    enc = tokenizer.encode(x, generator=generator)

    # 2. Targets (SPEC 8 step 2): ``online`` teacher-forces E-bar onto the
    #    online chain (detached, before corruption); ``ema_clean`` uses E-bar's
    #    own noise-free chain (the v1.3 definition).
    source = tokenizer if target_tokenizer is None else target_tokenizer
    with torch.no_grad():
        if cfg.jepa.target_chain == "online":
            targets = encode_online_targets(
                source, x, [z.detach() for z in enc.zhat]
            )
        elif cfg.jepa.target_chain == "ema_clean":
            targets, _ = encode_clean(source, x)
        else:
            raise ValueError(
                f"jepa.target_chain must be one of {TARGET_CHAINS}, got "
                f"{cfg.jepa.target_chain!r}"
            )
    targets = [target.detach() for target in targets]

    # 3. Corrupt the online chain once, then predict all levels packed.
    corrupted = corrupt_codes(
        enc.codes,
        cfg.generator.corrupt,
        generator=generator,
        quantizer=cfg.tokenizer.quantizer,
    )
    level_inputs = backbone.embed_levels(tokenizer, corrupted)
    hidden = backbone.forward_levels(level_inputs)

    # 4. L_pred against the stop-gradient target with SPEC 5.5 weights.
    _, per_level = head.loss(hidden, targets, generator=generator)
    loss_pred = weighted_total(per_level, backbone.schedule, cfg.generator.alpha)

    # 5. L_dec with nested dropout; optional detach closes the E path.
    cut = sample_cutoff(tokenizer.K, cfg.tokenizer.p_drop, generator=generator)
    decode_input = enc.zhat[cut]
    if not cfg.jepa.decoder_through_E:
        decode_input = decode_input.detach()
    logits = tokenizer.decode(decode_input)
    loss_dec = F.cross_entropy(
        logits.reshape(-1, tokenizer.V), x.reshape(-1)
    )

    # 6. VICReg on c_k and on c-hat_k (SPEC 8 step 6).
    predictions: list[torch.Tensor] = []
    loss_vic = enc.zhat[0].new_zeros(())
    if cfg.jepa.vicreg:
        predictions = predicted_codes(head, hidden, generator=generator)
        for code, prediction in zip(enc.targets, predictions):
            loss_vic = loss_vic + vicreg_loss(
                code.reshape(-1, code.shape[-1]).float(),
                gamma=cfg.jepa.gamma,
                lambda_var=cfg.jepa.lambda_var,
                lambda_cov=cfg.jepa.lambda_cov,
            )
            loss_vic = loss_vic + vicreg_loss(
                prediction.reshape(-1, prediction.shape[-1]).float(),
                gamma=cfg.jepa.gamma,
                lambda_var=cfg.jepa.lambda_var,
                lambda_cov=cfg.jepa.lambda_cov,
            )

    loss = loss_dec + cfg.jepa.lambda_pred * loss_pred + loss_vic
    return JointStep(
        loss=loss,
        loss_dec=loss_dec,
        loss_pred=loss_pred,
        loss_vic=loss_vic,
        per_level_pred=[value.detach() for value in per_level],
        codes=[code.detach() for code in enc.targets],
        targets=targets,
        hidden=[h.detach() for h in hidden],
        predictions=[p.detach() for p in predictions],
        cut=cut,
    )
