---
status: reference
updated: 2026-09-26T21:29:45-04:00
scope: Published Threes! ONNX policy
---

# Published Threes! ONNX policy

The local `onnx_model.onnx` file was retrieved from [pseudonam-gc/threes-web](https://github.com/pseudonam-gc/threes-web) at commit `9527295de72333c91592fff45e8ccbcabd6ba80d` for local research evaluation.

- Size: 55,993,340 bytes.
- SHA-256: `63bff54b23de09801fb2bc77ca8856ed781c2abaa2efe896c3da15e5d810eff2`.
- Runtime: ONNX Runtime CPU; input is 21 `int64` values and two 1 × 1024 `float32` LSTM states.
- The upstream README does not specify a code or model license. Keep the binary local; do not redistribute it without clarifying the terms.
- The upstream README describes a 24-dimensional input, while its browser adapter, C observation writer, and the ONNX file use **21** values. The browser adapter and ONNX signature were used for this integration.
- The upstream README reports a later 21.5% rate of tile 6,144 on 10,000 games. It does not identify which training checkpoint the checked-in ONNX file contains. Evaluate the checked-in file directly; do not assign that reported rate to it without evidence.

The historical adapter in [`src/external_agent.py`](../../src/external_agent.py) uses the older GUI simulator; its [compatibility audit](../../docs/archive/LEGACY_MODEL_COMPATIBILITY.md) describes transfer differences. Current recurrent evaluations and the policy inspector use the separate audited v2 simulator. See the [fidelity audit](../../docs/FIDELITY_AUDIT.md) for what is verified. Interpret scores using the evaluation's recorded engine and protocol.

To retrieve the same file for local evaluation in a fresh checkout, use PowerShell from the project root and verify the hash above:

```powershell
New-Item -ItemType Directory -Force external_models\kiok | Out-Null
Invoke-WebRequest 'https://raw.githubusercontent.com/pseudonam-gc/threes-web/9527295de72333c91592fff45e8ccbcabd6ba80d/model/onnx_model.onnx' -OutFile external_models\kiok\onnx_model.onnx
(Get-FileHash external_models\kiok\onnx_model.onnx -Algorithm SHA256).Hash
```
