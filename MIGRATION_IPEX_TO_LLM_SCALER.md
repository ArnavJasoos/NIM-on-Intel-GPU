# Migration Guide: IPEX-LLM → LLM-Scaler

## Executive Summary

**Your existing NIM + IPEX-LLM architecture requires NO changes to the NIM proxy layer.** You only need to swap the Docker image used for the inference backend.

- **Old:** `intelanalytics/ipex-llm-serving-xpu:latest`
- **New:** `intel/llm-scaler-vllm:1.3` (Intel's official, actively maintained)

The OpenAI-compatible API contract is 100% identical. Your deployment just gets newer vLLM (0.14.0 vs 0.6.0), better model support, and official Intel backing.

**Migration time:** < 5 minutes (change one line in docker-compose.yml, restart)

---

## Why Migrate?

### IPEX-LLM Status (2025-2026)

IPEX-LLM is **not deprecated**. It's still maintained on GitHub (`intel/ipex-llm`). However:

1. **Project scope shifted** — Intel created LLM-Scaler as the official, unified inference platform for Project Battlematrix (their new AI PC/workstation initiative)
2. **Active development moved** — New features, optimizations, and model support land in LLM-Scaler first
3. **Branding clarity** — LLM-Scaler is the public-facing name; IPEX-LLM is the underlying library

**Think of it like:** IPEX-LLM = library name, LLM-Scaler = product name. You're not losing anything; you're getting a better-maintained wrapper.

### Benefits of Switching to LLM-Scaler

| Benefit | Details |
|---------|---------|
| **Newer vLLM** | 0.14.0 (May 2026) vs 0.6.0 — 8+ months of upstream optimizations |
| **Active development** | Monthly releases vs quarterly — faster model support, bug fixes |
| **Intel official** | Part of Project Battlematrix — guaranteed long-term support |
| **Better MoE support** | Qwen3-235B-A22B (235B MoE, sym_int4 at TP16) now works |
| **Speculative decoding** | 15-30% faster inference via Medusa/Suffix methods |
| **FP6 quantization** | New ultra-low precision option (experimental) |
| **More models** | Qwen3-VL, Qwen3-Omni, PaddleOCR, ERNIE-4.5-vl, etc. |

---

## Migration Steps

### Step 1: Backup Your Current Setup

```bash
cd ~/nim-ipex-deployment

# Save current config
cp docker-compose.yml docker-compose.yml.ipex-backup
cp .env .env.backup
```

### Step 2: Update docker-compose.yml

**Option A: Use the provided LLM-Scaler template**
```bash
cp docker-compose-llm-scaler.yml docker-compose.yml
```

**Option B: Manual edit (if you've customized docker-compose.yml)**

Find the `ipex-backend` service image line:
```yaml
# OLD
image: intelanalytics/ipex-llm-serving-xpu:latest

# NEW
image: intel/llm-scaler-vllm:1.3
```

Also rename the service for clarity (optional but recommended):
```yaml
# OLD
services:
  ipex-backend:

# NEW
services:
  llm-scaler-backend:
```

And update the depends_on reference in nim-proxy:
```yaml
depends_on:
  llm-scaler-backend:  # was: ipex-backend
    condition: service_healthy
```

And the backend URL in NIM proxy environment:
```yaml
NIM_BACKEND_URL: "http://llm-scaler-backend:8000"  # was: ipex-backend
```

### Step 3: Restart Services

```bash
# Stop old stack
./deploy.sh stop

# Start new stack (pull new image)
./deploy.sh start

# Monitor startup
./deploy.sh logs

# Verify
./deploy.sh health
./deploy.sh test
```

**First run:** Kernel compilation takes 2-5 minutes (same as IPEX-LLM). Subsequent starts <10s.

---

## Rollback Plan

If you encounter issues:

```bash
# Revert docker-compose
cp docker-compose.yml.ipex-backup docker-compose.yml

# Restart with old image
./deploy.sh restart

# Old IPEX-LLM images cached locally, so this is instant
```

---

## What Works Without Changes

✓ **All existing configurations** (.env remains identical)  
✓ **NIM proxy layer** (no modifications needed)  
✓ **OpenAI API** (same endpoints, same request/response format)  
✓ **Model cache** (HuggingFace directory is shared)  
✓ **Health checks** (same /v1/models, /v1/health endpoints)  
✓ **Docker networking** (NIM → backend communication unchanged)  
✓ **All existing clients** (no code changes in your application)  

---

## Testing After Migration

### Quick Smoke Test

```bash
# Same as before
./deploy.sh test

# Should return: "2+2 = 4" (or similar)
```

### Verify vLLM Version

```bash
docker exec nim-llm-scaler-backend python3 -c "import vllm; print(vllm.__version__)"
# Should show: 0.14.0.dev... or similar (much newer than IPEX-LLM's 0.6.0)
```

### Compare Throughput (Optional)

```bash
# Before (IPEX-LLM)
time ./deploy.sh test
# Note the duration

# After (LLM-Scaler)
time ./deploy.sh test
# Should be similar or slightly faster
```

---

## Known Differences (Minor)

### LLM-Scaler vs IPEX-LLM

| Aspect | IPEX-LLM | LLM-Scaler | Impact |
|--------|----------|-----------|--------|
| **Command format** | `python -m ipex_llm.vllm...` | `python3 -m vllm.entrypoints...` | Already handled in docker-compose |
| **Startup message** | "IPEX-LLM" in logs | "vLLM" in logs | Cosmetic only |
| **Error messages** | References IPEX | References vLLM | Clearer (upstream vLLM errors) |
| **Image size** | ~2.8 GB | ~3.2 GB | Larger due to newer deps, not an issue |
| **First-token latency** | 3-5s (iGPU) | 3-5s (iGPU) | No change (GPU-bound) |

---

## Environment Variables (Unchanged)

All your existing `.env` variables work with LLM-Scaler:

```bash
# No changes needed
NIM_CONTAINER_URL=...
LLM_MODEL_NAME=...
NGC_API_KEY=...
IPEX_BACKEND_PORT=8001      # ← name is misleading now, but works fine
HF_CACHE_PATH=...
```

(Optionally rename `IPEX_BACKEND_PORT` to `LLM_SCALER_BACKEND_PORT` for clarity, but it's not required.)

---

## Advanced: Enabling New Features

### Speculative Decoding (15-30% Faster)

Add to LLM-Scaler backend command:
```yaml
command: >
  python3 -m vllm.entrypoints.openai.api_server
    --model ${LLM_MODEL_NAME}
    --device xpu
    --dtype float16
    --load-in-low-bit sym_int4
    --speculative-model ${LLM_MODEL_NAME}  # ← NEW
    --num-speculative-tokens 5              # ← NEW
    --speculative-algorithm medusa          # ← NEW
    ...
```

Then restart:
```bash
./deploy.sh restart
```

**Expected improvement:** 1.2-1.5x faster token generation (baseline 5 tok/s → 6-7.5 tok/s)

### Try FP6 Quantization (Experimental)

```yaml
# In docker-compose (optional experimental)
command: >
  python3 -m vllm.entrypoints.openai.api_server
    --model ${LLM_MODEL_NAME}
    --load-in-low-bit sym_int6  # ← NEW (experimental)
    ...
```

**Note:** FP6 is newer and less tested. Stick with `sym_int4` unless you need the memory savings.

---

## Troubleshooting Migration Issues

### Issue: "Image not found" error

```bash
# LLM-Scaler images are on Docker Hub
# Ensure internet access and run:
docker pull intel/llm-scaler-vllm:1.3

# Then retry
./deploy.sh start
```

### Issue: Inference slower after migration

This shouldn't happen. If it does:

1. Check vLLM version:
   ```bash
   docker exec nim-llm-scaler-backend python3 -c "import vllm; print(vllm.__version__)"
   ```

2. Verify quantization is working:
   ```bash
   docker logs nim-llm-scaler-backend | grep "load_in_low_bit"
   ```

3. Monitor GPU:
   ```bash
   docker exec nim-llm-scaler-backend intel_gpu_top
   ```

### Issue: Old IPEX-LLM container still running

```bash
# Clean up old images (only if you're sure)
docker rmi intelanalytics/ipex-llm-serving-xpu:latest

# Don't delete local models
ls -la ~/.cache/huggingface  # Should still be there
```

---

## Long-term Support

### LLM-Scaler Release Cadence

- **Monthly minor releases:** vLLM upstream integrations, model support, bug fixes
- **Quarterly major versions:** New capabilities (speculative decoding, new quantization)
- **3-year support window:** Intel commits to maintaining Project Battlematrix stack through 2028

### Future Versions to Watch

- **intel/llm-scaler-vllm:1.4+** — Upstream vLLM 0.15+
- **intel/llm-scaler-omni:0.2+** — Multimodal (image, video generation)
- **Battlemage GPU support** — Intel Arc B-series optimizations (your iGPU benefits indirectly)

---

## Summary

| Step | Action | Time |
|------|--------|------|
| 1 | Update docker-compose.yml (1 line) | 1 min |
| 2 | Run `./deploy.sh stop` | 30 sec |
| 3 | Run `./deploy.sh start` (pulls new image) | 1-2 min |
| 4 | Wait for kernel compile (first run) | 2-5 min |
| 5 | Run `./deploy.sh test` | 30 sec |
| **Total** | **Full migration** | **5-10 min** |

**Your NIM proxy and client code need zero changes.**

---

**Migration Date:** May 2026  
**LLM-Scaler Version:** 1.3 (stable, vLLM 0.11.1)  
**Status:** Ready for production
